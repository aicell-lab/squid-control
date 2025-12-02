import argparse
import asyncio
import fractions
import io
import json
import logging
import logging.handlers
import os
import sys
import time
import traceback
from pathlib import Path

import cv2
import dotenv
import numpy as np
from hypha_rpc import connect_to_server, login, register_rtc_service
from PIL import Image

# Import from squid_control package (now relative since we're inside the package)
# Handle both module and script execution
try:
    from .control.camera import TriggerModeSetting
    from .control.config import CONFIG, ChannelMapper
    from .hypha_tools.snapshot_utils import SnapshotManager
    from .squid_controller import SquidController
except ImportError:
    # Fallback for direct script execution from project root
    import os
    import sys
    # Add the project root to Python path
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)

    from .control.config import ChannelMapper
    from .squid_controller import SquidController

import base64
import signal
from collections import deque
from typing import Any, Dict, List, Optional

# WebRTC imports
import aiohttp
from aiortc import MediaStreamTrack
from av import VideoFrame
from hypha_rpc.utils.schema import schema_function
from pydantic import Field

dotenv.load_dotenv()
ENV_FILE = dotenv.find_dotenv()
if ENV_FILE:
    dotenv.load_dotenv(ENV_FILE)
import uuid  # noqa: E402

# Set up logging
from squid_control.utils.logging_utils import setup_logging
from squid_control.utils.video_utils import VideoBuffer, VideoFrameProcessor

logger = setup_logging("squid_control_service.log")

class MicroscopeVideoTrack(MediaStreamTrack):
    """
    A video stream track that provides real-time microscope images.
    """

    kind = "video"

    def __init__(self, microscope_instance):
        super().__init__()  # Initialize parent MediaStreamTrack
        self.microscope_instance = microscope_instance
        self.running = True
        self.fps = 5  # Default to 5 FPS
        self.count = 0
        self.start_time = None
        self.frame_width = 750
        self.frame_height = 750
        logger.info(f"MicroscopeVideoTrack initialized with FPS: {self.fps}")

    def draw_crosshair(self, img, center_x, center_y, size=20, color=[255, 255, 255]):
        """Draw a crosshair on the image"""
        import cv2  # noqa: PLC0415
        # Draw horizontal line
        cv2.line(img, (center_x - size, center_y), (center_x + size, center_y), color, 2)
        # Draw vertical line
        cv2.line(img, (center_x, center_y - size), (center_x, center_y + size), color, 2)

    async def recv(self):
        if not self.running:
            logger.warning("MicroscopeVideoTrack: recv() called but track is not running")
            raise Exception("Track stopped")

        try:
            if self.start_time is None:
                self.start_time = time.time()

            next_frame_time = self.start_time + (self.count / self.fps)
            sleep_duration = next_frame_time - time.time()
            if sleep_duration > 0:
                await asyncio.sleep(sleep_duration)

            # Get compressed frame data WITH METADATA from microscope
            frame_response = await self.microscope_instance.get_video_frame(
                frame_width=self.frame_width,
                frame_height=self.frame_height
            )

            # Extract frame data and metadata
            if isinstance(frame_response, dict) and 'data' in frame_response:
                frame_data = frame_response
                frame_metadata = frame_response.get('metadata', {})
            else:
                # Fallback for backward compatibility
                frame_data = frame_response
                frame_metadata = {}

            # Decompress JPEG data to numpy array for WebRTC
            processed_frame = self.microscope_instance.decode_video_frame(frame_data)

            current_time = time.time()
            # Use a 90kHz timebase, common for video, to provide accurate frame timing.
            # This prevents video from speeding up if frame acquisition is slow.
            time_base = fractions.Fraction(1, 90000)
            pts = int((current_time - self.start_time) * time_base.denominator)

            # Create VideoFrame
            new_video_frame = VideoFrame.from_ndarray(processed_frame, format="rgb24")
            new_video_frame.pts = pts
            new_video_frame.time_base = time_base

            # SEND METADATA VIA WEBRTC DATA CHANNEL
            # Send metadata through data channel instead of embedding in video frame
            if frame_metadata and hasattr(self.microscope_instance, 'metadata_data_channel'):
                try:
                    # Metadata already includes gray level statistics calculated in background acquisition
                    metadata_json = json.dumps(frame_metadata)
                    # Send metadata via WebRTC data channel
                    asyncio.create_task(self._send_metadata_via_datachannel(metadata_json))
                    logger.debug(f"Sent metadata via data channel: {len(metadata_json)} bytes (with gray level stats)")
                except Exception as e:
                    logger.warning(f"Failed to send metadata via data channel: {e}")

            if self.count % (self.fps * 5) == 0:  # Log every 5 seconds
                duration = current_time - self.start_time
                if duration > 0:
                    actual_fps = (self.count + 1) / duration
                    logger.info(f"MicroscopeVideoTrack: Sent frame {self.count}, actual FPS: {actual_fps:.2f}")
                    if frame_metadata:
                        stage_pos = frame_metadata.get('stage_position', {})
                        x_mm = stage_pos.get('x_mm')
                        y_mm = stage_pos.get('y_mm')
                        z_mm = stage_pos.get('z_mm')
                        # Handle None values in position logging
                        x_str = f"{x_mm:.2f}" if x_mm is not None else "None"
                        y_str = f"{y_mm:.2f}" if y_mm is not None else "None"
                        z_str = f"{z_mm:.2f}" if z_mm is not None else "None"
                        logger.info(f"Frame metadata: stage=({x_str}, {y_str}, {z_str}), "
                                   f"channel={frame_metadata.get('channel')}, intensity={frame_metadata.get('intensity')}")
                else:
                    logger.info(f"MicroscopeVideoTrack: Sent frame {self.count}")

            self.count += 1
            return new_video_frame

        except Exception as e:
            logger.error(f"MicroscopeVideoTrack: Error in recv(): {e}", exc_info=True)
            self.running = False
            raise

    def update_fps(self, new_fps):
        """Update the FPS of the video track"""
        self.fps = new_fps
        logger.info(f"MicroscopeVideoTrack FPS updated to {new_fps}")

    async def _send_metadata_via_datachannel(self, metadata_json):
        """Send metadata via WebRTC data channel"""
        try:
            if hasattr(self.microscope_instance, 'metadata_data_channel') and self.microscope_instance.metadata_data_channel:
                if self.microscope_instance.metadata_data_channel.readyState == 'open':
                    self.microscope_instance.metadata_data_channel.send(metadata_json)
                    logger.debug(f"Metadata sent via data channel: {len(metadata_json)} bytes")
                else:
                    logger.debug(f"Data channel not ready, state: {self.microscope_instance.metadata_data_channel.readyState}")
        except Exception as e:
            logger.warning(f"Error sending metadata via data channel: {e}")

    def stop(self):
        logger.info("MicroscopeVideoTrack stop() called.")
        self.running = False
        # Mark WebRTC as disconnected
        self.microscope_instance.webrtc_connected = False

class MicroscopeHyphaService:
    def __init__(self, is_simulation, is_local):  # noqa: PLR0915
        self.current_x = 0
        self.current_y = 0
        self.current_z = 0
        self.current_theta = 0
        self.current_illumination_channel = None
        self.current_intensity = None
        self.is_illumination_on = False
        self.is_simulation = is_simulation
        self.is_local = is_local
        self.squidController = SquidController(is_simulation=is_simulation)
        self.squidController.move_to_well('C',3)

        # Determine if this is a Squid+ microscope
        self.is_squid_plus = self._is_squid_plus_microscope()
        self.dx = 1
        self.dy = 1
        self.dz = 1
        self.BF_intensity_exposure = [50, 100]
        self.F405_intensity_exposure = [50, 100]
        self.F488_intensity_exposure = [50, 100]
        self.F561_intensity_exposure = [50, 100]
        self.F638_intensity_exposure = [50, 100]
        self.F730_intensity_exposure = [50, 100]
        self.channel_param_map = ChannelMapper.get_id_to_param_map()
        self.parameters = {
            'current_x': self.current_x,
            'current_y': self.current_y,
            'current_z': self.current_z,
            'current_theta': self.current_theta,
            'is_illumination_on': self.is_illumination_on,
            'dx': self.dx,
            'dy': self.dy,
            'dz': self.dz,
            'BF_intensity_exposure': self.BF_intensity_exposure,
            'F405_intensity_exposure': self.F405_intensity_exposure,
            'F488_intensity_exposure': self.F488_intensity_exposure,
            'F561_intensity_exposure': self.F561_intensity_exposure,
            'F638_intensity_exposure': self.F638_intensity_exposure,
            'F730_intensity_exposure': self.F730_intensity_exposure,
        }
        self.authorized_emails = self.load_authorized_emails()
        logger.info(f"Authorized emails: {self.authorized_emails}")
        self.snapshot_manager = None  # Will be initialized after artifact_manager in setup()
        self.server_url =  "http://192.168.2.1:9527" if is_local else "https://hypha.aicell.io/"
        self.server = None
        self.service_id = os.environ.get("MICROSCOPE_SERVICE_ID")
        self.setup_task = None  # Track the setup task

        # WebRTC related attributes
        self.video_track = None
        self.webrtc_service_id = None
        self.is_streaming = False
        self.video_contrast_min = 0
        self.video_contrast_max = None
        self.metadata_data_channel = None  # WebRTC data channel for metadata

        # Video buffering attributes
        self.video_buffer = VideoBuffer(max_size=5)
        self.frame_acquisition_task = None
        self.frame_acquisition_running = False
        self.buffer_fps = 5  # Background frame acquisition FPS
        self.last_parameters_update = 0
        self.parameters_update_interval = 1.0  # Update parameters every 1 second

        # Adjustable frame size attributes - replaces hardcoded 750x750
        self.buffer_frame_width = 750  # Current buffer frame width
        self.buffer_frame_height = 750  # Current buffer frame height
        self.default_frame_width = 750  # Default frame size
        self.default_frame_height = 750

        # Auto-stop video buffering attributes
        self.last_video_request_time = None
        self.video_idle_timeout = 1  # Increase to 1 seconds to prevent rapid cycling
        self.video_idle_check_task = None
        self.webrtc_connected = False
        self.buffering_start_time = None
        self.min_buffering_duration = 1.0  # Minimum time to keep buffering active

        # Scanning control attributes
        self.scanning_in_progress = False  # Flag to prevent video buffering during scans

        # Unified scan state tracking (single scan at a time)
        self.scan_state = {
            'state': 'idle',  # idle, running, completed, failed
            'error_message': None,
            'scan_task': None,  # asyncio.Task for the running scan
            'saved_data_type': None,  # raw_images, full_zarr, quick_zarr
        }

        # Segmentation state tracking (single segmentation operation at a time)
        self.segmentation_state = {
            'state': 'idle',  # idle, running, completed, failed
            'error_message': None,
            'segmentation_task': None,  # asyncio.Task for the running segmentation
            'progress': {
                'total_wells': 0,
                'completed_wells': 0,
                'current_well': None,
                'source_channel': None,
                'source_experiment': None,
                'segmentation_experiment': None,
                # Similarity search tracking
                'similarity_search_enabled': False,
                'similarity_search_status': 'idle',  # idle, extracting, embedding, uploading, completed, failed
                'total_polygons': 0,
                'extracted_count': 0,
                'embedding_count': 0,
                'uploaded_count': 0,
                'failed_count': 0,
                'current_stage': None  # e.g., "Extracting images", "Generating embeddings", etc.
            }
        }

        # Initialize coverage tracking if in test mode
        if os.environ.get('SQUID_TEST_MODE'):
            self.coverage_enabled = True
            print("✅ Service coverage tracking enabled")
        else:
            self.coverage_enabled = False

    def load_authorized_emails(self):
        """Load authorized user emails from environment variable.

        Returns:
            list: List of authorized email addresses, or None if no restrictions
        """
        authorized_users = os.environ.get("AUTHORIZED_USERS")

        if not authorized_users:
            logger.info("No AUTHORIZED_USERS environment variable set - allowing all authenticated users")
            return None

        try:
            # Parse the AUTHORIZED_USERS environment variable as a list of emails
            if isinstance(authorized_users, str):
                # Handle comma-separated string format (primary format)
                if ',' in authorized_users:
                    authorized_emails = [email.strip() for email in authorized_users.split(',') if email.strip()]
                else:
                    # Single email without comma
                    authorized_emails = [authorized_users.strip()] if authorized_users.strip() else []
            else:
                # If it's already a list, use it directly
                authorized_emails = authorized_users

            # Validate that we have a list of strings
            if not isinstance(authorized_emails, list):
                logger.warning("AUTHORIZED_USERS must be a list of emails - allowing all authenticated users")
                return None

            # Filter out empty strings and validate email format
            valid_emails = []
            for email in authorized_emails:
                if isinstance(email, str) and email.strip() and '@' in email:
                    valid_emails.append(email.strip())
                else:
                    logger.warning(f"Skipping invalid email format: {email}")

            if valid_emails:
                logger.info(f"Loaded {len(valid_emails)} authorized emails from AUTHORIZED_USERS")
                return valid_emails
            else:
                logger.warning("No valid emails found in AUTHORIZED_USERS - allowing all authenticated users")
                return None

        except Exception as e:
            logger.error(f"Error parsing AUTHORIZED_USERS environment variable: {e} - allowing all authenticated users")
            return None

    def check_permission(self, user):
        # Handle None or empty user context
        if self.is_simulation:
            logger.info("No user context provided in simulation mode - allowing access")
            return True

        # If no authorized emails are set, allow all authenticated users
        if self.authorized_emails is None:
            return True

        # Check if user email is in authorized list
        user_email = user.get("email")
        if not user_email:
            logger.warning("No email found in user context - denying access")
            return False

        if user_email in self.authorized_emails:
            return True
        else:
            return False

    def _is_squid_plus_microscope(self):
        """
        Determine if this is a Squid+ microscope based on configuration settings.
        Squid+ microscopes have specific features like filter wheel and objective switcher.
        """
        try:
            from squid_control.control.config import CONFIG

            # Check for Squid+ specific configuration settings
            has_filter_wheel = getattr(CONFIG, 'FILTER_CONTROLLER_ENABLE', False)
            has_objective_switcher = getattr(CONFIG, 'USE_XERYON', False)

            # If either Squid+ specific feature is enabled, it's a Squid+ microscope
            is_squid_plus = has_filter_wheel or has_objective_switcher

            if is_squid_plus:
                logger.info("🔬 Detected Squid+ microscope - Squid+ specific endpoints will be registered")
            else:
                logger.info("🔬 Detected original Squid microscope - Squid+ specific endpoints will not be registered")

            return is_squid_plus

        except Exception as e:
            logger.warning(f"Could not determine microscope type: {e} - assuming original Squid")
            return False

    async def is_service_healthy(self, context=None):
        """Check if all services are healthy"""
        try:
            microscope_svc = await self.server.get_service(self.service_id)
            if microscope_svc is None:
                raise RuntimeError("Microscope service not found")

            result = await microscope_svc.ping()
            if result != "pong":
                raise RuntimeError(f"Microscope service returned unexpected response: {result}")

            # Check artifact manager connection if available
            if self.artifact_manager is not None:
                try:
                    # Test artifact manager connection by listing galleries
                    # Use a simple gallery listing to test the connection
                    default_gallery_id = "agent-lens/microscope-snapshots"
                    gallery_contents = await self.artifact_manager._svc.list(parent_id=default_gallery_id)
                    if gallery_contents is None:
                        logger.warning("Artifact manager health check: No gallery_contents found, but connection is working")
                    else:
                        logger.info(f"Artifact manager health check: Found {len(gallery_contents)} datasets")
                except Exception as artifact_error:
                    logger.warning(f"Artifact manager health check failed: {str(artifact_error)}")
                    # Don't raise an error for artifact manager issues as it's not critical for basic operation
            else:
                logger.info("Artifact manager not initialized, skipping health check")

            logger.info("All services are healthy")
            return {"status": "ok", "message": "All services are healthy"}
        except Exception as e:
            logger.error(f"Health check failed: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
            raise RuntimeError(f"Service health check failed: {str(e)}")

    @schema_function(skip_self=True)
    def ping(self, context=None):
        """
        Check if the microscope service is responsive.
        Returns: String 'pong' confirming service availability.
        """
        return "pong"

    @schema_function(skip_self=True)
    def move_by_distance(self, x: float=Field(1.0, description="Distance to move along X axis in millimeters (positive=right, negative=left)"), y: float=Field(1.0, description="Distance to move along Y axis in millimeters (positive=downside, negative=upside)"), z: float=Field(1.0, description="Distance to move along Z axis in millimeters (positive=up toward sample, negative=down)"), context=None):
        """
        Move the microscope stage by specified distances relative to current position.
        Returns: Dictionary with success status, movement message, initial position, and final position in millimeters.
        Notes: Movement is validated against software safety limits and will fail if target position is out of range.
        """
        try:
            # Check authentication
            if context and not self.check_permission(context.get("user", {})):
                raise Exception("User not authorized to access this service")

            is_success, x_pos, y_pos, z_pos, x_des, y_des, z_des = self.squidController.move_by_distance_limited(x, y, z)
            if is_success:
                result = f'The stage moved ({x},{y},{z})mm through x,y,z axis, from ({x_pos},{y_pos},{z_pos})mm to ({x_des},{y_des},{z_des})mm'
                return {
                    "success": True,
                    "message": result,
                    "initial_position": {"x": x_pos, "y": y_pos, "z": z_pos},
                    "final_position": {"x": x_des, "y": y_des, "z": z_des}
                }
            else:
                result = f'The stage cannot move ({x},{y},{z})mm through x,y,z axis, from ({x_pos},{y_pos},{z_pos})mm to ({x_des},{y_des},{z_des})mm because out of the range.'
                raise Exception(result)
        except Exception as e:
            logger.error(f"Failed to move by distance: {e}")
            raise e

    @schema_function(skip_self=True)
    def move_to_position(self, x:float=Field(1.0, description="Absolute X coordinate in millimeters (0 disables X movement)"), y:float=Field(1.0, description="Absolute Y coordinate in millimeters (0 disables Y movement)"), z:float=Field(1.0, description="Absolute Z coordinate in millimeters (0 disables Z movement)"), context=None):
        """
        Move the microscope stage to absolute coordinates in the stage reference frame.
        Returns: Dictionary with success status, movement message, initial position, and final position in millimeters.
        Notes: Each axis is moved sequentially (X, then Y, then Z). Movement validates against safety limits per axis.
        """
        try:
            # Check authentication
            if context and not self.check_permission(context.get("user", {})):
                raise Exception("User not authorized to access this service")

            self.get_status()
            initial_x = self.parameters['current_x']
            initial_y = self.parameters['current_y']
            initial_z = self.parameters['current_z']

            if x != 0:
                is_success, x_pos, y_pos, z_pos, x_des = self.squidController.move_x_to_limited(x)
                if not is_success:
                    raise Exception(f'The stage cannot move to position ({x},{y},{z})mm from ({initial_x},{initial_y},{initial_z})mm because out of the limit of X axis.')

            if y != 0:
                is_success, x_pos, y_pos, z_pos, y_des = self.squidController.move_y_to_limited(y)
                if not is_success:
                    raise Exception(f'X axis moved successfully, the stage is now at ({x_pos},{y_pos},{z_pos})mm. But aimed position is out of the limit of Y axis and the stage cannot move to position ({x},{y},{z})mm.')

            if z != 0:
                is_success, x_pos, y_pos, z_pos, z_des = self.squidController.move_z_to_limited(z)
                if not is_success:
                    raise Exception(f'X and Y axis moved successfully, the stage is now at ({x_pos},{y_pos},{z_pos})mm. But aimed position is out of the limit of Z axis and the stage cannot move to position ({x},{y},{z})mm.')

            return {
                "success": True,
                "message": f'The stage moved to position ({x},{y},{z})mm from ({initial_x},{initial_y},{initial_z})mm successfully.',
                "initial_position": {"x": initial_x, "y": initial_y, "z": initial_z},
                "final_position": {"x": x_pos, "y": y_pos, "z": z_pos}
            }
        except Exception as e:
            logger.error(f"Failed to move to position: {e}")
            raise e

    @schema_function(skip_self=True)
    def get_status(self, context=None):
        """
        Retrieve comprehensive microscope status including position, illumination, and scan state.
        Returns: Dictionary with stage position (x, y, z, theta in mm), illumination state, current channel, video buffering status, well location, and scan status.
        """
        try:
            # Check authentication
            if context and not self.check_permission(context.get("user", {})):
                raise Exception("User not authorized to access this service")

            current_x, current_y, current_z, current_theta = self.squidController.navigationController.update_pos(microcontroller=self.squidController.microcontroller)
            is_illumination_on = self.squidController.liveController.illumination_on
            #scan_channel = self.squidController.multipointController.selected_configurations
            # Get current well location information
            well_info = self.squidController.get_well_from_position('96')  # Default to 96-well plate

            self.parameters = {
                'current_x': current_x,
                'current_y': current_y,
                'current_z': current_z,
                'current_theta': current_theta,
                'is_illumination_on': is_illumination_on,
                'dx': self.dx,
                'dy': self.dy,
                'dz': self.dz,
                'current_channel': self.squidController.current_channel,
                'current_channel_name': self.channel_param_map[self.squidController.current_channel],
                'BF_intensity_exposure': self.BF_intensity_exposure,
                'F405_intensity_exposure': self.F405_intensity_exposure,
                'F488_intensity_exposure': self.F488_intensity_exposure,
                'F561_intensity_exposure': self.F561_intensity_exposure,
                'F638_intensity_exposure': self.F638_intensity_exposure,
                'F730_intensity_exposure': self.F730_intensity_exposure,
                'video_fps': self.buffer_fps,
                'video_buffering_active': self.frame_acquisition_running,
                'current_well_location': well_info,  # Add well location information
                # Unified scan status
                'scan_status': {
                    'state': self.scan_state['state'],
                    'saved_data_type': self.scan_state['saved_data_type'],
                    'error_message': self.scan_state['error_message']
                }
            }
            return self.parameters
        except Exception as e:
            logger.error(f"Failed to get status: {e}")
            raise e

    @schema_function(skip_self=True)
    def update_parameters_from_client(self,
                                    new_parameters: dict = Field(..., description="Dictionary of parameters to update with key-value pairs (e.g., {'BF_intensity_exposure': [28, 20], 'dx': 0.5})"),
                                    context=None):
        """
        Update microscope acquisition parameters for channels, step sizes, and illumination settings.
        Returns: Dictionary with success status, count of updated parameters, updated parameter values, and any failed parameter updates.
        Notes: Only updates parameters that exist in the current parameter set. Changes take effect immediately for subsequent acquisitions.
        """
        try:
            # Check authentication
            if context and not self.check_permission(context.get("user", {})):
                raise Exception("User not authorized to access this service")

            if not isinstance(new_parameters, dict):
                raise ValueError("new_parameters must be a dictionary")

            if not new_parameters:
                raise ValueError("new_parameters cannot be empty")

            if self.parameters is None:
                self.parameters = {}

            updated_params = {}
            failed_params = {}

            # Update only the specified keys
            for key, value in new_parameters.items():
                if key in self.parameters:
                    self.parameters[key] = value
                    updated_params[key] = value
                    logger.info(f"Updated parameter '{key}' to '{value}'")

                    # Update the corresponding instance variable if it exists
                    if hasattr(self, key):
                        setattr(self, key, value)
                    else:
                        logger.warning(f"Instance attribute '{key}' does not exist, parameter updated in config only")
                else:
                    failed_params[key] = f"Parameter '{key}' not found in available parameters"
                    logger.warning(f"Parameter '{key}' not found in available parameters")

            result = {
                "success": True,
                "message": f"Updated {len(updated_params)} parameters successfully",
                "updated_parameters": updated_params
            }

            if failed_params:
                result["failed_parameters"] = failed_params
                result["message"] += f", {len(failed_params)} parameters failed to update"

            return result

        except Exception as e:
            logger.error(f"Failed to update parameters: {e}")
            raise e

    def set_simulated_sample_data_alias(self, sample_data_alias: str="agent-lens/20250824-example-data-20250824-221822", context=None):
        """
        Configure which virtual Zarr sample dataset to use for simulation mode imaging.
        Returns: String confirmation message with the set sample alias.
        Notes: Only functional in simulation mode. Changes which Zarr-based virtual sample appears under the microscope.
        """
        self.squidController.set_simulated_sample_data_alias(sample_data_alias)
        return f"The alias of simulated sample is set to {sample_data_alias}"

    def get_simulated_sample_data_alias(self, context=None):
        """
        Query the currently active virtual sample dataset alias for simulation mode.
        Returns: String with Zarr dataset alias (format: 'workspace/dataset-id').
        Notes: Only relevant in simulation mode.
        """
        return self.squidController.get_simulated_sample_data_alias()

    @schema_function(skip_self=True)
    def set_wellplate_offset(self, 
                            offset_x_mm: float = Field(0.0, description="X-axis offset in millimeters"), 
                            offset_y_mm: float = Field(0.0, description="Y-axis offset in millimeters"),
                            context=None):
        """Set wellplate offset values for X and Y axes. Offsets are added to well positions in non-simulation mode."""
        from configparser import ConfigParser
        from pathlib import Path
        from .control.config import load_config
        
        CONFIG.WELLPLATE_OFFSET_X_MM = float(offset_x_mm)
        CONFIG.WELLPLATE_OFFSET_Y_MM = float(offset_y_mm)
        
        # Get config file path
        config_file_path = None
        try:
            if CONFIG.CACHE_CONFIG_FILE_PATH and os.path.exists(CONFIG.CACHE_CONFIG_FILE_PATH):
                with open(CONFIG.CACHE_CONFIG_FILE_PATH) as f:
                    config_file_path = f.read().strip()
        except Exception:
            pass
        
        if not config_file_path or not os.path.exists(config_file_path):
            config_dir = Path(__file__).parent.parent
            for name in ["configuration_HCS_v2.ini", "configuration_Squid+.ini"]:
                for path in [config_dir / name, config_dir / "config" / name]:
                    if path.exists():
                        config_file_path = str(path)
                        break
                if config_file_path:
                    break
        
        # Update INI file
        if config_file_path and os.path.exists(config_file_path):
            cfp = ConfigParser()
            cfp.read(config_file_path)
            # Find section with wellplate_offset or use GENERAL
            section = next((s for s in cfp.sections() + ['DEFAULT'] if cfp.has_option(s, 'wellplate_offset_x_mm')), 
                          'GENERAL' if 'GENERAL' in cfp.sections() else (cfp.sections()[0] if cfp.sections() else 'DEFAULT'))
            if section not in cfp.sections() and section != 'DEFAULT':
                cfp.add_section(section)
            cfp.set(section, 'wellplate_offset_x_mm', str(offset_x_mm))
            cfp.set(section, 'wellplate_offset_y_mm', str(offset_y_mm))
            with open(config_file_path, 'w') as f:
                cfp.write(f)
            logger.info(f"Updated config file with wellplate offsets: X={offset_x_mm} mm, Y={offset_y_mm} mm")
        
        # Reload configuration
        load_config(config_file_path, False)
        return {"offset_x_mm": CONFIG.WELLPLATE_OFFSET_X_MM, "offset_y_mm": CONFIG.WELLPLATE_OFFSET_Y_MM}

    async def one_new_frame(self, context=None):
        """
        Get an image from the microscope
        Returns: A numpy array with preserved bit depth
        """
        # Check authentication
        if context and not self.check_permission(context.get("user", {})):
            raise Exception("User not authorized to access this service")

        # Stop video buffering to prevent camera overload
        if self.frame_acquisition_running:
            logger.info("Stopping video buffering for one_new_frame operation to prevent camera conflicts")
            await self.stop_video_buffering()
            # Wait a moment for the buffering to fully stop
            await asyncio.sleep(0.1)

        channel = self.squidController.current_channel
        intensity, exposure_time = 50, 100  # Default values
        try:
            #update the current illumination channel and intensity
            param_name = self.channel_param_map.get(channel)
            if param_name:
                stored_params = getattr(self, param_name, None)
                if stored_params and isinstance(stored_params, list) and len(stored_params) == 2:
                    intensity, exposure_time = stored_params
                else:
                    logger.warning(f"Parameter {param_name} for channel {channel} is not properly initialized. Using defaults.")
            else:
                logger.warning(f"Unknown channel {channel} in one_new_frame. Using default intensity/exposure.")

            # Get the image from camera - snap_image() returns already cropped image by default
            # This is exactly what snap() does
            gray_img = await self.squidController.snap_image(channel, intensity, exposure_time)

            self.get_status()

            # Return the numpy array directly with preserved bit depth
            return gray_img

        except Exception as e:
            logger.error(f"Failed to get new frame: {e}")
            raise e

    async def get_video_frame(self, frame_width: int=750, frame_height: int=750, context=None):
        """
        Get compressed frame data with metadata from the microscope using video buffering
        Returns: Compressed frame data (JPEG bytes) with associated metadata including stage position and timestamp
        """
        try:
            # Check authentication
            if context and not self.check_permission(context.get("user", {})):
                raise Exception("User not authorized to access this service")

            # If scanning is in progress, return a scanning placeholder immediately
            if self.scanning_in_progress:
                logger.debug("Scanning in progress, returning scanning placeholder frame")
                placeholder = VideoFrameProcessor._create_placeholder_frame(frame_width, frame_height, "Scanning in Progress...")
                placeholder_compressed = VideoFrameProcessor.encode_frame_jpeg(placeholder, quality=85)

                # Create metadata for scanning placeholder frame
                scanning_metadata = {
                    'stage_position': {'x_mm': None, 'y_mm': None, 'z_mm': None},
                    'timestamp': time.time(),
                    'channel': None,
                    'intensity': None,
                    'exposure_time_ms': None,
                    'scanning_status': 'in_progress'
                }

                return {
                    'format': placeholder_compressed['format'],
                    'data': placeholder_compressed['data'],
                    'width': frame_width,
                    'height': frame_height,
                    'size_bytes': placeholder_compressed['size_bytes'],
                    'compression_ratio': placeholder_compressed.get('compression_ratio', 1.0),
                    'metadata': scanning_metadata
                }

            # Update last video request time for auto-stop functionality (only when not scanning)
            self.last_video_request_time = time.time()

            # Start video buffering if not already running and not scanning
            if not self.frame_acquisition_running:
                logger.info("Starting video buffering for remote video frame request")
                await self.start_video_buffering()

            # Start idle checking task if not running
            if self.video_idle_check_task is None or self.video_idle_check_task.done():
                self.video_idle_check_task = asyncio.create_task(self._monitor_video_idle())

            # Get compressed frame data and metadata from buffer
            frame_data, frame_metadata = self.video_buffer.get_frame_data()

            if frame_data is not None:
                # Check if we need to resize the frame
                # Use current buffer frame size instead of hardcoded values
                buffered_width = self.buffer_frame_width
                buffered_height = self.buffer_frame_height

                if frame_width != buffered_width or frame_height != buffered_height:
                    # Need to resize - decompress, resize, and recompress
                    decompressed_frame = VideoFrameProcessor.decode_frame_jpeg(frame_data)
                    if decompressed_frame is not None:
                        # Resize the frame to requested dimensions
                        resized_frame = cv2.resize(decompressed_frame, (frame_width, frame_height), interpolation=cv2.INTER_AREA)
                        # Recompress at requested size
                        resized_compressed = VideoFrameProcessor.encode_frame_jpeg(resized_frame, quality=85)
                        return {
                            'format': resized_compressed['format'],
                            'data': resized_compressed['data'],
                            'width': frame_width,
                            'height': frame_height,
                            'size_bytes': resized_compressed['size_bytes'],
                            'compression_ratio': resized_compressed.get('compression_ratio', 1.0),
                            'metadata': frame_metadata
                        }
                    else:
                        # Fallback to placeholder if decompression fails
                        placeholder = VideoFrameProcessor._create_placeholder_frame(frame_width, frame_height, "Frame decompression failed")
                        placeholder_compressed = VideoFrameProcessor.encode_frame_jpeg(placeholder, quality=85)
                        return {
                            'format': placeholder_compressed['format'],
                            'data': placeholder_compressed['data'],
                            'width': frame_width,
                            'height': frame_height,
                            'size_bytes': placeholder_compressed['size_bytes'],
                            'compression_ratio': placeholder_compressed.get('compression_ratio', 1.0),
                            'metadata': frame_metadata
                        }
                else:
                    # Return buffered frame directly (no resize needed)
                    return {
                        'format': frame_data['format'],
                        'data': frame_data['data'],
                        'width': frame_width,
                        'height': frame_height,
                        'size_bytes': frame_data['size_bytes'],
                        'compression_ratio': frame_data.get('compression_ratio', 1.0),
                        'metadata': frame_metadata
                    }
            else:
                # No buffered frame available, create and compress placeholder
                logger.warning("No buffered frame available")
                placeholder = VideoFrameProcessor._create_placeholder_frame(frame_width, frame_height, "No buffered frame available")
                placeholder_compressed = VideoFrameProcessor.encode_frame_jpeg(placeholder, quality=85)

                # Create metadata for placeholder frame
                placeholder_metadata = {
                    'stage_position': {'x_mm': None, 'y_mm': None, 'z_mm': None},
                    'timestamp': time.time(),
                    'channel': None,
                    'intensity': None,
                    'exposure_time_ms': None,
                    'error': 'No buffered frame available'
                }

                return {
                    'format': placeholder_compressed['format'],
                    'data': placeholder_compressed['data'],
                    'width': frame_width,
                    'height': frame_height,
                    'size_bytes': placeholder_compressed['size_bytes'],
                    'compression_ratio': placeholder_compressed.get('compression_ratio', 1.0),
                    'metadata': placeholder_metadata
                }

        except Exception as e:
            logger.error(f"Error getting video frame: {e}", exc_info=True)
            # Create error placeholder and compress it
            raise e

    def configure_video_buffer(self, buffer_fps: int = 5, buffer_size: int = 5, context=None):
        """Configure video buffering parameters for optimal streaming performance."""
        try:
            self.buffer_fps_target = max(1, min(30, buffer_fps))  # Clamp between 1-30 FPS

            # Update buffer size
            old_size = self.frame_buffer.maxlen
            self.frame_buffer = deque(maxlen=max(1, min(20, buffer_size)))  # Clamp between 1-20 frames

            logger.info(f"Video buffer configured: FPS={self.buffer_fps_target}, buffer_size={self.frame_buffer.maxlen} (was {old_size})")

            return {
                "success": True,
                "message": f"Video buffer configured with {self.buffer_fps_target} FPS target and {self.frame_buffer.maxlen} frame buffer size",
                "buffer_fps": self.buffer_fps_target,
                "buffer_size": self.frame_buffer.maxlen
            }
        except Exception as e:
            logger.error(f"Failed to configure video buffer: {e}")
            raise e




    def configure_video_idle_timeout(self, idle_timeout: float = 5.0, context=None):
        """Configure how long to wait before automatically stopping video buffering when inactive."""
        try:
            self.video_idle_timeout = max(0, idle_timeout)  # Ensure non-negative
            logger.info(f"Video idle timeout set to {self.video_idle_timeout} seconds")

            return {
                "success": True,
                "message": f"Video idle timeout configured to {self.video_idle_timeout} seconds",
                "idle_timeout": self.video_idle_timeout,
                "automatic_stop": self.video_idle_timeout > 0
            }
        except Exception as e:
            logger.error(f"Failed to configure video idle timeout: {e}")
            raise e

    async def set_video_fps(self, fps: int = 5, context=None):
        """
        Set the video acquisition frame rate for smooth streaming.
        This controls how fast the microscope acquires frames for video streaming.
        Higher FPS provides smoother video but uses more resources.
        """

        try:
            # Check authentication
            if context and not self.check_permission(context.get("user", {})):
                raise Exception("User not authorized to access this service")

            # Validate FPS range
            if not isinstance(fps, int) or fps < 1 or fps > 30:
                raise ValueError(f"Invalid FPS value: {fps}. Must be an integer between 1 and 30.")

            # Store old FPS for comparison
            old_fps = self.buffer_fps
            was_running = self.frame_acquisition_running

            # Update FPS setting
            self.buffer_fps = fps
            logger.info(f"Video FPS updated from {old_fps} to {fps}")

            # Update any active WebRTC video tracks with the new FPS
            if hasattr(self, 'video_track') and self.video_track is not None:
                self.video_track.update_fps(fps)
                logger.info("Updated WebRTC video track FPS")

            # If video buffering is currently running, restart it with new FPS
            if was_running:
                logger.info("Restarting video buffering with new FPS settings")
                await self.stop_video_buffering()
                # Brief pause to ensure clean shutdown
                await asyncio.sleep(0.2)
                await self.start_video_buffering()
                logger.info(f"Video buffering restarted with {fps} FPS")

            return {
                "success": True,
                "message": f"Video FPS successfully updated from {old_fps} to {fps} FPS",
                "old_fps": old_fps,
                "new_fps": fps,
                "buffering_restarted": was_running
            }

        except Exception as e:
            logger.error(f"Failed to set video FPS: {e}")
            raise e



    def _reset_video_activity_tracking(self):
        """Reset video activity tracking (internal method)."""
        self.last_video_request_time = None
        logger.info("Video activity tracking reset")

    async def cleanup_for_tests(self):
        """Cleanup method specifically for test environments."""
        try:
            # Stop video buffering if running
            if self.buffer_acquisition_running:
                logger.info("Stopping video buffering for test cleanup")
                await self.stop_frame_buffer_acquisition()

            # Close camera resources properly (if any)
            if hasattr(self, 'squidController') and self.squidController:
                if hasattr(self.squidController, 'camera') and self.squidController.camera:
                    logger.info("Camera resources available for cleanup")

            # Log coverage tracking status
            if hasattr(self, 'coverage_enabled') and self.coverage_enabled:
                logger.info("Service coverage tracking was active during test")
        except Exception as e:
            logger.error(f"Error during test cleanup: {e}")

    async def start_video_buffering(self, context=None):
        """Start video buffering for smooth video streaming"""
        try:
            # Check authentication
            if context and not self.check_permission(context.get("user", {})):
                raise Exception("User not authorized to access this service")

            await self._start_video_buffering_internal()
            return {"success": True, "message": "Video buffering started successfully"}
        except Exception as e:
            logger.error(f"Failed to start video buffering: {e}")
            raise e

    async def stop_video_buffering(self, context=None):
        """Manually stop video buffering to save resources."""
        try:
            # Check authentication
            if context and not self.check_permission(context.get("user", {})):
                raise Exception("User not authorized to access this service")

            if not self.frame_acquisition_running:
                return {
                    "success": True,
                    "message": "Video buffering is already stopped",
                    "was_already_stopped": True
                }

            await self._stop_video_buffering_internal()
            logger.info("Video buffering stopped manually")

            return {
                "success": True,
                "message": "Video buffering stopped successfully"
            }
        except Exception as e:
            logger.error(f"Failed to stop video buffering: {e}")
            raise e

    def get_video_buffering_status(self, context=None):
        """Get the current video buffering status"""
        try:
            # Check authentication
            if context and not self.check_permission(context.get("user", {})):
                raise Exception("User not authorized to access this service")

            buffer_size = len(self.video_buffer.buffer) if self.video_buffer else 0
            frame_age = self.video_buffer.get_frame_age() if self.video_buffer else float('inf')

            return {
                "buffering_active": self.frame_acquisition_running,
                "buffer_size": buffer_size,
                "max_buffer_size": self.video_buffer.max_size if self.video_buffer else 0,
                "frame_age_seconds": frame_age if frame_age != float('inf') else None,
                "buffer_fps": self.buffer_fps,
                "has_frames": buffer_size > 0
            }
        except Exception as e:
            logger.error(f"Failed to get video buffering status: {e}")
            return {
                "buffering_active": False,
                "buffer_size": 0,
                "max_buffer_size": 0,
                "frame_age_seconds": None,
                "buffer_fps": 0,
                "has_frames": False,
                "error": str(e)
            }

    def adjust_video_frame(self, min_val: int = 0, max_val: Optional[int] = None, context=None):
        """Adjust the contrast of the video stream by setting min and max intensity values."""
        try:
            # Check authentication
            if context and not self.check_permission(context.get("user", {})):
                raise Exception("User not authorized to access this service")

            self.video_contrast_min = min_val
            self.video_contrast_max = max_val
            logger.info(f"Video contrast adjusted: min={min_val}, max={max_val}")
            return {"success": True, "message": f"Video contrast adjusted to min={min_val}, max={max_val}."}
        except Exception as e:
            logger.error(f"Failed to adjust video frame: {e}")
            raise e

    @schema_function(skip_self=True)
    async def snap(self, exposure_time: Optional[int]=Field(None, description="Camera exposure time in milliseconds (range: 1-900). If None, uses current microscope setting."), channel: Optional[str]=Field(None, description="Channel canonical name (e.g., 'BF_LED_matrix_full', 'Fluorescence_405_nm_Ex', 'Fluorescence_488_nm_Ex', 'Fluorescence_638_nm_Ex', 'Fluorescence_561_nm_Ex', 'Fluorescence_730_nm_Ex'). If None, uses current microscope channel."), intensity: Optional[int]=Field(None, description="LED illumination intensity percentage (range: 0-100). If None, uses current microscope setting."), context=None):
        """
        Capture a single microscope image and save it to artifact manager with public access.
        Returns: HTTP URL string pointing to the captured 2048x2048 PNG image with public read access.
        Notes: Stops video buffering during acquisition to prevent camera conflicts. Image is automatically cropped and resized.
        Snapshots are stored in daily datasets (snapshots-{service_id}-{date}) in the artifact manager with position metadata.
        If channel, intensity, or exposure_time are None, the current microscope settings will be used.
        The provided channel, intensity, and exposure_time settings are applied to the microscope and update the current parameters.
        """

        # Check authentication
        if context and not self.check_permission(context.get("user", {})):
            raise Exception("User not authorized to access this service")

        # Stop video buffering to prevent camera overload
        if self.frame_acquisition_running:
            logger.info("Stopping video buffering for snap operation to prevent camera conflicts")
            await self.stop_video_buffering()
            # Wait a moment for the buffering to fully stop
            await asyncio.sleep(0.1)

        try:
            # Convert channel (canonical name string) to channel_id (int) if provided, otherwise use current channel
            if channel is None:
                channel_id = self.squidController.current_channel
                # Get canonical name from current channel for metadata
                try:
                    channel = ChannelMapper.id_to_zarr_name(channel_id)
                    logger.info(f"Using current channel: {channel_id} ({channel})")
                except ValueError:
                    logger.warning(f"Current channel {channel_id} has no canonical name mapping")
                    channel = None
            else:
                # Convert channel (canonical name string) to channel_id (int)
                try:
                    channel_id = ChannelMapper.zarr_name_to_id(channel)
                    logger.info(f"Using channel: {channel_id} ({channel})")
                except ValueError:
                    valid_names = ChannelMapper.get_all_zarr_names()
                    raise Exception(f"Invalid channel: {channel}. Valid channel names are: {valid_names}")
            
            # Get channel parameter name to access current intensity and exposure
            param_name = self.channel_param_map.get(channel_id)
            if param_name is None:
                raise Exception(f"Invalid channel ID: {channel_id}. Valid channel IDs are: {list(self.channel_param_map.keys())}")
            
            # Get current intensity and exposure from channel parameters
            current_params = getattr(self, param_name, [50, 100])  # Default fallback
            if not (isinstance(current_params, list) and len(current_params) == 2):
                logger.warning(f"Parameter {param_name} for channel {channel_id} was not a list of two items. Using defaults.")
                current_params = [50, 100]
            
            if intensity is None:
                intensity = current_params[0]
                logger.info(f"Using current intensity: {intensity}")
            
            if exposure_time is None:
                # Get exposure time from channel parameters (second element is exposure_time)
                exposure_time = current_params[1]
                logger.info(f"Using current exposure_time: {exposure_time}")

            # Update current channel and parameters
            self.squidController.current_channel = channel_id
            if param_name:
                setattr(self, param_name, [intensity, exposure_time])

            # Capture image
            gray_img = await self.squidController.snap_image(channel_id, intensity, exposure_time)
            logger.info('The image is snapped')
            # Image is already uint8 from snap_image method
            # Resize the image to a standard size
            resized_img = cv2.resize(gray_img, (2048, 2048))

            # Encode the image directly to PNG without converting to BGR
            _, png_image = cv2.imencode('.png', resized_img)

            # Save using artifact manager (REQUIRED)
            if not self.snapshot_manager:
                raise Exception("Snapshot manager not available. Ensure AGENT_LENS_WORKSPACE_TOKEN is set.")

            # Get current position for metadata
            status = self.get_status()
            metadata = {
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime()),
                "channel_id": channel_id,
                "channel": channel,
                "intensity": intensity,
                "exposure_time": exposure_time,
                "position_x": status.get("current_x"),
                "position_y": status.get("current_y"),
                "position_z": status.get("current_z"),
                "microscope_service_id": self.service_id
            }

            # Save using artifact manager
            data_url = await self.snapshot_manager.save_snapshot(
                microscope_service_id=self.service_id,
                image_bytes=png_image.tobytes(),
                metadata=metadata
            )
            logger.info(f'The image is snapped and saved to artifact manager as {data_url}')

            return data_url
        except Exception as e:
            logger.error(f"Failed to snap image: {e}")
            raise e

    @schema_function(skip_self=True)
    def turn_on_illumination(self, context=None):
        """
        Turn on the microscope illumination using the currently set channel and intensity.
        Returns: String confirmation message that brightfield illumination is on.
        """
        try:
            # Check authentication
            if context and not self.check_permission(context.get("user", {})):
                raise Exception("User not authorized to access this service")

            self.squidController.liveController.turn_on_illumination()
            logger.info('Bright field illumination turned on.')
            return 'Bright field illumination turned on.'
        except Exception as e:
            logger.error(f"Failed to open illumination: {e}")
            raise e

    @schema_function(skip_self=True)
    def turn_off_illumination(self, context=None):
        """
        Turn off all microscope illumination sources.
        Returns: String confirmation message that illumination is off.
        """
        try:
            # Check authentication
            if context and not self.check_permission(context.get("user", {})):
                raise Exception("User not authorized to access this service")

            self.squidController.liveController.turn_off_illumination()
            logger.info('Illumination turned off.')
            return 'Illumination turned off.'
        except Exception as e:
            logger.error(f"Failed to close illumination: {e}")
            raise e

    async def scan_plate_save_raw_images(self, well_plate_type: str = "96", illumination_settings: List[dict] = None, do_contrast_autofocus: bool = False, do_reflection_af: bool = True, wells_to_scan: List[str] = None, Nx: int = 3, Ny: int = 3, dx: float = 0.8, dy: float = 0.8, action_ID: str = 'testPlateScan', context=None):
        """
        
        Scan the well plate according to the specified wells with custom illumination settings
        
        Args:
            well_plate_type: Type of well plate ('96', '384', etc.)
            illumination_settings: List of dictionaries with illumination settings
            do_contrast_autofocus: Whether to perform contrast-based autofocus
            do_reflection_af: Whether to perform reflection-based autofocus
            wells_to_scan: List of wells to scan (e.g., ['A1', 'B2', 'C3'])
            Nx: Number of X positions per well
            Ny: Number of Y positions per well
            dx: Distance between X positions in mm
            dy: Distance between Y positions in mm
            action_ID: Identifier for this scan
            
        Returns: The message of the action
        """
        logger.warning("DEPRECATED: scan_plate_save_raw_images is deprecated and will be removed in a future release. Use scan_start() with saved_data_type='raw_images' instead.")
        try:
            # Check authentication
            if context and not self.check_permission(context.get("user", {})):
                raise Exception("User not authorized to access this service")

            if illumination_settings is None:
                logger.warning("No illumination settings provided, using default settings")
                illumination_settings = [
                    {'channel': 'BF LED matrix full', 'intensity': 28.0, 'exposure_time': 20.0},
                    {'channel': 'Fluorescence 488 nm Ex', 'intensity': 27.0, 'exposure_time': 60.0},
                    {'channel': 'Fluorescence 561 nm Ex', 'intensity': 98.0, 'exposure_time': 100.0},
                ]

            if wells_to_scan is None:
                wells_to_scan = ['A1']

            # Check if video buffering is active and stop it during scanning
            video_buffering_was_active = self.frame_acquisition_running
            if video_buffering_was_active:
                logger.info("Video buffering is active, stopping it temporarily during well plate scanning")
                await self.stop_video_buffering()
                # Wait additional time to ensure camera fully settles after stopping video buffering
                logger.info("Waiting for camera to settle after stopping video buffering...")
                await asyncio.sleep(0.5)

            # Set scanning flag to prevent automatic video buffering restart during scan
            self.scanning_in_progress = True

            logger.info("Start scanning well plate with custom illumination settings")

            # Run the blocking plate_scan operation in a separate thread executor
            # This prevents the asyncio event loop from being blocked during long scans
            await asyncio.get_event_loop().run_in_executor(
                None,  # Use default ThreadPoolExecutor
                self.squidController.plate_scan,
                well_plate_type,
                illumination_settings,
                do_contrast_autofocus,
                do_reflection_af,
                wells_to_scan,
                Nx,
                Ny,
                dx,
                dy,
                action_ID
            )

            logger.info("Well plate scanning completed")
            return "Well plate scanning completed"
        except Exception as e:
            logger.error(f"Failed to scan well plate: {e}")
            raise e
        finally:
            # Always reset the scanning flag, regardless of success or failure
            self.scanning_in_progress = False
            logger.info("Well plate scanning completed, video buffering auto-start is now re-enabled")

    @schema_function(skip_self=True)
    def set_illumination(self, channel: int=Field(0, description="Illumination channel: 0=Brightfield, 11=405nm, 12=488nm, 13=638nm, 14=561nm, 15=730nm"), intensity: int=Field(50, description="LED illumination intensity percentage (range: 0-100)"), context=None):
        """
        Configure illumination channel and intensity for the microscope.
        Returns: String confirmation message with the channel number and set intensity.
        Notes: If illumination is already on, it will be toggled off then back on with new settings.
        """
        try:
            # Check authentication
            if context and not self.check_permission(context.get("user", {})):
                raise Exception("User not authorized to access this service")

            # if light is on, turn it off first
            if self.squidController.liveController.illumination_on:
                self.squidController.liveController.turn_off_illumination()
                time.sleep(0.005)
                self.squidController.liveController.set_illumination(channel, intensity)
                self.squidController.liveController.turn_on_illumination()
                time.sleep(0.005)
            else:
                self.squidController.liveController.set_illumination(channel, intensity)
                time.sleep(0.005)

            param_name = self.channel_param_map.get(channel)
            self.squidController.current_channel = channel
            if param_name:
                current_params = getattr(self, param_name, [intensity, 100]) # Default exposure if not found
                if not (isinstance(current_params, list) and len(current_params) == 2):
                    logger.warning(f"Parameter {param_name} for channel {channel} was not a list of two items. Resetting with default exposure.")
                    current_params = [intensity, 100] # Default exposure
                setattr(self, param_name, [intensity, current_params[1]])
            else:
                logger.warning(f"Unknown channel {channel} in set_illumination, parameters not updated for intensity attributes.")

            logger.info(f'The intensity of the channel {channel} illumination is set to {intensity}.')
            return f'The intensity of the channel {channel} illumination is set to {intensity}.'
        except Exception as e:
            logger.error(f"Failed to set illumination: {e}")
            raise e

    @schema_function(skip_self=True)
    def set_camera_exposure(self, channel: int=Field(..., description="Illumination channel to associate with this exposure: 0=Brightfield, 11=405nm, 12=488nm, 13=638nm, 14=561nm, 15=730nm"), exposure_time: int=Field(..., description="Camera exposure time in milliseconds (range: 1-900)"), context=None):
        """
        Configure camera exposure time for a specific illumination channel.
        Returns: String confirmation message with the set exposure time.
        Notes: Updates internal parameter storage for the specified channel. Changes apply to future acquisitions.
        """
        try:
            # Check authentication
            if context and not self.check_permission(context.get("user", {})):
                raise Exception("User not authorized to access this service")

            self.squidController.camera.set_exposure_time(exposure_time)

            param_name = self.channel_param_map.get(channel)
            self.squidController.current_channel = channel
            if param_name:
                current_params = getattr(self, param_name, [50, exposure_time]) # Default intensity if not found
                if not (isinstance(current_params, list) and len(current_params) == 2):
                    logger.warning(f"Parameter {param_name} for channel {channel} was not a list of two items. Resetting with default intensity.")
                    current_params = [50, exposure_time] # Default intensity
                setattr(self, param_name, [current_params[0], exposure_time])
            else:
                logger.warning(f"Unknown channel {channel} in set_camera_exposure, parameters not updated for exposure attributes.")

            logger.info(f'The exposure time of the camera is set to {exposure_time}.')
            return f'The exposure time of the camera is set to {exposure_time}.'
        except Exception as e:
            logger.error(f"Failed to set camera exposure: {e}")
            raise e

    @schema_function(skip_self=True)
    def stop_scan(self, context=None):
        """
        Abort the currently running well plate scan operation.
        Returns: String confirmation message that scanning has stopped.
        Notes: Deprecated - use scan_cancel() for unified scan operations instead.
        """
        try:
            # Check authentication
            if context and not self.check_permission(context.get("user", {})):
                raise Exception("User not authorized to access this service")

            self.squidController.liveController.stop_live()
            self.multipointController.abort_acqusition_requested=True
            logger.info("Stop scanning well plate")
            return "Stop scanning well plate"
        except Exception as e:
            logger.error(f"Failed to stop scan: {e}")
            raise e

    @schema_function(skip_self=True)
    async def home_stage(self, context=None):
        """
        Execute homing sequence to move stage to hardware home/zero position on all axes.
        Returns: String confirmation message that stage has moved to home position in Z, Y, and X axes.
        Notes: Runs in background thread to prevent event loop blocking. Stage moves sequentially in Z, Y, X order for safety.
        """
        try:
            # Check authentication
            if context and not self.check_permission(context.get("user", {})):
                raise Exception("User not authorized to access this service")

            # Run the blocking home_stage operation in a separate thread executor
            # This prevents the asyncio event loop from being blocked during homing
            await asyncio.get_event_loop().run_in_executor(
                None,  # Use default ThreadPoolExecutor
                self.squidController.home_stage
            )
            logger.info('The stage moved to home position in z, y, and x axis')
            return 'The stage moved to home position in z, y, and x axis'
        except Exception as e:
            logger.error(f"Failed to home stage: {e}")
            raise e

    @schema_function(skip_self=True)
    async def return_stage(self, context=None):
        """
        Move the stage to the configured initial imaging position.
        Returns: String confirmation message that stage has moved to the initial position.
        Notes: Runs in background thread to prevent event loop blocking. Initial position is defined in configuration.
        """

        try:
            # Check authentication
            if context and not self.check_permission(context.get("user", {})):
                raise Exception("User not authorized to access this service")

            # Run the blocking return_stage operation in a separate thread executor
            # This prevents the asyncio event loop from being blocked during stage movement
            await asyncio.get_event_loop().run_in_executor(
                None,  # Use default ThreadPoolExecutor
                self.squidController.return_stage
            )
            logger.info('The stage moved to the initial position')
            return 'The stage moved to the initial position'
        except Exception as e:
            logger.error(f"Failed to return stage: {e}")
            raise e

    @schema_function(skip_self=True)
    async def move_to_loading_position(self, context=None):
        """
        Move the stage to the slide loading position for safe sample insertion/removal.
        Returns: String confirmation message that stage has moved to loading position.
        Notes: Runs in background thread to prevent event loop blocking. Loading position provides clearance for manual sample handling.
        """

        try:
            # Check authentication
            if context and not self.check_permission(context.get("user", {})):
                raise Exception("User not authorized to access this service")

            # Run the blocking move_to_slide_loading_position operation in a separate thread executor
            # This prevents the asyncio event loop from being blocked during stage movement
            await asyncio.get_event_loop().run_in_executor(
                None,  # Use default ThreadPoolExecutor
                self.squidController.slidePositionController.move_to_slide_loading_position
            )
            logger.info('The stage moved to loading position')
            return 'The stage moved to loading position'
        except Exception as e:
            logger.error(f"Failed to move to loading position: {e}")
            raise e

    @schema_function(skip_self=True)
    async def contrast_autofocus(self, context=None):
        """
        Perform contrast-based autofocus by analyzing image sharpness across Z positions.
        Returns: String confirmation message that camera is auto-focused.
        Notes: Scans through Z range, calculates focus metrics, and moves to optimal position. Best for samples with visible features.
        """

        try:
            # Check authentication
            if context and not self.check_permission(context.get("user", {})):
                raise Exception("User not authorized to access this service")

            await self.squidController.contrast_autofocus()
            logger.info('The camera is auto-focused')
            return 'The camera is auto-focused'
        except Exception as e:
            logger.error(f"Failed to auto focus: {e}")
            raise e

    @schema_function(skip_self=True)
    async def reflection_autofocus(self, context=None):
        """
        Perform reflection-based (laser) autofocus using IR laser reflection from sample surface.
        Returns: String confirmation message that camera is auto-focused.
        Notes: Fast and accurate focus method using laser displacement sensor. Requires prior reference setting. Best for flat samples.
        """

        try:
            # Check authentication
            if context and not self.check_permission(context.get("user", {})):
                raise Exception("User not authorized to access this service")

            await self.squidController.reflection_autofocus()
            logger.info('The camera is auto-focused')
            return 'The camera is auto-focused'
        except Exception as e:
            logger.error(f"Failed to do laser autofocus: {e}")
            raise e

    @schema_function(skip_self=True)
    async def autofocus_set_reflection_reference(self, context=None):
        """
        Calibrate the reflection autofocus system by setting current Z position as focus reference.
        Returns: String confirmation message that laser reference is set.
        Notes: Must be called at a focused position before using reflection_autofocus(). Reference is maintained until next calibration.
        """

        try:
            # Check authentication
            if context and not self.check_permission(context.get("user", {})):
                raise Exception("User not authorized to access this service")

            if self.is_simulation:
                pass
            else:
                # Run the potentially blocking set_reference operation in a separate thread executor
                # This prevents the asyncio event loop from being blocked during laser reference setting
                await asyncio.get_event_loop().run_in_executor(
                    None,  # Use default ThreadPoolExecutor
                    self.squidController.laserAutofocusController.set_reference
                )
            logger.info('The laser reference is set')
            return 'The laser reference is set'
        except Exception as e:
            logger.error(f"Failed to set laser reference: {e}")
            raise e

    @schema_function(skip_self=True)
    async def navigate_to_well(self, row: str=Field('A', description="Well row letter (e.g., 'A', 'B', 'C', ... 'H' for 96-well)"), col: int=Field(1, description="Well column number (e.g., 1-12 for 96-well)"), well_plate_type: str=Field('96', description="Well plate format: '6', '12', '24', '96', or '384'"), context=None):
        """
        Navigate the stage to the center of a specific well in the well plate.
        Returns: String confirmation message with the well position (e.g., 'The stage moved to well position (A,1)').
        Notes: Runs in background thread. Well coordinates are automatically calculated based on standard plate dimensions.
        """

        try:
            # Check authentication
            if context and not self.check_permission(context.get("user", {})):
                raise Exception("User not authorized to access this service")

            if well_plate_type is None:
                well_plate_type = '96'
            # Run the blocking move_to_well operation in a separate thread executor
            # This prevents the asyncio event loop from being blocked during stage movement
            await asyncio.get_event_loop().run_in_executor(
                None,  # Use default ThreadPoolExecutor
                self.squidController.move_to_well,
                row,
                col,
                well_plate_type
            )
            logger.info(f'The stage moved to well position ({row},{col})')
            return f'The stage moved to well position ({row},{col})'
        except Exception as e:
            logger.error(f"Failed to navigate to well: {e}")
            raise e

    async def fetch_ice_servers(self):
        """Fetch ICE servers from the coturn service"""
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get('https://ai.imjoy.io/public/services/coturn/get_rtc_ice_servers') as response:
                    if response.status == 200:
                        ice_servers = await response.json()
                        logger.info("Successfully fetched ICE servers")
                        return ice_servers
                    else:
                        logger.warning(f"Failed to fetch ICE servers, status: {response.status}")
                        return None
        except Exception as e:
            logger.error(f"Error fetching ICE servers: {e}")
            return None

    @schema_function(skip_self=True)
    async def inspect_tool(self, images: List[dict]=Field(..., description="A list of images to be inspected, each dictionary must contain 'http_url' (required) and optionally 'title' (optional)"), query: str=Field(..., description="User query about the images for GPT-5.1 vision model analysis"), context_description: str=Field(..., description="Context description for the visual inspection task, typically describing that images are taken from the microscope"), context=None):
        """
        Inspect images using GPT's vision model (GPT-5.1) for analysis and description.
        Returns: String response from the vision model containing image analysis based on the query.
        Notes: All image URLs must be HTTP/HTTPS accessible. The method validates URLs and processes images through the GPT-5.1 vision API.
        """
        try:
            from squid_control.hypha_tools.vision_inspection import inspect_images
            return await inspect_images(
                images=images,
                query=query,
                context_description=context_description,
                check_permission=self.check_permission,
                context=context
            )
        except Exception as e:
            logger.error(f"Failed to inspect images: {e}")
            raise e

    async def start_hypha_service(self, server, service_id, run_in_executor=None):
        self.server = server
        self.service_id = service_id

        # Default to True for production, False for tests (identified by "test" in service_id)
        if run_in_executor is None:
            run_in_executor = "test" not in service_id.lower()

        # Build the service configuration
        # Always require context for proper authentication and schema generation
        visibility = "public" if self.is_simulation else "protected"
        require_context = True  # Always require context for consistent schema

        if self.is_simulation:
            logger.info("Running in simulation mode: service will be public but require context")
        else:
            logger.info("Running in production mode: service will be protected and require context")

        # Generate description based on simulation mode and microscope type
        if self.is_simulation:
            description = "A microscope control service for the Squid automated microscope operating in simulation mode. This service provides complete control over stage positioning, multi-channel illumination (brightfield and fluorescence), camera operations, autofocus systems, and well plate navigation. In simulation mode, the service uses virtual Zarr-based sample data to provide realistic microscope behavior without physical hardware, enabling development, testing, and demonstration of advanced microscopy workflows including automated scanning, image stitching, and multi-channel fluorescence imaging."
        elif self.is_squid_plus:
            description = "A microscope control service for the Squid+ automated microscope with advanced hardware integration. This service provides real-time control over precision stage positioning, multi-channel illumination (brightfield and fluorescence), high-resolution camera operations, dual autofocus systems (reflection and contrast-based), and automated well plate navigation. The Squid+ system includes motorized filter wheels, objective switchers, and enhanced optics for advanced microscopy applications including automated scanning, image stitching, and multi-channel fluorescence imaging with professional-grade hardware control."
        else:
            description = "A microscope control service for the Squid automated microscope with real hardware integration. This service provides real-time control over precision stage positioning, multi-channel illumination (brightfield and fluorescence), high-resolution camera operations, dual autofocus systems (reflection and contrast-based), and automated well plate navigation. The system enables advanced microscopy workflows including automated scanning, image stitching, and multi-channel fluorescence imaging with professional-grade hardware control and real-time feedback."

        service_config = {
            "name": "Microscope Control Service",
            "id": service_id,
            "description": description,
            "config": {
                "visibility": visibility,
                "require_context": require_context,  # Always require context
                "run_in_executor": run_in_executor
            },
            "type": "service",
            "ping": self.ping,
            "is_service_healthy": self.is_service_healthy,
            "move_by_distance": self.move_by_distance,
            "snap": self.snap,
            "one_new_frame": self.one_new_frame,
            "get_video_frame": self.get_video_frame,
            "turn_off_illumination": self.turn_off_illumination,
            "turn_on_illumination": self.turn_on_illumination,
            "set_illumination": self.set_illumination,
            "set_camera_exposure": self.set_camera_exposure,
            "scan_plate_save_raw_images": self.scan_plate_save_raw_images,
            "home_stage": self.home_stage,
            "return_stage": self.return_stage,
            "navigate_to_well": self.navigate_to_well,
            "move_to_position": self.move_to_position,
            "move_to_loading_position": self.move_to_loading_position,
            "set_simulated_sample_data_alias": self.set_simulated_sample_data_alias,
            "get_simulated_sample_data_alias": self.get_simulated_sample_data_alias,
            "set_wellplate_offset": self.set_wellplate_offset,
            "contrast_autofocus": self.contrast_autofocus,
            "reflection_autofocus": self.reflection_autofocus,
            "autofocus_set_reflection_reference": self.autofocus_set_reflection_reference,
            "get_status": self.get_status,
            "update_parameters_from_client": self.update_parameters_from_client,
            "adjust_video_frame": self.adjust_video_frame,
            "start_video_buffering": self.start_video_buffering,
            "stop_video_buffering": self.stop_video_buffering,
            "get_video_buffering_status": self.get_video_buffering_status,
            "set_video_fps": self.set_video_fps,
            "get_current_well_location": self.get_current_well_location,
            "inspect_tool": self.inspect_tool,
            "get_microscope_configuration": self.get_microscope_configuration,
            "set_stage_velocity": self.set_stage_velocity,
            # Unified Scan API
            "scan_start": self.scan_start,
            "scan_get_status": self.scan_get_status,
            "scan_cancel": self.scan_cancel,
            # Stitching functions
            "scan_region_to_zarr": self.scan_region_to_zarr,
            "quick_scan_region_to_zarr": self.quick_scan_region_to_zarr,
            "get_stitched_region": self.get_stitched_region,
            # Experiment management functions (replaces zarr fileset management)
            "create_experiment": self.create_experiment,
            "list_experiments": self.list_experiments,
            "set_active_experiment": self.set_active_experiment,
            "remove_experiment": self.remove_experiment,
            "reset_experiment": self.reset_experiment,
            "get_experiment_info": self.get_experiment_info,
            #Artifact manager functions
            "upload_zarr_dataset": self.upload_zarr_dataset,
            # Offline processing functions
            "process_timelapse_offline": self.process_timelapse_offline,
            # Segmentation functions
            "segmentation_start": self.segmentation_start,
            "segmentation_get_status": self.segmentation_get_status,
            "segmentation_cancel": self.segmentation_cancel,
            "segmentation_get_polygons": self.segmentation_get_polygons,
        }

        # Conditionally register Squid+ specific endpoints
        if self.is_squid_plus:
            squid_plus_endpoints = {
                "set_filter_wheel_position": self.set_filter_wheel_position,
                "get_filter_wheel_position": self.get_filter_wheel_position,
                "next_filter_position": self.next_filter_position,
                "previous_filter_position": self.previous_filter_position,
                "switch_objective": self.switch_objective,
                "get_current_objective": self.get_current_objective,
                "get_available_objectives": self.get_available_objectives,
            }
            service_config.update(squid_plus_endpoints)
            logger.info(f"Registered {len(squid_plus_endpoints)} Squid+ specific endpoints")


        svc = await server.register_service(service_config)

        logger.info(
            f"Service (service_id={service_id}) started successfully, available at {self.server_url}{server.config.workspace}/services"
        )

        logger.info(f'You can use this service using the service id: {svc.id}')
        id = svc.id.split(":")[1]

        logger.info(f"You can also test the service via the HTTP proxy: {self.server_url}{server.config.workspace}/services/{id}")

    async def start_webrtc_service(self, server, webrtc_service_id_arg):
        self.webrtc_service_id = webrtc_service_id_arg

        async def on_init(peer_connection):
            logger.info("WebRTC peer connection initialized")
            # Mark as connected when peer connection starts
            self.webrtc_connected = True

            # Create data channel for metadata transmission
            self.metadata_data_channel = peer_connection.createDataChannel("metadata", ordered=True)
            logger.info("Created metadata data channel")

            @self.metadata_data_channel.on("open")
            def on_data_channel_open():
                logger.info("Metadata data channel opened")

            @self.metadata_data_channel.on("close")
            def on_data_channel_close():
                logger.info("Metadata data channel closed")

            @self.metadata_data_channel.on("error")
            def on_data_channel_error(error):
                logger.error(f"Metadata data channel error: {error}")

            @peer_connection.on("connectionstatechange")
            async def on_connectionstatechange():
                logger.info(f"WebRTC connection state changed to: {peer_connection.connectionState}")
                if peer_connection.connectionState in ["closed", "failed", "disconnected"]:
                    # Mark as disconnected
                    self.webrtc_connected = False
                    self.metadata_data_channel = None
                    if self.video_track and self.video_track.running:
                        logger.info(f"Connection state is {peer_connection.connectionState}. Stopping video track.")
                        self.video_track.stop()
                elif peer_connection.connectionState in ["connected"]:
                    # Mark as connected
                    self.webrtc_connected = True

            @peer_connection.on("track")
            def on_track(track):
                logger.info(f"Track {track.kind} received from client")

                if self.video_track and self.video_track.running:
                    self.video_track.stop()

                self.video_track = MicroscopeVideoTrack(self)
                peer_connection.addTrack(self.video_track)
                logger.info("Added MicroscopeVideoTrack to peer connection")
                self.is_streaming = True

                # Start video buffering when WebRTC starts
                asyncio.create_task(self.start_video_buffering())

                @track.on("ended")
                def on_ended():
                    logger.info(f"Client track {track.kind} ended")
                    if self.video_track:
                        logger.info("Stopping MicroscopeVideoTrack.")
                        self.video_track.stop()  # Now synchronous
                        self.video_track = None
                    self.is_streaming = False
                    self.metadata_data_channel = None

                    # Stop video buffering when WebRTC ends
                    asyncio.create_task(self.stop_video_buffering())

        ice_servers = await self.fetch_ice_servers()
        if not ice_servers:
            logger.warning("Using fallback ICE servers")
            ice_servers = [{"urls": ["stun:stun.l.google.com:19302"]}]

        try:
            await register_rtc_service(
                server,
                service_id=self.webrtc_service_id,
                config={
                    "visibility": "public",
                    "ice_servers": ice_servers,
                    "on_init": on_init,
                },
            )
            logger.info(f"WebRTC service registered with id: {self.webrtc_service_id}")
        except Exception as e:
            logger.error(f"Failed to register WebRTC service ({self.webrtc_service_id}): {e}")
            if "Service already exists" in str(e):
                logger.info(f"WebRTC service {self.webrtc_service_id} already exists. Attempting to retrieve it.")
                try:
                    _ = await server.get_service(self.webrtc_service_id)
                    logger.info(f"Successfully retrieved existing WebRTC service: {self.webrtc_service_id}")
                except Exception as get_e:
                    logger.error(f"Failed to retrieve existing WebRTC service {self.webrtc_service_id}: {get_e}")
                    raise
            else:
                raise

    async def setup(self):

        # Determine workspace and token based on simulation mode
        if self.is_simulation and not self.is_local:
            remote_token = os.environ.get("AGENT_LENS_WORKSPACE_TOKEN")
            remote_workspace = "agent-lens"
        else:
            remote_token = os.environ.get("SQUID_WORKSPACE_TOKEN")
            remote_workspace = "squid-control"

        remote_server = await connect_to_server(
                {"client_id": f"squid-remote-server-{self.service_id}-{uuid.uuid4()}", "server_url": "https://hypha.aicell.io", "token": remote_token, "workspace": remote_workspace, "ping_interval": 30}
            )
        self.remote_server = remote_server
        if not self.service_id:
            raise ValueError("MICROSCOPE_SERVICE_ID is not set in the environment variables.")
        if self.is_local:
            token = os.environ.get("REEF_LOCAL_TOKEN")
            workspace = os.environ.get("REEF_LOCAL_WORKSPACE")
            server = await connect_to_server(
                {"client_id": f"squid-local-server-{self.service_id}-{uuid.uuid4()}", "server_url": self.server_url, "token": token, "workspace": workspace, "ping_interval": 30}
            )
        else:
            # Determine workspace and token based on simulation mode
            if self.is_simulation:
                try:
                    token = os.environ.get("AGENT_LENS_WORKSPACE_TOKEN")
                except:
                    token = await login({"server_url": self.server_url})
                workspace = "agent-lens"
            else:
                try:
                    token = os.environ.get("SQUID_WORKSPACE_TOKEN")
                except:
                    token = await login({"server_url": self.server_url})
                workspace = "squid-control"

            server = await connect_to_server(
                {"client_id": f"squid-control-server-{self.service_id}-{uuid.uuid4()}", "server_url": self.server_url, "token": token, "workspace": workspace,  "ping_interval": 30}
            )

        self.server = server

        # Setup artifact manager for dataset and snapshot upload functionality
        try:
            from .hypha_tools.artifact_manager.artifact_manager import (
                SquidArtifactManager,
            )
            self.artifact_manager = SquidArtifactManager()

            # Connect to agent-lens workspace for artifact uploads (zarr datasets + snapshots)
            artifact_token = os.environ.get("AGENT_LENS_WORKSPACE_TOKEN")
            if artifact_token:
                artifact_server = await connect_to_server({
                    "server_url": "https://hypha.aicell.io",
                    "token": artifact_token,
                    "workspace": "agent-lens",
                    "ping_interval": 30
                })
                await self.artifact_manager.connect_server(artifact_server)
                logger.info("Artifact manager initialized successfully")

                # Pass the artifact manager to the squid controller for zarr uploads
                self.squidController.zarr_artifact_manager = self.artifact_manager
                logger.info("Artifact manager passed to squid controller")
            else:
                logger.warning("AGENT_LENS_WORKSPACE_TOKEN not found, artifact upload functionality disabled")
                self.artifact_manager = None
        except Exception as e:
            logger.warning(f"Failed to initialize artifact manager: {e}")
            self.artifact_manager = None

        # Initialize snapshot manager if artifact manager is available
        if self.artifact_manager:
            try:
                self.snapshot_manager = SnapshotManager(self.artifact_manager)
                logger.info("Snapshot manager initialized successfully")
            except Exception as e:
                logger.warning(f"Failed to initialize snapshot manager: {e}")
                self.snapshot_manager = None
        else:
            self.snapshot_manager = None
            logger.warning("Snapshot manager not initialized (artifact manager unavailable)")

        if self.is_simulation:
            await self.start_hypha_service(self.server, service_id=self.service_id)
        else:
            await self.start_hypha_service(self.server, service_id=self.service_id)

        webrtc_id = f"video-track-{self.service_id}"
        if not self.is_local: # only start webrtc service in remote mode
            await self.start_webrtc_service(self.server, webrtc_id)


    # For zarr visualization, use the vizarr package.

    async def _start_video_buffering_internal(self):
        """Start the background frame acquisition task for video buffering"""
        if self.frame_acquisition_running:
            logger.info("Video buffering already running")
            return

        self.frame_acquisition_running = True
        self.buffering_start_time = time.time()
        self.frame_acquisition_task = asyncio.create_task(self._background_frame_acquisition())
        logger.info("Video buffering started")

    async def _stop_video_buffering_internal(self):
        """Stop the background frame acquisition task"""
        if not self.frame_acquisition_running:
            logger.info("Video buffering not running")
            return

        self.frame_acquisition_running = False

        # Stop idle monitoring task
        if self.video_idle_check_task and not self.video_idle_check_task.done():
            self.video_idle_check_task.cancel()
            try:
                await self.video_idle_check_task
            except asyncio.CancelledError:
                pass
            self.video_idle_check_task = None

        # Stop frame acquisition task
        if self.frame_acquisition_task:
            try:
                await asyncio.wait_for(self.frame_acquisition_task, timeout=2.0)
            except asyncio.TimeoutError:
                logger.warning("Frame acquisition task did not stop gracefully, cancelling")
                self.frame_acquisition_task.cancel()
                try:
                    await self.frame_acquisition_task
                except asyncio.CancelledError:
                    pass

        self.video_buffer.clear()
        self.last_video_request_time = None
        self.buffering_start_time = None
        logger.info("Video buffering stopped")

    async def _background_frame_acquisition(self):
        """Background task that continuously acquires frames and stores them in buffer"""
        logger.info("Background frame acquisition started")
        consecutive_failures = 0

        while self.frame_acquisition_running:
            try:
                # Control frame acquisition rate with adaptive timing
                start_time = time.time()

                # Reduce frequency if camera is struggling
                if consecutive_failures > 3:
                    current_fps = max(1, self.buffer_fps / 2)  # Halve the FPS if struggling
                    logger.warning(f"Camera struggling, reducing acquisition rate to {current_fps} FPS")
                else:
                    current_fps = self.buffer_fps

                # Get current parameters
                channel = self.squidController.current_channel
                param_name = self.channel_param_map.get(channel)
                intensity, exposure_time = 10, 10  # Default values

                if param_name:
                    stored_params = getattr(self, param_name, None)
                    if stored_params and isinstance(stored_params, list) and len(stored_params) == 2:
                        intensity, exposure_time = stored_params

                # Acquire frame
                try:
                    # LATENCY MEASUREMENT: Start timing background frame acquisition
                    T_cam_start = time.time()

                    if self.is_simulation:
                        # Use existing simulation method for video buffering
                        raw_frame = await self.squidController.get_camera_frame_simulation(
                            channel, intensity, exposure_time
                        )
                    else:
                        # For real hardware, run in executor to avoid blocking
                        raw_frame = await asyncio.get_event_loop().run_in_executor(
                            None, self.squidController.get_camera_frame, channel, intensity, exposure_time
                        )

                    # LATENCY MEASUREMENT: End timing background frame acquisition
                    T_cam_read_complete = time.time()

                    # Calculate frame acquisition time and frame size (only if frame is valid)
                    if raw_frame is not None:
                        frame_acquisition_time_ms = (T_cam_read_complete - T_cam_start) * 1000
                        frame_size_bytes = raw_frame.nbytes
                        frame_size_kb = frame_size_bytes / 1024

                        # Log timing and size information for latency analysis (less frequent to avoid spam)
                        if consecutive_failures == 0:  # Only log on successful acquisitions
                            logger.info(f"LATENCY_MEASUREMENT: Background frame acquisition took {frame_acquisition_time_ms:.2f}ms, "
                                       f"frame size: {frame_size_kb:.2f}KB, exposure_time: {exposure_time}ms, "
                                       f"channel: {channel}, intensity: {intensity}")
                    else:
                        frame_acquisition_time_ms = (T_cam_read_complete - T_cam_start) * 1000
                        logger.info(f"LATENCY_MEASUREMENT: Background frame acquisition failed after {frame_acquisition_time_ms:.2f}ms, "
                                   f"exposure_time: {exposure_time}ms, channel: {channel}, intensity: {intensity}")

                    # Check if frame acquisition was successful
                    if raw_frame is None:
                        consecutive_failures += 1
                        logger.warning(f"Camera frame acquisition returned None - camera may be overloaded (failure #{consecutive_failures})")
                        # Create placeholder frame on None return
                        placeholder_frame = VideoFrameProcessor._create_placeholder_frame(
                            self.buffer_frame_width, self.buffer_frame_height, "Camera Overloaded"
                        )
                        compressed_placeholder = VideoFrameProcessor.encode_frame_jpeg(placeholder_frame, quality=85)

                        # Calculate gray level statistics for placeholder frame
                        placeholder_gray_stats = VideoFrameProcessor._calculate_gray_level_statistics(placeholder_frame)

                        # Create placeholder metadata
                        placeholder_metadata = {
                            'stage_position': {'x_mm': None, 'y_mm': None, 'z_mm': None},
                            'timestamp': time.time(),
                            'channel': channel,
                            'intensity': intensity,
                            'exposure_time_ms': exposure_time,
                            'gray_level_stats': placeholder_gray_stats,
                            'error': 'Camera Overloaded'
                        }
                        self.video_buffer.put_frame(compressed_placeholder, placeholder_metadata)

                        # If too many failures, wait longer before next attempt
                        if consecutive_failures >= 5:
                            await asyncio.sleep(2.0)  # Wait 2 seconds before retry
                            consecutive_failures = max(0, consecutive_failures - 2)  # Gradually recover

                    else:
                        # Process frame normally and reset failure counter
                        consecutive_failures = 0

                        # LATENCY MEASUREMENT: Start timing image processing
                        T_process_start = time.time()

                        processed_frame, gray_level_stats = VideoFrameProcessor.process_raw_frame(
                            raw_frame, frame_width=self.buffer_frame_width, frame_height=self.buffer_frame_height,
                            video_contrast_min=self.video_contrast_min, video_contrast_max=self.video_contrast_max
                        )

                        # LATENCY MEASUREMENT: End timing image processing
                        T_process_complete = time.time()

                        # LATENCY MEASUREMENT: Start timing JPEG compression
                        T_compress_start = time.time()

                        # Compress frame for efficient storage and transmission
                        compressed_frame = VideoFrameProcessor.encode_frame_jpeg(processed_frame, quality=85)

                        # LATENCY MEASUREMENT: End timing JPEG compression
                        T_compress_complete = time.time()

                        # METADATA CAPTURE: Get current stage position and create metadata
                        frame_timestamp = time.time()
                        try:
                            # Update position and get current coordinates
                            self.squidController.navigationController.update_pos(microcontroller=self.squidController.microcontroller)
                            current_x = self.squidController.navigationController.x_pos_mm
                            current_y = self.squidController.navigationController.y_pos_mm
                            current_z = self.squidController.navigationController.z_pos_mm
                            print(f"current_x: {current_x}, current_y: {current_y}, current_z: {current_z}")
                            frame_metadata = {
                                'stage_position': {
                                    'x_mm': current_x,
                                    'y_mm': current_y,
                                    'z_mm': current_z
                                },
                                'timestamp': frame_timestamp,
                                'channel': channel,
                                'intensity': intensity,
                                'exposure_time_ms': exposure_time,
                                'gray_level_stats': gray_level_stats
                            }
                        except Exception as e:
                            logger.warning(f"Failed to capture stage position for metadata: {e}")
                            # Fallback metadata without stage position
                            frame_metadata = {
                                'stage_position': {
                                    'x_mm': None,
                                    'y_mm': None,
                                    'z_mm': None
                                },
                                'timestamp': frame_timestamp,
                                'channel': channel,
                                'intensity': intensity,
                                'exposure_time_ms': exposure_time,
                                'gray_level_stats': gray_level_stats
                            }

                        # Calculate timing statistics
                        processing_time_ms = (T_process_complete - T_process_start) * 1000
                        compression_time_ms = (T_compress_complete - T_compress_start) * 1000
                        total_time_ms = (T_compress_complete - T_cam_start) * 1000

                        # Log comprehensive performance statistics
                        logger.info(f"LATENCY_PROCESSING: Background frame processing took {processing_time_ms:.2f}ms, "
                                   f"compression took {compression_time_ms:.2f}ms, "
                                   f"total_time={total_time_ms:.2f}ms, "
                                   f"compression_ratio={compressed_frame['compression_ratio']:.1f}x, "
                                   f"size: {compressed_frame['original_size']//1024}KB -> {compressed_frame['size_bytes']//1024}KB")

                        # Store compressed frame with metadata in buffer
                        self.video_buffer.put_frame(compressed_frame, frame_metadata)

                except Exception as e:
                    consecutive_failures += 1
                    logger.error(f"Error in background frame acquisition: {e}")
                    # Create placeholder frame on error
                    placeholder_frame = VideoFrameProcessor._create_placeholder_frame(
                        self.buffer_frame_width, self.buffer_frame_height, f"Acquisition Error: {str(e)}"
                    )
                    compressed_placeholder = VideoFrameProcessor.encode_frame_jpeg(placeholder_frame, quality=85)

                    # Calculate gray level statistics for placeholder frame
                    placeholder_gray_stats = VideoFrameProcessor._calculate_gray_level_statistics(placeholder_frame)

                    # Create placeholder metadata for error case
                    error_metadata = {
                        'stage_position': {'x_mm': None, 'y_mm': None, 'z_mm': None},
                        'timestamp': time.time(),
                        'channel': channel if 'channel' in locals() else 0,
                        'intensity': intensity if 'intensity' in locals() else 0,
                        'exposure_time_ms': exposure_time if 'exposure_time' in locals() else 0,
                        'gray_level_stats': placeholder_gray_stats,
                        'error': f"Acquisition Error: {str(e)}"
                    }
                    self.video_buffer.put_frame(compressed_placeholder, error_metadata)

                # Control frame rate with adaptive timing
                elapsed = time.time() - start_time
                sleep_time = max(0.1, (1.0 / current_fps) - elapsed)  # Minimum 100ms between attempts
                if sleep_time > 0:
                    await asyncio.sleep(sleep_time)

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Unexpected error in background frame acquisition: {e}")
                await asyncio.sleep(1.0)  # Wait 1 second on unexpected error

        logger.info("Background frame acquisition stopped")


    def decode_video_frame(self, frame_data):
        """
        Decode compressed video frame data back to numpy array.
        This is a public method for testing and external use.
        
        Args:
            frame_data: dict with compressed frame info from get_video_frame()
        
        Returns:
            numpy array: RGB image data
        """
        return VideoFrameProcessor.decode_frame_jpeg(frame_data)

    async def _monitor_video_idle(self):
        """Monitor video request activity and stop buffering after idle timeout"""
        while self.frame_acquisition_running:
            try:
                await asyncio.sleep(1.0)  # Check every 1 second instead of 500ms

                # Don't stop video buffering during scanning
                if self.scanning_in_progress:
                    continue

                if self.last_video_request_time is None:
                    continue

                # Check if we've been buffering for minimum duration
                if self.buffering_start_time is not None:
                    buffering_duration = time.time() - self.buffering_start_time
                    if buffering_duration < self.min_buffering_duration:
                        continue  # Don't stop yet, maintain minimum buffering time

                # Check if video has been idle too long
                idle_time = time.time() - self.last_video_request_time
                if idle_time > self.video_idle_timeout:
                    logger.info(f"Video idle for {idle_time:.1f}s (timeout: {self.video_idle_timeout}s), stopping buffering")
                    await self.stop_video_buffering()
                    break


            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in video idle monitoring: {e}")
                await asyncio.sleep(2.0)  # Longer sleep on error

        logger.info("Video idle monitoring stopped")

    @schema_function(skip_self=True)
    def get_current_well_location(self, well_plate_type: str=Field('96', description="Well plate format: '6', '12', '24', '96', or '384'"), context=None):
        """
        Determine which well the stage is currently positioned over based on coordinates.
        Returns: Dictionary with well row, column, well_id (e.g., 'A1'), and position_status ('in_well', 'between_wells', or 'outside_plate').
        """
        try:
            # Check authentication
            if context and not self.check_permission(context.get("user", {})):
                raise Exception("User not authorized to access this service")

            well_info = self.squidController.get_well_from_position(well_plate_type)
            logger.info(f'Current well location: {well_info["well_id"]} ({well_info["position_status"]})')
            return well_info
        except Exception as e:
            logger.error(f"Failed to get current well location: {e}")
            raise e

    @schema_function(skip_self=True)
    def configure_video_buffer_frame_size(self, frame_width: int = Field(750, description="Width of the video buffer frames"), frame_height: int = Field(750, description="Height of the video buffer frames"), context=None):
        """Configure video buffer frame size for optimal streaming performance."""
        try:
            # Check authentication
            if context and not self.check_permission(context.get("user", {})):
                raise Exception("User not authorized to access this service")

            # Validate frame size parameters
            frame_width = max(64, min(4096, frame_width))  # Clamp between 64-4096 pixels
            frame_height = max(64, min(4096, frame_height))  # Clamp between 64-4096 pixels

            old_width = self.buffer_frame_width
            old_height = self.buffer_frame_height

            # Update buffer frame size
            self.buffer_frame_width = frame_width
            self.buffer_frame_height = frame_height

            # If buffer is running and size changed, restart it to use new size
            restart_needed = (frame_width != old_width or frame_height != old_height) and self.frame_acquisition_running

            if restart_needed:
                logger.info(f"Buffer frame size changed from {old_width}x{old_height} to {frame_width}x{frame_height}, restarting buffer")
                # Clear existing buffer to remove old-sized frames
                self.video_buffer.clear()
                # Note: The frame acquisition loop will automatically use the new size for subsequent frames

            # Update WebRTC video track if it exists
            if hasattr(self, 'video_track') and self.video_track:
                self.video_track.frame_width = frame_width
                self.video_track.frame_height = frame_height
                logger.info(f"Updated WebRTC video track frame size to {frame_width}x{frame_height}")

            logger.info(f"Video buffer frame size configured: {frame_width}x{frame_height} (was {old_width}x{old_height})")

            return {
                "success": True,
                "message": f"Video buffer frame size configured to {frame_width}x{frame_height}",
                "previous_size": {"width": old_width, "height": old_height},
                "new_size": {"width": frame_width, "height": frame_height},
                "buffer_restarted": restart_needed
            }
        except Exception as e:
            logger.error(f"Failed to configure video buffer frame size: {e}")
            raise e

    @schema_function(skip_self=True)
    def get_microscope_configuration(self, config_section: str = Field("all", description="Configuration section: 'all', 'camera', 'stage', 'illumination', 'acquisition', 'limits', 'hardware', 'wellplate', 'optics', 'autofocus', or 'microscope_type'"), include_defaults: bool = Field(True, description="Include default values from config.py (True) or only user-configured values (False)"), context=None):
        """
        Retrieve microscope hardware and software configuration parameters.
        Returns: Dictionary with requested configuration section data including microscope type ('Squid' or 'Squid+'), hardware capabilities, and operational parameters.
        """
        try:
            # Check authentication
            if context and not self.check_permission(context.get("user", {})):
                raise Exception("User not authorized to access this service")

            try:
                from .control.config import get_microscope_configuration_data
            except ImportError:
                from .control.config import get_microscope_configuration_data

            # Call the configuration function from config.py
            result = get_microscope_configuration_data(
                config_section=config_section,
                include_defaults=include_defaults,
                is_simulation=self.is_simulation,
                is_local=self.is_local,
                squid_controller=self.squidController
            )

            logger.info(f"Retrieved microscope configuration for section: {config_section}")

            return result

        except Exception as e:
            logger.error(f"Failed to get microscope configuration: {e}")
            raise e

    @schema_function(skip_self=True)
    def set_stage_velocity(self, velocity_x_mm_per_s: Optional[float] = Field(None, description="Maximum X axis velocity in mm/s (None uses config default)"), velocity_y_mm_per_s: Optional[float] = Field(None, description="Maximum Y axis velocity in mm/s (None uses config default)"), context=None):
        """
        Configure maximum stage movement velocities for X and Y axes.
        Returns: Dictionary with success status and current velocity settings for both axes in mm/s.
        Notes: Lower velocities increase precision but slow movement. Higher velocities enable faster navigation. Changes apply immediately.
        """
        try:
            # Check authentication
            if context and not self.check_permission(context.get("user", {})):
                raise Exception("User not authorized to access this service")

            return self.squidController.set_stage_velocity(
                velocity_x_mm_per_s=velocity_x_mm_per_s,
                velocity_y_mm_per_s=velocity_y_mm_per_s
            )
        except Exception as e:
            logger.error(f"Error setting stage velocity: {e}")
            raise e


    @schema_function(skip_self=True)
    async def upload_zarr_dataset(self,
                                experiment_name: str = Field(..., description="Name of existing experiment to upload (becomes dataset name with timestamp)"),
                                description: str = Field("", description="Optional human-readable description for the dataset"),
                                include_acquisition_settings: bool = Field(True, description="Include microscope settings (channels, pixel size) as dataset metadata (True recommended)"),
                                context=None):
        """
        Upload the experiment's zarr canvas folder directly to the artifact manager gallery using hypha-artifact.
        Returns: Dictionary with success status, experiment name, dataset name (with timestamp), total size (MB), and acquisition settings.
        Notes: Dataset named '{experiment_name}-{date-time}'. Requires AGENT_LENS_WORKSPACE_TOKEN environment variable.
        """
        try:
            # Check authentication
            if context and not self.check_permission(context.get("user", {})):
                raise Exception("User not authorized to access this service")

            # Check if experiment manager is initialized
            if not hasattr(self.squidController, 'experiment_manager') or self.squidController.experiment_manager is None:
                raise Exception("Experiment manager not initialized. Start a scanning operation first to create data.")

            # Get experiment information
            experiment_info = self.squidController.experiment_manager.get_experiment_info(experiment_name)
            
            # Check if zarr data exists
            zarr_path = Path(experiment_info.get("zarr_path", ""))
            if not zarr_path.exists():
                raise Exception(f"No zarr data found for experiment '{experiment_name}'. Start a scanning operation first to create data.")

            total_size_mb = experiment_info.get("total_size_mb", 0)
            logger.info(f"Uploading experiment '{experiment_name}' ({total_size_mb:.2f} MB)")

            # Prepare acquisition settings if requested
            acquisition_settings = None
            if include_acquisition_settings:
                try:
                    from .control.config import CONFIG, ChannelMapper
                    
                    # Get canvas to extract metadata
                    canvas = self.squidController.experiment_manager.get_canvas(experiment_name)
                    export_info = canvas.get_export_info() if canvas else {}
                    
                    acquisition_settings = {
                        "pixel_size_xy_um": export_info.get("canvas_dimensions", {}).get("pixel_size_um", self.squidController.pixel_size_xy),
                        "channels": export_info.get("channels", ChannelMapper.get_all_human_names()),
                        "canvas_dimensions": export_info.get("canvas_dimensions", {}),
                        "num_scales": export_info.get("num_scales", 6),
                        "microscope_service_id": self.service_id,
                        "experiment_name": experiment_name,
                        "stage_limits": self.squidController.experiment_manager.stage_limits
                    }
                except Exception as e:
                    logger.warning(f"Could not get detailed acquisition settings: {e}")
                    acquisition_settings = {
                        "microscope_service_id": self.service_id,
                        "experiment_name": experiment_name,
                        "total_size_mb": total_size_mb
                    }

            # Generate dataset name with timestamp
            dataset_name = f"{experiment_name}-{time.strftime('%Y%m%d-%H%M%S')}"
            
            # Use hypha-artifact for direct folder upload
            from hypha_artifact import AsyncHyphaArtifact
            
            token = os.environ.get("AGENT_LENS_WORKSPACE_TOKEN")
            if not token:
                raise Exception("AGENT_LENS_WORKSPACE_TOKEN environment variable not set")
            
            logger.info(f"Creating artifact '{dataset_name}' for upload")
            
            artifact = AsyncHyphaArtifact(
                artifact_id=dataset_name,
                workspace="agent-lens",
                token=token,
                server_url=self.server_url
            )
            
            # Edit mode for staging changes
            await artifact.edit(stage=True)
            
            # Upload the zarr folder recursively
            logger.info(f"Uploading zarr folder: {zarr_path}")
            await artifact.put(str(zarr_path), "/data.zarr", recursive=True)
            
            # Add manifest with acquisition settings if available
            if acquisition_settings:
                import json
                manifest_content = json.dumps({
                    "name": experiment_name,
                    "description": description or f"Experiment {experiment_name}",
                    "acquisition_settings": acquisition_settings
                }, indent=2)
                await artifact.put(manifest_content.encode(), "/manifest.json")
            
            # Commit the changes
            await artifact.commit(comment=description or f"Experiment {experiment_name}")
            
            logger.info(f"Successfully uploaded experiment '{experiment_name}' as '{dataset_name}'")

            return {
                "success": True,
                "experiment_name": experiment_name,
                "dataset_name": dataset_name,
                "total_size_mb": total_size_mb,
                "acquisition_settings": acquisition_settings,
                "description": description or f"Experiment {experiment_name}"
            }

        except Exception as e:
            logger.error(f"Error uploading experiment dataset: {e}")
            raise e

    async def scan_region_to_zarr(self, start_x_mm: float = 20, start_y_mm: float = 20, Nx: int = 5, Ny: int = 5, dx_mm: float = 0.9, dy_mm: float = 0.9, illumination_settings: Optional[List[dict]] = None, do_contrast_autofocus: bool = False, do_reflection_af: bool = False, action_ID: str = 'normal_scan_stitching', timepoint: int = 0, experiment_name: Optional[str] = None, wells_to_scan: List[str] = None, well_plate_type: str = '96', well_padding_mm: float = 1.0, uploading: bool = False, reset_application: bool = True, context=None):
        """
        DEPRECATED: Use scan_start() with saved_data_type='full_zarr' instead.
        
        Perform region scan with live stitching to OME-Zarr. Automatically triggers
        segmentation and similarity search (cell extraction, embeddings, Weaviate upload) after scan.
        
        Args:
            start_x_mm: Starting X position in millimeters
            start_y_mm: Starting Y position in millimeters
            Nx: Number of positions to scan in X direction
            Ny: Number of positions to scan in Y direction
            dx_mm: Distance between positions in X direction (millimeters)
            dy_mm: Distance between positions in Y direction (millimeters)
            illumination_settings: List of dictionaries with channel settings (optional)
            do_contrast_autofocus: Enable contrast-based autofocus
            do_reflection_af: Enable reflection-based autofocus
            action_ID: Unique identifier for this scan
            timepoint: Timepoint index for this scan (default 0)
            experiment_name: Name of the experiment to use. If None, uses active experiment or 'default' as fallback
            wells_to_scan: List of wells to scan (e.g., ['A1', 'B2', 'C3'])
            well_plate_type: Well plate type ('6', '12', '24', '96', '384')
            well_padding_mm: Padding around well in mm
            uploading: Enable upload after scanning is complete
            reset_application: Reset Weaviate application before upload (default: True)
            
        Returns:
            dict: Status of the scan, including segmentation_result with segmentation status
        """
        logger.warning("DEPRECATED: scan_region_to_zarr is deprecated and will be removed in a future release. Use scan_start() with saved_data_type='full_zarr' instead.")
        try:
            # Check authentication
            if context and not self.check_permission(context.get("user", {})):
                raise Exception("User not authorized to access this service")

            # Set default illumination settings if not provided
            if illumination_settings is None:
                illumination_settings = [{'channel': 'BF LED matrix full', 'intensity': 50, 'exposure_time': 100}]

            # Set default wells to scan if not provided
            if wells_to_scan is None:
                wells_to_scan = ['A1']

            logger.info(f"Starting region scan to Zarr: {Nx}x{Ny} positions from ({start_x_mm}, {start_y_mm})")

            # Check if video buffering is active and stop it during scanning
            video_buffering_was_active = self.frame_acquisition_running
            if video_buffering_was_active:
                logger.info("Video buffering is active, stopping it temporarily during scanning")
                await self.stop_video_buffering()
                # Wait additional time to ensure camera fully settles after stopping video buffering
                logger.info("Waiting for camera to settle after stopping video buffering...")
                await asyncio.sleep(0.5)

            # Set scanning flag to prevent automatic video buffering restart during scan
            self.scanning_in_progress = True

            # Perform the region scan to Zarr
            await self.squidController.scan_region_to_zarr(
                start_x_mm=start_x_mm,
                start_y_mm=start_y_mm,
                Nx=Nx,
                Ny=Ny,
                dx_mm=dx_mm,
                dy_mm=dy_mm,
                illumination_settings=illumination_settings,
                do_contrast_autofocus=do_contrast_autofocus,
                do_reflection_af=do_reflection_af,
                action_ID=action_ID,
                timepoint=timepoint,
                experiment_name=experiment_name,
                wells_to_scan=wells_to_scan,
                well_plate_type=well_plate_type,
                well_padding_mm=well_padding_mm
            )

            actual_experiment_name = experiment_name or self.squidController.experiment_manager.current_experiment_name
            
            # Auto-trigger segmentation after scan
            segmentation_result = None
            try:
                channel_configs = [{
                    'channel': illum.get('channel', 'BF LED matrix full'),
                    'min_percentile': 1.0,
                    'max_percentile': 99.0
                } for illum in (illumination_settings or [{'channel': 'BF LED matrix full'}])]
                
                # Convert wells_to_scan from tuples to strings for segmentation
                # e.g., [('B', 2)] -> ['B2']
                wells_to_segment_str = [f"{row}{col}" for row, col in wells_to_scan] if wells_to_scan else []
                
                segmentation_task = asyncio.create_task(
                    self._run_segmentation_background(
                        source_experiment=actual_experiment_name,
                        wells_to_segment=wells_to_segment_str,
                        channel_configs=channel_configs,
                        scale_level=1,
                        timepoint=timepoint,
                        well_plate_type=well_plate_type,
                        well_padding_mm=well_padding_mm,
                        reset_application=reset_application,
                        context=context
                    )
                )
                self.segmentation_state['segmentation_task'] = segmentation_task
                self.segmentation_state['state'] = 'running'
                segmentation_result = {"success": True, "experiment_name": actual_experiment_name}
            except Exception as e:
                logger.error(f"Auto-segmentation failed: {e}", exc_info=True)
                segmentation_result = {"success": False, "error": str(e)}
            
            # Upload the experiment if uploading is enabled
            upload_result = None
            if uploading:
                try:
                    logger.info("Uploading experiment after normal scan completion")
                    upload_result = await self.upload_zarr_dataset(
                        experiment_name=actual_experiment_name,
                        description=f"Normal scan with stitching - {action_ID}",
                        include_acquisition_settings=True
                    )
                    logger.info("Successfully uploaded experiment after normal scan")
                except Exception as e:
                    logger.error(f"Failed to upload experiment after normal scan: {e}")
                    # Don't raise the exception - continue with response

            return {
                "success": True,
                "message": "Normal scan with stitching completed successfully",
                "scan_parameters": {
                    "start_position": {"x_mm": start_x_mm, "y_mm": start_y_mm},
                    "grid_size": {"nx": Nx, "ny": Ny},
                    "step_size": {"dx_mm": dx_mm, "dy_mm": dy_mm},
                    "total_area_mm2": (Nx * dx_mm) * (Ny * dy_mm),
                    "experiment_name": actual_experiment_name,
                    "wells_scanned": wells_to_scan
                },
                "segmentation_result": segmentation_result,
                "upload_result": upload_result
            }
        except Exception as e:
            logger.error(f"Failed to perform normal scan with stitching: {e}")
            raise e
        finally:
            # Always reset the scanning flag, regardless of success or failure
            self.scanning_in_progress = False
            logger.info("Normal scanning completed, video buffering auto-start is now re-enabled")


    @schema_function(skip_self=True)
    def reset_stitching_canvas(self, context=None):
        """
        Reset the stitching canvas, clearing all stored images.
        
        This will delete the existing zarr canvas and prepare for a new scan.
        
        Returns:
            dict: Status of the reset operation
        """
        try:
            if hasattr(self.squidController, 'zarr_canvas') and self.squidController.zarr_canvas is not None:
                # Close the existing canvas
                self.squidController.zarr_canvas.close()

                # Delete the zarr directory
                import shutil
                if self.squidController.zarr_canvas.zarr_path.exists():
                    shutil.rmtree(self.squidController.zarr_canvas.zarr_path)

                # Clear the reference
                self.squidController.zarr_canvas = None

                logger.info("Stitching canvas reset successfully")
                return {
                    "success": True,
                    "message": "Stitching canvas has been reset"
                }
            else:
                return {
                    "success": True,
                    "message": "No stitching canvas to reset"
                }
        except Exception as e:
            logger.error(f"Failed to reset stitching canvas: {e}")
            raise e


    async def quick_scan_region_to_zarr(self, start_x_mm: float, start_y_mm: float,
                                        scan_width_mm: float, scan_height_mm: float,
                                        dy_mm: float = 0.85, exposure_time: float = 5,
                                        intensity: float = 70, fps_target: int = 10,
                                        velocity_scan_mm_per_s: float = 7.0,
                                        experiment_name: Optional[str] = None,
                                        uploading: bool = False,
                                        context=None):
        """
        Quick brightfield scan of a rectangular region without well plates.
        Simplified version that scans a continuous rectangular area in horizontal stripes.
        No autofocus, no well plate logic - just a simple fast scan.
        
        Args:
            start_x_mm: Starting X position in mm (left edge of scan area)
            start_y_mm: Starting Y position in mm (bottom edge of scan area)
            scan_width_mm: Width of scan area in mm (X direction)
            scan_height_mm: Height of scan area in mm (Y direction)
            dy_mm: Y increment between horizontal stripes in mm (default 0.85)
            exposure_time: Camera exposure time in ms (max 30ms)
            intensity: Brightfield LED intensity (0-100)
            fps_target: Target frame rate for acquisition (default 10fps)
            velocity_scan_mm_per_s: Stage velocity during scanning in mm/s (default 7.0)
            experiment_name: Name of the experiment. If None, uses active experiment
            uploading: Enable upload after scanning is complete
            
        Returns:
            dict: Status of the scan with performance metrics
        """
        try:
            # Check authentication
            if context and not self.check_permission(context.get("user", {})):
                raise Exception("User not authorized to access this service")
            
            # Validate exposure time early
            if exposure_time > 30:
                raise ValueError(f"Quick scan exposure time must not exceed 30ms (got {exposure_time}ms)")
            
            logger.info(f"Starting quick region scan: start=({start_x_mm}, {start_y_mm}), "
                       f"size={scan_width_mm}x{scan_height_mm}mm, dy={dy_mm}mm, "
                       f"velocity={velocity_scan_mm_per_s}mm/s, fps={fps_target}")
            
            # Check if video buffering is active and stop it during scanning
            video_buffering_was_active = self.frame_acquisition_running
            if video_buffering_was_active:
                logger.info("Video buffering is active, stopping it temporarily during quick region scanning")
                await self.stop_video_buffering()
                # Wait for camera to settle after stopping video buffering
                logger.info("Waiting for camera to settle after stopping video buffering...")
                await asyncio.sleep(0.5)
            
            # Set scanning flag to prevent automatic video buffering restart during scan
            self.scanning_in_progress = True
            
            # Record start time for performance metrics
            start_time = time.time()
            
            # Perform the quick region scan to Zarr
            result = await self.squidController.quick_scan_region_to_zarr(
                start_x_mm=start_x_mm,
                start_y_mm=start_y_mm,
                scan_width_mm=scan_width_mm,
                scan_height_mm=scan_height_mm,
                dy_mm=dy_mm,
                exposure_time=exposure_time,
                intensity=intensity,
                fps_target=fps_target,
                velocity_scan_mm_per_s=velocity_scan_mm_per_s,
                experiment_name=experiment_name
            )
            
            # Calculate performance metrics
            scan_duration = time.time() - start_time
            n_stripes = result.get('stripes_scanned', 0)
            total_frames = result.get('total_frames', 0)
            
            actual_experiment_name = experiment_name or self.squidController.experiment_manager.current_experiment_name
            
            # Upload the experiment if uploading is enabled
            upload_result = None
            if uploading:
                try:
                    logger.info("Uploading experiment after quick region scan completion")
                    upload_result = await self.upload_zarr_dataset(
                        experiment_name=actual_experiment_name,
                        description=f"Quick region scan - {scan_width_mm}x{scan_height_mm}mm",
                        include_acquisition_settings=True
                    )
                    logger.info("Successfully uploaded experiment after quick region scan")
                except Exception as e:
                    logger.error(f"Failed to upload experiment after quick region scan: {e}")
                    # Don't raise the exception - continue with response
            
            return {
                "success": True,
                "message": "Quick region scan completed successfully",
                "scan_parameters": {
                    "start_position": {"x_mm": start_x_mm, "y_mm": start_y_mm},
                    "scan_size": {"width_mm": scan_width_mm, "height_mm": scan_height_mm},
                    "scan_area_mm2": scan_width_mm * scan_height_mm,
                    "stripes_scanned": n_stripes,
                    "dy_mm": dy_mm,
                    "exposure_time_ms": exposure_time,
                    "intensity": intensity,
                    "scan_velocity_mm_per_s": velocity_scan_mm_per_s,
                    "target_fps": fps_target,
                    "experiment_name": actual_experiment_name
                },
                "performance_metrics": {
                    "total_scan_time_seconds": round(scan_duration, 2),
                    "scan_time_per_stripe_seconds": round(scan_duration / n_stripes, 2) if n_stripes > 0 else 0,
                    "total_frames_acquired": total_frames,
                    "average_fps": round(total_frames / scan_duration, 2) if scan_duration > 0 else 0
                },
                "stitching_info": {
                    "zarr_scales_updated": "1-5 (scale 0 skipped for performance)",
                    "channel": "BF LED matrix full",
                    "experiment_name": actual_experiment_name
                },
                "upload_result": upload_result
            }
            
        except ValueError as e:
            logger.error(f"Validation error in quick region scan: {e}")
            raise e
        except Exception as e:
            logger.error(f"Failed to perform quick region scan: {e}")
            raise e
        finally:
            # Always reset the scanning flag, regardless of success or failure
            self.scanning_in_progress = False
            logger.info("Quick region scanning completed, video buffering auto-start is now re-enabled")

    @schema_function(skip_self=True)
    def get_stitched_region(self, center_x_mm: float = Field(..., description="Region center X in stage coordinates (mm)"),
                           center_y_mm: float = Field(..., description="Region center Y in stage coordinates (mm)"),
                           width_mm: float = Field(5.0, description="Region width in millimeters"),
                           height_mm: float = Field(5.0, description="Region height in millimeters"),
                           well_plate_type: str = Field('96', description="Well plate format: '6', '12', '24', '96', or '384'"),
                           scale_level: int = Field(0, description="Pyramid scale level: 0=full, 1=1/4x, 2=1/16x, 3=1/64x, 4=1/256x, 5=1/1024x resolution"),
                           channel_name: str = Field('BF LED matrix full', description="Channel name(s): single or comma-separated (e.g., 'BF LED matrix full' or 'BF LED matrix full,Fluorescence 488 nm Ex')"),
                           timepoint: int = Field(0, description="Time-lapse timepoint index (0 for single timepoint)"),
                           well_padding_mm: float = Field(1.0, description="Well boundary padding in millimeters"),
                           output_format: str = Field('base64', description="Output format: 'base64' (PNG image) or 'array' (numpy list)"),
                           experiment_name: Optional[str] = Field(None, description="Experiment name (None uses active experiment)"),
                           context=None):
        """
        Extract and merge stitched image data from one or more wells at specified coordinates.
        Returns: Dictionary with success status, base64 PNG or array data, shape, dtype, is_rgb flag, channels_used list, and region metadata.
        Notes: Automatically spans multiple wells if region crosses boundaries. Multiple channels merge into RGB with channel-specific colors (BF=white, 405nm=blue, 488nm=green, 561nm=yellow, 638nm=red, 730nm=magenta).
        """
        try:
            # Check authentication
            if context and not self.check_permission(context.get("user", {})):
                logger.warning("User not authorized to access this service")
                raise Exception("User not authorized to access this service")

            # Parse channel_name string into a list
            if isinstance(channel_name, str):
                # Split by comma and strip whitespace, filter out empty strings
                channel_list = [ch.strip() for ch in channel_name.split(',') if ch.strip()]
            else:
                # If it's already a list, use it as is
                channel_list = list(channel_name)

            # Validate channel names
            if not channel_list:
                logger.warning("No valid channel names found - returning error")
                return {
                    "success": False,
                    "message": "At least one channel name must be specified",
                    "region": {
                        "center_x_mm": center_x_mm,
                        "center_y_mm": center_y_mm,
                        "width_mm": width_mm,
                        "height_mm": height_mm,
                        "well_plate_type": well_plate_type,
                        "scale_level": scale_level,
                        "channels": channel_list,
                        "timepoint": timepoint,
                        "well_padding_mm": well_padding_mm
                    }
                }

            # Get regions for each channel
            channel_regions = []
            for i, ch_name in enumerate(channel_list):
                logger.debug(f"Processing channel {i+1}/{len(channel_list)}: '{ch_name}'")
                region = self.squidController.get_stitched_region(
                    center_x_mm=center_x_mm,
                    center_y_mm=center_y_mm,
                    width_mm=width_mm,
                    height_mm=height_mm,
                    well_plate_type=well_plate_type,
                    scale_level=scale_level,
                    channel_name=ch_name,
                    timepoint=timepoint,
                    well_padding_mm=well_padding_mm,
                    experiment_name=experiment_name
                )

                if region is None:
                    logger.warning(f"No data available for channel '{ch_name}' at ({center_x_mm:.2f}, {center_y_mm:.2f})")
                    continue

                logger.debug(f"Retrieved region for channel '{ch_name}': shape={region.shape if hasattr(region, 'shape') else 'unknown'}")
                channel_regions.append((ch_name, region))

            if not channel_regions:
                logger.warning(f"No data available for any channels at ({center_x_mm:.2f}, {center_y_mm:.2f}) with size ({width_mm:.2f}x{height_mm:.2f})")
                return {
                    "success": False,
                    "message": f"No data available for any channels at ({center_x_mm:.2f}, {center_y_mm:.2f}) with size ({width_mm:.2f}x{height_mm:.2f})",
                    "region": {
                        "center_x_mm": center_x_mm,
                        "center_y_mm": center_y_mm,
                        "width_mm": width_mm,
                        "height_mm": height_mm,
                        "well_plate_type": well_plate_type,
                        "scale_level": scale_level,
                        "channels": channel_list,
                        "timepoint": timepoint,
                        "well_padding_mm": well_padding_mm
                    }
                }

            # Merge channels if multiple channels are specified
            if len(channel_regions) == 1:
                # Single channel - return as grayscale
                merged_region = channel_regions[0][1]
                is_rgb = False
            else:
                # Multiple channels - merge into RGB
                logger.debug(f"Merging {len(channel_regions)} channels into RGB")
                merged_region = self._merge_channels_to_rgb(channel_regions)
                is_rgb = True

            if merged_region is None:
                logger.error("Failed to merge channels - merged_region is None")
                return {
                    "success": False,
                    "message": "Failed to merge channels",
                    "region": {
                        "center_x_mm": center_x_mm,
                        "center_y_mm": center_y_mm,
                        "width_mm": width_mm,
                        "height_mm": height_mm,
                        "well_plate_type": well_plate_type,
                        "scale_level": scale_level,
                        "channels": channel_list,
                        "timepoint": timepoint,
                        "well_padding_mm": well_padding_mm
                    }
                }

            # Process output format
            if output_format == 'base64':
                # Convert to base64 encoded PNG
                import base64
                import io

                from PIL import Image

                if merged_region.dtype != np.uint8:
                    if is_rgb:
                        # RGB image - normalize each channel independently
                        normalized = np.zeros_like(merged_region, dtype=np.uint8)
                        for c in range(merged_region.shape[2]):
                            channel_data = merged_region[:, :, c]
                            if channel_data.max() > 0:
                                normalized[:, :, c] = (channel_data / channel_data.max() * 255).astype(np.uint8)
                        merged_region = normalized
                    else:
                        # Grayscale image
                        merged_region = (merged_region / merged_region.max() * 255).astype(np.uint8) if merged_region.max() > 0 else merged_region.astype(np.uint8)

                if is_rgb:
                    img = Image.fromarray(merged_region, 'RGB')
                else:
                    img = Image.fromarray(merged_region, 'L')

                buffer = io.BytesIO()
                img.save(buffer, format='PNG')
                img_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
                logger.debug(f"Base64 encoded PNG: {len(img_base64)} chars, shape={merged_region.shape}")
                return {
                    "success": True,
                    "data": img_base64,
                    "format": "png_base64",
                    "shape": merged_region.shape,
                    "dtype": str(merged_region.dtype),
                    "is_rgb": is_rgb,
                    "channels_used": [ch for ch, _ in channel_regions],
                    "region": {
                        "center_x_mm": center_x_mm,
                        "center_y_mm": center_y_mm,
                        "width_mm": width_mm,
                        "height_mm": height_mm,
                        "well_plate_type": well_plate_type,
                        "scale_level": scale_level,
                        "channels": channel_list,
                        "timepoint": timepoint,
                        "well_padding_mm": well_padding_mm
                    }
                }
            else:
                logger.debug("Returning array format result")
                return {
                    "success": True,
                    "data": merged_region.tolist(),
                    "format": "array",
                    "shape": merged_region.shape,
                    "dtype": str(merged_region.dtype),
                    "is_rgb": is_rgb,
                    "channels_used": [ch for ch, _ in channel_regions],
                    "region": {
                        "center_x_mm": center_x_mm,
                        "center_y_mm": center_y_mm,
                        "width_mm": width_mm,
                        "height_mm": height_mm,
                        "well_plate_type": well_plate_type,
                        "scale_level": scale_level,
                        "channels": channel_list,
                        "timepoint": timepoint,
                        "well_padding_mm": well_padding_mm
                    }
                }

        except Exception as e:
            logger.error(f"Failed to get stitched region: {e}", exc_info=True)
            raise e

    def _merge_channels_to_rgb(self, channel_regions):
        """
        Merge multiple channel regions into a single RGB image using the channel color scheme.
        
        Args:
            channel_regions: List of tuples (channel_name, region_data)
            
        Returns:
            np.ndarray: RGB image with shape (height, width, 3)
        """
        try:
            if not channel_regions:
                return None

            # Get the first region to determine dimensions
            first_region = channel_regions[0][1]
            height, width = first_region.shape

            # Create RGB output image
            rgb_image = np.zeros((height, width, 3), dtype=np.float32)

            # Channel color mapping based on initialize_canvas
            channel_colors = {
                'BF LED matrix full': [1.0, 1.0, 1.0],      # White
                'Fluorescence 405 nm Ex': [0.5, 0.0, 1.0],   # Blue-violet
                'Fluorescence 488 nm Ex': [0.0, 1.0, 0.0],   # Green
                'Fluorescence 638 nm Ex': [1.0, 0.0, 0.0],   # Red
                'Fluorescence 561 nm Ex': [1.0, 1.0, 0.0],   # Yellow
                'Fluorescence 730 nm Ex': [1.0, 0.0, 1.0],   # Magenta
            }

            # Process each channel
            for ch_name, region_data in channel_regions:
                # Normalize region data to 0-1 range
                if region_data.max() > 0:
                    normalized_region = region_data.astype(np.float32) / region_data.max()
                else:
                    normalized_region = region_data.astype(np.float32)

                # Get color for this channel
                if ch_name in channel_colors:
                    color = channel_colors[ch_name]
                else:
                    # Default to white for unknown channels
                    color = [1.0, 1.0, 1.0]

                # Add weighted contribution to RGB image
                for c in range(3):
                    rgb_image[:, :, c] += normalized_region * color[c]

            # Clip to 0-1 range and convert to uint8
            rgb_image = np.clip(rgb_image, 0, 1)
            rgb_image = (rgb_image * 255).astype(np.uint8)

            logger.debug(f"Merged {len(channel_regions)} channels into RGB image")
            return rgb_image

        except Exception as e:
            logger.error(f"Error merging channels to RGB: {e}")
            return None

    @schema_function(skip_self=True)
    async def create_experiment(self, experiment_name: str = Field(..., description="Unique name for the new experiment folder"), context=None):
        """
        Create a new experiment container for organizing multi-well scanning data.
        Returns: Dictionary with success status, experiment name, path, and creation timestamp.
        Notes: Experiment becomes active automatically. All subsequent scans are saved to this experiment until changed.
        """
        try:
            # Check authentication
            if context and not self.check_permission(context.get("user", {})):
                raise Exception("User not authorized to access this service")

            result = self.squidController.create_experiment(experiment_name)
            logger.info(f"Created experiment: {experiment_name}")
            return result
        except Exception as e:
            logger.error(f"Failed to create experiment: {e}")
            raise e

    @schema_function(skip_self=True)
    async def list_experiments(self, context=None):
        """
        Retrieve all experiments in the workspace with their metadata.
        Returns: Dictionary with list of experiments (name, path, size, well count), active experiment name, and total count.
        """
        try:
            # Check authentication
            if context and not self.check_permission(context.get("user", {})):
                raise Exception("User not authorized to access this service")

            result = self.squidController.list_experiments()
            logger.info(f"Listed experiments: {result['total_count']} found")
            return result
        except Exception as e:
            logger.error(f"Failed to list experiments: {e}")
            raise e

    @schema_function(skip_self=True)
    async def set_active_experiment(self, experiment_name: str = Field(..., description="Name of existing experiment to make active"), context=None):
        """
        Designate an existing experiment as the active target for all subsequent scan operations.
        Returns: Dictionary with success status, experiment name, and path.
        Notes: All new scan data will be saved to this experiment until a different one is activated.
        """
        try:
            # Check authentication
            if context and not self.check_permission(context.get("user", {})):
                raise Exception("User not authorized to access this service")

            result = self.squidController.set_active_experiment(experiment_name)
            logger.info(f"Set active experiment: {experiment_name}")
            return result
        except Exception as e:
            logger.error(f"Failed to set active experiment: {e}")
            raise e

    @schema_function(skip_self=True)
    async def remove_experiment(self, experiment_name: str = Field(..., description="Name of experiment to permanently delete"), context=None):
        """
        Permanently delete an experiment and all its associated data from disk.
        Returns: Dictionary with success status, deleted experiment name, and freed disk space.
        Notes: This operation is irreversible. All well canvases and metadata will be permanently removed.
        """
        try:
            # Check authentication
            if context and not self.check_permission(context.get("user", {})):
                raise Exception("User not authorized to access this service")

            result = self.squidController.remove_experiment(experiment_name)
            logger.info(f"Removed experiment: {experiment_name}")
            return result
        except Exception as e:
            logger.error(f"Failed to remove experiment: {e}")
            raise e

    @schema_function(skip_self=True)
    async def reset_experiment(self, experiment_name: str = Field(..., description="Name of experiment to clear all data from"), context=None):
        """
        Clear all scan data from an experiment while preserving the experiment container.
        Returns: Dictionary with success status, experiment name, and confirmation that data was cleared.
        Notes: Deletes all well canvases and resets to empty state. Experiment structure and metadata remain intact.
        """
        try:
            # Check authentication
            if context and not self.check_permission(context.get("user", {})):
                raise Exception("User not authorized to access this service")

            result = self.squidController.reset_experiment(experiment_name)
            logger.info(f"Reset experiment: {experiment_name}")
            return result
        except Exception as e:
            logger.error(f"Failed to reset experiment: {e}")
            raise e

    @schema_function(skip_self=True)
    async def get_experiment_info(self, experiment_name: str = Field(..., description="Name of experiment to query"), context=None):
        """
        Retrieve detailed metadata and statistics for a specific experiment.
        Returns: Dictionary with experiment name, path, total size, well canvas list (name, size, path), and well count.
        """
        try:
            # Check authentication
            if context and not self.check_permission(context.get("user", {})):
                raise Exception("User not authorized to access this service")

            result = self.squidController.get_experiment_info(experiment_name)
            logger.info(f"Retrieved experiment info: {experiment_name}")
            return result
        except Exception as e:
            logger.error(f"Failed to get experiment info: {e}")
            raise e

    @schema_function(skip_self=True)
    async def process_timelapse_offline(self,
        experiment_id: str = Field(..., description="Experiment ID prefix to search for matching folders (e.g., 'test-drug' matches 'test-drug-20250822T...')"),
        upload_immediately: bool = Field(True, description="Upload each run to gallery immediately after stitching completes (True recommended)"),
        cleanup_temp_files: bool = Field(True, description="Delete temporary zarr files after successful upload to save disk space (True recommended)"),
        use_parallel_wells: bool = Field(True, description="Process 3 wells concurrently for faster completion (True=parallel, False=sequential)"),
        context=None):
        """
        Process time-lapse experiment raw images offline by stitching into OME-Zarr and uploading to gallery.
        Returns: Dictionary with success status, total datasets processed, gallery information, processing mode (parallel/sequential), and per-run details.
        Notes: Runs in background thread to prevent network disconnections. Searches for all folders matching experiment_id prefix. Each run uploads as separate dataset to 'experiment-{experiment_id}' gallery. Processes 3 wells in parallel by default for speed.
        """
        try:
            # Check authentication
            if context and not self.check_permission(context.get("user", {})):
                raise Exception("User not authorized to access this service")

            # Check if zarr artifact manager is available
            if self.artifact_manager is None:
                raise Exception("Artifact manager not initialized. Check that AGENT_LENS_WORKSPACE_TOKEN is set.")

            logger.info(f"Starting offline processing for experiment ID: {experiment_id}")
            logger.info(f"Parameters: upload_immediately={upload_immediately}, cleanup_temp_files={cleanup_temp_files}, use_parallel_wells={use_parallel_wells}")
            logger.info("🧵 Running offline processing in separate thread to prevent network disconnections")

            # Define the blocking processing function to run in a thread
            def run_offline_processing():
                """
                Run the offline processing in a separate thread.
                This prevents blocking the main event loop and maintains network connections.
                """
                try:
                    # Import and create the offline processor
                    from .offline_processing import OfflineProcessor
                    processor = OfflineProcessor(
                        self.squidController,
                        self.artifact_manager,
                        self.service_id
                    )
                    logger.info("OfflineProcessor created successfully in worker thread")

                    # Create a new event loop for this thread since offline processing uses async operations
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)

                    try:
                        # Run the offline processing in the new event loop
                        logger.info("Calling processor.stitch_and_upload_timelapse in worker thread...")
                        result = loop.run_until_complete(
                            processor.stitch_and_upload_timelapse(
                                experiment_id, upload_immediately, cleanup_temp_files,
                                use_parallel_wells=use_parallel_wells
                            )
                        )

                        logger.info(f"Offline processing completed in worker thread: {result.get('total_datasets', 0)} datasets processed")
                        logger.info(f"Processing mode: {result.get('processing_mode', 'unknown')}")
                        return result

                    finally:
                        # Clean up the event loop
                        loop.close()

                except Exception as e:
                    logger.error(f"Error in offline processing worker thread: {e}")
                    return {
                        "success": False,
                        "error": str(e),
                        "experiment_id": experiment_id
                    }

            # Run the processing function in a separate thread using asyncio.to_thread
            logger.info("🚀 Launching offline processing in worker thread...")
            result = await asyncio.to_thread(run_offline_processing)

            logger.info(f"🎉 Offline processing thread completed: {result}")
            return result

        except Exception as e:
            logger.error(f"Error in offline stitching and upload service method: {e}")
            raise e

    # ===== Unified Scan API Methods =====

    async def _run_scan_with_state_tracking(self, scan_method, *args, **kwargs):
        """
        Internal wrapper that updates scan state during execution.
        
        This method wraps existing scan methods to provide unified state tracking
        for the scan_start/scan_get_status/scan_cancel API.
        
        Args:
            scan_method: The async scan method to execute
            *args, **kwargs: Arguments to pass to the scan method
            
        Returns:
            The result from the scan method, or None if failed
        """
        try:
            self.scan_state['state'] = 'running'
            self.scan_state['error_message'] = None

            logger.info(f"Starting scan with method: {scan_method.__name__}")

            # Run the actual scan method
            result = await scan_method(*args, **kwargs)

            self.scan_state['state'] = 'completed'
            logger.info(f"Scan completed successfully: {scan_method.__name__}")
            return result

        except asyncio.CancelledError:
            self.scan_state['state'] = 'failed'
            self.scan_state['error_message'] = 'Scan was cancelled by user'
            logger.info("Scan was cancelled by user")
            raise

        except Exception as e:
            self.scan_state['state'] = 'failed'
            self.scan_state['error_message'] = str(e)
            logger.error(f"Scan failed: {e}", exc_info=True)

        finally:
            self.scan_state['scan_task'] = None

    @schema_function(skip_self=True)
    async def scan_start(self,
                        config: dict = Field(..., description="Scan configuration dictionary containing all scan parameters"),
                        context=None):
        """
        Launch a background scanning operation with one of three profiles: raw_images, full_zarr, or quick_zarr.
        Returns: Dictionary with success status, profile type, action_ID, and scan state ('running').
        Notes: Scan executes asynchronously. Use scan_get_status() to monitor progress and scan_cancel() to abort.
        
        Config dictionary must contain 'saved_data_type' (str): 'raw_images', 'full_zarr', or 'quick_zarr'
        
        Common parameters:
        - action_ID (str): Unique identifier for this scan operation
        - well_plate_type (str): Well plate format: '6', '12', '24', '96', or '384'
        - do_contrast_autofocus (bool): Enable contrast-based autofocus
        - do_reflection_af (bool): Enable reflection-based laser autofocus
        
        For 'raw_images':
        - illumination_settings (List[dict]): Illumination settings
        - wells_to_scan (List[str]): List of wells to scan (e.g., ['A1', 'B2', 'C3'])
        - Nx, Ny (int): Grid dimensions
        - dx, dy (float): Position intervals in mm
        
        For 'full_zarr':
        - start_x_mm, start_y_mm (float, optional): Starting position in mm (relative to well center).
          If None, the grid will be automatically centered around the well center (similar to plate_scan behavior).
        - Nx, Ny (int): Grid dimensions
        - dx_mm, dy_mm (float): Position intervals in mm
        - illumination_settings (List[dict]): Illumination settings
        - wells_to_scan (List[str]): List of wells to scan
        - well_padding_mm (float): Padding around well in mm
        - experiment_name (str): Experiment name
        - uploading (bool): Enable upload after scanning
        - timepoint (int): Timepoint index for time-lapse
        
        For 'quick_zarr':
        - start_x_mm (float, required): Starting X position in mm (left edge of scan area)
        - start_y_mm (float, required): Starting Y position in mm (bottom edge of scan area)
        - scan_width_mm (float, required): Width of scan area in mm (X direction)
        - scan_height_mm (float, required): Height of scan area in mm (Y direction)
        - dy_mm (float): Y increment between horizontal stripes in mm (default 0.85)
        - exposure_time (float): Camera exposure time in ms (max 30ms, default 5)
        - intensity (float): Brightfield LED intensity 0-100 (default 70)
        - fps_target (int): Target frame rate for acquisition (default 10)
        - velocity_scan_mm_per_s (float): Stage velocity during scanning in mm/s (default 7.0)
        - experiment_name (str): Experiment name
        - uploading (bool): Enable upload after scanning
        """
        try:
            # Check authentication
            if context and not self.check_permission(context.get("user", {})):
                raise Exception("User not authorized to access this service")

            # Check if scan already running
            if self.scan_state['state'] == 'running':
                raise Exception("A scan is already in progress. Use scan_cancel() to stop it first.")

            # Extract saved_data_type from config
            saved_data_type = config.get('saved_data_type')
            if not saved_data_type:
                raise ValueError("Config must contain 'saved_data_type' parameter")

            # Validate saved_data_type
            valid_types = ['raw_images', 'full_zarr', 'quick_zarr']
            if saved_data_type not in valid_types:
                raise ValueError(f"Invalid saved_data_type '{saved_data_type}'. Must be one of: {valid_types}")

            # Store the scan type
            self.scan_state['saved_data_type'] = saved_data_type

            # Extract common parameters with defaults
            action_ID = config.get('action_ID', 'unified_scan')
            do_contrast_autofocus = config.get('do_contrast_autofocus', False)
            do_reflection_af = config.get('do_reflection_af', True)
            well_plate_type = config.get('well_plate_type', '96')

            logger.info(f"Starting unified scan with profile: {saved_data_type}")

            # Route to appropriate scan method based on profile
            if saved_data_type == 'raw_images':
                # Extract raw_images specific parameters
                illumination_settings = config.get('illumination_settings')
                wells_to_scan = config.get('wells_to_scan', ['A1'])
                Nx = config.get('Nx', 3)
                Ny = config.get('Ny', 3)
                dx = config.get('dx', 0.8)
                dy = config.get('dy', 0.8)

                scan_coro = self._run_scan_with_state_tracking(
                    self.scan_plate_save_raw_images,
                    well_plate_type=well_plate_type,
                    illumination_settings=illumination_settings,
                    do_contrast_autofocus=do_contrast_autofocus,
                    do_reflection_af=do_reflection_af,
                    wells_to_scan=wells_to_scan,
                    Nx=Nx,
                    Ny=Ny,
                    dx=dx,
                    dy=dy,
                    action_ID=action_ID,
                    context=context
                )

            elif saved_data_type == 'full_zarr':
                # Extract full_zarr specific parameters
                # start_x_mm and start_y_mm are optional - if not provided, grid will be centered
                start_x_mm = config.get('start_x_mm', None)  # None = auto-center
                start_y_mm = config.get('start_y_mm', None)  # None = auto-center
                Nx = config.get('Nx', 3)
                Ny = config.get('Ny', 3)
                dx_mm = config.get('dx_mm', 0.9)
                dy_mm = config.get('dy_mm', 0.9)
                illumination_settings = config.get('illumination_settings')
                wells_to_scan = config.get('wells_to_scan', ['A1'])
                well_padding_mm = config.get('well_padding_mm', 1.0)
                experiment_name = config.get('experiment_name')
                uploading = config.get('uploading', False)
                timepoint = config.get('timepoint', 0)

                scan_coro = self._run_scan_with_state_tracking(
                    self.scan_region_to_zarr,
                    start_x_mm=start_x_mm,
                    start_y_mm=start_y_mm,
                    Nx=Nx,
                    Ny=Ny,
                    dx_mm=dx_mm,
                    dy_mm=dy_mm,
                    illumination_settings=illumination_settings,
                    do_contrast_autofocus=do_contrast_autofocus,
                    do_reflection_af=do_reflection_af,
                    action_ID=action_ID,
                    timepoint=timepoint,
                    experiment_name=experiment_name,
                    wells_to_scan=wells_to_scan,
                    well_plate_type=well_plate_type,
                    well_padding_mm=well_padding_mm,
                    uploading=uploading,
                    context=context
                )

            elif saved_data_type == 'quick_zarr':
                # Extract quick_zarr specific parameters (now using region-based scanning)
                start_x_mm = config.get('start_x_mm')
                start_y_mm = config.get('start_y_mm')
                scan_width_mm = config.get('scan_width_mm')
                scan_height_mm = config.get('scan_height_mm')
                dy_mm = config.get('dy_mm', 0.85)
                exposure_time = config.get('exposure_time', 5)
                intensity = config.get('intensity', 70)
                fps_target = config.get('fps_target', 10)
                velocity_scan_mm_per_s = config.get('velocity_scan_mm_per_s', 7.0)
                experiment_name = config.get('experiment_name')
                uploading = config.get('uploading', False)

                # Validate required parameters for quick_zarr
                if start_x_mm is None or start_y_mm is None:
                    raise ValueError("quick_zarr scan requires 'start_x_mm' and 'start_y_mm' parameters")
                if scan_width_mm is None or scan_height_mm is None:
                    raise ValueError("quick_zarr scan requires 'scan_width_mm' and 'scan_height_mm' parameters")

                scan_coro = self._run_scan_with_state_tracking(
                    self.quick_scan_region_to_zarr,
                    start_x_mm=start_x_mm,
                    start_y_mm=start_y_mm,
                    scan_width_mm=scan_width_mm,
                    scan_height_mm=scan_height_mm,
                    dy_mm=dy_mm,
                    exposure_time=exposure_time,
                    intensity=intensity,
                    fps_target=fps_target,
                    velocity_scan_mm_per_s=velocity_scan_mm_per_s,
                    experiment_name=experiment_name,
                    uploading=uploading,
                    context=context
                )

            # Launch scan in background
            self.scan_state['scan_task'] = asyncio.create_task(scan_coro)

            logger.info(f"Scan started in background with profile: {saved_data_type}")

            return {
                "success": True,
                "message": f"Scan started with profile '{saved_data_type}'",
                "saved_data_type": saved_data_type,
                "action_ID": action_ID,
                "state": "running"
            }

        except Exception as e:
            logger.error(f"Failed to start scan: {e}")
            raise e

    @schema_function(skip_self=True)
    def scan_get_status(self, context=None):
        """
        Query the current state of the background scanning operation.
        Returns: Dictionary with success flag, scan state ('idle', 'running', 'completed', 'failed'), error_message (if failed), and saved_data_type.
        Notes: State persists across client disconnections. Poll this endpoint to monitor long-running scans.
        """
        try:
            # Check authentication
            if context and not self.check_permission(context.get("user", {})):
                raise Exception("User not authorized to access this service")

            return {
                "success": True,
                "state": self.scan_state['state'],
                "error_message": self.scan_state['error_message'],
                "saved_data_type": self.scan_state['saved_data_type']
            }

        except Exception as e:
            logger.error(f"Failed to get scan status: {e}")
            raise e

    @schema_function(skip_self=True)
    async def scan_cancel(self, context=None):
        """
        Abort the currently running scan operation and return microscope to idle state.
        Returns: Dictionary with success status, confirmation message, and final scan state.
        Notes: Works with any scan profile (raw_images, full_zarr, quick_zarr). Scan stops gracefully where possible. Partial data may be retained.
        """
        try:
            # Check authentication
            if context and not self.check_permission(context.get("user", {})):
                raise Exception("User not authorized to access this service")

            # Check if scan is running
            if self.scan_state['state'] != 'running':
                return {
                    "success": True,
                    "message": f"No scan to cancel. Current state: {self.scan_state['state']}",
                    "state": self.scan_state['state']
                }

            logger.info("Scan cancellation requested")

            # Set stop flags at controller level
            if hasattr(self.squidController, 'scan_stop_requested'):
                self.squidController.scan_stop_requested = True
                logger.info("Set squidController.scan_stop_requested = True")

            # Call controller's stop method
            self.squidController.stop_scan_and_stitching()

            # Cancel the scan task
            if self.scan_state['scan_task'] and not self.scan_state['scan_task'].done():
                self.scan_state['scan_task'].cancel()
                logger.info("Cancelled scan task")

                # Wait a moment for cancellation to propagate
                try:
                    await asyncio.wait_for(self.scan_state['scan_task'], timeout=1.0)
                except (asyncio.CancelledError, asyncio.TimeoutError):
                    pass

            # Update state
            self.scan_state['state'] = 'failed'
            self.scan_state['error_message'] = 'Scan cancelled by user'
            self.scan_state['scan_task'] = None

            logger.info("Scan cancelled successfully")

            return {
                "success": True,
                "message": "Scan cancelled successfully",
                "state": self.scan_state['state']
            }

        except Exception as e:
            logger.error(f"Failed to cancel scan: {e}")
            raise e

    # ===== Segmentation API Methods =====

    async def _run_segmentation_background(self, source_experiment, wells_to_segment, channel_configs,
                                          scale_level, timepoint, well_plate_type, well_padding_mm, reset_application=True, context=None):
        """
        Internal method for background segmentation processing with multi-channel support.
        
        This method runs in a separate task to segment experiment wells using microSAM service.
        
        Args:
            source_experiment: Name of the source experiment
            wells_to_segment: List of well identifiers to segment
            channel_configs: List of channel configurations for merging
            scale_level: Pyramid scale level
            timepoint: Timepoint index
            well_plate_type: Well plate format
            well_padding_mm: Well padding in millimeters
            context: Request context
            
        Returns:
            results: Dictionary with segmentation results
        """
        try:
            self.segmentation_state['state'] = 'running'
            self.segmentation_state['error_message'] = None

            logger.info(f"🚀 Starting background segmentation for experiment '{source_experiment}'")
            logger.info(f"Event loop status: running={asyncio.get_event_loop().is_running()}")

            # Import microsam_client module
            from squid_control.hypha_tools.microsam_client import (
                connect_to_microsam,
                segment_experiment_wells,
            )

            # Verify server connection before starting
            if self.remote_server is None:
                raise Exception("Remote server connection not available")

            # Connect to microSAM service using remote_server (agent-lens workspace)
            logger.info("Connecting to microSAM service...")
            logger.info(f"Remote server workspace: {self.remote_server.config.workspace}")

            microsam_service = await connect_to_microsam(self.remote_server)

            # Define progress callback
            def progress_callback(well_id, completed, total):
                self.segmentation_state['progress']['completed_wells'] = completed
                self.segmentation_state['progress']['current_well'] = well_id
                logger.info(f"Progress: {completed}/{total} wells completed (current: {well_id})")

            # Define similarity search progress callback
            def similarity_search_callback(message, current, total):
                self.segmentation_state['progress']['current_stage'] = message
                # Update specific counts based on stage
                if 'Extracting' in message or 'Loaded' in message:
                    self.segmentation_state['progress']['similarity_search_status'] = 'extracting'
                    self.segmentation_state['progress']['total_polygons'] = total
                    self.segmentation_state['progress']['extracted_count'] = current
                elif 'embedding' in message.lower():
                    self.segmentation_state['progress']['similarity_search_status'] = 'embedding'
                    self.segmentation_state['progress']['embedding_count'] = current
                elif 'Uploading' in message or 'Setting up' in message:
                    self.segmentation_state['progress']['similarity_search_status'] = 'uploading'
                    self.segmentation_state['progress']['uploaded_count'] = current
                elif 'Complete' in message:
                    self.segmentation_state['progress']['similarity_search_status'] = 'completed'
                    self.segmentation_state['progress']['uploaded_count'] = current
                logger.info(f"Similarity search: {message} ({current}/{total})")

            # Initialize progress tracking
            self.segmentation_state['progress']['total_wells'] = len(wells_to_segment)
            self.segmentation_state['progress']['completed_wells'] = 0
            self.segmentation_state['progress']['channel_configs'] = channel_configs
            self.segmentation_state['progress']['source_experiment'] = source_experiment
            self.segmentation_state['progress']['similarity_search_enabled'] = True

            # Run segmentation with multi-channel support
            results = await segment_experiment_wells(
                microsam_service=microsam_service,
                experiment_manager=self.squidController.experiment_manager,
                source_experiment=source_experiment,
                wells_to_segment=wells_to_segment,
                channel_configs=channel_configs,
                scale_level=scale_level,
                timepoint=timepoint,
                well_plate_type=well_plate_type,
                well_padding_mm=well_padding_mm,
                progress_callback=progress_callback,
                enable_similarity_search=True,
                similarity_search_callback=similarity_search_callback,
                reset_application=reset_application
            )

            # Update state with similarity search results
            if 'similarity_search_results' in results:
                sim_results = results['similarity_search_results']
                self.segmentation_state['progress']['total_polygons'] = sim_results.get('total_polygons', 0)
                self.segmentation_state['progress']['extracted_count'] = sim_results.get('extracted_count', 0)
                self.segmentation_state['progress']['embedding_count'] = sim_results.get('embedding_success_count', 0)
                self.segmentation_state['progress']['uploaded_count'] = sim_results.get('uploaded_count', 0)
                self.segmentation_state['progress']['failed_count'] = sim_results.get('failed_count', 0)
                
                if sim_results.get('success'):
                    self.segmentation_state['progress']['similarity_search_status'] = 'completed'
                else:
                    self.segmentation_state['progress']['similarity_search_status'] = 'failed'
            
            # Update state to completed
            self.segmentation_state['state'] = 'completed'
            logger.info("✅ Segmentation completed successfully!")
            logger.info(f"  Successful wells: {results['successful_wells']}/{results['total_wells']}")
            
            # Log similarity search results if available
            if 'similarity_search_results' in results:
                sim_results = results['similarity_search_results']
                if sim_results.get('success'):
                    logger.info(f"  Similarity search: {sim_results['uploaded_count']} cells uploaded to Weaviate")
                else:
                    logger.warning(f"  Similarity search: Failed with {sim_results.get('failed_count', 0)} errors")

            return results

        except asyncio.CancelledError:
            self.segmentation_state['state'] = 'failed'
            self.segmentation_state['error_message'] = 'Segmentation was cancelled by user'
            logger.info("Segmentation was cancelled by user")
            raise

        except Exception as e:
            self.segmentation_state['state'] = 'failed'
            self.segmentation_state['error_message'] = str(e)
            logger.error(f"Segmentation failed: {e}", exc_info=True)
            raise

        finally:
            self.segmentation_state['segmentation_task'] = None

    @schema_function(skip_self=True)
    async def segmentation_start(self,
        experiment_name: str = Field(..., description="Source experiment name to process"),
        wells_to_segment: Optional[List[str]] = Field(None, description="List of wells (e.g., ['A1', 'B2']). None = all wells in experiment"),
        channel_configs: List[Dict[str, Any]] = Field(
            ...,
            description="Channel configurations for merging. Each dict must have 'channel' (name), and optionally 'min_percentile' (default 1.0), 'max_percentile' (default 99.0), 'weight' (default 1.0). Example: [{'channel': 'BF LED matrix full', 'min_percentile': 2.0, 'max_percentile': 98.0}]"
        ),
        scale_level: int = Field(1, description="Pyramid scale level: 0=full resolution, 1=1/4x, 2=1/16x, etc. Higher scales process faster."),
        well_plate_type: str = Field("96", description="Well plate format: '6', '12', '24', '96', or '384'"),
        well_padding_mm: float = Field(1.0, description="Well boundary padding in millimeters"),
        context=None):
        """
        Launch multi-channel segmentation with color-mapped merging using microSAM BioEngine service.
        Returns: Dictionary with success status, source experiment name, state, and total wells count.
        Notes: Segmentation executes asynchronously. Polygon results saved to '{experiment_name}/polygons.json'. Use segmentation_get_status() to monitor progress and segmentation_cancel() to abort.
        """
        try:
            # Check authentication
            if context and not self.check_permission(context.get("user", {})):
                raise Exception("User not authorized to access this service")

            # Check if segmentation already running
            if self.segmentation_state['state'] == 'running':
                raise Exception("A segmentation is already in progress. Use segmentation_cancel() to stop it first.")

            logger.info(f"Segmentation start requested for experiment '{experiment_name}'")

            # Validate source experiment exists
            if not hasattr(self.squidController, 'experiment_manager'):
                raise Exception("Experiment manager not initialized")

            experiment_path = self.squidController.experiment_manager.base_path / experiment_name
            if not experiment_path.exists():
                raise ValueError(f"Source experiment '{experiment_name}' does not exist")

            # Auto-detect wells if not specified
            if wells_to_segment is None:
                logger.info("No wells specified, auto-detecting all wells in experiment...")

                # List all well zarr filesets in the experiment directory
                detected_wells = []
                for item in experiment_path.iterdir():
                    if item.is_dir() and item.suffix == '.zarr' and item.stem.startswith('well_'):
                        # Parse well identifier from fileset name (e.g., "well_A1_96.zarr" -> "A1")
                        import re
                        match = re.match(r'well_([A-Z]+)(\d+)_\d+', item.stem)
                        if match:
                            well_id = f"{match.group(1)}{match.group(2)}"
                            detected_wells.append(well_id)

                if not detected_wells:
                    raise ValueError(f"No wells found in experiment '{experiment_name}'")

                wells_to_segment = sorted(detected_wells)
                logger.info(f"Auto-detected {len(wells_to_segment)} wells: {wells_to_segment}")

            # Validate wells_to_segment is a list
            if not isinstance(wells_to_segment, list) or len(wells_to_segment) == 0:
                raise ValueError("wells_to_segment must be a non-empty list")

            # Validate channel_configs
            if not channel_configs or len(channel_configs) == 0:
                raise ValueError("channel_configs must contain at least one channel configuration")

            for config in channel_configs:
                if 'channel' not in config:
                    raise ValueError("Each channel config must have 'channel' key")
                # Set defaults
                config.setdefault('min_percentile', 1.0)
                config.setdefault('max_percentile', 99.0)
                config.setdefault('weight', 1.0)

                # Validate percentile ranges
                if not (0 <= config['min_percentile'] <= 100):
                    raise ValueError(f"min_percentile must be between 0 and 100, got {config['min_percentile']}")
                if not (0 <= config['max_percentile'] <= 100):
                    raise ValueError(f"max_percentile must be between 0 and 100, got {config['max_percentile']}")
                if config['min_percentile'] >= config['max_percentile']:
                    raise ValueError("min_percentile must be less than max_percentile")

            logger.info("Segmentation configuration:")
            logger.info(f"  Source experiment: '{experiment_name}'")
            logger.info(f"  Wells to segment: {wells_to_segment}")
            logger.info(f"  Channels: {len(channel_configs)} channel(s)")
            for config in channel_configs:
                logger.info(f"    - {config['channel']}: {config['min_percentile']}%-{config['max_percentile']}%, weight={config['weight']}")
            logger.info(f"  Scale level: {scale_level}")
            logger.info(f"  Well plate type: '{well_plate_type}'")

            # Launch segmentation in background (always use timepoint=0 for single timepoint experiments)
            self.segmentation_state['segmentation_task'] = asyncio.create_task(
                self._run_segmentation_background(
                    source_experiment=experiment_name,
                    wells_to_segment=wells_to_segment,
                    channel_configs=channel_configs,
                    scale_level=scale_level,
                    timepoint=0,  # Always use timepoint 0 for single timepoint experiments
                    well_plate_type=well_plate_type,
                    well_padding_mm=well_padding_mm,
                    context=context
                )
            )

            logger.info("✅ Segmentation task launched in background")

            return {
                "success": True,
                "message": f"Segmentation started for experiment '{experiment_name}'",
                "source_experiment": experiment_name,
                "state": "running",
                "total_wells": len(wells_to_segment),
                "wells_to_segment": wells_to_segment
            }

        except Exception as e:
            logger.error(f"Failed to start segmentation: {e}", exc_info=True)
            raise e

    @schema_function(skip_self=True)
    def segmentation_get_status(self, context=None):
        """
        Query the current state of the background segmentation operation.
        Returns: Dictionary with success flag, segmentation state ('idle', 'running', 'completed', 'failed'), progress details, and error_message (if failed).
        Notes: State persists across client disconnections. Poll this endpoint to monitor long-running segmentations.
        """
        try:
            # Check authentication
            if context and not self.check_permission(context.get("user", {})):
                raise Exception("User not authorized to access this service")

            return {
                "success": True,
                "state": self.segmentation_state['state'],
                "error_message": self.segmentation_state['error_message'],
                "progress": self.segmentation_state['progress']
            }

        except Exception as e:
            logger.error(f"Failed to get segmentation status: {e}")
            raise e

    @schema_function(skip_self=True)
    async def segmentation_cancel(self, context=None):
        """
        Abort the currently running segmentation operation.
        Returns: Dictionary with success status, confirmation message, and final segmentation state.
        Notes: Segmentation stops gracefully where possible. Partial data may be retained in the segmentation experiment.
        """
        try:
            # Check authentication
            if context and not self.check_permission(context.get("user", {})):
                raise Exception("User not authorized to access this service")

            # Check if segmentation is running
            if self.segmentation_state['state'] != 'running':
                return {
                    "success": True,
                    "message": f"No segmentation to cancel. Current state: {self.segmentation_state['state']}",
                    "state": self.segmentation_state['state']
                }

            logger.info("Segmentation cancellation requested")

            # Cancel the segmentation task
            if self.segmentation_state['segmentation_task'] and not self.segmentation_state['segmentation_task'].done():
                self.segmentation_state['segmentation_task'].cancel()
                logger.info("Cancelled segmentation task")

                # Wait a moment for cancellation to propagate
                try:
                    await asyncio.wait_for(self.segmentation_state['segmentation_task'], timeout=1.0)
                except (asyncio.CancelledError, asyncio.TimeoutError):
                    pass

            # Update state
            self.segmentation_state['state'] = 'failed'
            self.segmentation_state['error_message'] = 'Segmentation cancelled by user'
            self.segmentation_state['segmentation_task'] = None

            logger.info("Segmentation cancelled successfully")

            return {
                "success": True,
                "message": "Segmentation cancelled successfully",
                "state": self.segmentation_state['state']
            }

        except Exception as e:
            logger.error(f"Failed to cancel segmentation: {e}")
            raise e

    async def segmentation_get_polygons(self, experiment_name: str, well_id: str = None, context=None):
        """
        Retrieve polygon annotations from a completed segmentation.
        
        This endpoint reads polygon data from the polygons.json file in the source
        experiment folder. Polygons are in WKT format with well-relative millimeter coordinates.
        
        Args:
            experiment_name: Name of the source experiment
            well_id: Optional well identifier to filter results (e.g., "A1", "B2")
            context: Request context for authentication
        
        Returns:
            Dictionary with:
            - success: Boolean indicating if operation succeeded
            - polygons: List of polygon objects with "well_id" and "polygon_wkt" fields
            - total_count: Total number of polygons returned
            - experiment_name: Name of the source experiment
        
        Note: This endpoint does NOT use @schema_function decorator as requested.
        """
        try:
            # Check authentication
            if context and not self.check_permission(context.get("user", {})):
                raise Exception("User not authorized to access this service")

            logger.info(f"Fetching polygons from experiment: '{experiment_name}'")
            if well_id:
                logger.info(f"Filtering for well: {well_id}")

            # Get path to polygons.json in source experiment
            if not hasattr(self.squidController, 'experiment_manager'):
                raise Exception("Experiment manager not initialized")

            experiment_path = self.squidController.experiment_manager.base_path / experiment_name
            json_path = experiment_path / "polygons.json"

            # Check if file exists
            if not json_path.exists():
                logger.info(f"No polygons.json found in '{experiment_name}', returning empty list")
                return {
                    "success": True,
                    "polygons": [],
                    "total_count": 0,
                    "experiment_name": experiment_name,
                    "message": "No polygon data available (polygons.json not found)"
                }

            # Read polygons from JSON file
            try:
                with open(json_path) as f:
                    data = json.load(f)

                all_polygons = data.get("polygons", [])

                # Filter by well_id if specified
                if well_id:
                    filtered_polygons = [p for p in all_polygons if p.get("well_id") == well_id]
                    logger.info(f"Filtered {len(filtered_polygons)} polygons for well {well_id} "
                               f"(out of {len(all_polygons)} total)")
                else:
                    filtered_polygons = all_polygons
                    logger.info(f"Returning all {len(filtered_polygons)} polygons")

                return {
                    "success": True,
                    "polygons": filtered_polygons,
                    "total_count": len(filtered_polygons),
                    "experiment_name": experiment_name
                }

            except json.JSONDecodeError as json_err:
                logger.error(f"Failed to parse polygons.json: {json_err}")
                raise Exception(f"Corrupt polygons.json file: {json_err}")

        except Exception as e:
            logger.error(f"Failed to get polygons: {e}", exc_info=True)
            raise e

    async def stop_scan_and_stitching(self, context=None):
        """
        [DEPRECATED] Stop scan and stitching operations.
        
        This endpoint is deprecated. Use scan_cancel() instead for unified scan operations.
        This method is kept for backward compatibility and routes to scan_cancel().
        
        Returns:
            dict: Status with success flag and message
        """
        logger.warning("stop_scan_and_stitching is deprecated. Use scan_cancel() instead.")

        try:
            # Check authentication
            if context and not self.check_permission(context.get("user", {})):
                raise Exception("User not authorized to access this service")

            # Route to the unified scan_cancel method
            result = await self.scan_cancel(context)

            # Update the message to indicate this was called via deprecated endpoint
            if result.get("success"):
                result["message"] = f"[DEPRECATED] {result['message']} (Use scan_cancel() instead)"

            return result

        except Exception as e:
            logger.error(f"Failed to stop scan and stitching: {e}")
            raise e

    # ===== Squid+ Specific API Methods =====

    @schema_function(skip_self=True)
    async def set_filter_wheel_position(
        self,
        position: int = Field(..., description="Target filter wheel position (range: 1-8)"),
        context=None
    ):
        """
        Move the motorized filter wheel to a specific position (Squid+ only).
        Returns: Dictionary with success status, position number, and confirmation message.
        Notes: Only available on Squid+ microscopes with filter wheel hardware. Validates position is within 1-8 range.
        """
        try:
            if context and not self.check_permission(context.get("user", {})):
                raise Exception("User not authorized to access this service")

            if self.squidController.filter_wheel is None:
                raise Exception("Filter wheel not available on this microscope")

            # Handle case where parameters might be passed as a dictionary
            if isinstance(position, dict):
                # Extract position from dictionary
                logger.info(f"Received dictionary parameters: {position}")
                position_value = position.get('position', 1)
            else:
                # Convert position to int in case it's an ObjectProxy
                logger.info(f"Received position as: {type(position)} = {position}")
                position_value = int(position)

            success = self.squidController.filter_wheel.set_filter_position(position_value)

            if success:
                logger.info(f"Filter wheel moved to position {position_value}")
                return {
                    "success": True,
                    "position": position_value,
                    "message": f"Filter wheel set to position {position_value}"
                }
            else:
                raise Exception(f"Failed to set filter wheel to position {position_value}")

        except Exception as e:
            logger.error(f"Error setting filter wheel position: {e}")
            raise e

    @schema_function(skip_self=True)
    async def get_filter_wheel_position(self, context=None):
        """
        Query the current filter wheel position (Squid+ only).
        Returns: Dictionary with success status and current position number (1-8).
        Notes: Only available on Squid+ microscopes with filter wheel hardware.
        """
        try:
            if context and not self.check_permission(context.get("user", {})):
                raise Exception("User not authorized to access this service")

            if self.squidController.filter_wheel is None:
                raise Exception("Filter wheel not available on this microscope")

            position = self.squidController.filter_wheel.get_filter_position()
            return {
                "success": True,
                "position": position
            }

        except Exception as e:
            logger.error(f"Error getting filter wheel position: {e}")
            raise e

    @schema_function(skip_self=True)
    async def next_filter_position(self, context=None):
        """
        Advance the filter wheel to the next sequential position (Squid+ only).
        Returns: Dictionary with success status, new position number, and confirmation message.
        Notes: Only available on Squid+ microscopes. Wraps around from position 8 to position 1.
        """
        try:
            if context and not self.check_permission(context.get("user", {})):
                raise Exception("User not authorized to access this service")

            if self.squidController.filter_wheel is None:
                raise Exception("Filter wheel not available on this microscope")

            success = self.squidController.filter_wheel.next_position()

            if success:
                position = self.squidController.filter_wheel.get_filter_position()
                return {
                    "success": True,
                    "position": position,
                    "message": f"Moved to filter position {position}"
                }
            else:
                raise Exception("Failed to move to next filter position")

        except Exception as e:
            logger.error(f"Error moving to next filter position: {e}")
            raise e

    @schema_function(skip_self=True)
    async def previous_filter_position(self, context=None):
        """
        Move the filter wheel to the previous sequential position (Squid+ only).
        Returns: Dictionary with success status, new position number, and confirmation message.
        Notes: Only available on Squid+ microscopes. Wraps around from position 1 to position 8.
        """
        try:
            if context and not self.check_permission(context.get("user", {})):
                raise Exception("User not authorized to access this service")

            if self.squidController.filter_wheel is None:
                raise Exception("Filter wheel not available on this microscope")

            success = self.squidController.filter_wheel.previous_position()

            if success:
                position = self.squidController.filter_wheel.get_filter_position()
                return {
                    "success": True,
                    "position": position,
                    "message": f"Moved to filter position {position}"
                }
            else:
                raise Exception("Failed to move to previous filter position")

        except Exception as e:
            logger.error(f"Error moving to previous filter position: {e}")
            raise e

    @schema_function(skip_self=True)
    async def switch_objective(
        self,
        objective_name: str = Field(..., description="Target objective identifier (e.g., '4x', '10x', '20x', '40x')"),
        move_z: bool = Field(True, description="Automatically adjust Z stage to compensate for objective height difference (True recommended)"),
        context=None
    ):
        """
        Switch the motorized objective turret to a different magnification (Squid+ only).
        Returns: Dictionary with success status, objective name, position number, pixel size (µm), and confirmation message.
        Notes: Only available on Squid+ microscopes with objective switcher. Automatically updates pixel size and imaging parameters. Set move_z=True to maintain focus position.
        """
        try:
            if context and not self.check_permission(context.get("user", {})):
                raise Exception("User not authorized to access this service")

            if self.squidController.objective_switcher is None:
                raise Exception("Objective switcher not available on this microscope")

            # Handle case where parameters might be passed as a dictionary
            if isinstance(objective_name, dict):
                # Extract parameters from dictionary
                logger.info(f"Received dictionary parameters: {objective_name}")
                objective_name_str = str(objective_name.get('objective_name', ''))
                move_z = objective_name.get('move_z', True)
            else:
                # Convert objective_name to string in case it's an ObjectProxy
                logger.info(f"Received objective_name as: {type(objective_name)} = {objective_name}")
                objective_name_str = str(objective_name)

            # Validate that we have a valid objective name
            if not objective_name_str or objective_name_str.strip() == '':
                raise Exception("No objective name provided")

            logger.info(f"Looking for objective: '{objective_name_str}'")

            # Get available objectives and their positions
            position_names = self.squidController.objective_switcher.get_position_names()
            logger.info(f"Available objectives: {position_names}")

            # Find the position for the requested objective
            position = None
            for pos, name in position_names.items():
                if name.lower() == objective_name_str.lower():
                    position = pos
                    break

            if position is None:
                available_objectives = list(position_names.values())
                raise Exception(f"Objective '{objective_name_str}' not found. Available objectives: {available_objectives}")

            # Move to the found position
            if position == 1:
                success = self.squidController.objective_switcher.move_to_position_1(move_z=move_z)
            elif position == 2:
                success = self.squidController.objective_switcher.move_to_position_2(move_z=move_z)
            else:
                raise Exception(f"Invalid objective position: {position}")

            if success:
                logger.info(f"Objective switcher switched to {objective_name_str} (position {position})")

                # Update objective-related parameters after successful switch
                try:
                    # Update the current objective in objectiveStore
                    self.squidController.objectiveStore.current_objective = objective_name_str
                    logger.info(f"Updated objectiveStore.current_objective to: {objective_name_str}")

                    # Recalculate pixel size and related parameters
                    self.squidController.get_pixel_size()
                    logger.info(f"Recalculated pixel size: {self.squidController.pixel_size_xy} µm")

                    # Log the updated parameters
                    logger.info(f"Objective switch completed - New objective: {objective_name_str}, "
                              f"Pixel size: {self.squidController.pixel_size_xy} µm")

                except Exception as param_error:
                    logger.warning(f"Failed to update objective parameters: {param_error}")
                    # Don't fail the entire operation if parameter update fails

                return {
                    "success": True,
                    "objective_name": objective_name_str,
                    "position": position,
                    "pixel_size_xy": getattr(self.squidController, 'pixel_size_xy', None),
                    "message": f"Switched to objective {objective_name_str}"
                }
            else:
                raise Exception(f"Failed to switch to objective {objective_name_str}")

        except Exception as e:
            logger.error(f"Error switching objective: {e}")
            raise e

    @schema_function(skip_self=True)
    async def get_current_objective(self, context=None):
        """
        Query the currently active objective and list all available objectives (Squid+ only).
        Returns: Dictionary with success status, current objective name, position number, list of available objectives, and confirmation message.
        Notes: Only available on Squid+ microscopes with objective switcher.
        """
        try:
            if context and not self.check_permission(context.get("user", {})):
                raise Exception("User not authorized to access this service")

            if self.squidController.objective_switcher is None:
                raise Exception("Objective switcher not available on this microscope")

            position = self.squidController.objective_switcher.get_current_position()
            position_names = self.squidController.objective_switcher.get_position_names()
            objective_name = position_names.get(position, "Unknown") if position else "Not set"

            # Get all available objectives
            available_objectives = list(position_names.values())

            return {
                "success": True,
                "current_objective": objective_name,
                "position": position,
                "available_objectives": available_objectives,
                "message": f"Current objective: {objective_name}"
            }

        except Exception as e:
            logger.error(f"Error getting current objective: {e}")
            raise e

    @schema_function
    async def set_objective_switcher_speed(
        self,
        speed: float = Field(..., description="Speed value for objective switcher"),
        context=None
    ):
        """
        Set the movement speed of the objective switcher (Squid+ only)
        
        Args:
            speed: Speed value
            
        Returns:
            dict: Status message
        """
        try:
            if context and not self.check_permission(context.get("user", {})):
                raise Exception("User not authorized to access this service")

            if self.squidController.objective_switcher is None:
                raise Exception("Objective switcher not available on this microscope")

            success = self.squidController.objective_switcher.set_speed(speed)

            if success:
                return {
                    "success": True,
                    "speed": speed,
                    "message": f"Objective switcher speed set to {speed}"
                }
            else:
                raise Exception("Failed to set objective switcher speed")

        except Exception as e:
            logger.error(f"Error setting objective switcher speed: {e}")
            raise e

    @schema_function(skip_self=True)
    async def get_available_objectives(self, context=None):
        """
        List all objectives installed on the motorized turret (Squid+ only).
        Returns: Dictionary with success status, list of objectives (each with position and name), objective names list, position numbers, and confirmation message.
        Notes: Only available on Squid+ microscopes with objective switcher.
        """
        try:
            if context and not self.check_permission(context.get("user", {})):
                raise Exception("User not authorized to access this service")

            if self.squidController.objective_switcher is None:
                raise Exception("Objective switcher not available on this microscope")

            positions = self.squidController.objective_switcher.get_available_positions()
            position_names = self.squidController.objective_switcher.get_position_names()

            # Create a more user-friendly response
            available_objectives = []
            for pos in positions:
                objective_name = position_names.get(pos, f"Position {pos}")
                available_objectives.append({
                    "position": pos,
                    "objective_name": objective_name
                })

            return {
                "success": True,
                "available_objectives": available_objectives,
                "objective_names": list(position_names.values()),
                "positions": positions,
                "message": f"Available objectives: {list(position_names.values())}"
            }

        except Exception as e:
            logger.error(f"Error getting available objectives: {e}")
            raise e

# Global variable to hold the microscope instance
_microscope_instance = None

# Define a signal handler for graceful shutdown
def signal_handler(sig, frame):
    global _microscope_instance
    logger.info('Signal received, shutting down gracefully...')

    # Stop video buffering
    if _microscope_instance and hasattr(_microscope_instance, 'frame_acquisition_running') and _microscope_instance.frame_acquisition_running:
        logger.info('Stopping video buffering...')
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                loop.create_task(_microscope_instance.stop_video_buffering())
            else:
                loop.run_until_complete(_microscope_instance.stop_video_buffering())
                loop.close()
        except Exception as e:
            logger.error(f'Error stopping video buffering: {e}')

    if _microscope_instance and hasattr(_microscope_instance, 'squidController'):
        _microscope_instance.squidController.close()
    sys.exit(0)

# Register the signal handler for SIGINT and SIGTERM
signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)

def main():
    """Main entry point for the microscope service"""
    global _microscope_instance

    parser = argparse.ArgumentParser(
        description="Squid microscope control services for Hypha."
    )
    parser.add_argument(
        "--simulation",
        dest="simulation",
        action="store_true",
        default=False,
        help="Run in simulation mode (default: False)"
    )
    parser.add_argument(
        "--local",
        dest="local",
        action="store_true",
        default=False,
        help="Run with local server URL (default: False)"
    )
    parser.add_argument("--verbose", "-v", action="count")
    args = parser.parse_args()

    if args.verbose:
        logging.basicConfig(level=logging.DEBUG)
    else:
        logging.basicConfig(level=logging.INFO)

    microscope = MicroscopeHyphaService(is_simulation=args.simulation, is_local=args.local)
    _microscope_instance = microscope  # Set the global variable

    loop = asyncio.get_event_loop()

    async def async_main():
        try:
            microscope.setup_task = asyncio.create_task(microscope.setup())
            await microscope.setup_task
        except Exception:
            traceback.print_exc()

    loop.create_task(async_main())
    loop.run_forever()

if __name__ == "__main__":
    main()
