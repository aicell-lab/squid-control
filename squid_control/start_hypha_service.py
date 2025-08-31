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
from functools import partial
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
    from .hypha_tools.artifact_manager.artifact_manager import SquidArtifactManager
    from .hypha_tools.chatbot.aask import aask
    from .hypha_tools.hypha_storage import HyphaDataStore
    from .squid_controller import SquidController
except ImportError:
    # Fallback for direct script execution from project root
    import os
    import sys
    # Add the project root to Python path
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)
    
    from squid_control.control.camera import TriggerModeSetting
    from squid_control.control.config import CONFIG, ChannelMapper
    from squid_control.hypha_tools.artifact_manager.artifact_manager import (
        SquidArtifactManager,
    )
    from squid_control.hypha_tools.chatbot.aask import aask
    from squid_control.hypha_tools.hypha_storage import HyphaDataStore
    from squid_control.squid_controller import SquidController

import base64
import signal
import threading
from collections import deque
from typing import List, Optional

# WebRTC imports
import aiohttp
from aiortc import MediaStreamTrack
from av import VideoFrame
from hypha_rpc.utils.schema import schema_function
from pydantic import BaseModel, Field

dotenv.load_dotenv()
ENV_FILE = dotenv.find_dotenv()
if ENV_FILE:
    dotenv.load_dotenv(ENV_FILE)
import uuid  # noqa: E402

# Set up logging

def setup_logging(log_file="squid_control_service.log", max_bytes=100000, backup_count=3):
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)

    # Rotating file handler
    file_handler = logging.handlers.RotatingFileHandler(log_file, maxBytes=max_bytes, backupCount=backup_count)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    return logger

logger = setup_logging()

class VideoBuffer:
    """
    Video buffer to store and manage compressed microscope frames for smooth video streaming
    """
    def __init__(self, max_size=5):
        self.max_size = max_size
        self.buffer = deque(maxlen=max_size)
        self.lock = threading.Lock()
        self.last_frame_data = None  # Store compressed frame data
        self.last_metadata = None  # Store metadata for last frame
        self.frame_timestamp = 0

    def put_frame(self, frame_data, metadata=None):
        """Add a compressed frame and its metadata to the buffer

        Args:
            frame_data: dict with compressed frame info from _encode_frame_jpeg()
            metadata: dict with frame metadata including stage position and timestamp
        """
        with self.lock:
            self.buffer.append({
                'frame_data': frame_data,
                'metadata': metadata,
                'timestamp': time.time()
            })
            self.last_frame_data = frame_data
            self.last_metadata = metadata
            self.frame_timestamp = time.time()
            
    def get_frame_data(self):
        """Get the most recent compressed frame data and metadata from buffer
        
        Returns:
            tuple: (frame_data, metadata) or (None, None) if no frame available
        """
        with self.lock:
            if self.buffer:
                buffer_entry = self.buffer[-1]
                return buffer_entry['frame_data'], buffer_entry.get('metadata')
            elif self.last_frame_data is not None:
                return self.last_frame_data, self.last_metadata
            else:
                return None, None
    
    def get_frame(self):
        """Get the most recent decompressed frame from buffer (for backward compatibility)"""
        frame_data, _ = self.get_frame_data()  # Ignore metadata for backward compatibility
        if frame_data is None:
            return None
            
        # Decode JPEG back to numpy array
        try:
            if frame_data['format'] == 'jpeg':
                # Decode JPEG data
                nparr = np.frombuffer(frame_data['data'], np.uint8)
                bgr_frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                if bgr_frame is not None:
                    # Convert BGR back to RGB
                    return cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)
            elif frame_data['format'] == 'raw':
                # Raw numpy data
                return np.frombuffer(frame_data['data'], dtype=np.uint8).reshape((-1, 750, 3))
        except Exception as e:
            logger.error(f"Error decoding frame: {e}")
        
        return None
                
    def get_frame_age(self):
        """Get the age of the most recent frame in seconds"""
        with self.lock:
            if self.frame_timestamp > 0:
                return time.time() - self.frame_timestamp
            else:
                return float('inf')
                
    def clear(self):
        """Clear the buffer"""
        with self.lock:
            self.buffer.clear()
            self.last_frame_data = None
            self.last_metadata = None
            self.frame_timestamp = 0

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
        import cv2
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
            processed_frame = self.microscope_instance._decode_frame_jpeg(frame_data)

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

class Microscope:
    def __init__(self, is_simulation, is_local):
        self.current_x = 0
        self.current_y = 0
        self.current_z = 0
        self.current_theta = 0
        self.current_illumination_channel = None
        self.current_intensity = None
        self.is_illumination_on = False
        self.chatbot_service_url = None
        self.is_simulation = is_simulation
        self.is_local = is_local
        self.squidController = SquidController(is_simulation=is_simulation)
        self.squidController.move_to_well('C',3)
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
        self.datastore = None
        self.server_url = "http://192.168.2.1:9527" if is_local else "https://hypha.aicell.io/"
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
        if user['is_anonymous']:
            return False
        if self.authorized_emails is None or user["email"] in self.authorized_emails:
            return True
        else:
            return False
    
    async def is_service_healthy(self, context=None):
        """Check if all services are healthy"""
        try:
            # Check authentication
            if context and not self.check_permission(context.get("user", {})):
                raise Exception("User not authorized to access this service")
            
            microscope_svc = await self.server.get_service(self.service_id)
            if microscope_svc is None:
                raise RuntimeError("Microscope service not found")
            
            result = await microscope_svc.ping()
            if result != "pong":
                raise RuntimeError(f"Microscope service returned unexpected response: {result}")
            
            datastore_id = f'data-store-{"simu" if self.is_simulation else "real"}-{self.service_id}'
            datastore_svc = await self.server.get_service(datastore_id)
            if datastore_svc is None:
                raise RuntimeError("Datastore service not found")
            
            # Shorten chatbot service ID to avoid OpenAI API limits
            short_service_id = self.service_id[:20] if len(self.service_id) > 20 else self.service_id
            chatbot_id = f"sq-cb-{'simu' if self.is_simulation else 'real'}-{short_service_id}"
            
            chatbot_server_url = "https://chat.bioimage.io"
            try:
                chatbot_token = os.environ.get("WORKSPACE_TOKEN_CHATBOT")
                if not chatbot_token:
                    logger.warning("Chatbot token not found, skipping chatbot health check")
                else:
                    chatbot_server = await connect_to_server({
                        "client_id": f"squid-chatbot-{self.service_id}-{uuid.uuid4()}",
                        "server_url": chatbot_server_url, 
                        "token": chatbot_token,
                        "ping_interval": None
                    })
                    chatbot_svc = await asyncio.wait_for(chatbot_server.get_service(chatbot_id), 10)
                    if chatbot_svc is None:
                        raise RuntimeError("Chatbot service not found")
            except Exception as chatbot_error:
                raise RuntimeError(f"Chatbot service health check failed: {str(chatbot_error)}")
            


            logger.info("All services are healthy")
            return {"status": "ok", "message": "All services are healthy"}
        except Exception as e:
            logger.error(f"Health check failed: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
            raise RuntimeError(f"Service health check failed: {str(e)}")
    
    @schema_function(skip_self=True)
    def ping(self, context=None):
        """Ping the service"""
        return "pong"
    
    @schema_function(skip_self=True)
    def move_by_distance(self, x: float=Field(1.0, description="disntance through X axis, unit: milimeter"), y: float=Field(1.0, description="disntance through Y axis, unit: milimeter"), z: float=Field(1.0, description="disntance through Z axis, unit: milimeter"), context=None):
        """
        Move the stage by a distances in x, y, z axis
        Returns: Result information
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
    def move_to_position(self, x:float=Field(1.0,description="Unit: milimeter"), y:float=Field(1.0,description="Unit: milimeter"), z:float=Field(1.0,description="Unit: milimeter"), context=None):
        """
        Move the stage to a position in x, y, z axis
        Returns: The result of the movement
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
        Get the current status of the microscope
        Returns: Status of the microscope
        """
        try:
            # Check authentication
            if context and not self.check_permission(context.get("user", {})):
                raise Exception("User not authorized to access this service")
            
            current_x, current_y, current_z, current_theta = self.squidController.navigationController.update_pos(microcontroller=self.squidController.microcontroller)
            is_illumination_on = self.squidController.liveController.illumination_on
            scan_channel = self.squidController.multipointController.selected_configurations
            is_busy = self.squidController.is_busy
            # Get current well location information
            well_info = self.squidController.get_well_from_position('96')  # Default to 96-well plate
            
            self.parameters = {
                'is_busy': is_busy,
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
            }
            return self.parameters
        except Exception as e:
            logger.error(f"Failed to get status: {e}")
            raise e

    @schema_function(skip_self=True)
    def update_parameters_from_client(self, new_parameters: dict=Field(description="the dictionary parameters user want to update"), context=None):
        """
        Update the parameters from the client side
        Returns: Updated parameters in the microscope
        """
        try:
            # Check authentication
            if context and not self.check_permission(context.get("user", {})):
                raise Exception("User not authorized to access this service")
            
            if self.parameters is None:
                self.parameters = {}

            # Update only the specified keys
            for key, value in new_parameters.items():
                if key in self.parameters:
                    self.parameters[key] = value
                    logger.info(f"Updated {key} to {value}")

                    # Update the corresponding instance variable if it exists
                    if hasattr(self, key):
                        setattr(self, key, value)
                    else:
                        logger.error(f"Attribute {key} does not exist on self, skipping update.")
                else:
                    logger.error(f"Key {key} not found in parameters, skipping update.")

            return {"success": True, "message": "Parameters updated successfully.", "updated_parameters": new_parameters}
        except Exception as e:
            logger.error(f"Failed to update parameters: {e}")
            raise e

    @schema_function(skip_self=True)
    def set_simulated_sample_data_alias(self, sample_data_alias: str=Field("agent-lens/20250506-scan-time-lapse-2025-05-06_17-56-38", description="The alias of the sample data"), context=None):
        """
        Set the alias of simulated sample
        """
        self.squidController.set_simulated_sample_data_alias(sample_data_alias)
        return f"The alias of simulated sample is set to {sample_data_alias}"
    
    @schema_function(skip_self=True)
    def get_simulated_sample_data_alias(self, context=None):
        """
        Get the alias of simulated sample
        """
        return self.squidController.get_simulated_sample_data_alias()
    
    @schema_function(skip_self=True)
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
            
            # Get the raw image from the camera with original bit depth preserved and full frame
            raw_img = await self.squidController.snap_image(channel, intensity, exposure_time, full_frame=True)
            
            # In simulation mode, resize small images to expected camera resolution
            if self.squidController.is_simulation:
                height, width = raw_img.shape[:2]
                # If image is too small, resize it to expected camera dimensions
                expected_width = 3000  # Expected camera width
                expected_height = 3000  # Expected camera height
                if width < expected_width or height < expected_height:
                    raw_img = cv2.resize(raw_img, (expected_width, expected_height), interpolation=cv2.INTER_LINEAR)
            
            # Crop the image before resizing, similar to squid_controller.py approach
            crop_height = CONFIG.Acquisition.CROP_HEIGHT
            crop_width = CONFIG.Acquisition.CROP_WIDTH
            height, width = raw_img.shape[:2]  # Support both grayscale and color images
            start_x = width // 2 - crop_width // 2
            start_y = height // 2 - crop_height // 2
            
            # Ensure crop coordinates are within bounds
            start_x = max(0, start_x)
            start_y = max(0, start_y)
            end_x = min(width, start_x + crop_width)
            end_y = min(height, start_y + crop_height)
            
            cropped_img = raw_img[start_y:end_y, start_x:end_x]
            
            self.get_status()
            
            # Return the numpy array directly with preserved bit depth
            return cropped_img
            
        except Exception as e:
            logger.error(f"Failed to get new frame: {e}")
            raise e

    @schema_function(skip_self=True)
    async def get_video_frame(self, frame_width: int=Field(750, description="Width of the video frame"), frame_height: int=Field(750, description="Height of the video frame"), context=None):
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
                placeholder = self._create_placeholder_frame(frame_width, frame_height, "Scanning in Progress...")
                placeholder_compressed = self._encode_frame_jpeg(placeholder, quality=85)
                
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
                    decompressed_frame = self._decode_frame_jpeg(frame_data)
                    if decompressed_frame is not None:
                        # Resize the frame to requested dimensions
                        resized_frame = cv2.resize(decompressed_frame, (frame_width, frame_height), interpolation=cv2.INTER_AREA)
                        # Recompress at requested size
                        resized_compressed = self._encode_frame_jpeg(resized_frame, quality=85)
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
                        placeholder = self._create_placeholder_frame(frame_width, frame_height, "Frame decompression failed")
                        placeholder_compressed = self._encode_frame_jpeg(placeholder, quality=85)
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
                placeholder = self._create_placeholder_frame(frame_width, frame_height, "No buffered frame available")
                placeholder_compressed = self._encode_frame_jpeg(placeholder, quality=85)
                
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

    @schema_function(skip_self=True)
    def configure_video_buffer(self, buffer_fps: int = Field(5, description="Target FPS for buffer acquisition"), buffer_size: int = Field(5, description="Maximum number of frames to keep in buffer"), context=None):
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

    @schema_function(skip_self=True)
    def get_video_buffer_status(self, context=None):
        """Get the current status of the video buffer."""
        try:
            buffer_fill = len(self.video_buffer.frame_buffer)
            buffer_capacity = self.video_buffer.max_size
            
            return {
                "success": True,
                "buffer_running": self.frame_acquisition_running,
                "buffer_fill": buffer_fill,
                "buffer_capacity": buffer_capacity,
                "buffer_fill_percent": (buffer_fill / buffer_capacity * 100) if buffer_capacity > 0 else 0,
                "buffer_fps": self.buffer_fps,
                "frame_dimensions": {
                    "width": self.buffer_frame_width,
                    "height": self.buffer_frame_height
                },
                "video_idle_timeout": self.video_idle_timeout,
                "last_video_request": self.last_video_request_time,
                "webrtc_connected": self.webrtc_connected
            }
        except Exception as e:
            logger.error(f"Failed to get video buffer status: {e}")
            raise e

    @schema_function(skip_self=True)
    async def start_video_buffering(self, context=None):
        """Manually start video buffering for smooth streaming."""
        try:
            if self.buffer_acquisition_running:
                return {
                    "success": True,
                    "message": "Video buffering is already running",
                    "was_already_running": True
                }
            
            await self.start_frame_buffer_acquisition()
            logger.info("Video buffering started manually")
            
            return {
                "success": True,
                "message": "Video buffering started successfully",
                "buffer_fps": self.buffer_fps_target,
                "buffer_size": self.frame_buffer.maxlen
            }
        except Exception as e:
            logger.error(f"Failed to start video buffering: {e}")
            raise e

    @schema_function(skip_self=True)
    async def stop_video_buffering(self, context=None):
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
        
    @schema_function(skip_self=True)
    def configure_video_idle_timeout(self, idle_timeout: float = Field(5.0, description="Idle timeout in seconds (0 to disable automatic stop)"), context=None):
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

    @schema_function(skip_self=True)
    async def set_video_fps(self, fps: int = Field(5, description="Target frames per second for video acquisition (1-30 FPS)"), context=None):
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
            
            # Close camera resources properly
            if hasattr(self, 'squidController') and self.squidController:
                if hasattr(self.squidController, 'camera') and self.squidController.camera:
                    camera = self.squidController.camera
                    if hasattr(camera, 'cleanup_zarr_resources_async'):
                        try:
                            await asyncio.wait_for(camera.cleanup_zarr_resources_async(), timeout=5.0)
                            logger.info("ZarrImageManager resources cleaned up")
                        except asyncio.TimeoutError:
                            logger.warning("ZarrImageManager cleanup timed out")
                        except Exception as e:
                            logger.warning(f"ZarrImageManager cleanup error: {e}")
        except Exception as e:
            logger.error(f"Error during test cleanup: {e}")

    @schema_function(skip_self=True)
    async def start_video_buffering_api(self, context=None):
        """Start video buffering for smooth video streaming"""
        try:
            # Check authentication
            if context and not self.check_permission(context.get("user", {})):
                raise Exception("User not authorized to access this service")
            
            await self.start_video_buffering()
            return {"success": True, "message": "Video buffering started successfully"}
        except Exception as e:
            logger.error(f"Failed to start video buffering: {e}")
            raise e

    @schema_function(skip_self=True)
    async def stop_video_buffering_api(self, context=None):
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
            
            await self.stop_video_buffering()
            logger.info("Video buffering stopped manually")
            
            return {
                "success": True,
                "message": "Video buffering stopped successfully"
            }
        except Exception as e:
            logger.error(f"Failed to stop video buffering: {e}")
            raise e

    @schema_function(skip_self=True)
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

    @schema_function(skip_self=True)
    def adjust_video_frame(self, min_val: int = Field(0, description="Minimum intensity value for contrast stretching"), max_val: Optional[int] = Field(None, description="Maximum intensity value for contrast stretching"), context=None):
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
    async def snap(self, exposure_time: int=Field(100, description="Exposure time, in milliseconds"), channel: int=Field(0, description="Light source (0 for Bright Field, Fluorescence channels: 11 for 405 nm, 12 for 488 nm, 13 for 638nm, 14 for 561 nm, 15 for 730 nm)"), intensity: int=Field(50, description="Intensity of the illumination source"), context=None):
        """
        Get an image from microscope
        Returns: the URL of the image
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
            gray_img = await self.squidController.snap_image(channel, intensity, exposure_time)
            logger.info('The image is snapped')
            gray_img = gray_img.astype(np.uint8)
            # Resize the image to a standard size
            resized_img = cv2.resize(gray_img, (2048, 2048))

            # Encode the image directly to PNG without converting to BGR
            _, png_image = cv2.imencode('.png', resized_img)

            # Store the PNG image
            file_id = self.datastore.put('file', png_image.tobytes(), 'snapshot.png', "Captured microscope image in PNG format")
            data_url = self.datastore.get_url(file_id)
            logger.info(f'The image is snapped and saved as {data_url}')
            
            #update the current illumination channel and intensity
            self.squidController.current_channel = channel
            param_name = self.channel_param_map.get(channel)
            if param_name:
                setattr(self, param_name, [intensity, exposure_time])
            else:
                logger.warning(f"Unknown channel {channel} in snap, parameters not updated for intensity/exposure attributes.")
            
            self.get_status()
            return data_url
        except Exception as e:
            logger.error(f"Failed to snap image: {e}")
            raise e

    @schema_function(skip_self=True)
    def open_illumination(self, context=None):
        """
        Turn on the illumination
        Returns: The message of the action
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
    def close_illumination(self, context=None):
        """
        Turn off the illumination
        Returns: The message of the action
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

    @schema_function(skip_self=True)
    async def scan_well_plate(self, well_plate_type: str=Field("96", description="Type of the well plate (e.g., '6', '12', '24', '96', '384')"), illumination_settings: List[dict]=Field(default_factory=lambda: [{'channel': 'BF LED matrix full', 'intensity': 28.0, 'exposure_time': 20.0}, {'channel': 'Fluorescence 488 nm Ex', 'intensity': 27.0, 'exposure_time': 60.0}, {'channel': 'Fluorescence 561 nm Ex', 'intensity': 98.0, 'exposure_time': 100.0}], description="Illumination settings with channel name, intensity (0-100), and exposure time (ms) for each channel"), do_contrast_autofocus: bool=Field(False, description="Whether to do contrast based autofocus"), do_reflection_af: bool=Field(True, description="Whether to do reflection based autofocus"), scanning_zone: List[tuple]=Field(default_factory=lambda: [(0,0),(0,0)], description="The scanning zone of the well plate, for 96 well plate, it should be[(0,0),(7,11)] "), Nx: int=Field(3, description="Number of columns to scan"), Ny: int=Field(3, description="Number of rows to scan"), action_ID: str=Field('testPlateScan', description="The ID of the action"), context=None):
        """
        Scan the well plate according to the pre-defined position list with custom illumination settings
        Returns: The message of the action
        """
        try:
            # Check authentication
            if context and not self.check_permission(context.get("user", {})):
                raise Exception("User not authorized to access this service")
            
            if illumination_settings is None:
                logger.warning("No illumination settings provided, using default settings")
                illumination_settings = [
                    {'channel': 'BF LED matrix full', 'intensity': 18, 'exposure_time': 10},
                    {'channel': 'Fluorescence 405 nm Ex', 'intensity': 45, 'exposure_time': 30},
                    {'channel': 'Fluorescence 488 nm Ex', 'intensity': 30, 'exposure_time': 100},
                    {'channel': 'Fluorescence 561 nm Ex', 'intensity': 100, 'exposure_time': 200},
                    {'channel': 'Fluorescence 638 nm Ex', 'intensity': 100, 'exposure_time': 200},
                    {'channel': 'Fluorescence 730 nm Ex', 'intensity': 100, 'exposure_time': 200},
                ]
            
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
                scanning_zone,
                Nx,
                Ny,
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
    def scan_well_plate_simulated(self, context=None):
        """
        Scan the well plate according to the pre-defined position list
        Returns: The message of the action
        """
        try:
            # Check authentication
            if context and not self.check_permission(context.get("user", {})):
                raise Exception("User not authorized to access this service")
            
            time.sleep(600)
            return "Well plate scanning completed"
        except Exception as e:
            logger.error(f"Failed to scan well plate: {e}")
            raise e


    @schema_function(skip_self=True)
    def set_illumination(self, channel: int=Field(0, description="Light source (e.g., 0 for Bright Field, Fluorescence channels: 11 for 405 nm, 12 for 488 nm, 13 for 638nm, 14 for 561 nm, 15 for 730 nm)"), intensity: int=Field(50, description="Intensity of the illumination source"), context=None):
        """
        Set the intensity of light source
        Returns:A string message
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
    def set_camera_exposure(self,channel: int=Field(..., description="Light source (e.g., 0 for Bright Field, Fluorescence channels: 11 for 405 nm, 12 for 488 nm, 13 for 638nm, 14 for 561 nm, 15 for 730 nm)"), exposure_time: int=Field(..., description="Exposure time in milliseconds"), context=None):
        """
        Set the exposure time of the camera
        Returns: A string message
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
        Stop the scanning of the well plate.
        Returns: A string message
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
        Move the stage to home/zero position
        Returns: A string message
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
        Move the stage to the initial position for imaging.
        Returns: A string message
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
        Move the stage to the loading position.
        Returns: A  string message
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
    async def auto_focus(self, context=None):
        """
        Do contrast-based autofocus
        Returns: A string message
        """

        try:
            # Check authentication
            if context and not self.check_permission(context.get("user", {})):
                raise Exception("User not authorized to access this service")
            
            await self.squidController.do_autofocus()
            logger.info('The camera is auto-focused')
            return 'The camera is auto-focused'
        except Exception as e:
            logger.error(f"Failed to auto focus: {e}")
            raise e
    
    @schema_function(skip_self=True)
    async def do_laser_autofocus(self, context=None):
        """
        Do reflection-based autofocus
        Returns: A string message
        """

        try:
            # Check authentication
            if context and not self.check_permission(context.get("user", {})):
                raise Exception("User not authorized to access this service")
            
            await self.squidController.do_laser_autofocus()
            logger.info('The camera is auto-focused')
            return 'The camera is auto-focused'
        except Exception as e:
            logger.error(f"Failed to do laser autofocus: {e}")
            raise e
        
    @schema_function(skip_self=True)
    async def set_laser_reference(self, context=None):
        """
        Set the reference of the laser
        Returns: A string message
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
    async def navigate_to_well(self, row: str=Field('A', description="Row number of the well position (e.g., 'A')"), col: int=Field(1, description="Column number of the well position"), wellplate_type: str=Field('96', description="Type of the well plate (e.g., '6', '12', '24', '96', '384')"), context=None):
        """
        Navigate to the specified well position in the well plate.
        Returns: A string message
        """

        try:
            # Check authentication
            if context and not self.check_permission(context.get("user", {})):
                raise Exception("User not authorized to access this service")
            
            if wellplate_type is None:
                wellplate_type = '96'
            # Run the blocking move_to_well operation in a separate thread executor
            # This prevents the asyncio event loop from being blocked during stage movement
            await asyncio.get_event_loop().run_in_executor(
                None,  # Use default ThreadPoolExecutor
                self.squidController.move_to_well,
                row,
                col,
                wellplate_type
            )
            logger.info(f'The stage moved to well position ({row},{col})')
            return f'The stage moved to well position ({row},{col})'
        except Exception as e:
            logger.error(f"Failed to navigate to well: {e}")
            raise e

    @schema_function(skip_self=True)
    def get_chatbot_url(self, context=None):
        """
        Get the URL of the chatbot service.
        Returns: A URL string
        """

        try:
            # Check authentication
            if context and not self.check_permission(context.get("user", {})):
                raise Exception("User not authorized to access this service")
            
            logger.info(f"chatbot_service_url: {self.chatbot_service_url}")
            return self.chatbot_service_url
        except Exception as e:
            logger.error(f"Failed to get chatbot URL: {e}")
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
    
    class MoveByDistanceInput(BaseModel):
        """Move the stage by a distance in x, y, z axis."""
        x: float = Field(0, description="Move the stage along X axis")
        y: float = Field(0, description="Move the stage along Y axis")
        z: float = Field(0, description="Move the stage along Z axis")

    class MoveToPositionInput(BaseModel):
        """Move the stage to a position in x, y, z axis."""
        x: Optional[float] = Field(None, description="Move the stage to the X coordinate")
        y: Optional[float] = Field(None, description="Move the stage to the Y coordinate")
        z: float = Field(3.35, description="Move the stage to the Z coordinate")

    class SetSimulatedSampleDataAliasInput(BaseModel):
        """Set the alias of simulated sample"""
        sample_data_alias: str = Field("agent-lens/20250506-scan-time-lapse-2025-05-06_17-56-38", description="The alias of the sample data")

    class AutoFocusInput(BaseModel):
        """Reflection based autofocus."""
        N: int = Field(10, description="Number of discrete focus positions")
        delta_Z: float = Field(1.524, description="Step size in the Z-axis in micrometers")

    class SnapImageInput(BaseModel):
        """Snap an image from the camera, and display it in the chatbot."""
        exposure: int = Field(..., description="Exposure time in milliseconds")
        channel: int = Field(..., description="Light source (e.g., 0 for Bright Field, Fluorescence channels: 11 for 405 nm, 12 for 488 nm, 13 for 638nm, 14 for 561 nm, 15 for 730 nm)")
        intensity: int = Field(..., description="Intensity of the illumination source")

    class InspectToolInput(BaseModel):
        """Inspect the images with GPT4-o's vision model."""
        images: List[dict] = Field(..., description="A list of images to be inspected, each with a 'http_url' and 'title'")
        query: str = Field(..., description="User query about the image")
        context_description: str = Field(..., description="Context for the visual inspection task, inspect images taken from the microscope")

    class NavigateToWellInput(BaseModel):
        """Navigate to a well position in the well plate."""
        row: str = Field(..., description="Row number of the well position (e.g., 'A')")
        col: int = Field(..., description="Column number of the well position")
        wellplate_type: str = Field('96', description="Type of the well plate (e.g., '6', '12', '24', '96', '384')")

    class MoveToLoadingPositionInput(BaseModel):
        """Move the stage to the loading position."""

    class SetIlluminationInput(BaseModel):
        """Set the intensity of light source."""
        channel: int = Field(..., description="Light source (e.g., 0 for Bright Field, Fluorescence channels: 11 for 405 nm, 12 for 488 nm, 13 for 638nm, 14 for 561 nm, 15 for 730 nm)")
        intensity: int = Field(..., description="Intensity of the illumination source")

    class SetCameraExposureInput(BaseModel):
        """Set the exposure time of the camera."""
        channel: int = Field(..., description="Light source (e.g., 0 for Bright Field, Fluorescence channels: 11 for 405 nm, 12 for 488 nm, 13 for 638nm, 14 for 561 nm, 15 for 730 nm)")
        exposure_time: int = Field(..., description="Exposure time in milliseconds")

    class DoLaserAutofocusInput(BaseModel):
        """Do reflection-based autofocus."""

    class SetLaserReferenceInput(BaseModel):
        """Set the reference of the laser."""

    class GetStatusInput(BaseModel):
        """Get the current status of the microscope."""

    class HomeStageInput(BaseModel):
        """Home the stage in z, y, and x axis."""

    class ReturnStageInput(BaseModel):
        """Return the stage to the initial position."""

    class ImageInfo(BaseModel):
        """Image information."""
        url: str = Field(..., description="The URL of the image.")
        title: Optional[str] = Field(None, description="The title of the image.")

    class GetCurrentWellLocationInput(BaseModel):
        """Get the current well location based on the stage position."""
        wellplate_type: str = Field('96', description="Type of the well plate (e.g., '6', '12', '24', '96', '384')")

    class GetMicroscopeConfigurationInput(BaseModel):
        """Get microscope configuration information in JSON format."""
        config_section: str = Field('all', description="Configuration section to retrieve ('all', 'camera', 'stage', 'illumination', 'acquisition', 'limits', 'hardware', 'wellplate', 'optics', 'autofocus')")
        include_defaults: bool = Field(True, description="Whether to include default values from config.py")

    class SetStageVelocityInput(BaseModel):
        """Set the maximum velocity for X and Y stage axes."""
        velocity_x_mm_per_s: Optional[float] = Field(None, description="Maximum velocity for X axis in mm/s (default: uses configuration value)")
        velocity_y_mm_per_s: Optional[float] = Field(None, description="Maximum velocity for Y axis in mm/s (default: uses configuration value)")

    async def inspect_tool(self, images: List[dict], query: str, context_description: str) -> str:
        image_infos = [
            self.ImageInfo(url=image_dict['http_url'], title=image_dict.get('title'))
            for image_dict in images
        ]
        for image_info_obj in image_infos:
            assert image_info_obj.url.startswith("http"), "Image URL must start with http."
        response = await aask(image_infos, [context_description, query])
        return response

    def move_by_distance_schema(self, config: MoveByDistanceInput, context=None):
        self.get_status()
        x_pos = self.parameters['current_x']
        y_pos = self.parameters['current_y']
        z_pos = self.parameters['current_z']
        result = self.move_by_distance(config.x, config.y, config.z, context)
        return result['message']

    def move_to_position_schema(self, config: MoveToPositionInput, context=None):
        self.get_status()
        x_pos = self.parameters['current_x']
        y_pos = self.parameters['current_y']
        z_pos = self.parameters['current_z']
        x = config.x if config.x is not None else 0
        y = config.y if config.y is not None else 0
        z = config.z if config.z is not None else 0
        result = self.move_to_position(x, y, z, context)
        return result['message']
    
    async def auto_focus_schema(self, config: AutoFocusInput, context=None):
        await self.auto_focus(context)
        return "Auto-focus completed."

    async def snap_image_schema(self, config: SnapImageInput, context=None):
        image_url = await self.snap(config.exposure, config.channel, config.intensity, context)
        return f"![Image]({image_url})"

    async def navigate_to_well_schema(self, config: NavigateToWellInput, context=None):
        await self.navigate_to_well(config.row, config.col, config.wellplate_type, context)
        return f'The stage moved to well position ({config.row},{config.col})'

    async def inspect_tool_schema(self, config: InspectToolInput, context=None):
        response = await self.inspect_tool(config.images, config.query, config.context_description)
        return {"result": response}

    async def home_stage_schema(self, context=None):
        response = await self.home_stage(context)
        return {"result": response}

    async def return_stage_schema(self, context=None):
        response = await self.return_stage(context)
        return {"result": response}

    def set_illumination_schema(self, config: SetIlluminationInput, context=None):
        response = self.set_illumination(config.channel, config.intensity, context)
        return {"result": response}

    def set_camera_exposure_schema(self, config: SetCameraExposureInput, context=None):
        response = self.set_camera_exposure(config.channel, config.exposure_time, context)
        return {"result": response}

    async def do_laser_autofocus_schema(self, context=None):
        response = await self.do_laser_autofocus(context)
        return {"result": response}

    async def set_laser_reference_schema(self, context=None):
        response = await self.set_laser_reference(context)
        return {"result": response}

    def get_status_schema(self, context=None):
        response = self.get_status(context)
        return {"result": response}

    def get_current_well_location_schema(self, config: GetCurrentWellLocationInput, context=None):
        response = self.get_current_well_location(config.wellplate_type, context)
        return {"result": response}

    def get_microscope_configuration_schema(self, config: GetMicroscopeConfigurationInput, context=None):
        response = self.get_microscope_configuration(config.config_section, config.include_defaults, context)
        return {"result": response}

    def get_schema(self, context=None):
        return {
            "move_by_distance": self.MoveByDistanceInput.model_json_schema(),
            "move_to_position": self.MoveToPositionInput.model_json_schema(),
            "home_stage": self.HomeStageInput.model_json_schema(),
            "return_stage": self.ReturnStageInput.model_json_schema(),
            "auto_focus": self.AutoFocusInput.model_json_schema(),
            "snap_image": self.SnapImageInput.model_json_schema(),
            "inspect_tool": self.InspectToolInput.model_json_schema(),
            "load_position": self.MoveToLoadingPositionInput.model_json_schema(),
            "navigate_to_well": self.NavigateToWellInput.model_json_schema(),
            "set_illumination": self.SetIlluminationInput.model_json_schema(),
            "set_camera_exposure": self.SetCameraExposureInput.model_json_schema(),
            "do_laser_autofocus": self.DoLaserAutofocusInput.model_json_schema(),
            "set_laser_reference": self.SetLaserReferenceInput.model_json_schema(),
            "get_status": self.GetStatusInput.model_json_schema(),
            "get_current_well_location": self.GetCurrentWellLocationInput.model_json_schema(),
            "get_microscope_configuration": self.GetMicroscopeConfigurationInput.model_json_schema(),
            "set_stage_velocity": self.SetStageVelocityInput.model_json_schema(),
        }

    async def start_hypha_service(self, server, service_id, run_in_executor=None):
        self.server = server
        self.service_id = service_id
        
        # Default to True for production, False for tests (identified by "test" in service_id)
        if run_in_executor is None:
            run_in_executor = "test" not in service_id.lower()
        
        # Build the service configuration
        service_config = {
            "name": "Microscope Control Service",
            "id": service_id,
            "config": {
                "visibility": "protected",
                "require_context": True,  # Enable user context for authentication
                "run_in_executor": run_in_executor
            },
            "type": "echo",
            "ping": self.ping,
            "is_service_healthy": self.is_service_healthy,
            "move_by_distance": self.move_by_distance,
            "snap": self.snap,
            "one_new_frame": self.one_new_frame,
            "get_video_frame": self.get_video_frame,
            "off_illumination": self.close_illumination,
            "on_illumination": self.open_illumination,
            "set_illumination": self.set_illumination,
            "set_camera_exposure": self.set_camera_exposure,
            "scan_well_plate": self.scan_well_plate,
            "scan_well_plate_simulated": self.scan_well_plate_simulated,
            "stop_scan": self.stop_scan,
            "home_stage": self.home_stage,
            "return_stage": self.return_stage,
            "navigate_to_well": self.navigate_to_well,
            "move_to_position": self.move_to_position,
            "move_to_loading_position": self.move_to_loading_position,
            "set_simulated_sample_data_alias": self.set_simulated_sample_data_alias,
            "get_simulated_sample_data_alias": self.get_simulated_sample_data_alias,
            "auto_focus": self.auto_focus,
            "do_laser_autofocus": self.do_laser_autofocus,
            "set_laser_reference": self.set_laser_reference,
            "get_status": self.get_status,
            "update_parameters_from_client": self.update_parameters_from_client,
            "get_chatbot_url": self.get_chatbot_url,
            "adjust_video_frame": self.adjust_video_frame,
            "start_video_buffering": self.start_video_buffering_api,
            "stop_video_buffering": self.stop_video_buffering_api,
            "get_video_buffering_status": self.get_video_buffering_status,
            "set_video_fps": self.set_video_fps,
            "get_current_well_location": self.get_current_well_location,
            "get_microscope_configuration": self.get_microscope_configuration,
            "set_stage_velocity": self.set_stage_velocity,
            # Stitching functions
            "normal_scan_with_stitching": self.normal_scan_with_stitching,
            "quick_scan_with_stitching": self.quick_scan_with_stitching,
            "stop_scan_and_stitching": self.stop_scan_and_stitching,
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
            "list_microscope_galleries": self.list_microscope_galleries,
            "list_gallery_datasets": self.list_gallery_datasets,
        }
        
        # Only register get_canvas_chunk when not in local mode
        if not self.is_local:
            service_config["get_canvas_chunk"] = self.get_canvas_chunk
            logger.info("Registered get_canvas_chunk service (remote mode)")
        else:
            logger.info("Skipped get_canvas_chunk service registration (local mode)")

        svc = await server.register_service(service_config)

        logger.info(
            f"Service (service_id={service_id}) started successfully, available at {self.server_url}{server.config.workspace}/services"
        )

        logger.info(f'You can use this service using the service id: {svc.id}')
        id = svc.id.split(":")[1]

        logger.info(f"You can also test the service via the HTTP proxy: {self.server_url}{server.config.workspace}/services/{id}")

    async def start_chatbot_service(self, server, service_id):
        chatbot_extension = {
            "_rintf": True,
            "id": service_id,
            "type": "bioimageio-chatbot-extension",
            "name": "Squid Microscope Control",
            "description": "You are an AI agent controlling microscope. Automate tasks, adjust imaging parameters, and make decisions based on live visual feedback. Solve all the problems from visual feedback; the user only wants to see good results.",
            "config": {"visibility": "public", "require_context": True},
            "get_schema": self.get_schema,
            "tools": {
                "move_by_distance": self.move_by_distance_schema,
                "move_to_position": self.move_to_position_schema,
                "auto_focus": self.auto_focus_schema,
                "snap_image": self.snap_image_schema,
                "home_stage": self.home_stage_schema,
                "return_stage": self.return_stage_schema,
                "load_position": self.move_to_loading_position,
                "navigate_to_well": self.navigate_to_well_schema,
                "inspect_tool": self.inspect_tool_schema,
                "set_illumination": self.set_illumination_schema,
                "set_camera_exposure": self.set_camera_exposure_schema,
                "do_laser_autofocus": self.do_laser_autofocus_schema,
                "set_laser_reference": self.set_laser_reference_schema,
                "get_status": self.get_status_schema,
                "get_current_well_location": self.get_current_well_location_schema,
                "get_microscope_configuration": self.get_microscope_configuration_schema,
                "set_stage_velocity": self.set_stage_velocity_schema,
            }
        }

        svc = await server.register_service(chatbot_extension)
        self.chatbot_service_url = f"https://bioimage.io/chat?server=https://chat.bioimage.io&extension={svc.id}&assistant=Skyler"
        logger.info(f"Extension service registered with id: {svc.id}, you can visit the service at:\n {self.chatbot_service_url}")

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
                {"client_id": f"squid-remote-server-{self.service_id}-{uuid.uuid4()}", "server_url": "https://hypha.aicell.io", "token": remote_token, "workspace": remote_workspace, "ping_interval": None}
            )
        if not self.service_id:
            raise ValueError("MICROSCOPE_SERVICE_ID is not set in the environment variables.")
        if self.is_local:
            token = os.environ.get("REEF_LOCAL_TOKEN")
            workspace = os.environ.get("REEF_LOCAL_WORKSPACE")
            server = await connect_to_server(
                {"client_id": f"squid-local-server-{self.service_id}-{uuid.uuid4()}", "server_url": self.server_url, "token": token, "workspace": workspace, "ping_interval": None}
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
                {"client_id": f"squid-control-server-{self.service_id}-{uuid.uuid4()}", "server_url": self.server_url, "token": token, "workspace": workspace,  "ping_interval": None}
            )
        
        self.server = server
        
        # Setup zarr artifact manager for dataset upload functionality
        try:
            from .hypha_tools.artifact_manager.artifact_manager import SquidArtifactManager
            self.zarr_artifact_manager = SquidArtifactManager()
            
            # Connect to agent-lens workspace for zarr uploads
            zarr_token = os.environ.get("AGENT_LENS_WORKSPACE_TOKEN")
            if zarr_token:
                zarr_server = await connect_to_server({
                    "server_url": "https://hypha.aicell.io",
                    "token": zarr_token,
                    "workspace": "agent-lens",
                    "ping_interval": None
                })
                await self.zarr_artifact_manager.connect_server(zarr_server)
                logger.info("Zarr artifact manager initialized successfully")
                
                # Pass the zarr artifact manager to the squid controller
                self.squidController.zarr_artifact_manager = self.zarr_artifact_manager
                logger.info("Zarr artifact manager passed to squid controller")
            else:
                logger.warning("AGENT_LENS_WORKSPACE_TOKEN not found, zarr upload functionality disabled")
                self.zarr_artifact_manager = None
        except Exception as e:
            logger.warning(f"Failed to initialize zarr artifact manager: {e}")
            self.zarr_artifact_manager = None
        
        if self.is_simulation:
            await self.start_hypha_service(self.server, service_id=self.service_id)
            datastore_id = f'data-store-simu-{self.service_id}'
            # Shorten chatbot service ID to avoid OpenAI API limits
            short_service_id = self.service_id[:20] if len(self.service_id) > 20 else self.service_id
            chatbot_id = f"sq-cb-simu-{short_service_id}"
        else:
            await self.start_hypha_service(self.server, service_id=self.service_id)
            datastore_id = f'data-store-real-{self.service_id}'
            # Shorten chatbot service ID to avoid OpenAI API limits
            short_service_id = self.service_id[:20] if len(self.service_id) > 20 else self.service_id
            chatbot_id = f"sq-cb-real-{short_service_id}"
        
        self.datastore = HyphaDataStore()
        try:
            await self.datastore.setup(remote_server, service_id=datastore_id)
        except TypeError as e:
            if "Future" in str(e):
                config = await asyncio.wrap_future(server.config)
                await self.datastore.setup(remote_server, service_id=datastore_id, config=config)
            else:
                raise e
    
        chatbot_server_url = "https://chat.bioimage.io"
        try:
            chatbot_token= os.environ.get("WORKSPACE_TOKEN_CHATBOT")
        except:
            chatbot_token = await login({"server_url": chatbot_server_url})
        chatbot_server = await connect_to_server({"client_id": f"squid-chatbot-{self.service_id}-{uuid.uuid4()}", "server_url": chatbot_server_url, "token": chatbot_token,  "ping_interval": None})
        await self.start_chatbot_service(chatbot_server, chatbot_id)
        webrtc_id = f"video-track-{self.service_id}"
        if not self.is_local: # only start webrtc service in remote mode
            await self.start_webrtc_service(self.server, webrtc_id)


    async def initialize_zarr_manager(self, camera):
        from .hypha_tools.artifact_manager.artifact_manager import ZarrImageManager
        
        camera.zarr_image_manager = ZarrImageManager()
        
        init_success = await camera.zarr_image_manager.connect(
            server_url=self.server_url
        )
        
        if not init_success:
            raise RuntimeError("Failed to initialize ZarrImageManager")
        
        if hasattr(camera, 'scale_level'):
            camera.zarr_image_manager.scale_key = f'scale{camera.scale_level}'
        
        logger.info("ZarrImageManager initialized successfully for health check")
        return camera.zarr_image_manager

    async def start_video_buffering(self):
        """Start the background frame acquisition task for video buffering"""
        if self.frame_acquisition_running:
            logger.info("Video buffering already running")
            return
            
        self.frame_acquisition_running = True
        self.buffering_start_time = time.time()
        self.frame_acquisition_task = asyncio.create_task(self._background_frame_acquisition())
        logger.info("Video buffering started")
        
    async def stop_video_buffering(self):
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
                        placeholder_frame = self._create_placeholder_frame(
                            self.buffer_frame_width, self.buffer_frame_height, "Camera Overloaded"
                        )
                        compressed_placeholder = self._encode_frame_jpeg(placeholder_frame, quality=85)
                        
                        # Calculate gray level statistics for placeholder frame
                        placeholder_gray_stats = self._calculate_gray_level_statistics(placeholder_frame)
                        
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
                        
                        processed_frame, gray_level_stats = self._process_raw_frame(
                            raw_frame, frame_width=self.buffer_frame_width, frame_height=self.buffer_frame_height
                        )
                        
                        # LATENCY MEASUREMENT: End timing image processing
                        T_process_complete = time.time()
                        
                        # LATENCY MEASUREMENT: Start timing JPEG compression
                        T_compress_start = time.time()
                        
                        # Compress frame for efficient storage and transmission
                        compressed_frame = self._encode_frame_jpeg(processed_frame, quality=85)
                        
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
                    placeholder_frame = self._create_placeholder_frame(
                        self.buffer_frame_width, self.buffer_frame_height, f"Acquisition Error: {str(e)}"
                    )
                    compressed_placeholder = self._encode_frame_jpeg(placeholder_frame, quality=85)
                    
                    # Calculate gray level statistics for placeholder frame
                    placeholder_gray_stats = self._calculate_gray_level_statistics(placeholder_frame)
                    
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
        
    def _process_raw_frame(self, raw_frame, frame_width=750, frame_height=750):
        """Process raw frame for video streaming - OPTIMIZED"""
        try:
            # OPTIMIZATION 1: Crop FIRST, then resize to reduce data for all subsequent operations
            crop_height = CONFIG.Acquisition.CROP_HEIGHT
            crop_width = CONFIG.Acquisition.CROP_WIDTH
            height, width = raw_frame.shape[:2]  # Support both grayscale and color images
            start_x = width // 2 - crop_width // 2
            start_y = height // 2 - crop_height // 2
            
            # Ensure crop coordinates are within bounds
            start_x = max(0, start_x)
            start_y = max(0, start_y)
            end_x = min(width, start_x + crop_width)
            end_y = min(height, start_y + crop_height)
            
            cropped_frame = raw_frame[start_y:end_y, start_x:end_x]
            
            # Now resize the cropped frame to target dimensions
            if cropped_frame.shape[:2] != (frame_height, frame_width):
                # Use INTER_AREA for downsampling (faster than INTER_LINEAR)
                processed_frame = cv2.resize(cropped_frame, (frame_width, frame_height), interpolation=cv2.INTER_AREA)
            else:
                processed_frame = cropped_frame.copy()
            
            # Calculate gray level statistics on original frame BEFORE min/max adjustments
            gray_level_stats = self._calculate_gray_level_statistics(processed_frame)
            
            # OPTIMIZATION 2: Robust contrast adjustment (fixed)
            min_val = self.video_contrast_min
            max_val = self.video_contrast_max

            if max_val is None:
                if processed_frame.dtype == np.uint16:
                    max_val = 65535
                else:
                    max_val = 255
            
            # OPTIMIZATION 3: Improved contrast scaling with proper range handling
            if max_val > min_val:
                # Clip values to the specified range
                processed_frame = np.clip(processed_frame, min_val, max_val)
                
                # Scale to 0-255 range using float for precision, then convert to uint8
                if max_val > min_val:
                    # Use float32 for accurate scaling, then convert to uint8
                    processed_frame = ((processed_frame.astype(np.float32) - min_val) / (max_val - min_val) * 255).astype(np.uint8)
                else:
                    # Edge case: min_val == max_val
                    processed_frame = np.full_like(processed_frame, 127, dtype=np.uint8)
            else:
                # Edge case: max_val <= min_val, return mid-gray
                height, width = processed_frame.shape[:2]
                processed_frame = np.full((height, width), 127, dtype=np.uint8)
            
            # Ensure we have uint8 output
            if processed_frame.dtype != np.uint8:
                processed_frame = processed_frame.astype(np.uint8)
            
            # OPTIMIZATION 4: Fast color space conversion
            if len(processed_frame.shape) == 2:
                # Direct array manipulation is faster than cv2.cvtColor for grayscale->RGB
                processed_frame = np.stack([processed_frame] * 3, axis=2)
            elif processed_frame.shape[2] == 1:
                processed_frame = np.repeat(processed_frame, 3, axis=2)
            
            return processed_frame, gray_level_stats
            
        except Exception as e:
            logger.error(f"Error processing frame: {e}")
            placeholder_frame = self._create_placeholder_frame(frame_width, frame_height, f"Processing Error: {str(e)}")
            placeholder_stats = self._calculate_gray_level_statistics(placeholder_frame)
            return placeholder_frame, placeholder_stats
            
    def _create_placeholder_frame(self, width, height, message="No Frame Available"):
        """Create a placeholder frame with error message"""
        placeholder_img = np.zeros((height, width, 3), dtype=np.uint8)
        cv2.putText(placeholder_img, message, (10, height//2), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (128, 128, 128), 2)
        return placeholder_img
    
    def _decode_frame_jpeg(self, frame_data):
        """
        Decode compressed frame data back to numpy array
        
        Args:
            frame_data: dict from _encode_frame_jpeg() or get_video_frame()
        
        Returns:
            numpy array: RGB image data
        """
        try:
            if frame_data['format'] == 'jpeg':
                # Decode JPEG data
                nparr = np.frombuffer(frame_data['data'], np.uint8)
                bgr_frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                if bgr_frame is not None:
                    # Convert BGR back to RGB
                    return cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)
            elif frame_data['format'] == 'raw':
                # Raw numpy data
                height = frame_data.get('height', 750)
                width = frame_data.get('width', 750)
                return np.frombuffer(frame_data['data'], dtype=np.uint8).reshape((height, width, 3))
        except Exception as e:
            logger.error(f"Error decoding frame: {e}")
        
        # Return placeholder on error
        width = frame_data.get('width', self.buffer_frame_width)
        height = frame_data.get('height', self.buffer_frame_height)
        return self._create_placeholder_frame(width, height, "Decode Error")

    def _calculate_gray_level_statistics(self, rgb_frame):
        """Calculate comprehensive gray level statistics for microscope analysis"""
        try:
            import numpy as np
            
            # Convert RGB to grayscale for analysis (standard luminance formula)
            if len(rgb_frame.shape) == 3:
                # RGB to grayscale: Y = 0.299*R + 0.587*G + 0.114*B
                gray_frame = np.dot(rgb_frame[...,:3], [0.299, 0.587, 0.114])
            else:
                gray_frame = rgb_frame
            
            # Ensure we have a valid grayscale image
            if gray_frame.size == 0:
                return None
                
            # Convert to 0-100% range for analysis
            gray_normalized = (gray_frame / 255.0) * 100.0
            
            # Calculate comprehensive statistics
            stats = {
                'mean_percent': float(np.mean(gray_normalized)),
                'std_percent': float(np.std(gray_normalized)),
                'min_percent': float(np.min(gray_normalized)),
                'max_percent': float(np.max(gray_normalized)),
                'median_percent': float(np.median(gray_normalized)),
                'percentiles': {
                    'p5': float(np.percentile(gray_normalized, 5)),
                    'p25': float(np.percentile(gray_normalized, 25)),
                    'p75': float(np.percentile(gray_normalized, 75)),
                    'p95': float(np.percentile(gray_normalized, 95))
                },
                'histogram': {
                    'bins': 20,  # 20 bins for 0-100% range (5% per bin)
                    'counts': [],
                    'bin_edges': []
                }
            }
            
            # Calculate histogram (20 bins from 0-100%)
            hist_counts, bin_edges = np.histogram(gray_normalized, bins=20, range=(0, 100))
            stats['histogram']['counts'] = hist_counts.tolist()
            stats['histogram']['bin_edges'] = bin_edges.tolist()
            
            # Additional microscope-specific metrics
            stats['dynamic_range_percent'] = stats['max_percent'] - stats['min_percent']
            stats['contrast_ratio'] = stats['std_percent'] / stats['mean_percent'] if stats['mean_percent'] > 0 else 0
            
            # Exposure quality indicators
            stats['exposure_quality'] = {
                'underexposed_pixels_percent': float(np.sum(gray_normalized < 5) / gray_normalized.size * 100),
                'overexposed_pixels_percent': float(np.sum(gray_normalized > 95) / gray_normalized.size * 100),
                'well_exposed_pixels_percent': float(np.sum((gray_normalized >= 5) & (gray_normalized <= 95)) / gray_normalized.size * 100)
            }
            
            return stats
            
        except Exception as e:
            logger.warning(f"Error calculating gray level statistics: {e}")
            return None

    def _encode_frame_jpeg(self, frame, quality=85):
        """
        Encode frame to JPEG format for efficient network transmission
        
        Args:
            frame: RGB numpy array
            quality: JPEG quality (1-100, higher = better quality, larger size)
        
        Returns:
            dict: {
                'format': 'jpeg',
                'data': bytes,
                'size_bytes': int,
                'compression_ratio': float
            }
        """
        try:
            # Convert RGB to BGR for OpenCV JPEG encoding
            if len(frame.shape) == 3 and frame.shape[2] == 3:
                bgr_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            else:
                bgr_frame = frame
            
            # Encode to JPEG with specified quality
            encode_params = [cv2.IMWRITE_JPEG_QUALITY, quality]
            success, encoded_img = cv2.imencode('.jpg', bgr_frame, encode_params)
            
            if not success:
                raise ValueError("Failed to encode frame to JPEG")
            
            # Calculate compression statistics
            original_size = frame.nbytes
            compressed_size = len(encoded_img)
            compression_ratio = original_size / compressed_size if compressed_size > 0 else 1.0
            
            return {
                'format': 'jpeg',
                'data': encoded_img.tobytes(),
                'size_bytes': compressed_size,
                'compression_ratio': compression_ratio,
                'original_size': original_size
            }
            
        except Exception as e:
            logger.error(f"Error encoding frame to JPEG: {e}")
            # Return uncompressed as fallback
            raise e

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
    def get_current_well_location(self, wellplate_type: str=Field('96', description="Type of the well plate (e.g., '6', '12', '24', '96', '384')"), context=None):
        """
        Get the current well location based on the stage position.
        Returns: Dictionary with well location information including row, column, well_id, and position status
        """
        try:
            # Check authentication
            if context and not self.check_permission(context.get("user", {})):
                raise Exception("User not authorized to access this service")
            
            well_info = self.squidController.get_well_from_position(wellplate_type)
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
    def get_microscope_configuration(self, config_section: str = Field("all", description="Configuration section to retrieve ('all', 'camera', 'stage', 'illumination', 'acquisition', 'limits', 'hardware', 'wellplate', 'optics', 'autofocus')"), include_defaults: bool = Field(True, description="Whether to include default values from config.py"), context=None):
        """
        Get microscope configuration information in JSON format.
        Input: config_section: str = Field("all", description="Configuration section to retrieve ('all', 'camera', 'stage', 'illumination', 'acquisition', 'limits', 'hardware', 'wellplate', 'optics', 'autofocus')"), include_defaults: bool = Field(True, description="Whether to include default values from config.py")
        Returns: Configuration data as a JSON object
        """
        try:
            # Check authentication
            if context and not self.check_permission(context.get("user", {})):
                raise Exception("User not authorized to access this service")
            
            try:
                from .control.config import get_microscope_configuration_data
            except ImportError:
                from squid_control.control.config import get_microscope_configuration_data
            
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
    async def get_canvas_chunk(self, x_mm: float = Field(..., description="X coordinate of the stage location in millimeters"), y_mm: float = Field(..., description="Y coordinate of the stage location in millimeters"), scale_level: int = Field(1, description="Scale level for the chunk (0-2, where 0 is highest resolution)"), context=None):
        """Get a canvas chunk based on microscope stage location (available only in simulation mode when not running locally)"""
        
        # Check if this function is available in current mode
        if self.is_local:
            raise Exception("get_canvas_chunk is not available in local mode")
        
        if not self.is_simulation:
            raise Exception("get_canvas_chunk is only available in simulation mode")
        
        try:
            # Check authentication
            if context and not self.check_permission(context.get("user", {})):
                raise Exception("User not authorized to access this service")
            
            logger.info(f"Getting canvas chunk at position: x={x_mm}mm, y={y_mm}mm, scale_level={scale_level}")
            
            # Initialize ZarrImageManager if not already initialized
            if not hasattr(self, 'zarr_image_manager') or self.zarr_image_manager is None:
                try:
                    from .hypha_tools.artifact_manager.artifact_manager import ZarrImageManager
                except ImportError:
                    from squid_control.hypha_tools.artifact_manager.artifact_manager import ZarrImageManager
                self.zarr_image_manager = ZarrImageManager()
                success = await self.zarr_image_manager.connect(server_url=self.server_url)
                if not success:
                    raise RuntimeError("Failed to connect to ZarrImageManager")
                logger.info("ZarrImageManager initialized for get_canvas_chunk")
            
            # Use the current simulated sample data alias
            dataset_id = self.get_simulated_sample_data_alias()
            channel_name = 'BF_LED_matrix_full'  # Always use brightfield channel
            
            # Use parameters similar to the simulation camera
            pixel_size_um = 0.333  # Default pixel size used in simulation
            
            # Get scale factor based on scale level
            scale_factors = {0: 1, 1: 4, 2: 16}  # scale0=1x, scale1=1/4x, scale2=1/16x
            scale_factor = scale_factors.get(scale_level, 4)  # Default to scale1
            
            # Convert microscope coordinates (mm) to pixel coordinates
            pixel_x = int((x_mm / pixel_size_um) * 1000 / scale_factor)
            pixel_y = int((y_mm / pixel_size_um) * 1000 / scale_factor)
            
            # Convert pixel coordinates to chunk coordinates
            chunk_size = 256  # Default chunk size used by ZarrImageManager
            chunk_x = pixel_x // chunk_size
            chunk_y = pixel_y // chunk_size
            
            logger.info(f"Converted coordinates: x={x_mm}mm, y={y_mm}mm to pixel coords: x={pixel_x}, y={pixel_y}, chunk coords: x={chunk_x}, y={chunk_y} (scale{scale_level})")
            
            # Get the single chunk data from ZarrImageManager
            region_data = await self.zarr_image_manager.get_region_np_data(
                dataset_id, 
                channel_name, 
                scale_level,
                chunk_x,  # Chunk X coordinate
                chunk_y,  # Chunk Y coordinate
                direct_region=None,  # Don't use direct_region, use chunk coordinates instead
                width=chunk_size,
                height=chunk_size
            )
            
            if region_data is None:
                raise Exception("Failed to retrieve chunk data from Zarr storage")
            
            # Convert numpy array to base64 encoded PNG for transmission
            try:
                # Ensure data is in uint8 format
                if region_data.dtype != np.uint8:
                    if region_data.dtype == np.float32 or region_data.dtype == np.float64:
                        # Normalize floating point data
                        if region_data.max() > 0:
                            region_data = (region_data / region_data.max() * 255).astype(np.uint8)
                        else:
                            region_data = np.zeros(region_data.shape, dtype=np.uint8)
                    else:
                        # For other integer types, scale appropriately
                        region_data = (region_data / region_data.max() * 255).astype(np.uint8) if region_data.max() > 0 else region_data.astype(np.uint8)
                        
                # Convert to PIL Image and then to base64
                pil_image = Image.fromarray(region_data)
                buffer = io.BytesIO()
                pil_image.save(buffer, format="PNG")
                img_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
                
                return {
                    "data": img_base64,
                    "format": "png_base64",
                    "scale_level": scale_level,
                    "stage_location": {"x_mm": x_mm, "y_mm": y_mm},
                    "chunk_coordinates": {"chunk_x": chunk_x, "chunk_y": chunk_y}
                }
                
            except Exception as e:
                logger.error(f"Error converting chunk data to base64: {e}")
                raise e
                
        except Exception as e:
            logger.error(f"Error in get_canvas_chunk: {e}")
            import traceback
            traceback.print_exc()
            raise e

    @schema_function(skip_self=True)
    def set_stage_velocity(self, velocity_x_mm_per_s: Optional[float] = Field(None, description="Maximum velocity for X axis in mm/s (default: uses configuration value)"), velocity_y_mm_per_s: Optional[float] = Field(None, description="Maximum velocity for Y axis in mm/s (default: uses configuration value)"), context=None):
        """
        Set the maximum velocity for X and Y stage axes.
        
        This function allows you to control how fast the microscope stage moves.
        Lower velocities provide more precision but slower movement.
        Higher velocities enable faster navigation but may reduce precision.
        
        Args:
            velocity_x_mm_per_s: Maximum velocity for X axis in mm/s. If not specified, uses default from configuration.
            velocity_y_mm_per_s: Maximum velocity for Y axis in mm/s. If not specified, uses default from configuration.
            
        Returns:
            dict: Status and current velocity settings
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
                                experiment_name: str = Field(..., description="Name of the experiment to upload (this becomes the dataset name)"),
                                description: str = Field("", description="Description of the dataset"),
                                include_acquisition_settings: bool = Field(True, description="Whether to include current acquisition settings as metadata"),
                                context=None):
        """
        Upload an experiment's well canvases as individual zip files to a single dataset in the artifact manager.
        
        This function uploads each well canvas from the experiment as a separate zip file
        within a single dataset. The dataset name will be '{experiment_name}-{date and time}'.
        
        Args:
            experiment_name: Name of the experiment to upload (becomes the dataset name)
            description: Description of the dataset
            include_acquisition_settings: Whether to include current acquisition settings as metadata
            
        Returns:
            dict: Upload result information with details about uploaded well canvases
        """
        
        try:
            # Check authentication
            if context and not self.check_permission(context.get("user", {})):
                raise Exception("User not authorized to access this service")
            
            # Check if experiment manager is initialized
            if not hasattr(self.squidController, 'experiment_manager') or self.squidController.experiment_manager is None:
                raise Exception("Experiment manager not initialized. Start a scanning operation first to create data.")
            
            # Check if zarr artifact manager is available
            if self.zarr_artifact_manager is None:
                raise Exception("Zarr artifact manager not initialized. Check that AGENT_LENS_WORKSPACE_TOKEN is set.")
            
            # Get experiment information
            experiment_info = self.squidController.experiment_manager.get_experiment_info(experiment_name)
            
            if not experiment_info.get("well_canvases"):
                raise Exception(f"No well canvases found in experiment '{experiment_name}'. Start a scanning operation first to create data.")
            

            logger.info(f"Uploading experiment '{experiment_name}' with {len(experiment_info['well_canvases'])} well canvases to single dataset")
            
            # Prepare acquisition settings if requested
            acquisition_settings = None
            if include_acquisition_settings:
                # Get settings from the first available well canvas
                first_well = experiment_info['well_canvases'][0]
                well_path = Path(first_well['path'])
                
                # Try to get canvas info from the first well
                try:
                    # Create a temporary canvas instance to get export info
                    try:
                        from .stitching.zarr_canvas import WellZarrCanvas
                        from .control.config import ChannelMapper, CONFIG
                    except ImportError:
                        from squid_control.stitching.zarr_canvas import WellZarrCanvas
                        from squid_control.control.config import ChannelMapper, CONFIG
                    
                    # Parse well info from path (e.g., "well_A1_96.zarr" -> A, 1, 96)
                    well_name = well_path.stem  # "well_A1_96"
                    if well_name.startswith("well_"):
                        well_info = well_name[5:]  # "A1_96"
                        if "_" in well_info:
                            well_part, wellplate_type = well_info.rsplit("_", 1)
                            if len(well_part) >= 2:
                                well_row = well_part[0]
                                well_column = int(well_part[1:])
                                
                                # Create temporary canvas to get export info
                                temp_canvas = WellZarrCanvas(
                                    well_row=well_row,
                                    well_column=well_column,
                                    wellplate_type=wellplate_type,
                                    padding_mm=1.0,
                                    base_path=str(well_path.parent),
                                    pixel_size_xy_um=self.squidController.pixel_size_xy,
                                    channels=ChannelMapper.get_all_human_names(),
                                    rotation_angle_deg=CONFIG.STITCHING_ROTATION_ANGLE_DEG
                                )
                                
                                # Get export info from the temporary canvas
                                export_info = temp_canvas.get_export_info()
                                temp_canvas.close()
                                
                                acquisition_settings = {
                                    "pixel_size_xy_um": export_info.get("canvas_dimensions", {}).get("pixel_size_um"),
                                    "channels": export_info.get("channels", []),
                                    "canvas_dimensions": export_info.get("canvas_dimensions", {}),
                                    "num_scales": export_info.get("num_scales"),
                                    "microscope_service_id": self.service_id,
                                    "experiment_name": experiment_name,
                                    "wellplate_type": wellplate_type
                                }
                except Exception as e:
                    logger.warning(f"Could not get detailed acquisition settings: {e}")
                    # Fallback to basic settings
                    acquisition_settings = {
                        "microscope_service_id": self.service_id,
                        "experiment_name": experiment_name,
                        "total_wells": len(experiment_info['well_canvases']),
                        "total_size_mb": total_size_mb
                    }
            
            # Prepare all well canvases for upload to single dataset
            zarr_files_info = []
            well_info_list = []
            
            for well_info in experiment_info['well_canvases']:
                well_name = well_info['name']
                well_path = Path(well_info['path'])
                well_size_mb = well_info['size_mb']
                
                logger.info(f"Preparing well canvas: {well_name} ({well_size_mb:.2f} MB)")
                
                try:
                    # Create a temporary canvas instance to export the well
                    try:
                        from .stitching.zarr_canvas import WellZarrCanvas
                        from .control.config import ChannelMapper, CONFIG
                    except ImportError:
                        from squid_control.stitching.zarr_canvas import WellZarrCanvas
                        from squid_control.control.config import ChannelMapper, CONFIG
                    
                    # Parse well info from name (e.g., "well_A1_96" -> A, 1, 96)
                    if well_name.startswith("well_"):
                        well_info_part = well_name[5:]  # "A1_96"
                        if "_" in well_info_part:
                            well_part, wellplate_type = well_info_part.rsplit("_", 1)
                            if len(well_part) >= 2:
                                well_row = well_part[0]
                                well_column = int(well_part[1:])
                                
                                # Create temporary canvas for export
                                temp_canvas = WellZarrCanvas(
                                    well_row=well_row,
                                    well_column=well_column,
                                    wellplate_type=wellplate_type,
                                    padding_mm=1.0,
                                    base_path=str(well_path.parent),
                                    pixel_size_xy_um=self.squidController.pixel_size_xy,
                                    channels=ChannelMapper.get_all_human_names(),
                                    rotation_angle_deg=CONFIG.STITCHING_ROTATION_ANGLE_DEG
                                )
                                
                                # Export the well canvas as zip using asyncio.to_thread to avoid blocking
                                well_zip_content = await asyncio.to_thread(temp_canvas.export_as_zip)
                                temp_canvas.close()
                                
                                # Add to files info for batch upload
                                zarr_files_info.append({
                                    'name': well_name,
                                    'content': well_zip_content,
                                    'size_mb': well_size_mb
                                })
                                
                                well_info_list.append({
                                    "well_name": well_name,
                                    "well_row": well_row,
                                    "well_column": well_column,
                                    "wellplate_type": wellplate_type,
                                    "size_mb": well_size_mb
                                })
                                
                                logger.info(f"Successfully prepared well {well_name}")
                                
                            else:
                                logger.warning(f"Could not parse well name: {well_name}")
                        else:
                            logger.warning(f"Could not parse well name: {well_name}")
                    else:
                        logger.warning(f"Unexpected well name format: {well_name}")
                        
                except Exception as e:
                    logger.error(f"Failed to prepare well {well_name}: {e}")
                    # Continue with other wells
                    continue
            
            if not zarr_files_info:
                raise Exception("No well canvases were successfully prepared for upload")
            
            # Upload all well canvases to a single dataset
            logger.info(f"Uploading {len(zarr_files_info)} well canvases to single dataset...")
            
            # Add well information to acquisition settings
            if acquisition_settings:
                acquisition_settings["wells"] = well_info_list
            
            upload_result = await self.zarr_artifact_manager.upload_multiple_zarr_files_to_dataset(
                microscope_service_id=self.service_id,
                experiment_id=experiment_name,
                zarr_files_info=zarr_files_info,
                acquisition_settings=acquisition_settings,
                description=description or f"Experiment {experiment_name} with {len(zarr_files_info)} well canvases"
            )
            
            logger.info(f"Successfully uploaded experiment '{experiment_name}' to single dataset")
            
            return {
                "success": True,
                "experiment_name": experiment_name,
                "dataset_name": upload_result["dataset_name"],
                "uploaded_wells": well_info_list,
                "total_wells": len(well_info_list),
                "total_size_mb": upload_result["total_size_mb"],
                "acquisition_settings": acquisition_settings,
                "description": description or f"Experiment {experiment_name} with {len(well_info_list)} well canvases",
                "upload_result": upload_result
            }
            
        except Exception as e:
            logger.error(f"Error uploading experiment dataset: {e}")
            raise e
    
    @schema_function(skip_self=True)
    async def list_microscope_galleries(self, microscope_service_id: str = Field(..., description="Microscope service ID to list galleries for"), context=None):
        """
        List all galleries (collections) available for a given microscope's service ID.
        This includes both standard microscope galleries and experiment-based galleries.
        Returns a list of gallery info dicts.
        """
        try:
            # Check authentication
            if context and not self.check_permission(context.get("user", {})):
                raise Exception("User not authorized to access this service")
            
            if self.zarr_artifact_manager is None:
                raise Exception("Zarr artifact manager not initialized. Check that AGENT_LENS_WORKSPACE_TOKEN is set.")

            # List all collections in the agent-lens workspace (top-level)
            all_collections = await self.zarr_artifact_manager.navigate_collections(parent_id=None)
            galleries = []
            
            # Check if microscope service ID ends with a number
            import re
            number_match = re.search(r'-(\d+)$', microscope_service_id)
            
            for coll in all_collections:
                manifest = coll.get('manifest', {})
                alias = coll.get('alias', '')
                
                # Standard gallery
                if alias == f"agent-lens/microscope-gallery-{microscope_service_id}":
                    galleries.append(coll)
                # Experiment-based gallery (for microscope IDs ending with numbers)
                elif number_match:
                    gallery_number = number_match.group(1)
                    if alias.startswith(f"agent-lens/{gallery_number}-"):
                        # Check manifest for matching microscope_service_id
                        if manifest.get('microscope_service_id') == microscope_service_id:
                            galleries.append(coll)
                # Fallback: check manifest field
                elif manifest.get('microscope_service_id') == microscope_service_id:
                    galleries.append(coll)
                    
            return {
                "success": True,
                "microscope_service_id": microscope_service_id,
                "galleries": galleries,
                "total": len(galleries)
            }
        except Exception as e:
            logger.error(f"Error listing galleries: {e}")
            raise e

    @schema_function(skip_self=True)
    async def list_gallery_datasets(self, gallery_id: str = Field(None, description="Gallery (collection) artifact ID, e.g. agent-lens/1-..."), microscope_service_id: str = Field(None, description="Microscope service ID (optional, used to find gallery if gallery_id not given)"), experiment_id: str = Field(None, description="Experiment ID (optional, used to find gallery if gallery_id not given)"), context=None):
        """
        List all datasets in a gallery (collection).
        You can specify the gallery by its artifact ID, or provide microscope_service_id and/or experiment_id to find the gallery.
        Returns a list of datasets in the gallery.
        """
        try:
            # Check authentication
            if context and not self.check_permission(context.get("user", {})):
                raise Exception("User not authorized to access this service")
            
            if self.zarr_artifact_manager is None:
                raise Exception("Zarr artifact manager not initialized. Check that AGENT_LENS_WORKSPACE_TOKEN is set.")

            # Find the gallery if not given
            gallery = None
            if gallery_id:
                # Try to read the gallery directly
                gallery = await self.zarr_artifact_manager._svc.read(artifact_id=gallery_id)
            else:
                # Use microscope_service_id and/or experiment_id to find the gallery
                if microscope_service_id is None and experiment_id is None:
                    raise Exception("You must provide either gallery_id, microscope_service_id, or experiment_id.")
                gallery = await self.zarr_artifact_manager.create_or_get_microscope_gallery(
                    microscope_service_id or '', experiment_id=experiment_id)
            # List datasets in the gallery
            datasets = await self.zarr_artifact_manager._svc.list(gallery["id"])
            return {
                "success": True,
                "gallery_id": gallery["id"],
                "gallery_alias": gallery.get("alias"),
                "gallery_name": gallery.get("manifest", {}).get("name"),
                "datasets": datasets,
                "total": len(datasets)
            }
        except Exception as e:
            logger.error(f"Error listing gallery datasets: {e}")
            raise e

    def get_microscope_configuration_schema(self, config: GetMicroscopeConfigurationInput, context=None):
        return self.get_microscope_configuration(config.config_section, config.include_defaults, context)

    def set_stage_velocity_schema(self, config: SetStageVelocityInput, context=None):
        """Set the maximum velocity for X and Y stage axes with schema validation."""
        return self.set_stage_velocity(config.velocity_x_mm_per_s, config.velocity_y_mm_per_s, context)

    @schema_function(skip_self=True)
    async def normal_scan_with_stitching(self, start_x_mm: float = Field(20, description="Starting X position in millimeters"), 
                                       start_y_mm: float = Field(20, description="Starting Y position in millimeters"),
                                       Nx: int = Field(5, description="Number of positions in X direction"),
                                       Ny: int = Field(5, description="Number of positions in Y direction"),
                                       dx_mm: float = Field(0.9, description="Interval between positions in X (millimeters)"),
                                       dy_mm: float = Field(0.9, description="Interval between positions in Y (millimeters)"),
                                       illumination_settings: Optional[List[dict]] = Field(None, description="List of channel settings"),
                                       do_contrast_autofocus: bool = Field(False, description="Whether to perform contrast-based autofocus"),
                                       do_reflection_af: bool = Field(False, description="Whether to perform reflection-based autofocus"),
                                       action_ID: str = Field('normal_scan_stitching', description="Identifier for this scan"),
                                       timepoint: int = Field(0, description="Timepoint index for this scan (default 0)"),
                                       experiment_name: Optional[str] = Field(None, description="Name of the experiment to use. If None, uses active experiment or 'default' as fallback"),
                                       wells_to_scan: List[str] = Field(default_factory=lambda: ['A1'], description="List of wells to scan (e.g., ['A1', 'B2', 'C3'])"),
                                       wellplate_type: str = Field('96', description="Well plate type ('6', '12', '24', '96', '384')"),
                                       well_padding_mm: float = Field(1.0, description="Padding around well in mm"),
                                       uploading: bool = Field(False, description="Enable upload after scanning is complete"),
                                       context=None):
        """
        Perform a normal scan with live stitching to OME-Zarr canvas using well-based approach.
        The images are saved to well-specific zarr canvases within an experiment folder.
        
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
            wellplate_type: Well plate type ('6', '12', '24', '96', '384')
            well_padding_mm: Padding around well in mm
            uploading: Enable upload after scanning is complete
            
        Returns:
            dict: Status of the scan
        """
        try:
            # Check authentication
            if context and not self.check_permission(context.get("user", {})):
                raise Exception("User not authorized to access this service")
            
            # Set default illumination settings if not provided
            if illumination_settings is None:
                illumination_settings = [{'channel': 'BF LED matrix full', 'intensity': 50, 'exposure_time': 100}]
            
            logger.info(f"Starting normal scan with stitching: {Nx}x{Ny} positions from ({start_x_mm}, {start_y_mm})")
            
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
            
            # Perform the normal scan
            await self.squidController.normal_scan_with_stitching(
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
                wellplate_type=wellplate_type,
                well_padding_mm=well_padding_mm
            )
            
            # Upload the experiment if uploading is enabled
            upload_result = None
            if uploading:
                try:
                    logger.info("Uploading experiment after normal scan completion")
                    upload_result = await self.upload_zarr_dataset(
                        experiment_name=experiment_name or self.squidController.experiment_manager.current_experiment_name,
                        description=f"Normal scan with stitching - {action_ID}",
                        include_acquisition_settings=True
                    )
                    logger.info("Successfully uploaded experiment after normal scan")
                except Exception as e:
                    logger.error(f"Failed to upload experiment after normal scan: {e}")
                    # Don't raise the exception - continue with response
            
            return {
                "success": True,
                "message": f"Normal scan with stitching completed successfully",
                "scan_parameters": {
                    "start_position": {"x_mm": start_x_mm, "y_mm": start_y_mm},
                    "grid_size": {"nx": Nx, "ny": Ny},
                    "step_size": {"dx_mm": dx_mm, "dy_mm": dy_mm},
                    "total_area_mm2": (Nx * dx_mm) * (Ny * dy_mm),
                    "experiment_name": self.squidController.experiment_manager.current_experiment_name,  # Include the actual experiment used
                    "wells_scanned": wells_to_scan
                },
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

    @schema_function(skip_self=True)
    async def quick_scan_with_stitching(self, wellplate_type: str = Field('96', description="Well plate type ('6', '12', '24', '96', '384')"),
                                      exposure_time: float = Field(5, description="Camera exposure time in milliseconds (max 30ms)"),
                                      intensity: float = Field(70, description="Brightfield LED intensity (0-100)"),
                                      fps_target: int = Field(10, description="Target frame rate for acquisition (default 10fps)"),
                                      action_ID: str = Field('quick_scan_stitching', description="Identifier for this scan"),
                                      n_stripes: int = Field(4, description="Number of stripes per well (default 4)"),
                                      stripe_width_mm: float = Field(4.0, description="Length of each stripe inside a well in mm (default 4.0)"),
                                      dy_mm: float = Field(0.9, description="Y increment between stripes in mm (default 0.9)"),
                                      velocity_scan_mm_per_s: float = Field(7.0, description="Stage velocity during stripe scanning in mm/s (default 7.0)"),
                                      do_contrast_autofocus: bool = Field(False, description="Whether to perform contrast-based autofocus"),
                                      do_reflection_af: bool = Field(False, description="Whether to perform reflection-based autofocus"),
                                      experiment_name: Optional[str] = Field(None, description="Name of the experiment to use. If None, uses active experiment or 'default' as fallback"),
                                      well_padding_mm: float = Field(1.0, description="Padding around each well in mm"),
                                      uploading: bool = Field(False, description="Enable upload after scanning is complete"),
                                      context=None):
        """
        Perform a quick scan with live stitching to OME-Zarr canvas - brightfield only.
        Uses 4-stripe x 4 mm scanning pattern with serpentine motion per well.
        Only supports brightfield channel with exposure time ≤ 30ms.
        Always uses well-based approach with individual canvases per well.
        
        Args:
            wellplate_type: Well plate format ('6', '12', '24', '96', '384')
            exposure_time: Camera exposure time in milliseconds (must be ≤ 30ms)
            intensity: Brightfield LED intensity (0-100)
            fps_target: Target frame rate for acquisition (default 10fps)
            action_ID: Unique identifier for this scan
            n_stripes: Number of stripes per well (default 4)
            stripe_width_mm: Length of each stripe inside a well in mm (default 4.0)
            dy_mm: Y increment between stripes in mm (default 0.9)
            velocity_scan_mm_per_s: Stage velocity during stripe scanning in mm/s (default 7.0)
            do_contrast_autofocus: Whether to perform contrast-based autofocus at each well
            do_reflection_af: Whether to perform reflection-based autofocus at each well
            experiment_name: Name of the experiment to use. If None, uses active experiment or 'default' as fallback
            well_padding_mm: Padding around each well in mm
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
            
            logger.info(f"Starting quick scan with stitching: {wellplate_type} well plate, {n_stripes} stripes × {stripe_width_mm}mm, dy={dy_mm}mm, scan_velocity={velocity_scan_mm_per_s}mm/s, fps={fps_target}")
            
            # Check if video buffering is active and stop it during scanning
            video_buffering_was_active = self.frame_acquisition_running
            if video_buffering_was_active:
                logger.info("Video buffering is active, stopping it temporarily during quick scanning")
                await self.stop_video_buffering()
                # Wait for camera to settle after stopping video buffering
                logger.info("Waiting for camera to settle after stopping video buffering...")
                await asyncio.sleep(0.5)
            
            # Set scanning flag to prevent automatic video buffering restart during scan
            self.scanning_in_progress = True
            
            # Record start time for performance metrics
            start_time = time.time()
            
            # Perform the quick scan
            await self.squidController.quick_scan_with_stitching(
                wellplate_type=wellplate_type,
                exposure_time=exposure_time,
                intensity=intensity,
                fps_target=fps_target,
                action_ID=action_ID,
                n_stripes=n_stripes,
                stripe_width_mm=stripe_width_mm,
                dy_mm=dy_mm,
                velocity_scan_mm_per_s=velocity_scan_mm_per_s,
                do_contrast_autofocus=do_contrast_autofocus,
                do_reflection_af=do_reflection_af,
                experiment_name=experiment_name,
                well_padding_mm=well_padding_mm
            )
            
            # Calculate performance metrics
            scan_duration = time.time() - start_time
            
            # Calculate well plate dimensions for area estimation
            wellplate_configs = {
                '6': {'rows': 2, 'cols': 3},
                '12': {'rows': 3, 'cols': 4},
                '24': {'rows': 4, 'cols': 6},
                '96': {'rows': 8, 'cols': 12},
                '384': {'rows': 16, 'cols': 24}
            }
            
            # Convert wellplate_type to string to avoid ObjectProxy issues
            wellplate_type_str = str(wellplate_type)
            config = wellplate_configs.get(wellplate_type_str, wellplate_configs['96'])
            total_wells = config['rows'] * config['cols']
            total_stripes = total_wells * n_stripes
            
            # Upload the experiment if uploading is enabled
            upload_result = None
            if uploading:
                try:
                    logger.info("Uploading experiment after quick scan completion")
                    upload_result = await self.upload_zarr_dataset(
                        experiment_name=experiment_name or self.squidController.experiment_manager.current_experiment_name,
                        description=f"Quick scan with stitching - {action_ID}",
                        include_acquisition_settings=True
                    )
                    logger.info("Successfully uploaded experiment after quick scan")
                except Exception as e:
                    logger.error(f"Failed to upload experiment after quick scan: {e}")
                    # Don't raise the exception - continue with response
            
            return {
                "success": True,
                "message": f"Quick scan with stitching completed successfully",
                "scan_parameters": {
                    "wellplate_type": wellplate_type_str,
                    "wells_scanned": total_wells,
                    "stripes_per_well": n_stripes,
                    "stripe_width_mm": stripe_width_mm,
                    "dy_mm": dy_mm,
                    "total_stripes": total_stripes,
                    "exposure_time_ms": exposure_time,
                    "intensity": intensity,
                    "scan_velocity_mm_per_s": velocity_scan_mm_per_s,
                    "target_fps": fps_target,
                    "inter_well_velocity_mm_per_s": 30.0
                },
                "performance_metrics": {
                    "total_scan_time_seconds": round(scan_duration, 2),
                    "scan_time_per_well_seconds": round(scan_duration / total_wells, 2),
                    "scan_time_per_stripe_seconds": round(scan_duration / total_stripes, 2),
                    "estimated_frames_acquired": int(scan_duration * fps_target)
                },
                "stitching_info": {
                    "zarr_scales_updated": "1-5 (scale 0 skipped for performance)",
                    "channel": "BF LED matrix full",
                    "action_id": action_ID,
                    "pattern": f"{n_stripes}-stripe × {stripe_width_mm}mm serpentine per well",
                    "experiment_name": self.squidController.experiment_manager.current_experiment_name
                },
                "upload_result": upload_result
            }
            
        except ValueError as e:
            logger.error(f"Validation error in quick scan: {e}")
            raise e
        except Exception as e:
            logger.error(f"Failed to perform quick scan with stitching: {e}")
            raise e
        finally:
            # Always reset the scanning flag, regardless of success or failure
            self.scanning_in_progress = False
            logger.info("Quick scanning completed, video buffering auto-start is now re-enabled")

    @schema_function(skip_self=True)
    def stop_scan_and_stitching(self, context=None):
        """
        Stop any ongoing scanning and stitching processes.
        This will interrupt normal_scan_with_stitching and quick_scan_with_stitching if they are running.
        
        Returns:
            dict: Status of the stop request
        """
        try:
            # Check authentication
            if context and not self.check_permission(context.get("user", {})):
                raise Exception("User not authorized to access this service")
            
            logger.info("Stop scan and stitching requested")
            
            # Call the controller's stop method
            result = self.squidController.stop_scan_and_stitching()
            
            # Also reset the scanning flag at service level
            if hasattr(self, 'scanning_in_progress'):
                self.scanning_in_progress = False
                logger.info("Service scanning flag reset")
            
            return {
                "success": True,
                "message": "Scan stop requested - ongoing scans will be interrupted",
                "controller_response": result
            }
            
        except Exception as e:
            logger.error(f"Failed to stop scan and stitching: {e}")
            raise e

    @schema_function(skip_self=True)
    def get_stitched_region(self, center_x_mm: float = Field(..., description="Center X position in absolute stage coordinates (mm)"),
                           center_y_mm: float = Field(..., description="Center Y position in absolute stage coordinates (mm)"),
                           width_mm: float = Field(5.0, description="Width of region in mm"),
                           height_mm: float = Field(5.0, description="Height of region in mm"),
                           wellplate_type: str = Field('96', description="Well plate type ('6', '12', '24', '96', '384')"),
                           scale_level: int = Field(0, description="Scale level (0=full resolution, 1=1/4, 2=1/16, etc)"),
                           channel_name: str = Field('BF LED matrix full', description="Channel names to retrieve and merge (comma-separated string or single channel name, e.g., 'BF LED matrix full' or 'BF LED matrix full,Fluorescence 488 nm Ex')"),
                           timepoint: int = Field(0, description="Timepoint index to retrieve (default 0)"),
                           well_padding_mm: float = Field(1.0, description="Padding around wells in mm"),
                           output_format: str = Field('base64', description="Output format: 'base64' or 'array'"),
                           context=None):
        """
        Get a stitched region that may span multiple wells by determining which wells 
        are needed and combining their data. Supports merging multiple channels with proper colors.
        
        This function automatically determines which wells intersect with the requested region
        and stitches together the data from multiple wells if necessary. When multiple channels
        are specified, they are merged into a single RGB image using the channel color scheme.
        
        Args:
            center_x_mm: Center X position in absolute stage coordinates (mm)
            center_y_mm: Center Y position in absolute stage coordinates (mm)
            width_mm: Width of region in mm
            height_mm: Height of region in mm
            wellplate_type: Well plate type ('6', '12', '24', '96', '384')
            scale_level: Scale level (0=full resolution, 1=1/4, 2=1/16, etc)
            channel_name: Channel names to retrieve and merge (comma-separated string or single channel name)
            timepoint: Timepoint index to retrieve (default 0)
            well_padding_mm: Padding around wells in mm
            output_format: Output format ('base64' for compressed image, 'array' for numpy array)
            
        Returns:
            dict: Retrieved stitched image data with metadata and region information
        """
        try:
            # Check authentication
            if context and not self.check_permission(context.get("user", {})):
                raise Exception("User not authorized to access this service")
            
            # Parse channel_name string into a list
            if isinstance(channel_name, str):
                # Split by comma and strip whitespace, filter out empty strings
                channel_list = [ch.strip() for ch in channel_name.split(',') if ch.strip()]
                logger.info(f"Parsed channel names: '{channel_name}' -> {channel_list}")
            else:
                # If it's already a list, use it as is
                channel_list = list(channel_name)
                logger.info(f"Using channel list: {channel_list}")
            
            # Validate channel names
            if not channel_list:
                return {
                    "success": False,
                    "message": "At least one channel name must be specified",
                    "region": {
                        "center_x_mm": center_x_mm,
                        "center_y_mm": center_y_mm,
                        "width_mm": width_mm,
                        "height_mm": height_mm,
                        "wellplate_type": wellplate_type,
                        "scale_level": scale_level,
                        "channels": channel_list,
                        "timepoint": timepoint,
                        "well_padding_mm": well_padding_mm
                    }
                }
            
            # Get regions for each channel
            channel_regions = []
            for ch_name in channel_list:
                region = self.squidController.get_stitched_region(
                    center_x_mm=center_x_mm,
                    center_y_mm=center_y_mm,
                    width_mm=width_mm,
                    height_mm=height_mm,
                    wellplate_type=wellplate_type,
                    scale_level=scale_level,
                    channel_name=ch_name,
                    timepoint=timepoint,
                    well_padding_mm=well_padding_mm
                )
                
                if region is None:
                    logger.warning(f"No data available for channel '{ch_name}' at ({center_x_mm:.2f}, {center_y_mm:.2f})")
                    continue
                    
                channel_regions.append((ch_name, region))
            
            if not channel_regions:
                return {
                    "success": False,
                    "message": f"No data available for any channels at ({center_x_mm:.2f}, {center_y_mm:.2f}) with size ({width_mm:.2f}x{height_mm:.2f})",
                    "region": {
                        "center_x_mm": center_x_mm,
                        "center_y_mm": center_y_mm,
                        "width_mm": width_mm,
                        "height_mm": height_mm,
                        "wellplate_type": wellplate_type,
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
                merged_region = self._merge_channels_to_rgb(channel_regions)
                is_rgb = True
            
            if merged_region is None:
                return {
                    "success": False,
                    "message": "Failed to merge channels",
                    "region": {
                        "center_x_mm": center_x_mm,
                        "center_y_mm": center_y_mm,
                        "width_mm": width_mm,
                        "height_mm": height_mm,
                        "wellplate_type": wellplate_type,
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
                from PIL import Image
                import io
                
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
                        "wellplate_type": wellplate_type,
                        "scale_level": scale_level,
                        "channels": channel_list,
                        "timepoint": timepoint,
                        "well_padding_mm": well_padding_mm
                    }
                }
            else:
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
                        "wellplate_type": wellplate_type,
                        "scale_level": scale_level,
                        "channels": channel_list,
                        "timepoint": timepoint,
                        "well_padding_mm": well_padding_mm
                    }
                }
                
        except Exception as e:
            logger.error(f"Failed to get stitched region: {e}")
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
            
            logger.info(f"Successfully merged {len(channel_regions)} channels into RGB image")
            return rgb_image
            
        except Exception as e:
            logger.error(f"Error merging channels to RGB: {e}")
            return None

    @schema_function(skip_self=True)
    async def create_experiment(self, experiment_name: str = Field(..., description="Name for the new experiment"), context=None):
        """
        Create a new experiment with the given name.
        
        Args:
            experiment_name: Name for the new experiment
            
        Returns:
            dict: Information about the created experiment
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
        List all available experiments.
        
        Returns:
            dict: List of experiments and their status
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
    async def set_active_experiment(self, experiment_name: str = Field(..., description="Name of the experiment to activate"), context=None):
        """
        Set the active experiment for operations.
        
        Args:
            experiment_name: Name of the experiment to activate
            
        Returns:
            dict: Information about the activated experiment
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
    async def remove_experiment(self, experiment_name: str = Field(..., description="Name of the experiment to remove"), context=None):
        """
        Remove an experiment.
        
        Args:
            experiment_name: Name of the experiment to remove
            
        Returns:
            dict: Information about the removed experiment
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
    async def reset_experiment(self, experiment_name: str = Field(..., description="Name of the experiment to reset"), context=None):
        """
        Reset an experiment.
        
        Args:
            experiment_name: Name of the experiment to reset
            
        Returns:
            dict: Information about the reset experiment
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
    async def get_experiment_info(self, experiment_name: str = Field(..., description="Name of the experiment to retrieve information about"), context=None):
        """
        Get information about an experiment.
        
        Args:
            experiment_name: Name of the experiment to retrieve information about
            
        Returns:
            dict: Information about the experiment
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

# Define a signal handler for graceful shutdown
def signal_handler(sig, frame):
    logger.info('Signal received, shutting down gracefully...')
    
    # Stop video buffering
    if hasattr(microscope, 'frame_acquisition_running') and microscope.frame_acquisition_running:
        logger.info('Stopping video buffering...')
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                loop.create_task(microscope.stop_video_buffering())
            else:
                loop.run_until_complete(microscope.stop_video_buffering())
        except Exception as e:
            logger.error(f'Error stopping video buffering: {e}')
    
    microscope.squidController.close()
    sys.exit(0)

# Register the signal handler for SIGINT and SIGTERM
signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)

def main():
    """Main entry point for the microscope service"""
    parser = argparse.ArgumentParser(
        description="Squid microscope control services for Hypha."
    )
    parser.add_argument(
        "--simulation",
        dest="simulation",
        action="store_true",
        default=True,
        help="Run in simulation mode (default: True)"
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

    microscope = Microscope(is_simulation=args.simulation, is_local=args.local)

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