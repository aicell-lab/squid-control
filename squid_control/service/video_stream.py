import asyncio
import fractions
import json
import logging
import time

import cv2  # noqa: F401
import numpy as np  # noqa: F401
from aiortc import MediaStreamTrack
from av import VideoFrame

from squid_control.utils.logging_utils import setup_logging

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
