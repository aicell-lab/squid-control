"""
Toupcam Camera Driver for Squid Control System

This module provides a camera driver for Toupcam cameras, adapted to work with the 
squid-control system architecture. It interfaces with the toupcam library and provides
the same API as the default camera module.

Author: Adapted from official Squid software for squid-control
"""

import math
import time
import threading
from typing import Optional, Tuple, Dict
import numpy as np

# Import toupcam library from squid_control
import squid_control.control.camera.toupcam as toupcam
from squid_control.control.camera.toupcam_exceptions import hresult_checker
from squid_control.control.camera import TriggerModeSetting
from squid_control.control.config import CONFIG
from squid_control.utils.logging_utils import setup_logging

logger = setup_logging("camera_toupcam.log")


def get_sn_by_model(model_name):
    """
    Get camera serial number by model name.
    
    Args:
        model_name: The camera model name to search for
        
    Returns:
        Camera serial number (ID) if found, None otherwise
    """
    try:
        device_list = toupcam.Toupcam.EnumV2()
    except:
        logger.error("Problem generating Toupcam device list")
        return None
    
    for dev in device_list:
        if model_name in dev.displayname:
            return dev.id
    
    return None  # return None if no device with the specified model_name is connected


class Camera(object):
    """
    Toupcam Camera driver adapted for squid-control system.
    
    This class provides the same interface as the default Camera class but uses
    Toupcam hardware instead of Daheng/gxipy cameras.
    """
    
    def __init__(
        self, sn=None, is_global_shutter=False, rotate_image_angle=None, flip_image=None
    ):
        """
        Initialize Toupcam camera driver.
        
        Args:
            sn: Camera serial number (optional, will open first camera if None)
            is_global_shutter: Whether camera has global shutter (unused for Toupcam)
            rotate_image_angle: Angle to rotate acquired images
            flip_image: Whether to flip images
        """
        logger.info(f"Initializing Toupcam camera with SN={sn}")
        
        # Camera hardware
        self.sn = sn
        self.camera = None
        self._capabilities = None
        self.is_global_shutter = is_global_shutter
        
        # Image transformation parameters
        self.rotate_image_angle = rotate_image_angle
        self.flip_image = flip_image
        
        # Camera state
        self.exposure_time = 1  # unit: ms
        self.analog_gain = 0
        self.frame_ID = -1
        self.frame_ID_software = -1
        self.frame_ID_offset_hardware_trigger = 0
        self.timestamp = 0
        
        # Frame management
        self.image_locked = False
        self.current_frame = None
        self._internal_read_buffer = None
        
        # Callback management
        self.callback_is_enabled = False
        self.is_streaming = False
        self.new_image_callback_external = None
        self._raw_camera_stream_started = False
        self._raw_frame_callback_lock = threading.Lock()
        self._trigger_sent = False
        self._last_trigger_timestamp = 0
        
        # Frame synchronization for blocking read_frame()
        self._frame_ready_event = threading.Event()
        self._last_read_frame_id = -1
        
        # Camera limits
        self.GAIN_MAX = 40  # User gain range 0-40
        self.GAIN_MIN = 0
        self.GAIN_STEP = 0.01
        self.EXPOSURE_TIME_MS_MIN = 0.01
        self.EXPOSURE_TIME_MS_MAX = 4000
        
        # Trigger mode
        self.trigger_mode = None
        self.pixel_size_byte = 2  # Default to 16-bit
        
        # Toupcam specific parameters
        self._binning = (1, 1)  # Default binning
        self._pixel_format = "MONO16"
        
        # Live mode flag (for compatibility with original camera)
        self.is_live = False
        
        # Strobe timing info (calculated after camera opens)
        self._strobe_info = None
        
        # Temperature monitoring
        self.temperature_reading_callback = None
        self.terminate_read_temperature_thread = False
        self.thread_read_temperature = None
        
        # Camera dimensions (will be set after opening)
        self.Width = 0
        self.Height = 0
        self.WidthMax = 0
        self.HeightMax = 0
        self.OffsetX = 0
        self.OffsetY = 0
        
        # Camera color info
        self.is_color = False
        
        self.is_live = False

    def open(self, index=0):
        """
        Open the camera and initialize it.
        
        Args:
            index: Camera index (0 for first camera)
        """
        logger.info(f"Opening Toupcam with index={index}, sn={self.sn}")
        
        try:
            # Enumerate available cameras
            devices = toupcam.Toupcam.EnumV2()
            if len(devices) <= 0:
                raise RuntimeError("No Toupcam devices found! Is the camera connected and powered on?")
            
            # Log available cameras
            for idx, device in enumerate(devices):
                logger.info(
                    f"Camera {idx}: {device.displayname}: flag = {device.model.flag:#x}, "
                    f"preview = {device.model.preview}, still = {device.model.still}"
                )
            
            # Determine which camera to open
            device_to_open = None
            if self.sn is not None:
                # Find camera by serial number
                for idx, device in enumerate(devices):
                    if device.id == self.sn:
                        device_to_open = device
                        break
                if device_to_open is None:
                    all_sns = [d.id for d in devices]
                    raise RuntimeError(f"Could not find camera with SN={self.sn}, available: {all_sns}")
            else:
                # Use camera at specified index
                if index >= len(devices):
                    raise RuntimeError(f"Camera index {index} out of range (have {len(devices)} cameras)")
                device_to_open = devices[index]
            
            # Build capabilities from device info
            self._capabilities = self._build_capabilities(device_to_open)
            
            # Open the camera
            self.camera = toupcam.Toupcam.Open(device_to_open.id)
            logger.info(f"Successfully opened camera: {device_to_open.displayname}")
            
            # Configure camera to initial state
            self._configure_camera()
            
            # Get camera dimensions
            self.Width, self.Height = self.camera.get_Size()
            self.WidthMax = self._capabilities['max_resolution'][0]
            self.HeightMax = self._capabilities['max_resolution'][1]
            self.OffsetX, self.OffsetY, _, _ = self.camera.get_Roi()
            
            # Determine if camera is color
            self.is_color = (device_to_open.model.flag & toupcam.TOUPCAM_FLAG_MONO) == 0
            
            # Start temperature monitoring thread
            if self._capabilities['has_TEC']:
                self.thread_read_temperature = threading.Thread(target=self._check_temperature, daemon=True)
                self.thread_read_temperature.start()
            
            # CRITICAL: Start the raw camera stream to enable frame acquisition
            # This must be done after configuration to receive frame callbacks
            self._start_raw_camera_stream()
            
            # Update internal settings to prepare buffers and exposure
            self._update_internal_settings()
            
            logger.info(f"Camera initialization complete. Resolution: {self.Width}x{self.Height}, streaming started")
            
        except Exception as e:
            logger.error(f"Failed to open camera: {e}")
            raise
    
    def _start_raw_camera_stream(self):
        """
        Start the raw camera stream for frame acquisition.
        
        This must be called after the camera is opened and configured.
        It sets up the callback to receive frames from the camera.
        """
        if self.camera is None:
            logger.error("Cannot start raw stream: camera not opened")
            return
            
        if not self._raw_camera_stream_started:
            try:
                logger.debug("Starting raw stream in PullModeWithCallback")
                self.camera.StartPullModeWithCallback(self._event_callback, self)
                self._raw_camera_stream_started = True
                logger.info("Raw camera stream started successfully")
            except toupcam.HRESULTException as ex:
                self._raw_camera_stream_started = False
                logger.error(f"Failed to start raw camera stream: {ex}")
                raise
    
    def _update_internal_settings(self):
        """
        Update internal buffer and settings based on current camera configuration.
        
        This prepares the camera for frame acquisition by:
        - Creating the internal read buffer
        - Setting the exposure time
        - Setting the analog gain
        """
        if self.camera is None:
            logger.warning("Cannot update settings: camera not opened")
            return
            
        try:
            # Update internal buffer size based on current ROI and pixel format
            self._update_internal_buffer()
            
            # Set initial exposure time (convert ms to us)
            exposure_time_us = int(self.exposure_time * 1000)
            self.camera.put_ExpoTime(exposure_time_us)
            logger.debug(f"Exposure time set to {self.exposure_time}ms")
            
            # Set initial analog gain if it's been set
            if hasattr(self, 'analog_gain') and self.analog_gain > 0:
                self.set_analog_gain(self.analog_gain)
                
            logger.debug(f"Internal settings updated: buffer ready, exposure={self.exposure_time}ms")
            
        except Exception as e:
            logger.error(f"Failed to update internal settings: {e}")
            # Don't raise - let camera continue with defaults
    
    def _build_capabilities(self, device):
        """
        Build camera capabilities dictionary from device info.
        
        Args:
            device: Toupcam device info object
            
        Returns:
            Dictionary of camera capabilities
        """
        # Build resolution list
        resolution_list = []
        for r in device.model.res:
            resolution_list.append((r.width, r.height))
        
        if len(resolution_list) == 0:
            raise ValueError("No resolutions found for camera")
        
        resolution_list.sort(key=lambda x: x[0] * x[1], reverse=True)
        highest_res = resolution_list[0]
        
        # Build binning to resolution mapping
        binning_res = {}
        for res in resolution_list:
            x_binning = int(highest_res[0] / res[0])
            y_binning = int(highest_res[1] / res[1])
            binning_res[(x_binning, y_binning)] = res
        
        capabilities = {
            'binning_to_resolution': binning_res,
            'max_resolution': highest_res,
            'has_fan': (device.model.flag & toupcam.TOUPCAM_FLAG_FAN) > 0,
            'has_TEC': (device.model.flag & toupcam.TOUPCAM_FLAG_TEC_ONOFF) > 0,
            'has_low_noise_mode': (device.model.flag & toupcam.TOUPCAM_FLAG_LOW_NOISE) > 0,
            'has_black_level': (device.model.flag & toupcam.TOUPCAM_FLAG_BLACKLEVEL) > 0,
        }
        
        logger.info(f"Camera capabilities: {capabilities}")
        return capabilities
    
    def _configure_camera(self):
        """
        Configure camera to initial state using CONFIG values.
        """
        try:
            # Set low noise mode if available
            if self._capabilities['has_low_noise_mode']:
                self.camera.put_Option(toupcam.TOUPCAM_OPTION_LOW_NOISE, 0)
            
            # Set temperature and fan if available
            if self._capabilities['has_TEC']:
                try:
                    fan_speed = getattr(CONFIG.CAMERA_CONFIG, 'FAN_SPEED_DEFAULT', 1)
                    temp = getattr(CONFIG.CAMERA_CONFIG, 'TEMPERATURE_DEFAULT', 20)
                    self._set_fan_speed(fan_speed)
                    self.set_temperature(temp)
                except Exception as e:
                    logger.warning(f"Failed to configure temperature/fan (continuing anyway): {e}")
            
            # Set frame format to RAW
            self.camera.put_Option(toupcam.TOUPCAM_OPTION_RAW, 1)  # 1 = RAW mode
            
            # Set pixel format from CONFIG
            pixel_format = getattr(CONFIG.CAMERA_CONFIG, 'PIXEL_FORMAT_DEFAULT', 'MONO16')
            self._set_pixel_format(pixel_format)
            
            # Set black level if available
            if self._capabilities['has_black_level']:
                black_level = getattr(CONFIG.CAMERA_CONFIG, 'BLACKLEVEL_VALUE_DEFAULT', 3)
                self._set_black_level(black_level)
            
            # Set binning from CONFIG
            # This automatically sets the resolution to match the binning level
            # binning_factor_default is the actual binning factor: 1=no binning, 2=2x2, 4=4x4, etc.
            binning_factor = getattr(CONFIG.CAMERA_CONFIG, 'BINNING_FACTOR_DEFAULT', 2)
            self.set_binning(binning_factor)
            
            # NOTE: ROI setting is intentionally skipped here to match official Squid software behavior.
            # Setting ROI after binning causes errors because ROI dimensions from config are for unbinned resolution.
            # The binning operation already sets the camera to the correct resolution for the binning level.
            # Hardware cropping (ROI) should be implemented separately if needed, with binning-aware calculations.
            
            logger.info("Camera configuration complete")
            
        except Exception as e:
            logger.error(f"Error configuring camera: {e}")
            raise
    
    def _check_temperature(self):
        """Background thread to monitor camera temperature."""
        while not self.terminate_read_temperature_thread:
            time.sleep(2)
            try:
                temperature = self.get_temperature()
                if self.temperature_reading_callback is not None:
                    self.temperature_reading_callback(temperature)
            except Exception as e:
                logger.error(f"Temperature monitoring error: {e}")
    
    def close(self):
        """Close the camera and cleanup resources."""
        logger.info("Closing camera")
        
        # Stop temperature monitoring
        if self.thread_read_temperature is not None:
            self.terminate_read_temperature_thread = True
            self.thread_read_temperature.join(timeout=5)
        
        # Turn off fan
        if self.camera and self._capabilities and self._capabilities['has_fan']:
            try:
                self._set_fan_speed(0)
            except:
                pass
        
        # Close camera
        if self.camera:
            try:
                if self.is_streaming:
                    self.stop_streaming()
                self.camera.Close()
            except Exception as e:
                logger.error(f"Error closing camera: {e}")
        
        self.camera = None
        self.current_frame = None

    def set_callback(self, function):
        """Set external callback function for new frames."""
        self.new_image_callback_external = function

    def enable_callback(self):
        """Enable callback mode for frame acquisition."""
        if not self.callback_is_enabled:
            # Stop streaming if active
            was_streaming = self.is_streaming
            if was_streaming:
                self.stop_streaming()
            
            # Enable callback
            try:
                self.camera.StartPullModeWithCallback(self._event_callback, self)
                self.callback_is_enabled = True
                self._raw_camera_stream_started = True
                logger.info("Callback mode enabled")
            except toupcam.HRESULTException as ex:
                logger.error(f"Failed to enable callback: {ex}")
                raise
            
            # Resume streaming if it was active
            if was_streaming:
                self.is_streaming = True

    def disable_callback(self):
        """Disable callback mode."""
        if self.callback_is_enabled:
            was_streaming = self.is_streaming
            if was_streaming:
                self.stop_streaming()
            
            self.callback_is_enabled = False
            self._raw_camera_stream_started = False
            
            if was_streaming:
                self.start_streaming()

    @staticmethod
    def _event_callback(event_number, camera):
        """
        Static callback function called by toupcam library.
        
        Args:
            event_number: Event type from toupcam
            camera: Camera instance (self)
        """
        if event_number == toupcam.TOUPCAM_EVENT_IMAGE:
            camera._on_frame_callback()

    def _on_frame_callback(self):
        """Handle new frame from camera."""
        with self._raw_frame_callback_lock:
            self._raw_camera_stream_started = True
            self._trigger_sent = False
            
            try:
                # Pull image from camera
                pixel_size_bits = self._get_pixel_size_in_bytes() * 8
                self.camera.PullImageV2(self._internal_read_buffer, pixel_size_bits, None)
                
                # Update frame ID and timestamp
                self.frame_ID += 1
                self.timestamp = time.time()
                
                # Get ROI dimensions
                _, _, width, height = self.camera.get_Roi()
                
                # Convert buffer to numpy array
                if self._get_pixel_size_in_bytes() == 1:
                    raw_image = np.frombuffer(self._internal_read_buffer, dtype="uint8")
                elif self._get_pixel_size_in_bytes() == 2:
                    raw_image = np.frombuffer(self._internal_read_buffer, dtype="uint16")
                
                current_image = raw_image.reshape(height, width)
                
                # Apply image transformations
                if self.rotate_image_angle is not None and self.rotate_image_angle != 0:
                    # OpenCV rotation (k=1 means 90° CCW, k=2 means 180°, k=3 means 270° CCW)
                    k = int(self.rotate_image_angle / 90)
                    current_image = np.rot90(current_image, k)
                
                if self.flip_image:
                    current_image = np.fliplr(current_image)
                
                self.current_frame = current_image
                
                # Signal that a new frame is ready for read_frame()
                self._frame_ready_event.set()
                
                # Call external callback if set and enabled
                # Note: Toupcam doesn't use is_live flag - callback fires for all frames when enabled
                if self.new_image_callback_external is not None and self.callback_is_enabled:
                    try:
                        self.new_image_callback_external(self)  # Pass self (camera object) like default camera does
                    except Exception as e:
                        logger.error(f"External callback error: {e}")
                
            except toupcam.HRESULTException as ex:
                logger.error(f"Pull image failed: {ex}")
            except Exception as e:
                logger.error(f"Frame callback error: {e}")

    def start_streaming(self):
        """
        Start camera streaming.
        
        Note: If a callback has been set via set_callback(), it will be automatically
        enabled when streaming starts. This matches the Toupcam usage pattern.
        """
        if not self._raw_camera_stream_started:
            try:
                self.camera.StartPullModeWithCallback(self._event_callback, self)
                self._raw_camera_stream_started = True
                self.is_streaming = True
                
                # Auto-enable callback if one has been set
                if self.new_image_callback_external is not None:
                    self.callback_is_enabled = True
                    logger.info("Streaming started with callback enabled")
                else:
                    logger.info("Streaming started")
                    
            except toupcam.HRESULTException as ex:
                logger.error(f"Failed to start streaming: {ex}")
                raise
        else:
            self.is_streaming = True

    def stop_streaming(self):
        """Stop camera streaming."""
        if self._raw_camera_stream_started:
            try:
                self.camera.Stop()
                self._raw_camera_stream_started = False
                self.is_streaming = False
                logger.info("Streaming stopped")
            except Exception as e:
                logger.error(f"Error stopping streaming: {e}")

    def set_exposure_time(self, exposure_time):
        """
        Set camera exposure time.
        
        Args:
            exposure_time: Exposure time in milliseconds
        """
        self.exposure_time = exposure_time
        
        # Toupcam expects exposure time in microseconds
        exposure_time_us = int(exposure_time * 1000)
        
        try:
            self.camera.put_ExpoTime(exposure_time_us)
            logger.debug(f"Exposure time set to {exposure_time} ms")
        except toupcam.HRESULTException as ex:
            logger.error(f"Failed to set exposure time: {ex}")
            raise

    def set_analog_gain(self, analog_gain):
        """
        Set camera analog gain.
        
        Args:
            analog_gain: Gain value (0-40 user range)
        """
        self.analog_gain = analog_gain
        
        # Convert user gain (0-40) to toupcam gain (100-10000)
        toupcam_gain = int(100 * (10 ** (analog_gain / 20)))
        
        try:
            self.camera.put_ExpoAGain(toupcam_gain)
            logger.debug(f"Analog gain set to {analog_gain} (toupcam: {toupcam_gain})")
        except toupcam.HRESULTException as ex:
            logger.error(f"Failed to set analog gain: {ex}")
            raise

    def get_temperature(self):
        """Get camera sensor temperature in Celsius."""
        try:
            return self.camera.get_Temperature() / 10
        except toupcam.HRESULTException as ex:
            logger.error(f"Failed to get temperature: {ex}")
            return None

    def set_temperature(self, temperature):
        """
        Set camera target temperature.
        
        Args:
            temperature: Target temperature in Celsius
        """
        try:
            self.camera.put_Temperature(int(temperature * 10))
            logger.info(f"Target temperature set to {temperature}°C")
        except toupcam.HRESULTException as ex:
            logger.error(f"Failed to set temperature: {ex}")
            raise

    def _set_fan_speed(self, speed):
        """Set camera fan speed (0-3)."""
        try:
            self.camera.put_Option(toupcam.TOUPCAM_OPTION_FAN, speed)
        except toupcam.HRESULTException as ex:
            logger.error(f"Failed to set fan speed: {ex}")

    def _set_pixel_format(self, pixel_format):
        """
        Set camera pixel format.
        
        Args:
            pixel_format: One of 'MONO8', 'MONO12', 'MONO14', 'MONO16'
        """
        self._pixel_format = pixel_format
        
        try:
            if pixel_format == "MONO8":
                self.camera.put_Option(toupcam.TOUPCAM_OPTION_BITDEPTH, 0)
                self.pixel_size_byte = 1
            elif pixel_format in ["MONO12", "MONO14", "MONO16"]:
                self.camera.put_Option(toupcam.TOUPCAM_OPTION_BITDEPTH, 1)
                self.pixel_size_byte = 2
            else:
                raise ValueError(f"Unsupported pixel format: {pixel_format}")
            
            logger.info(f"Pixel format set to {pixel_format}")
            self._update_internal_buffer()
            
        except toupcam.HRESULTException as ex:
            logger.error(f"Failed to set pixel format: {ex}")
            raise

    def _set_black_level(self, black_level):
        """Set camera black level."""
        if not self._capabilities['has_black_level']:
            logger.warning("Camera does not support black level adjustment")
            return
        
        # Get black level factor based on pixel format
        factor = self._get_black_level_factor()
        raw_black_level = int(black_level * factor)
        
        try:
            self.camera.put_Option(toupcam.TOUPCAM_OPTION_BLACKLEVEL, raw_black_level)
            logger.info(f"Black level set to {black_level} (raw: {raw_black_level})")
        except toupcam.HRESULTException as ex:
            logger.error(f"Failed to set black level: {ex}")

    def _get_black_level_factor(self):
        """Get black level multiplication factor based on pixel format."""
        format_factors = {
            "MONO8": 1,
            "MONO12": 16,
            "MONO14": 64,
            "MONO16": 256
        }
        return format_factors.get(self._pixel_format, 1)

    def _get_pixel_size_in_bytes(self):
        """Get pixel size in bytes based on current pixel format."""
        if self._pixel_format == "MONO8":
            return 1
        else:  # MONO12, MONO14, MONO16
            return 2

    def set_binning(self, binning_factor):
        """
        Set camera binning.
        
        Args:
            binning_factor: Binning factor (1, 2, 4, etc.)
        """
        binning_tuple = (binning_factor, binning_factor)
        
        if binning_tuple not in self._capabilities['binning_to_resolution']:
            logger.error(f"Binning {binning_factor} not supported")
            raise ValueError(f"Binning {binning_factor} not supported by camera")
        
        width, height = self._capabilities['binning_to_resolution'][binning_tuple]
        self._binning = binning_tuple
        
        try:
            self.camera.put_Size(width, height)
            self.Width = width
            self.Height = height
            logger.info(f"Binning set to {binning_factor} -> {width}x{height}")
            self._update_internal_buffer()
        except toupcam.HRESULTException as ex:
            logger.error(f"Failed to set binning: {ex}")
            raise

    def set_ROI(self, offset_x, offset_y, width, height):
        """
        Set camera region of interest.
        
        Args:
            offset_x: X offset in pixels (must be even)
            offset_y: Y offset in pixels (must be even)
            width: ROI width in pixels (must be even)
            height: ROI height in pixels (must be even)
        """
        # Toupcam requires even values
        offset_x = (offset_x // 2) * 2
        offset_y = (offset_y // 2) * 2
        width = (width // 2) * 2
        height = (height // 2) * 2
        
        try:
            self.camera.put_Roi(offset_x, offset_y, width, height)
            self.OffsetX = offset_x
            self.OffsetY = offset_y
            self.Width = width
            self.Height = height
            logger.info(f"ROI set to {offset_x},{offset_y} size {width}x{height}")
            self._update_internal_buffer()
        except toupcam.HRESULTException as ex:
            logger.error(f"Failed to set ROI: {ex}")

    def _update_internal_buffer(self):
        """Update internal read buffer based on current camera settings."""
        _, _, width, height = self.camera.get_Roi()
        pixel_size = self._get_pixel_size_in_bytes()
        buffer_size = width * pixel_size * height
        self._internal_read_buffer = bytes(buffer_size)
        logger.debug(f"Internal buffer updated: {buffer_size} bytes for {width}x{height}")

    def set_continuous_acquisition(self):
        """Set camera to continuous acquisition mode."""
        try:
            self.camera.put_Option(toupcam.TOUPCAM_OPTION_TRIGGER, 0)
            self.trigger_mode = TriggerModeSetting.CONTINUOUS
            logger.info("Continuous acquisition mode set")
        except toupcam.HRESULTException as ex:
            logger.error(f"Failed to set continuous mode: {ex}")
            raise

    def set_software_triggered_acquisition(self):
        """Set camera to software trigger mode."""
        try:
            self.camera.put_Option(toupcam.TOUPCAM_OPTION_TRIGGER, 1)
            self.trigger_mode = TriggerModeSetting.HARDWARE  # Using HARDWARE enum but software trigger
            logger.info("Software trigger mode set")
        except toupcam.HRESULTException as ex:
            logger.error(f"Failed to set software trigger mode: {ex}")
            raise

    def send_trigger(self):
        """Send software trigger to camera."""
        if not self.get_ready_for_trigger():
            logger.warning("Camera not ready for trigger, waiting...")
            return False
        
        try:
            # Clear the frame ready event before triggering
            self._frame_ready_event.clear()
            
            self.camera.Trigger(1)
            self._trigger_sent = True
            self._last_trigger_timestamp = time.time()
            logger.debug("Software trigger sent")
            return True
        except toupcam.HRESULTException as ex:
            logger.error(f"Failed to send trigger: {ex}")
            return False

    def get_ready_for_trigger(self):
        """Check if camera is ready for next trigger."""
        # Simple check: if trigger was sent, not ready yet
        # Camera callback will clear _trigger_sent when frame arrives
        return not self._trigger_sent

    @property
    def pixel_format(self):
        """Get the current pixel format."""
        return self._pixel_format

    @pixel_format.setter
    def pixel_format(self, value):
        """Set the pixel format."""
        self._set_pixel_format(value)

    def read_frame(self):
        """
        Read the current frame from the camera.
        
        This method blocks until a new frame is available after send_trigger(),
        matching the behavior of the original Daheng camera implementation.
        
        The old camera's get_image() blocks indefinitely until the frame is ready.
        We do the same here - just wait for the frame ready event without timeout.
        
        Returns:
            numpy array with the current image
        """
        # Wait for new frame to be ready (blocks indefinitely like old camera)
        self._frame_ready_event.wait()
        return self.current_frame


# Simulation camera class
class Camera_Simulation(object):
    """Simulation camera for testing without hardware."""
    
    def __init__(self, rotate_image_angle=None, flip_image=None):
        """Initialize simulation camera."""
        logger.info("Initializing Toupcam simulation camera")
        
        # Use the default camera simulation
        from squid_control.control.camera.camera_default import Camera_Simulation as DefaultSimulation
        self._sim = DefaultSimulation(rotate_image_angle=rotate_image_angle, flip_image=flip_image)
        
        # Expose all methods from default simulation
        for attr in dir(self._sim):
            if not attr.startswith('_') and callable(getattr(self._sim, attr)):
                setattr(self, attr, getattr(self._sim, attr))
        
        # Copy important attributes
        self.Width = getattr(self._sim, 'Width', 3104)
        self.Height = getattr(self._sim, 'Height', 2084)
        self.WidthMax = getattr(self._sim, 'WidthMax', 3104)
        self.HeightMax = getattr(self._sim, 'HeightMax', 2084)
        self.is_color = False
        self.is_streaming = False
        self.callback_is_enabled = False
        
        # Copy rotation and flip attributes
        self.rotate_image_angle = rotate_image_angle
        self.flip_image = flip_image

    @property
    def pixel_format(self):
        """Get the current pixel format."""
        return getattr(self._sim, 'pixel_format', 'MONO8')

    @pixel_format.setter
    def pixel_format(self, value):
        """Set the pixel format."""
        if hasattr(self._sim, 'pixel_format'):
            self._sim.pixel_format = value

