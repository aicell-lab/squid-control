"""Image acquisition and autofocus mixin for SquidController."""
import asyncio
import logging

import cv2
import numpy as np

from squid_control.hardware.config import CONFIG, SIMULATED_CAMERA
from squid_control.utils.image_processing import rotate_and_flip_image

logger = logging.getLogger('squid_controller')


class AcquisitionMixin:
    """Mixin providing image acquisition and autofocus for SquidController."""

    async def send_trigger_simulation(self, channel=0, intensity=100, exposure_time=100):
        logger.debug('Getting simulated image')
        current_x, current_y, current_z, *_ = self.navigationController.update_pos(microcontroller=self.microcontroller)
        self.dz = current_z - SIMULATED_CAMERA.ORIN_Z
        self.current_channel = channel
        magnification_factor = SIMULATED_CAMERA.MAGNIFICATION_FACTOR
        self.current_exposure_time = exposure_time
        self.current_intensity = intensity
        await self.camera.send_trigger(current_x, current_y, self.dz, self.pixel_size_xy, channel, intensity, exposure_time, magnification_factor)
        logger.debug(f'For simulated camera, exposure_time={exposure_time}, intensity={intensity}, magnification_factor={magnification_factor}, current position: {current_x},{current_y},{current_z}')

    async def contrast_autofocus(self):

        if self.is_simulation:
            await self.contrast_autofocus_simulation()
        else:
            self.autofocusController.set_deltaZ(3.048)
            self.autofocusController.set_N(19)
            self.autofocusController.autofocus()
            self.autofocusController.wait_till_autofocus_has_completed()

    async def contrast_autofocus_simulation(self):

        random_z = SIMULATED_CAMERA.ORIN_Z + np.random.normal(0,0.001)
        self.navigationController.move_z_to(random_z)
        await self.send_trigger_simulation(self.current_channel, self.current_intensity, self.current_exposure_time)

    def init_laser_autofocus(self):
        self.laserAutofocusController.initialize_auto()

    async def reflection_autofocus(self):
        if self.is_simulation:
            await self.contrast_autofocus_simulation()
        else:
            # Check if focus map is enabled and use it if available
            if self.autofocusController.use_focus_map:
                # Use focus map interpolation for faster focusing
                self.autofocusController.autofocus(focus_map_override=False)
                self.autofocusController.wait_till_autofocus_has_completed()
            else:
                # Use laser autofocus as normal
                self.laserAutofocusController.move_to_target(0)

    def measure_displacement(self):
        self.laserAutofocusController.measure_displacement()

    async def snap_image(self, channel=0, intensity=100, exposure_time=100, full_frame=False):
        # Check if illumination was already on before we start
        illumination_was_already_on = self.liveController.illumination_on

        # Turn off illumination if it's on to ensure clean state
        if illumination_was_already_on:
            self.liveController.turn_off_illumination()
            while self.microcontroller.is_busy():
                await asyncio.sleep(0.005)

        # Set up camera and illumination for the photo
        self.camera.set_exposure_time(exposure_time)
        self.liveController.set_illumination(channel, intensity)
        self.liveController.turn_on_illumination()
        while self.microcontroller.is_busy():
            await asyncio.sleep(0.005)

        # Take the photo
        if self.is_simulation:
            await self.send_trigger_simulation(channel, intensity, exposure_time)
        else:
            self.camera.send_trigger()

        while self.microcontroller.is_busy():
            await asyncio.sleep(0.005)

        # Read the captured image (this blocks until frame is ready for real camera)
        # Run in executor to avoid blocking the async event loop
        loop = asyncio.get_event_loop()
        gray_img = await loop.run_in_executor(None, self.camera.read_frame)
        # Apply rotation and flip first
        gray_img = rotate_and_flip_image(gray_img, self.camera.rotate_image_angle, self.camera.flip_image)

        # In simulation mode, resize small images to expected camera resolution
        if self.is_simulation:
            height, width = gray_img.shape[:2]
            # If image is too small, resize it to expected camera dimensions
            expected_width = 3000  # Expected camera width
            expected_height = 3000  # Expected camera height
            if width < expected_width or height < expected_height:
                gray_img = cv2.resize(gray_img, (expected_width, expected_height), interpolation=cv2.INTER_LINEAR)

        # Return full frame if requested, otherwise crop using configuration settings
        if full_frame:
            result_img = gray_img
        else:
            # Crop using configuration-based dimensions with proper bounds checking
            crop_height = CONFIG.ACQUISITION.CROP_HEIGHT
            crop_width = CONFIG.ACQUISITION.CROP_WIDTH
            height, width = gray_img.shape[:2]
            start_x = width // 2 - crop_width // 2
            start_y = height // 2 - crop_height // 2

            # Add bounds checking
            start_x = max(0, start_x)
            start_y = max(0, start_y)
            end_x = min(width, start_x + crop_width)
            end_y = min(height, start_y + crop_height)

            result_img = gray_img[start_y:end_y, start_x:end_x]

        # Turn off illumination after taking the photo
        self.liveController.turn_off_illumination()
        while self.microcontroller.is_busy():
            await asyncio.sleep(0.005)

        # Restore illumination state if it was on before
        if illumination_was_already_on:
            self.liveController.turn_on_illumination()
            while self.microcontroller.is_busy():
                await asyncio.sleep(0.005)

        # Ensure the image is uint8 before returning
        if result_img.dtype != np.uint8:
            if result_img.dtype == np.uint16:
                # Scale 16-bit to 8-bit
                result_img = (result_img / 256).astype(np.uint8)
            else:
                result_img = result_img.astype(np.uint8)

        return result_img

    async def get_camera_frame_simulation(self, channel=0, intensity=100, exposure_time=100):
        self.camera.set_exposure_time(exposure_time)
        self.liveController.set_illumination(channel, intensity)
        await self.send_trigger_simulation(channel, intensity, exposure_time)
        gray_img = self.camera.read_frame()
        gray_img = rotate_and_flip_image(gray_img, self.camera.rotate_image_angle, self.camera.flip_image)

        # In simulation mode, resize small images to expected camera resolution
        height, width = gray_img.shape[:2]
        # If image is too small, resize it to expected camera dimensions
        expected_width = 3000  # Expected camera width
        expected_height = 3000  # Expected camera height
        if width < expected_width or height < expected_height:
            gray_img = cv2.resize(gray_img, (expected_width, expected_height), interpolation=cv2.INTER_LINEAR)

        # Ensure the image is uint8 before returning
        if gray_img.dtype != np.uint8:
            if gray_img.dtype == np.uint16:
                # Scale 16-bit to 8-bit
                gray_img = (gray_img / 256).astype(np.uint8)
            else:
                gray_img = gray_img.astype(np.uint8)

        return gray_img

    def get_camera_frame(self, channel=0, intensity=100, exposure_time=100):
        try:
            self.camera.send_trigger()
            gray_img = self.camera.read_frame()
            if gray_img is None:
                print(f"Warning: read_frame() returned None for channel {channel}")
                # Return a placeholder image instead of None to prevent crashes
                return np.zeros((self.camera.Height, self.camera.Width), dtype=np.uint8)
            gray_img = rotate_and_flip_image(gray_img, self.camera.rotate_image_angle, self.camera.flip_image)

            # Ensure the image is uint8 before returning
            if gray_img.dtype != np.uint8:
                if gray_img.dtype == np.uint16:
                    # Scale 16-bit to 8-bit
                    gray_img = (gray_img / 256).astype(np.uint8)
                else:
                    gray_img = gray_img.astype(np.uint8)

            return gray_img
        except Exception as e:
            print(f"Error in get_camera_frame: {e}")
            # Return a placeholder image on error
            return np.zeros((self.camera.Height, self.camera.Width), dtype=np.uint8)
