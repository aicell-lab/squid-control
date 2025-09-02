import asyncio
import importlib.util
import json
import math
import os
import threading
import time
from datetime import datetime
from importlib import import_module
from pathlib import Path
from queue import Queue
from typing import Callable

import cv2
import imageio as iio
import numpy as np
import pandas as pd
import scipy
import scipy.signal
from lxml import etree as ET

from squid_control.control import utils, utils_config
from squid_control.control.camera import TriggerModeSetting
from squid_control.control.config import CONFIG
from squid_control.control.microcontroller import LIMIT_CODE
from squid_control.control.processing_handler import ProcessingHandler


class EventEmitter:
    def __init__(self):
        self._callbacks = {}

    def connect(self, event_name: str, callback: Callable):
        if event_name not in self._callbacks:
            self._callbacks[event_name] = []
        self._callbacks[event_name].append(callback)

    def emit(self, event_name: str, *args, **kwargs):
        if event_name in self._callbacks:
            for callback in self._callbacks[event_name]:
                try:
                    callback(*args, **kwargs)
                except Exception as e:
                    print(f"Error in callback for {event_name}: {e}")

    def disconnect(self, event_name: str, callback: Callable = None):
        if event_name in self._callbacks:
            if callback:
                try:
                    self._callbacks[event_name].remove(callback)
                except ValueError:
                    pass
            else:
                self._callbacks[event_name].clear()


def _load_multipoint_function(module_path, entrypoint):
    if module_path.endswith(".py"):
        spec = importlib.util.spec_from_file_location("user_module", module_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    else:
        module = import_module(module_path)

    return getattr(module, entrypoint)

def load_multipoint_custom_script(startup_function_uri: str):
    parts = startup_function_uri.split(":")
    module_path = parts[0]
    assert not module_path.endswith(".py") or os.path.exists(
        module_path
    ), f"Module {module_path} does not exist"
    entrypoint = parts[1] if len(parts) > 1 else None
    assert (
        entrypoint
    ), f"Entrypoint is required for {startup_function_uri}, please use {startup_function_uri}:entrypoint_function"

    # load the python module and get the entrypoint
    load_func = _load_multipoint_function(module_path, entrypoint)

    print(f"Successfully executed the startup function: {startup_function_uri}")
    return load_func

class ObjectiveStore:
    def __init__(
        self,
        objectives_dict=None,
        default_objective=None,
    ):
        # Get current CONFIG values at runtime instead of at class definition time
        self.objectives_dict = objectives_dict if objectives_dict is not None else CONFIG.OBJECTIVES
        self.default_objective = default_objective if default_objective is not None else CONFIG.DEFAULT_OBJECTIVE
        self.current_objective = self.default_objective


class StreamHandler:
    """Handle image streams with callback-based events"""

    def __init__(
        self,
        crop_width=CONFIG.Acquisition.CROP_WIDTH,
        crop_height=CONFIG.Acquisition.CROP_HEIGHT,
        display_resolution_scaling=1,
    ):
        self.events = EventEmitter()
        self.fps_display = 1
        self.fps_save = 1
        self.fps_track = 1
        self.timestamp_last_display = 0
        self.timestamp_last_save = 0
        self.timestamp_last_track = 0

        self.crop_width = crop_width
        self.crop_height = crop_height
        self.display_resolution_scaling = display_resolution_scaling

        self.save_image_flag = False
        self.track_flag = False
        self.handler_busy = False

        # for fps measurement
        self.timestamp_last = 0
        self.counter = 0
        self.fps_real = 0
        # Direct callback for WebRTC
        self.webrtc_frame_callback = None
        self.general_frame_callback = None # New callback for general frame updates

        # Callback functions
        self.image_to_display_callback = None
        self.packet_image_to_write_callback = None
        self.packet_image_for_tracking_callback = None

    def connect_image_to_display(self, callback):
        self.image_to_display_callback = callback

    def connect_packet_image_to_write(self, callback):
        self.packet_image_to_write_callback = callback

    def connect_packet_image_for_tracking(self, callback):
        self.packet_image_for_tracking_callback = callback

    def start_recording(self):
        self.save_image_flag = True

    def stop_recording(self):
        self.save_image_flag = False

    def start_tracking(self):
        self.tracking_flag = True

    def stop_tracking(self):
        self.tracking_flag = False

    def set_display_fps(self, fps):
        self.fps_display = fps

    def set_save_fps(self, fps):
        self.fps_save = fps

    def set_crop(self, crop_width, crop_height):
        self.crop_width = crop_width
        self.crop_height = crop_height

    def set_display_resolution_scaling(self, display_resolution_scaling):
        self.display_resolution_scaling = display_resolution_scaling / 100
        print(self.display_resolution_scaling)

    def set_webrtc_frame_callback(self, callback):
        """Set a direct callback for WebRTC frame handling"""
        self.webrtc_frame_callback = callback

    def remove_webrtc_frame_callback(self):
        """Remove the WebRTC frame callback"""
        self.webrtc_frame_callback = None

    def set_general_frame_callback(self, callback):
        """Set a direct callback for general frame updates"""
        self.general_frame_callback = callback

    def remove_general_frame_callback(self):
        """Remove the general frame callback"""
        self.general_frame_callback = None

    def on_new_frame(self, camera):

        if camera.is_live:

            camera.image_locked = True
            self.handler_busy = True

            # measure real fps
            timestamp_now = round(time.time())
            if timestamp_now == self.timestamp_last:
                self.counter = self.counter + 1
            else:
                self.timestamp_last = timestamp_now
                self.fps_real = self.counter
                self.counter = 0
                print("real camera fps is " + str(self.fps_real))

            # crop image
            image_cropped = utils.crop_image(
                camera.current_frame, self.crop_width, self.crop_height
            )
            image_cropped = np.squeeze(image_cropped)

            # rotate and flip
            image_cropped = utils.rotate_and_flip_image(
                image_cropped,
                rotate_image_angle=camera.rotate_image_angle,
                flip_image=camera.flip_image,
            )

            # Send the raw cropped frame for WebRTC using direct callback instead of signal
            if self.webrtc_frame_callback is not None:
                try:
                    self.webrtc_frame_callback(image_cropped.copy())  # Send a copy for safety
                except Exception as e:
                    print(f"Error in WebRTC frame callback: {e}")

            # send image to display
            time_now = time.time()
            if time_now - self.timestamp_last_display >= 1 / self.fps_display:
                display_image = utils.crop_image(
                    image_cropped,
                    round(self.crop_width * self.display_resolution_scaling),
                    round(self.crop_height * self.display_resolution_scaling),
                )
                if self.image_to_display_callback:
                    self.image_to_display_callback(display_image)
                self.timestamp_last_display = time_now

            # send image to write
            if (
                self.save_image_flag
                and time_now - self.timestamp_last_save >= 1 / self.fps_save
            ):
                if camera.is_color:
                    image_cropped = cv2.cvtColor(image_cropped, cv2.COLOR_RGB2BGR)
                if self.packet_image_to_write_callback:
                    self.packet_image_to_write_callback(
                        image_cropped, camera.frame_ID, camera.timestamp
                    )
                self.timestamp_last_save = time_now

            # Call the general frame callback if it's set
            if self.general_frame_callback:
                try:
                    self.general_frame_callback(image_cropped, camera.frame_ID, camera.timestamp)
                except Exception as e:
                    print(f"Error in general frame callback: {e}")

            # send image to track
            if (
                self.track_flag
                and time_now - self.timestamp_last_track >= 1 / self.fps_track
            ):
                if self.packet_image_for_tracking_callback:
                    self.packet_image_for_tracking_callback(
                        image_cropped, camera.frame_ID, camera.timestamp
                    )
                self.timestamp_last_track = time_now

            self.handler_busy = False
            camera.image_locked = False

class ImageSaver:

    def __init__(self, image_format=CONFIG.Acquisition.IMAGE_FORMAT):
        self.base_path = "./"
        self.experiment_ID = ""
        self.image_format = image_format
        self.max_num_image_per_folder = 1000
        self.queue = Queue(10)  # max 10 items in the queue
        self.image_lock = threading.Lock()
        self.stop_signal_received = False
        self.thread = threading.Thread(target=self.process_queue)
        self.thread.start()
        self.counter = 0
        self.recording_start_time = 0
        self.recording_time_limit = -1

    def process_queue(self):
        while True:
            # stop the thread if stop signal is received
            if self.stop_signal_received:
                return
            # process the queue
            try:
                [image, frame_ID, timestamp] = self.queue.get(timeout=0.1)
                self.image_lock.acquire(True)
                folder_ID = int(self.counter / self.max_num_image_per_folder)
                file_ID = int(self.counter % self.max_num_image_per_folder)
                # create a new folder
                if file_ID == 0:
                    os.mkdir(
                        os.path.join(self.base_path, self.experiment_ID, str(folder_ID))
                    )

                if image.dtype == np.uint16:
                    # need to use tiff when saving 16 bit images
                    saving_path = os.path.join(
                        self.base_path,
                        self.experiment_ID,
                        str(folder_ID),
                        str(file_ID) + "_" + str(frame_ID) + ".tiff",
                    )
                    iio.imwrite(saving_path, image)
                else:
                    saving_path = os.path.join(
                        self.base_path,
                        self.experiment_ID,
                        str(folder_ID),
                        str(file_ID) + "_" + str(frame_ID) + "." + self.image_format,
                    )
                    cv2.imwrite(saving_path, image)

                self.counter = self.counter + 1
                self.queue.task_done()
                self.image_lock.release()
            except:
                pass

    def enqueue(self, image, frame_ID, timestamp):
        try:
            self.queue.put_nowait([image, frame_ID, timestamp])
            if (self.recording_time_limit > 0) and (
                time.time() - self.recording_start_time >= self.recording_time_limit
            ):
                #self.stop_recording.emit()
                pass
            # when using self.queue.put(str_), program can be slowed down despite multithreading because of the block and the GIL
        except:
            print("imageSaver queue is full, image discarded")

    def set_base_path(self, path):
        self.base_path = path

    def set_recording_time_limit(self, time_limit):
        self.recording_time_limit = time_limit

    def start_new_experiment(self, experiment_ID, add_timestamp=True):
        if add_timestamp:
            # generate unique experiment ID
            self.experiment_ID = (
                experiment_ID + "_" + datetime.now().strftime("%Y-%m-%d_%H-%M-%-S.%f")
            )
        else:
            self.experiment_ID = experiment_ID
        self.recording_start_time = time.time()
        # create a new folder
        try:
            os.mkdir(os.path.join(self.base_path, self.experiment_ID))
            # to do: save configuration
        except:
            pass
        # reset the counter
        self.counter = 0

    def close(self):
        self.queue.join()
        self.stop_signal_received = True
        self.thread.join()


class ImageSaver_Tracking:
    def __init__(self, base_path, image_format="bmp"):
        self.base_path = base_path
        self.image_format = image_format
        self.max_num_image_per_folder = 1000
        self.queue = Queue(100)  # max 100 items in the queue
        self.image_lock = threading.Lock()
        self.stop_signal_received = False
        self.thread = threading.Thread(target=self.process_queue)
        self.thread.start()

    def process_queue(self):
        while True:
            # stop the thread if stop signal is received
            if self.stop_signal_received:
                return
            # process the queue
            try:
                [image, frame_counter, postfix] = self.queue.get(timeout=0.1)
                self.image_lock.acquire(True)
                folder_ID = int(frame_counter / self.max_num_image_per_folder)
                file_ID = int(frame_counter % self.max_num_image_per_folder)
                # create a new folder
                if file_ID == 0:
                    os.mkdir(os.path.join(self.base_path, str(folder_ID)))
                if image.dtype == np.uint16:
                    saving_path = os.path.join(
                        self.base_path,
                        str(folder_ID),
                        str(file_ID)
                        + "_"
                        + str(frame_counter)
                        + "_"
                        + postfix
                        + ".tiff",
                    )
                    iio.imwrite(saving_path, image)
                else:
                    saving_path = os.path.join(
                        self.base_path,
                        str(folder_ID),
                        str(file_ID)
                        + "_"
                        + str(frame_counter)
                        + "_"
                        + postfix
                        + "."
                        + self.image_format,
                    )
                    cv2.imwrite(saving_path, image)
                self.queue.task_done()
                self.image_lock.release()
            except:
                pass

    def enqueue(self, image, frame_counter, postfix):
        try:
            self.queue.put_nowait([image, frame_counter, postfix])
        except:
            print("imageSaver queue is full, image discarded")

    def close(self):
        self.queue.join()
        self.stop_signal_received = True
        self.thread.join()


"""
class ImageSaver_MultiPointAcquisition(QObject):
"""


class ImageDisplay:

    def __init__(self):
        self.queue = Queue(10)  # max 10 items in the queue
        self.image_lock = threading.Lock()
        self.stop_signal_received = False
        self.thread = threading.Thread(target=self.process_queue)
        self.thread.start()

    def process_queue(self):
        while True:
            # stop the thread if stop signal is received
            if self.stop_signal_received:
                return
            # process the queue
            try:
                [image, frame_ID, timestamp] = self.queue.get(timeout=0.1)
                self.image_lock.acquire(True)
                self.image_lock.release()
                self.queue.task_done()
            except:
                pass

    # def enqueue(self,image,frame_ID,timestamp):
    def enqueue(self, image):
        try:
            self.queue.put_nowait([image, None, None])
            # when using self.queue.put(str_) instead of try + nowait, program can be slowed down despite multithreading because of the block and the GIL
            pass
        except:
            print("imageDisplay queue is full, image discarded")

    def close(self):
        self.queue.join()
        self.stop_signal_received = True
        self.thread.join()


class Configuration:
    def __init__(
        self,
        mode_id=None,
        name=None,
        camera_sn=None,
        exposure_time=None,
        analog_gain=None,
        illumination_source=None,
        illumination_intensity=None,
        z_offset=None,
        pixel_format=None,
        _pixel_format_options=None,
    ):
        self.id = mode_id
        self.name = name
        self.exposure_time = exposure_time
        self.analog_gain = analog_gain
        self.illumination_source = illumination_source
        self.illumination_intensity = illumination_intensity
        self.camera_sn = camera_sn
        self.z_offset = z_offset
        self.pixel_format = pixel_format
        if self.pixel_format is None:
            self.pixel_format = "default"
        self._pixel_format_options = _pixel_format_options
        if _pixel_format_options is None:
            self._pixel_format_options = self.pixel_format


class LiveController:

    def __init__(
        self,
        camera,
        microcontroller,
        configurationManager,
        control_illumination=True,
        use_internal_timer_for_hardware_trigger=True,
        for_displacement_measurement=False,
    ):
        self.camera = camera
        self.microcontroller = microcontroller
        self.configurationManager = configurationManager
        self.currentConfiguration = None
        self.trigger_mode = TriggerModeSetting.SOFTWARE  # @@@ change to None
        self.is_live = False
        self.control_illumination = control_illumination
        self.illumination_on = False
        self.use_internal_timer_for_hardware_trigger = (
            use_internal_timer_for_hardware_trigger  # use QTimer vs timer in the MCU
        )
        self.for_displacement_measurement = for_displacement_measurement

        self.fps_trigger = 1
        self.timer_trigger_interval = (1 / self.fps_trigger) * 1000

        self._trigger_task = None # asyncio task for triggering

        self.trigger_ID = -1

        self.fps_real = 0
        self.counter = 0
        self.timestamp_last = 0

        self.display_resolution_scaling = CONFIG.DEFAULT_DISPLAY_CROP / 100

    # illumination control
    def turn_on_illumination(self):
        self.microcontroller.turn_on_illumination()
        print("illumination on")
        self.illumination_on = True

    def turn_off_illumination(self):
        self.microcontroller.turn_off_illumination()
        self.illumination_on = False

    def set_illumination(self, illumination_source, intensity):
        if illumination_source < 10:  # LED matrix
            self.microcontroller.set_illumination_led_matrix(
                illumination_source,
                r=(intensity / 100) * CONFIG.LED_MATRIX_R_FACTOR,
                g=(intensity / 100) * CONFIG.LED_MATRIX_G_FACTOR,
                b=(intensity / 100) * CONFIG.LED_MATRIX_B_FACTOR,
            )
        else:
            self.microcontroller.set_illumination(illumination_source, intensity)

    def start_live(self):
        self.is_live = True
        self.camera.is_live = True
        self.camera.start_streaming()
        if self.trigger_mode == TriggerModeSetting.SOFTWARE.value or (
            self.trigger_mode == TriggerModeSetting.HARDWARE
            and self.use_internal_timer_for_hardware_trigger
        ):
            self.camera.enable_callback()  # in case it's disabled e.g. by the laser CONFIG.AF controller
            self._start_triggerred_acquisition()
        # if controlling the laser displacement measurement camera
        if self.for_displacement_measurement:
            self.microcontroller.set_pin_level(CONFIG.MCU_PINS.AF_LASER, 1)

    def stop_live(self):
        if self.is_live:
            self.is_live = False
            self.camera.is_live = False
            if hasattr(self.camera, "stop_exposure"):
                self.camera.stop_exposure()
            if self.trigger_mode == TriggerModeSetting.SOFTWARE:
                self._stop_triggerred_acquisition()
            # self.camera.stop_streaming() # 20210113 this line seems to cause problems when using af with multipoint
            if self.trigger_mode == TriggerModeSetting.CONTINUOUS:
                self.camera.stop_streaming()
            if (self.trigger_mode == TriggerModeSetting.SOFTWARE) or (
                self.trigger_mode == TriggerModeSetting.HARDWARE
                and self.use_internal_timer_for_hardware_trigger
            ):
                self._stop_triggerred_acquisition()
            if self.control_illumination:
                self.turn_off_illumination()
            # if controlling the laser displacement measurement camera
            if self.for_displacement_measurement:
                self.microcontroller.set_pin_level(CONFIG.MCU_PINS.AF_LASER, 0)

    # software trigger related
    def trigger_acquisition(self):
        if self.trigger_mode == TriggerModeSetting.SOFTWARE:
            if self.control_illumination and self.illumination_on == False:
                self.turn_on_illumination()
            self.trigger_ID = self.trigger_ID + 1

            # Handle camera trigger: schedule if async (simulation), call directly if sync (real camera)
            if asyncio.iscoroutinefunction(self.camera.send_trigger):
                asyncio.create_task(self.camera.send_trigger())
            else:
                self.camera.send_trigger()

            # measure real fps
            timestamp_now = round(time.time())
            if timestamp_now == self.timestamp_last:
                self.counter = self.counter + 1
            else:
                self.timestamp_last = timestamp_now
                self.fps_real = self.counter
                self.counter = 0
                # print('real trigger fps is ' + str(self.fps_real))
        elif self.trigger_mode == TriggerModeSetting.HARDWARE:
            self.trigger_ID = self.trigger_ID + 1
            self.microcontroller.send_hardware_trigger(
                control_illumination=True,
                illumination_on_time_us=self.camera.exposure_time * 1000,
            )

    async def _trigger_loop(self):
        while self.is_live:
            try:
                if self.trigger_mode == TriggerModeSetting.SOFTWARE or \
                   (self.trigger_mode == TriggerModeSetting.HARDWARE and self.use_internal_timer_for_hardware_trigger):
                    self.trigger_acquisition()
                await asyncio.sleep(self.timer_trigger_interval / 1000.0) # interval is in ms
            except asyncio.CancelledError:
                break # Exit loop if cancelled
            except Exception as e:
                print(f"Error in trigger loop: {e}") # Log other errors
                break

    def _start_triggerred_acquisition(self):
        # self.timer_trigger.start()
        if self._trigger_task is None or self._trigger_task.done():
            self._trigger_task = asyncio.create_task(self._trigger_loop())

    def _set_trigger_fps(self, fps_trigger):
        self.fps_trigger = fps_trigger
        self.timer_trigger_interval = (1 / self.fps_trigger) * 1000
        # self.timer_trigger.setInterval(int(self.timer_trigger_interval))

    def _stop_triggerred_acquisition(self):
        # self.timer_trigger.stop()
        if self._trigger_task and not self._trigger_task.done():
            self._trigger_task.cancel()
        self._trigger_task = None

    # trigger mode and settings
    def set_trigger_mode(self, mode):
        if mode == TriggerModeSetting.SOFTWARE.value:
            if self.is_live and (
                self.trigger_mode == TriggerModeSetting.HARDWARE
                and self.use_internal_timer_for_hardware_trigger
            ):
                self._stop_triggerred_acquisition()
            self.camera.set_software_triggered_acquisition()
            if self.is_live:
                self._start_triggerred_acquisition()
        if mode == TriggerModeSetting.HARDWARE.value:
            if self.trigger_mode == TriggerModeSetting.SOFTWARE and self.is_live:
                self._stop_triggerred_acquisition()
            # self.camera.reset_camera_acquisition_counter()
            self.camera.set_hardware_triggered_acquisition()
            self.microcontroller.set_strobe_delay_us(self.camera.strobe_delay_us)
            if self.is_live and self.use_internal_timer_for_hardware_trigger:
                self._start_triggerred_acquisition()
        if mode == TriggerModeSetting.CONTINUOUS.value:
            if (self.trigger_mode == TriggerModeSetting.SOFTWARE) or (
                self.trigger_mode == TriggerModeSetting.HARDWARE
                and self.use_internal_timer_for_hardware_trigger
            ):
                self._stop_triggerred_acquisition()
            self.camera.set_continuous_acquisition()
        self.trigger_mode = mode

    def set_trigger_fps(self, fps):
        if (self.trigger_mode == TriggerModeSetting.SOFTWARE) or (
            self.trigger_mode == TriggerModeSetting.HARDWARE
            and self.use_internal_timer_for_hardware_trigger
        ):
            self._set_trigger_fps(fps)

    # set microscope mode
    # @@@ to do: change softwareTriggerGenerator to TriggerGeneratror
    def set_microscope_mode(self, configuration):

        self.currentConfiguration = configuration
        print("setting microscope mode to " + self.currentConfiguration.name)

        # temporarily stop live while changing mode
        if self.is_live is True:
            self._stop_triggerred_acquisition()
            if self.control_illumination:
                self.turn_off_illumination()

        # set camera exposure time and analog gain
        self.camera.set_exposure_time(self.currentConfiguration.exposure_time)
        self.camera.set_analog_gain(self.currentConfiguration.analog_gain)

        # set illumination
        if self.control_illumination:
            self.set_illumination(
                self.currentConfiguration.illumination_source,
                self.currentConfiguration.illumination_intensity,
            )

        # restart live
        if self.is_live is True:
            if self.control_illumination:
                self.turn_on_illumination()
            self._start_triggerred_acquisition()

    def get_trigger_mode(self):
        return self.trigger_mode

    # slot
    def on_new_frame(self):
        if self.fps_trigger <= 5:
            if self.control_illumination and self.illumination_on == True:
                self.turn_off_illumination()

    def set_display_resolution_scaling(self, display_resolution_scaling):
        self.display_resolution_scaling = display_resolution_scaling / 100


class NavigationController:

    xPos = None
    yPos = None
    zPos = None
    thetaPos = None
    xyPos = None
    signal_joystick_button_pressed = None

    # x y z axis pid enable flag
    pid_enable_flag = [False, False, False]

    def __init__(self, microcontroller, parent=None):
        # parent should be set to OctopiGUI instance to enable updates
        # to camera settings, e.g. binning, that would affect click-to-move
        self.microcontroller = microcontroller
        self.parent = parent
        self.x_pos_mm = 0
        self.y_pos_mm = 0
        self.z_pos_mm = 0
        self.z_pos = 0
        self.theta_pos_rad = 0
        self.x_microstepping = CONFIG.MICROSTEPPING_DEFAULT_X
        self.y_microstepping = CONFIG.MICROSTEPPING_DEFAULT_Y
        self.z_microstepping = CONFIG.MICROSTEPPING_DEFAULT_Z
        self.click_to_move = False
        self.theta_microstepping = CONFIG.MICROSTEPPING_DEFAULT_THETA
        self.enable_joystick_button_action = True

        # to be moved to gui for transparency
        self.microcontroller.set_callback(self.update_pos)

    def set_flag_click_to_move(self, flag):
        self.click_to_move = flag

    def move_from_click(self, click_x, click_y):
        if self.click_to_move:
            try:
                highest_res = (0, 0)
                for res in self.parent.camera.res_list:
                    if res[0] > highest_res[0] or res[1] > highest_res[1]:
                        highest_res = res
                resolution = self.parent.camera.resolution

                try:
                    pixel_binning_x = highest_res[0] / resolution[0]
                    pixel_binning_y = highest_res[1] / resolution[1]
                    pixel_binning_x = max(pixel_binning_x, 1)
                    pixel_binning_y = max(pixel_binning_y, 1)
                except:
                    pixel_binning_x = 1
                    pixel_binning_y = 1
            except AttributeError:
                pixel_binning_x = 1
                pixel_binning_y = 1

            try:
                current_objective = self.parent.objectiveStore.current_objective
                objective_info = self.parent.objectiveStore.objectives_dict.get(
                    current_objective, {}
                )
            except (AttributeError, KeyError):
                objective_info = CONFIG.OBJECTIVES[CONFIG.DEFAULT_OBJECTIVE]
            magnification = objective_info["magnification"]
            objective_tube_lens_mm = objective_info["tube_lens_f_mm"]
            tube_lens_mm = CONFIG.TUBE_LENS_MM
            pixel_size_um = CONFIG.CAMERA_PIXEL_SIZE_UM[CONFIG.CAMERA_SENSOR]

            pixel_size_xy = pixel_size_um / (
                magnification / (objective_tube_lens_mm / tube_lens_mm)
            )

            pixel_size_x = pixel_size_xy * pixel_binning_x
            pixel_size_y = pixel_size_xy * pixel_binning_y

            pixel_sign_x = 1
            pixel_sign_y = 1 if CONFIG.INVERTED_OBJECTIVE else -1

            delta_x = pixel_sign_x * pixel_size_x * click_x / 1000.0
            delta_y = pixel_sign_y * pixel_size_y * click_y / 1000.0

            self.move_x(delta_x)
            self.move_y(delta_y)

    def move_to_cached_position(self):
        if not os.path.isfile(CONFIG.LAST_COORDS_PATH):
            return
        with open(CONFIG.LAST_COORDS_PATH) as f:
            for line in f:
                try:
                    x, y, z = line.strip("\n").strip().split(",")
                    x = float(x)
                    y = float(y)
                    z = float(z)
                    self.move_to(x, y)
                    self.move_z_to(z)
                    break
                except:
                    pass
                break

    def cache_current_position(self):
        with open(CONFIG.LAST_COORDS_PATH, "w") as f:
            f.write(
                ",".join([str(self.x_pos_mm), str(self.y_pos_mm), str(self.z_pos_mm)])
            )

    def move_x(self, delta):
        self.microcontroller.move_x_usteps(
            int(
                delta
                / (
                    CONFIG.SCREW_PITCH_X_MM
                    / (self.x_microstepping * CONFIG.FULLSTEPS_PER_REV_X)
                )
            )
        )

    def move_y(self, delta):
        self.microcontroller.move_y_usteps(
            int(
                delta
                / (
                    CONFIG.SCREW_PITCH_Y_MM
                    / (self.y_microstepping * CONFIG.FULLSTEPS_PER_REV_Y)
                )
            )
        )

    def move_z(self, delta):
        self.microcontroller.move_z_usteps(
            int(
                delta
                / (
                    CONFIG.SCREW_PITCH_Z_MM
                    / (self.z_microstepping * CONFIG.FULLSTEPS_PER_REV_Z)
                )
            )
        )

    def move_x_to(self, delta):
        self.microcontroller.move_x_to_usteps(
            CONFIG.STAGE_MOVEMENT_SIGN_X
            * int(
                delta
                / (
                    CONFIG.SCREW_PITCH_X_MM
                    / (self.x_microstepping * CONFIG.FULLSTEPS_PER_REV_X)
                )
            )
        )

    def move_y_to(self, delta):
        self.microcontroller.move_y_to_usteps(
            CONFIG.STAGE_MOVEMENT_SIGN_Y
            * int(
                delta
                / (
                    CONFIG.SCREW_PITCH_Y_MM
                    / (self.y_microstepping * CONFIG.FULLSTEPS_PER_REV_Y)
                )
            )
        )

    def move_z_to(self, delta):
        self.microcontroller.move_z_to_usteps(
            CONFIG.STAGE_MOVEMENT_SIGN_Z
            * int(
                delta
                / (
                    CONFIG.SCREW_PITCH_Z_MM
                    / (self.z_microstepping * CONFIG.FULLSTEPS_PER_REV_Z)
                )
            )
        )

    def move_x_limited(self, delta):
        self.microcontroller.move_x_usteps_limited(
            int(
                delta
                / (
                    CONFIG.SCREW_PITCH_X_MM
                    / (self.x_microstepping * CONFIG.FULLSTEPS_PER_REV_X)
                )
            )
        )

    def move_y_limited(self, delta):
        self.microcontroller.move_y_usteps_limited(
            int(
                delta
                / (
                    CONFIG.SCREW_PITCH_Y_MM
                    / (self.y_microstepping * CONFIG.FULLSTEPS_PER_REV_Y)
                )
            )
        )

    def move_z_limited(self, delta):
        self.microcontroller.move_z_usteps_limited(
            int(
                delta
                / (
                    CONFIG.SCREW_PITCH_Z_MM
                    / (self.z_microstepping * CONFIG.FULLSTEPS_PER_REV_Z)
                )
            )
        )

    def move_x_to_limited(self, delta):
        self.microcontroller.move_x_to_usteps_limited(
            CONFIG.STAGE_MOVEMENT_SIGN_X
            * int(
                delta
                / (
                    CONFIG.SCREW_PITCH_X_MM
                    / (self.x_microstepping * CONFIG.FULLSTEPS_PER_REV_X)
                )
            )
        )
    def move_x_continuous(self, delta, velocity_mm_s):
        self.microcontroller.move_x_continuous_usteps(
            int(
                delta
                / (
                    CONFIG.SCREW_PITCH_X_MM
                    / (self.x_microstepping * CONFIG.FULLSTEPS_PER_REV_X)
                )
            ),
            velocity_mm_s
        )

    def move_y_to_limited(self, delta):
        self.microcontroller.move_y_to_usteps_limited(
            CONFIG.STAGE_MOVEMENT_SIGN_Y
            * int(
                delta
                / (
                    CONFIG.SCREW_PITCH_Y_MM
                    / (self.y_microstepping * CONFIG.FULLSTEPS_PER_REV_Y)
                )
            )
        )

    def move_z_to_limited(self, delta):
        self.microcontroller.move_z_to_usteps_limited(
            CONFIG.STAGE_MOVEMENT_SIGN_Z
            * int(
                delta
                / (
                    CONFIG.SCREW_PITCH_Z_MM
                    / (self.z_microstepping * CONFIG.FULLSTEPS_PER_REV_Z)
                )
            )
        )

    def move_x_usteps(self, usteps):
        self.microcontroller.move_x_usteps(usteps)

    def move_y_usteps(self, usteps):
        self.microcontroller.move_y_usteps(usteps)

    def move_z_usteps(self, usteps):
        self.microcontroller.move_z_usteps(usteps)

    def update_pos(self, microcontroller):
        # get position from the microcontroller
        x_pos, y_pos, z_pos, theta_pos = microcontroller.get_pos()
        self.z_pos = z_pos
        # calculate position in mm or rad
        if CONFIG.USE_ENCODER_X:
            self.x_pos_mm = (
                x_pos * CONFIG.ENCODER_POS_SIGN_X * CONFIG.ENCODER_STEP_SIZE_X_MM
            )
        else:
            self.x_pos_mm = (
                x_pos
                * CONFIG.STAGE_POS_SIGN_X
                * (
                    CONFIG.SCREW_PITCH_X_MM
                    / (self.x_microstepping * CONFIG.FULLSTEPS_PER_REV_X)
                )
            )
        if CONFIG.USE_ENCODER_Y:
            self.y_pos_mm = (
                y_pos * CONFIG.ENCODER_POS_SIGN_Y * CONFIG.ENCODER_STEP_SIZE_Y_MM
            )
        else:
            self.y_pos_mm = (
                y_pos
                * CONFIG.STAGE_POS_SIGN_Y
                * (
                    CONFIG.SCREW_PITCH_Y_MM
                    / (self.y_microstepping * CONFIG.FULLSTEPS_PER_REV_Y)
                )
            )
        if CONFIG.USE_ENCODER_Z:
            self.z_pos_mm = (
                z_pos * CONFIG.ENCODER_POS_SIGN_Z * CONFIG.ENCODER_STEP_SIZE_Z_MM
            )
        else:
            self.z_pos_mm = (
                z_pos
                * CONFIG.STAGE_POS_SIGN_Z
                * (
                    CONFIG.SCREW_PITCH_Z_MM
                    / (self.z_microstepping * CONFIG.FULLSTEPS_PER_REV_Z)
                )
            )
        if CONFIG.USE_ENCODER_THETA:
            self.theta_pos_rad = (
                theta_pos
                * CONFIG.ENCODER_POS_SIGN_THETA
                * CONFIG.ENCODER_STEP_SIZE_THETA
            )
        else:
            self.theta_pos_rad = (
                theta_pos
                * CONFIG.STAGE_POS_SIGN_THETA
                * (
                    2
                    * math.pi
                    / (self.theta_microstepping * CONFIG.FULLSTEPS_PER_REV_THETA)
                )
            )
        # emit the updated position
        self.xPos = self.x_pos_mm
        self.yPos = self.y_pos_mm
        self.zPos = self.z_pos_mm * 1000
        self.thetaPos = self.theta_pos_rad * 360 / (2 * math.pi)
        self.xyPos = (self.x_pos_mm, self.y_pos_mm)

        if microcontroller.signal_joystick_button_pressed_event:
            if self.enable_joystick_button_action and self.signal_joystick_button_pressed:
                self.signal_joystick_button_pressed()
            print("joystick button pressed")
            microcontroller.signal_joystick_button_pressed_event = False

        return self.x_pos_mm, self.y_pos_mm, self.z_pos_mm, self.theta_pos_rad

    def home_x(self):
        self.microcontroller.home_x()

    def home_y(self):
        self.microcontroller.home_y()

    def home_z(self):
        self.microcontroller.home_z()

    def home_theta(self):
        self.microcontroller.home_theta()

    def home_xy(self):
        self.microcontroller.home_xy()

    def zero_x(self):
        self.microcontroller.zero_x()

    def zero_y(self):
        self.microcontroller.zero_y()

    def zero_z(self):
        self.microcontroller.zero_z()

    def zero_theta(self):
        self.microcontroller.zero_tehta()

    def home(self):
        pass

    def set_x_limit_pos_mm(self, value_mm):
        if CONFIG.STAGE_MOVEMENT_SIGN_X > 0:
            self.microcontroller.set_lim(
                LIMIT_CODE.X_POSITIVE,
                int(
                    value_mm
                    / (
                        CONFIG.SCREW_PITCH_X_MM
                        / (self.x_microstepping * CONFIG.FULLSTEPS_PER_REV_X)
                    )
                ),
            )
        else:
            self.microcontroller.set_lim(
                LIMIT_CODE.X_NEGATIVE,
                CONFIG.STAGE_MOVEMENT_SIGN_X
                * int(
                    value_mm
                    / (
                        CONFIG.SCREW_PITCH_X_MM
                        / (self.x_microstepping * CONFIG.FULLSTEPS_PER_REV_X)
                    )
                ),
            )

    def set_x_limit_neg_mm(self, value_mm):
        if CONFIG.STAGE_MOVEMENT_SIGN_X > 0:
            self.microcontroller.set_lim(
                LIMIT_CODE.X_NEGATIVE,
                int(
                    value_mm
                    / (
                        CONFIG.SCREW_PITCH_X_MM
                        / (self.x_microstepping * CONFIG.FULLSTEPS_PER_REV_X)
                    )
                ),
            )
        else:
            self.microcontroller.set_lim(
                LIMIT_CODE.X_POSITIVE,
                CONFIG.STAGE_MOVEMENT_SIGN_X
                * int(
                    value_mm
                    / (
                        CONFIG.SCREW_PITCH_X_MM
                        / (self.x_microstepping * CONFIG.FULLSTEPS_PER_REV_X)
                    )
                ),
            )

    def set_y_limit_pos_mm(self, value_mm):
        if CONFIG.STAGE_MOVEMENT_SIGN_Y > 0:
            self.microcontroller.set_lim(
                LIMIT_CODE.Y_POSITIVE,
                int(
                    value_mm
                    / (
                        CONFIG.SCREW_PITCH_Y_MM
                        / (self.y_microstepping * CONFIG.FULLSTEPS_PER_REV_Y)
                    )
                ),
            )
        else:
            self.microcontroller.set_lim(
                LIMIT_CODE.Y_NEGATIVE,
                CONFIG.STAGE_MOVEMENT_SIGN_Y
                * int(
                    value_mm
                    / (
                        CONFIG.SCREW_PITCH_Y_MM
                        / (self.y_microstepping * CONFIG.FULLSTEPS_PER_REV_Y)
                    )
                ),
            )

    def set_y_limit_neg_mm(self, value_mm):
        if CONFIG.STAGE_MOVEMENT_SIGN_Y > 0:
            self.microcontroller.set_lim(
                LIMIT_CODE.Y_NEGATIVE,
                int(
                    value_mm
                    / (
                        CONFIG.SCREW_PITCH_Y_MM
                        / (self.y_microstepping * CONFIG.FULLSTEPS_PER_REV_Y)
                    )
                ),
            )
        else:
            self.microcontroller.set_lim(
                LIMIT_CODE.Y_POSITIVE,
                CONFIG.STAGE_MOVEMENT_SIGN_Y
                * int(
                    value_mm
                    / (
                        CONFIG.SCREW_PITCH_Y_MM
                        / (self.y_microstepping * CONFIG.FULLSTEPS_PER_REV_Y)
                    )
                ),
            )

    def set_z_limit_pos_mm(self, value_mm):
        if CONFIG.STAGE_MOVEMENT_SIGN_Z > 0:
            self.microcontroller.set_lim(
                LIMIT_CODE.Z_POSITIVE,
                int(
                    value_mm
                    / (
                        CONFIG.SCREW_PITCH_Z_MM
                        / (self.z_microstepping * CONFIG.FULLSTEPS_PER_REV_Z)
                    )
                ),
            )
        else:
            self.microcontroller.set_lim(
                LIMIT_CODE.Z_NEGATIVE,
                CONFIG.STAGE_MOVEMENT_SIGN_Z
                * int(
                    value_mm
                    / (
                        CONFIG.SCREW_PITCH_Z_MM
                        / (self.z_microstepping * CONFIG.FULLSTEPS_PER_REV_Z)
                    )
                ),
            )

    def set_z_limit_neg_mm(self, value_mm):
        if CONFIG.STAGE_MOVEMENT_SIGN_Z > 0:
            self.microcontroller.set_lim(
                LIMIT_CODE.Z_NEGATIVE,
                int(
                    value_mm
                    / (
                        CONFIG.SCREW_PITCH_Z_MM
                        / (self.z_microstepping * CONFIG.FULLSTEPS_PER_REV_Z)
                    )
                ),
            )
        else:
            self.microcontroller.set_lim(
                LIMIT_CODE.Z_POSITIVE,
                CONFIG.STAGE_MOVEMENT_SIGN_Z
                * int(
                    value_mm
                    / (
                        CONFIG.SCREW_PITCH_Z_MM
                        / (self.z_microstepping * CONFIG.FULLSTEPS_PER_REV_Z)
                    )
                ),
            )

    def move_to(self, x_mm, y_mm):
        self.move_x_to(x_mm)
        self.move_y_to(y_mm)

    def configure_encoder(self, axis, transitions_per_revolution, flip_direction):
        self.microcontroller.configure_stage_pid(
            axis,
            transitions_per_revolution=int(transitions_per_revolution),
            flip_direction=flip_direction,
        )

    def set_pid_control_enable(self, axis, enable_flag):
        self.pid_enable_flag[axis] = enable_flag
        if self.pid_enable_flag[axis] is True:
            self.microcontroller.turn_on_stage_pid(axis)
        else:
            self.microcontroller.turn_off_stage_pid(axis)

    def turnoff_axis_pid_control(self):
        for i in range(len(self.pid_enable_flag)):
            if self.pid_enable_flag[i] is True:
                self.microcontroller.turn_off_stage_pid(i)

    def get_pid_control_flag(self, axis):
        return self.pid_enable_flag[axis]


class SlidePositionControlWorker:

    def __init__(self, slidePositionController, home_x_and_y_separately=False):
        self.slidePositionController = slidePositionController
        self.navigationController = slidePositionController.navigationController
        self.microcontroller = self.navigationController.microcontroller
        self.liveController = self.slidePositionController.liveController
        self.home_x_and_y_separately = home_x_and_y_separately

    def wait_till_operation_is_completed(
        self, timestamp_start, slide_position_switching_timeout_limit_s
    ):
        while self.microcontroller.is_busy():
            time.sleep(CONFIG.SLEEP_TIME_S)
            if time.time() - timestamp_start > slide_position_switching_timeout_limit_s:
                print("Error - slide position switching timeout, the program will exit")
                self.navigationController.move_x(0)
                self.navigationController.move_y(0)
                exit()

    def move_to_slide_loading_position(self):
        was_live = self.liveController.is_live
        if was_live:
            self.signal_stop_live()

        # retract z
        timestamp_start = time.time()
        self.slidePositionController.z_pos = (
            self.navigationController.z_pos
        )  # zpos at the beginning of the scan
        self.navigationController.move_z_to(CONFIG.OBJECTIVE_RETRACTED_POS_MM)
        self.wait_till_operation_is_completed(
            timestamp_start, CONFIG.SLIDE_POTISION_SWITCHING_TIMEOUT_LIMIT_S
        )
        print("z retracted")
        self.slidePositionController.objective_retracted = True

        # move to position
        # for well plate
        if self.slidePositionController.is_for_wellplate:
            # reset limits
            self.navigationController.set_x_limit_pos_mm(100)
            self.navigationController.set_x_limit_neg_mm(-100)
            self.navigationController.set_y_limit_pos_mm(100)
            self.navigationController.set_y_limit_neg_mm(-100)
            # home for the first time
            if self.slidePositionController.homing_done == False:
                print("running homing first")
                timestamp_start = time.time()
                # x needs to be at > + 20 mm when homing y
                self.navigationController.move_x(20)
                self.wait_till_operation_is_completed(
                    timestamp_start, CONFIG.SLIDE_POTISION_SWITCHING_TIMEOUT_LIMIT_S
                )
                # home y
                self.navigationController.home_y()
                self.wait_till_operation_is_completed(
                    timestamp_start, CONFIG.SLIDE_POTISION_SWITCHING_TIMEOUT_LIMIT_S
                )
                self.navigationController.zero_y()
                # home x
                self.navigationController.home_x()
                self.wait_till_operation_is_completed(
                    timestamp_start, CONFIG.SLIDE_POTISION_SWITCHING_TIMEOUT_LIMIT_S
                )
                self.navigationController.zero_x()
                self.slidePositionController.homing_done = True
            # homing done previously
            else:
                timestamp_start = time.time()
                self.navigationController.move_x_to(20)
                self.wait_till_operation_is_completed(
                    timestamp_start, CONFIG.SLIDE_POTISION_SWITCHING_TIMEOUT_LIMIT_S
                )
                self.navigationController.move_y_to(CONFIG.SLIDE_POSITION.LOADING_Y_MM)
                self.wait_till_operation_is_completed(
                    timestamp_start, CONFIG.SLIDE_POTISION_SWITCHING_TIMEOUT_LIMIT_S
                )
                self.navigationController.move_x_to(CONFIG.SLIDE_POSITION.LOADING_X_MM)
                self.wait_till_operation_is_completed(
                    timestamp_start, CONFIG.SLIDE_POTISION_SWITCHING_TIMEOUT_LIMIT_S
                )
            # set limits again
            self.navigationController.set_x_limit_pos_mm(
                CONFIG.SOFTWARE_POS_LIMIT.X_POSITIVE
            )
            self.navigationController.set_x_limit_neg_mm(
                CONFIG.SOFTWARE_POS_LIMIT.X_NEGATIVE
            )
            self.navigationController.set_y_limit_pos_mm(
                CONFIG.SOFTWARE_POS_LIMIT.Y_POSITIVE
            )
            self.navigationController.set_y_limit_neg_mm(
                CONFIG.SOFTWARE_POS_LIMIT.Y_NEGATIVE
            )

        # for glass slide
        elif (
            self.slidePositionController.homing_done == False
            or CONFIG.SLIDE_POTISION_SWITCHING_HOME_EVERYTIME
        ):
            if self.home_x_and_y_separately:
                timestamp_start = time.time()
                self.navigationController.home_x()
                self.wait_till_operation_is_completed(
                    timestamp_start, CONFIG.SLIDE_POTISION_SWITCHING_TIMEOUT_LIMIT_S
                )
                self.navigationController.zero_x()
                self.navigationController.move_x(CONFIG.SLIDE_POSITION.LOADING_X_MM)
                self.wait_till_operation_is_completed(
                    timestamp_start, CONFIG.SLIDE_POTISION_SWITCHING_TIMEOUT_LIMIT_S
                )
                self.navigationController.home_y()
                self.wait_till_operation_is_completed(
                    timestamp_start, CONFIG.SLIDE_POTISION_SWITCHING_TIMEOUT_LIMIT_S
                )
                self.navigationController.zero_y()
                self.navigationController.move_y(CONFIG.SLIDE_POSITION.LOADING_Y_MM)
                self.wait_till_operation_is_completed(
                    timestamp_start, CONFIG.SLIDE_POTISION_SWITCHING_TIMEOUT_LIMIT_S
                )
            else:
                timestamp_start = time.time()
                self.navigationController.home_xy()
                self.wait_till_operation_is_completed(
                    timestamp_start, CONFIG.SLIDE_POTISION_SWITCHING_TIMEOUT_LIMIT_S
                )
                self.navigationController.zero_x()
                self.navigationController.zero_y()
                self.navigationController.move_x(CONFIG.SLIDE_POSITION.LOADING_X_MM)
                self.wait_till_operation_is_completed(
                    timestamp_start, CONFIG.SLIDE_POTISION_SWITCHING_TIMEOUT_LIMIT_S
                )
                self.navigationController.move_y(CONFIG.SLIDE_POSITION.LOADING_Y_MM)
                self.wait_till_operation_is_completed(
                    timestamp_start, CONFIG.SLIDE_POTISION_SWITCHING_TIMEOUT_LIMIT_S
                )
            self.slidePositionController.homing_done = True
        else:
            timestamp_start = time.time()
            self.navigationController.move_y(
                CONFIG.SLIDE_POSITION.LOADING_Y_MM
                - self.navigationController.y_pos_mm
            )
            self.wait_till_operation_is_completed(
                timestamp_start, CONFIG.SLIDE_POTISION_SWITCHING_TIMEOUT_LIMIT_S
            )
            self.navigationController.move_x(
                CONFIG.SLIDE_POSITION.LOADING_X_MM
                - self.navigationController.x_pos_mm
            )
            self.wait_till_operation_is_completed(
                timestamp_start, CONFIG.SLIDE_POTISION_SWITCHING_TIMEOUT_LIMIT_S
            )

        if was_live:
            self.signal_resume_live()

        self.slidePositionController.slide_loading_position_reached = True

    def move_to_slide_scanning_position(self):
        was_live = self.liveController.is_live
        if was_live:
            self.signal_stop_live()

        # move to position
        # for well plate
        if self.slidePositionController.is_for_wellplate:
            # home for the first time
            if self.slidePositionController.homing_done == False:
                timestamp_start = time.time()

                # x needs to be at > + 20 mm when homing y
                self.navigationController.move_x(20)
                self.wait_till_operation_is_completed(
                    timestamp_start, CONFIG.SLIDE_POTISION_SWITCHING_TIMEOUT_LIMIT_S
                )
                # home y
                self.navigationController.home_y()
                self.wait_till_operation_is_completed(
                    timestamp_start, CONFIG.SLIDE_POTISION_SWITCHING_TIMEOUT_LIMIT_S
                )
                self.navigationController.zero_y()
                # home x
                self.navigationController.home_x()
                self.wait_till_operation_is_completed(
                    timestamp_start, CONFIG.SLIDE_POTISION_SWITCHING_TIMEOUT_LIMIT_S
                )
                self.navigationController.zero_x()
                self.slidePositionController.homing_done = True
                # move to scanning position
                self.navigationController.move_x_to(CONFIG.SLIDE_POSITION.SCANNING_X_MM)
                self.wait_till_operation_is_completed(
                    timestamp_start, CONFIG.SLIDE_POTISION_SWITCHING_TIMEOUT_LIMIT_S
                )

                self.navigationController.move_y_to(CONFIG.SLIDE_POSITION.SCANNING_Y_MM)
                self.wait_till_operation_is_completed(
                    timestamp_start, CONFIG.SLIDE_POTISION_SWITCHING_TIMEOUT_LIMIT_S
                )

            else:
                timestamp_start = time.time()
                self.navigationController.move_x_to(CONFIG.SLIDE_POSITION.SCANNING_X_MM)
                self.wait_till_operation_is_completed(
                    timestamp_start, CONFIG.SLIDE_POTISION_SWITCHING_TIMEOUT_LIMIT_S
                )
                self.navigationController.move_y_to(CONFIG.SLIDE_POSITION.SCANNING_Y_MM)
                self.wait_till_operation_is_completed(
                    timestamp_start, CONFIG.SLIDE_POTISION_SWITCHING_TIMEOUT_LIMIT_S
                )
        elif (
            self.slidePositionController.homing_done == False
            or CONFIG.SLIDE_POTISION_SWITCHING_HOME_EVERYTIME
        ):
            if self.home_x_and_y_separately:
                timestamp_start = time.time()
                self.navigationController.home_y()
                self.wait_till_operation_is_completed(
                    timestamp_start, CONFIG.SLIDE_POTISION_SWITCHING_TIMEOUT_LIMIT_S
                )
                self.navigationController.zero_y()
                self.navigationController.move_y(
                    CONFIG.SLIDE_POSITION.SCANNING_Y_MM
                )
                self.wait_till_operation_is_completed(
                    timestamp_start, CONFIG.SLIDE_POTISION_SWITCHING_TIMEOUT_LIMIT_S
                )
                self.navigationController.home_x()
                self.wait_till_operation_is_completed(
                    timestamp_start, CONFIG.SLIDE_POTISION_SWITCHING_TIMEOUT_LIMIT_S
                )
                self.navigationController.zero_x()
                self.navigationController.move_x(
                    CONFIG.SLIDE_POSITION.SCANNING_X_MM
                )
                self.wait_till_operation_is_completed(
                    timestamp_start, CONFIG.SLIDE_POTISION_SWITCHING_TIMEOUT_LIMIT_S
                )
            else:
                timestamp_start = time.time()
                self.navigationController.home_xy()
                self.wait_till_operation_is_completed(
                    timestamp_start, CONFIG.SLIDE_POTISION_SWITCHING_TIMEOUT_LIMIT_S
                )
                self.navigationController.zero_x()
                self.navigationController.zero_y()
                self.navigationController.move_y(
                    CONFIG.SLIDE_POSITION.SCANNING_Y_MM
                )
                self.wait_till_operation_is_completed(
                    timestamp_start, CONFIG.SLIDE_POTISION_SWITCHING_TIMEOUT_LIMIT_S
                )
                self.navigationController.move_x(
                    CONFIG.SLIDE_POSITION.SCANNING_X_MM
                )
                self.wait_till_operation_is_completed(
                    timestamp_start, CONFIG.SLIDE_POTISION_SWITCHING_TIMEOUT_LIMIT_S
                )
            self.slidePositionController.homing_done = True
        else:
            timestamp_start = time.time()
            self.navigationController.move_y(
                CONFIG.SLIDE_POSITION.SCANNING_Y_MM
                - self.navigationController.y_pos_mm
            )
            self.wait_till_operation_is_completed(
                timestamp_start, CONFIG.SLIDE_POTISION_SWITCHING_TIMEOUT_LIMIT_S
            )
            self.navigationController.move_x(
                CONFIG.SLIDE_POSITION.SCANNING_X_MM
                - self.navigationController.x_pos_mm
            )
            self.wait_till_operation_is_completed(
                timestamp_start, CONFIG.SLIDE_POTISION_SWITCHING_TIMEOUT_LIMIT_S
            )

        # restore z
        if self.slidePositionController.objective_retracted:
            if self.navigationController.get_pid_control_flag(2) is False:
                _usteps_to_clear_backlash = max(
                    160, 20 * self.navigationController.z_microstepping
                )
                self.navigationController.microcontroller.move_z_to_usteps(
                    self.slidePositionController.z_pos
                    - CONFIG.STAGE_MOVEMENT_SIGN_Z * _usteps_to_clear_backlash
                )
                self.wait_till_operation_is_completed(
                    timestamp_start, CONFIG.SLIDE_POTISION_SWITCHING_TIMEOUT_LIMIT_S
                )
                self.navigationController.move_z_usteps(_usteps_to_clear_backlash)
                self.wait_till_operation_is_completed(
                    timestamp_start, CONFIG.SLIDE_POTISION_SWITCHING_TIMEOUT_LIMIT_S
                )
            else:
                self.navigationController.microcontroller.move_z_to_usteps(
                    self.slidePositionController.z_pos
                )
                self.wait_till_operation_is_completed(
                    timestamp_start, CONFIG.SLIDE_POTISION_SWITCHING_TIMEOUT_LIMIT_S
                )
            self.slidePositionController.objective_retracted = False
            print("z position restored")

        if was_live:
            self.signal_resume_live()

        self.slidePositionController.slide_scanning_position_reached = True


class SlidePositionController:

    signal_slide_loading_position_reached = None
    signal_slide_scanning_position_reached = None
    signal_clear_slide = None

    def __init__(self, navigationController, liveController, is_for_wellplate=False):
        self.navigationController = navigationController
        self.liveController = liveController
        self.slide_loading_position_reached = False
        self.slide_scanning_position_reached = False
        self.homing_done = False
        self.is_for_wellplate = is_for_wellplate
        self.retract_objective_before_moving = (
            CONFIG.RETRACT_OBJECTIVE_BEFORE_MOVING_TO_LOADING_POSITION
        )
        self.objective_retracted = False
        self.thread = None

    def move_to_slide_loading_position(self):
        # create a worker object
        self.slidePositionControlWorker = SlidePositionControlWorker(self)
        # Set up callbacks
        self.slidePositionControlWorker.signal_stop_live = self.slot_stop_live
        self.slidePositionControlWorker.signal_resume_live = self.slot_resume_live
        # create and start the thread
        self.thread = threading.Thread(
            target=self.slidePositionControlWorker.move_to_slide_loading_position
        )
        self.thread.start()

    def move_to_slide_scanning_position(self):
        # create a worker object
        self.slidePositionControlWorker = SlidePositionControlWorker(self)
        # Set up callbacks
        self.slidePositionControlWorker.signal_stop_live = self.slot_stop_live
        self.slidePositionControlWorker.signal_resume_live = self.slot_resume_live
        # create and start the thread
        self.thread = threading.Thread(
            target=self.slidePositionControlWorker.move_to_slide_scanning_position
        )
        print("before thread.start()")
        self.thread.start()
        if self.signal_clear_slide:
            self.signal_clear_slide()

    def slot_stop_live(self):
        self.liveController.stop_live()

    def slot_resume_live(self):
        self.liveController.start_live()

    # def threadFinished(self):
    # 	print('========= threadFinished ========= ')


class AutofocusWorker:

    def __init__(self, autofocusController):
        self.autofocusController = autofocusController

        self.camera = self.autofocusController.camera
        self.microcontroller = (
            self.autofocusController.navigationController.microcontroller
        )
        self.navigationController = self.autofocusController.navigationController
        self.liveController = self.autofocusController.liveController

        self.N = self.autofocusController.N
        self.deltaZ = self.autofocusController.deltaZ
        self.deltaZ_usteps = self.autofocusController.deltaZ_usteps

        self.crop_width = self.autofocusController.crop_width
        self.crop_height = self.autofocusController.crop_height

    def run(self):
        self.run_autofocus()

    def wait_till_operation_is_completed(self):
        while self.microcontroller.is_busy():
            time.sleep(CONFIG.SLEEP_TIME_S)

    def run_autofocus(self):
        # @@@ to add: increase gain, decrease exposure time
        # @@@ can move the execution into a thread - done 08/21/2021
        focus_measure_vs_z = [0] * self.N
        focus_measure_max = 0

        z_af_offset_usteps = self.deltaZ_usteps * round(self.N / 2)
        # self.navigationController.move_z_usteps(-z_af_offset_usteps) # combine with the back and forth maneuver below
        # self.wait_till_operation_is_completed()

        # maneuver for achiving uniform step size and repeatability when using open-loop control
        # can be moved to the firmware
        if self.navigationController.get_pid_control_flag(2) is False:
            _usteps_to_clear_backlash = max(
                160, 20 * self.navigationController.z_microstepping
            )
            self.navigationController.move_z_usteps(
                -_usteps_to_clear_backlash - z_af_offset_usteps
            )
            self.wait_till_operation_is_completed()
            self.navigationController.move_z_usteps(_usteps_to_clear_backlash)
            self.wait_till_operation_is_completed()
        else:
            self.navigationController.move_z_usteps(-z_af_offset_usteps)
            self.wait_till_operation_is_completed()

        #check if the illumination is on
        illumination_on = self.liveController.illumination_on
        if illumination_on:
            self.liveController.turn_off_illumination()
            self.wait_till_operation_is_completed()

        steps_moved = 0
        for i in range(self.N):
            self.navigationController.move_z_usteps(self.deltaZ_usteps)
            self.wait_till_operation_is_completed()
            steps_moved = steps_moved + 1
            # trigger acquisition (including turning on the illumination)
            if self.liveController.trigger_mode == TriggerModeSetting.SOFTWARE:
                self.liveController.turn_on_illumination()
                self.wait_till_operation_is_completed()
                self.camera.send_trigger()
            elif self.liveController.trigger_mode == TriggerModeSetting.HARDWARE:
                self.microcontroller.send_hardware_trigger(
                    control_illumination=True,
                    illumination_on_time_us=self.camera.exposure_time * 1000,
                )
            # read camera frame
            image = self.camera.read_frame()
            if image is None:
                continue
            # tunr of the illumination if using software trigger
            if self.liveController.trigger_mode == TriggerModeSetting.SOFTWARE:
                self.liveController.turn_off_illumination()
            image = utils.crop_image(image, self.crop_width, self.crop_height)
            image = utils.rotate_and_flip_image(
                image,
                rotate_image_angle=self.camera.rotate_image_angle,
                flip_image=self.camera.flip_image,
            )
            timestamp_0 = time.time()
            focus_measure = utils.calculate_focus_measure(
                image, CONFIG.FOCUS_MEASURE_OPERATOR
            )
            timestamp_1 = time.time()
            print(
                "             calculating focus measure took "
                + str(timestamp_1 - timestamp_0)
                + " second"
            )
            focus_measure_vs_z[i] = focus_measure
            print(i, focus_measure)
            focus_measure_max = max(focus_measure, focus_measure_max)
            if focus_measure < focus_measure_max * CONFIG.AF.STOP_THRESHOLD:
                break

        # move to the starting location
        # self.navigationController.move_z_usteps(-steps_moved*self.deltaZ_usteps) # combine with the back and forth maneuver below
        # self.wait_till_operation_is_completed()

        # maneuver for achiving uniform step size and repeatability when using open-loop control
        if self.navigationController.get_pid_control_flag(2) is False:
            _usteps_to_clear_backlash = max(
                160, 20 * self.navigationController.z_microstepping
            )
            self.navigationController.move_z_usteps(
                -_usteps_to_clear_backlash - steps_moved * self.deltaZ_usteps
            )
            # determine the in-focus position
            idx_in_focus = focus_measure_vs_z.index(max(focus_measure_vs_z))
            self.wait_till_operation_is_completed()
            self.navigationController.move_z_usteps(
                _usteps_to_clear_backlash + (idx_in_focus + 1) * self.deltaZ_usteps
            )
            self.wait_till_operation_is_completed()
        else:
            # determine the in-focus position
            idx_in_focus = focus_measure_vs_z.index(max(focus_measure_vs_z))
            self.navigationController.move_z_usteps(
                (idx_in_focus + 1) * self.deltaZ_usteps
                - steps_moved * self.deltaZ_usteps
            )
            self.wait_till_operation_is_completed()
        #turn on the illumination if the illumination was on before the autofocus
        if illumination_on:
            self.liveController.turn_on_illumination()
            self.wait_till_operation_is_completed()
        # move to the calculated in-focus position
        # self.navigationController.move_z_usteps(idx_in_focus*self.deltaZ_usteps)
        # self.wait_till_operation_is_completed() # combine with the movement above
        if idx_in_focus == 0:
            print("moved to the bottom end of the CONFIG.AF range")
        if idx_in_focus == self.N - 1:
            print("moved to the top end of the CONFIG.AF range")


class AutoFocusController:

    z_pos = None
    autofocusFinished = None
    image_to_display = None

    def __init__(self, camera, navigationController, liveController):
        self.camera = camera
        self.navigationController = navigationController
        self.liveController = liveController
        self.N = None
        self.deltaZ = None
        self.deltaZ_usteps = None
        self.crop_width = CONFIG.AF.CROP_WIDTH
        self.crop_height = CONFIG.AF.CROP_HEIGHT
        self.autofocus_in_progress = False
        self.focus_map_coords = []
        self.use_focus_map = False

    def set_N(self, N):
        self.N = N

    def set_deltaZ(self, deltaZ_um):
        mm_per_ustep_Z = CONFIG.SCREW_PITCH_Z_MM / (
            self.navigationController.z_microstepping * CONFIG.FULLSTEPS_PER_REV_Z
        )
        self.deltaZ = deltaZ_um / 1000
        self.deltaZ_usteps = round((deltaZ_um / 1000) / mm_per_ustep_Z)

    def set_crop(self, crop_width, crop_height):
        self.crop_width = crop_width
        self.crop_height = crop_height

    def autofocus(self, focus_map_override=False):
        if self.use_focus_map and (not focus_map_override):
            self.autofocus_in_progress = True
            self.navigationController.microcontroller.wait_till_operation_is_completed()
            x = self.navigationController.x_pos_mm
            y = self.navigationController.y_pos_mm

            # z here is in mm because that's how the navigation controller stores it
            target_z = utils.interpolate_plane(*self.focus_map_coords[:3], (x, y))
            print(
                f"Interpolated target z as {target_z} mm from focus map, moving there."
            )
            self.navigationController.move_z_to(target_z)
            self.navigationController.microcontroller.wait_till_operation_is_completed()
            self.autofocus_in_progress = False
            return
        # stop live
        if self.liveController.is_live:
            self.was_live_before_autofocus = True
            self.liveController.stop_live()
        else:
            self.was_live_before_autofocus = False

        # temporarily disable call back -> image does not go through streamHandler
        if self.camera.callback_is_enabled:
            self.callback_was_enabled_before_autofocus = True
            self.camera.disable_callback()
        else:
            self.callback_was_enabled_before_autofocus = False

        self.autofocus_in_progress = True

        try:
            if hasattr(self, 'thread') and self.thread and self.thread.is_alive():
                print('*** autofocus thread is still running ***')
                # For standard threading, we can't forcefully terminate, just wait
                self.thread.join(timeout=1.0)
                print('*** autofocus threaded manually stopped ***')
        except:
            pass

        # create a worker object
        self.autofocusWorker = AutofocusWorker(self)

        self.autofocusWorker.run()
        self._on_autofocus_completed()





    def _on_autofocus_completed(self):
        # re-enable callback
        if self.callback_was_enabled_before_autofocus:
            self.camera.enable_callback()

        # re-enable live if it's previously on
        if self.was_live_before_autofocus:
            self.liveController.start_live()

        # emit the autofocus finished signal to enable the UI
        if self.autofocusFinished:
            self.autofocusFinished()
        print("autofocus finished")

        # update the state
        self.autofocus_in_progress = False

    def wait_till_autofocus_has_completed(self):
        while self.autofocus_in_progress == True:
            time.sleep(0.005)
        print("autofocus wait has completed, exit wait")

    def set_focus_map_use(self, enable):
        if not enable:
            print("Disabling focus map.")
            self.use_focus_map = False
            return
        if len(self.focus_map_coords) < 3:
            print(
                "Not enough coordinates (less than 3) for focus map generation, disabling focus map."
            )
            self.use_focus_map = False
            return
        x1, y1, _ = self.focus_map_coords[0]
        x2, y2, _ = self.focus_map_coords[1]
        x3, y3, _ = self.focus_map_coords[2]

        detT = (y2 - y3) * (x1 - x3) + (x3 - x2) * (y1 - y3)
        if detT == 0:
            print(
                "Your 3 x-y coordinates are linear, cannot use to interpolate, disabling focus map."
            )
            self.use_focus_map = False
            return

        if enable:
            print("Enabling focus map.")
            self.use_focus_map = True

    def clear_focus_map(self):
        self.focus_map_coords = []
        self.set_focus_map_use(False)

    def gen_focus_map(self, coord1, coord2, coord3):
        """
        Navigate to 3 coordinates and get your focus-map coordinates
        by autofocusing there and saving the z-values.
        :param coord1-3: Tuples of (x,y) values, coordinates in mm.
        :raise: ValueError if coordinates are all on the same line
        """
        x1, y1 = coord1
        x2, y2 = coord2
        x3, y3 = coord3
        detT = (y2 - y3) * (x1 - x3) + (x3 - x2) * (y1 - y3)
        if detT == 0:
            raise ValueError("Your 3 x-y coordinates are linear")

        self.focus_map_coords = []

        for coord in [coord1, coord2, coord3]:
            print(
                f"Navigating to coordinates ({coord[0]},{coord[1]}) to sample for focus map"
            )
            self.navigationController.move_to(coord[0], coord[1])
            self.navigationController.microcontroller.wait_till_operation_is_completed()
            print("Autofocusing")
            self.autofocus(True)
            self.wait_till_autofocus_has_completed()
            # self.navigationController.microcontroller.wait_till_operation_is_completed()
            x = self.navigationController.x_pos_mm
            y = self.navigationController.y_pos_mm
            z = self.navigationController.z_pos_mm
            print(f"Adding coordinates ({x},{y},{z}) to focus map")
            self.focus_map_coords.append((x, y, z))

        print("Generated focus map.")

    def add_current_coords_to_focus_map(self):
        if len(self.focus_map_coords) >= 3:
            print("Replacing last coordinate on focus map.")
        self.navigationController.microcontroller.wait_till_operation_is_completed()
        print("Autofocusing")
        self.autofocus(True)
        self.wait_till_autofocus_has_completed()
        # self.navigationController.microcontroller.wait_till_operation_is_completed()
        x = self.navigationController.x_pos_mm
        y = self.navigationController.y_pos_mm
        z = self.navigationController.z_pos_mm
        if len(self.focus_map_coords) >= 2:
            x1, y1, _ = self.focus_map_coords[0]
            x2, y2, _ = self.focus_map_coords[1]
            x3 = x
            y3 = y

            detT = (y2 - y3) * (x1 - x3) + (x3 - x2) * (y1 - y3)
            if detT == 0:
                raise ValueError(
                    "Your 3 x-y coordinates are linear. Navigate to a different coordinate or clear and try again."
                )
        if len(self.focus_map_coords) >= 3:
            self.focus_map_coords.pop()
        self.focus_map_coords.append((x, y, z))
        print(f"Added triple ({x},{y},{z}) to focus map")


class MultiPointWorker:

    def __init__(self, multiPointController):
        self.multiPointController = multiPointController
        self.update_stats_callback = None
        self.image_to_display_callback = None
        self.spectrum_to_display_callback = None
        self.image_to_display_multi_callback = None
        self.signal_current_configuration_callback = None
        self.signal_register_current_fov_callback = None
        self.signal_detection_stats_callback = None
        self.start_time = 0
        self.processingHandler = multiPointController.processingHandler
        self.camera = self.multiPointController.camera
        self.microcontroller = self.multiPointController.microcontroller
        self.usb_spectrometer = self.multiPointController.usb_spectrometer
        self.navigationController = self.multiPointController.navigationController
        self.liveController = self.multiPointController.liveController
        self.autofocusController = self.multiPointController.autofocusController
        self.configurationManager = self.multiPointController.configurationManager
        self.NX = self.multiPointController.NX
        self.NY = self.multiPointController.NY
        self.NZ = self.multiPointController.NZ
        self.Nt = self.multiPointController.Nt
        self.deltaX = self.multiPointController.deltaX
        self.deltaX_usteps = self.multiPointController.deltaX_usteps
        self.deltaY = self.multiPointController.deltaY
        self.deltaY_usteps = self.multiPointController.deltaY_usteps
        self.deltaZ = self.multiPointController.deltaZ
        self.deltaZ_usteps = self.multiPointController.deltaZ_usteps
        self.dt = self.multiPointController.deltat
        self.do_autofocus = self.multiPointController.do_autofocus
        self.do_reflection_af = self.multiPointController.do_reflection_af
        self.crop_width = self.multiPointController.crop_width
        self.crop_height = self.multiPointController.crop_height
        self.display_resolution_scaling = (
            self.multiPointController.display_resolution_scaling
        )
        self.counter = self.multiPointController.counter
        self.experiment_ID = self.multiPointController.experiment_ID
        self.base_path = self.multiPointController.base_path
        self.selected_configurations = self.multiPointController.selected_configurations
        self.detection_stats = {}
        self.async_detection_stats = {}

        self.timestamp_acquisition_started = (
            self.multiPointController.timestamp_acquisition_started
        )
        self.time_point = 0

        self.microscope = self.multiPointController.parent

        self.t_dpc = []
        self.t_inf = []
        self.t_over = []

    def update_stats(self, new_stats):
        for k in new_stats.keys():
            try:
                self.detection_stats[k] += new_stats[k]
            except:
                self.detection_stats[k] = 0
                self.detection_stats[k] += new_stats[k]
        if (
            "Total RBC" in self.detection_stats
            and "Total Positives" in self.detection_stats
        ):
            self.detection_stats["Positives per 5M RBC"] = 5e6 * (
                self.detection_stats["Total Positives"]
                / self.detection_stats["Total RBC"]
            )
        if self.signal_detection_stats_callback:
            self.signal_detection_stats_callback(self.detection_stats)

    def run(self):
        self.time_point = 0 #NOTE: reset time point to 0
        self.start_time = time.perf_counter_ns()
        if self.camera.is_streaming == False:
            self.camera.start_streaming()

        if self.multiPointController.location_list is None:
            # use scanCoordinates for well plates or regular multipoint scan
            if self.multiPointController.scanCoordinates != None:
                # use scan coordinates for the scan
                self.multiPointController.scanCoordinates.get_selected_wells_to_coordinates()
                self.scan_coordinates_mm = (
                    self.multiPointController.scanCoordinates.coordinates_mm
                )
                self.scan_coordinates_name = (
                    self.multiPointController.scanCoordinates.name
                )
                self.use_scan_coordinates = True
            else:
                # use the current position for the scan
                self.scan_coordinates_mm = [
                    (
                        self.navigationController.x_pos_mm,
                        self.navigationController.y_pos_mm,
                    )
                ]
                self.scan_coordinates_name = [""]
                self.use_scan_coordinates = False
        else:
            # use location_list specified by the multipoint controlller
            self.scan_coordinates_mm = self.multiPointController.location_list
            self.scan_coordinates_name = None
            self.use_scan_coordinates = True

        while self.time_point < self.Nt:
            # check if abort acquisition has been requested
            if self.multiPointController.abort_acqusition_requested:
                break
            # run single time point
            try:
                self.run_single_time_point()
            except Exception as e:
                print("Error in run_single_time_point: " + str(e))
            print("single time point done")
            self.time_point = self.time_point + 1
            # continous acquisition
            if self.dt == 0:
                pass
            # timed acquisition
            else:
                # check if the aquisition has taken longer than dt or integer multiples of dt, if so skip the next time point(s)
                while (
                    time.time()
                    > self.timestamp_acquisition_started + self.time_point * self.dt
                ):
                    print("skip time point " + str(self.time_point + 1))
                    self.time_point = self.time_point + 1
                # check if it has reached Nt
                if self.time_point == self.Nt:
                    break  # no waiting after taking the last time point
                # wait until it's time to do the next acquisition
                while (
                    time.time()
                    < self.timestamp_acquisition_started + self.time_point * self.dt
                ):
                    if self.multiPointController.abort_acqusition_requested:
                        break
                    time.sleep(0.05)
        #self.processingHandler.processing_queue.join()
        #self.processingHandler.upload_queue.join()
        elapsed_time = time.perf_counter_ns() - self.start_time
        print("Time taken for acquisition/processing: " + str(elapsed_time / 10**9))

    def wait_till_operation_is_completed(self):
        while self.microcontroller.is_busy():
            time.sleep(CONFIG.SLEEP_TIME_S)

    def run_single_time_point(self):
        start = time.time()
        print(time.time())
        # disable joystick button action
        self.navigationController.enable_joystick_button_action = False

        print("multipoint acquisition - time point " + str(self.time_point + 1))

        # for each time point, create a new folder
        current_path = os.path.join(
            self.base_path, self.experiment_ID, str(self.time_point)
        )
        os.mkdir(current_path)

        slide_path = os.path.join(self.base_path, self.experiment_ID)

        # create a dataframe to save coordinates
        self.coordinates_pd = pd.DataFrame(
            columns=["i", "j", "k", "x (mm)", "y (mm)", "z (um)", "time"]
        )

        n_regions = len(self.scan_coordinates_mm)

        for coordinate_id in range(n_regions):

            coordiante_mm = self.scan_coordinates_mm[coordinate_id]
            print(coordiante_mm)

            if self.scan_coordinates_name is None:
                # flexible scan, use a sequencial ID
                coordiante_name = str(coordinate_id)
            else:
                coordiante_name = self.scan_coordinates_name[coordinate_id]

            if self.use_scan_coordinates:
                # move to the specified coordinate
                self.navigationController.move_x_to(
                    coordiante_mm[0] - self.deltaX * (self.NX - 1) / 2
                )
                self.navigationController.move_y_to(
                    coordiante_mm[1] - self.deltaY * (self.NY - 1) / 2
                )
                # check if z is included in the coordinate
                if len(coordiante_mm) == 3:
                    if coordiante_mm[2] >= self.navigationController.z_pos_mm:
                        self.navigationController.move_z_to(coordiante_mm[2])
                        self.wait_till_operation_is_completed()
                    else:
                        self.navigationController.move_z_to(coordiante_mm[2])
                        self.wait_till_operation_is_completed()
                        # remove backlash
                        if self.navigationController.get_pid_control_flag(2) is False:
                            _usteps_to_clear_backlash = max(
                                160, 20 * self.navigationController.z_microstepping
                            )
                            self.navigationController.move_z_usteps(
                                -_usteps_to_clear_backlash
                            )  # to-do: combine this with the above
                            self.wait_till_operation_is_completed()
                            self.navigationController.move_z_usteps(
                                _usteps_to_clear_backlash
                            )
                            self.wait_till_operation_is_completed()
                else:
                    self.wait_till_operation_is_completed()
                time.sleep(CONFIG.SCAN_STABILIZATION_TIME_MS_Y / 1000)
                if len(coordiante_mm) == 3:
                    time.sleep(CONFIG.SCAN_STABILIZATION_TIME_MS_Z / 1000)
                # add '_' to the coordinate name
                original_coordiante_name = coordiante_name
                coordiante_name = coordiante_name + "_"

            self.x_scan_direction = 1
            self.dx_usteps = 0  # accumulated x displacement
            self.dy_usteps = 0  # accumulated y displacement
            self.dz_usteps = 0  # accumulated z displacement
            z_pos = self.navigationController.z_pos  # zpos at the beginning of the scan

            # z stacking config
            if CONFIG.Z_STACKING_CONFIG == "FROM TOP":
                self.deltaZ_usteps = -abs(self.deltaZ_usteps)

            # along y
            for i in range(self.NY):

                self.FOV_counter = 0  # for CONFIG.AF, so that CONFIG.AF at the beginning of each new row

                # along x
                for j in range(self.NX):

                    if (
                        CONFIG.RUN_CUSTOM_MULTIPOINT
                    ):
                        assert CONFIG.CUSTOM_MULTIPOINT_FUNCTION is not None
                        multipoint_custom_script_entry = load_multipoint_custom_script(CONFIG.CUSTOM_MULTIPOINT_FUNCTION)

                        print("run custom multipoint")
                        multipoint_custom_script_entry(
                            self,
                            self.time_point,
                            current_path,
                            coordinate_id,
                            coordiante_name,
                            i,
                            j,
                        )

                    else:

                        # autofocus
                        if self.do_reflection_af == False:
                            # contrast-based CONFIG.AF; perform CONFIG.AF only if when not taking z stack or doing z stack from center
                            if (
                                (
                                    (self.NZ == 1)
                                    or CONFIG.Z_STACKING_CONFIG == "FROM CENTER"
                                )
                                and (self.do_autofocus)
                                and (
                                    self.FOV_counter
                                    % CONFIG.Acquisition.NUMBER_OF_FOVS_PER_AF
                                    == 0
                                )
                            ):
                                # temporary: replace the above line with the line below to CONFIG.AF every FOV
                                # if (self.NZ == 1) and (self.do_autofocus):
                                configuration_name_AF = (
                                    CONFIG.MULTIPOINT_AUTOFOCUS_CHANNEL
                                )
                                config_AF = next(

                                        config
                                        for config in self.configurationManager.configurations
                                        if config.name == configuration_name_AF

                                )
                                self.autofocusController.set_microscope_mode(config_AF)
                                print(f"autofocus at {coordiante_name}{i}_{j}, configuration: {configuration_name_AF},{config_AF}")
                                if (
                                    self.FOV_counter
                                    % CONFIG.Acquisition.NUMBER_OF_FOVS_PER_AF
                                    == 0
                                ) or self.autofocusController.use_focus_map:
                                    self.autofocusController.autofocus()
                                    self.autofocusController.wait_till_autofocus_has_completed()
                                # upate z location of scan_coordinates_mm after CONFIG.AF
                                if len(coordiante_mm) == 3:
                                    self.scan_coordinates_mm[coordinate_id, 2] = (
                                        self.navigationController.z_pos_mm
                                    )
                                    # update the coordinate in the widget
                                    try:
                                        self.microscope.multiPointWidget2._update_z(
                                            coordinate_id,
                                            self.navigationController.z_pos_mm,
                                        )
                                    except:
                                        pass
                        # initialize laser autofocus if it has not been done
                        elif (
                            self.microscope.laserAutofocusController.is_initialized
                            == False
                        ):
                            # initialize the reflection CONFIG.AF
                            self.microscope.laserAutofocusController.initialize_auto()
                            # do contrast CONFIG.AF for the first FOV (if contrast CONFIG.AF box is checked)
                            if self.do_autofocus and (
                                (self.NZ == 1)
                                or CONFIG.Z_STACKING_CONFIG == "FROM CENTER"
                            ):
                                configuration_name_AF = (
                                    CONFIG.MULTIPOINT_AUTOFOCUS_CHANNEL
                                )
                                config_AF = next(

                                        config
                                        for config in self.configurationManager.configurations
                                        if config.name == configuration_name_AF

                                )
                                self.autofocusController.set_microscope_mode(config_AF)
                                self.autofocusController.autofocus()
                                self.autofocusController.wait_till_autofocus_has_completed()
                            # set the current plane as reference
                            self.microscope.laserAutofocusController.set_reference()
                        else:
                            try:
                                if (
                                    self.navigationController.get_pid_control_flag(
                                        2
                                    )
                                    is False
                                ):
                                    self.microscope.laserAutofocusController.move_to_target(
                                        0
                                    )
                                    self.microscope.laserAutofocusController.move_to_target(
                                        0
                                    )  # for stepper in open loop mode, repeat the operation to counter backlash
                                else:
                                    self.microscope.laserAutofocusController.move_to_target(
                                        0
                                    )
                            except:
                                file_ID = (
                                    coordiante_name
                                    + str(i)
                                    + "_"
                                    + str(
                                        j
                                        if self.x_scan_direction == 1
                                        else self.NX - 1 - j
                                    )
                                )
                                saving_path = os.path.join(
                                    current_path, file_ID + "_focus_camera.bmp"
                                )
                                iio.imwrite(
                                    saving_path,
                                    self.microscope.laserAutofocusController.image,
                                )
                                print(
                                    "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! laser CONFIG.AF failed !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
                                    #raise Exception("laser CONFIG.AF failed")


                        if self.NZ > 1:
                            # move to bottom of the z stack
                            if CONFIG.Z_STACKING_CONFIG == "FROM CENTER":
                                self.navigationController.move_z_usteps(
                                    -self.deltaZ_usteps * round((self.NZ - 1) / 2)
                                )
                                self.wait_till_operation_is_completed()
                                time.sleep(CONFIG.SCAN_STABILIZATION_TIME_MS_Z / 1000)
                            # maneuver for achiving uniform step size and repeatability when using open-loop control
                            self.navigationController.move_z_usteps(-160)
                            self.wait_till_operation_is_completed()
                            self.navigationController.move_z_usteps(160)
                            self.wait_till_operation_is_completed()
                            time.sleep(CONFIG.SCAN_STABILIZATION_TIME_MS_Z / 1000)

                        # z-stack
                        for k in range(self.NZ):

                            # Ensure that i/y-indexing is always top to bottom
                            sgn_i = -1 if self.deltaY >= 0 else 1
                            if CONFIG.INVERTED_OBJECTIVE:
                                sgn_i = -sgn_i
                            sgn_j = (
                                self.x_scan_direction
                                if self.deltaX >= 0
                                else -self.x_scan_direction
                            )
                            file_ID = (
                                coordiante_name
                                + str(self.NY - 1 - i if sgn_i == -1 else i)
                                + "_"
                                + str(j if sgn_j == 1 else self.NX - 1 - j)
                                + "_"
                                + str(k)
                            )
                            # metadata = dict(x = self.navigationController.x_pos_mm, y = self.navigationController.y_pos_mm, z = self.navigationController.z_pos_mm)
                            # metadata = json.dumps(metadata)

                            # laser af characterization mode
                            if CONFIG.LASER_AF_CHARACTERIZATION_MODE:
                                image = (
                                    self.microscope.laserAutofocusController.get_image()
                                )
                                saving_path = os.path.join(
                                    current_path, file_ID + "_laser af camera" + ".bmp"
                                )
                                iio.imwrite(saving_path, image)

                            current_round_images = {}
                            # iterate through selected modes
                            for config in self.selected_configurations:
                                if (
                                    config.z_offset is not None
                                ):  # perform z offset for config, assume
                                    # z_offset is in um
                                    if config.z_offset != 0.0:
                                        print(
                                            "Moving to Z offset " + str(config.z_offset)
                                        )
                                        self.navigationController.move_z(
                                            config.z_offset / 1000
                                        )
                                        self.wait_till_operation_is_completed()
                                        time.sleep(
                                            CONFIG.SCAN_STABILIZATION_TIME_MS_Z / 1000
                                        )

                                if "USB Spectrometer" not in config.name:
                                    # update the current configuration
                                    print("current configuration: " + config.name)
                                    self.wait_till_operation_is_completed()
                                    self.liveController.set_microscope_mode(config)
                                    self.wait_till_operation_is_completed()
                                    # trigger acquisition (including turning on the illumination)
                                    if (
                                        self.liveController.trigger_mode
                                        == TriggerModeSetting.SOFTWARE
                                    ):
                                        self.liveController.turn_on_illumination()
                                        self.wait_till_operation_is_completed()
                                        self.camera.send_trigger()
                                    elif (
                                        self.liveController.trigger_mode
                                        == TriggerModeSetting.HARDWARE
                                    ):
                                        self.microcontroller.send_hardware_trigger(
                                            control_illumination=True,
                                            illumination_on_time_us=self.camera.exposure_time
                                            * 1000,
                                        )
                                    # read camera frame
                                    old_pixel_format = self.camera.pixel_format
                                    if config.pixel_format is not None:
                                        if (
                                            config.pixel_format != ""
                                            and config.pixel_format.lower() != "default"
                                        ):
                                            self.camera.set_pixel_format(
                                                config.pixel_format
                                            )

                                    image = self.camera.read_frame()

                                    if config.pixel_format is not None:
                                        if (
                                            config.pixel_format != ""
                                            and config.pixel_format.lower() != "default"
                                        ):
                                            self.camera.set_pixel_format(
                                                old_pixel_format
                                            )
                                    if image is None:
                                        print("self.camera.read_frame() returned None")
                                        continue
                                    # tunr of the illumination if using software trigger
                                    if (
                                        self.liveController.trigger_mode
                                        == TriggerModeSetting.SOFTWARE
                                    ):
                                        self.liveController.turn_off_illumination()
                                    # process the image -  @@@ to move to camera
                                    image = utils.crop_image(
                                        image, self.crop_width, self.crop_height
                                    )
                                    image = utils.rotate_and_flip_image(
                                        image,
                                        rotate_image_angle=self.camera.rotate_image_angle,
                                        flip_image=self.camera.flip_image,
                                    )
                                    image_to_display = utils.crop_image(
                                        image,
                                        round(
                                            self.crop_width
                                            * self.display_resolution_scaling
                                        ),
                                        round(
                                            self.crop_height
                                            * self.display_resolution_scaling
                                        ),
                                    )
                                    if image.dtype == np.uint16:
                                        saving_path = os.path.join(
                                            current_path,
                                            file_ID
                                            + "_"
                                            + str(config.name).replace(" ", "_")
                                            + ".tiff",
                                        )
                                        if self.camera.is_color:
                                            if "BF LED matrix" in config.name:
                                                if (
                                                    CONFIG.MULTIPOINT_BF_SAVING_OPTION
                                                    == "RGB2GRAY"
                                                ):
                                                    image = cv2.cvtColor(
                                                        image, cv2.COLOR_RGB2GRAY
                                                    )
                                                elif (
                                                    CONFIG.MULTIPOINT_BF_SAVING_OPTION
                                                    == "Green Channel Only"
                                                ):
                                                    image = image[:, :, 1]
                                        iio.imwrite(saving_path, image)
                                    else:
                                        saving_path = os.path.join(
                                            current_path,
                                            file_ID
                                            + "_"
                                            + str(config.name).replace(" ", "_")
                                            + "."
                                            + CONFIG.Acquisition.IMAGE_FORMAT,
                                        )
                                        if self.camera.is_color:
                                            if "BF LED matrix" in config.name:
                                                if (
                                                    CONFIG.MULTIPOINT_BF_SAVING_OPTION
                                                    == "Raw"
                                                ):
                                                    image = cv2.cvtColor(
                                                        image, cv2.COLOR_RGB2BGR
                                                    )
                                                elif (
                                                    CONFIG.MULTIPOINT_BF_SAVING_OPTION
                                                    == "RGB2GRAY"
                                                ):
                                                    image = cv2.cvtColor(
                                                        image, cv2.COLOR_RGB2GRAY
                                                    )
                                                elif (
                                                    CONFIG.MULTIPOINT_BF_SAVING_OPTION
                                                    == "Green Channel Only"
                                                ):
                                                    image = image[:, :, 1]
                                            else:
                                                image = cv2.cvtColor(
                                                    image, cv2.COLOR_RGB2BGR
                                                )
                                        cv2.imwrite(saving_path, image)

                                    current_round_images[config.name] = np.copy(image)

                                elif self.usb_spectrometer != None:
                                    for l in range(CONFIG.N_SPECTRUM_PER_POINT):
                                        data = self.usb_spectrometer.read_spectrum()
                                        saving_path = os.path.join(
                                            current_path,
                                            file_ID
                                            + "_"
                                            + str(config.name).replace(" ", "_")
                                            + "_"
                                            + str(l)
                                            + ".csv",
                                        )
                                        np.savetxt(saving_path, data, delimiter=",")

                                if config.z_offset is not None:  # undo Z offset
                                    # assume z_offset is in um
                                    if config.z_offset != 0.0:
                                        print(
                                            "Moving back from Z offset "
                                            + str(config.z_offset)
                                        )
                                        self.navigationController.move_z(
                                            -config.z_offset / 1000
                                        )
                                        self.wait_till_operation_is_completed()
                                        time.sleep(
                                            CONFIG.SCAN_STABILIZATION_TIME_MS_Z / 1000
                                        )

                            # add the coordinate of the current location
                            new_row = pd.DataFrame(
                                {
                                    "region": original_coordiante_name,
                                    "i": [self.NY - 1 - i if sgn_i == -1 else i],
                                    "j": [j if sgn_j == 1 else self.NX - 1 - j],
                                    "k": [k],
                                    "x (mm)": [self.navigationController.x_pos_mm],
                                    "y (mm)": [self.navigationController.y_pos_mm],
                                    "z (um)": [
                                        self.navigationController.z_pos_mm * 1000
                                    ],
                                    "time": datetime.now().strftime(
                                        "%Y-%m-%d_%H-%M-%-S.%f"
                                    ),
                                },
                            )
                            self.coordinates_pd = pd.concat(
                                [self.coordinates_pd, new_row], ignore_index=True
                            )


                            # check if the acquisition should be aborted
                            if self.multiPointController.abort_acqusition_requested:
                                self.liveController.turn_off_illumination()
                                self.navigationController.move_x_usteps(-self.dx_usteps)
                                self.wait_till_operation_is_completed()
                                self.navigationController.move_y_usteps(-self.dy_usteps)
                                self.wait_till_operation_is_completed()

                                if (
                                    self.navigationController.get_pid_control_flag(2)
                                    is False
                                ):
                                    _usteps_to_clear_backlash = max(
                                        160,
                                        20 * self.navigationController.z_microstepping,
                                    )
                                    self.navigationController.move_z_usteps(
                                        -self.dz_usteps - _usteps_to_clear_backlash
                                    )
                                    self.wait_till_operation_is_completed()
                                    self.navigationController.move_z_usteps(
                                        _usteps_to_clear_backlash
                                    )
                                    self.wait_till_operation_is_completed()
                                else:
                                    self.navigationController.move_z_usteps(
                                        -self.dz_usteps
                                    )
                                    self.wait_till_operation_is_completed()

                                self.coordinates_pd.to_csv(
                                    os.path.join(current_path, "coordinates.csv"),
                                    index=False,
                                    header=True,
                                )
                                self.navigationController.enable_joystick_button_action = (
                                    True
                                )
                                return

                            if self.NZ > 1:
                                # move z
                                if k < self.NZ - 1:
                                    self.navigationController.move_z_usteps(
                                        self.deltaZ_usteps
                                    )
                                    self.wait_till_operation_is_completed()
                                    time.sleep(
                                        CONFIG.SCAN_STABILIZATION_TIME_MS_Z / 1000
                                    )
                                    self.dz_usteps = self.dz_usteps + self.deltaZ_usteps

                        if self.NZ > 1:
                            # move z back
                            _usteps_to_clear_backlash = max(
                                160, 20 * self.navigationController.z_microstepping
                            )
                            if CONFIG.Z_STACKING_CONFIG == "FROM CENTER":
                                if (
                                    self.navigationController.get_pid_control_flag(2)
                                    is False
                                ):
                                    _usteps_to_clear_backlash = max(
                                        160,
                                        20 * self.navigationController.z_microstepping,
                                    )
                                    self.navigationController.move_z_usteps(
                                        -self.deltaZ_usteps * (self.NZ - 1)
                                        + self.deltaZ_usteps * round((self.NZ - 1) / 2)
                                        - _usteps_to_clear_backlash
                                    )
                                    self.wait_till_operation_is_completed()
                                    self.navigationController.move_z_usteps(
                                        _usteps_to_clear_backlash
                                    )
                                    self.wait_till_operation_is_completed()
                                else:
                                    self.navigationController.move_z_usteps(
                                        -self.deltaZ_usteps * (self.NZ - 1)
                                        + self.deltaZ_usteps * round((self.NZ - 1) / 2)
                                    )
                                    self.wait_till_operation_is_completed()

                                self.dz_usteps = (
                                    self.dz_usteps
                                    - self.deltaZ_usteps * (self.NZ - 1)
                                    + self.deltaZ_usteps * round((self.NZ - 1) / 2)
                                )
                            else:
                                if (
                                    self.navigationController.get_pid_control_flag(2)
                                    is False
                                ):
                                    _usteps_to_clear_backlash = max(
                                        160,
                                        20 * self.navigationController.z_microstepping,
                                    )
                                    self.navigationController.move_z_usteps(
                                        -self.deltaZ_usteps * (self.NZ - 1)
                                        - _usteps_to_clear_backlash
                                    )
                                    self.wait_till_operation_is_completed()
                                    self.navigationController.move_z_usteps(
                                        _usteps_to_clear_backlash
                                    )
                                    self.wait_till_operation_is_completed()
                                else:
                                    self.navigationController.move_z_usteps(
                                        -self.deltaZ_usteps * (self.NZ - 1)
                                    )
                                    self.wait_till_operation_is_completed()

                                self.dz_usteps = self.dz_usteps - self.deltaZ_usteps * (
                                    self.NZ - 1
                                )

                        # update FOV counter
                        self.FOV_counter = self.FOV_counter + 1

                    if self.NX > 1:
                        # move x
                        if j < self.NX - 1:
                            self.navigationController.move_x_usteps(
                                self.x_scan_direction * self.deltaX_usteps
                            )
                            self.wait_till_operation_is_completed()
                            time.sleep(CONFIG.SCAN_STABILIZATION_TIME_MS_X / 1000)
                            self.dx_usteps = (
                                self.dx_usteps
                                + self.x_scan_direction * self.deltaX_usteps
                            )

                # finished X scan
                """
                # instead of move back, reverse scan direction (12/29/2021)
                if self.NX > 1:
                    # move x back
                    self.navigationController.move_x_usteps(-self.deltaX_usteps*(self.NX-1))
                    self.wait_till_operation_is_completed()
                    time.sleep(CONFIG.SCAN_STABILIZATION_TIME_MS_X/1000)
                """
                self.x_scan_direction = -self.x_scan_direction

                if self.NY > 1:
                    # move y
                    if i < self.NY - 1:
                        self.navigationController.move_y_usteps(self.deltaY_usteps)
                        self.wait_till_operation_is_completed()
                        time.sleep(CONFIG.SCAN_STABILIZATION_TIME_MS_Y / 1000)
                        self.dy_usteps = self.dy_usteps + self.deltaY_usteps

            # finished XY scan
            if n_regions == 1:
                # only move to the start position if there's only one region in the scan
                if self.NY > 1:
                    # move y back
                    self.navigationController.move_y_usteps(
                        -self.deltaY_usteps * (self.NY - 1)
                    )
                    self.wait_till_operation_is_completed()
                    time.sleep(CONFIG.SCAN_STABILIZATION_TIME_MS_Y / 1000)
                    self.dy_usteps = self.dy_usteps - self.deltaY_usteps * (self.NY - 1)

                # move x back at the end of the scan
                if self.x_scan_direction == -1:
                    self.navigationController.move_x_usteps(
                        -self.deltaX_usteps * (self.NX - 1)
                    )
                    self.wait_till_operation_is_completed()
                    time.sleep(CONFIG.SCAN_STABILIZATION_TIME_MS_X / 1000)

                # move z back
                if self.navigationController.get_pid_control_flag(2) is False:
                    _usteps_to_clear_backlash = max(
                        160, 20 * self.navigationController.z_microstepping
                    )
                    self.navigationController.microcontroller.move_z_to_usteps(
                        z_pos - CONFIG.STAGE_MOVEMENT_SIGN_Z * _usteps_to_clear_backlash
                    )
                    self.wait_till_operation_is_completed()
                    self.navigationController.move_z_usteps(_usteps_to_clear_backlash)
                    self.wait_till_operation_is_completed()
                else:
                    self.navigationController.microcontroller.move_z_to_usteps(z_pos)
                    self.wait_till_operation_is_completed()

        # finished region scan
        self.coordinates_pd.to_csv(
            os.path.join(current_path, "coordinates.csv"), index=False, header=True
        )
        self.navigationController.enable_joystick_button_action = True
        print(time.time())
        print(time.time() - start)
        return

class MultiPointController:

    acquisitionFinished = None
    image_to_display = None
    image_to_display_multi = None
    spectrum_to_display = None
    signal_current_configuration = None
    signal_register_current_fov = None
    detection_stats = None

    def __init__(
        self,
        camera,
        navigationController,
        liveController,
        autofocusController,
        configurationManager,
        usb_spectrometer=None,
        scanCoordinates=None,
        parent=None,
    ):
        self.camera = camera
        self.processingHandler = ProcessingHandler()
        self.microcontroller = (
            navigationController.microcontroller
        )  # to move to gui for transparency
        self.navigationController = navigationController
        self.liveController = liveController
        self.autofocusController = autofocusController
        self.configurationManager = configurationManager
        self.NX = 1
        self.NY = 1
        self.NZ = 1
        self.Nt = 1
        mm_per_ustep_X = CONFIG.SCREW_PITCH_X_MM / (
            self.navigationController.x_microstepping * CONFIG.FULLSTEPS_PER_REV_X
        )
        mm_per_ustep_Y = CONFIG.SCREW_PITCH_Y_MM / (
            self.navigationController.y_microstepping * CONFIG.FULLSTEPS_PER_REV_Y
        )
        mm_per_ustep_Z = CONFIG.SCREW_PITCH_Z_MM / (
            self.navigationController.z_microstepping * CONFIG.FULLSTEPS_PER_REV_Z
        )
        self.deltaX = CONFIG.Acquisition.DX
        self.deltaX_usteps = round(self.deltaX / mm_per_ustep_X)
        self.deltaY = CONFIG.Acquisition.DY
        self.deltaY_usteps = round(self.deltaY / mm_per_ustep_Y)
        self.deltaZ = CONFIG.Acquisition.DZ / 1000
        self.deltaZ_usteps = round(self.deltaZ / mm_per_ustep_Z)
        self.deltat = 0
        self.do_autofocus = False
        self.do_reflection_af = False
        self.gen_focus_map = False
        self.focus_map_storage = []
        self.already_using_fmap = False
        self.crop_width = CONFIG.Acquisition.CROP_WIDTH
        self.crop_height = CONFIG.Acquisition.CROP_HEIGHT
        self.display_resolution_scaling = (
            CONFIG.Acquisition.IMAGE_DISPLAY_SCALING_FACTOR
        )
        self.counter = 0
        self.experiment_ID = None
        self.base_path = None
        self.selected_configurations = []
        self.usb_spectrometer = usb_spectrometer
        self.scanCoordinates = scanCoordinates
        self.parent = parent

        self.old_images_per_page = 1
        try:
            if self.parent is not None:
                self.old_images_per_page = self.parent.dataHandler.n_images_per_page
        except:
            pass
        self.location_list = None  # for flexible multipoint

    def set_NX(self, N):
        self.NX = N

    def set_NY(self, N):
        self.NY = N

    def set_NZ(self, N):
        self.NZ = N

    def set_Nt(self, N):
        self.Nt = N

    def set_deltaX(self, delta):
        mm_per_ustep_X = CONFIG.SCREW_PITCH_X_MM / (
            self.navigationController.x_microstepping * CONFIG.FULLSTEPS_PER_REV_X
        )
        self.deltaX = delta
        self.deltaX_usteps = round(delta / mm_per_ustep_X)

    def set_deltaY(self, delta):
        mm_per_ustep_Y = CONFIG.SCREW_PITCH_Y_MM / (
            self.navigationController.y_microstepping * CONFIG.FULLSTEPS_PER_REV_Y
        )
        self.deltaY = delta
        self.deltaY_usteps = round(delta / mm_per_ustep_Y)

    def set_deltaZ(self, delta_um):
        mm_per_ustep_Z = CONFIG.SCREW_PITCH_Z_MM / (
            self.navigationController.z_microstepping * CONFIG.FULLSTEPS_PER_REV_Z
        )
        self.deltaZ = delta_um / 1000
        self.deltaZ_usteps = round((delta_um / 1000) / mm_per_ustep_Z)

    def set_deltat(self, delta):
        self.deltat = delta

    def set_af_flag(self, flag):
        self.do_autofocus = flag

    def set_reflection_af_flag(self, flag):
        self.do_reflection_af = flag

    def set_gen_focus_map_flag(self, flag):
        self.gen_focus_map = flag
        if not flag:
            self.autofocusController.set_focus_map_use(False)

    def set_crop(self, crop_width, crop_height):
        self.crop_width = crop_width
        self.crop_height = crop_height

    def set_base_path(self, path):
        self.base_path = path

    def start_new_experiment(
        self, experiment_ID
    ):  # @@@ to do: change name to prepare_folder_for_new_experiment
        # generate unique experiment ID
        self.experiment_ID = (
            experiment_ID.replace(" ", "_")
            + "_"
            + datetime.now().strftime("%Y-%m-%d_%H-%M-%-S.%f")
        )
        self.recording_start_time = time.time()
        # create a new folder
        os.mkdir(os.path.join(self.base_path, self.experiment_ID))
        configManagerThrowaway = ConfigurationManager(
            self.configurationManager.config_filename
        )
        configManagerThrowaway.write_configuration_selected(
            self.selected_configurations,
            os.path.join(self.base_path, self.experiment_ID) + "/configurations.xml",
        )  # save the configuration for the experiment
        acquisition_parameters = {
            "dx(mm)": self.deltaX,
            "Nx": self.NX,
            "dy(mm)": self.deltaY,
            "Ny": self.NY,
            "dz(um)": self.deltaZ * 1000,
            "Nz": self.NZ,
            "dt(s)": self.deltat,
            "Nt": self.Nt,
            "with CONFIG.AF": self.do_autofocus,
            "with reflection CONFIG.AF": self.do_reflection_af,
        }
        try:  # write objective data if it is available
            current_objective = self.parent.objectiveStore.current_objective
            objective_info = self.parent.objectiveStore.objectives_dict.get(
                current_objective, {}
            )
            acquisition_parameters["objective"] = {}
            for k in objective_info.keys():
                acquisition_parameters["objective"][k] = objective_info[k]
            acquisition_parameters["objective"]["name"] = current_objective
        except:
            try:
                objective_info = CONFIG.OBJECTIVES[CONFIG.DEFAULT_OBJECTIVE]
                acquisition_parameters["objective"] = {}
                for k in objective_info.keys():
                    acquisition_parameters["objective"][k] = objective_info[k]
                acquisition_parameters["objective"]["name"] = CONFIG.DEFAULT_OBJECTIVE
            except:
                pass
        acquisition_parameters["sensor_pixel_size_um"] = CONFIG.CAMERA_PIXEL_SIZE_UM[
            CONFIG.CAMERA_SENSOR
        ]
        acquisition_parameters["tube_lens_mm"] = CONFIG.TUBE_LENS_MM
        f = open(
            os.path.join(self.base_path, self.experiment_ID)
            + "/acquisition parameters.json",
            "w",
        )
        f.write(json.dumps(acquisition_parameters))
        f.close()

    def set_selected_configurations(self, selected_configurations_name):
        self.selected_configurations = []
        for configuration_name in selected_configurations_name:
            self.selected_configurations.append(
                next(

                        config
                        for config in self.configurationManager.configurations
                        if config.name == configuration_name

                )
            )

    def set_selected_configurations_with_settings(self, illumination_settings):
        """
        Set selected configurations with custom illumination settings.
        Updates the original configurations directly so the custom settings 
        will be saved in the experiment metadata.
        
        Args:
            illumination_settings (list): List of dictionaries containing:
                - 'channel': Channel name (str)
                - 'intensity': Illumination intensity (float, 0-100)
                - 'exposure_time': Exposure time in ms (float)
        """
        self.selected_configurations = []

        for setting in illumination_settings:
            channel_name = setting['channel']
            intensity = setting['intensity']
            exposure_time = setting['exposure_time']

            # Find the original configuration by name
            original_config = None
            for cfg in self.configurationManager.configurations:
                if cfg.name == channel_name:
                    original_config = cfg
                    break

            if original_config is None:
                print(f"Warning: Configuration '{channel_name}' not found, skipping...")
                continue

            # UPDATE the original configuration directly with new settings
            # This ensures the custom values will be saved in the experiment metadata
            original_config.illumination_intensity = float(intensity)
            original_config.exposure_time = float(exposure_time)

            # Add the updated configuration to selected configurations
            self.selected_configurations.append(original_config)

            print(f"Updated configuration '{channel_name}': intensity={intensity}, exposure_time={exposure_time}")

        print(f"Selected {len(self.selected_configurations)} configurations with custom settings")

    def run_acquisition(self, location_list=None):
        print('start acquisition')
        self.tile_stitchers = {}
        print(str(self.Nt) + '_' + str(self.NX) + '_' + str(self.NY) + '_' + str(self.NZ))
        if location_list is not None:
            self.location_list = location_list


        self.abort_acqusition_requested = False

        # Store current configuration to restore later
        self.configuration_before_running_multipoint = self.liveController.currentConfiguration

        # Stop live view if active
        if self.liveController.is_live:
            self.liveController_was_live_before_multipoint = True
            self.liveController.stop_live()
        else:
            self.liveController_was_live_before_multipoint = False

        # Disable camera callback
        if self.camera.callback_is_enabled:
            self.camera_callback_was_enabled_before_multipoint = True
            self.camera.disable_callback()
        else:
            self.camera_callback_was_enabled_before_multipoint = False

        # Handle spectrometer if present
        if self.usb_spectrometer != None:
            if self.usb_spectrometer.streaming_started == True and self.usb_spectrometer.streaming_paused == False:
                self.usb_spectrometer.pause_streaming()
                self.usb_spectrometer_was_streaming = True
            else:
                self.usb_spectrometer_was_streaming = False

        if self.parent is not None:
            try:
                self.parent.imageDisplayTabs.setCurrentWidget(self.parent.imageArrayDisplayWindow.widget)
            except:
                pass
            try:
                self.parent.recordTabWidget.setCurrentWidget(self.parent.statsDisplayWidget)
            except:
                pass

        # Start acquisition
        self.timestamp_acquisition_started = time.time()

        # Start processing
        #self.processingHandler.start_processing()
        #self.processingHandler.start_uploading()

        # Create worker but run directly without thread
        self.multiPointWorker = MultiPointWorker(self)

        # Connect signals directly - they'll still work for direct method calls
        #self.multiPointWorker.signal_detection_stats.connect(self.slot_detection_stats)
        #self.multiPointWorker.image_to_display.connect(self.slot_image_to_display)
        #self.multiPointWorker.image_to_display_multi.connect(self.slot_image_to_display_multi)
        #self.multiPointWorker.spectrum_to_display.connect(self.slot_spectrum_to_display)
        #self.multiPointWorker.signal_current_configuration.connect(self.slot_current_configuration)
        #self.multiPointWorker.signal_register_current_fov.connect(self.slot_register_current_fov)

        try:
            # Run the acquisition directly without threading
            self.multiPointWorker.run()
        except Exception as e:
            print(f"Error in acquisition: {str(e)}")
        finally:
            # Always clean up properly
            self._on_acquisition_completed()

    def _on_acquisition_completed(self):
        # restore the previous selected mode
        if self.gen_focus_map:
            self.autofocusController.clear_focus_map()
            for x, y, z in self.focus_map_storage:
                self.autofocusController.focus_map_coords.append((x, y, z))
            self.autofocusController.use_focus_map = self.already_using_fmap
        # self.signal_current_configuration.emit(
        #     self.configuration_before_running_multipoint
        # )

        # re-enable callback
        if self.camera_callback_was_enabled_before_multipoint:
            self.camera.enable_callback()
            self.camera_callback_was_enabled_before_multipoint = False

        # re-enable live if it's previously on
        if self.liveController_was_live_before_multipoint:
            self.liveController.start_live()

        if self.usb_spectrometer != None:
            if self.usb_spectrometer_was_streaming:
                self.usb_spectrometer.resume_streaming()

        # emit the acquisition finished signal to enable the UI
        self.processingHandler.end_processing()
        if self.parent is not None:
            try:
                self.parent.dataHandler.set_number_of_images_per_page(
                    self.old_images_per_page
                )
                self.parent.dataHandler.sort("Sort by prediction score")
            except:
                pass

    def request_abort_aquisition(self):
        self.abort_acqusition_requested = True

class ConfigurationManager:
    def __init__(self, filename=CONFIG.CHANNEL_CONFIGURATIONS_PATH):
        # Ensure we have an absolute path to prevent working directory issues
        if not os.path.isabs(filename):
            # Convert relative path to absolute path relative to the package directory
            # __file__ is in squid_control/control/core.py, so we need to go up 1 level to get to squid_control/
            package_dir = os.path.dirname(__file__)  # This gives us squid_control/control/
            package_dir = os.path.dirname(package_dir)  # This gives us squid_control/
            self.config_filename = os.path.join(package_dir, os.path.basename(filename))
        else:
            self.config_filename = filename

        print(f"Illumination configurations file: {self.config_filename}")
        self.configurations = []
        self.read_configurations()

    def save_configurations(self):
        self.write_configuration(self.config_filename)

    def write_configuration(self, filename):
        self.config_xml_tree.write(
            filename, encoding="utf-8", xml_declaration=True, pretty_print=True
        )

    def read_configurations(self):
        if not os.path.isfile(self.config_filename):
            # Don't auto-generate files during testing - this can cause issues
            if 'PYTEST_CURRENT_TEST' in os.environ:
                raise FileNotFoundError(f"Configuration file not found: {self.config_filename}. "
                                      f"Please ensure the file exists in the squid_control package directory.")
            else:
                utils_config.generate_default_configuration(self.config_filename)
        self.config_xml_tree = ET.parse(self.config_filename)
        self.config_xml_tree_root = self.config_xml_tree.getroot()
        self.num_configurations = 0
        for mode in self.config_xml_tree_root.iter("mode"):
            self.num_configurations = self.num_configurations + 1
            self.configurations.append(
                Configuration(
                    mode_id=mode.get("ID"),
                    name=mode.get("Name"),
                    exposure_time=float(mode.get("ExposureTime")),
                    analog_gain=float(mode.get("AnalogGain")),
                    illumination_source=int(mode.get("IlluminationSource")),
                    illumination_intensity=float(mode.get("IlluminationIntensity")),
                    camera_sn=mode.get("CameraSN"),
                    z_offset=float(mode.get("ZOffset")),
                    pixel_format=mode.get("PixelFormat"),
                    _pixel_format_options=mode.get("_PixelFormat_options"),
                )
            )

    def update_configuration(self, configuration_id, attribute_name, new_value):
        conf_list = self.config_xml_tree_root.xpath(
            "//mode[contains(@ID," + "'" + str(configuration_id) + "')]"
        )
        mode_to_update = conf_list[0]
        mode_to_update.set(attribute_name, str(new_value))
        self.save_configurations()

    def update_configuration_without_writing(
        self, configuration_id, attribute_name, new_value
    ):
        conf_list = self.config_xml_tree_root.xpath(
            "//mode[contains(@ID," + "'" + str(configuration_id) + "')]"
        )
        mode_to_update = conf_list[0]
        mode_to_update.set(attribute_name, str(new_value))

    def write_configuration_selected(
        self, selected_configurations, filename
    ):  # to be only used with a throwaway instance
        # of this class
        for conf in self.configurations:
            self.update_configuration_without_writing(conf.id, "Selected", 0)
        for conf in selected_configurations:
            # Update the actual configuration values from the selected configurations
            # This ensures custom illumination settings are saved in the XML
            self.update_configuration_without_writing(conf.id, "ExposureTime", conf.exposure_time)
            self.update_configuration_without_writing(conf.id, "IlluminationIntensity", conf.illumination_intensity)
            self.update_configuration_without_writing(conf.id, "Selected", 1)
        self.write_configuration(filename)
        for conf in selected_configurations:
            self.update_configuration_without_writing(conf.id, "Selected", 0)


class PlateReaderNavigationController:

    signal_homing_complete = None
    signal_current_well = None

    def __init__(self, microcontroller):
        self.microcontroller = microcontroller
        self.x_pos_mm = 0
        self.y_pos_mm = 0
        self.z_pos_mm = 0
        self.z_pos = 0
        self.x_microstepping = CONFIG.MICROSTEPPING_DEFAULT_X
        self.y_microstepping = CONFIG.MICROSTEPPING_DEFAULT_Y
        self.z_microstepping = CONFIG.MICROSTEPPING_DEFAULT_Z
        self.column = ""
        self.row = ""

        # to be moved to gui for transparency
        self.microcontroller.set_callback(self.update_pos)

        self.is_homing = False
        self.is_scanning = False

    def move_x_usteps(self, usteps):
        self.microcontroller.move_x_usteps(usteps)

    def move_y_usteps(self, usteps):
        self.microcontroller.move_y_usteps(usteps)

    def move_z_usteps(self, usteps):
        self.microcontroller.move_z_usteps(usteps)

    def move_x_to_usteps(self, usteps):
        self.microcontroller.move_x_to_usteps(usteps)

    def move_y_to_usteps(self, usteps):
        self.microcontroller.move_y_to_usteps(usteps)

    def move_z_to_usteps(self, usteps):
        self.microcontroller.move_z_to_usteps(usteps)

    def moveto(self, column, row):
        if column != "":
            mm_per_ustep_X = CONFIG.SCREW_PITCH_X_MM / (
                self.x_microstepping * CONFIG.FULLSTEPS_PER_REV_X
            )
            x_mm = (
                CONFIG.PLATE_READER.OFFSET_COLUMN_1_MM
                + (int(column) - 1) * CONFIG.PLATE_READER.COLUMN_SPACING_MM
            )
            x_usteps = CONFIG.STAGE_MOVEMENT_SIGN_X * round(x_mm / mm_per_ustep_X)
            self.move_x_to_usteps(x_usteps)
        if row != "":
            mm_per_ustep_Y = CONFIG.SCREW_PITCH_Y_MM / (
                self.y_microstepping * CONFIG.FULLSTEPS_PER_REV_Y
            )
            y_mm = (
                CONFIG.PLATE_READER.OFFSET_ROW_A_MM
                + (ord(row) - ord("A")) * CONFIG.PLATE_READER.ROW_SPACING_MM
            )
            y_usteps = CONFIG.STAGE_MOVEMENT_SIGN_Y * round(y_mm / mm_per_ustep_Y)
            self.move_y_to_usteps(y_usteps)

    def moveto_row(self, row):
        # row: int, starting from 0
        mm_per_ustep_Y = CONFIG.SCREW_PITCH_Y_MM / (
            self.y_microstepping * CONFIG.FULLSTEPS_PER_REV_Y
        )
        y_mm = (
            CONFIG.PLATE_READER.OFFSET_ROW_A_MM
            + row * CONFIG.PLATE_READER.ROW_SPACING_MM
        )
        y_usteps = round(y_mm / mm_per_ustep_Y)
        self.move_y_to_usteps(y_usteps)

    def moveto_column(self, column):
        # column: int, starting from 0
        mm_per_ustep_X = CONFIG.SCREW_PITCH_X_MM / (
            self.x_microstepping * CONFIG.FULLSTEPS_PER_REV_X
        )
        x_mm = (
            CONFIG.PLATE_READER.OFFSET_COLUMN_1_MM
            + column * CONFIG.PLATE_READER.COLUMN_SPACING_MM
        )
        x_usteps = round(x_mm / mm_per_ustep_X)
        self.move_x_to_usteps(x_usteps)

    def update_pos(self, microcontroller):
        # get position from the microcontroller
        x_pos, y_pos, z_pos, theta_pos = microcontroller.get_pos()
        self.z_pos = z_pos
        # calculate position in mm or rad
        if CONFIG.USE_ENCODER_X:
            self.x_pos_mm = (
                x_pos * CONFIG.STAGE_POS_SIGN_X * CONFIG.ENCODER_STEP_SIZE_X_MM
            )
        else:
            self.x_pos_mm = (
                x_pos
                * CONFIG.STAGE_POS_SIGN_X
                * (
                    CONFIG.SCREW_PITCH_X_MM
                    / (self.x_microstepping * CONFIG.FULLSTEPS_PER_REV_X)
                )
            )
        if CONFIG.USE_ENCODER_Y:
            self.y_pos_mm = (
                y_pos * CONFIG.STAGE_POS_SIGN_Y * CONFIG.ENCODER_STEP_SIZE_Y_MM
            )
        else:
            self.y_pos_mm = (
                y_pos
                * CONFIG.STAGE_POS_SIGN_Y
                * (
                    CONFIG.SCREW_PITCH_Y_MM
                    / (self.y_microstepping * CONFIG.FULLSTEPS_PER_REV_Y)
                )
            )
        if CONFIG.USE_ENCODER_Z:
            self.z_pos_mm = (
                z_pos * CONFIG.STAGE_POS_SIGN_Z * CONFIG.ENCODER_STEP_SIZE_Z_MM
            )
        else:
            self.z_pos_mm = (
                z_pos
                * CONFIG.STAGE_POS_SIGN_Z
                * (
                    CONFIG.SCREW_PITCH_Z_MM
                    / (self.z_microstepping * CONFIG.FULLSTEPS_PER_REV_Z)
                )
            )
        # check homing status
        if (
            self.is_homing
            and self.microcontroller.mcu_cmd_execution_in_progress == False
            and self.signal_homing_complete
        ):
            self.signal_homing_complete()
        # for debugging
        # print('X: ' + str(self.x_pos_mm) + ' Y: ' + str(self.y_pos_mm))
        # check and emit current position
        column = round(
            (self.x_pos_mm - CONFIG.PLATE_READER.OFFSET_COLUMN_1_MM)
            / CONFIG.PLATE_READER.COLUMN_SPACING_MM
        )
        if column >= 0 and column <= CONFIG.PLATE_READER.NUMBER_OF_COLUMNS:
            column = str(column + 1)
        else:
            column = " "
        row = round(
            (self.y_pos_mm - CONFIG.PLATE_READER.OFFSET_ROW_A_MM)
            / CONFIG.PLATE_READER.ROW_SPACING_MM
        )
        if row >= 0 and row <= CONFIG.PLATE_READER.NUMBER_OF_ROWS:
            row = chr(ord("A") + row)
        else:
            row = " "


    def home(self):
        self.is_homing = True
        self.microcontroller.home_xy()

    def home_x(self):
        self.microcontroller.home_x()

    def home_y(self):
        self.microcontroller.home_y()

class WellSelector:
    def __init__(self, rows=8, columns=12):
        self.rows = rows
        self.columns = columns
        self.selected_wells = []  # Initialize as an empty list
        self.selected_wells_names = []

    def get_selected_wells(self):
        list_of_selected_cells = []
        self.selected_wells_names = []
        if not self.selected_wells:
            print("No wells selected, will call 'set_selected_wells' first")
            self.set_selected_wells((0, 0), (self.rows, self.columns))
            print("selected wells:", self.selected_wells)
        for well in self.selected_wells:
            row, col = well
            list_of_selected_cells.append((row, col))
            self.selected_wells_names.append(chr(ord("A") + row) + str(col + 1))
        if list_of_selected_cells:
            print("cells:", list_of_selected_cells)
        else:
            print("no cells")
        return list_of_selected_cells

    def set_selected_wells(self, start, stop):
        """
        Set the selected wells based on the start and stop coordinates
        input:
        start: tuple, (row, column)
        stop: tuple, (row, column)

        """
        self.selected_wells = []
        start_row, start_col = start
        stop_row, stop_col = stop
        for row in range(start_row, stop_row + 1):
            for col in range(start_col, stop_col + 1):
                self.selected_wells.append((row, col))

class ScanCoordinates:
    def __init__(self):
        self.coordinates_mm = []
        self.name = []
        self.well_selector = WellSelector()

    def add_well_selector(self, well_selector):
        self.well_selector = well_selector

    def get_selected_wells_to_coordinates(self, wellplate_type='96', is_simulation=False):
        """
        Convert selected wells to coordinates using the same logic as move_to_well function.
        
        Args:
            wellplate_type (str): Type of well plate ('6', '12', '24', '96', '384')
            is_simulation (bool): Whether in simulation mode (affects offset application)
        """
        # Import wellplate format classes
        from squid_control.control.config import (
            CONFIG,
            WELLPLATE_FORMAT_6,
            WELLPLATE_FORMAT_12,
            WELLPLATE_FORMAT_24,
            WELLPLATE_FORMAT_96,
            WELLPLATE_FORMAT_384,
        )

        # Get well plate format configuration - same logic as move_to_well
        if wellplate_type == '6':
            wellplate_format = WELLPLATE_FORMAT_6
        elif wellplate_type == '12':
            wellplate_format = WELLPLATE_FORMAT_12
        elif wellplate_type == '24':
            wellplate_format = WELLPLATE_FORMAT_24
        elif wellplate_type == '96':
            wellplate_format = WELLPLATE_FORMAT_96
        elif wellplate_type == '384':
            wellplate_format = WELLPLATE_FORMAT_384
        else:
            # Default to 96-well plate if unsupported type is provided
            wellplate_format = WELLPLATE_FORMAT_96

        # get selected wells from the widget
        selected_wells = self.well_selector.get_selected_wells()
        selected_wells = np.array(selected_wells)
        # clear the previous selection
        self.coordinates_mm = []
        self.name = []
        # populate the coordinates
        rows = np.unique(selected_wells[:, 0])
        _increasing = True
        for row in rows:
            items = selected_wells[selected_wells[:, 0] == row]
            columns = items[:, 1]
            columns = np.sort(columns)
            if _increasing == False:
                columns = np.flip(columns)
            for column in columns:
                # Use the same coordinate calculation as move_to_well function
                if is_simulation:
                    x_mm = wellplate_format.A1_X_MM + column * wellplate_format.WELL_SPACING_MM
                    y_mm = wellplate_format.A1_Y_MM + row * wellplate_format.WELL_SPACING_MM
                else:
                    x_mm = wellplate_format.A1_X_MM + column * wellplate_format.WELL_SPACING_MM + CONFIG.WELLPLATE_OFFSET_X_MM
                    y_mm = wellplate_format.A1_Y_MM + row * wellplate_format.WELL_SPACING_MM + CONFIG.WELLPLATE_OFFSET_Y_MM

                self.coordinates_mm.append((x_mm, y_mm))
                self.name.append(chr(ord("A") + row) + str(column + 1))
            _increasing = not _increasing


class LaserAutofocusController:

    image_to_display = None
    signal_displacement_um = None

    def __init__(
        self,
        microcontroller,
        camera,
        liveController,
        navigationController,
        has_two_interfaces=True,
        use_glass_top=True,
        look_for_cache=True,
    ):
        self.microcontroller = microcontroller
        self.camera = camera
        self.liveController = liveController
        self.navigationController = navigationController

        self.is_initialized = False
        self.x_reference = 0
        self.pixel_to_um = 1
        self.x_offset = 0
        self.y_offset = 0
        self.x_width = 3088
        self.y_width = 2064

        self.has_two_interfaces = has_two_interfaces  # e.g. air-glass and glass water, set to false when (1) using oil immersion (2) using 1 mm thick slide (3) using metal coated slide or Si wafer
        self.use_glass_top = use_glass_top
        self.spot_spacing_pixels = (
            None  # spacing between the spots from the two interfaces (unit: pixel)
        )

        self.look_for_cache = look_for_cache

        self.image = None  # for saving the focus camera image for debugging when centroid cannot be found

        if look_for_cache:
            cache_path = "cache/laser_af_reference_plane.txt"
            try:
                with open(cache_path) as cache_file:
                    for line in cache_file:
                        value_list = line.split(",")
                        x_offset = float(value_list[0])
                        y_offset = float(value_list[1])
                        width = int(value_list[2])
                        height = int(value_list[3])
                        pixel_to_um = float(value_list[4])
                        x_reference = float(value_list[5])
                        self.initialize_manual(
                            x_offset, y_offset, width, height, pixel_to_um, x_reference,write_to_cache=False
                        )
                        break
            except (FileNotFoundError, ValueError, IndexError) as e:
                print("Unable to read laser CONFIG.AF state cache, exception below:")
                print(e)
                pass

    def initialize_manual(
        self,
        x_offset,
        y_offset,
        width,
        height,
        pixel_to_um,
        x_reference,
        write_to_cache=True,
    ):
        cache_string = ",".join(
            [
                str(x_offset),
                str(y_offset),
                str(width),
                str(height),
                str(pixel_to_um),
                str(x_reference),
            ]
        )
        if write_to_cache:
            cache_path = Path("cache/laser_af_reference_plane.txt")
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            cache_path.write_text(cache_string)
        # x_reference is relative to the full sensor
        self.pixel_to_um = pixel_to_um
        self.x_offset = int((x_offset // 8) * 8)
        self.y_offset = int((y_offset // 2) * 2)
        self.width = int((width // 8) * 8)
        self.height = int((height // 2) * 2)
        self.x_reference = (
            x_reference - self.x_offset
        )  # self.x_reference is relative to the cropped region
        self.camera.set_ROI(self.x_offset, self.y_offset, self.width, self.height)
        self.camera.set_exposure_time(CONFIG.FOCUS_CAMERA_EXPOSURE_TIME_MS)
        self.camera.set_analog_gain(CONFIG.FOCUS_CAMERA_ANALOG_GAIN)

        self.is_initialized = True

    def initialize_auto(self):

        # first find the region to crop
        # then calculate the convert factor

        # set camera to use full sensor
        self.camera.set_ROI(0, 0, None, None)  # set offset first
        self.camera.set_ROI(0, 0, 3088, 2064)
        # update camera settings
        self.camera.set_exposure_time(CONFIG.FOCUS_CAMERA_EXPOSURE_TIME_MS)
        self.camera.set_analog_gain(CONFIG.FOCUS_CAMERA_ANALOG_GAIN)

        # turn on the laser
        self.microcontroller.turn_on_AF_laser()
        self.wait_till_operation_is_completed()

        # get laser spot location
        x, y = self._get_laser_spot_centroid()

        # turn off the laser
        self.microcontroller.turn_off_AF_laser()
        self.wait_till_operation_is_completed()

        x_offset = x - CONFIG.LASER_AF_CROP_WIDTH / 2
        y_offset = y - CONFIG.LASER_AF_CROP_HEIGHT / 2
        print(
            "laser spot location on the full sensor is ("
            + str(int(x))
            + ","
            + str(int(y))
            + ")"
        )

        # set camera crop
        self.initialize_manual(
            x_offset,
            y_offset,
            CONFIG.LASER_AF_CROP_WIDTH,
            CONFIG.LASER_AF_CROP_HEIGHT,
            1,
            x,
        )

        # turn on laser
        self.microcontroller.turn_on_AF_laser()
        self.wait_till_operation_is_completed()

        # move z to - 6 um
        self.navigationController.move_z(-0.018)
        self.wait_till_operation_is_completed()
        self.navigationController.move_z(0.012)
        self.wait_till_operation_is_completed()
        time.sleep(0.02)

        # measure
        x0, y0 = self._get_laser_spot_centroid()

        # move z to 6 um
        self.navigationController.move_z(0.006)
        self.wait_till_operation_is_completed()
        time.sleep(0.02)

        # measure
        x1, y1 = self._get_laser_spot_centroid()

        # turn off laser
        self.microcontroller.turn_off_AF_laser()
        self.wait_till_operation_is_completed()

        # calculate the conversion factor
        self.pixel_to_um = 6.0 / (x1 - x0)
        print("pixel to um conversion factor is " + str(self.pixel_to_um) + " um/pixel")
        # for simulation
        if x1 - x0 == 0:
            self.pixel_to_um = 0.4

        # set reference
        self.x_reference = x1

        if self.look_for_cache:
            cache_path = "cache/laser_af_reference_plane.txt"
            try:
                x_offset = None
                y_offset = None
                width = None
                height = None
                pixel_to_um = None
                x_reference = None
                with open(cache_path) as cache_file:
                    for line in cache_file:
                        value_list = line.split(",")
                        x_offset = float(value_list[0])
                        y_offset = float(value_list[1])
                        width = int(value_list[2])
                        height = int(value_list[3])
                        pixel_to_um = self.pixel_to_um
                        x_reference = self.x_reference + self.x_offset
                        break
                cache_string = ",".join(
                    [
                        str(x_offset),
                        str(y_offset),
                        str(width),
                        str(height),
                        str(pixel_to_um),
                        str(x_reference),
                    ]
                )
                cache_path = Path("cache/laser_af_reference_plane.txt")
                cache_path.parent.mkdir(parents=True, exist_ok=True)
                cache_path.write_text(cache_string)
            except (FileNotFoundError, ValueError, IndexError) as e:
                print("Unable to read laser CONFIG.AF state cache, exception below:")
                print(e)
                pass

    def measure_displacement(self):
        # turn on the laser
        self.microcontroller.turn_on_AF_laser()
        self.wait_till_operation_is_completed()
        # get laser spot location
        x, y = self._get_laser_spot_centroid()
        # turn off the laser
        self.microcontroller.turn_off_AF_laser()
        self.wait_till_operation_is_completed()
        # calculate displacement
        displacement_um = (x - self.x_reference) * self.pixel_to_um
        return displacement_um

    def move_to_target(self, target_um):
        current_displacement_um = self.measure_displacement()
        um_to_move = target_um - current_displacement_um
        # limit the range of movement
        um_to_move = min(um_to_move, 200)
        um_to_move = max(um_to_move, -200)
        self.navigationController.move_z(um_to_move / 1000)
        self.wait_till_operation_is_completed()
        # update the displacement measurement
        self.measure_displacement()

    def set_reference(self):
        # turn on the laser
        self.microcontroller.turn_on_AF_laser()
        self.wait_till_operation_is_completed()
        # get laser spot location
        x, y = self._get_laser_spot_centroid()
        # turn off the laser
        self.microcontroller.turn_off_AF_laser()
        self.wait_till_operation_is_completed()
        self.x_reference = x

    def _caculate_centroid(self, image):
        if self.has_two_interfaces == False:
            h, w = image.shape
            x, y = np.meshgrid(range(w), range(h))
            I = image.astype(float)
            I = I - np.amin(I)
            I[I / np.amax(I) < 0.2] = 0
            x = np.sum(x * I) / np.sum(I)
            y = np.sum(y * I) / np.sum(I)
            return x, y
        else:
            I = image
            # get the y position of the spots
            tmp = np.sum(I, axis=1)
            y0 = np.argmax(tmp)
            # crop along the y axis
            I = I[y0 - 96 : y0 + 96, :]
            # signal along x
            tmp = np.sum(I, axis=0)
            # find peaks
            peak_locations, _ = scipy.signal.find_peaks(tmp, distance=100)
            idx = np.argsort(tmp[peak_locations])
            peak_0_location = peak_locations[idx[-1]]
            peak_1_location = peak_locations[
                idx[-2]
            ]  # for air-glass-water, the smaller peak corresponds to the glass-water interface
            self.spot_spacing_pixels = peak_1_location - peak_0_location
            """
            # find peaks - alternative
            if self.spot_spacing_pixels is not None:
                peak_locations,_ = scipy.signal.find_peaks(tmp,distance=100)
                idx = np.argsort(tmp[peak_locations])
                peak_0_location = peak_locations[idx[-1]]
                peak_1_location = peak_locations[idx[-2]] # for air-glass-water, the smaller peak corresponds to the glass-water interface
                self.spot_spacing_pixels = peak_1_location-peak_0_location
            else:
                peak_0_location = np.argmax(tmp)
                peak_1_location = peak_0_location + self.spot_spacing_pixels
            """
            # choose which surface to use
            if self.use_glass_top:
                x1 = peak_1_location
            else:
                x1 = peak_0_location
            # find centroid
            h, w = I.shape
            x, y = np.meshgrid(range(w), range(h))
            I = I[:, max(0, x1 - 64) : min(w - 1, x1 + 64)]
            x = x[:, max(0, x1 - 64) : min(w - 1, x1 + 64)]
            y = y[:, max(0, x1 - 64) : min(w - 1, x1 + 64)]
            I = I.astype(float)
            I = I - np.amin(I)
            I[I / np.amax(I) < 0.1] = 0
            x1 = np.sum(x * I) / np.sum(I)
            y1 = np.sum(y * I) / np.sum(I)
            return x1, y0 - 96 + y1

    def _get_laser_spot_centroid(self):
        # disable camera callback
        self.camera.disable_callback()
        tmp_x = 0
        tmp_y = 0
        for i in range(CONFIG.LASER_AF_AVERAGING_N):
            # send camera trigger
            if self.liveController.trigger_mode == TriggerModeSetting.SOFTWARE:
                self.camera.send_trigger()
            elif self.liveController.trigger_mode == TriggerModeSetting.HARDWARE:
                # self.microcontroller.send_hardware_trigger(control_illumination=True,illumination_on_time_us=self.camera.exposure_time*1000)
                pass  # to edit
            # read camera frame
            image = self.camera.read_frame()
            self.image = image
            # optionally display the image
            if CONFIG.LASER_AF_DISPLAY_SPOT_IMAGE:
                pass
            # calculate centroid
            x, y = self._caculate_centroid(image)
            tmp_x = tmp_x + x
            tmp_y = tmp_y + y
        x = tmp_x / CONFIG.LASER_AF_AVERAGING_N
        y = tmp_y / CONFIG.LASER_AF_AVERAGING_N
        return x, y

    def wait_till_operation_is_completed(self):
        while self.microcontroller.is_busy():
            time.sleep(CONFIG.SLEEP_TIME_S)

    def get_image(self):
        # turn on the laser
        self.microcontroller.turn_on_AF_laser()
        self.wait_till_operation_is_completed()
        # send trigger, grab image and display image
        self.camera.send_trigger()
        image = self.camera.read_frame()
        # turn off the laser
        self.microcontroller.turn_off_AF_laser()
        self.wait_till_operation_is_completed()
        return image
