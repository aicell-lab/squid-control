import os 
import shutil
# app specific libraries
import squid_control.control.core_reef as core
import squid_control.control.microcontroller as microcontroller
from squid_control.control.config import *
from squid_control.control.camera import get_camera
from squid_control.control.utils import rotate_and_flip_image
from squid_control.control.config import ChannelMapper
import cv2
import logging
import logging.handlers
# Import serial_peripherals conditionally based on simulation mode
import sys
import numpy as np
from squid_control.stitching.zarr_canvas import ZarrCanvas, WellZarrCanvas
from pathlib import Path

_is_simulation_mode = (
    "--simulation" in sys.argv or 
    os.environ.get("SQUID_SIMULATION_MODE", "").lower() in ["true", "1", "yes"] or
    os.environ.get("PYTEST_CURRENT_TEST") is not None  # Running in pytest
)

if _is_simulation_mode:
    print("Simulation mode detected - skipping hardware peripheral imports")
    SERIAL_PERIPHERALS_AVAILABLE = False
    serial_peripherals = None
else:
    try:
        import squid_control.control.serial_peripherals as serial_peripherals
        SERIAL_PERIPHERALS_AVAILABLE = True
    except ImportError as e:
        print(f"serial_peripherals import error - hardware peripheral functionality not available: {e}")
        SERIAL_PERIPHERALS_AVAILABLE = False
        serial_peripherals = None
if CONFIG.SUPPORT_LASER_AUTOFOCUS:
    import squid_control.control.core_displacement_measurement as core_displacement_measurement

import time
import asyncio
#using os to set current working directory
#find the current path
path=os.path.abspath(__file__)
# Get the directory where config.py is located
config_dir = os.path.join(os.path.dirname(os.path.dirname(path)), 'squid_control/control')
cache_file_path = os.path.join(config_dir, 'cache_config_file_path.txt')

# Try to read the cached config path
config_path = None
if os.path.exists(cache_file_path):
    try:
        with open(cache_file_path, 'r') as f:
            config_path = f.readline().strip()
    except:
        pass

def setup_logging(log_file="squid_controller.log", max_bytes=100000, backup_count=3):
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

class SquidController:
    fps_software_trigger= 10

    def __init__(self,is_simulation, *args, **kwargs):
        super().__init__(*args,**kwargs)
        self.data_channel = None
        self.is_simulation = is_simulation
        self.is_busy = False
        self.scan_stop_requested = False  # Flag to stop ongoing scans
        self.zarr_artifact_manager = None  # Initialize zarr artifact manager to None
        if is_simulation:
            config_path = os.path.join(os.path.dirname(path), 'configuration_HCS_v2_example.ini')
        else:
            config_path = os.path.join(os.path.dirname(path), 'configuration_HCS_v2.ini')

        print(f"Loading configuration from: {config_path}")
        load_config(config_path, False)
        
        # Create objects after configuration is loaded to use updated CONFIG values
        self.objectiveStore = core.ObjectiveStore()
        print(f"ObjectiveStore initialized with default objective: {self.objectiveStore.current_objective}")
        camera, camera_fc = get_camera(CONFIG.CAMERA_TYPE)
        # load objects
        if self.is_simulation:
            if CONFIG.ENABLE_SPINNING_DISK_CONFOCAL and SERIAL_PERIPHERALS_AVAILABLE:
                self.xlight = serial_peripherals.XLight_Simulation()
            if CONFIG.SUPPORT_LASER_AUTOFOCUS:
                self.camera = camera.Camera_Simulation(
                    rotate_image_angle=CONFIG.ROTATE_IMAGE_ANGLE,
                    flip_image=CONFIG.FLIP_IMAGE,
                )
                self.camera_focus = camera_fc.Camera_Simulation()
            else:
                self.camera = camera.Camera_Simulation(
                    rotate_image_angle=CONFIG.ROTATE_IMAGE_ANGLE,
                    flip_image=CONFIG.FLIP_IMAGE,
                )
            self.microcontroller = microcontroller.Microcontroller_Simulation()
        else:
            if CONFIG.ENABLE_SPINNING_DISK_CONFOCAL and SERIAL_PERIPHERALS_AVAILABLE:
                self.xlight = serial_peripherals.XLight()
            try:
                if CONFIG.SUPPORT_LASER_AUTOFOCUS:
                    sn_camera_main = camera.get_sn_by_model(CONFIG.MAIN_CAMERA_MODEL)
                    sn_camera_focus = camera_fc.get_sn_by_model(
                        CONFIG.FOCUS_CAMERA_MODEL
                    )
                    self.camera = camera.Camera(
                        sn=sn_camera_main,
                        rotate_image_angle=CONFIG.ROTATE_IMAGE_ANGLE,
                        flip_image=CONFIG.FLIP_IMAGE,
                    )
                    self.camera.open()
                    self.camera_focus = camera_fc.Camera(sn=sn_camera_focus)
                    self.camera_focus.open()
                else:
                    self.camera = camera.Camera(
                        rotate_image_angle=CONFIG.ROTATE_IMAGE_ANGLE,
                        flip_image=CONFIG.FLIP_IMAGE,
                    )
                    self.camera.open()
            except:
                if CONFIG.SUPPORT_LASER_AUTOFOCUS:
                    self.camera = camera.Camera_Simulation(
                        rotate_image_angle=CONFIG.ROTATE_IMAGE_ANGLE,
                        flip_image=CONFIG.FLIP_IMAGE,
                    )
                    self.camera.open()
                    self.camera_focus = camera.Camera_Simulation()
                    self.camera_focus.open()
                else:
                    self.camera = camera.Camera_Simulation(
                        rotate_image_angle=CONFIG.ROTATE_IMAGE_ANGLE,
                        flip_image=CONFIG.FLIP_IMAGE,
                    )
                    self.camera.open()
                print("! camera not detected, using simulated camera !")
            self.microcontroller = microcontroller.Microcontroller(
                version=CONFIG.CONTROLLER_VERSION
            )

        # reset the MCU
        self.microcontroller.reset()
        time.sleep(0.5)

        # reinitialize motor drivers and DAC (in particular for V2.1 driver board where PG is not functional)
        self.microcontroller.initialize_drivers()
        time.sleep(0.5)

        # configure the actuators
        self.microcontroller.configure_actuators()

        self.configurationManager = core.ConfigurationManager(
            filename="./u2os_fucci_illumination_configurations.xml"
        )

        self.streamHandler = core.StreamHandler(
            display_resolution_scaling=CONFIG.DEFAULT_DISPLAY_CROP / 100
        )
        self.liveController = core.LiveController(
            self.camera, self.microcontroller, self.configurationManager
        )
        self.navigationController = core.NavigationController(
            self.microcontroller, parent=self
        )
        self.slidePositionController = core.SlidePositionController(
            self.navigationController, self.liveController, is_for_wellplate=True
        )
        self.autofocusController = core.AutoFocusController(
            self.camera, self.navigationController, self.liveController
        )
        self.scanCoordinates = core.ScanCoordinates()
        self.multipointController = core.MultiPointController(
            self.camera,
            self.navigationController,
            self.liveController,
            self.autofocusController,
            self.configurationManager,
            scanCoordinates=self.scanCoordinates,
            parent=self,
        )
        if CONFIG.ENABLE_TRACKING:
            self.trackingController = core.TrackingController(
                self.camera,
                self.microcontroller,
                self.navigationController,
                self.configurationManager,
                self.liveController,
                self.autofocusController,
            )

        # retract the objective
        self.navigationController.home_z()
        # wait for the operation to finish
        t0 = time.time()
        while self.microcontroller.is_busy():
            time.sleep(0.005)
            if time.time() - t0 > 10:
                print("z homing timeout, the program will exit")
                exit()
        print("objective retracted")

        # set encoder arguments
        # set axis pid control enable
        # only CONFIG.ENABLE_PID_X and CONFIG.HAS_ENCODER_X are both enable, can be enable to PID
        if CONFIG.HAS_ENCODER_X == True:
            self.navigationController.configure_encoder(
                0,
                (CONFIG.SCREW_PITCH_X_MM * 1000) / CONFIG.ENCODER_RESOLUTION_UM_X,
                CONFIG.ENCODER_FLIP_DIR_X,
            )
            self.navigationController.set_pid_control_enable(0, CONFIG.ENABLE_PID_X)
        if CONFIG.HAS_ENCODER_Y == True:
            self.navigationController.configure_encoder(
                1,
                (CONFIG.SCREW_PITCH_Y_MM * 1000) / CONFIG.ENCODER_RESOLUTION_UM_Y,
                CONFIG.ENCODER_FLIP_DIR_Y,
            )
            self.navigationController.set_pid_control_enable(1, CONFIG.ENABLE_PID_Y)
        if CONFIG.HAS_ENCODER_Z == True:
            self.navigationController.configure_encoder(
                2,
                (CONFIG.SCREW_PITCH_Z_MM * 1000) / CONFIG.ENCODER_RESOLUTION_UM_Z,
                CONFIG.ENCODER_FLIP_DIR_Z,
            )
            self.navigationController.set_pid_control_enable(2, CONFIG.ENABLE_PID_Z)
        time.sleep(0.5)

        self.navigationController.set_z_limit_pos_mm(
            CONFIG.SOFTWARE_POS_LIMIT.Z_POSITIVE
        )

        # home XY, set zero and set software limit
        print("home xy")
        timestamp_start = time.time()
        # x needs to be at > + 20 mm when homing y
        self.navigationController.move_x(20)  # to-do: add blocking code
        while self.microcontroller.is_busy():
            time.sleep(0.005)
        # home y
        self.navigationController.home_y()
        t0 = time.time()
        while self.microcontroller.is_busy():
            time.sleep(0.005)
            if time.time() - t0 > 10:
                print("y homing timeout, the program will exit")
                exit()
        self.navigationController.zero_y()
        # home x
        self.navigationController.home_x()
        t0 = time.time()
        while self.microcontroller.is_busy():
            time.sleep(0.005)
            if time.time() - t0 > 10:
                print("y homing timeout, the program will exit")
                exit()
        self.navigationController.zero_x()
        self.slidePositionController.homing_done = True
        print("home xy done")

        # move to scanning position
        self.navigationController.move_x(32.3)
        while self.microcontroller.is_busy():
            time.sleep(0.005)
        self.navigationController.move_y(29.35)
        while self.microcontroller.is_busy():
            time.sleep(0.005)

        # move z
        self.navigationController.move_z_to(CONFIG.DEFAULT_Z_POS_MM)
        # wait for the operation to finish
        
        # FIXME: This is failing right now, z return timeout
        t0 = time.time()
        while self.microcontroller.is_busy():
            time.sleep(0.005)
            if time.time() - t0 > 5:
                print("z return timeout, the program will exit")
                exit()

        # set output's gains
        div = 1 if CONFIG.OUTPUT_GAINS.REFDIV is True else 0
        gains = CONFIG.OUTPUT_GAINS.CHANNEL0_GAIN << 0
        gains += CONFIG.OUTPUT_GAINS.CHANNEL1_GAIN << 1
        gains += CONFIG.OUTPUT_GAINS.CHANNEL2_GAIN << 2
        gains += CONFIG.OUTPUT_GAINS.CHANNEL3_GAIN << 3
        gains += CONFIG.OUTPUT_GAINS.CHANNEL4_GAIN << 4
        gains += CONFIG.OUTPUT_GAINS.CHANNEL5_GAIN << 5
        gains += CONFIG.OUTPUT_GAINS.CHANNEL6_GAIN << 6
        gains += CONFIG.OUTPUT_GAINS.CHANNEL7_GAIN << 7
        self.microcontroller.configure_dac80508_refdiv_and_gain(div, gains)

        # set illumination intensity factor
        self.microcontroller.set_dac80508_scaling_factor_for_illumination(
            CONFIG.ILLUMINATION_INTENSITY_FACTOR
        )

        # open the camera
        # camera start streaming
        # self.camera.set_reverse_x(CAMERA_REVERSE_X) # these are not implemented for the cameras in use
        # self.camera.set_reverse_y(CAMERA_REVERSE_Y) # these are not implemented for the cameras in use
        self.camera.set_software_triggered_acquisition()  # self.camera.set_continuous_acquisition()
        self.camera.set_callback(self.streamHandler.on_new_frame)
        #self.camera.enable_callback()
        self.camera.start_streaming()     

        if CONFIG.SUPPORT_LASER_AUTOFOCUS:

            # controllers
            self.configurationManager_focus_camera = core.ConfigurationManager(filename='./focus_camera_configurations.xml')
            self.streamHandler_focus_camera = core.StreamHandler()
            self.liveController_focus_camera = core.LiveController(self.camera_focus,self.microcontroller,self.configurationManager_focus_camera,control_illumination=False,for_displacement_measurement=True)
            self.multipointController = core.MultiPointController(self.camera,self.navigationController,self.liveController,self.autofocusController,self.configurationManager,scanCoordinates=self.scanCoordinates,parent=self)
            self.displacementMeasurementController = core_displacement_measurement.DisplacementMeasurementController()
            self.laserAutofocusController = core.LaserAutofocusController(self.microcontroller,self.camera_focus,self.liveController_focus_camera,self.navigationController,has_two_interfaces=CONFIG.HAS_TWO_INTERFACES,use_glass_top=CONFIG.USE_GLASS_TOP)

            # camera
            self.camera_focus.set_software_triggered_acquisition() #self.camera.set_continuous_acquisition()
            self.camera_focus.set_callback(self.streamHandler_focus_camera.on_new_frame)
            #self.camera_focus.enable_callback()
            self.camera_focus.start_streaming()


        # set software limits        
        self.navigationController.set_x_limit_pos_mm(CONFIG.SOFTWARE_POS_LIMIT.X_POSITIVE)
        self.navigationController.set_x_limit_neg_mm(CONFIG.SOFTWARE_POS_LIMIT.X_NEGATIVE)
        self.navigationController.set_y_limit_pos_mm(CONFIG.SOFTWARE_POS_LIMIT.Y_POSITIVE)
        self.navigationController.set_y_limit_neg_mm(CONFIG.SOFTWARE_POS_LIMIT.Y_NEGATIVE)

        # set the default infomation, this will be used for the simulated camera
        self.dz = 0
        self.current_channel = 0
        self.current_exposure_time = 100
        self.current_intensity = 100
        self.pixel_size_xy = 0.333
        # drift correction for image map
        self.drift_correction_x = -1.6
        self.drift_correction_y = -2.1
        # simulated sample data alias
        self.sample_data_alias = "agent-lens/20250506-scan-time-lapse-2025-05-06_17-56-38"
        self.get_pixel_size()

        # Initialize experiment-based zarr management
        zarr_path = os.getenv('ZARR_PATH', '/tmp/zarr_canvas')
        from squid_control.stitching.zarr_canvas import ExperimentManager
        self.experiment_manager = ExperimentManager(zarr_path, self.pixel_size_xy)
        
        # Initialize legacy well_canvases attribute for backward compatibility
        self.well_canvases = {}
        
        # Initialize legacy zarr canvas attributes for backward compatibility
        self.zarr_canvases = {}
        self.zarr_canvas = None
        self.active_canvas_name = None
        
        # Clean up ZARR_PATH directory on startup
        #self._cleanup_zarr_directory() # Disabled for now

    def get_pixel_size(self):
        """Calculate pixel size based on imaging parameters."""
        try:
            tube_lens_mm = float(CONFIG.TUBE_LENS_MM)
            pixel_size_um = float(CONFIG.CAMERA_PIXEL_SIZE_UM[CONFIG.CAMERA_SENSOR])
            
            object_dict_key = self.objectiveStore.current_objective
            objective = self.objectiveStore.objectives_dict[object_dict_key]
            magnification =  float(objective['magnification'])
            objective_tube_lens_mm = float(objective['tube_lens_f_mm'])
            print(f"Using objective: {object_dict_key}")
            print(f"CONFIG.DEFAULT_OBJECTIVE: {CONFIG.DEFAULT_OBJECTIVE}")
            print(f"Tube lens: {tube_lens_mm} mm, Objective tube lens: {objective_tube_lens_mm} mm, Pixel size: {pixel_size_um} µm, Magnification: {magnification}")
        except Exception as e:
            logger.error(f"Missing required parameters for pixel size calculation: {e}")
            return

        self.pixel_size_xy = pixel_size_um / (magnification / (objective_tube_lens_mm / tube_lens_mm))
        self.pixel_size_xy = self.pixel_size_xy * CONFIG.PIXEL_SIZE_ADJUSTMENT_FACTOR
        print(f"Pixel size: {self.pixel_size_xy} µm (adjustment factor: {CONFIG.PIXEL_SIZE_ADJUSTMENT_FACTOR})")
        
                
    def move_to_scaning_position(self):
        # move to scanning position
        self.navigationController.move_z_to(0.4)
        self.navigationController.move_x(20)
        while self.microcontroller.is_busy():
            time.sleep(0.005)
        self.navigationController.move_y(20)
        while self.microcontroller.is_busy():
            time.sleep(0.005)

        # move z
        self.navigationController.move_z_to(CONFIG.DEFAULT_Z_POS_MM)
        # wait for the operation to finish
        t0 = time.time() 
        while self.microcontroller.is_busy():
            time.sleep(0.005)
            if time.time() - t0 > 5:
                print('z return timeout, the program will exit')
                exit()

    def plate_scan(self, well_plate_type='96', illumination_settings=None, do_contrast_autofocus=False, do_reflection_af=True, scanning_zone=[(0,0),(2,2)], Nx=3, Ny=3, action_ID='testPlateScanNew'):
        """
        New well plate scanning function with custom illumination settings.
        
        Args:
            well_plate_type (str): Type of well plate ('96', '384', etc.)
            illumination_settings (list): List of dictionaries with illumination settings
                Each dict should contain:
                {
                    'channel': 'BF LED matrix full',  # Channel name
                    'intensity': 50.0,               # Illumination intensity (0-100)
                    'exposure_time': 25.0            # Exposure time in ms
                }
            do_contrast_autofocus (bool): Whether to perform contrast-based autofocus
            do_reflection_af (bool): Whether to perform reflection-based autofocus
            scanning_zone (list): List of two tuples [(start_row, start_col), (end_row, end_col)]
            Nx (int): Number of X positions per well
            Ny (int): Number of Y positions per well
            action_ID (str): Identifier for this scan
        """
        if illumination_settings is None:
            logger.warning("No illumination settings provided, using default settings")
            # Default settings if none provided
            illumination_settings = [
                {'channel': 'BF LED matrix full', 'intensity': 18, 'exposure_time': 37},
                {'channel': 'Fluorescence 405 nm Ex', 'intensity': 45, 'exposure_time': 30},
                {'channel': 'Fluorescence 488 nm Ex', 'intensity': 30, 'exposure_time': 100},
                {'channel': 'Fluorescence 561 nm Ex', 'intensity': 100, 'exposure_time': 200},
                {'channel': 'Fluorescence 638 nm Ex', 'intensity': 100, 'exposure_time': 200},
                {'channel': 'Fluorescence 730 nm Ex', 'intensity': 100, 'exposure_time': 200},
            ]
        
        # Update configurations with custom settings
        self.multipointController.set_selected_configurations_with_settings(illumination_settings)
        
        # Move to scanning position
        self.move_to_scaning_position()
        
        # Set up scan coordinates
        self.scanCoordinates.well_selector.set_selected_wells(scanning_zone[0], scanning_zone[1])
        self.scanCoordinates.get_selected_wells_to_coordinates(wellplate_type=well_plate_type, is_simulation=self.is_simulation)
        
        # Configure multipoint controller
        self.multipointController.set_base_path(CONFIG.DEFAULT_SAVING_PATH)
        self.multipointController.do_autofocus = do_contrast_autofocus
        self.multipointController.do_reflection_af = do_reflection_af
        self.multipointController.set_NX(Nx)
        self.multipointController.set_NY(Ny)
        self.multipointController.start_new_experiment(action_ID)
        
        # Start scanning
        self.is_busy = True
        print('Starting new plate scan with custom illumination settings')
        self.multipointController.run_acquisition()
        print('New plate scan completed')
        self.is_busy = False
    
    def stop_plate_scan(self):
        self.multipointController.abort_acqusition_requested = True
        self.is_busy = False
        print('Plate scan stopped')
        
    def stop_scan_well_plate_new(self):
        """Stop the new well plate scan - alias for stop_plate_scan"""
        self.stop_plate_scan()
        
    async def send_trigger_simulation(self, channel=0, intensity=100, exposure_time=100):
        print('Getting simulated image')
        current_x, current_y, current_z, *_ = self.navigationController.update_pos(microcontroller=self.microcontroller)
        self.dz = current_z - SIMULATED_CAMERA.ORIN_Z
        self.current_channel = channel
        magnification_factor = SIMULATED_CAMERA.MAGNIFICATION_FACTOR
        self.current_exposure_time = exposure_time
        self.current_intensity = intensity
        corrected_x = current_x + self.drift_correction_x
        corrected_y = current_y + self.drift_correction_y
        await self.camera.send_trigger(corrected_x, corrected_y, self.dz, self.pixel_size_xy, channel, intensity, exposure_time, magnification_factor, sample_data_alias=self.sample_data_alias)
        print(f'For simulated camera, exposure_time={exposure_time}, intensity={intensity}, magnification_factor={magnification_factor}, current position: {current_x},{current_y},{current_z}')
   
    def set_simulated_sample_data_alias(self, sample_data_alias):
        self.sample_data_alias = sample_data_alias

    def get_simulated_sample_data_alias(self):
        return self.sample_data_alias

    async def do_autofocus(self):
        
        if self.is_simulation:
            await self.do_autofocus_simulation()
        else:
            self.autofocusController.set_deltaZ(1.524)
            self.autofocusController.set_N(15)
            self.autofocusController.autofocus()
            self.autofocusController.wait_till_autofocus_has_completed()

    async def do_autofocus_simulation(self):
        
        random_z = SIMULATED_CAMERA.ORIN_Z + np.random.normal(0,0.001)
        self.navigationController.move_z_to(random_z)
        await self.send_trigger_simulation(self.current_channel, self.current_intensity, self.current_exposure_time)
        
    def init_laser_autofocus(self):
        self.laserAutofocusController.initialize_auto()

    async def do_laser_autofocus(self):
        if self.is_simulation:
            await self.do_autofocus_simulation()
        else:
            self.laserAutofocusController.move_to_target(0)
    
    def measure_displacement(self):
        self.laserAutofocusController.measure_displacement()     
        
    def move_to_well(self,row,column, wellplate_type='96'):
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
        
        if column != 0 and column != None:
            mm_per_ustep_X = CONFIG.SCREW_PITCH_X_MM/(self.navigationController.x_microstepping*CONFIG.FULLSTEPS_PER_REV_X)
            if self.is_simulation:
                x_mm = wellplate_format.A1_X_MM + (int(column)-1)*wellplate_format.WELL_SPACING_MM
            else:
                x_mm = wellplate_format.A1_X_MM + (int(column)-1)*wellplate_format.WELL_SPACING_MM + CONFIG.WELLPLATE_OFFSET_X_MM
            x_usteps = CONFIG.STAGE_MOVEMENT_SIGN_X*round(x_mm/mm_per_ustep_X)
            self.microcontroller.move_x_to_usteps(x_usteps)
        if row != 0 and row != None:
            mm_per_ustep_Y = CONFIG.SCREW_PITCH_Y_MM/(self.navigationController.y_microstepping*CONFIG.FULLSTEPS_PER_REV_Y)
            if self.is_simulation:
                y_mm = wellplate_format.A1_Y_MM + (ord(row) - ord('A'))*wellplate_format.WELL_SPACING_MM
            else:
                y_mm = wellplate_format.A1_Y_MM + (ord(row) - ord('A'))*wellplate_format.WELL_SPACING_MM + CONFIG.WELLPLATE_OFFSET_Y_MM
            y_usteps = CONFIG.STAGE_MOVEMENT_SIGN_Y*round(y_mm/mm_per_ustep_Y)
            self.microcontroller.move_y_to_usteps(y_usteps)
            while self.microcontroller.is_busy():
                time.sleep(0.005)

    async def move_to_well_async(self, row, column, wellplate_type='96'):
        """
        Async version of move_to_well that doesn't block the event loop.
        
        Args:
            row: Row letter (e.g., 'A', 'B', 'C')
            column: Column number (e.g., 1, 2, 3)
            wellplate_type: Type of well plate ('6', '12', '24', '96', '384')
        """
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
        
        if column != 0 and column != None:
            mm_per_ustep_X = CONFIG.SCREW_PITCH_X_MM/(self.navigationController.x_microstepping*CONFIG.FULLSTEPS_PER_REV_X)
            if self.is_simulation:
                x_mm = wellplate_format.A1_X_MM + (int(column)-1)*wellplate_format.WELL_SPACING_MM
            else:
                x_mm = wellplate_format.A1_X_MM + (int(column)-1)*wellplate_format.WELL_SPACING_MM + CONFIG.WELLPLATE_OFFSET_X_MM
            x_usteps = CONFIG.STAGE_MOVEMENT_SIGN_X*round(x_mm/mm_per_ustep_X)
            self.microcontroller.move_x_to_usteps(x_usteps)
        if row != 0 and row != None:
            mm_per_ustep_Y = CONFIG.SCREW_PITCH_Y_MM/(self.navigationController.y_microstepping*CONFIG.FULLSTEPS_PER_REV_Y)
            if self.is_simulation:
                y_mm = wellplate_format.A1_Y_MM + (ord(row) - ord('A'))*wellplate_format.WELL_SPACING_MM
            else:
                y_mm = wellplate_format.A1_Y_MM + (ord(row) - ord('A'))*wellplate_format.WELL_SPACING_MM + CONFIG.WELLPLATE_OFFSET_Y_MM
            y_usteps = CONFIG.STAGE_MOVEMENT_SIGN_Y*round(y_mm/mm_per_ustep_Y)
            self.microcontroller.move_y_to_usteps(y_usteps)
            # Use async sleep to avoid blocking the event loop
            while self.microcontroller.is_busy():
                await asyncio.sleep(0.005)

    async def move_to_well_center_for_autofocus(self, row, column, wellplate_type='96', velocity_mm_per_s=30.0):
        """
        Optimized method to move to well center for autofocus operations.
        Sets velocity, moves to well center, and waits for completion.
        
        Args:
            row: Row letter (e.g., 'A', 'B', 'C')
            column: Column number (e.g., 1, 2, 3)
            wellplate_type: Type of well plate ('6', '12', '24', '96', '384')
            velocity_mm_per_s: Velocity for movement (default 30.0 mm/s)
        """
        # Set high speed velocity for moving to well center
        velocity_result = self.set_stage_velocity(velocity_mm_per_s, velocity_mm_per_s)
        if not velocity_result['success']:
            logger.warning(f"Failed to set high-speed velocity for autofocus: {velocity_result['message']}")
        
        # Move to well center using async method
        await self.move_to_well_async(row, column, wellplate_type)
        
        logger.info(f'Moved to well {row}{column} center for autofocus')

    def get_well_from_position(self, wellplate_type='96', x_pos_mm=None, y_pos_mm=None, well_padding_mm=1.0):
        """
        Calculate which well position corresponds to the given X,Y coordinates, considering canvas padding.
        This is used for stitching where we want to accept positions within the padded canvas area.
        
        Args:
            wellplate_type (str): Type of well plate ('6', '12', '24', '96', '384')
            x_pos_mm (float, optional): X position in mm. If None, uses current position.
            y_pos_mm (float, optional): Y position in mm. If None, uses current position.
            well_padding_mm (float): Padding around well boundaries in mm
            
        Returns:
            dict: Same format as get_well_from_position but with padded boundaries
        """
        # Get well plate format configuration
        if wellplate_type == '6':
            wellplate_format = WELLPLATE_FORMAT_6
            max_rows = 2  # A-B
            max_cols = 3  # 1-3
        elif wellplate_type == '12':
            wellplate_format = WELLPLATE_FORMAT_12
            max_rows = 3  # A-C
            max_cols = 4  # 1-4
        elif wellplate_type == '24':
            wellplate_format = WELLPLATE_FORMAT_24
            max_rows = 4  # A-D
            max_cols = 6  # 1-6
        elif wellplate_type == '96':
            wellplate_format = WELLPLATE_FORMAT_96
            max_rows = 8  # A-H
            max_cols = 12  # 1-12
        elif wellplate_type == '384':
            wellplate_format = WELLPLATE_FORMAT_384
            max_rows = 16  # A-P
            max_cols = 24  # 1-24
        else:
            # Default to 96-well plate if unsupported type is provided
            wellplate_format = WELLPLATE_FORMAT_96
            max_rows = 8
            max_cols = 12
            wellplate_type = '96'
        
        # Get current position if not provided
        if x_pos_mm is None or y_pos_mm is None:
            current_x, current_y, current_z, current_theta = self.navigationController.update_pos(
                microcontroller=self.microcontroller
            )
            if x_pos_mm is None:
                x_pos_mm = current_x
            if y_pos_mm is None:
                y_pos_mm = current_y
        
        # Apply well plate offset for hardware mode
        if self.is_simulation:
            x_offset = 0
            y_offset = 0
        else:
            x_offset = CONFIG.WELLPLATE_OFFSET_X_MM
            y_offset = CONFIG.WELLPLATE_OFFSET_Y_MM
        
        # Calculate which well this position corresponds to
        # Reverse of the move_to_well calculation
        x_relative = x_pos_mm - (wellplate_format.A1_X_MM + x_offset)
        y_relative = y_pos_mm - (wellplate_format.A1_Y_MM + y_offset)
        
        # Calculate well indices (0-based initially)
        col_index = round(x_relative / wellplate_format.WELL_SPACING_MM)
        row_index = round(y_relative / wellplate_format.WELL_SPACING_MM)
        
        # Initialize result dictionary
        result = {
            'row': None,
            'column': None,
            'well_id': None,
            'is_inside_well': False,
            'distance_from_center': float('inf'),
            'position_status': 'outside_plate',
            'x_mm': x_pos_mm,
            'y_mm': y_pos_mm,
            'plate_type': wellplate_type
        }
        
        # Check if the calculated well indices are valid
        if 0 <= col_index < max_cols and 0 <= row_index < max_rows:
            # Convert to 1-based column and letter-based row
            column = col_index + 1
            row = chr(ord('A') + row_index)
            
            result['row'] = row
            result['column'] = column
            result['well_id'] = f"{row}{column}"
            
            # Calculate the exact center position of this well
            well_center_x = wellplate_format.A1_X_MM + x_offset + col_index * wellplate_format.WELL_SPACING_MM
            well_center_y = wellplate_format.A1_Y_MM + y_offset + row_index * wellplate_format.WELL_SPACING_MM
            
            # Calculate distance from well center
            dx = x_pos_mm - well_center_x
            dy = y_pos_mm - well_center_y
            distance_from_center = np.sqrt(dx**2 + dy**2)
            result['distance_from_center'] = distance_from_center
            
            # Check if position is inside the PADDED well boundary (for stitching purposes)
            well_radius = wellplate_format.WELL_SIZE_MM / 2.0
            padded_radius = well_radius + well_padding_mm  # Include padding in boundary check
            if distance_from_center <= padded_radius:
                result['is_inside_well'] = True
                result['position_status'] = 'in_well'
            else:
                result['is_inside_well'] = False
                result['position_status'] = 'between_wells'
        else:
            # Position is outside the valid well range
            result['position_status'] = 'outside_plate'
            
            # Find the closest valid well for reference
            closest_col = max(0, min(max_cols - 1, col_index))
            closest_row = max(0, min(max_rows - 1, row_index))
            
            closest_well_center_x = wellplate_format.A1_X_MM + x_offset + closest_col * wellplate_format.WELL_SPACING_MM
            closest_well_center_y = wellplate_format.A1_Y_MM + y_offset + closest_row * wellplate_format.WELL_SPACING_MM
            
            dx = x_pos_mm - closest_well_center_x
            dy = y_pos_mm - closest_well_center_y
            result['distance_from_center'] = np.sqrt(dx**2 + dy**2)
        
        return result

    def move_x_to_limited(self, x):
        
        x_pos_before,y_pos_before, z_pos_before, *_ = self.navigationController.update_pos(microcontroller=self.microcontroller)

        self.navigationController.move_x_to_limited(x)
        while self.microcontroller.is_busy():
            time.sleep(0.005) 
        
        x_pos,y_pos, z_pos, *_ = self.navigationController.update_pos(microcontroller=self.microcontroller)

        if abs(x_pos - x) < CONFIG.STAGE_MOVED_THRESHOLD:
            return True, x_pos_before, y_pos_before, z_pos_before, x

        return False, x_pos_before, y_pos_before, z_pos_before, x
    
    def move_y_to_limited(self, y):
        x_pos_before,y_pos_before, z_pos_before, *_ = self.navigationController.update_pos(microcontroller=self.microcontroller)
        self.navigationController.move_y_to_limited(y)

        while self.microcontroller.is_busy():
            time.sleep(0.005)
        x_pos,y_pos, z_pos, *_ = self.navigationController.update_pos(microcontroller=self.microcontroller)

        if abs(y_pos - y) < CONFIG.STAGE_MOVED_THRESHOLD:
            return True, x_pos_before, y_pos_before, z_pos_before, y
    
        return False, x_pos_before, y_pos_before, z_pos_before, y
    
    def move_z_to_limited(self, z):
        x_pos_before,y_pos_before, z_pos_before, *_ = self.navigationController.update_pos(microcontroller=self.microcontroller)
        self.navigationController.move_z_to_limited(z)

        while self.microcontroller.is_busy():
            time.sleep(0.005)
        x_pos,y_pos, z_pos, *_ = self.navigationController.update_pos(microcontroller=self.microcontroller)

        if abs(z_pos - z) < CONFIG.STAGE_MOVED_THRESHOLD:
            return True, x_pos_before, y_pos_before, z_pos_before, z

        return False, x_pos_before, y_pos_before, z_pos_before, z
    

    def move_by_distance_limited(self, dx, dy, dz):
        x_pos_before,y_pos_before, z_pos_before, *_ = self.navigationController.update_pos(microcontroller=self.microcontroller)

        self.navigationController.move_x_limited(dx)
        while self.microcontroller.is_busy():
            time.sleep(0.005)
        self.navigationController.move_y_limited(dy)
        while self.microcontroller.is_busy():
            time.sleep(0.005)
        self.navigationController.move_z_limited(dz)
        while self.microcontroller.is_busy():
            time.sleep(0.005)
        
        x_pos,y_pos, z_pos, *_ = self.navigationController.update_pos(microcontroller=self.microcontroller)

        if abs(x_pos-x_pos_before)<CONFIG.STAGE_MOVED_THRESHOLD and dx!=0:
            return False, x_pos_before, y_pos_before, z_pos_before, x_pos_before+dx, y_pos_before+dy, z_pos_before+dz
        if abs(y_pos-y_pos_before)<CONFIG.STAGE_MOVED_THRESHOLD and dy!=0:
            return False, x_pos_before, y_pos_before, z_pos_before, x_pos_before+dx, y_pos_before+dy, z_pos_before+dz
        if abs(z_pos-z_pos_before)<CONFIG.STAGE_MOVED_THRESHOLD and dz!=0:
            return False, x_pos_before, y_pos_before, z_pos_before, x_pos_before+dx, y_pos_before+dy, z_pos_before+dz


        return True, x_pos_before, y_pos_before, z_pos_before, x_pos, y_pos, z_pos
    
    def home_stage(self):
        # retract the object
        self.navigationController.home_z()
        # wait for the operation to finish
        t0 = time.time()
        while self.microcontroller.is_busy():
            time.sleep(0.005)
            if time.time() - t0 > 10:
                print('z homing timeout, the program will exit')
                exit()
        print('objective retracted')
        self.navigationController.set_z_limit_pos_mm(CONFIG.SOFTWARE_POS_LIMIT.Z_POSITIVE)

        # home XY, set zero and set software limit
        print('home xy')
        timestamp_start = time.time()
        # x needs to be at > + 20 mm when homing y
        self.navigationController.move_x(20) # to-do: add blocking code
        while self.microcontroller.is_busy():
            time.sleep(0.005)
        # home y
        self.navigationController.home_y()
        t0 = time.time()
        while self.microcontroller.is_busy():
            time.sleep(0.005)
            if time.time() - t0 > 10:
                print('y homing timeout, the program will exit')
                exit()
        self.navigationController.zero_y()
        # home x
        self.navigationController.home_x()
        t0 = time.time()
        while self.microcontroller.is_busy():
            time.sleep(0.005)
            if time.time() - t0 > 10:
                print('y homing timeout, the program will exit')
                exit()
        self.navigationController.zero_x()
        self.slidePositionController.homing_done = True
        print('home xy done')
        
    def return_stage(self):
        # move to scanning position
        self.navigationController.move_x(30.26)
        while self.microcontroller.is_busy():
            time.sleep(0.005)
        self.navigationController.move_y(29.1)
        while self.microcontroller.is_busy():
            time.sleep(0.005)

        # move z
        self.navigationController.move_z_to(CONFIG.DEFAULT_Z_POS_MM)
        # wait for the operation to finish
        t0 = time.time() 
        while self.microcontroller.is_busy():
            time.sleep(0.005)
            if time.time() - t0 > 5:
                print('z return timeout, the program will exit')

    async def snap_image(self, channel=0, intensity=100, exposure_time=100, full_frame=False):
        # turn off the illumination if it is on
        need_to_turn_illumination_back = False
        if self.liveController.illumination_on:
            need_to_turn_illumination_back = True
            self.liveController.turn_off_illumination()
            while self.microcontroller.is_busy():
                await asyncio.sleep(0.005)
        self.camera.set_exposure_time(exposure_time)
        self.liveController.set_illumination(channel, intensity)
        self.liveController.turn_on_illumination()
        while self.microcontroller.is_busy():
            await asyncio.sleep(0.005)

        if self.is_simulation:
            await self.send_trigger_simulation(channel, intensity, exposure_time)
        else:
            self.camera.send_trigger()
        await asyncio.sleep(0.005)

        while self.microcontroller.is_busy():
            await asyncio.sleep(0.005)

        gray_img = self.camera.read_frame()
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
            crop_height = CONFIG.Acquisition.CROP_HEIGHT
            crop_width = CONFIG.Acquisition.CROP_WIDTH
            height, width = gray_img.shape[:2]
            start_x = width // 2 - crop_width // 2
            start_y = height // 2 - crop_height // 2
            
            # Add bounds checking
            start_x = max(0, start_x)
            start_y = max(0, start_y)
            end_x = min(width, start_x + crop_width)
            end_y = min(height, start_y + crop_height)
            
            result_img = gray_img[start_y:end_y, start_x:end_x]

        if not need_to_turn_illumination_back:
            self.liveController.turn_off_illumination()
            while self.microcontroller.is_busy():
                await asyncio.sleep(0.005)

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
            return gray_img
        except Exception as e:
            print(f"Error in get_camera_frame: {e}")
            # Return a placeholder image on error
            return np.zeros((self.camera.Height, self.camera.Width), dtype=np.uint8)
    

    def close(self):
        # In simulation mode, skip stage movements to avoid delays
        if self.is_simulation:
            print("Simulation mode: Skipping stage close operations")
            # Close only essential components that don't cause delays
            if hasattr(self, 'liveController'):
                # LiveController doesn't have close method, just stop live
                if hasattr(self.liveController, 'stop_live'):
                    self.liveController.stop_live()
            if hasattr(self, 'camera'):
                self.camera.close()
            # Close experiment manager
            if hasattr(self, 'experiment_manager'):
                self.experiment_manager.close()
            return
        
        # Normal close operations for real hardware
        print("closing the system")
        if hasattr(self, 'liveController'):
            # LiveController doesn't have close method, just stop live
            if hasattr(self.liveController, 'stop_live'):
                self.liveController.stop_live()
        if hasattr(self, 'camera'):
            self.camera.close()
        
        # Move to safe position synchronously (no threading)
        if hasattr(self, 'navigationController') and hasattr(self, 'microcontroller'):
            try:
                self.navigationController.move_x_to(30)
                while self.microcontroller.is_busy():
                    time.sleep(0.005)
                self.navigationController.move_y_to(30)
                while self.microcontroller.is_busy():
                    time.sleep(0.005)
            except Exception as e:
                print(f"Error moving to safe position during close: {e}")
        
        if hasattr(self, 'camera_focus'):
            self.camera_focus.close()
        
        if hasattr(self, 'microcontroller'):
            self.microcontroller.close()
        
        # Close experiment manager
        if hasattr(self, 'experiment_manager'):
            self.experiment_manager.close()

    def set_stage_velocity(self, velocity_x_mm_per_s=None, velocity_y_mm_per_s=None):
        """
        Set the maximum velocity for X and Y stage axes.
        
        Args:
            velocity_x_mm_per_s (float, optional): Maximum velocity for X axis in mm/s. 
                                                 If None, uses default from configuration.
            velocity_y_mm_per_s (float, optional): Maximum velocity for Y axis in mm/s.
                                                 If None, uses default from configuration.
        
        Returns:
            dict: Status and current velocity settings
        """
        # Use default values from configuration if not specified
        if velocity_x_mm_per_s is None:
            velocity_x_mm_per_s = CONFIG.MAX_VELOCITY_X_MM
        if velocity_y_mm_per_s is None:
            velocity_y_mm_per_s = CONFIG.MAX_VELOCITY_Y_MM
            
        # Validate velocity ranges (microcontroller limit is 65535/100 = 655.35 mm/s)
        max_velocity_limit = 655.35
        if velocity_x_mm_per_s > max_velocity_limit or velocity_x_mm_per_s <= 0:
            return {
                "success": False,
                "message": f"X velocity must be between 0 and {max_velocity_limit} mm/s (exclusive of 0)",
                "velocity_x_mm_per_s": velocity_x_mm_per_s,
                "velocity_y_mm_per_s": velocity_y_mm_per_s
            }
        if velocity_y_mm_per_s > max_velocity_limit or velocity_y_mm_per_s <= 0:
            return {
                "success": False,
                "message": f"Y velocity must be between 0 and {max_velocity_limit} mm/s (exclusive of 0)",
                "velocity_x_mm_per_s": velocity_x_mm_per_s,
                "velocity_y_mm_per_s": velocity_y_mm_per_s
            }
        
        try:
            # Set X axis velocity (keeping default acceleration)
            self.microcontroller.set_max_velocity_acceleration(
                microcontroller.AXIS.X, velocity_x_mm_per_s, CONFIG.MAX_ACCELERATION_X_MM
            )
            self.microcontroller.wait_till_operation_is_completed()
            
            # Set Y axis velocity (keeping default acceleration)  
            self.microcontroller.set_max_velocity_acceleration(
                microcontroller.AXIS.Y, velocity_y_mm_per_s, CONFIG.MAX_ACCELERATION_Y_MM
            )
            self.microcontroller.wait_till_operation_is_completed()
            
            return {
                "success": True,
                "message": "Stage velocity updated successfully",
                "velocity_x_mm_per_s": velocity_x_mm_per_s,
                "velocity_y_mm_per_s": velocity_y_mm_per_s,
                "acceleration_x_mm_per_s2": CONFIG.MAX_ACCELERATION_X_MM,
                "acceleration_y_mm_per_s2": CONFIG.MAX_ACCELERATION_Y_MM
            }
            
        except Exception as e:
            return {
                "success": False, 
                "message": f"Failed to set stage velocity: {str(e)}",
                "velocity_x_mm_per_s": velocity_x_mm_per_s,
                "velocity_y_mm_per_s": velocity_y_mm_per_s
            }

    async def normal_scan_with_stitching(self, start_x_mm, start_y_mm, Nx, Ny, dx_mm, dy_mm, 
                                        illumination_settings=None, do_contrast_autofocus=False, 
                                        do_reflection_af=False, action_ID='normal_scan_stitching',
                                        timepoint=0, experiment_name=None, wells_to_scan=None,
                                        wellplate_type='96', well_padding_mm=1.0):
        """
        Normal scan with live stitching to well-specific OME-Zarr canvases.
        Scans specified wells one by one, creating individual zarr canvases for each well.
        
        Args:
            start_x_mm (float): Starting X position in mm (relative to well center)
            start_y_mm (float): Starting Y position in mm (relative to well center)
            Nx (int): Number of positions in X
            Ny (int): Number of positions in Y
            dx_mm (float): Interval between positions in X (mm)
            dy_mm (float): Interval between positions in Y (mm)
            illumination_settings (list): List of channel settings
            do_contrast_autofocus (bool): Whether to perform contrast-based autofocus
            do_reflection_af (bool): Whether to perform reflection-based autofocus
            action_ID (str): Identifier for this scan
            timepoint (int): Timepoint index for the scan (default 0)
            experiment_name (str, optional): Name of the experiment to use. If None, uses active experiment or creates "default"
            wells_to_scan (list): List of well strings (e.g., ['A1', 'B2']) or (row, column) tuples. If None, scans single well at current position
            wellplate_type (str): Well plate type ('6', '12', '24', '96', '384')
            well_padding_mm (float): Padding around well in mm
        """
        if illumination_settings is None:
            illumination_settings = [
                {'channel': 'BF LED matrix full', 'intensity': 50, 'exposure_time': 100}
            ]
        
        # Ensure we have an active experiment
        self.ensure_active_experiment(experiment_name)
        
        # Determine wells to scan
        if wells_to_scan is None:
            # Get current position and determine which well we're in
            current_x, current_y, current_z, _ = self.navigationController.update_pos(self.microcontroller)
            well_info = self.get_well_from_position(wellplate_type, current_x, current_y)
            
            if well_info['position_status'] != 'in_well':
                raise RuntimeError(f"Current position ({current_x:.2f}, {current_y:.2f}) is not inside a well. Please specify wells_to_scan or move to a well first.")
            
            wells_to_scan = [(well_info['row'], well_info['column'])]
            logger.info(f"Auto-detected current well: {well_info['row']}{well_info['column']}")
        
        # Convert wells_to_scan to tuple format if needed
        wells_to_scan = self._convert_well_strings_to_tuples(wells_to_scan)
        
        # Validate wells_to_scan format
        if not isinstance(wells_to_scan, list) or not wells_to_scan:
            raise ValueError("wells_to_scan must be a non-empty list of well strings or (row, column) tuples")
        
        logger.info(f"Normal scan with stitching for experiment '{self.experiment_manager.current_experiment}', scanning {len(wells_to_scan)} wells")
        
        # Map channel names to indices
        channel_map = ChannelMapper.get_human_to_id_map()
        
        # Start scanning wells one by one
        try:
            self.is_busy = True
            self.scan_stop_requested = False  # Reset stop flag at start of scan
            logger.info(f'Starting normal scan with stitching: {Nx}x{Ny} positions per well, dx={dx_mm}mm, dy={dy_mm}mm, timepoint={timepoint}')
            
            for well_idx, (well_row, well_column) in enumerate(wells_to_scan):
                if self.scan_stop_requested:
                    logger.info("Scan stopped by user request")
                    self._restore_original_velocity(CONFIG.MAX_VELOCITY_X_MM, CONFIG.MAX_VELOCITY_Y_MM)
                    break
                
                logger.info(f"Scanning well {well_row}{well_column} ({well_idx + 1}/{len(wells_to_scan)})")
                
                # Get well canvas for this well
                canvas = self.experiment_manager.get_well_canvas(well_row, well_column, wellplate_type, well_padding_mm)
                
                # Validate channels are available in this canvas
                for settings in illumination_settings:
                    channel_name = settings['channel']
                    if channel_name not in canvas.channel_to_zarr_index:
                        logger.error(f"Requested channel '{channel_name}' not found in well canvas!")
                        logger.error(f"Available channels: {list(canvas.channel_to_zarr_index.keys())}")
                        raise ValueError(f"Channel '{channel_name}' not available in well canvas")
                
                # Start stitching for this well
                await canvas.start_stitching()
                
                # Move to well center first
                await self.move_to_well_async(well_row, well_column, wellplate_type)
                
                # Get well center coordinates for relative positioning
                well_center_x = canvas.well_center_x
                well_center_y = canvas.well_center_y
                
                try:
                    # Scan pattern: snake pattern for efficiency
                    for i in range(Ny):
                        # Check for stop request before each row
                        if self.scan_stop_requested:
                            logger.info("Scan stopped by user request")
                            break
                            
                        for j in range(Nx):
                            # Check for stop request before each position
                            if self.scan_stop_requested:
                                logger.info("Scan stopped by user request")
                                break
                            
                            # Calculate position (snake pattern - reverse X on odd rows)
                            if i % 2 == 0:
                                x_idx = j
                            else:
                                x_idx = Nx - 1 - j
                            
                            # Calculate absolute position (well center + relative offset)
                            absolute_x_mm = well_center_x + start_x_mm + x_idx * dx_mm
                            absolute_y_mm = well_center_y + start_y_mm + i * dy_mm
                            
                            # Move to position
                            self.navigationController.move_x_to(absolute_x_mm)
                            self.navigationController.move_y_to(absolute_y_mm)
                            while self.microcontroller.is_busy():
                                await asyncio.sleep(0.005)
                            
                            # Let stage settle
                            await asyncio.sleep(CONFIG.SCAN_STABILIZATION_TIME_MS_X / 1000)
                            
                            # Update position from microcontroller to get actual stage position
                            actual_x_mm, actual_y_mm, actual_z_mm, _ = self.navigationController.update_pos(self.microcontroller)
                            
                            # Autofocus if requested (first position or periodically)
                            if do_reflection_af and (i == 0 and j == 0):
                                if hasattr(self, 'laserAutofocusController'):
                                    await self.do_laser_autofocus()
                                    # Update position again after autofocus
                                    actual_x_mm, actual_y_mm, actual_z_mm, _ = self.navigationController.update_pos(self.microcontroller)
                            elif do_contrast_autofocus and ((i * Nx + j) % CONFIG.Acquisition.NUMBER_OF_FOVS_PER_AF == 0):
                                await self.do_autofocus()
                                # Update position again after autofocus
                                actual_x_mm, actual_y_mm, actual_z_mm, _ = self.navigationController.update_pos(self.microcontroller)
                            
                            # Acquire images for each channel
                            for idx, settings in enumerate(illumination_settings):
                                channel_name = settings['channel']
                                intensity = settings['intensity']
                                exposure_time = settings['exposure_time']
                                
                                # Get global channel index for snap_image (uses global channel IDs)
                                global_channel_idx = channel_map.get(channel_name, 0)
                                
                                # Get local zarr channel index (0, 1, 2, etc.)
                                try:
                                    zarr_channel_idx = canvas.get_zarr_channel_index(channel_name)
                                except ValueError as e:
                                    logger.error(f"Channel mapping error: {e}")
                                    continue
                                
                                # Snap image using global channel ID with full frame for stitching
                                image = await self.snap_image(global_channel_idx, intensity, exposure_time, full_frame=True)
                                
                                # Convert to 8-bit if needed
                                if image.dtype != np.uint8:
                                    # Scale to 8-bit
                                    if image.dtype == np.uint16:
                                        image = (image / 256).astype(np.uint8)
                                    else:
                                        image = image.astype(np.uint8)
                                
                                # Add to stitching queue using the new well-based routing method (like quick scan)
                                await self._add_image_to_zarr_normal_well_based(
                                    image, actual_x_mm, actual_y_mm,
                                    zarr_channel_idx=zarr_channel_idx,
                                    timepoint=timepoint,
                                    wellplate_type=wellplate_type,
                                    well_padding_mm=well_padding_mm,
                                    channel_name=channel_name
                                )
                                
                                logger.debug(f'Added image at position ({actual_x_mm:.2f}, {actual_y_mm:.2f}) for well {well_row}{well_column}, channel {channel_name}, timepoint={timepoint}')
                
                finally:
                    # Stop stitching for this well
                    await canvas.stop_stitching()
                    logger.info(f'Completed scanning well {well_row}{well_column}')
            
            logger.info('Normal scan with stitching completed for all wells')
            
        finally:
            self.is_busy = False
            # Additional delay to ensure all zarr operations are complete
            logger.info('Waiting for all zarr operations to stabilize...')
            await asyncio.sleep(0.5)  # 500ms buffer to ensure filesystem operations complete
            logger.info('Normal scan with stitching fully completed - zarr data ready for export')
    

    
    def stop_scan_and_stitching(self):
        """
        Stop any ongoing scanning and stitching processes.
        This will interrupt normal_scan_with_stitching and quick_scan_with_stitching.
        """
        self.scan_stop_requested = True
        logger.info("Scan stop requested - ongoing scans will be interrupted")
        self._restore_original_velocity(CONFIG.MAX_VELOCITY_X_MM, CONFIG.MAX_VELOCITY_Y_MM)
        return {"success": True, "message": "Scan stop requested"}
    
    async def _add_image_to_zarr_quick_well_based(self, image: np.ndarray, x_mm: float, y_mm: float,
                                                 zarr_channel_idx: int, timepoint: int = 0, 
                                                 wellplate_type='96', well_padding_mm=1.0, channel_name='BF LED matrix full'):
        """
        Add image to well canvas stitching queue for quick scan - only updates scales 1-5 (skips scale 0).
        The input image should already be at scale1 resolution (1/4 of original).
        
        Args:
            image: Processed image at scale1 resolution
            x_mm: Absolute X position in mm
            y_mm: Absolute Y position in mm
            zarr_channel_idx: Zarr channel index
            timepoint: Timepoint index
            wellplate_type: Well plate type
            well_padding_mm: Well padding in mm
            channel_name: Channel name for validation
        """
        logger.info(f'ZARR_QUEUE: Attempting to queue image at position ({x_mm:.2f}, {y_mm:.2f}), timepoint={timepoint}, channel={channel_name}')
        
        # Determine which well this position belongs to using padded boundaries for stitching
        well_info = self.get_well_from_position(wellplate_type, x_mm, y_mm, well_padding_mm)
        
        logger.info(f'ZARR_QUEUE: Well detection result - status={well_info["position_status"]}, well={well_info.get("well_id", "None")}, distance={well_info["distance_from_center"]:.2f}mm')
        
        if well_info["position_status"] == "in_well":
            well_row = well_info["row"]
            well_column = well_info["column"]
            
            logger.info(f'ZARR_QUEUE: Position is inside well {well_row}{well_column}')
            
            # Get or create well canvas
            try:
                well_canvas = self.experiment_manager.get_well_canvas(well_row, well_column, wellplate_type, well_padding_mm)
                logger.info(f'ZARR_QUEUE: Got well canvas for {well_row}{well_column}, stitching_active={well_canvas.is_stitching}')
            except Exception as e:
                logger.error(f"ZARR_QUEUE: Failed to get well canvas for {well_row}{well_column}: {e}")
                return f"Failed to get well canvas: {e}"
            
            # Validate channel exists in this well canvas
            if channel_name not in well_canvas.channel_to_zarr_index:
                logger.warning(f"ZARR_QUEUE: Channel '{channel_name}' not found in well canvas {well_row}{well_column}")
                logger.warning(f"ZARR_QUEUE: Available channels: {list(well_canvas.channel_to_zarr_index.keys())}")
                return f"Channel {channel_name} not found"
            
            # Note: WellZarrCanvas will handle coordinate conversion internally
            logger.info(f'ZARR_QUEUE: Using absolute coordinates: ({x_mm:.2f}, {y_mm:.2f}), well_center: ({well_canvas.well_center_x:.2f}, {well_canvas.well_center_y:.2f})')
            
            # Add to stitching queue with quick_scan flag
            try:
                queue_item = {
                    'image': image.copy(),
                    'x_mm': x_mm,  # Use absolute coordinates - WellZarrCanvas will convert to well-relative
                    'y_mm': y_mm,  # Use absolute coordinates - WellZarrCanvas will convert to well-relative
                    'channel_idx': zarr_channel_idx,
                    'z_idx': 0,
                    'timepoint': timepoint,
                    'timestamp': time.time(),
                    'quick_scan': True  # Flag to indicate this is for quick scan (scales 1-5 only)
                }
                
                # Check queue size before adding
                queue_size_before = well_canvas.stitch_queue.qsize()
                await well_canvas.stitch_queue.put(queue_item)
                queue_size_after = well_canvas.stitch_queue.qsize()
                
                logger.info(f'ZARR_QUEUE: Successfully queued image for well {well_row}{well_column} at absolute coords ({x_mm:.2f}, {y_mm:.2f})')
                logger.info(f'ZARR_QUEUE: Queue size before={queue_size_before}, after={queue_size_after}')
                return f"Queued for well {well_row}{well_column}"
                
            except Exception as e:
                logger.error(f"ZARR_QUEUE: Failed to add image to stitching queue for well {well_row}{well_column}: {e}")
                return f"Failed to queue: {e}"
        else:
            # Image is outside wells - log and skip
            logger.warning(f'ZARR_QUEUE: Image at ({x_mm:.2f}, {y_mm:.2f}) is {well_info["position_status"]} - skipping')
            return f"Position outside well: {well_info['position_status']}"

    def debug_stitching_status(self):
        """Debug method to check stitching status of all well canvases."""
        logger.info("STITCHING_DEBUG: Checking stitching status for all well canvases")
        
        if hasattr(self, 'experiment_manager') and hasattr(self.experiment_manager, 'well_canvases'):
            for well_id, well_canvas in self.experiment_manager.well_canvases.items():
                queue_size = well_canvas.stitch_queue.qsize()
                is_stitching = well_canvas.is_stitching
                logger.info(f"STITCHING_DEBUG: Well {well_id} - stitching_active={is_stitching}, queue_size={queue_size}")
                
                # Check if stitching task is running
                if hasattr(well_canvas, 'stitching_task'):
                    task_done = well_canvas.stitching_task.done() if well_canvas.stitching_task else True
                    logger.info(f"STITCHING_DEBUG: Well {well_id} - stitching_task_done={task_done}")
        else:
            logger.warning("STITCHING_DEBUG: No well canvases found in experiment manager")

    def _cleanup_zarr_directory(self):
        # Clean up .zarr folders within ZARR_PATH directory on startup
        zarr_path = os.getenv('ZARR_PATH', '/tmp/zarr_canvas')
        if os.path.exists(zarr_path):
            try:
                # Only delete .zarr folders, not the entire directory
                deleted_count = 0
                for item in os.listdir(zarr_path):
                    item_path = os.path.join(zarr_path, item)
                    if os.path.isdir(item_path) and item.endswith('.zarr'):
                        try:
                            shutil.rmtree(item_path)
                            deleted_count += 1
                            logger.info(f'Cleaned up zarr folder: {item_path}')
                        except Exception as e:
                            logger.warning(f'Failed to clean up zarr folder {item_path}: {e}')
                
                if deleted_count > 0:
                    logger.info(f'Cleaned up {deleted_count} zarr folders in {zarr_path}')
                else:
                    logger.info(f'No zarr folders found to clean up in {zarr_path}')
                    
            except Exception as e:
                logger.error(f'Failed to access ZARR_PATH directory {zarr_path}: {e}')

    def create_zarr_fileset(self, fileset_name):
        """Create a new zarr fileset with the given name.
        
        Args:
            fileset_name: Name for the new fileset
            
        Returns:
            dict: Information about the created fileset
            
        Raises:
            ValueError: If fileset already exists
            RuntimeError: If fileset creation fails
        """
        if fileset_name in self.zarr_canvases:
            raise ValueError(f"Fileset '{fileset_name}' already exists")
        
        try:
            # Initialize new canvas
            self._initialize_empty_canvas(fileset_name)
            
            return {
                "fileset_name": fileset_name,
                "is_active": self.active_canvas_name == fileset_name,
                "message": f"Created new zarr fileset '{fileset_name}'"
            }
            
        except Exception as e:
            logger.error(f"Failed to create zarr fileset '{fileset_name}': {e}")
            raise RuntimeError(f"Failed to create fileset '{fileset_name}': {str(e)}") from e
    
    def list_zarr_filesets(self):
        """List all available zarr filesets.
        
        Returns:
            dict: List of filesets and their status
            
        Raises:
            RuntimeError: If listing filesets fails
        """
        try:
            zarr_path = os.getenv('ZARR_PATH', '/tmp/zarr_canvas')
            zarr_dir = Path(zarr_path)
            
            filesets = []
            
            # List from memory (already loaded)
            for name, canvas in self.zarr_canvases.items():
                filesets.append({
                    "name": name,
                    "is_active": name == self.active_canvas_name,
                    "loaded": True,
                    "path": str(canvas.zarr_path),
                    "channels": len(canvas.channels),
                    "timepoints": len(canvas.available_timepoints)
                })
            
            # Also check disk for any zarr directories not in memory
            if zarr_dir.exists():
                for item in zarr_dir.iterdir():
                    if item.is_dir() and item.suffix == '.zarr':
                        name = item.stem
                        if name not in self.zarr_canvases:
                            filesets.append({
                                "name": name,
                                "is_active": False,
                                "loaded": False,
                                "path": str(item)
                            })
            
            return {
                "filesets": filesets,
                "active_fileset": self.active_canvas_name,
                "total_count": len(filesets)
            }
            
        except Exception as e:
            logger.error(f"Failed to list zarr filesets: {e}")
            raise RuntimeError(f"Failed to list filesets: {str(e)}") from e
    
    def set_active_zarr_fileset(self, fileset_name):
        """Set the active zarr fileset for operations.
        
        Args:
            fileset_name: Name of the fileset to activate
            
        Returns:
            dict: Information about the activated fileset
            
        Raises:
            ValueError: If fileset not found
            RuntimeError: If activation fails
        """
        try:
            # Check if already active
            if self.active_canvas_name == fileset_name:
                return {
                    "message": f"Fileset '{fileset_name}' is already active",
                    "fileset_name": fileset_name,
                    "was_already_active": True
                }
            
            # Check if fileset exists in memory
            if fileset_name in self.zarr_canvases:
                self.active_canvas_name = fileset_name
                self.zarr_canvas = self.zarr_canvases[fileset_name]
                return {
                    "message": f"Activated fileset '{fileset_name}'",
                    "fileset_name": fileset_name,
                    "was_already_active": False
                }
            
            # Try to load from disk
            zarr_path = os.getenv('ZARR_PATH', '/tmp/zarr_canvas')
            zarr_dir = Path(zarr_path)
            fileset_path = zarr_dir / f"{fileset_name}.zarr"
            
            if fileset_path.exists():
                # Open existing canvas without deleting data
                stage_limits = {
                    'x_positive': 120,
                    'x_negative': 0,
                    'y_positive': 86,
                    'y_negative': 0,
                    'z_positive': 6
                }
                default_channels = ChannelMapper.get_all_human_names()
                canvas = ZarrCanvas(
                    base_path=zarr_path,
                    pixel_size_xy_um=self.pixel_size_xy,
                    stage_limits=stage_limits,
                    channels=default_channels,
                    rotation_angle_deg=CONFIG.STITCHING_ROTATION_ANGLE_DEG,
                    initial_timepoints=20,
                    timepoint_expansion_chunk=10,
                    fileset_name=fileset_name,
                    initialize_new=False
                )
                self.zarr_canvases[fileset_name] = canvas
                self.active_canvas_name = fileset_name
                self.zarr_canvas = canvas
                return {
                    "message": f"Loaded and activated fileset '{fileset_name}' from disk",
                    "fileset_name": fileset_name,
                    "was_already_active": False
                }
            raise ValueError(f"Fileset '{fileset_name}' not found")
        except Exception as e:
            logger.error(f"Failed to set active zarr fileset '{fileset_name}': {e}")
            if isinstance(e, ValueError):
                raise
            raise RuntimeError(f"Failed to activate fileset '{fileset_name}': {str(e)}") from e
    
    def remove_zarr_fileset(self, fileset_name):
        """Remove a zarr fileset.
        
        Args:
            fileset_name: Name of the fileset to remove
            
        Returns:
            dict: Information about the removed fileset
            
        Raises:
            ValueError: If trying to remove active fileset
            RuntimeError: If removal fails
        """
        # Can't remove active fileset
        if self.active_canvas_name == fileset_name:
            raise ValueError(f"Cannot remove active fileset '{fileset_name}'. Please switch to another fileset first.")
        
        try:
            removed_from_memory = False
            removed_from_disk = False
            
            # Remove from memory if loaded
            if fileset_name in self.zarr_canvases:
                canvas = self.zarr_canvases[fileset_name]
                canvas.close()
                del self.zarr_canvases[fileset_name]
                removed_from_memory = True
            
            # Remove from disk
            zarr_path = os.getenv('ZARR_PATH', '/tmp/zarr_canvas')
            zarr_dir = Path(zarr_path)
            fileset_path = zarr_dir / f"{fileset_name}.zarr"
            
            if fileset_path.exists():
                shutil.rmtree(fileset_path)
                logger.info(f"Removed zarr fileset '{fileset_name}' from disk")
                removed_from_disk = True
            
            return {
                "message": f"Removed zarr fileset '{fileset_name}'",
                "fileset_name": fileset_name,
                "removed_from_memory": removed_from_memory,
                "removed_from_disk": removed_from_disk
            }
            
        except Exception as e:
            logger.error(f"Failed to remove zarr fileset '{fileset_name}': {e}")
            raise RuntimeError(f"Failed to remove fileset '{fileset_name}': {str(e)}") from e
    
    def get_active_canvas(self):
        """Get the currently active zarr canvas, ensuring one exists.
        
        Returns:
            ZarrCanvas: The active canvas, or None if no canvas is active
        """
        if self.zarr_canvas is None and self.active_canvas_name is None:
            # No canvas exists, create default one
            logger.info("No active zarr canvas, creating default fileset")
            self.create_zarr_fileset("default")
        
        return self.zarr_canvas

    def get_well_canvas(self, well_row: str, well_column: int, wellplate_type: str = '96', 
                       padding_mm: float = 1.0):
        """
        Get or create a well-specific canvas.
        
        Args:
            well_row: Well row (e.g., 'A', 'B')
            well_column: Well column (e.g., 1, 2, 3)
            wellplate_type: Well plate type ('6', '12', '24', '96', '384')
            padding_mm: Padding around well in mm
            
        Returns:
            WellZarrCanvas: The well-specific canvas
        """
        well_id = f"{well_row}{well_column}_{wellplate_type}"
        
        if well_id not in self.well_canvases:
            # Create new well canvas
            zarr_path = os.getenv('ZARR_PATH', '/tmp/zarr_canvas')
            all_channels = ChannelMapper.get_all_human_names()
            
            canvas = WellZarrCanvas(
                well_row=well_row,
                well_column=well_column,
                wellplate_type=wellplate_type,
                padding_mm=padding_mm,
                base_path=zarr_path,
                pixel_size_xy_um=self.pixel_size_xy,
                channels=all_channels,
                rotation_angle_deg=CONFIG.STITCHING_ROTATION_ANGLE_DEG,
                initial_timepoints=20,
                timepoint_expansion_chunk=10
            )
            
            self.well_canvases[well_id] = canvas
            logger.info(f"Created well canvas for {well_row}{well_column} ({wellplate_type})")
            
        return self.well_canvases[well_id]
    
    def create_well_canvas(self, well_row: str, well_column: int, wellplate_type: str = '96',
                          padding_mm: float = 1.0):
        """
        Create a new well-specific canvas (replaces existing if present).
        
        Args:
            well_row: Well row (e.g., 'A', 'B')
            well_column: Well column (e.g., 1, 2, 3)
            wellplate_type: Well plate type ('6', '12', '24', '96', '384')
            padding_mm: Padding around well in mm
            
        Returns:
            dict: Information about the created canvas
        """
        well_id = f"{well_row}{well_column}_{wellplate_type}"
        
        # Close existing canvas if present
        if well_id in self.well_canvases:
            self.well_canvases[well_id].close()
            logger.info(f"Closed existing well canvas for {well_row}{well_column}")
        
        # Create new canvas
        canvas = self.get_well_canvas(well_row, well_column, wellplate_type, padding_mm)
        
        return {
            "well_id": well_id,
            "well_row": well_row,
            "well_column": well_column,
            "wellplate_type": wellplate_type,
            "padding_mm": padding_mm,
            "canvas_path": str(canvas.zarr_path),
            "message": f"Created well canvas for {well_row}{well_column}"
        }
    
    def list_well_canvases(self):
        """
        List all active well canvases.
        
        Returns:
            dict: Information about all well canvases
        """
        canvases = []
        
        for well_id, canvas in self.well_canvases.items():
            well_info = canvas.get_well_info()
            canvases.append({
                "well_id": well_id,
                "well_row": canvas.well_row,
                "well_column": canvas.well_column,
                "wellplate_type": canvas.wellplate_type,
                "canvas_path": str(canvas.zarr_path),
                "well_center_x_mm": canvas.well_center_x,
                "well_center_y_mm": canvas.well_center_y,
                "padding_mm": canvas.padding_mm,
                "channels": len(canvas.channels),
                "timepoints": len(canvas.available_timepoints)
            })
        
        return {
            "well_canvases": canvases,
            "total_count": len(canvases)
        }
    
    def remove_well_canvas(self, well_row: str, well_column: int, wellplate_type: str = '96'):
        """
        Remove a well-specific canvas.
        
        Args:
            well_row: Well row (e.g., 'A', 'B')
            well_column: Well column (e.g., 1, 2, 3)
            wellplate_type: Well plate type
            
        Returns:
            dict: Information about the removed canvas
        """
        well_id = f"{well_row}{well_column}_{wellplate_type}"
        
        if well_id not in self.well_canvases:
            raise ValueError(f"Well canvas for {well_row}{well_column} ({wellplate_type}) not found")
        
        canvas = self.well_canvases[well_id]
        canvas.close()
        del self.well_canvases[well_id]
        
        # Also remove from disk
        try:
            import shutil
            if canvas.zarr_path.exists():
                shutil.rmtree(canvas.zarr_path)
                logger.info(f"Removed well canvas directory: {canvas.zarr_path}")
        except Exception as e:
            logger.warning(f"Failed to remove well canvas directory: {e}")
        
        return {
            "well_id": well_id,
            "well_row": well_row,
            "well_column": well_column,
            "wellplate_type": wellplate_type,
            "message": f"Removed well canvas for {well_row}{well_column}"
        }

    def _convert_well_strings_to_tuples(self, wells_to_scan):
        """
        Convert a list of well strings (e.g., ['A1', 'B2', 'C3']) to a list of tuples (e.g., [('A', 1), ('B', 2), ('C', 3)]).
        
        Args:
            wells_to_scan: List of well strings or tuples
            
        Returns:
            List of (row, column) tuples
        """
        if not wells_to_scan:
            return []
        
        converted_wells = []
        for well in wells_to_scan:
            if isinstance(well, str):
                # Parse string format like 'A1', 'B2', etc.
                if len(well) >= 2:
                    row = well[0].upper()  # First character is row (A, B, C, etc.)
                    try:
                        column = int(well[1:])  # Rest is column number
                        converted_wells.append((row, column))
                    except ValueError:
                        logger.warning(f"Invalid well format '{well}', skipping")
                        continue
                else:
                    logger.warning(f"Invalid well format '{well}', skipping")
                    continue
            elif isinstance(well, (list, tuple)) and len(well) == 2:
                # Already in tuple format
                row, column = well
                if isinstance(row, str) and isinstance(column, (int, str)):
                    if isinstance(column, str):
                        try:
                            column = int(column)
                        except ValueError:
                            logger.warning(f"Invalid column number '{column}' in well {well}, skipping")
                            continue
                    converted_wells.append((row, column))
                else:
                    logger.warning(f"Invalid well format {well}, skipping")
                    continue
            else:
                logger.warning(f"Invalid well format {well}, skipping")
                continue
        
        return converted_wells

    async def _add_image_to_zarr_normal_well_based(self, image: np.ndarray, x_mm: float, y_mm: float,
                                                 zarr_channel_idx: int, timepoint: int = 0, 
                                                 wellplate_type='96', well_padding_mm=1.0, channel_name='BF LED matrix full'):
        """
        Add image to well canvas stitching queue for normal scan - updates all scales (0-5).
        Uses the same routing logic as quick scan but with full scale processing.
        
        Args:
            image: Processed image (original resolution)
            x_mm: Absolute X position in mm
            y_mm: Absolute Y position in mm
            zarr_channel_idx: Zarr channel index
            timepoint: Timepoint index
            wellplate_type: Well plate type
            well_padding_mm: Well padding in mm
            channel_name: Channel name for validation
        """
        logger.info(f'ZARR_NORMAL: Attempting to queue image at position ({x_mm:.2f}, {y_mm:.2f}), timepoint={timepoint}, channel={channel_name}')
        
        # Determine which well this position belongs to using padded boundaries for stitching
        well_info = self.get_well_from_position(wellplate_type, x_mm, y_mm, well_padding_mm)
        
        logger.info(f'ZARR_NORMAL: Well detection result - status={well_info["position_status"]}, well={well_info.get("well_id", "None")}, distance={well_info["distance_from_center"]:.2f}mm')
        
        if well_info["position_status"] == "in_well":
            well_row = well_info["row"]
            well_column = well_info["column"]
            
            logger.info(f'ZARR_NORMAL: Position is inside well {well_row}{well_column}')
            
            # Get or create well canvas
            try:
                well_canvas = self.experiment_manager.get_well_canvas(well_row, well_column, wellplate_type, well_padding_mm)
                logger.info(f'ZARR_NORMAL: Got well canvas for {well_row}{well_column}, stitching_active={well_canvas.is_stitching}')
            except Exception as e:
                logger.error(f"ZARR_NORMAL: Failed to get well canvas for {well_row}{well_column}: {e}")
                return f"Failed to get well canvas: {e}"
            
            # Validate channel exists in this well canvas
            if channel_name not in well_canvas.channel_to_zarr_index:
                logger.warning(f"ZARR_NORMAL: Channel '{channel_name}' not found in well canvas {well_row}{well_column}")
                logger.warning(f"ZARR_NORMAL: Available channels: {list(well_canvas.channel_to_zarr_index.keys())}")
                return f"Channel {channel_name} not found"
            
            logger.info(f'ZARR_NORMAL: Well center: ({well_canvas.well_center_x:.2f}, {well_canvas.well_center_y:.2f})')
            
            # Add to stitching queue with normal scan flag (all scales)
            try:
                queue_item = {
                    'image': image.copy(),
                    'x_mm': x_mm,  # Use absolute coordinates - WellZarrCanvas will convert to well-relative (same as quick scan)
                    'y_mm': y_mm,  # Use absolute coordinates - WellZarrCanvas will convert to well-relative (same as quick scan)
                    'channel_idx': zarr_channel_idx,
                    'z_idx': 0,
                    'timepoint': timepoint,
                    'timestamp': time.time(),
                    'quick_scan': False  # Flag to indicate this is normal scan (all scales)
                }
                
                # Check queue size before adding
                queue_size_before = well_canvas.stitch_queue.qsize()
                await well_canvas.stitch_queue.put(queue_item)
                queue_size_after = well_canvas.stitch_queue.qsize()
                
                logger.info(f'ZARR_NORMAL: Queue size before={queue_size_before}, after={queue_size_after}')
                return f"Queued for well {well_row}{well_column}"
                
            except Exception as e:
                logger.error(f"ZARR_NORMAL: Failed to add image to stitching queue for well {well_row}{well_column}: {e}")
                return f"Failed to queue: {e}"
        else:
            # Image is outside wells - log and skip
            logger.warning(f'ZARR_NORMAL: Image at ({x_mm:.2f}, {y_mm:.2f}) is {well_info["position_status"]} - skipping')
            return f"Position outside well: {well_info['position_status']}"

    def ensure_active_experiment(self, experiment_name: str = None):
        """
        Ensure there's an active experiment, creating a default one if needed.
        
        Args:
            experiment_name: Name of experiment to create/activate (default: creates "default")
        """
        if experiment_name is None:
            experiment_name = "default"
        
        if self.experiment_manager.current_experiment is None:
            # No active experiment, create or set one
            try:
                self.experiment_manager.set_active_experiment(experiment_name)
                logger.info(f"Set active experiment to existing '{experiment_name}'")
            except ValueError:
                # Experiment doesn't exist, create it
                self.experiment_manager.create_experiment(experiment_name)
                logger.info(f"Created new experiment '{experiment_name}'")
    
    def _check_well_canvas_exists(self, well_row: str, well_column: int, wellplate_type: str = '96'):
        """
        Check if a well canvas exists on disk for the current experiment.
        
        Args:
            well_row: Well row (e.g., 'A', 'B')
            well_column: Well column (e.g., 1, 2, 3)
            wellplate_type: Well plate type ('6', '12', '24', '96', '384')
            
        Returns:
            bool: True if the well canvas exists on disk, False otherwise
        """
        if self.experiment_manager.current_experiment is None:
            return False
        
        # Calculate the expected canvas path
        experiment_path = self.experiment_manager.base_path / self.experiment_manager.current_experiment
        fileset_name = f"well_{well_row}{well_column}_{wellplate_type}"
        canvas_path = experiment_path / f"{fileset_name}.zarr"
        
        return canvas_path.exists()

    def get_stitched_region(self, center_x_mm: float, center_y_mm: float, 
                           width_mm: float, height_mm: float,
                           wellplate_type: str = '96', scale_level: int = 0, 
                           channel_name: str = 'BF LED matrix full', 
                           timepoint: int = 0, well_padding_mm: float = 2.0):
        """
        Get a stitched region that may span multiple wells by determining which wells 
        are needed and combining their data.
        
        Args:
            center_x_mm: Center X position in absolute stage coordinates (mm)
            center_y_mm: Center Y position in absolute stage coordinates (mm)
            width_mm: Width of region in mm
            height_mm: Height of region in mm
            wellplate_type: Well plate type ('6', '12', '24', '96', '384')
            scale_level: Scale level (0=full res, 1=1/4, 2=1/16, etc)
            channel_name: Name of channel to retrieve
            timepoint: Timepoint index (default 0)
            well_padding_mm: Padding around wells in mm
            
        Returns:
            np.ndarray: The requested region, or None if not available
        """
        try:
            # Calculate the bounding box of the requested region
            half_width = width_mm / 2.0
            half_height = height_mm / 2.0
            
            region_min_x = center_x_mm - half_width
            region_max_x = center_x_mm + half_width
            region_min_y = center_y_mm - half_height
            region_max_y = center_y_mm + half_height
            
            logger.info(f"Requested region: center=({center_x_mm:.2f}, {center_y_mm:.2f}), "
                       f"size=({width_mm:.2f}x{height_mm:.2f}), "
                       f"bounds=({region_min_x:.2f}-{region_max_x:.2f}, {region_min_y:.2f}-{region_max_y:.2f})")
            
            # Get well plate format configuration
            if wellplate_type == '6':
                wellplate_format = WELLPLATE_FORMAT_6
                max_rows = 2  # A-B
                max_cols = 3  # 1-3
            elif wellplate_type == '12':
                wellplate_format = WELLPLATE_FORMAT_12
                max_rows = 3  # A-C
                max_cols = 4  # 1-4
            elif wellplate_type == '24':
                wellplate_format = WELLPLATE_FORMAT_24
                max_rows = 4  # A-D
                max_cols = 6  # 1-6
            elif wellplate_type == '96':
                wellplate_format = WELLPLATE_FORMAT_96
                max_rows = 8  # A-H
                max_cols = 12  # 1-12
            elif wellplate_type == '384':
                wellplate_format = WELLPLATE_FORMAT_384
                max_rows = 16  # A-P
                max_cols = 24  # 1-24
            else:
                wellplate_format = WELLPLATE_FORMAT_96
                max_rows = 8
                max_cols = 12
                wellplate_type = '96'
            
            # Apply well plate offset for hardware mode
            if self.is_simulation:
                x_offset = 0
                y_offset = 0
            else:
                x_offset = CONFIG.WELLPLATE_OFFSET_X_MM
                y_offset = CONFIG.WELLPLATE_OFFSET_Y_MM
            
            # Find all wells that intersect with the requested region
            wells_to_query = []
            well_regions = []
            
            for row_idx in range(max_rows):
                for col_idx in range(max_cols):
                    # Calculate well center position
                    well_center_x = wellplate_format.A1_X_MM + x_offset + col_idx * wellplate_format.WELL_SPACING_MM
                    well_center_y = wellplate_format.A1_Y_MM + y_offset + row_idx * wellplate_format.WELL_SPACING_MM
                    
                    # Calculate well boundaries with padding
                    well_radius = wellplate_format.WELL_SIZE_MM / 2.0
                    padded_radius = well_radius + well_padding_mm
                    
                    well_min_x = well_center_x - padded_radius
                    well_max_x = well_center_x + padded_radius
                    well_min_y = well_center_y - padded_radius
                    well_max_y = well_center_y + padded_radius
                    
                    # Check if this well intersects with the requested region
                    if (well_max_x >= region_min_x and well_min_x <= region_max_x and
                        well_max_y >= region_min_y and well_min_y <= region_max_y):
                        
                        well_row = chr(ord('A') + row_idx)
                        well_column = col_idx + 1
                        
                        # Calculate the intersection region in well-relative coordinates
                        intersection_min_x = max(region_min_x, well_min_x)
                        intersection_max_x = min(region_max_x, well_max_x)
                        intersection_min_y = max(region_min_y, well_min_y)
                        intersection_max_y = min(region_max_y, well_max_y)
                        
                        # Convert to well-relative coordinates
                        well_rel_center_x = ((intersection_min_x + intersection_max_x) / 2.0) - well_center_x
                        well_rel_center_y = ((intersection_min_y + intersection_max_y) / 2.0) - well_center_y
                        well_rel_width = intersection_max_x - intersection_min_x
                        well_rel_height = intersection_max_y - intersection_min_y
                        
                        wells_to_query.append((well_row, well_column))
                        well_regions.append({
                            'well_row': well_row,
                            'well_column': well_column,
                            'well_center_x': well_center_x,
                            'well_center_y': well_center_y,
                            'well_rel_center_x': well_rel_center_x,
                            'well_rel_center_y': well_rel_center_y,
                            'well_rel_width': well_rel_width,
                            'well_rel_height': well_rel_height,
                            'abs_min_x': intersection_min_x,
                            'abs_max_x': intersection_max_x,
                            'abs_min_y': intersection_min_y,
                            'abs_max_y': intersection_max_y
                        })
            
            if not wells_to_query:
                logger.warning(f"No wells found that intersect with requested region")
                return None
            
            logger.info(f"Found {len(wells_to_query)} wells that intersect with requested region: {wells_to_query}")
            
            # If only one well, get the region directly
            if len(wells_to_query) == 1:
                well_info = well_regions[0]
                well_row = well_info['well_row']
                well_column = well_info['well_column']
                
                # Check if the well canvas exists
                if not self._check_well_canvas_exists(well_row, well_column, wellplate_type):
                    logger.warning(f"Well canvas for {well_row}{well_column} ({wellplate_type}) does not exist")
                    return None
                
                # Get well canvas and extract region
                canvas = self.experiment_manager.get_well_canvas(well_row, well_column, wellplate_type, well_padding_mm)
                
                # Calculate absolute coordinates for the intersection region
                intersection_center_x = (well_info['abs_min_x'] + well_info['abs_max_x']) / 2.0
                intersection_center_y = (well_info['abs_min_y'] + well_info['abs_max_y']) / 2.0
                intersection_width = well_info['abs_max_x'] - well_info['abs_min_x']
                intersection_height = well_info['abs_max_y'] - well_info['abs_min_y']
                
                region = canvas.get_canvas_region_by_channel_name(
                    intersection_center_x, intersection_center_y, 
                    intersection_width, intersection_height,
                    channel_name, scale=scale_level, timepoint=timepoint
                )
                
                if region is None:
                    logger.warning(f"Failed to get region from well {well_row}{well_column}")
                    return None
                
                logger.info(f"Retrieved single-well region from {well_row}{well_column}, shape: {region.shape}")
                return region
            
            # Multiple wells - need to stitch them together
            logger.info(f"Stitching regions from {len(wells_to_query)} wells")
            
            # Calculate the output image dimensions at the requested scale
            scale_factor = 4 ** scale_level  # Each scale level is 4x smaller
            output_width_pixels = int(width_mm / (self.pixel_size_xy / 1000) / scale_factor)
            output_height_pixels = int(height_mm / (self.pixel_size_xy / 1000) / scale_factor)
            
            # Create output image
            output_image = np.zeros((output_height_pixels, output_width_pixels), dtype=np.uint8)
            
            # Process each well and place its data in the output image
            for well_info in well_regions:
                well_row = well_info['well_row']
                well_column = well_info['well_column']
                
                # Check if the well canvas exists
                if not self._check_well_canvas_exists(well_row, well_column, wellplate_type):
                    continue
                
                # Get well canvas and extract region
                canvas = self.experiment_manager.get_well_canvas(well_row, well_column, wellplate_type, well_padding_mm)
                
                # Calculate absolute coordinates for the intersection region
                intersection_center_x = (well_info['abs_min_x'] + well_info['abs_max_x']) / 2.0
                intersection_center_y = (well_info['abs_min_y'] + well_info['abs_max_y']) / 2.0
                intersection_width = well_info['abs_max_x'] - well_info['abs_min_x']
                intersection_height = well_info['abs_max_y'] - well_info['abs_min_y']
                
                well_region = canvas.get_canvas_region_by_channel_name(
                    intersection_center_x, intersection_center_y, 
                    intersection_width, intersection_height,
                    channel_name, scale=scale_level, timepoint=timepoint
                )
                
                if well_region is None:
                    logger.warning(f"Failed to get region from well {well_row}{well_column} - skipping")
                    continue
                
                # Calculate where to place this region in the output image
                # Convert absolute coordinates to output image coordinates
                rel_min_x = well_info['abs_min_x'] - region_min_x
                rel_min_y = well_info['abs_min_y'] - region_min_y
                
                # Convert to pixel coordinates
                start_x_px = int(rel_min_x / (self.pixel_size_xy / 1000) / scale_factor)
                start_y_px = int(rel_min_y / (self.pixel_size_xy / 1000) / scale_factor)
                
                # Ensure we don't go out of bounds
                start_x_px = max(0, min(start_x_px, output_width_pixels))
                start_y_px = max(0, min(start_y_px, output_height_pixels))
                
                end_x_px = min(start_x_px + well_region.shape[1], output_width_pixels)
                end_y_px = min(start_y_px + well_region.shape[0], output_height_pixels)
                
                # Crop the well region if needed to fit in output
                well_width = end_x_px - start_x_px
                well_height = end_y_px - start_y_px
                
                if well_width > 0 and well_height > 0:
                    cropped_well_region = well_region[:well_height, :well_width]
                    output_image[start_y_px:end_y_px, start_x_px:end_x_px] = cropped_well_region
                    
                    logger.info(f"Placed region from well {well_row}{well_column} at ({start_x_px}, {start_y_px}) "
                               f"with size ({well_width}, {well_height})")
            
            logger.info(f"Successfully stitched region from {len(well_regions)} wells, "
                       f"output shape: {output_image.shape}")
            
            return output_image
            
        except Exception as e:
            logger.error(f"Error getting stitched region: {e}")
            return None
    
    def initialize_experiment_if_needed(self, experiment_name: str = None):
        """
        Initialize an experiment if needed.
        This can be called early in the service lifecycle to ensure an experiment is ready.
        
        Args:
            experiment_name: Name of experiment to initialize (default: "default")
        """
        self.ensure_active_experiment(experiment_name)
        logger.info(f"Experiment '{self.experiment_manager.current_experiment}' is ready")

    # Experiment management methods
    def create_experiment(self, experiment_name: str, wellplate_type: str = '96', 
                         well_padding_mm: float = 1.0, initialize_all_wells: bool = False):
        """
        Create a new experiment.
        
        Args:
            experiment_name: Name of the experiment
            wellplate_type: Well plate type ('6', '12', '24', '96', '384')
            well_padding_mm: Padding around each well in mm
            initialize_all_wells: If True, create canvases for all wells in the plate
            
        Returns:
            dict: Information about the created experiment
        """
        return self.experiment_manager.create_experiment(
            experiment_name, wellplate_type, well_padding_mm, initialize_all_wells
        )
    
    def list_experiments(self):
        """List all available experiments."""
        return self.experiment_manager.list_experiments()
    
    def set_active_experiment(self, experiment_name: str):
        """Set the active experiment."""
        return self.experiment_manager.set_active_experiment(experiment_name)
    
    def remove_experiment(self, experiment_name: str):
        """Remove an experiment."""
        return self.experiment_manager.remove_experiment(experiment_name)
    
    def reset_experiment(self, experiment_name: str = None):
        """Reset an experiment by removing all well canvases."""
        return self.experiment_manager.reset_experiment(experiment_name)
    
    def get_experiment_info(self, experiment_name: str = None):
        """
        Get detailed information about an experiment.
        
        Args:
            experiment_name: Name of the experiment (default: current experiment)
            
        Returns:
            dict: Detailed experiment information
        """
        return self.experiment_manager.get_experiment_info(experiment_name)

    async def quick_scan_with_stitching(self, wellplate_type='96', exposure_time=5, intensity=50, 
                                      fps_target=10, action_ID='quick_scan_stitching',
                                      n_stripes=4, stripe_width_mm=4.0, dy_mm=0.9, velocity_scan_mm_per_s=7.0,
                                      do_contrast_autofocus=False, do_reflection_af=False, timepoint=0, 
                                      experiment_name=None, well_padding_mm=1.0):
        """
        Quick scan with live stitching to well-specific OME-Zarr canvases - brightfield only.
        Scans entire well plate, creating individual zarr canvases for each well.
        Uses 4-stripe × 4 mm scanning pattern with serpentine motion per well.
        
        Args:
            wellplate_type (str): Well plate type ('6', '12', '24', '96', '384')
            exposure_time (float): Camera exposure time in ms (max 30ms)
            intensity (float): Brightfield LED intensity (0-100)
            fps_target (int): Target frame rate for acquisition (default 10fps)
            action_ID (str): Identifier for this scan
            n_stripes (int): Number of stripes per well (default 4)
            stripe_width_mm (float): Length of each stripe inside a well in mm (default 4.0)
            dy_mm (float): Y increment between stripes in mm (default 0.9)
            velocity_scan_mm_per_s (float): Stage velocity during stripe scanning in mm/s (default 7.0)
            do_contrast_autofocus (bool): Whether to perform contrast-based autofocus
            do_reflection_af (bool): Whether to perform reflection-based autofocus
            timepoint (int): Timepoint index for the scan (default 0)
            experiment_name (str, optional): Name of the experiment to use. If None, uses active experiment or creates "default"
            well_padding_mm (float): Padding around each well in mm
        """
        
        # Validate exposure time
        if exposure_time > 30:
            raise ValueError("Quick scan exposure time must not exceed 30ms")
        
        # Get well plate format configuration
        if wellplate_type == '6':
            wellplate_format = WELLPLATE_FORMAT_6
            max_rows = 2  # A-B
            max_cols = 3  # 1-3
        elif wellplate_type == '12':
            wellplate_format = WELLPLATE_FORMAT_12
            max_rows = 3  # A-C
            max_cols = 4  # 1-4
        elif wellplate_type == '24':
            wellplate_format = WELLPLATE_FORMAT_24
            max_rows = 4  # A-D
            max_cols = 6  # 1-6
        elif wellplate_type == '96':
            wellplate_format = WELLPLATE_FORMAT_96
            max_rows = 8  # A-H
            max_cols = 12  # 1-12
        elif wellplate_type == '384':
            wellplate_format = WELLPLATE_FORMAT_384
            max_rows = 16  # A-P
            max_cols = 24  # 1-24
        else:
            # Default to 96-well plate if unsupported type is provided
            wellplate_format = WELLPLATE_FORMAT_96
            max_rows = 8
            max_cols = 12
            wellplate_type = '96'
        
        # Ensure we have an active experiment
        self.ensure_active_experiment(experiment_name)
        
        # Always use well-based approach - create well canvases dynamically as we encounter wells
        logger.info(f"Quick scan with stitching for experiment '{self.experiment_manager.current_experiment}': individual canvases for each well ({wellplate_type})")
        
        # Validate that brightfield channel is available (we'll check per well canvas)
        channel_name = 'BF LED matrix full'
        # Store original velocity settings for restoration
        original_velocity_result = self.set_stage_velocity()
        original_velocity_x = original_velocity_result.get('velocity_x_mm_per_s', CONFIG.MAX_VELOCITY_X_MM)
        original_velocity_y = original_velocity_result.get('velocity_y_mm_per_s', CONFIG.MAX_VELOCITY_Y_MM)
        
        # Define velocity constants
        HIGH_SPEED_VELOCITY_MM_PER_S = 30.0  # For moving between wells
        scan_velocity = velocity_scan_mm_per_s  # For scanning within wells
        
        try:
            self.is_busy = True
            self.scan_stop_requested = False  # Reset stop flag at start of scan
            logger.info(f'Starting quick scan with stitching: {wellplate_type} well plate, {n_stripes} stripes × {stripe_width_mm}mm, dy={dy_mm}mm, scan_velocity={scan_velocity}mm/s, fps={fps_target}, timepoint={timepoint}')
            
            if do_contrast_autofocus:
                logger.info('Contrast autofocus enabled for quick scan')
            if do_reflection_af:
                logger.info('Reflection autofocus enabled for quick scan')
            
            # 1. Before starting scanning, read the position of z axis
            original_x_mm, original_y_mm, original_z_mm, _ = self.navigationController.update_pos(self.microcontroller)
            logger.info(f'Original Z position before autofocus: {original_z_mm:.3f}mm')
            
            # Set camera exposure time
            self.camera.set_exposure_time(exposure_time)
            
            # Calculate well plate parameters
            well_spacing = wellplate_format.WELL_SPACING_MM
            x_offset = CONFIG.WELLPLATE_OFFSET_X_MM
            y_offset = CONFIG.WELLPLATE_OFFSET_Y_MM
            
            # Calculate frame acquisition timing
            frame_interval = 1.0 / fps_target  # seconds between frames
            
            # Get software limits for safety
            limit_x_pos = CONFIG.SOFTWARE_POS_LIMIT.X_POSITIVE
            limit_x_neg = CONFIG.SOFTWARE_POS_LIMIT.X_NEGATIVE
            limit_y_pos = CONFIG.SOFTWARE_POS_LIMIT.Y_POSITIVE
            limit_y_neg = CONFIG.SOFTWARE_POS_LIMIT.Y_NEGATIVE
            
            # Scan each well using snake pattern for rows
            for row_idx in range(max_rows):
                if self.scan_stop_requested:
                    logger.info("Quick scan stopped by user request")
                    self._restore_original_velocity(CONFIG.MAX_VELOCITY_X_MM, CONFIG.MAX_VELOCITY_Y_MM)
                    break
                    
                row_letter = chr(ord('A') + row_idx)
                
                # Snake pattern: alternate direction for each row
                if row_idx % 2 == 0:
                    # Even rows (0, 2, 4...): left to right (A1 → A12, C1 → C12, etc.)
                    col_range = range(max_cols)
                    direction = "left-to-right"
                else:
                    # Odd rows (1, 3, 5...): right to left (B12 → B1, D12 → D1, etc.)
                    col_range = range(max_cols - 1, -1, -1)
                    direction = "right-to-left"
                
                logger.info(f'Scanning row {row_letter} ({direction})')
                
                for col_idx in col_range:
                    if self.scan_stop_requested:
                        logger.info("Quick scan stopped by user request")
                        self._restore_original_velocity(CONFIG.MAX_VELOCITY_X_MM, CONFIG.MAX_VELOCITY_Y_MM)
                        break
                        
                    col_number = col_idx + 1
                    well_name = f"{row_letter}{col_number}"
                    
                    # Calculate well center position
                    well_center_x = wellplate_format.A1_X_MM + x_offset + col_idx * well_spacing
                    well_center_y = wellplate_format.A1_Y_MM + y_offset + row_idx * well_spacing
                    
                    # Calculate stripe boundaries within the well
                    stripe_half_width = stripe_width_mm / 2
                    stripe_start_x = well_center_x - stripe_half_width
                    stripe_end_x = well_center_x + stripe_half_width
                    
                    # Clamp stripe boundaries to software limits
                    stripe_start_x = max(min(stripe_start_x, limit_x_pos), limit_x_neg)
                    stripe_end_x = max(min(stripe_end_x, limit_x_pos), limit_x_neg)
                    
                    # Calculate starting Y position for stripes (centered around well)
                    stripe_start_y = well_center_y - ((n_stripes - 1) * dy_mm) / 2
                    
                    logger.info(f'Scanning well {well_name}: {n_stripes} stripes × {stripe_width_mm}mm at Y positions starting from {stripe_start_y:.2f}mm')
                    
                    # Autofocus workflow: move to well center first if autofocus is requested
                    if do_contrast_autofocus or do_reflection_af:
                        logger.info(f'Moving to well {well_name} center for autofocus')
                        
                        # Set high speed velocity for moving to well center
                        velocity_result = self.set_stage_velocity(HIGH_SPEED_VELOCITY_MM_PER_S, HIGH_SPEED_VELOCITY_MM_PER_S)
                        if not velocity_result['success']:
                            logger.warning(f"Failed to set high-speed velocity for autofocus: {velocity_result['message']}")
                        
                        # Move to well center using move_to_well function
                        self.move_to_well(row_letter, col_number, wellplate_type)
                        
                        # Wait for movement to complete
                        while self.microcontroller.is_busy():
                            await asyncio.sleep(0.005)
                        
                        # Perform autofocus
                        if do_reflection_af:
                            logger.info(f'Performing reflection autofocus at well {well_name}')
                            if hasattr(self, 'laserAutofocusController'):
                                await self.do_laser_autofocus()
                            else:
                                logger.warning('Reflection autofocus requested but laserAutofocusController not available')
                        elif do_contrast_autofocus:
                            logger.info(f'Performing contrast autofocus at well {well_name}')
                            await self.do_autofocus()
                        
                        # Update position after autofocus
                        actual_x_mm, actual_y_mm, actual_z_mm, _ = self.navigationController.update_pos(self.microcontroller)
                        logger.info(f'Autofocus completed at well {well_name}, current position: ({actual_x_mm:.2f}, {actual_y_mm:.2f}, {actual_z_mm:.2f})')
                    
                    # Get well canvas for this well and validate brightfield channel
                    canvas = self.experiment_manager.get_well_canvas(row_letter, col_number, wellplate_type, well_padding_mm)
                    
                    # Validate that brightfield channel is available in this canvas
                    if channel_name not in canvas.channel_to_zarr_index:
                        logger.error(f"Requested channel '{channel_name}' not found in well canvas!")
                        logger.error(f"Available channels: {list(canvas.channel_to_zarr_index.keys())}")
                        raise ValueError(f"Channel '{channel_name}' not available in well canvas")
                    
                    # Get local zarr channel index for brightfield
                    try:
                        zarr_channel_idx = canvas.get_zarr_channel_index(channel_name)
                    except ValueError as e:
                        logger.error(f"Channel mapping error: {e}")
                        continue
                    
                    # Start stitching for this well
                    logger.info(f'QUICK_SCAN: Starting stitching for well {well_name}')
                    await canvas.start_stitching()
                    logger.info(f'QUICK_SCAN: Stitching started for well {well_name}, is_stitching={canvas.is_stitching}')
                    
                    # Move to well stripe start position at high speed
                    await self._move_to_well_at_high_speed(well_name, stripe_start_x, stripe_start_y, 
                                                            HIGH_SPEED_VELOCITY_MM_PER_S, limit_y_neg, limit_y_pos)
                    
                    # Set scan velocity for stripe scanning
                    velocity_result = self.set_stage_velocity(scan_velocity, scan_velocity)
                    if not velocity_result['success']:
                        logger.warning(f"Failed to set scanning velocity: {velocity_result['message']}")
                    
                    # Scan all stripes within the well with continuous frame acquisition
                    total_frames = await self._scan_well_with_continuous_acquisition(
                        well_name, n_stripes, stripe_start_x, stripe_end_x, 
                        stripe_start_y, dy_mm, intensity, frame_interval, 
                        zarr_channel_idx, limit_y_neg, limit_y_pos, timepoint=timepoint,
                        wellplate_type=wellplate_type, well_padding_mm=well_padding_mm, channel_name=channel_name)
                    
                    logger.info(f'Well {well_name} completed with {n_stripes} stripes, total frames: {total_frames}')
                    
                    # Debug stitching status after completing well
                    self.debug_stitching_status()
                    
                    # 3. After scanning for this well is done, move the z axis back to the remembered position
                    if do_contrast_autofocus or do_reflection_af:
                        logger.info(f'Restoring Z position to original: {original_z_mm:.3f}mm')
                        self.navigationController.move_z_to(original_z_mm)
                        while self.microcontroller.is_busy():
                            await asyncio.sleep(0.005)
            
            logger.info('Quick scan with stitching completed')
            
            # Allow time for final images to be queued for stitching
            logger.info('Allowing time for final images to be queued for stitching...')
            await asyncio.sleep(0.5)
            
        finally:
            self.is_busy = False
            
            # Turn off illumination if still on
            self.liveController.turn_off_illumination()
            
            # Restore original velocity settings
            self._restore_original_velocity(original_velocity_x, original_velocity_y)
            
            # Debug stitching status before stopping
            logger.info('QUICK_SCAN: Final stitching status before stopping:')
            self.debug_stitching_status()
            
            # Stop stitching for all active well canvases in the experiment
            for well_id, well_canvas in self.experiment_manager.well_canvases.items():
                if well_canvas.is_stitching:
                    logger.info(f'QUICK_SCAN: Stopping stitching for well canvas {well_id}, queue_size={well_canvas.stitch_queue.qsize()}')
                    await well_canvas.stop_stitching()
                    logger.info(f'QUICK_SCAN: Stopped stitching for well canvas {well_id}')
            
            # Final stitching status after stopping
            logger.info('QUICK_SCAN: Final stitching status after stopping:')
            self.debug_stitching_status()
            
            # CRITICAL: Additional delay after stitching stops to ensure all zarr operations are complete
            # This prevents race conditions with ZIP export when scanning finishes normally
            logger.info('Waiting additional time for all zarr operations to stabilize...')
            await asyncio.sleep(0.5)  # 500ms buffer to ensure filesystem operations complete
            logger.info('Quick scan with stitching fully completed - zarr data ready for export')
    
    async def _move_to_well_at_high_speed(self, well_name, start_x, start_y, high_speed_velocity, limit_y_neg, limit_y_pos):
        """Move to well at high speed (30 mm/s) for efficient inter-well movement."""
        logger.info(f'Moving to well {well_name} at high speed ({high_speed_velocity} mm/s)')
        
        velocity_result = self.set_stage_velocity(high_speed_velocity, high_speed_velocity)
        if not velocity_result['success']:
            logger.warning(f"Failed to set high-speed velocity: {velocity_result['message']}")
        
        # Clamp Y position to limits
        clamped_y = max(min(start_y, limit_y_pos), limit_y_neg)
        
        # Move to first stripe start position
        self.navigationController.move_x_to(start_x)
        self.navigationController.move_y_to(clamped_y)
        
        # Wait for movement to complete
        while self.microcontroller.is_busy():
            await asyncio.sleep(0.005)
        
        logger.info(f'Moved to well {well_name} start position ({start_x:.2f}, {clamped_y:.2f})')
    
    async def _scan_well_with_continuous_acquisition(self, well_name, n_stripes, stripe_start_x, stripe_end_x, 
                                                   stripe_start_y, dy_mm, intensity, frame_interval, 
                                                   zarr_channel_idx, limit_y_neg, limit_y_pos, timepoint=0,
                                                   wellplate_type='96', well_padding_mm=1.0, channel_name='BF LED matrix full'):
        """Scan all stripes within a well with continuous frame acquisition."""
        total_frames = 0
        
        # Turn on brightfield illumination once for the entire well
        self.liveController.set_illumination(0, intensity)  # Channel 0 = brightfield
        await asyncio.sleep(0.01)  # Small delay for illumination to stabilize
        self.liveController.turn_on_illumination()
        
        # Start continuous frame acquisition
        last_frame_time = time.time()
        
        try:
            for stripe_idx in range(n_stripes):
                if self.scan_stop_requested:
                    logger.info("Quick scan stopped by user request")
                    self._restore_original_velocity(CONFIG.MAX_VELOCITY_X_MM, CONFIG.MAX_VELOCITY_Y_MM)
                    break
                    
                stripe_y = stripe_start_y + stripe_idx * dy_mm
                stripe_y = max(min(stripe_y, limit_y_pos), limit_y_neg)
                
                # Serpentine pattern: alternate direction for each stripe
                if stripe_idx % 2 == 0:
                    # Even stripes: left to right
                    start_x, end_x = stripe_start_x, stripe_end_x
                    direction = "left-to-right"
                else:
                    # Odd stripes: right to left
                    start_x, end_x = stripe_end_x, stripe_start_x
                    direction = "right-to-left"
                
                logger.info(f'Well {well_name}, stripe {stripe_idx + 1}/{n_stripes} ({direction}) from X={start_x:.2f}mm to X={end_x:.2f}mm at Y={stripe_y:.2f}mm')
                
                # Move to stripe start position
                self.navigationController.move_x_to(start_x)
                self.navigationController.move_y_to(stripe_y)
                
                # Wait for positioning to complete
                while self.microcontroller.is_busy():
                    await asyncio.sleep(0.005)
                
                # Let stage settle briefly
                await asyncio.sleep(0.05)
                
                # Start continuous movement to end of stripe
                self.navigationController.move_x_to(end_x)
                
                # Acquire frames while moving along this stripe
                stripe_frames = 0
                while self.microcontroller.is_busy():
                    if self.scan_stop_requested:
                        logger.info("Quick scan stopped during stripe movement")
                        self._restore_original_velocity(CONFIG.MAX_VELOCITY_X_MM, CONFIG.MAX_VELOCITY_Y_MM)
                        break
                        
                    current_time = time.time()
                    
                    # Check if it's time for next frame
                    if current_time - last_frame_time >= frame_interval:
                        frame_acquired = await self._acquire_and_process_frame(
                            zarr_channel_idx, timepoint, wellplate_type, well_padding_mm, channel_name
                        )
                        if frame_acquired:
                            stripe_frames += 1
                            total_frames += 1
                        # Update timing AFTER frame acquisition completes, not before
                        last_frame_time = time.time()
                    
                    # Small delay to prevent overwhelming the system
                    await asyncio.sleep(0.001)
                
                logger.info(f'Well {well_name}, stripe {stripe_idx + 1}/{n_stripes} completed, acquired {stripe_frames} frames')
                
                # Continue to next stripe without stopping illumination or frame acquisition
                
        finally:
            # Turn off illumination only when done with the entire well
            self.liveController.turn_off_illumination()
        
        return total_frames
    
    async def _acquire_and_process_frame(self, zarr_channel_idx, timepoint=0, 
                                       wellplate_type='96', well_padding_mm=1.0, channel_name='BF LED matrix full'):
        """Acquire a single frame and add it to the stitching queue for quick scan."""
        # Get position before frame acquisition
        pos_before_x_mm, pos_before_y_mm, pos_before_z_mm, _ = self.navigationController.update_pos(self.microcontroller)
        
        # Read frame from camera
        self.camera.send_trigger()
        gray_img = self.camera.read_frame()
        
        # Get position after frame acquisition
        pos_after_x_mm, pos_after_y_mm, pos_after_z_mm, _ = self.navigationController.update_pos(self.microcontroller)
        
        # Calculate average position during frame acquisition
        avg_x_mm = (pos_before_x_mm + pos_after_x_mm) / 2.0
        avg_y_mm = (pos_before_y_mm + pos_after_y_mm) / 2.0
        
        logger.info(f'FRAME_ACQ: Position before=({pos_before_x_mm:.2f}, {pos_before_y_mm:.2f}), after=({pos_after_x_mm:.2f}, {pos_after_y_mm:.2f}), avg=({avg_x_mm:.2f}, {avg_y_mm:.2f})')
        
        if gray_img is not None:
            logger.info(f'FRAME_ACQ: Camera frame acquired successfully, shape={gray_img.shape}, dtype={gray_img.dtype}')
            
            # Process and add image to stitching queue using quick scan method
            processed_img = self._process_frame_for_stitching(gray_img)
            logger.info(f'FRAME_ACQ: Image processed for stitching, new shape={processed_img.shape}, dtype={processed_img.dtype}')
            
            # Add to stitching queue for quick scan (using well-based approach)
            result = await self._add_image_to_zarr_quick_well_based(
                processed_img, avg_x_mm, avg_y_mm, 
                zarr_channel_idx, timepoint, wellplate_type, well_padding_mm, channel_name
            )
            
            logger.info(f'FRAME_ACQ: Frame processing completed at position ({avg_x_mm:.2f}, {avg_y_mm:.2f}), timepoint={timepoint}, result={result}')
            return True
        else:
            logger.warning(f'FRAME_ACQ: Camera frame is None at position ({avg_x_mm:.2f}, {avg_y_mm:.2f})')
        
        return False
    
    def _process_frame_for_stitching(self, gray_img):
        """Process a frame for stitching (resize, rotate, flip, convert to 8-bit)."""
        # Immediately rescale to scale1 resolution (1/4 of original)
        original_height, original_width = gray_img.shape[:2]
        scale1_width = original_width // 4
        scale1_height = original_height // 4
        
        # Resize image to scale1 resolution
        scaled_img = cv2.resize(gray_img, (scale1_width, scale1_height), interpolation=cv2.INTER_AREA)
        
        # Apply rotate and flip transformations
        processed_img = rotate_and_flip_image(
            scaled_img,
            rotate_image_angle=self.camera.rotate_image_angle,
            flip_image=self.camera.flip_image
        )
        
        # Convert to 8-bit if needed
        if processed_img.dtype != np.uint8:
            if processed_img.dtype == np.uint16:
                processed_img = (processed_img / 256).astype(np.uint8)
            else:
                processed_img = processed_img.astype(np.uint8)
        
        return processed_img
    
    def _restore_original_velocity(self, original_velocity_x, original_velocity_y):
        """Restore the original stage velocity settings."""
        restore_result = self.set_stage_velocity(original_velocity_x, original_velocity_y)
        if restore_result['success']:
            logger.info(f'Restored original stage velocity: X={original_velocity_x}mm/s, Y={original_velocity_y}mm/s')
        else:
            logger.warning(f'Failed to restore original stage velocity: {restore_result["message"]}')
    
    def stop_scan_and_stitching(self):
        """
        Stop any ongoing scanning and stitching processes.
        This will interrupt normal_scan_with_stitching and quick_scan_with_stitching.
        """
        self.scan_stop_requested = True
        logger.info("Scan stop requested - ongoing scans will be interrupted")
        self._restore_original_velocity(CONFIG.MAX_VELOCITY_X_MM, CONFIG.MAX_VELOCITY_Y_MM)
        return {"success": True, "message": "Scan stop requested"}

async def try_microscope():
    squid_controller = SquidController(is_simulation=False)

    custom_illumination_settings = [
        {'channel': 'BF LED matrix full', 'intensity': 35.0, 'exposure_time': 15.0},
        {'channel': 'Fluorescence 488 nm Ex', 'intensity': 50.0, 'exposure_time': 80.0},
        {'channel': 'Fluorescence 561 nm Ex', 'intensity': 75.0, 'exposure_time': 120.0}
    ]
    
    squid_controller.scan_well_plate_new(
        well_plate_type='96',
        illumination_settings=custom_illumination_settings,
        do_contrast_autofocus=False,
        do_reflection_af=True,
        scanning_zone=[(0,0),(1,1)],  # Scan wells A1 to B2
        Nx=2,
        Ny=2,
        action_ID='customIlluminationTest'
    )
    
    squid_controller.close()


if __name__ == "__main__":
    asyncio.run(try_microscope())

