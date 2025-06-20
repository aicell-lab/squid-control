import os 
# app specific libraries
import squid_control.control.core_reef as core
import squid_control.control.microcontroller as microcontroller
from squid_control.control.config import *
from squid_control.control.camera import get_camera
from squid_control.control.utils import rotate_and_flip_image
import cv2
import logging
import matplotlib.path as mpath
# Import serial_peripherals conditionally based on simulation mode
import sys
import numpy as np

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


class SquidController:
    fps_software_trigger= 10

    def __init__(self,is_simulation, *args, **kwargs):
        super().__init__(*args,**kwargs)
        self.data_channel = None
        #load objects
        self.objectiveStore = core.ObjectiveStore()
        camera, camera_fc = get_camera(CONFIG.CAMERA_TYPE)
        self.is_simulation = is_simulation
        self.is_busy = False
        if is_simulation:
            config_path = os.path.join(os.path.dirname(path), 'configuration_HCS_v2_example.ini')
        else:
            config_path = os.path.join(os.path.dirname(path), 'configuration_HCS_v2.ini')

        print(f"Loading configuration from: {config_path}")
        load_config(config_path, False)
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
        self.pixel_size_adjument_factor = 0.936
        # drift correction for image map
        self.drift_correction_x = -1.6
        self.drift_correction_y = -2.1
        # simulated sample data alias
        self.sample_data_alias = "agent-lens/20250506-scan-time-lapse-2025-05-06_17-56-38"
        self.get_pixel_size()


    def get_pixel_size(self):
        """Calculate pixel size based on imaging parameters."""
        try:
            tube_lens_mm = float(CONFIG.TUBE_LENS_MM)
            pixel_size_um = float(CONFIG.CAMERA_PIXEL_SIZE_UM[CONFIG.CAMERA_SENSOR])
            
            object_dict_key = self.objectiveStore.current_objective
            objective = self.objectiveStore.objectives_dict[object_dict_key]
            magnification =  float(objective['magnification'])
            objective_tube_lens_mm = float(objective['tube_lens_f_mm'])
            print(f"Tube lens: {tube_lens_mm} mm, Objective tube lens: {objective_tube_lens_mm} mm, Pixel size: {pixel_size_um} µm, Magnification: {magnification}")
        except:
            raise ValueError("Missing required parameters for pixel size calculation.")

        self.pixel_size_xy = pixel_size_um / (magnification / (objective_tube_lens_mm / tube_lens_mm))
        self.pixel_size_xy = self.pixel_size_xy * self.pixel_size_adjument_factor
        print(f"Pixel size: {self.pixel_size_xy} µm")
        
                
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
            # Default settings if none provided
            illumination_settings = [
                {'channel': 'BF LED matrix full', 'intensity': 28.0, 'exposure_time': 20.0},
                {'channel': 'Fluorescence 488 nm Ex', 'intensity': 27.0, 'exposure_time': 60.0},
                {'channel': 'Fluorescence 561 nm Ex', 'intensity': 98.0, 'exposure_time': 100.0}
            ]
        
        # Update configurations with custom settings
        self.multipointController.set_selected_configurations_with_settings(illumination_settings)
        
        # Move to scanning position
        self.move_to_scaning_position()
        
        # Set up scan coordinates
        self.scanCoordinates.well_selector.set_selected_wells(scanning_zone[0], scanning_zone[1])
        self.scanCoordinates.get_selected_wells_to_coordinates()
        
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
            while self.microcontroller.is_busy():
                time.sleep(0.005)
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

    async def snap_image(self, channel=0, intensity=100, exposure_time=100):
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
        gray_img = rotate_and_flip_image(gray_img, self.camera.rotate_image_angle, self.camera.flip_image)
        
        if not need_to_turn_illumination_back:
            self.liveController.turn_off_illumination()
            while self.microcontroller.is_busy():
                await asyncio.sleep(0.005)

        return gray_img
    
    async def get_camera_frame_simulation(self, channel=0, intensity=100, exposure_time=100):
        self.camera.set_exposure_time(exposure_time)
        self.liveController.set_illumination(channel, intensity)
        await self.send_trigger_simulation(channel, intensity, exposure_time)
        gray_img = self.camera.read_frame() 
        gray_img = rotate_and_flip_image(gray_img, self.camera.rotate_image_angle, self.camera.flip_image)
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
        if not self.is_simulation:
            # move the objective to a defined position upon exit
            self.navigationController.move_x(0.1) # temporary bug fix - move_x needs to be called before move_x_to if the stage has been moved by the joystick
            while self.microcontroller.is_busy():
                time.sleep(0.005)
            self.navigationController.move_x_to(28)
            while self.microcontroller.is_busy():
                time.sleep(0.005)
            self.navigationController.move_y(0.1) # temporary bug fix - move_y needs to be called before move_y_to if the stage has been moved by the joystick
            while self.microcontroller.is_busy():
                time.sleep(0.005)
            self.navigationController.move_y_to(19)
            while self.microcontroller.is_busy():
                time.sleep(0.005)

        self.liveController.stop_live()
        self.camera.close()  # This now includes proper Hypha cleanup
        #self.imageSaver.close()
        if CONFIG.SUPPORT_LASER_AUTOFOCUS:
            self.camera_focus.close()
            #self.imageDisplayWindow_focus.close()
        self.microcontroller.close()
        
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
