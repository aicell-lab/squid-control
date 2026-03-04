"""Plate scanning and well navigation mixin for SquidController."""
import logging
import time

from squid_control.hardware.config import CONFIG

logger = logging.getLogger('squid_controller')


class ScanningMixin:
    """Mixin providing plate scanning methods for SquidController."""

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
                logger.error('z return timeout, the program will exit')
                exit()

    def plate_scan(self, well_plate_type='96', illumination_settings=None, do_contrast_autofocus=False, do_reflection_af=True, wells_to_scan=['A1'], Nx=3, Ny=3, dx=0.8, dy=0.8, action_ID='testPlateScanNew'):
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
            wells_to_scan (list): List of wells to scan (e.g., ['A1', 'B2', 'C3'])
            Nx (int): Number of X positions per well
            Ny (int): Number of Y positions per well
            dx (float): Distance between X positions in mm (default: 0.8)
            dy (float): Distance between Y positions in mm (default: 0.8)
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

        # Set up scan coordinates using wells_to_scan
        self.scanCoordinates.well_selector.set_selected_wells_from_list(wells_to_scan)
        self.scanCoordinates.get_selected_wells_to_coordinates(well_plate_type=well_plate_type, is_simulation=self.is_simulation)

        # Configure multipoint controller
        self.multipointController.set_base_path(CONFIG.DEFAULT_SAVING_PATH)
        self.multipointController.contrast_autofocus = do_contrast_autofocus
        self.multipointController.do_reflection_af = do_reflection_af
        self.multipointController.set_NX(Nx)
        self.multipointController.set_NY(Ny)
        self.multipointController.set_deltaX(dx)
        self.multipointController.set_deltaY(dy)
        self.multipointController.start_new_experiment(action_ID)

        # Clear location_list to ensure we use well plate coordinates from scanCoordinates
        # This prevents using stale coordinates from previous flexible position scans
        self.multipointController.location_list = None

        # Start scanning
        self.is_busy = True
        logger.info('Starting new plate scan with custom illumination settings')
        self.multipointController.run_acquisition()
        logger.info('New plate scan completed')
        self.is_busy = False

    def stop_plate_scan(self):
        self.multipointController.abort_acqusition_requested = True
        self.is_busy = False
        logger.info('Plate scan stopped')

    def stop_scan_plate_save_raw_images_new(self):
        """Stop the plate scan that saves raw images - alias for stop_plate_scan"""
        self.stop_plate_scan()

    def flexible_position_scan(self, positions=None, illumination_settings=None, do_contrast_autofocus=False, do_reflection_af=True, action_ID='flexibleScan', move_for_autofocus=False):
        """
        Flexible position scanning function that allows arbitrary positions with individual grid parameters.
        No well plate constraints - user specifies exact positions and grid parameters.

        Args:
            positions (list): List of position dictionaries, each containing:
                {
                    'x': 10.0,        # X position in mm (absolute stage coordinate)
                    'y': 20.0,        # Y position in mm (absolute stage coordinate)
                    'z': 5.0,         # Z position in mm (absolute stage coordinate, optional)
                    'Nx': 3,          # Number of X grid points for this position (default: 1)
                    'Ny': 3,          # Number of Y grid points for this position (default: 1)
                    'Nz': 1,          # Number of Z grid points for this position (default: 1)
                    'dx': 0.8,        # X spacing in mm for this position (default: 0.8)
                    'dy': 0.8,        # Y spacing in mm for this position (default: 0.8)
                    'dz': 0.01,       # Z spacing in mm for this position (default: 0.01)
                    'name': 'pos1'    # Optional name for this position (default: 'position_N')
                }
            illumination_settings (list): List of dictionaries with illumination settings
                Each dict should contain:
                {
                    'channel': 'BF LED matrix full',  # Channel name
                    'intensity': 50.0,               # Illumination intensity (0-100)
                    'exposure_time': 25.0            # Exposure time in ms
                }
            do_contrast_autofocus (bool): Whether to perform contrast-based autofocus
            do_reflection_af (bool): Whether to perform reflection-based autofocus
            action_ID (str): Identifier for this scan
            move_for_autofocus (bool): If True, move 0.2mm in X and Y before reflection autofocus, then move back.
                If False (default), perform reflection autofocus at current position only.
        """
        if positions is None or len(positions) == 0:
            logger.error("No positions provided for flexible scan")
            raise ValueError("positions list cannot be empty")

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

        # Build expanded location list - expand each position's grid into individual scan coordinates
        # This approach works with the existing multipoint controller architecture
        location_list = []
        location_names = []

        for idx, pos in enumerate(positions):
            # Extract position coordinates
            x_center = pos.get('x')
            y_center = pos.get('y')
            z_center = pos.get('z', None)  # Z is optional

            if x_center is None or y_center is None:
                logger.error(f"Position {idx} missing required x or y coordinate")
                raise ValueError(f"Position {idx} must have 'x' and 'y' coordinates")

            # Extract grid parameters with defaults
            Nx = pos.get('Nx', 1)
            Ny = pos.get('Ny', 1)
            Nz = pos.get('Nz', 1)
            dx = pos.get('dx', 0.8)
            dy = pos.get('dy', 0.8)
            dz = pos.get('dz', 0.01)
            name = pos.get('name', f'position_{idx+1}')

            logger.info(f"Processing position '{name}': center=({x_center},{y_center},{z_center}), grid={Nx}x{Ny}x{Nz}, spacing=({dx},{dy},{dz})")

            # Expand this position into a grid of individual coordinates
            # Grid is centered on the specified position
            for k in range(Nz):
                for i in range(Ny):
                    for j in range(Nx):
                        # Calculate offset from center for this grid point
                        x_offset = (j - (Nx - 1) / 2.0) * dx
                        y_offset = (i - (Ny - 1) / 2.0) * dy
                        z_offset = (k - (Nz - 1) / 2.0) * dz if z_center is not None else 0

                        # Calculate absolute position
                        x_abs = x_center + x_offset
                        y_abs = y_center + y_offset

                        # Create coordinate tuple
                        if z_center is not None:
                            z_abs = z_center + z_offset
                            coord = (x_abs, y_abs, z_abs)
                        else:
                            coord = (x_abs, y_abs)

                        location_list.append(coord)

                        # Create descriptive name for this grid point
                        if Nx > 1 or Ny > 1 or Nz > 1:
                            point_name = f"{name}_y{i}_x{j}" if Nz == 1 else f"{name}_z{k}_y{i}_x{j}"
                        else:
                            point_name = name
                        location_names.append(point_name)

            logger.info(f"Expanded position '{name}' into {Nx*Ny*Nz} scan points")

        logger.info(f"Total scan points: {len(location_list)}")

        # Configure multipoint controller
        self.multipointController.set_base_path(CONFIG.DEFAULT_SAVING_PATH)
        self.multipointController.contrast_autofocus = do_contrast_autofocus
        self.multipointController.do_reflection_af = do_reflection_af
        self.multipointController.move_for_autofocus = move_for_autofocus

        # Set NX=1, NY=1, NZ=1 since we've already expanded the grid
        self.multipointController.set_NX(1)
        self.multipointController.set_NY(1)
        self.multipointController.set_NZ(1)

        self.multipointController.start_new_experiment(action_ID)

        # Start scanning with expanded location_list
        self.is_busy = True
        logger.info(f'Starting flexible position scan with {len(location_list)} total scan points')

        # Store location names in scanCoordinates for proper naming
        if self.scanCoordinates is not None:
            self.scanCoordinates.coordinates_mm = location_list
            self.scanCoordinates.name = location_names

        # Pass location_list to run_acquisition
        self.multipointController.run_acquisition(location_list=location_list)

        logger.info('Flexible position scan completed')
        self.is_busy = False

    def stop_flexible_position_scan(self):
        """Stop the flexible position scan"""
        self.multipointController.abort_acqusition_requested = True
        self.is_busy = False
        logger.info('Flexible position scan stopped')
