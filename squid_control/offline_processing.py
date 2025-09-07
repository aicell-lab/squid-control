"""
Offline processing module for time-lapse experiment data.
Handles stitching and uploading of stored microscopy data.
"""

import asyncio
import io
import json
import logging
import os
import shutil
import tempfile
import time
import xml.etree.ElementTree as ET
import zipfile
from pathlib import Path
from typing import List

import cv2
import pandas as pd

# Use print statements for debugging since logger configuration seems problematic
logger = logging.getLogger(__name__)


class OfflineProcessor:
    """Handles offline stitching and uploading of time-lapse data."""

    def __init__(self, squid_controller, zarr_artifact_manager=None, service_id=None):
        print("🔧 OfflineProcessor.__init__ called")
        self.squid_controller = squid_controller
        self.zarr_artifact_manager = zarr_artifact_manager
        self.service_id = service_id
        self.logger = logger
        
        # Ensure configuration is loaded
        from squid_control.control.config import CONFIG
        self._ensure_config_loaded()

    def _ensure_config_loaded(self):
        """Ensure the configuration is properly loaded."""
        from squid_control.control.config import CONFIG, load_config
        import os
        
        # Check if DEFAULT_SAVING_PATH is already loaded
        if not CONFIG.DEFAULT_SAVING_PATH:
            print("Configuration not loaded, attempting to load HCS_v2 config...")
            try:
                # Try to load the config file
                current_dir = Path(__file__).parent
                config_path = current_dir / "config" / "configuration_HCS_v2.ini"
                if config_path.exists():
                    load_config(str(config_path), None)
                    print(f"Configuration loaded: DEFAULT_SAVING_PATH = {CONFIG.DEFAULT_SAVING_PATH}")
                else:
                    print(f"Config file not found at {config_path}")
            except Exception as e:
                print(f"Failed to load configuration: {e}")
        else:
            print(f"Configuration already loaded: DEFAULT_SAVING_PATH = {CONFIG.DEFAULT_SAVING_PATH}")

    def find_experiment_folders(self, experiment_id: str) -> List[Path]:
        """
        Find all experiment folders matching experiment_id prefix.
        
        Args:
            experiment_id: Experiment ID to search for (e.g., 'test-drug')
            
        Returns:
            Sorted list of Path objects like:
            [experiment_id-20250822T143055, experiment_id-20250822T163022, ...]
        """

        from squid_control.control.config import CONFIG
        print(f"CONFIG.DEFAULT_SAVING_PATH = {CONFIG.DEFAULT_SAVING_PATH}")
        base_path = Path(CONFIG.DEFAULT_SAVING_PATH)
        print(f"Searching in base path: {base_path}")
        if not base_path.exists():
            raise FileNotFoundError(f"Base path does not exist: {base_path}")

        pattern = f"{experiment_id}-*"
        print(f"Using pattern: {pattern}")
        folders = sorted(base_path.glob(pattern))
        print(f"Found {len(folders)} folders matching pattern: {[f.name for f in folders]}")

        # Filter to only directories that contain a '0' subfolder
        valid_folders = []
        for folder in folders:
            print(f"Checking folder: {folder.name}")
            if folder.is_dir():
                zero_folder = folder / "0"
                print(f"  Looking for '0' subfolder: {zero_folder}")
                if zero_folder.exists() and zero_folder.is_dir():
                    valid_folders.append(folder)
                    print(f"  ✓ Valid folder: {folder.name}")
                else:
                    print(f"  ✗ Skipping {folder.name}: no '0' subfolder found")
            else:
                print(f"  ✗ Skipping {folder.name}: not a directory")

        print(f"Found {len(valid_folders)} valid experiment folders for '{experiment_id}': {[f.name for f in valid_folders]}")
        return valid_folders

    def parse_acquisition_parameters(self, experiment_folder: Path) -> dict:
        """
        Parse acquisition parameters.json from the experiment folder.
        
        Args:
            experiment_folder: Path to experiment folder (e.g., test-drug-20250822T143055)
            
        Returns:
            Dictionary with acquisition parameters
        """
        # Try the main experiment folder first
        json_file = experiment_folder / "acquisition parameters.json"
        if not json_file.exists():
            # Fallback to the '0' subfolder
            json_file = experiment_folder / "0" / "acquisition parameters.json"
            if not json_file.exists():
                raise FileNotFoundError(f"No acquisition parameters.json found in {experiment_folder}/ or {experiment_folder}/0/")

        with open(json_file) as f:
            params = json.load(f)

        self.logger.info(f"Loaded acquisition parameters from {json_file}: Nx={params.get('Nx')}, Ny={params.get('Ny')}, dx={params.get('dx(mm)')}, dy={params.get('dy(mm)')}")
        return params

    def parse_configurations_xml(self, experiment_folder: Path) -> dict:
        """
        Parse configurations.xml and extract channel settings.
        
        Args:
            experiment_folder: Path to experiment folder
            
        Returns:
            Dictionary with channel configurations
        """
        # Try the main experiment folder first
        xml_file = experiment_folder / "configurations.xml"
        if not xml_file.exists():
            # Fallback to the '0' subfolder
            xml_file = experiment_folder / "0" / "configurations.xml"
            if not xml_file.exists():
                raise FileNotFoundError(f"No configurations.xml found in {experiment_folder}/ or {experiment_folder}/0/")

        return self._parse_xml_channels(xml_file)

    def _parse_xml_channels(self, xml_file: Path) -> dict:
        """Extract channel information from XML configuration file."""
        tree = ET.parse(xml_file)
        root = tree.getroot()

        channels = {}
        for mode in root.findall('mode'):
            if mode.get('Selected') == '1':  # Only selected channels
                channel_name = mode.get('Name')
                channels[channel_name] = {
                    'exposure_time': float(mode.get('ExposureTime', 0)),
                    'intensity': float(mode.get('IlluminationIntensity', 0)),
                    'illumination_source': mode.get('IlluminationSource'),
                    'analog_gain': float(mode.get('AnalogGain', 0)),
                    'mode_id': mode.get('ID')
                }

        self.logger.info(f"Found {len(channels)} selected channels: {list(channels.keys())}")
        return channels

    def parse_coordinates_csv(self, experiment_folder: Path) -> dict:
        """
        Parse coordinates.csv and group by well.
        
        Args:
            experiment_folder: Path to experiment folder
            
        Returns:
            Dictionary grouped by well: {well_id: [coordinate_data]}
        """
        csv_file = experiment_folder / "0" / "coordinates.csv"
        if not csv_file.exists():
            raise FileNotFoundError(f"No coordinates.csv found in {experiment_folder}/0/")

        df = pd.read_csv(csv_file)

        # Group by 'region' column (which contains well IDs like 'G9')
        wells_data = {}

        for region in df['region'].unique():
            well_data = df[df['region'] == region].copy()
            # Only process k=0 (single focal plane)
            well_data = well_data[well_data['k'] == 0]

            if len(well_data) > 0:
                wells_data[region] = well_data.to_dict('records')

        self.logger.info(f"Found {len(wells_data)} wells with data: {list(wells_data.keys())}")
        return wells_data

    def create_xml_to_channel_mapping(self, xml_channels: dict) -> dict:
        """
        Create mapping from filename channel names to ChannelMapper human names.
        
        Args:
            xml_channels: Dictionary of channel configurations from XML
            
        Returns:
            Dictionary mapping filename channel names (zarr format) to human names (expected by canvas)
        """

        from squid_control.control.config import ChannelMapper
        
        # Create mapping from zarr names (used in filenames) to human names (expected by canvas)
        filename_to_human_mapping = {}
        
        # Get all channel info and create zarr_name -> human_name mapping
        for channel_info in ChannelMapper.CHANNELS.values():
            filename_to_human_mapping[channel_info.zarr_name] = channel_info.human_name
        
        print(f"Filename to human name mapping: {filename_to_human_mapping}")
        return filename_to_human_mapping

    def create_temp_experiment_manager(self, temp_path: str):
        """
        Create temporary experiment manager for offline stitching.
        
        Args:
            temp_path: Path for temporary zarr storage
            
        Returns:
            ExperimentManager instance
        """

        # Ensure temp directory exists
        Path(temp_path).mkdir(parents=True, exist_ok=True)

        from squid_control.stitching.zarr_canvas import ExperimentManager
        temp_exp_manager = ExperimentManager(
            base_path=temp_path,
            pixel_size_xy_um=self.squid_controller.pixel_size_xy
        )

        return temp_exp_manager

    async def load_and_stitch_well_images(self, well_data: List[dict],
                                        experiment_folder: Path,
                                        canvas, channel_mapping: dict) -> None:
        """
        Load BMP images and add them to well canvas.
        
        Args:
            well_data: List of coordinate records for this well
            experiment_folder: Path to experiment folder
            canvas: WellZarrCanvas instance
            channel_mapping: XML to ChannelMapper name mapping
        """
        data_folder = experiment_folder / "0"

        # Get available channels for this canvas
        available_channels = list(canvas.channel_to_zarr_index.keys())
        print(f"Available channels for canvas: {available_channels}")

        images_added = 0
        for coord_record in well_data:
            i = int(coord_record['i'])
            j = int(coord_record['j'])
            k = int(coord_record['k'])
            x_mm = float(coord_record['x (mm)'])
            y_mm = float(coord_record['y (mm)'])
            well_id = coord_record['region']

            # Skip if not k=0 (single focal plane only)
            if k != 0:
                continue

            # Find all image files for this position
            pattern = f"{well_id}_{i}_{j}_{k}_*.bmp"
            image_files = list(data_folder.glob(pattern))
            print(f"Well {well_id} position ({i},{j},{k}): found {len(image_files)} images with pattern {pattern}")

            for img_file in image_files:
                # Extract channel name from filename
                # Format: G9_2_1_0_Fluorescence_488_nm_Ex.bmp
                filename_parts = img_file.stem.split('_')
                if len(filename_parts) >= 5:
                    # Channel name is everything after the position indices (in zarr format)
                    channel_name = '_'.join(filename_parts[4:])

                    # Map zarr channel name to human name (expected by canvas)
                    mapped_channel_name = channel_mapping.get(channel_name, channel_name)

                    # Check if this channel is available in the canvas
                    if mapped_channel_name in available_channels:
                        # Load image
                        image = cv2.imread(str(img_file), cv2.IMREAD_GRAYSCALE)
                        if image is not None:
                            # Get zarr channel index
                            zarr_channel_idx = canvas.get_zarr_channel_index(mapped_channel_name)

                            # Add image to canvas using async stitching (same as normal_scan_with_stitching)
                            await canvas.add_image_async(
                                image, x_mm - canvas.well_center_x, y_mm - canvas.well_center_y,
                                zarr_channel_idx, 0, 0  # z_idx=0, timepoint=0
                            )

                            images_added += 1
                            print(f"✓ Added image {img_file.name} to canvas at ({x_mm:.2f}, {y_mm:.2f}) - channel {mapped_channel_name} (idx {zarr_channel_idx})")
                        else:
                            print(f"✗ Failed to load image: {img_file}")
                    else:
                        print(f"✗ Channel '{mapped_channel_name}' not available in canvas, skipping {img_file.name}")
        
        print(f"Total images added to canvas: {images_added}")

    def extract_timestamp_from_folder(self, folder_path: Path) -> str:
        """
        Extract timestamp identifier from folder name.
        
        Args:
            folder_path: Path to experiment folder (e.g., test-drug-20250822T143055)
            
        Returns:
            Timestamp string (e.g., '20250822T143055')
        """
        folder_name = folder_path.name
        # Find the last occurrence of '-' and take everything after it
        if '-' in folder_name:
            timestamp = folder_name.split('-')[-1]
            return timestamp
        else:
            # Fallback to folder name if no timestamp pattern found
            return folder_name

    async def process_experiment_run(self, experiment_folder: Path,
                                   gallery_id: str, upload_immediately: bool = True,
                                   cleanup_temp_files: bool = True) -> dict:
        """
        Process a single experiment run (one timestamped folder).
        
        Args:
            experiment_folder: Path to experiment folder
            gallery_id: Gallery ID for uploads
            upload_immediately: Whether to upload after processing
            cleanup_temp_files: Whether to cleanup temp files
            
        Returns:
            Dictionary with processing results
        """
        self.logger.info(f"Processing experiment run: {experiment_folder.name}")

        try:
            # 1. Parse metadata
            acquisition_params = self.parse_acquisition_parameters(experiment_folder)
            xml_channels = self.parse_configurations_xml(experiment_folder)
            channel_mapping = self.create_xml_to_channel_mapping(xml_channels)
            coordinates_data = self.parse_coordinates_csv(experiment_folder)

            # 2. Create temporary experiment for this run
            # Use CONFIG.DEFAULT_SAVING_PATH if available, otherwise fall back to system temp
            from squid_control.control.config import CONFIG
            
            if CONFIG.DEFAULT_SAVING_PATH and Path(CONFIG.DEFAULT_SAVING_PATH).exists():
                base_temp_path = Path(CONFIG.DEFAULT_SAVING_PATH)
                temp_path = base_temp_path / f"offline_stitch_{experiment_folder.name}_{int(time.time())}"
                temp_path.mkdir(parents=True, exist_ok=True)
                print(f"Using configured saving path for temporary stitching: {temp_path}")
            else:
                temp_path = Path(tempfile.mkdtemp(prefix=f"offline_stitch_{experiment_folder.name}_"))
                print(f"Using system temp directory for stitching: {temp_path}")
                
            temp_exp_manager = self.create_temp_experiment_manager(str(temp_path))

            # 3. Process each well
            zarr_files_info = []

            for well_id, well_data in coordinates_data.items():
                self.logger.info(f"Processing well {well_id} with {len(well_data)} positions")

                # Parse well row and column from well_id (e.g., 'G9' -> 'G', 9)
                if len(well_id) >= 2:
                    well_row = well_id[0]
                    try:
                        well_column = int(well_id[1:])
                    except ValueError:
                        self.logger.warning(f"Could not parse column from well ID: {well_id}")
                        continue
                else:
                    self.logger.warning(f"Invalid well ID format: {well_id}")
                    continue

                # Create well canvas using the standard approach (same as normal_scan_with_stitching)
                # Use very large padding to accommodate absolute coordinates
                canvas = temp_exp_manager.get_well_canvas(
                    well_row, well_column, '96', 100.0  # Very large padding for absolute coordinates
                )

                # Start stitching
                await canvas.start_stitching()

                try:
                    # Load and stitch images for this well
                    await self.load_and_stitch_well_images(
                        well_data, experiment_folder, canvas, channel_mapping
                    )

                    # Small delay to let stitching complete
                    await asyncio.sleep(0.1)

                    # Export as zip using asyncio.to_thread to avoid blocking
                    well_zip_content = await asyncio.to_thread(canvas.export_as_zip)

                    zarr_files_info.append({
                        'name': f"well_{well_row}{well_column}_96",
                        'content': well_zip_content,
                        'size_mb': len(well_zip_content) / (1024 * 1024)
                    })

                    self.logger.info(f"Exported well {well_id} as {len(well_zip_content)/(1024*1024):.2f} MB zip")

                finally:
                    await canvas.stop_stitching()

            # 4. Upload if requested
            upload_result = None
            raw_data_upload_result = None
            if upload_immediately and zarr_files_info:
                timestamp = self.extract_timestamp_from_folder(experiment_folder)
                dataset_name = f"run-{timestamp}"

                self.logger.info(f"Uploading {len(zarr_files_info)} wells as dataset '{dataset_name}'")

                # 4a. First, create and upload raw data backup
                self.logger.info(f"Creating raw data backup for experiment folder: {experiment_folder.name}")
                raw_data_zip_content = await self._create_raw_data_backup(experiment_folder)

                # Add raw data backup to the files to upload
                zarr_files_info_with_backup = zarr_files_info.copy()
                zarr_files_info_with_backup.append({
                    'name': f"raw_data_backup_{experiment_folder.name}",
                    'content': raw_data_zip_content,
                    'size_mb': len(raw_data_zip_content) / (1024 * 1024)
                })

                self.logger.info(f"Raw data backup created: {len(raw_data_zip_content)/(1024*1024):.2f} MB")

                # 4b. Upload all files (well zarrs + raw data backup) using multipart upload
                upload_result = await self.zarr_artifact_manager.upload_multiple_zarr_files_to_dataset(
                    microscope_service_id=self.service_id,
                    experiment_id=dataset_name,
                    zarr_files_info=zarr_files_info_with_backup,
                    acquisition_settings={
                        "acquisition_parameters": acquisition_params,
                        "channel_configurations": xml_channels,
                        "experiment_run": experiment_folder.name,
                        "well_count": len(zarr_files_info),
                        "raw_data_backup_included": True,
                        "microscope_service_id": self.service_id,
                        "pixel_size_xy_um": self.squid_controller.pixel_size_xy
                    },
                    description=f"Offline processed experiment run: {experiment_folder.name} (includes raw data backup)"
                )

                self.logger.info(f"Upload completed for dataset '{dataset_name}' with raw data backup")

            # 5. Cleanup temporary files
            if cleanup_temp_files:
                shutil.rmtree(temp_path, ignore_errors=True)
                self.logger.debug(f"Cleaned up temporary files: {temp_path}")

            return {
                "success": True,
                "experiment_folder": experiment_folder.name,
                "wells_processed": len(zarr_files_info),
                "total_size_mb": sum(file_info['size_mb'] for file_info in zarr_files_info),
                "raw_data_backup_included": raw_data_upload_result is not None or (upload_result and upload_result.get("raw_data_backup_included", False)),
                "upload_result": upload_result
            }

        except Exception as e:
            self.logger.error(f"Error processing experiment run {experiment_folder.name}: {e}")
            return {
                "success": False,
                "experiment_folder": experiment_folder.name,
                "error": str(e)
            }

    async def stitch_and_upload_timelapse(self, experiment_id: str,
                                        upload_immediately: bool = True,
                                        cleanup_temp_files: bool = True) -> dict:
        """
        Main function for offline stitching and uploading.
        
        Args:
            experiment_id: Experiment ID to search for
            upload_immediately: Whether to upload each run after stitching
            cleanup_temp_files: Whether to delete temporary files after upload
            
        Returns:
            Dictionary with processing results
        """
        results = {
            "success": True,
            "experiment_id": experiment_id,
            "gallery_id": None,
            "processed_runs": [],
            "failed_runs": [],
            "total_datasets": 0,
            "total_size_mb": 0,
            "start_time": time.time()
        }

        try:
            print("=" * 60)
            print(f"OFFLINE PROCESSING STARTED for experiment_id: {experiment_id}")
            print(f"Parameters: upload_immediately={upload_immediately}, cleanup_temp_files={cleanup_temp_files}")
            print("=" * 60)
            
            # Find all experiment folders
            print("Searching for experiment folders...")
            experiment_folders = self.find_experiment_folders(experiment_id)

            if not experiment_folders:
                results["success"] = False
                results["message"] = f"No experiment folders found for ID: {experiment_id}"
                results["processing_time_seconds"] = time.time() - results["start_time"]
                self.logger.warning(f"No experiment folders found for ID: {experiment_id}")
                return results

            # Create gallery once if uploading
            gallery_id = None
            if upload_immediately:
                gallery_name = f"experiment-{experiment_id}"
                gallery_id = await self._create_or_get_gallery(gallery_name)
                results["gallery_id"] = gallery_id
                self.logger.info(f"Using gallery: {gallery_name} (ID: {gallery_id})")

            # Process each experiment run
            for i, exp_folder in enumerate(experiment_folders):
                self.logger.info(f"Processing run {i+1}/{len(experiment_folders)}: {exp_folder.name}")

                # Use asyncio.to_thread for CPU-intensive stitching operations
                run_result = await self.process_experiment_run(
                    exp_folder, gallery_id, upload_immediately, cleanup_temp_files
                )

                if run_result["success"]:
                    results["processed_runs"].append(run_result)
                    results["total_datasets"] += 1
                    results["total_size_mb"] += run_result.get("total_size_mb", 0)
                else:
                    results["failed_runs"].append(run_result)

            results["processing_time_seconds"] = time.time() - results["start_time"]

            self.logger.info(f"Completed processing: {results['total_datasets']} datasets, "
                           f"{len(results['failed_runs'])} failures, "
                           f"{results['total_size_mb']:.2f} MB total")

        except Exception as e:
            results["success"] = False
            results["error"] = str(e)
            self.logger.error(f"Offline processing failed: {e}")

        return results

    async def _create_or_get_gallery(self, gallery_name: str) -> str:
        """
        Create or get existing gallery for the experiment.
        
        Args:
            gallery_name: Name of the gallery to create/get
            
        Returns:
            Gallery ID
        """
        try:
            # Use the correct method from SquidArtifactManager
            gallery_result = await self.zarr_artifact_manager.create_or_get_microscope_gallery(
                self.service_id, gallery_name
            )
            gallery_id = gallery_result["id"]
            self.logger.info(f"Created or found gallery: {gallery_name} (ID: {gallery_id})")

        except Exception as e:
            # If that fails, try with a unique gallery name with timestamp
            self.logger.warning(f"Gallery creation failed: {e}")
            
            import datetime
            timestamp = datetime.datetime.now().strftime("%Y%m%dT%H%M%S")
            unique_gallery_name = f"{gallery_name}-{timestamp}"

            try:
                gallery_result = await self.zarr_artifact_manager.create_or_get_microscope_gallery(
                    self.service_id, unique_gallery_name
                )
                gallery_id = gallery_result["id"]
                self.logger.info(f"Created unique gallery: {unique_gallery_name} (ID: {gallery_id})")
            except Exception as e2:
                self.logger.error(f"Failed to create gallery even with unique name: {e2}")
                raise e2

        return gallery_id

    async def _create_raw_data_backup(self, experiment_folder: Path) -> bytes:
        """
        Create a zip backup of the raw experiment folder.
        
        Args:
            experiment_folder: Path to the experiment folder to backup
            
        Returns:
            bytes: ZIP file content as bytes
        """

        self.logger.info(f"Creating raw data backup for: {experiment_folder}")

        # Create ZIP in memory
        zip_buffer = io.BytesIO()

        with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED, allowZip64=True) as zipf:
            # Walk through all files in the experiment folder
            for root, dirs, files in os.walk(experiment_folder):
                for file in files:
                    file_path = Path(root) / file

                    # Calculate relative path from experiment folder
                    try:
                        relative_path = file_path.relative_to(experiment_folder)
                        # Create archive name with experiment folder name as root
                        arcname = f"{experiment_folder.name}/{relative_path}"

                        # Add file to zip
                        zipf.write(file_path, arcname=arcname)

                    except ValueError:
                        # Skip files that can't be made relative (shouldn't happen)
                        self.logger.warning(f"Could not make path relative: {file_path}")
                        continue

        # Get the ZIP content
        zip_content = zip_buffer.getvalue()
        zip_buffer.close()

        self.logger.info(f"Raw data backup created: {len(zip_content)/(1024*1024):.2f} MB")

        return zip_content
