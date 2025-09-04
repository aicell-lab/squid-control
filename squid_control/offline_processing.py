"""
Offline processing module for time-lapse experiment data.
Handles stitching and uploading of stored microscopy data.
"""

import asyncio
import json
import xml.etree.ElementTree as ET
import pandas as pd
import shutil
import logging
import cv2
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import tempfile
import time

logger = logging.getLogger(__name__)


class OfflineProcessor:
    """Handles offline stitching and uploading of time-lapse data."""
    
    def __init__(self, squid_controller):
        self.squid_controller = squid_controller
        self.logger = logger
    
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
        
        base_path = Path(CONFIG.DEFAULT_SAVING_PATH)
        if not base_path.exists():
            raise FileNotFoundError(f"Base path does not exist: {base_path}")
        
        pattern = f"{experiment_id}-*"
        folders = sorted(base_path.glob(pattern))
        
        # Filter to only directories that contain a '0' subfolder
        valid_folders = []
        for folder in folders:
            if folder.is_dir():
                zero_folder = folder / "0"
                if zero_folder.exists() and zero_folder.is_dir():
                    valid_folders.append(folder)
                else:
                    self.logger.warning(f"Skipping {folder.name}: no '0' subfolder found")
        
        self.logger.info(f"Found {len(valid_folders)} valid experiment folders for '{experiment_id}'")
        return valid_folders
    
    def parse_acquisition_parameters(self, experiment_folder: Path) -> dict:
        """
        Parse acquisition parameters.json from the '0' folder.
        
        Args:
            experiment_folder: Path to experiment folder (e.g., test-drug-20250822T143055)
            
        Returns:
            Dictionary with acquisition parameters
        """
        json_file = experiment_folder / "0" / "acquisition parameters.json"
        if not json_file.exists():
            raise FileNotFoundError(f"No acquisition parameters.json found in {experiment_folder}/0/")
        
        with open(json_file, 'r') as f:
            params = json.load(f)
        
        self.logger.info(f"Loaded acquisition parameters: Nx={params.get('Nx')}, Ny={params.get('Ny')}, dx={params.get('dx(mm)')}, dy={params.get('dy(mm)')}")
        return params
    
    def parse_configurations_xml(self, experiment_folder: Path) -> dict:
        """
        Parse configurations.xml and extract channel settings.
        
        Args:
            experiment_folder: Path to experiment folder
            
        Returns:
            Dictionary with channel configurations
        """
        xml_file = experiment_folder / "0" / "configurations.xml"
        if not xml_file.exists():
            raise FileNotFoundError(f"No configurations.xml found in {experiment_folder}/0/")
        
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
        Create mapping from XML channel names to ChannelMapper names.
        
        Args:
            xml_channels: Dictionary of channel configurations from XML
            
        Returns:
            Dictionary mapping XML names to ChannelMapper names
        """
        from squid_control.control.config import ChannelMapper
        
        channel_mapper_names = ChannelMapper.get_all_human_names()
        mapping = {}
        
        for xml_name in xml_channels.keys():
            # Try exact match first
            if xml_name in channel_mapper_names:
                mapping[xml_name] = xml_name
                self.logger.debug(f"Exact match: '{xml_name}' -> '{xml_name}'")
            else:
                # Try to find closest match or use fallback
                # For now, use the same name as fallback
                mapping[xml_name] = xml_name
                self.logger.warning(f"No exact match for '{xml_name}', using as-is")
        
        return mapping
    
    def create_temp_experiment_manager(self, temp_path: str):
        """
        Create temporary experiment manager for offline stitching.
        
        Args:
            temp_path: Path for temporary zarr storage
            
        Returns:
            ExperimentManager instance
        """
        from squid_control.stitching.zarr_canvas import ExperimentManager
        
        # Ensure temp directory exists
        Path(temp_path).mkdir(parents=True, exist_ok=True)
        
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
            
            for img_file in image_files:
                # Extract channel name from filename
                # Format: G9_2_1_0_Fluorescence_488_nm_Ex.bmp
                filename_parts = img_file.stem.split('_')
                if len(filename_parts) >= 5:
                    # Channel name is everything after the position indices
                    channel_name = '_'.join(filename_parts[4:])
                    
                    # Map XML channel name to ChannelMapper name
                    mapped_channel_name = channel_mapping.get(channel_name, channel_name)
                    
                    # Check if this channel is available in the canvas
                    if mapped_channel_name in available_channels:
                        # Load image
                        image = cv2.imread(str(img_file), cv2.IMREAD_GRAYSCALE)
                        if image is not None:
                            # Get zarr channel index
                            zarr_channel_idx = canvas.get_zarr_channel_index(mapped_channel_name)
                            
                            # Add image to canvas using absolute coordinates
                            await canvas.add_image_from_absolute_coords_async(
                                image, x_mm, y_mm, zarr_channel_idx, 0, 0  # timepoint=0
                            )
                            
                            self.logger.debug(f"Added image {img_file.name} to canvas at ({x_mm:.2f}, {y_mm:.2f})")
                        else:
                            self.logger.warning(f"Failed to load image: {img_file}")
                    else:
                        self.logger.debug(f"Channel '{mapped_channel_name}' not available in canvas, skipping {img_file.name}")
    
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
            temp_path = Path(tempfile.mkdtemp(prefix=f"offline_stitch_{experiment_folder.name}_"))
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
                
                # Create well canvas
                canvas = temp_exp_manager.get_well_canvas(
                    well_row, well_column, '96', 1.0  # Default to 96-well
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
            if upload_immediately and zarr_files_info:
                timestamp = self.extract_timestamp_from_folder(experiment_folder)
                dataset_name = f"run-{timestamp}"
                
                self.logger.info(f"Uploading {len(zarr_files_info)} wells as dataset '{dataset_name}'")
                
                upload_result = await self.squid_controller.zarr_artifact_manager.upload_multiple_zarr_files_to_dataset(
                    microscope_service_id=self.squid_controller.service_id,
                    experiment_id=dataset_name,
                    zarr_files_info=zarr_files_info,
                    acquisition_settings={
                        "acquisition_parameters": acquisition_params,
                        "channel_configurations": xml_channels,
                        "experiment_run": experiment_folder.name,
                        "well_count": len(zarr_files_info),
                        "microscope_service_id": self.squid_controller.service_id,
                        "pixel_size_xy_um": self.squid_controller.pixel_size_xy
                    },
                    description=f"Offline processed experiment run: {experiment_folder.name}"
                )
                
                self.logger.info(f"Upload completed for dataset '{dataset_name}'")
            
            # 5. Cleanup temporary files
            if cleanup_temp_files:
                shutil.rmtree(temp_path, ignore_errors=True)
                self.logger.debug(f"Cleaned up temporary files: {temp_path}")
            
            return {
                "success": True,
                "experiment_folder": experiment_folder.name,
                "wells_processed": len(zarr_files_info),
                "total_size_mb": sum(file_info['size_mb'] for file_info in zarr_files_info),
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
            # Find all experiment folders
            experiment_folders = self.find_experiment_folders(experiment_id)
            
            if not experiment_folders:
                raise ValueError(f"No experiment folders found for ID: {experiment_id}")
            
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
            # Try to create a new gallery - this will fail if it already exists
            gallery_result = await self.squid_controller.zarr_artifact_manager.create_gallery(
                self.squid_controller.service_id, gallery_name
            )
            gallery_id = gallery_result["gallery_id"]
            self.logger.info(f"Created new gallery: {gallery_name}")
            
        except Exception as e:
            # Gallery likely already exists, try to find it
            self.logger.info(f"Gallery creation failed (likely exists): {e}")
            
            # For now, create a unique gallery name with timestamp
            import datetime
            timestamp = datetime.datetime.now().strftime("%Y%m%dT%H%M%S")
            unique_gallery_name = f"{gallery_name}-{timestamp}"
            
            gallery_result = await self.squid_controller.zarr_artifact_manager.create_gallery(
                self.squid_controller.service_id, unique_gallery_name
            )
            gallery_id = gallery_result["gallery_id"]
            self.logger.info(f"Created unique gallery: {unique_gallery_name}")
        
        return gallery_id
