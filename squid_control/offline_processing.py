"""
Offline processing module for time-lapse experiment data.
Handles stitching and uploading of stored microscopy data.
"""
import os
import asyncio
import json
import logging
import shutil
import tempfile
import time
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import List

import cv2
import pandas as pd

# Use print statements for debugging since logger configuration seems problematic
logger = logging.getLogger(__name__)


class OfflineProcessor:
    """Handles offline stitching and uploading of time-lapse data."""

    def __init__(self, squid_controller, zarr_artifact_manager=None, service_id=None,
                 max_concurrent_wells=3, image_batch_size=5):
        print("🔧 OfflineProcessor.__init__ called")
        self.squid_controller = squid_controller
        self.zarr_artifact_manager = zarr_artifact_manager
        self.service_id = service_id
        self.logger = logger

        # Performance configuration
        self.max_concurrent_wells = max_concurrent_wells
        self.image_batch_size = image_batch_size

        # Ensure configuration is loaded
        self._ensure_config_loaded()

    def _ensure_config_loaded(self):
        """Ensure the configuration is properly loaded with auto-detection."""
        from squid_control.control.config import CONFIG, load_config

        # Check if DEFAULT_SAVING_PATH is already loaded
        if not CONFIG.DEFAULT_SAVING_PATH:
            print("Configuration not loaded, attempting auto-detection...")
            try:
                # Use the same auto-detection logic as SquidController
                current_dir = Path(__file__).parent

                # Check main squid_control directory first
                squid_plus_config_main = current_dir / "configuration_Squid+.ini"
                hcs_config_main = current_dir / "configuration_HCS_v2.ini"

                if squid_plus_config_main.exists():
                    config_path = squid_plus_config_main
                    print("🔬 OfflineProcessor detected Squid+ microscope - using Squid+ configuration")
                elif hcs_config_main.exists():
                    config_path = hcs_config_main
                    print("🔬 OfflineProcessor detected original Squid microscope - using HCS_v2 configuration")
                else:
                    # Fall back to config directory
                    squid_plus_config = current_dir / "config" / "configuration_Squid+.ini"
                    hcs_config = current_dir / "config" / "configuration_HCS_v2.ini"

                    if squid_plus_config.exists():
                        config_path = squid_plus_config
                        print("🔬 OfflineProcessor detected Squid+ microscope - using Squid+ configuration from config directory")
                    elif hcs_config.exists():
                        config_path = hcs_config
                        print("🔬 OfflineProcessor detected original Squid microscope - using HCS_v2 configuration from config directory")
                    else:
                        # Final fallback
                        config_path = hcs_config
                        print("⚠️ OfflineProcessor: No configuration file found, using default path (may fail)")

                if config_path.exists():
                    load_config(str(config_path), None)
                    print(f"Configuration loaded: DEFAULT_SAVING_PATH = {CONFIG.DEFAULT_SAVING_PATH}")
                else:
                    print(f"Config file not found at {config_path}")
            except Exception as e:
                print(f"Failed to load configuration: {e}")
        else:
            print(f"Configuration already loaded: DEFAULT_SAVING_PATH = {CONFIG.DEFAULT_SAVING_PATH}")

    async def _upload_zarr_with_hypha_artifact(self, zarr_folder_path: str, dataset_name: str, 
                                                acquisition_settings: dict = None, 
                                                description: str = "", server_url: str = None) -> dict:
        """
        Upload a zarr folder to the artifact manager using hypha-artifact.
        
        Args:
            zarr_folder_path: Path to the zarr folder to upload
            dataset_name: Name for the dataset
            acquisition_settings: Optional acquisition settings metadata
            description: Optional description for the dataset
            server_url: Server URL (defaults to https://hypha.aicell.io)
            
        Returns:
            dict with upload result information
        """
        from hypha_artifact import AsyncHyphaArtifact
        
        token = os.environ.get("AGENT_LENS_WORKSPACE_TOKEN")
        if not token:
            raise Exception("AGENT_LENS_WORKSPACE_TOKEN environment variable not set")
        
        server_url = server_url or "https://hypha.aicell.io"
        
        print(f"📤 Creating artifact '{dataset_name}' for upload...")
        
        artifact = AsyncHyphaArtifact(
            artifact_id=dataset_name,
            workspace="agent-lens",
            token=token,
            server_url=server_url
        )
        
        # Edit mode for staging changes
        await artifact.edit(stage=True)
        
        # Upload the zarr folder recursively
        print(f"📤 Uploading zarr folder: {zarr_folder_path}")
        await artifact.put(zarr_folder_path, "/data.zarr", recursive=True)
        
        # Add manifest with acquisition settings if available
        if acquisition_settings:
            manifest_content = json.dumps({
                "name": dataset_name,
                "description": description,
                "acquisition_settings": acquisition_settings
            }, indent=2)
            await artifact.put(manifest_content.encode(), "/manifest.json")
        
        # Commit the changes
        await artifact.commit(comment=description or f"Uploaded {dataset_name}")
        
        print(f"✅ Successfully uploaded '{dataset_name}'")
        
        return {
            "success": True,
            "dataset_name": dataset_name,
            "description": description
        }

    def find_experiment_folders(self, experiment_id: str) -> List[Path]:
        """
        Find all experiment folders matching experiment_id prefix.
        
        Args:
            experiment_id: Experiment ID to search for (e.g., 'test-drug')
            
        Returns:
            Sorted list of Path objects like:
            [experiment_id-20250822T143055, experiment_id_2025-08-22_14-30-55, ...]
        """

        from squid_control.control.config import CONFIG
        print(f"CONFIG.DEFAULT_SAVING_PATH = {CONFIG.DEFAULT_SAVING_PATH}")
        base_path = Path(CONFIG.DEFAULT_SAVING_PATH)
        print(f"Searching in base path: {base_path}")
        if not base_path.exists():
            raise FileNotFoundError(f"Base path does not exist: {base_path}")

        # Search for both hyphen and underscore separators
        # e.g., 'test-drug-20250822T143055' or 'test-drug_2025-08-22_14-30-55'
        pattern_hyphen = f"{experiment_id}-*"
        pattern_underscore = f"{experiment_id}_*"
        print(f"Using patterns: {pattern_hyphen} and {pattern_underscore}")
        
        folders_hyphen = list(base_path.glob(pattern_hyphen))
        folders_underscore = list(base_path.glob(pattern_underscore))
        
        # Combine and deduplicate
        all_folders = set(folders_hyphen + folders_underscore)
        folders = sorted(all_folders)
        print(f"Found {len(folders)} folders matching patterns: {[f.name for f in folders]}")

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

            # Handle both CSV formats: old format has 'k' column, new format has 'z_level'
            if 'k' in well_data.columns:
                # Old format: Only process k=0 (single focal plane)
                well_data = well_data[well_data['k'] == 0]
            elif 'z_level' in well_data.columns:
                # New format: Only process z_level=0 (single focal plane)
                well_data = well_data[well_data['z_level'] == 0]
            else:
                # If neither column exists, process all data
                self.logger.warning(f"No 'k' or 'z_level' column found in CSV, processing all data for region {region}")

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

    def _load_and_stitch_well_images_sync(self, well_data: List[dict],
                                         experiment_folder: Path,
                                         canvas, channel_mapping: dict) -> None:
        """
        Load BMP images and add them to single canvas using absolute stage coordinates.
        
        Supports multiple file naming conventions:
        1. Traditional: {region}_{i}_{j}_{k}_{channel}.bmp (e.g., A1_0_0_0_BF_LED_matrix_full.bmp)
        2. Numeric region: {region_number}_{i}_{j}_{k}_{channel}.bmp (e.g., 0_0_0_0_BF_LED_matrix_full.bmp)
        3. Position ID: {position_id}_{i}_{j}_{k}_{channel}.bmp (e.g., 402_0_0_0_BF_LED_matrix_full.bmp)
        
        Args:
            well_data: List of coordinate records (from all positions)
            experiment_folder: Path to experiment folder
            canvas: ZarrCanvas instance (single canvas for experiment)
            channel_mapping: XML to ChannelMapper name mapping
        """
        data_folder = experiment_folder / "0"

        # Get available channels for this canvas
        available_channels = list(canvas.channel_to_zarr_index.keys())
        print(f"Available channels for canvas: {available_channels}")

        # Build a mapping from position index to coordinate record
        # This handles the case where filenames use position index (0, 1, 2, ...) 
        # instead of well IDs
        position_to_coords = {}
        for idx, coord_record in enumerate(well_data):
            # Handle both CSV formats
            if 'i' in coord_record and 'j' in coord_record:
                i = int(coord_record['i'])
                j = int(coord_record['j'])
                k = int(coord_record.get('k', 0))
            else:
                i = int(coord_record['fov'])
                j = 0
                k = int(coord_record.get('z_level', 0))

            x_mm = float(coord_record['x (mm)'])
            y_mm = float(coord_record['y (mm)'])
            region = coord_record.get('region', str(idx))
            
            # Store both by region and by index for flexible matching
            position_to_coords[str(region)] = {
                'i': i, 'j': j, 'k': k, 'x_mm': x_mm, 'y_mm': y_mm, 'region': region
            }
            position_to_coords[str(idx)] = {
                'i': i, 'j': j, 'k': k, 'x_mm': x_mm, 'y_mm': y_mm, 'region': region
            }

        # Scan all BMP files and group by their position ID prefix
        all_bmp_files = list(data_folder.glob("*.bmp"))
        print(f"Found {len(all_bmp_files)} total BMP files in {data_folder}")
        
        if not all_bmp_files:
            print(f"⚠️ No BMP files found in {data_folder}")
            return

        # Group files by position ID (first part of filename before _i_j_k)
        position_files = {}
        for bmp_file in all_bmp_files:
            filename = bmp_file.stem  # Remove .bmp extension
            parts = filename.split('_')
            
            if len(parts) < 4:
                print(f"  ⚠️ Skipping file with unexpected format: {bmp_file.name}")
                continue
            
            # Extract position ID and i, j, k from filename
            # Format: {position_id}_{i}_{j}_{k}_{channel...}
            position_id = parts[0]
            try:
                file_i = int(parts[1])
                file_j = int(parts[2])
                file_k = int(parts[3])
            except ValueError:
                print(f"  ⚠️ Could not parse i,j,k from filename: {bmp_file.name}")
                continue
            
            # Skip non-zero k (z-stacks) for now
            if file_k != 0:
                continue
            
            key = (position_id, file_i, file_j, file_k)
            if key not in position_files:
                position_files[key] = []
            position_files[key].append(bmp_file)

        print(f"Grouped into {len(position_files)} unique positions")

        # Build a list of all image tasks to process
        image_tasks = []
        for (position_id, file_i, file_j, file_k), image_files in position_files.items():
            # Try to find matching coordinates
            coords = None
            
            if position_id in position_to_coords:
                coords = position_to_coords[position_id]
            else:
                # Try treating position_id as an index into well_data
                try:
                    idx = int(position_id)
                    if 0 <= idx < len(well_data):
                        coord_record = well_data[idx]
                        x_mm = float(coord_record['x (mm)'])
                        y_mm = float(coord_record['y (mm)'])
                        coords = {'x_mm': x_mm, 'y_mm': y_mm, 'region': position_id}
                except (ValueError, KeyError):
                    pass
            
            if coords is None:
                continue

            x_mm = coords['x_mm']
            y_mm = coords['y_mm']
            
            for img_file in image_files:
                image_tasks.append((img_file, x_mm, y_mm))

        total_images = len(image_tasks)
        print(f"📊 Prepared {total_images} images to process using multi-threading...")

        # Process images using thread pool for parallel loading
        import concurrent.futures
        from threading import Lock
        
        images_added = 0
        images_failed = 0
        counter_lock = Lock()
        last_progress_time = time.time()
        
        def process_single_image(task):
            """Process a single image - runs in thread pool"""
            nonlocal images_added, images_failed, last_progress_time
            
            img_file, x_mm, y_mm = task
            success = self._load_and_process_single_image_sync(
                img_file, x_mm, y_mm, canvas, channel_mapping, available_channels
            )
            
            with counter_lock:
                if success:
                    images_added += 1
                else:
                    images_failed += 1
                
                # Print progress every 5 seconds
                current_time = time.time()
                if current_time - last_progress_time >= 5.0:
                    queue_size = canvas.preprocessing_queue.qsize()
                    print(f"📈 Progress: {images_added}/{total_images} queued, {images_failed} failed, queue_size={queue_size}")
                    last_progress_time = current_time
            
            return success

        # Use threading for parallel image loading, all available cores-1 to avoid blocking the main thread
        num_workers = os.cpu_count() - 1
        if num_workers <= 0:
            num_workers = 1
        print(f"🚀 Starting {num_workers} worker threads for image loading...")
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as executor:
            # Submit all tasks using map
            executor.map(process_single_image, image_tasks)

    def _load_and_process_single_image_sync(self, img_file: Path, x_mm: float, y_mm: float,
                                           canvas, channel_mapping: dict, available_channels: list) -> bool:
        """
        Load and process a single image sequentially (no async operations).
        
        Returns:
            True if image was successfully added, False otherwise
        """
        try:
            # Extract channel name from filename
            filename_parts = img_file.stem.split('_')

            # Handle both filename formats by looking for channel keywords
            # Look for "Fluorescence" or "BF" to identify the start of channel name
            channel_start_idx = None
            for i, part in enumerate(filename_parts):
                if part in ['Fluorescence', 'BF']:
                    channel_start_idx = i
                    break

            if channel_start_idx is not None:
                channel_name = '_'.join(filename_parts[channel_start_idx:])
            else:
                # Silently skip files without channel keyword
                return False

            mapped_channel_name = channel_mapping.get(channel_name, channel_name)

            # Check if this channel is available in the canvas
            if mapped_channel_name not in available_channels:
                # Silently skip unavailable channels
                return False

            # Load image synchronously (no thread pool)
            image = cv2.imread(str(img_file), cv2.IMREAD_GRAYSCALE)

            if image is None:
                print(f"    Failed to load image file: {img_file}")
                return False

            # Get zarr channel index
            try:
                zarr_channel_idx = canvas.get_zarr_channel_index(mapped_channel_name)
            except Exception as e:
                print(f"    ❌ Failed to get zarr channel index for {mapped_channel_name}: {e}")
                import traceback
                traceback.print_exc()
                return False

            # Add image to canvas using put_nowait (non-blocking)
            # We're in sync context but need to add to async queue
            # put_nowait() doesn't require the event loop to be running
            import asyncio

            queue_item = {
                'image': image.copy(),
                'x_mm': x_mm,
                'y_mm': y_mm,
                'channel_idx': zarr_channel_idx,
                'z_idx': 0,
                'timepoint': 0,
                'timestamp': time.time(),
                'quick_scan': False  # Process all scales
            }

            # Try to add to queue - wait indefinitely if queue is full
            while True:
                try:
                    # Use put_nowait - doesn't block
                    canvas.preprocessing_queue.put_nowait(queue_item)
                    return True  # Success!
                except asyncio.QueueFull:
                    # Queue is full, wait and retry (no limit - just wait until space available)
                    time.sleep(0.5)
                    continue

        except Exception as e:
            print(f"    ❌ Failed to process image {img_file}: {type(e).__name__}: {e}")
            import traceback
            traceback.print_exc()
            return False

    async def _load_and_process_single_image(self, img_file: Path, x_mm: float, y_mm: float,
                                           canvas, channel_mapping: dict, available_channels: list) -> bool:
        """
        Load and process a single image asynchronously.
        
        Returns:
            True if image was successfully added, False otherwise
        """
        try:
            # Extract channel name from filename
            filename_parts = img_file.stem.split('_')

            # Handle both filename formats by looking for channel keywords
            # Look for "Fluorescence" or "BF" to identify the start of channel name
            channel_start_idx = None
            for i, part in enumerate(filename_parts):
                if part in ['Fluorescence', 'BF']:
                    channel_start_idx = i
                    break

            if channel_start_idx is not None:
                channel_name = '_'.join(filename_parts[channel_start_idx:])
            else:
                return False

            mapped_channel_name = channel_mapping.get(channel_name, channel_name)

            # Check if this channel is available in the canvas
            if mapped_channel_name not in available_channels:
                return False

            # Load image in thread pool to avoid blocking
            loop = asyncio.get_event_loop()
            image = await loop.run_in_executor(
                None, cv2.imread, str(img_file), cv2.IMREAD_GRAYSCALE
            )

            if image is None:
                return False

            # Get zarr channel index
            zarr_channel_idx = canvas.get_zarr_channel_index(mapped_channel_name)

            # Add image to canvas using stitching queue (same pattern as scan_region_to_zarr)
            await self._add_image_to_stitching_queue(canvas, image, x_mm, y_mm, zarr_channel_idx)

            return True

        except Exception as e:
            print(f"✗ Failed to process image {img_file}: {e}")
            return False

    def extract_timestamp_from_folder(self, folder_path: Path) -> str:
        """
        Extract timestamp identifier from folder name.
        
        Handles both old and new timestamp formats:
        - Old format: 20250718-U2OS-FUCCI-Eto-ER-20250720T095000_2025-07-20_09-51-13.673205
        - New format: 20250718-U2OS-FUCCI-Eto-ER_2025-07-20_09-51-13.673205
        
        Args:
            folder_path: Path to experiment folder
            
        Returns:
            Normalized timestamp string (e.g., '2025-07-20_09-51-13')
        """
        import re

        folder_name = folder_path.name

        # Pattern to match the normalized timestamp format: _YYYY-MM-DD_HH-MM-SS
        # This pattern will match both old and new formats
        timestamp_pattern = r'_(\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2}(?:\.\d+)?)'

        match = re.search(timestamp_pattern, folder_name)
        if match:
            # Extract the timestamp part (without the leading underscore)
            timestamp = match.group(1)
            # Remove microseconds if present for consistency
            if '.' in timestamp:
                timestamp = timestamp.split('.')[0]
            return timestamp
        else:
            # Fallback: try to find any timestamp-like pattern
            # Look for patterns like YYYYMMDDTHHMMSS or YYYY-MM-DD_HH-MM-SS
            fallback_patterns = [
                r'(\d{8}T\d{6})',  # YYYYMMDDTHHMMSS
                r'(\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2})',  # YYYY-MM-DD_HH-MM-SS
                r'(\d{4}-\d{2}-\d{2}_\d{2}:\d{2}:\d{2})',  # YYYY-MM-DD_HH:MM:SS
            ]

            for pattern in fallback_patterns:
                match = re.search(pattern, folder_name)
                if match:
                    return match.group(1)

            # Final fallback: use folder name
            return folder_name

    def create_normalized_dataset_name(self, experiment_folder: Path, experiment_id: str = None) -> str:
        """
        Create a normalized dataset name using the extracted timestamp.
        
        This helps group related experiments together by using a consistent
        timestamp format regardless of the original folder naming.
        
        Args:
            experiment_folder: Path to experiment folder
            experiment_id: Optional experiment ID prefix
            
        Returns:
            Normalized dataset name (e.g., 'experiment-20250720-095113')
        """
        # Extract the normalized timestamp
        timestamp = self.extract_timestamp_from_folder(experiment_folder)

        # Get folder name for reference
        folder_name = experiment_folder.name

        # Use experiment_id as prefix if provided, otherwise use folder name base
        if experiment_id:
            # Extract base name from experiment_id (before any timestamps)
            base_name = experiment_id.split('_')[0] if '_' in experiment_id else experiment_id
        else:
            # Extract base name from folder (everything before the first timestamp pattern)
            # Remove timestamp patterns to get base name
            import re
            base_name = re.sub(r'_\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2}(?:\.\d+)?.*$', '', folder_name)
            base_name = re.sub(r'-\d{8}T\d{6}.*$', '', base_name)

        # Convert timestamp to old format (YYYYMMDD-HHMMSS) if we have a valid timestamp
        if timestamp and timestamp != folder_name:
            # Check if timestamp is in the new format (YYYY-MM-DD_HH-MM-SS)
            if '_' in timestamp and '-' in timestamp:
                # Convert from YYYY-MM-DD_HH-MM-SS to YYYYMMDD-HHMMSS
                date_part, time_part = timestamp.split('_')
                date_compact = date_part.replace('-', '')
                time_compact = time_part.replace('-', '')
                old_format_timestamp = f"{date_compact}-{time_compact}"
            else:
                # Already in old format or some other format, use as-is
                old_format_timestamp = timestamp
            normalized_name = f"{base_name}-{old_format_timestamp}"
        else:
            normalized_name = base_name

        # Sanitize the final name
        return self.sanitize_dataset_name(normalized_name)

    def sanitize_dataset_name(self, name: str) -> str:
        """
        Sanitize dataset name to meet artifact manager requirements.
        
        Requirements: lowercase letters, numbers, hyphens, and colons only.
        Must start and end with alphanumeric character.
        
        Args:
            name: Original dataset name
            
        Returns:
            Sanitized dataset name
        """
        import re

        # Convert to lowercase
        sanitized = name.lower()

        # Replace invalid characters with hyphens
        # Keep only: lowercase letters, numbers, hyphens, colons
        sanitized = re.sub(r'[^a-z0-9\-:]', '-', sanitized)

        # Remove multiple consecutive hyphens
        sanitized = re.sub(r'-+', '-', sanitized)

        # Remove leading/trailing hyphens and colons
        sanitized = sanitized.strip('-:')

        # Ensure it starts and ends with alphanumeric
        if not sanitized[0].isalnum():
            sanitized = 'run-' + sanitized
        if not sanitized[-1].isalnum():
            sanitized = sanitized + '-1'

        # Ensure minimum length
        if len(sanitized) < 3:
            sanitized = f"run-{sanitized}-{int(time.time())}"

        return sanitized

    async def stitch_and_upload_timelapse(self, experiment_id: str,
                                        upload_immediately: bool = True,
                                        cleanup_temp_files: bool = True,
                                        max_concurrent_runs: int = 1,
                                        use_parallel_wells: bool = True) -> dict:
        """
        Parallel stitching and uploading
        
        Args:
            experiment_id: Experiment ID to search for
            upload_immediately: Whether to upload each well after stitching
            cleanup_temp_files: Whether to delete temporary files after upload
            max_concurrent_runs: Not used (kept for compatibility)
            use_parallel_wells: Whether to process wells in parallel (3 at a time)
            
        Returns:
            Dictionary with processing results
        """
        results = {
            "success": True,
            "experiment_id": experiment_id,
            "processed_runs": [],
            "failed_runs": [],
            "total_datasets": 0,
            "total_size_mb": 0,
            "start_time": time.time(),
            "processing_mode": "parallel_wells" if use_parallel_wells else "sequential_wells"
        }

        try:
            print("=" * 60)
            if use_parallel_wells:
                print(f"PARALLEL PROCESSING STARTED for experiment_id: {experiment_id}")
                print("Mode: One folder at a time, 3 wells at a time")
            else:
                print(f"SEQUENTIAL PROCESSING STARTED for experiment_id: {experiment_id}")
                print("Mode: One folder at a time, one well at a time")
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

            # Process each experiment folder (each folder = one dataset)
            for folder_index, exp_folder in enumerate(experiment_folders):
                print(f"\n📁 Processing folder {folder_index + 1}/{len(experiment_folders)}: {exp_folder.name}")

                # Choose processing method based on use_parallel_wells flag
                if use_parallel_wells:
                    run_result = await self.process_experiment_run_parallel(
                        exp_folder, upload_immediately, cleanup_temp_files, experiment_id
                    )
                else:
                    run_result = await self.process_experiment_run_sequential(
                        exp_folder, upload_immediately, cleanup_temp_files, experiment_id
                    )

                if run_result["success"]:
                    results["processed_runs"].append(run_result)
                    results["total_datasets"] += 1  # Each folder = one dataset
                    results["total_size_mb"] += run_result.get("total_size_mb", 0)
                    print(f"✅ Folder {exp_folder.name} completed successfully")
                    print(f"   Dataset: {run_result.get('dataset_name', 'Unknown')}")
                    print(f"   Wells: {run_result.get('wells_processed', 0)}")
                else:
                    results["failed_runs"].append(run_result)
                    print(f"❌ Folder {exp_folder.name} failed: {run_result.get('error', 'Unknown error')}")

                    # Stop processing if upload failed and upload_immediately is True
                    if upload_immediately and "upload" in run_result.get('error', '').lower():
                        print(f"🛑 Stopping processing due to upload failure in folder {exp_folder.name}")
                        results["success"] = False
                        results["message"] = f"Processing stopped due to upload failure in folder {exp_folder.name}"
                        break

            results["processing_time_seconds"] = time.time() - results["start_time"]

            processing_mode = "Parallel" if use_parallel_wells else "Sequential"
            self.logger.info(f"{processing_mode} processing completed: {results['total_datasets']} datasets created, "
                           f"{len(results['failed_runs'])} folder failures, "
                           f"{results['total_size_mb']:.2f} MB total")

        except Exception as e:
            results["success"] = False
            results["error"] = str(e)
            self.logger.error(f"Processing failed: {e}")

        return results

    def _cleanup_existing_temp_folders(self, experiment_folder_name: str):
        """
        Clean up any existing temporary offline_stitch folders for this experiment.
        
        Args:
            experiment_folder_name: Name of the experiment folder to clean up temp folders for
        """
        try:
            from squid_control.control.config import CONFIG

            if CONFIG.DEFAULT_SAVING_PATH and Path(CONFIG.DEFAULT_SAVING_PATH).exists():
                base_temp_path = Path(CONFIG.DEFAULT_SAVING_PATH)

                # Find all offline_stitch folders for this experiment
                pattern = f"offline_stitch_{experiment_folder_name}_*"
                temp_folders = list(base_temp_path.glob(pattern))

                if temp_folders:
                    print(f"🧹 Found {len(temp_folders)} temporary folders to clean up:")
                    for temp_folder in temp_folders:
                        try:
                            if temp_folder.is_dir():
                                shutil.rmtree(temp_folder, ignore_errors=True)
                                print(f"  🗑️ Cleaned up: {temp_folder.name}")
                        except Exception as e:
                            print(f"  ⚠️ Failed to cleanup {temp_folder.name}: {e}")
                else:
                    print(f"  ✅ No temporary folders found for {experiment_folder_name}")
            else:
                print("  ⚠️ Cannot cleanup temp folders: CONFIG.DEFAULT_SAVING_PATH not available")

        except Exception as e:
            print(f"  ⚠️ Error during temp folder cleanup: {e}")

    async def process_experiment_run_parallel(self, experiment_folder: Path,
                                            upload_immediately: bool = True,
                                            cleanup_temp_files: bool = True,
                                            experiment_id: str = None) -> dict:
        """
        Process a single experiment run - all images into a single canvas using absolute stage coordinates.
        
        Args:
            experiment_folder: Path to experiment folder
            upload_immediately: Whether to upload the dataset after processing
            cleanup_temp_files: Whether to cleanup temp files
            experiment_id: Experiment ID for naming
            
        Returns:
            Dictionary with processing results
        """
        self.logger.info(f"Processing experiment folder: {experiment_folder.name}")

        try:
            # 1. Parse metadata from this folder
            print(f"📋 Reading metadata from {experiment_folder.name}...")
            acquisition_params = self.parse_acquisition_parameters(experiment_folder)
            xml_channels = self.parse_configurations_xml(experiment_folder)
            channel_mapping = self.create_xml_to_channel_mapping(xml_channels)
            coordinates_data = self.parse_coordinates_csv(experiment_folder)

            # Flatten all positions from all wells into a single list
            all_positions = []
            for well_id, well_data in coordinates_data.items():
                for coord_record in well_data:
                    coord_record['well_id'] = well_id  # Keep track of original well for logging
                    all_positions.append(coord_record)

            print(f"Found {len(all_positions)} positions to process from {len(coordinates_data)} regions")

            # 2. Create/find stitch folder - use consistent name without timestamp for resume capability
            from squid_control.control.config import CONFIG

            if CONFIG.DEFAULT_SAVING_PATH and Path(CONFIG.DEFAULT_SAVING_PATH).exists():
                base_temp_path = Path(CONFIG.DEFAULT_SAVING_PATH)
                # Use consistent folder name (no timestamp) so we can resume if upload fails
                temp_path = base_temp_path / f"offline_stitch_{experiment_folder.name}"
            else:
                # For system temp, we can't easily resume, so use timestamp
                temp_path = Path(tempfile.mkdtemp(prefix=f"offline_stitch_{experiment_folder.name}_"))
            
            # Check if .done file exists - if so, skip zarr creation and just upload
            done_file = temp_path / ".done"
            zarr_already_created = done_file.exists()
            
            if zarr_already_created:
                print(f"✅ Found .done file - zarr already created, skipping to upload")
                print(f"   Using existing zarr at: {temp_path}")
                
                # Load existing canvas
                temp_exp_manager = self.create_temp_experiment_manager(str(temp_path))
                exp_name = experiment_folder.name
                # Don't create new experiment, just get existing canvas
                temp_exp_manager.current_experiment = exp_name
                canvas = temp_exp_manager.get_canvas(exp_name)
                
                # Get canvas info
                canvas_info = canvas.get_export_info()
                total_size_mb = canvas_info.get('total_size_mb', 0)
                images_processed = len(all_positions)
            else:
                # Create new zarr
                temp_path.mkdir(parents=True, exist_ok=True)
                print(f"Using stitch folder: {temp_path}")
                
                temp_exp_manager = self.create_temp_experiment_manager(str(temp_path))

                # Create a single experiment with one canvas
                exp_name = experiment_folder.name
                temp_exp_manager.create_experiment(exp_name)
                canvas = temp_exp_manager.get_canvas(exp_name)

                # 3. Start stitching before processing images
                print(f"🚀 Starting stitching pipeline...")
                await canvas.start_stitching()
                print(f"✅ Stitching pipeline started")

                # 4. Process all positions into the single canvas using absolute stage coordinates
                print(f"🚀 Processing {len(all_positions)} positions into single canvas...")

                # Run the image loading in a thread pool so it doesn't block the event loop
                # This allows the stitching background tasks to run while we load images
                import concurrent.futures
                loop = asyncio.get_event_loop()
                
                with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
                    # Run the sync function in a separate thread
                    await loop.run_in_executor(
                        executor,
                        self._load_and_stitch_well_images_sync,
                        all_positions, experiment_folder, canvas, channel_mapping
                    )

                # 5. Wait for all images to be processed
                print(f"⏳ Waiting for stitching queue to drain...")
                await self._wait_for_stitching_completion(canvas, timeout_seconds=600)  # 10 minute timeout for large datasets
                print(f"✅ All images processed")

                # Get canvas size
                canvas_info = canvas.get_export_info()
                total_size_mb = canvas_info.get('total_size_mb', 0)
                images_processed = len(all_positions)
                
                # Create .done file to mark zarr creation complete
                done_file.write_text(f"Zarr creation completed at {time.strftime('%Y-%m-%d %H:%M:%S')}\n"
                                    f"Positions: {images_processed}\n"
                                    f"Size: {total_size_mb:.2f} MB\n")
                print(f"✅ Created .done file - zarr creation complete")

            print(f"🎉 Processing complete: {images_processed} positions processed")

            # Create dataset name using normalized timestamp extraction
            dataset_name = self.create_normalized_dataset_name(experiment_folder, experiment_id)
            print(f"📝 Using dataset name: {dataset_name}")

            # 4. Upload experiment zarr data to artifact manager using hypha-artifact
            upload_result = None
            if images_processed > 0 and upload_immediately:
                print(f"\n📦 Uploading experiment zarr data to artifact manager...")

                try:
                    # Get the zarr folder path from the canvas
                    zarr_folder = canvas.zarr_path
                    upload_result = await self._upload_zarr_with_hypha_artifact(
                        zarr_folder_path=str(zarr_folder),
                        dataset_name=dataset_name,
                        acquisition_settings={
                            "microscope_service_id": self.service_id,
                            "experiment_name": experiment_folder.name,
                            "total_positions": images_processed,
                            "total_size_mb": total_size_mb,
                            "offline_processing": True
                        },
                        description=f"Offline processed experiment: {experiment_folder.name} with {images_processed} positions"
                    )

                    print(f"  ✅ Dataset upload complete: {upload_result.get('dataset_name')}")

                    # Clean up stitch folder only after successful upload
                    if cleanup_temp_files:
                        shutil.rmtree(temp_path, ignore_errors=True)
                        self.logger.debug(f"Cleaned up temporary files: {temp_path}")
                        print(f"  🗑️ Cleaned up stitch folder: {temp_path}")

                except Exception as upload_error:
                    print(f"  ❌ Dataset upload failed: {upload_error}")
                    print(f"  💡 Zarr data preserved at: {temp_path}")
                    print(f"  💡 Run again to retry upload (will skip zarr creation)")
                    upload_result = None

            return {
                "success": True,
                "experiment_folder": experiment_folder.name,
                "positions_processed": images_processed,
                "total_size_mb": total_size_mb,
                "dataset_name": dataset_name,
                "upload_result": upload_result
            }

        except Exception as e:
            self.logger.error(f"Error in processing {experiment_folder.name}: {e}")
            import traceback
            traceback.print_exc()
            return {
                "success": False,
                "experiment_folder": experiment_folder.name,
                "error": str(e)
            }

    async def process_experiment_run_sequential(self, experiment_folder: Path,
                                              upload_immediately: bool = True,
                                              cleanup_temp_files: bool = True,
                                              experiment_id: str = None) -> dict:
        """
        Process a single experiment run (sequential mode).
        
        This method is kept for backward compatibility. It now uses the same
        single-canvas approach as process_experiment_run_parallel.
        
        Args:
            experiment_folder: Path to experiment folder
            upload_immediately: Whether to upload the dataset after processing
            cleanup_temp_files: Whether to cleanup temp files
            experiment_id: Experiment ID for naming
            
        Returns:
            Dictionary with processing results
        """
        # Delegate to the parallel method which now uses a single canvas
        return await self.process_experiment_run_parallel(
            experiment_folder=experiment_folder,
            upload_immediately=upload_immediately,
            cleanup_temp_files=cleanup_temp_files,
            experiment_id=experiment_id
        )


    async def _wait_for_stitching_completion(self, canvas, timeout_seconds=60):
        """
        Wait for stitching to complete properly with timeout and progress monitoring.
        
        Args:
            canvas: WellZarrCanvas instance
            timeout_seconds: Maximum time to wait for stitching completion
        """
        start_time = time.time()
        last_queue_size = -1
        empty_queue_count = 0  # Count consecutive empty queue checks

        while time.time() - start_time < timeout_seconds:
            # Check if stitching is still active
            if not canvas.is_stitching:
                break

            # Check queue size
            current_queue_size = canvas.preprocessing_queue.qsize()

            # Log progress if queue size changed
            if current_queue_size != last_queue_size:
                if current_queue_size > 0:
                    print(f"Stitching queue has {current_queue_size} images remaining...")
                    empty_queue_count = 0  # Reset counter when queue has items
                last_queue_size = current_queue_size

            # If queue is empty, wait longer to ensure all zarr writes complete
            if current_queue_size == 0:
                empty_queue_count += 1
                print(f"Queue empty, waiting for zarr writes to complete... (check {empty_queue_count})")

                # Wait longer for zarr writes to complete
                await asyncio.sleep(2.0)  # Increased from 0.5 to 2.0 seconds

                # Only exit after 3 consecutive empty checks (6 seconds total)
                if empty_queue_count >= 3:
                    print("Queue empty for 3 consecutive checks, stitching should be complete")
                    break
            else:
                empty_queue_count = 0  # Reset counter when queue has items

            # Wait before checking again
            await asyncio.sleep(0.5)  # Increased from 0.1 to 0.5 seconds

        # Final check
        final_queue_size = canvas.preprocessing_queue.qsize()
        if final_queue_size > 0:
            print(f"Warning: {final_queue_size} images still in stitching queue after timeout")
        else:
            print("Stitching queue is completely empty")

        # Additional wait for zarr writes to complete
        print("Waiting additional 3 seconds for zarr writes to complete...")
        await asyncio.sleep(3.0)

        elapsed_time = time.time() - start_time
        print(f"Stitching completion wait took {elapsed_time:.2f} seconds")

    async def _add_image_to_stitching_queue(self, canvas, image, x_mm, y_mm, zarr_channel_idx, z_idx=0, timepoint=0):
        """
        Add image to stitching queue using the exact same pattern as scan_region_to_zarr.
        This ensures proper processing with all scales and coordinate conversion.
        
        Args:
            canvas: WellZarrCanvas instance
            image: Image array
            x_mm, y_mm: Absolute coordinates (like scan_region_to_zarr)
            zarr_channel_idx: Local zarr channel index
            z_idx: Z-slice index (default 0)
            timepoint: Timepoint index (default 0)
        """
        try:
            # Add to stitching queue with normal scan flag (all scales) - same as scan_region_to_zarr
            queue_item = {
                'image': image.copy(),
                'x_mm': x_mm,  # Use absolute coordinates - WellZarrCanvas will convert to well-relative
                'y_mm': y_mm,  # Use absolute coordinates - WellZarrCanvas will convert to well-relative
                'channel_idx': zarr_channel_idx,
                'z_idx': z_idx,
                'timepoint': timepoint,
                'timestamp': time.time(),
                'quick_scan': False  # Flag to indicate this is normal scan (all scales)
            }

            # Check queue size before adding
            queue_size_before = canvas.preprocessing_queue.qsize()
            await canvas.preprocessing_queue.put(queue_item)
            queue_size_after = canvas.preprocessing_queue.qsize()


            return "Queued for stitching (all scales)"

        except Exception as e:
            print(f"    ❌ Failed to add image to stitching queue: {e}")
            return f"Failed to queue: {e}"

    async def _add_image_with_backpressure(self, canvas, image, x_mm, y_mm, zarr_channel_idx, max_queue_size=100):
        """
        Add image to canvas with backpressure to prevent queue overflow.
        
        Args:
            canvas: WellZarrCanvas instance
            image: Image array
            x_mm, y_mm: Coordinates
            zarr_channel_idx: Channel index
            max_queue_size: Maximum queue size before applying backpressure
        """
        # Check queue size and apply backpressure if needed
        queue_size = canvas.preprocessing_queue.qsize()

        if queue_size > max_queue_size:
            # Wait for queue to drain a bit
            print(f"Stitching queue full ({queue_size} items), waiting for processing...")
            while canvas.preprocessing_queue.qsize() > max_queue_size // 2:
                await asyncio.sleep(0.1)

        # Add image to canvas
        await canvas.add_image_async(
            image, x_mm - canvas.well_center_x, y_mm - canvas.well_center_y,
            zarr_channel_idx, 0, 0  # z_idx=0, timepoint=0
        )

    def _export_well_to_zip_direct(self, canvas, filename: str) -> str:
        """
        Export a well canvas to a ZIP file on disk using CONFIG.DEFAULT_SAVING_PATH.
        
        Args:
            canvas: WellZarrCanvas instance
            filename: Desired filename (e.g., 'well_A1_96.zip')
            
        Returns:
            str: Path to the created ZIP file
        """
        from pathlib import Path

        from squid_control.control.config import CONFIG

        # Use CONFIG.DEFAULT_SAVING_PATH for output directory
        try:
            output_dir = Path(CONFIG.DEFAULT_SAVING_PATH) / "well_zips"
            output_dir.mkdir(parents=True, exist_ok=True)
            self.logger.info(f"Using output directory: {output_dir}")
        except Exception as e:
            # Fallback to temp directory if CONFIG not available
            import tempfile
            output_dir = Path(tempfile.gettempdir()) / "well_zips"
            output_dir.mkdir(parents=True, exist_ok=True)
            self.logger.warning(f"Could not use CONFIG.DEFAULT_SAVING_PATH, using temp: {e}")

        # Create full path for the ZIP file
        zip_file_path = output_dir / filename

        # Clean up any partial Zarr files before export
        if hasattr(canvas, 'zarr_path') and canvas.zarr_path.exists():
            import glob
            partial_files = glob.glob(str(canvas.zarr_path / "**" / "*.partial"), recursive=True)
            for partial_file in partial_files:
                try:
                    os.remove(partial_file)
                except:
                    pass

        # Export canvas to this specific file path
        self.logger.info(f"Exporting well canvas to: {zip_file_path}")
        canvas.export_to_zip(str(zip_file_path))

        return str(zip_file_path)

    async def _upload_existing_wells_from_directory(self, well_zips_path: Path, experiment_folder: Path,
                                                   experiment_id: str, upload_immediately: bool,
                                                   cleanup_temp_files: bool) -> dict:
        """
        DEPRECATED: This method is no longer used as ZIP file workflow has been removed.
        Zarr data is now uploaded directly using hypha-artifact.
        
        Args:
            well_zips_path: Path to directory containing well ZIP files
            experiment_folder: Original experiment folder for metadata
            experiment_id: Experiment ID for upload
            upload_immediately: Whether to upload immediately
            cleanup_temp_files: Whether to cleanup temp files
            
        Returns:
            Dictionary with processing results indicating deprecation
        """
        print("   Zarr data is now uploaded directly using hypha-artifact.")
        print("   Please re-process the experiment to generate zarr data for upload.")
        
        return {
            "success": False,
            "experiment_folder": experiment_folder.name,
            "error": "ZIP file upload workflow has been deprecated. Please re-process the experiment.",
            "deprecated": True
        }



def create_optimized_processor(squid_controller, zarr_artifact_manager=None, service_id=None,
                              max_concurrent_wells=3, image_batch_size=5):
    """
    Create an optimized OfflineProcessor instance with performance tuning.
    
    Args:
        squid_controller: SquidController instance
        zarr_artifact_manager: ZarrArtifactManager instance
        service_id: Service ID for uploads
        max_concurrent_wells: Maximum number of wells to process concurrently (default: 3)
        image_batch_size: Number of images to process in each batch (default: 10)
        
    Returns:
        Optimized OfflineProcessor instance
    """
    return OfflineProcessor(
        squid_controller=squid_controller,
        zarr_artifact_manager=zarr_artifact_manager,
        service_id=service_id,
        max_concurrent_wells=max_concurrent_wells,
        image_batch_size=image_batch_size
    )


def create_high_performance_processor(squid_controller, zarr_artifact_manager=None, service_id=None):
    """
    Create a high-performance OfflineProcessor instance optimized for speed.
    
    This configuration prioritizes speed over memory usage.
    
    Args:
        squid_controller: SquidController instance
        zarr_artifact_manager: ZarrArtifactManager instance
        service_id: Service ID for uploads
        
    Returns:
        High-performance OfflineProcessor instance
    """
    return OfflineProcessor(
        squid_controller=squid_controller,
        zarr_artifact_manager=zarr_artifact_manager,
        service_id=service_id,
        max_concurrent_wells=4,  # Higher concurrency
        image_batch_size=10      # Larger batches
    )


def create_memory_efficient_processor(squid_controller, zarr_artifact_manager=None, service_id=None):
    """
    Create a memory-efficient OfflineProcessor instance optimized for low memory usage.
    
    This configuration prioritizes memory usage over speed.
    
    Args:
        squid_controller: SquidController instance
        zarr_artifact_manager: ZarrArtifactManager instance
        service_id: Service ID for uploads
        
    Returns:
        Memory-efficient OfflineProcessor instance
    """
    return OfflineProcessor(
        squid_controller=squid_controller,
        zarr_artifact_manager=zarr_artifact_manager,
        service_id=service_id,
        max_concurrent_wells=1,  # Lower concurrency
        image_batch_size=5       # Smaller batches
    )
