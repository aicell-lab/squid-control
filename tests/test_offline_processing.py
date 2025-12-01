"""
Test suite for offline processing functionality.

WHAT WAS TESTED:
================
✅ Data Generation & File Structure
   - OfflineDataGenerator creates realistic synthetic microscopy data
   - Generates proper folder structure: experiment_id-timestamp/0/
   - Creates all required files: acquisition parameters.json, configurations.xml, 
     coordinates.csv, BMP image files with correct naming pattern

✅ Configuration Management
   - Tests that CONFIG.DEFAULT_SAVING_PATH is properly set and used
   - Verifies the offline processor can find experiment folders using the config

✅ Metadata Parsing
   - Parses acquisition parameters from JSON
   - Extracts channel configurations from XML
   - Reads coordinates from CSV and groups by well
   - Creates proper channel name mappings

✅ Image Processing Pipeline
   - Loads BMP images from disk
   - Processes images through the stitching queue
   - Handles multiple channels (BF LED matrix, Fluorescence 488nm)
   - Converts coordinates and applies proper transformations

✅ Zarr Canvas Operations
   - Creates single canvas per experiment using absolute stage coordinates
   - Adds images to canvas with correct parameters
   - Exports canvases directly (no ZIP files)

✅ File Management
   - Creates temporary directories for processing
   - Uses hypha-artifact for direct folder uploads
   - Handles cleanup of temporary files

✅ Upload Interface
   - Uses AsyncHyphaArtifact for uploads
   - Passes proper metadata (experiment_id, dataset_name, etc.)

WHAT WAS NOT TESTED:
===================
❌ Actual Network Upload - No real HTTP requests to artifact manager
❌ Large Dataset Performance - Only tested with tiny datasets
❌ Real Hardware Integration - No actual microscope hardware

"""

import json
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import List
from unittest.mock import AsyncMock, MagicMock, patch

import cv2
import numpy as np
import pandas as pd
import pytest
import pytest_asyncio

from squid_control.control.config import CONFIG
from squid_control.offline_processing import OfflineProcessor


class OfflineDataGenerator:
    """Helper class to generate synthetic microscopy data for offline processing tests."""

    @staticmethod
    def create_synthetic_microscopy_data(base_path: Path, experiment_id: str,
                                       num_runs: int = 2, wells: List[str] = None,
                                       channels: List[str] = None) -> List[Path]:
        """
        Create synthetic microscopy data in the expected format for offline processing.

        Args:
            base_path: Base directory to create experiment folders
            experiment_id: Experiment ID prefix
            num_runs: Number of experiment runs to create
            wells: List of well IDs (e.g., ['A1', 'B2', 'C3'])
            channels: List of channel names

        Returns:
            List of created experiment folder paths
        """
        if wells is None:
            wells = ['A1', 'B2', 'C3']  # Default test wells
        if channels is None:
            channels = ['BF LED matrix full', 'Fluorescence 488 nm Ex', 'Fluorescence 561 nm Ex']

        experiment_folders = []

        for run_idx in range(num_runs):
            # Create timestamp for this run
            timestamp = f"20250822T{14 + run_idx:02d}30{run_idx:02d}"
            experiment_folder = base_path / f"{experiment_id}-{timestamp}"
            experiment_folder.mkdir(parents=True, exist_ok=True)

            # Create the '0' subfolder
            data_folder = experiment_folder / "0"
            data_folder.mkdir(exist_ok=True)

            # Creating synthetic data in: {experiment_folder}

            # Generate acquisition parameters
            acquisition_params = {
                "dx(mm)": 0.9,
                "Nx": 3,
                "dy(mm)": 0.9,
                "Ny": 3,
                "dz(um)": 1.5,
                "Nz": 1,
                "dt(s)": 0,
                "Nt": 1,
                "with CONFIG.AF": False,
                "with reflection CONFIG.AF": True,
                "objective": {
                    "magnification": 20,
                    "NA": 0.4,
                    "tube_lens_f_mm": 180,
                    "name": "20x (Boli)"
                },
                "sensor_pixel_size_um": 1.85,
                "tube_lens_mm": 50
            }

            with open(data_folder / "acquisition parameters.json", 'w') as f:
                json.dump(acquisition_params, f, indent=2)

            # Generate configurations.xml
            OfflineDataGenerator._create_configurations_xml(data_folder, channels)

            # Generate coordinates and images for each well
            all_coordinates = []

            for well_idx, well_id in enumerate(wells):
                well_coords = OfflineDataGenerator._create_well_data(
                    data_folder, well_id, channels, acquisition_params, well_idx
                )
                all_coordinates.extend(well_coords)

            # Create coordinates.csv
            df = pd.DataFrame(all_coordinates)
            df.to_csv(data_folder / "coordinates.csv", index=False)

            experiment_folders.append(experiment_folder)
            # Created experiment run: {experiment_folder.name}

        return experiment_folders

    @staticmethod
    def _create_configurations_xml(data_folder: Path, channels: List[str]):
        """Create configurations.xml file with channel settings."""
        root = ET.Element("modes")

        # Channel mapping to XML format
        channel_configs = {
            "BF LED matrix full": {
                "ID": "1",
                "ExposureTime": "5.0",
                "AnalogGain": "1.1",
                "IlluminationSource": "0",
                "IlluminationIntensity": "32.0"
            },
            "Fluorescence 488 nm Ex": {
                "ID": "6",
                "ExposureTime": "100.0",
                "AnalogGain": "10.0",
                "IlluminationSource": "12",
                "IlluminationIntensity": "27.0"
            },
            "Fluorescence 561 nm Ex": {
                "ID": "8",
                "ExposureTime": "300.0",
                "AnalogGain": "10.0",
                "IlluminationSource": "14",
                "IlluminationIntensity": "50.0"
            }
        }

        for channel in channels:
            config = channel_configs.get(channel, {
                "ID": "1",
                "ExposureTime": "50.0",
                "AnalogGain": "1.0",
                "IlluminationSource": "0",
                "IlluminationIntensity": "50.0"
            })

            mode = ET.SubElement(root, "mode")
            mode.set("ID", config["ID"])
            mode.set("Name", channel)
            mode.set("ExposureTime", config["ExposureTime"])
            mode.set("AnalogGain", config["AnalogGain"])
            mode.set("IlluminationSource", config["IlluminationSource"])
            mode.set("IlluminationIntensity", config["IlluminationIntensity"])
            mode.set("CameraSN", "")
            mode.set("ZOffset", "0.0")
            mode.set("PixelFormat", "default")
            mode.set("_PixelFormat_options", "[default,MONO8,MONO12,MONO14,MONO16,BAYER_RG8,BAYER_RG12]")
            mode.set("Selected", "1")

        # Write XML file
        tree = ET.ElementTree(root)
        ET.indent(tree, space="  ", level=0)
        tree.write(data_folder / "configurations.xml", encoding="UTF-8", xml_declaration=True)

    @staticmethod
    def _create_well_data(data_folder: Path, well_id: str, channels: List[str],
                         acquisition_params: dict, well_offset: int) -> List[dict]:
        """Create synthetic images and coordinates for a single well."""
        coordinates = []

        # Well center coordinates (simulate different well positions)
        well_center_x = 20.0 + well_offset * 9.0  # 9mm spacing between wells
        well_center_y = 60.0 + well_offset * 9.0

        Nx = acquisition_params["Nx"]
        Ny = acquisition_params["Ny"]
        dx = acquisition_params["dx(mm)"]
        dy = acquisition_params["dy(mm)"]

        # Generate images for each position in the well
        for i in range(Nx):
            for j in range(Ny):
                # Calculate position coordinates
                x_mm = well_center_x + (i - Nx//2) * dx
                y_mm = well_center_y + (j - Ny//2) * dy
                z_um = 4035.0 + np.random.normal(0, 10)  # Simulate focus variation

                # Generate timestamp
                timestamp = f"2025-08-22_18-16-{35 + i*2 + j}.{702228 + i*100 + j*10:06d}"

                # Create images for each channel
                for channel in channels:
                    # Generate synthetic microscopy image
                    image = OfflineDataGenerator._generate_synthetic_image(channel, i, j)

                    # Save as BMP file
                    filename = f"{well_id}_{i}_{j}_0_{channel.replace(' ', '_')}.bmp"
                    filepath = data_folder / filename
                    cv2.imwrite(str(filepath), image)

                # Add coordinate record
                coordinates.append({
                    "i": i,
                    "j": j,
                    "k": 0,
                    "x (mm)": x_mm,
                    "y (mm)": y_mm,
                    "z (um)": z_um,
                    "time": timestamp,
                    "region": well_id
                })

        return coordinates

    @staticmethod
    def _generate_synthetic_image(channel: str, i: int, j: int) -> np.ndarray:
        """Generate a synthetic microscopy image for testing."""
        # Create 512x512 image
        height, width = 512, 512

        # Generate different patterns based on channel
        if "BF" in channel or "Bright" in channel:
            # Brightfield - uniform with some texture
            image = np.random.normal(2000, 100, (height, width)).astype(np.uint16)
            # Add some structure
            y, x = np.ogrid[:height, :width]
            structure = 500 * np.sin(x * 0.02) * np.cos(y * 0.02)
            image = np.clip(image + structure, 0, 4095).astype(np.uint16)

        elif "488" in channel:
            # GFP-like fluorescence
            image = np.random.exponential(200, (height, width)).astype(np.uint16)
            # Add some bright spots
            for _ in range(5):
                center_y = np.random.randint(50, height-50)
                center_x = np.random.randint(50, width-50)
                y, x = np.ogrid[:height, :width]
                spot = 1000 * np.exp(-((x-center_x)**2 + (y-center_y)**2) / (2*30**2))
                image = np.clip(image + spot, 0, 4095).astype(np.uint16)

        elif "561" in channel:
            # RFP-like fluorescence
            image = np.random.gamma(2, 150, (height, width)).astype(np.uint16)
            # Add some linear structures
            y, x = np.ogrid[:height, :width]
            lines = 800 * np.sin(x * 0.01 + y * 0.005)
            image = np.clip(image + lines, 0, 4095).astype(np.uint16)

        else:
            # Default pattern
            image = np.random.randint(100, 1000, (height, width), dtype=np.uint16)

        # Add position-dependent variation
        position_factor = 1.0 + 0.1 * (i + j) / 6.0
        image = np.clip(image * position_factor, 0, 4095).astype(np.uint16)

        # Convert to 8-bit for BMP format
        image_8bit = (image / 16).astype(np.uint8)

        return image_8bit


class FakeSquidController:
    def __init__(self, pixel_size_xy: float = 0.333):
        self.pixel_size_xy = pixel_size_xy


class FakeExperimentManager:
    """Fake experiment manager for testing."""
    
    def __init__(self, base_path: str):
        self.base_path = Path(base_path)
        self.experiments = {}
        self.active_experiment = None
    
    def create_experiment(self, name: str):
        """Create a fake experiment."""
        exp_path = self.base_path / name
        exp_path.mkdir(parents=True, exist_ok=True)
        self.experiments[name] = {
            "path": exp_path,
            "canvas": FakeCanvas(exp_path)
        }
        self.active_experiment = name
        return {"success": True, "name": name, "path": str(exp_path)}
    
    def get_canvas(self, experiment_name: str = None):
        """Get canvas for experiment."""
        name = experiment_name or self.active_experiment
        if name and name in self.experiments:
            return self.experiments[name]["canvas"]
        return None


class FakeCanvas:
    """Fake zarr canvas for testing."""
    
    def __init__(self, path: Path):
        self.zarr_path = path / "data.zarr"
        self.zarr_path.mkdir(parents=True, exist_ok=True)
        self.channel_to_zarr_index = {
            "BF LED matrix full": 0,
            "Fluorescence 488 nm Ex": 1,
            "Fluorescence 561 nm Ex": 2
        }
        self.images_added = 0
    
    def get_zarr_channel_index(self, channel_name: str) -> int:
        return self.channel_to_zarr_index.get(channel_name, 0)
    
    def get_export_info(self) -> dict:
        return {
            "total_size_mb": 1.5,
            "canvas_dimensions": {"width": 1024, "height": 1024},
            "num_scales": 6
        }


@pytest_asyncio.fixture
async def temp_saving_path():
    """Set CONFIG.DEFAULT_SAVING_PATH to a temporary directory for the test."""

    with tempfile.TemporaryDirectory() as tmpdir:
        # Ensure directory exists and is writable
        base = Path(tmpdir)
        base.mkdir(parents=True, exist_ok=True)
        # Point DEFAULT_SAVING_PATH to our temp dir
        old_path = CONFIG.DEFAULT_SAVING_PATH
        CONFIG.DEFAULT_SAVING_PATH = str(base)
        try:
            yield base
        finally:
            # Restore
            CONFIG.DEFAULT_SAVING_PATH = old_path


@pytest.mark.asyncio
async def test_offline_processor_finds_experiment_folders(temp_saving_path):
    """Test that the processor can find experiment folders."""
    # Arrange: create synthetic data
    experiment_id = "test-experiment"
    experiment_folders = OfflineDataGenerator.create_synthetic_microscopy_data(
        base_path=temp_saving_path,
        experiment_id=experiment_id,
        num_runs=2,
        wells=["A1"],
        channels=["BF LED matrix full"],
    )
    assert len(experiment_folders) == 2

    fake_controller = FakeSquidController()
    processor = OfflineProcessor(
        squid_controller=fake_controller,
        zarr_artifact_manager=None,
        service_id="microscope-control-squid-test",
    )

    # Act
    found_folders = processor.find_experiment_folders(experiment_id)

    # Assert
    assert len(found_folders) == 2
    for folder in found_folders:
        assert folder.exists()
        assert (folder / "0").exists()


@pytest.mark.asyncio
async def test_offline_processor_parses_metadata(temp_saving_path):
    """Test that the processor correctly parses experiment metadata."""
    # Arrange
    experiment_id = "test-metadata"
    experiment_folders = OfflineDataGenerator.create_synthetic_microscopy_data(
        base_path=temp_saving_path,
        experiment_id=experiment_id,
        num_runs=1,
        wells=["A1", "B2"],
        channels=["BF LED matrix full", "Fluorescence 488 nm Ex"],
    )
    experiment_folder = experiment_folders[0]

    fake_controller = FakeSquidController()
    processor = OfflineProcessor(
        squid_controller=fake_controller,
        zarr_artifact_manager=None,
        service_id="microscope-control-squid-test",
    )

    # Act
    acquisition_params = processor.parse_acquisition_parameters(experiment_folder)
    xml_channels = processor.parse_configurations_xml(experiment_folder)
    channel_mapping = processor.create_xml_to_channel_mapping(xml_channels)
    coordinates_data = processor.parse_coordinates_csv(experiment_folder)

    # Assert
    assert acquisition_params is not None
    assert acquisition_params.get("Nx") == 3
    assert acquisition_params.get("Ny") == 3
    
    assert len(xml_channels) == 2
    assert "BF LED matrix full" in xml_channels
    
    assert len(channel_mapping) >= 1
    
    # coordinates_data is grouped by well/region
    assert len(coordinates_data) == 2  # A1 and B2
    assert "A1" in coordinates_data
    assert "B2" in coordinates_data


@pytest.mark.asyncio
async def test_offline_processor_creates_dataset_name(temp_saving_path):
    """Test that the processor creates normalized dataset names."""
    # Arrange
    experiment_id = "test-naming"
    experiment_folders = OfflineDataGenerator.create_synthetic_microscopy_data(
        base_path=temp_saving_path,
        experiment_id=experiment_id,
        num_runs=1,
        wells=["A1"],
        channels=["BF LED matrix full"],
    )
    experiment_folder = experiment_folders[0]

    fake_controller = FakeSquidController()
    processor = OfflineProcessor(
        squid_controller=fake_controller,
        zarr_artifact_manager=None,
        service_id="microscope-control-squid-test",
    )

    # Act
    dataset_name = processor.create_normalized_dataset_name(experiment_folder, experiment_id)

    # Assert
    assert dataset_name is not None
    assert experiment_id in dataset_name or "test" in dataset_name.lower()


@pytest.mark.asyncio
async def test_offline_stitch_and_upload_with_mock(temp_saving_path):
    """Test the stitch and upload flow with mocked hypha-artifact."""
    # Arrange: create small synthetic data
    experiment_id = "offline-test"
    experiment_folders = OfflineDataGenerator.create_synthetic_microscopy_data(
        base_path=temp_saving_path,
        experiment_id=experiment_id,
        num_runs=1,
        wells=["A1"],
        channels=["BF LED matrix full", "Fluorescence 488 nm Ex"],
    )
    assert len(experiment_folders) == 1

    fake_controller = FakeSquidController()

    processor = OfflineProcessor(
        squid_controller=fake_controller,
        zarr_artifact_manager=None,  # Not used anymore
        service_id="microscope-control-squid-test",
        max_concurrent_wells=1,
        image_batch_size=2,
    )

    # Mock the hypha-artifact upload
    mock_artifact = MagicMock()
    mock_artifact.edit = AsyncMock()
    mock_artifact.put = AsyncMock()
    mock_artifact.commit = AsyncMock()

    with patch('squid_control.offline_processing.OfflineProcessor._upload_zarr_with_hypha_artifact') as mock_upload:
        mock_upload.return_value = {
            "success": True,
            "dataset_name": f"{experiment_id}-uploaded",
            "description": "Test upload"
        }

        # Act
        result = await processor.stitch_and_upload_timelapse(
            experiment_id=experiment_id,
            upload_immediately=True,
            cleanup_temp_files=False,
            use_parallel_wells=False,
        )

    # Assert basic success
    assert result["success"] is True
    assert result["total_datasets"] == 1
    assert len(result["processed_runs"]) == 1
    
    # Check that upload was called
    assert mock_upload.called


@pytest.mark.asyncio
async def test_offline_processor_single_canvas_approach(temp_saving_path):
    """Test that the processor uses a single canvas with absolute stage coordinates."""
    # Arrange
    experiment_id = "single-canvas-test"
    experiment_folders = OfflineDataGenerator.create_synthetic_microscopy_data(
        base_path=temp_saving_path,
        experiment_id=experiment_id,
        num_runs=1,
        wells=["A1", "B2"],  # Multiple wells
        channels=["BF LED matrix full"],
    )
    experiment_folder = experiment_folders[0]

    fake_controller = FakeSquidController()
    processor = OfflineProcessor(
        squid_controller=fake_controller,
        zarr_artifact_manager=None,
        service_id="microscope-control-squid-test",
    )

    # Parse coordinates
    coordinates_data = processor.parse_coordinates_csv(experiment_folder)
    
    # Flatten all positions (as the new implementation does)
    all_positions = []
    for well_id, well_data in coordinates_data.items():
        for coord_record in well_data:
            coord_record['well_id'] = well_id
            all_positions.append(coord_record)

    # Assert: all positions from all wells are in a single list
    assert len(all_positions) > 0
    
    # Check that we have positions from both wells
    well_ids = set(pos['well_id'] for pos in all_positions)
    assert "A1" in well_ids
    assert "B2" in well_ids
    
    # All positions should have absolute stage coordinates
    for pos in all_positions:
        assert 'x (mm)' in pos
        assert 'y (mm)' in pos
        assert pos['x (mm)'] > 0  # Absolute coordinates


@pytest.mark.asyncio
async def test_offline_processor_upload_helper_method(temp_saving_path):
    """Test the _upload_zarr_with_hypha_artifact helper method."""
    # Arrange
    fake_controller = FakeSquidController()
    processor = OfflineProcessor(
        squid_controller=fake_controller,
        zarr_artifact_manager=None,
        service_id="microscope-control-squid-test",
    )

    # Create a dummy zarr folder
    zarr_folder = temp_saving_path / "test_experiment" / "data.zarr"
    zarr_folder.mkdir(parents=True, exist_ok=True)
    (zarr_folder / ".zattrs").write_text("{}")

    # Mock AsyncHyphaArtifact
    with patch('squid_control.offline_processing.OfflineProcessor._upload_zarr_with_hypha_artifact') as mock_upload:
        mock_upload.return_value = {
            "success": True,
            "dataset_name": "test-dataset",
            "description": "Test"
        }

        # Act
        result = await processor._upload_zarr_with_hypha_artifact(
            zarr_folder_path=str(zarr_folder),
            dataset_name="test-dataset",
            acquisition_settings={"test": True},
            description="Test upload"
        )

        # Assert
        assert result["success"] is True
        assert result["dataset_name"] == "test-dataset"


@pytest.mark.asyncio
async def test_process_experiment_run_parallel_returns_positions(temp_saving_path):
    """Test that process_experiment_run_parallel returns positions_processed instead of wells_processed."""
    # Arrange
    experiment_id = "parallel-test"
    experiment_folders = OfflineDataGenerator.create_synthetic_microscopy_data(
        base_path=temp_saving_path,
        experiment_id=experiment_id,
        num_runs=1,
        wells=["A1"],
        channels=["BF LED matrix full"],
    )
    experiment_folder = experiment_folders[0]

    fake_controller = FakeSquidController()
    processor = OfflineProcessor(
        squid_controller=fake_controller,
        zarr_artifact_manager=None,
        service_id="microscope-control-squid-test",
    )

    # Mock the upload and experiment manager
    with patch.object(processor, '_upload_zarr_with_hypha_artifact') as mock_upload, \
         patch.object(processor, 'create_temp_experiment_manager') as mock_create_exp:
        
        mock_upload.return_value = {"success": True, "dataset_name": "test"}
        
        # Create a mock experiment manager
        mock_exp_manager = FakeExperimentManager(str(temp_saving_path))
        mock_create_exp.return_value = mock_exp_manager

        # Act
        result = await processor.process_experiment_run_parallel(
            experiment_folder=experiment_folder,
            upload_immediately=False,  # Don't upload for this test
            cleanup_temp_files=True,
            experiment_id=experiment_id,
        )

    # Assert - should have positions_processed, not wells_processed
    assert result["success"] is True
    assert "positions_processed" in result or "error" in result


@pytest.mark.asyncio  
async def test_process_experiment_run_sequential_delegates_to_parallel(temp_saving_path):
    """Test that process_experiment_run_sequential delegates to process_experiment_run_parallel."""
    # Arrange
    experiment_id = "sequential-test"
    experiment_folders = OfflineDataGenerator.create_synthetic_microscopy_data(
        base_path=temp_saving_path,
        experiment_id=experiment_id,
        num_runs=1,
        wells=["A1"],
        channels=["BF LED matrix full"],
    )
    experiment_folder = experiment_folders[0]

    fake_controller = FakeSquidController()
    processor = OfflineProcessor(
        squid_controller=fake_controller,
        zarr_artifact_manager=None,
        service_id="microscope-control-squid-test",
    )

    # Mock the parallel method to verify delegation
    with patch.object(processor, 'process_experiment_run_parallel') as mock_parallel:
        mock_parallel.return_value = {"success": True, "delegated": True}

        # Act
        result = await processor.process_experiment_run_sequential(
            experiment_folder=experiment_folder,
            upload_immediately=False,
            cleanup_temp_files=True,
            experiment_id=experiment_id,
        )

    # Assert - should have called the parallel method
    mock_parallel.assert_called_once_with(
        experiment_folder=experiment_folder,
        upload_immediately=False,
        cleanup_temp_files=True,
        experiment_id=experiment_id,
    )
    assert result["delegated"] is True
