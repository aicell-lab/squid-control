# Opera Phoenix to Zarr Import System

## Overview

This document explains how the Opera Phoenix high-content screening microscopy data is organized and imported into the Squid microscope's coordinate-based Zarr canvas system.

## Opera Phoenix OME-TIFF Structure

### File Format: Zarr-in-TIFF

The Opera Phoenix exports data as **OME-TIFF with embedded Zarr storage**. This is a special format where:
- The outer container is a TIFF file (`plate1.ome.tif`)
- Inside, images are stored as a Zarr array for efficient random access
- Metadata follows the OME-XML standard

### Array Dimensions

The Zarr array has shape: `(N_tiles, N_channels, tile_height, tile_width)`

For a typical 96-well plate scan:
```
Shape: (2592, 4, 1080, 1080)
       ↓     ↓   ↓     ↓
       tiles ch  height width
```

**Breaking down the 2592 tiles:**
- 96 wells × 9 fields-of-view (FoV) per well × 3 z-planes = 2,592 tiles
- Each well is imaged in a 3×3 grid (9 FoV total)
- Each FoV is captured at 3 focal planes (top, middle, bottom)

### Channel Organization

The Opera dataset contains 4 fluorescence channels:

| Channel Index | Wavelength | Stain | Biological Target |
|--------------|------------|-------|-------------------|
| 0 | 405nm (DAPI) | Blue | Cell nucleus |
| 1 | 488nm | Green | Target protein |
| 2 | 638nm | Red | SARS-CoV-2 Virus |
| 3 | 561nm | Yellow/Orange | Endoplasmic Reticulum (ER) |

## Tile-to-Well Mapping Algorithm

### The Challenge

Opera Phoenix stores tiles sequentially in a 1D array without explicit coordinate metadata. We need to determine:
1. Which well does each tile belong to?
2. Which field-of-view (FoV) within the well?
3. What are the absolute stage coordinates (in mm)?

### Tile Index Decoding

The tiles are organized in a **hierarchical order**:

```python
# Pseudo-code for understanding the tile organization
for each_well in plate (96 wells total):
    for each_fov in 3x3_grid (9 FoV per well):
        for each_z_plane in [0, 1, 2] (3 z-planes):
            tile_index = well_index * 27 + fov_index * 3 + z_plane_index
```

**Example:**
- Well A1, FoV 0, z-plane 0 → tile index 0
- Well A1, FoV 0, z-plane 1 → tile index 1
- Well A1, FoV 0, z-plane 2 → tile index 2
- Well A1, FoV 1, z-plane 0 → tile index 3
- ...
- Well A1, FoV 8, z-plane 2 → tile index 26
- Well A2, FoV 0, z-plane 0 → tile index 27

### Implementation: `get_opera_stage_coordinates()`

```python
def get_opera_stage_coordinates(tile_idx):
    """
    Decode tile index to well position and absolute stage coordinates.
    
    Parameters:
        tile_idx: Sequential tile index (0 to 2591)
    
    Returns:
        (stage_x_mm, stage_y_mm, well_id, fov_idx, z_idx)
    """
    # Constants for 96-well plate with 9 FoV per well
    WELLS_PER_ROW = 12          # 96-well plate: 8 rows × 12 columns
    NUM_ROWS = 8
    FOV_PER_WELL = 9            # 3×3 grid
    Z_PLANES_PER_FOV = 3        # Top, middle, bottom
    
    # Decode hierarchical structure
    tiles_per_well = FOV_PER_WELL * Z_PLANES_PER_FOV  # 27
    
    well_idx = tile_idx // tiles_per_well
    remainder = tile_idx % tiles_per_well
    
    fov_idx = remainder // Z_PLANES_PER_FOV
    z_idx = remainder % Z_PLANES_PER_FOV
    
    # Convert well index to row/column
    row = well_idx // WELLS_PER_ROW
    col = well_idx % WELLS_PER_ROW
    well_id = f"{chr(65 + row)}{col + 1}"  # e.g., "A1", "B6", "H12"
    
    # Calculate absolute stage coordinates
    stage_x_mm, stage_y_mm = calculate_stage_position(row, col, fov_idx)
    
    return stage_x_mm, stage_y_mm, well_id, fov_idx, z_idx
```

## Stage Coordinate Calculation

### Well Plate Geometry

For a standard 96-well plate:
- **Well spacing**: 9.0 mm (center-to-center)
- **Well A1 position**: (12.05 mm, 9.05 mm) - reference origin
- **Layout**: 8 rows (A-H) × 12 columns (1-12)

### Field-of-View (FoV) Grid within Each Well

Each well is imaged in a **3×3 grid of overlapping fields**:

```
Well interior (3×3 FoV grid):

   Column:  0        1        2
   Row 0:  [FoV 0]  [FoV 1]  [FoV 2]
   Row 1:  [FoV 3]  [FoV 4]  [FoV 5]
   Row 2:  [FoV 6]  [FoV 7]  [FoV 8]
```

### FoV Position Calculation

```python
def calculate_stage_position(well_row, well_col, fov_idx):
    """Calculate absolute stage coordinates for a specific FoV."""
    
    # Well plate geometry
    WELL_A1_X = 12.05  # mm
    WELL_A1_Y = 9.05   # mm
    WELL_SPACING = 9.0 # mm
    
    # FoV grid parameters (63X objective, Opera Phoenix)
    FOV_SPACING_X = 0.305  # mm between FoV centers
    FOV_SPACING_Y = 0.305  # mm
    
    # Well center position
    well_center_x = WELL_A1_X + (well_col * WELL_SPACING)
    well_center_y = WELL_A1_Y + (well_row * WELL_SPACING)
    
    # FoV offset from well center
    fov_row = fov_idx // 3
    fov_col = fov_idx % 3
    
    # Center the 3×3 grid around well center
    fov_offset_x = (fov_col - 1) * FOV_SPACING_X  # -1, 0, +1
    fov_offset_y = (fov_row - 1) * FOV_SPACING_Y  # -1, 0, +1
    
    # Final absolute stage coordinates
    stage_x = well_center_x + fov_offset_x
    stage_y = well_center_y + fov_offset_y
    
    return stage_x, stage_y
```

### Coordinate System Example

**Well A1 (12.05, 9.05):**
```
FoV 0: (11.745, 8.745)  FoV 1: (12.05, 8.745)  FoV 2: (12.355, 8.745)
FoV 3: (11.745, 9.05)   FoV 4: (12.05, 9.05)   FoV 5: (12.355, 9.05)
FoV 6: (11.745, 9.355)  FoV 7: (12.05, 9.355)  FoV 8: (12.355, 9.355)
```

**Well B6 (56.3, 18.05):**
```
FoV 0: (56.0, 17.745)  FoV 1: (56.3, 17.745)  FoV 2: (56.6, 17.745)
...
```

## Import Process Overview

### Step 1: Open Opera OME-TIFF as Zarr

```python
import tifffile
import zarr

with tifffile.TiffFile('plate1.ome.tif') as tif:
    store = tif.aszarr()
    zgroup = zarr.open(store, mode="r")
    zarr_arr = zgroup["0"]  # Shape: (2592, 4, 1080, 1080)
```

### Step 2: Create Squid Zarr Canvas

The Squid microscope uses a **coordinate-based canvas**:
- Canvas covers entire stage area: 120mm × 80mm
- Pixel size: 0.0435 µm/pixel (63X magnification)
- Images placed at absolute (x, y) coordinates in millimeters
- Multi-scale pyramid: 6 levels with 4× downsampling

```python
from squid_control.stitching.zarr_canvas import ExperimentManager

exp_manager = ExperimentManager(
    base_path="/mnt/shared_documents",
    pixel_size_xy_um=0.0435,
    stage_limits={
        'x_positive': 120.0,
        'x_negative': 0.0,
        'y_positive': 80.0,
        'y_negative': 0.0
    }
)

exp_manager.create_experiment(
    experiment_name="opera_import",
    well_plate_type="96",
    initialize_all_wells=False
)

canvas = exp_manager.get_canvas()
```

### Step 3: Process Each Tile

```python
for tile_idx in range(n_tiles):
    # Decode tile index to coordinates
    stage_x, stage_y, well_id, fov_idx, z_idx = get_opera_stage_coordinates(tile_idx)
    
    # Skip unwanted z-planes
    if z_idx != 1:  # Only import middle z-plane
        continue
    
    # Process each channel
    for opera_channel_idx in [0, 1, 2, 3]:
        # Read tile from Opera zarr
        tile_data = zarr_arr[tile_idx, opera_channel_idx, :, :]
        
        # Map Opera channel to Squid channel
        squid_channel_name = channel_mapping[opera_channel_idx]
        canvas_zarr_idx = canvas.channel_to_zarr_index[squid_channel_name]
        
        # Normalize uint16 → uint8
        normalized_tile = normalize_to_uint8(tile_data)
        
        # Add to canvas at absolute coordinates
        canvas.add_image_sync(
            image=normalized_tile,
            x_mm=stage_x,
            y_mm=stage_y,
            z_mm=3.323,  # Fixed z-height
            channel_idx=canvas_zarr_idx,
            timepoint=0
        )
```

### Step 4: Channel Mapping

Opera channels are mapped to Squid's standard channel structure:

| Opera Ch | Opera Wavelength | → | Squid Channel | Squid Index |
|----------|------------------|---|---------------|-------------|
| 0 | 405nm (DAPI) | → | Fluorescence 405 nm Ex | 1 |
| 1 | 488nm (Target) | → | Fluorescence 488 nm Ex | 2 |
| 2 | 638nm (Virus) | → | Fluorescence 638 nm Ex | 3 |
| 3 | 561nm (ER) | → | Fluorescence 561 nm Ex | 4 |

**Note:** Squid canvas index 0 is reserved for Brightfield (not in Opera data).

## Data Normalization

### uint16 → uint8 Conversion

Opera data is 16-bit (0-65535), but Squid canvas uses 8-bit (0-255):

```python
def normalize_to_uint8(tile_data):
    """Normalize uint16 to uint8 preserving dynamic range."""
    data_min = tile_data.min()
    data_max = tile_data.max()
    
    if data_max > data_min:
        # Scale to full 8-bit range
        normalized = ((tile_data - data_min) / (data_max - data_min) * 255)
        return normalized.astype(np.uint8)
    else:
        # Uniform tile (no data)
        return np.clip(tile_data // 256, 0, 255).astype(np.uint8)
```

## Verification

After import, verify the data:

```python
import zarr
import numpy as np

root = zarr.open_group('/mnt/shared_documents/opera_import/data.zarr', mode='r')
arr = root['3']  # Scale 3 for quick check

# Check each channel
for channel_idx in range(6):
    sample = arr[0, channel_idx, 0, 3200:3300, 4300:4400]
    print(f'Channel {channel_idx}: range=[{sample.min()}, {sample.max()}]')
```

Expected output:
```
Channel 0 (BF): range=[0, 0]         # Empty (no brightfield in Opera)
Channel 1 (405nm): range=[0, 119]    # DAPI data ✓
Channel 2 (488nm): range=[0, 98]     # Target protein data ✓
Channel 3 (638nm): range=[0, 143]    # Virus data ✓
Channel 4 (561nm): range=[0, 76]     # ER data ✓
Channel 5 (730nm): range=[0, 0]      # Empty (not in Opera)
```

## Usage

### Import All Channels

```bash
python scripts/import_opera_to_zarr.py \
  --opera-tif /path/to/plate1.ome.tif \
  --output-dir /mnt/shared_documents/ \
  --experiment-name opera_import \
  --channels "0,1,2,3" \
  --verbose
```

### Import Specific Channels Only

```bash
# Import only DAPI (channel 0)
python scripts/import_opera_to_zarr.py \
  --opera-tif /path/to/plate1.ome.tif \
  --output-dir /mnt/shared_documents/ \
  --experiment-name opera_dapi_only \
  --channels "0"
```

### Import Different Z-Plane

```bash
# Import top focal plane instead of middle
python scripts/import_opera_to_zarr.py \
  --opera-tif /path/to/plate1.ome.tif \
  --output-dir /mnt/shared_documents/ \
  --experiment-name opera_top_plane \
  --z-plane 0 \
  --channels "0,1,2,3"
```

## Configuration for 63X Simulation Mode

To use the imported Opera data with the Squid microscope simulation:

1. **Set wellplate offset** (`configuration_HCS_v2_63x.ini`):
   ```ini
   wellplate_offset_x_mm = -2.250
   wellplate_offset_y_mm = -2.310
   ```
   This aligns the microscope's well plate navigation with Opera's coordinate system.

2. **Start 63X microscope simulation**:
   ```bash
   python -m squid_control microscope --simulation --config HCS_v2_63x
   ```

3. **Navigate to any well and acquire images**:
   ```python
   # Via Hypha RPC
   await microscope.navigate_to_well(row='A', col=1, well_plate_type='96')
   image_url = await microscope.snap(channel='Fluorescence_405_nm_Ex')
   ```

## Technical Specifications

### Opera Phoenix System
- **Objective**: 63X water immersion, NA 1.4
- **Pixel size**: 0.0435 µm/pixel (calculated)
- **Tile size**: 1080 × 1080 pixels
- **Well plate**: 96-well format (8 rows × 12 columns)
- **Imaging pattern**: 3×3 FoV grid per well, 3 z-planes per FoV

### Squid Zarr Canvas
- **Canvas size**: 2,758,656 × 1,839,104 pixels (120mm × 80mm)
- **Scales**: 6 levels (4× downsampling each level)
- **Chunk size**: 256 × 256 pixels
- **Format**: OME-Zarr 0.4 specification
- **Coordinate system**: Absolute stage coordinates in millimeters

## Troubleshooting

### Issue: "Image contains all zeros"
**Cause**: Channel name mismatch (underscores vs spaces)
**Solution**: The camera now auto-normalizes channel names (`Fluorescence_405_nm_Ex` ↔ `Fluorescence 405 nm Ex`)

### Issue: "Only channel 0 has data"
**Cause**: Imported only one channel
**Solution**: Re-import with `--channels "0,1,2,3"` to get all 4 channels

### Issue: "Position mismatch - wrong well"
**Cause**: Wellplate offset not set
**Solution**: Set correct offset in `configuration_HCS_v2_63x.ini`:
```ini
wellplate_offset_x_mm = -2.250
wellplate_offset_y_mm = -2.310
```

## References

- [OME-TIFF Specification](https://docs.openmicroscopy.org/ome-model/latest/)
- [OME-Zarr Format](https://ngff.openmicroscopy.org/)
- [Opera Phoenix Documentation](https://www.revvity.com/product/operetta-cls-hh16000000)
- Squid Microscope: [GitHub Repository](https://github.com/hongquanli/octopi-research)

## Author

Created for the Squid microscope control system by the AI Cell Lab.

Last updated: February 15, 2026
