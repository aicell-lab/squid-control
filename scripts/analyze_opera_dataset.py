#!/usr/bin/env python3
"""
Analyze the Opera Phoenix dataset structure to understand:
1. How tiles map to wells and FoVs
2. The coordinate system needed for Zarr canvas
3. Pixel size calculations for 63X objective

Dataset info:
- 384-well plate  
- 63X magnification
- 9 FoV per well
- 3 z-planes per FoV
- 4 channels (DAPI, ER, Virus, Target)
- Zarr shape: (2592, 4, 1080, 1080)
"""

import json
from pathlib import Path
import tifffile
import zarr
import numpy as np

# Dataset paths
DATA_DIR = Path("/media/reef/harddisk/immunofluorescence_data/14315777")
OME_TIF = DATA_DIR / "plate1.ome.tif"
OFFSETS_JSON = DATA_DIR / "plate1.ome.tif_offsets.json"

def analyze_dataset():
    """Analyze the Opera dataset structure."""
    
    print("="*80)
    print("OPERA PHOENIX DATASET ANALYSIS")
    print("="*80)
    
    # 1. Load Zarr from OME-TIFF
    print("\n1. Loading Zarr array...")
    with tifffile.TiffFile(OME_TIF) as tif:
        store = tif.aszarr()
        zgroup = zarr.open(store, mode="r")
        zarr_arr = zgroup["0"]
        
    print(f"   Zarr shape: {zarr_arr.shape}")
    print(f"   Dtype: {zarr_arr.dtype}")
    print(f"   Chunks: {zarr_arr.chunks}")
    
    n_tiles, n_channels, tile_h, tile_w = zarr_arr.shape
    print(f"\n   Total tiles: {n_tiles}")
    print(f"   Channels: {n_channels}")
    print(f"   Tile size: {tile_h} x {tile_w} pixels")
    
    # 2. Calculate expected structure
    print("\n2. Dataset structure analysis:")
    print(f"   384-well plate = 16 rows x 24 columns")
    print(f"   9 FoV per well")
    print(f"   3 z-planes per FoV")
    print(f"   Expected total tiles: 384 wells × 9 FoV × 3 z = {384 * 9 * 3}")
    
    expected_tiles = 384 * 9 * 3
    wells_384 = 384
    fov_per_well = 9
    z_per_fov = 3
    
    if n_tiles == expected_tiles:
        print(f"   ✓ Tile count matches! {n_tiles} tiles")
    else:
        print(f"   ✗ Tile count mismatch: got {n_tiles}, expected {expected_tiles}")
        # Try to infer structure
        if n_tiles % z_per_fov == 0:
            n_fov_total = n_tiles // z_per_fov
            n_wells_actual = n_fov_total // fov_per_well
            print(f"   → If 3 z-planes: {n_fov_total} FoV total")
            print(f"   → If 9 FoV per well: {n_wells_actual} wells")
            print(f"   → Might be scanning {n_wells_actual} wells instead of full 384")
    
    # 3. Calculate tile organization
    print("\n3. Tile organization hypothesis:")
    print(f"   Tiles are likely organized as:")
    print(f"   tile_idx = (well_idx * 9 * 3) + (fov_idx * 3) + z_idx")
    print(f"   where:")
    print(f"     - well_idx: 0 to {wells_384-1} (row-major: A1, A2, ..., A24, B1, B2, ...)")
    print(f"     - fov_idx: 0 to 8 (9 fields of view per well)")
    print(f"     - z_idx: 0 to 2 (3 z-planes, we use z_idx=1 for middle plane)")
    
    # 4. 384-well plate layout (from config.py)
    print("\n4. 384-well plate physical layout:")
    print(f"   Well spacing: 4.5 mm")
    print(f"   Well size: 3.3 mm diameter")
    print(f"   A1 position: (12.05, 9.05) mm (from config)")
    print(f"   Layout: 16 rows (A-P) × 24 columns (1-24)")
    
    # 5. Field of view calculations for 63X
    print("\n5. Field of view calculations (63X objective):")
    
    # From config: pixel size calculation
    # For 63X objective:
    tube_lens_mm = 180  # Standard (from CONFIG)
    pixel_size_um = 2.74  # Camera sensor (typical for scientific cameras)
    magnification = 63
    objective_tube_lens_mm = 180  # Standard for 63X
    binning = 1  # No binning
    
    # Calculate pixel size
    pixel_size_xy = pixel_size_um / (magnification / (objective_tube_lens_mm / tube_lens_mm))
    pixel_size_xy = pixel_size_xy * binning
    
    print(f"   Camera pixel size: {pixel_size_um} µm")
    print(f"   Magnification: {magnification}X")
    print(f"   Calculated pixel size: {pixel_size_xy:.4f} µm/pixel")
    
    # FOV size
    fov_width_um = tile_w * pixel_size_xy
    fov_height_um = tile_h * pixel_size_xy
    fov_width_mm = fov_width_um / 1000
    fov_height_mm = fov_height_um / 1000
    
    print(f"   FOV size: {fov_width_um:.1f} x {fov_height_um:.1f} µm")
    print(f"   FOV size: {fov_width_mm:.3f} x {fov_height_mm:.3f} mm")
    
    # 6. 9-field layout within well (typical 3x3 grid)
    print("\n6. 9-field layout within well (typical 3x3 grid):")
    print(f"   Assuming 3x3 grid arrangement:")
    print(f"   Grid spacing ≈ FOV size with small overlap")
    
    # Typical overlap for stitching is 10-20%
    overlap_percent = 10
    effective_fov_mm = fov_width_mm * (1 - overlap_percent / 100)
    
    print(f"   With {overlap_percent}% overlap:")
    print(f"   Effective FOV step: {effective_fov_mm:.3f} mm")
    print(f"   3x3 grid total coverage: {effective_fov_mm * 2:.3f} x {effective_fov_mm * 2:.3f} mm")
    print(f"   (Fits within well diameter of 3.3 mm)")
    
    # 7. Sample a few tiles to check
    print("\n7. Sampling tiles to verify structure:")
    sample_tiles = [0, 27, 54, 81]  # First tile, then every well*fov
    for tile_idx in sample_tiles:
        if tile_idx < n_tiles:
            # Calculate well and FoV from tile index
            well_idx = tile_idx // (fov_per_well * z_per_fov)
            remainder = tile_idx % (fov_per_well * z_per_fov)
            fov_idx = remainder // z_per_fov
            z_idx = remainder % z_per_fov
            
            # Well position (row-major)
            well_row_idx = well_idx // 24  # 24 columns
            well_col_idx = well_idx % 24
            well_row = chr(ord('A') + well_row_idx)
            well_col = well_col_idx + 1
            
            tile_data = zarr_arr[tile_idx, 0, :, :]  # Channel 0
            mean_val = np.mean(tile_data)
            max_val = np.max(tile_data)
            
            print(f"   Tile {tile_idx:4d}: Well {well_row}{well_col:2d}, FoV {fov_idx}, Z {z_idx} | "
                  f"Mean: {mean_val:6.1f}, Max: {max_val:5d}")
    
    print("\n" + "="*80)
    print("PROPOSED SOLUTION")
    print("="*80)
    
    print("""
1. TILE TO COORDINATE MAPPING:
   - Parse tile index to (well_idx, fov_idx, z_idx)
   - Calculate well center: (A1_x + col*spacing, A1_y + row*spacing)
   - Calculate FoV offset within well using 3x3 grid pattern
   - Use only z_idx=1 (middle z-plane)

2. ZARR CANVAS SETUP:
   - Pixel size: {:.4f} µm/pixel
   - Canvas size: cover full 384-well plate area
   - Channels: ['DAPI', 'ER', 'Virus', 'Target'] (or original names)
   - Multi-scale pyramid: 4x downsampling between scales

3. COORDINATE CALCULATION:
   def get_stage_coordinates(tile_idx, well_spacing_mm=4.5, a1_x_mm=12.05, a1_y_mm=9.05):
       well_idx = tile_idx // 27  # 27 = 9 FoV × 3 z-planes
       fov_idx = (tile_idx % 27) // 3
       z_idx = tile_idx % 3
       
       if z_idx != 1:  # Skip if not middle z-plane
           return None
           
       # Well position (row-major, 24 columns)
       well_row = well_idx // 24
       well_col = well_idx % 24
       well_center_x = a1_x_mm + well_col * well_spacing_mm
       well_center_y = a1_y_mm + well_row * well_spacing_mm
       
       # FoV offset (3x3 grid: -1, 0, +1 in each dimension)
       fov_row = fov_idx // 3  # 0, 1, 2
       fov_col = fov_idx % 3   # 0, 1, 2
       fov_offset_x = (fov_col - 1) * {:.3f}  # ±effective_fov_mm
       fov_offset_y = (fov_row - 1) * {:.3f}
       
       stage_x = well_center_x + fov_offset_x
       stage_y = well_center_y + fov_offset_y
       
       return (stage_x, stage_y)

4. IMPLEMENTATION STEPS:
   a. Create import script that reads plate1.ome.tif
   b. For each tile where z_idx=1:
      - Calculate stage coordinates
      - Read tile from Zarr
      - Add to canvas using canvas.add_image()
   c. Generate multi-scale pyramid
   d. Export as OME-Zarr with metadata

5. EXPECTED RESULT:
   - {} images added to canvas (using middle z-plane only)
   - 96-well subset stitched with coordinate system
   - Can overlay with squid microscope data for comparison
""".format(
        pixel_size_xy,
        effective_fov_mm,
        effective_fov_mm,
        n_tiles // 3
    ))
    
    return {
        'n_tiles': n_tiles,
        'n_channels': n_channels,
        'tile_size': (tile_h, tile_w),
        'pixel_size_xy_um': pixel_size_xy,
        'fov_size_mm': (fov_width_mm, fov_height_mm),
        'effective_fov_step_mm': effective_fov_mm,
    }

if __name__ == "__main__":
    result = analyze_dataset()
    print("\n✓ Analysis complete. Ready to implement import script.")
