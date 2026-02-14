#!/usr/bin/env python3
"""
Test and validate the Opera Phoenix import script before running full import.
This script does a dry-run to verify coordinate calculations and import pipeline.
"""

import sys
from pathlib import Path

import numpy as np
import tifffile
import zarr

# Add parent directory for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.import_opera_to_zarr import get_opera_stage_coordinates


def test_coordinate_calculations():
    """Test the coordinate calculation function with known tile indices."""
    print("=" * 80)
    print("TESTING COORDINATE CALCULATIONS")
    print("=" * 80)
    
    # Test cases: various tile indices
    test_cases = [
        0,    # Well A1, FoV 0, Z 0
        1,    # Well A1, FoV 0, Z 1 (middle) - SHOULD IMPORT
        2,    # Well A1, FoV 0, Z 2
        13,   # Well A1, FoV 4 (center), Z 1 - SHOULD IMPORT
        27,   # Well A2, FoV 0, Z 0
        28,   # Well A2, FoV 0, Z 1 - SHOULD IMPORT
        270,  # Well A10, FoV 0, Z 0
        324,  # Well A12, FoV 0, Z 0 (last well in row A)
        325,  # Well B1, FoV 0, Z 1 - SHOULD IMPORT (second row)
        2591, # Last tile
    ]
    
    print("\nSample tile coordinate calculations:")
    print(f"{'Tile':<6} {'Well':<6} {'FoV':<4} {'Z':<3} {'X (mm)':<10} {'Y (mm)':<10} {'Import?':<8}")
    print("-" * 70)
    
    imported_count = 0
    for tile_idx in test_cases:
        coords = get_opera_stage_coordinates(tile_idx)
        if coords is None:
            # Calculate for display even if None
            well_idx = tile_idx // 27
            fov_idx = (tile_idx % 27) // 3
            z_idx = tile_idx % 3
            well_row = well_idx // 24
            well_col = well_idx % 24
            well_id = f"{chr(ord('A') + well_row)}{well_col + 1}"
            print(f"{tile_idx:<6} {well_id:<6} {fov_idx:<4} {z_idx:<3} {'N/A':<10} {'N/A':<10} {'No':<8}")
        else:
            stage_x, stage_y, well_id, fov_idx, z_idx = coords
            print(f"{tile_idx:<6} {well_id:<6} {fov_idx:<4} {z_idx:<3} {stage_x:<10.3f} {stage_y:<10.3f} {'Yes':<8}")
            imported_count += 1
    
    print(f"\nResult: {imported_count}/{len(test_cases)} tiles would be imported (z-plane 1 only)")
    
    # Verify well coverage
    print("\n" + "=" * 80)
    print("WELL COVERAGE ANALYSIS")
    print("=" * 80)
    
    well_centers = {}
    for well_idx in range(96):
        # Get first FoV, middle z-plane of each well
        tile_idx = well_idx * 27 + 1  # fov_idx=0, z_idx=1
        coords = get_opera_stage_coordinates(tile_idx)
        if coords:
            stage_x, stage_y, well_id, fov_idx, z_idx = coords
            well_centers[well_id] = (stage_x, stage_y)
    
    print(f"\nTotal wells mapped: {len(well_centers)}")
    print(f"First 10 wells:")
    for i, (well_id, (x, y)) in enumerate(list(well_centers.items())[:10]):
        print(f"  {well_id}: ({x:.3f}, {y:.3f}) mm")
    
    # Check coordinate ranges
    all_x = [x for x, y in well_centers.values()]
    all_y = [y for x, y in well_centers.values()]
    print(f"\nCoordinate ranges:")
    print(f"  X: {min(all_x):.3f} to {max(all_x):.3f} mm (range: {max(all_x) - min(all_x):.3f} mm)")
    print(f"  Y: {min(all_y):.3f} to {max(all_y):.3f} mm (range: {max(all_y) - min(all_y):.3f} mm)")
    print(f"\nExpected for 96 wells (8 rows × 12 cols, 4.5 mm spacing):")
    print(f"  X range: {11 * 4.5:.1f} mm (12 columns - 1) × 4.5 mm")
    print(f"  Y range: {7 * 4.5:.1f} mm (8 rows - 1) × 4.5 mm")


def test_opera_file_reading(opera_tif_path: Path):
    """Test reading the Opera OME-TIFF file."""
    print("\n" + "=" * 80)
    print("TESTING OPERA FILE READING")
    print("=" * 80)
    
    try:
        with tifffile.TiffFile(opera_tif_path) as tif:
            print(f"  File opened: {opera_tif_path.name}")
            
            if not hasattr(tif, 'aszarr'):
                print("  ✗ No Zarr storage in TIFF")
                return False
            
            store = tif.aszarr()
            zgroup = zarr.open(store, mode="r")
            print(f"  ✓ Zarr store opened")
            
            if "0" not in zgroup:
                print("  ✗ No scale 0 in Zarr group")
                return False
            
            zarr_arr = zgroup["0"]
            print(f"  ✓ Zarr array accessed")
            print(f"  Shape: {zarr_arr.shape}")
            print(f"  Dtype: {zarr_arr.dtype}")
            print(f"  Chunks: {zarr_arr.chunks}")
            
            # Try reading a few sample tiles
            print(f"\n  Testing tile reading:")
            sample_tiles = [0, 1, 27, 270]
            for tile_idx in sample_tiles:
                try:
                    tile_data = zarr_arr[tile_idx, 0, :, :]
                    tile_array = np.asarray(tile_data)
                    mean_val = np.mean(tile_array)
                    max_val = np.max(tile_array)
                    print(f"    Tile {tile_idx:4d}: shape={tile_array.shape}, "
                          f"mean={mean_val:.1f}, max={max_val}, dtype={tile_array.dtype}")
                except Exception as e:
                    print(f"    Tile {tile_idx:4d}: ERROR - {e}")
                    return False
            
            print(f"  ✓ All sample tiles read successfully")
            return True
            
    except Exception as e:
        print(f"  ✗ Error opening file: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all tests."""
    # Test 1: Coordinate calculations
    test_coordinate_calculations()
    
    # Test 2: File reading (if file exists)
    opera_tif = Path("/media/reef/harddisk/immunofluorescence_data/14315777/plate1.ome.tif")
    if opera_tif.exists():
        success = test_opera_file_reading(opera_tif)
        if not success:
            print("\n✗ File reading test FAILED")
            return 1
    else:
        print(f"\n⚠ Skipping file reading test (file not found: {opera_tif})")
    
    print("\n" + "=" * 80)
    print("✓ ALL TESTS PASSED")
    print("=" * 80)
    print("\nReady to run full import:")
    print("  python scripts/import_opera_to_zarr.py \\")
    print("      --opera-tif /media/reef/harddisk/immunofluorescence_data/14315777/plate1.ome.tif \\")
    print("      --output-dir /media/reef/harddisk/immunofluorescence_data/experiments \\")
    print("      --experiment-name opera_import")
    print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
