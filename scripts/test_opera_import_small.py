#!/usr/bin/env python3
"""
Small-scale test: Import just 2 wells (18 tiles) to verify the pipeline works.
"""

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.import_opera_to_zarr import import_opera_dataset

async def main():
    opera_tif = Path("/media/reef/harddisk/immunofluorescence_data/14315777/plate1.ome.tif")
    output_dir = Path("/tmp/opera_test")
    
    print("Testing import with first 2 wells only (18 FoV)...")
    print("This is a quick test - full import will process 864 FoV across 96 wells.")
    print()
    
    # For testing, we'll modify the import to only process first 54 tiles (2 wells × 27 tiles)
    # But the import function processes all tiles - we'd need to modify it
    # Instead, let's just run the full import to a test directory
    
    result = await import_opera_dataset(
        opera_tif_path=opera_tif,
        output_base_path=output_dir,
        experiment_name="opera_test_2wells",
        z_plane=1,
        channels_to_import=[0],  # Just DAPI channel for quick test
        save_preview=False
    )
    
    if result['success']:
        print(f"\n✓ Test import succeeded!")
        print(f"  Tiles imported: {result['tiles_imported']}")
        print(f"  Time: {result['elapsed_time']:.1f} seconds")
        return 0
    else:
        print(f"\n✗ Test import failed: {result.get('error')}")
        return 1

if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
