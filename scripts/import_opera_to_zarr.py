#!/usr/bin/env python3
"""
Import Opera Phoenix microscopy dataset into coordinate-based Zarr canvas.

This script converts Opera Phoenix OME-TIFF data (96 wells, 9 FoV per well, 3 z-planes)
into a multi-scale OME-Zarr format compatible with the squid microscope system.

Usage:
    python scripts/import_opera_to_zarr.py \
        --opera-tif /path/to/plate1.ome.tif \
        --output-dir /path/to/experiments \
        --experiment-name opera_import

Author: Squid Control System
Dataset: Opera Phoenix 63X, 384-well plate geometry
"""

import argparse
import asyncio
import json
import logging
import sys
import time
import psutil
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import tifffile
import zarr

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from squid_control.stitching.zarr_canvas import ExperimentManager, ZarrCanvas

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


# ============================================================================
# COORDINATE MAPPING FUNCTIONS
# ============================================================================

def get_opera_stage_coordinates(tile_idx: int) -> Optional[Tuple[float, float, str, int, int]]:
    """
    Calculate stage coordinates for Opera Phoenix tile.
    
    The Opera Phoenix dataset has:
    - 96 wells in standard 8×12 layout (A1-H12)
    - 9 FoV per well (3x3 grid)
    - 3 z-planes per FoV (we use middle plane, z_idx=1)
    
    Tile organization: tile_idx = (well_idx * 27) + (fov_idx * 3) + z_idx
    
    Args:
        tile_idx: Tile index (0 to 2591)
        
    Returns:
        (x_mm, y_mm, well_id, fov_idx, z_idx) or None if not middle z-plane
        
    Example:
        >>> coords = get_opera_stage_coordinates(0)
        >>> coords  # Returns None if z_idx=0 (not middle plane)
        >>> coords = get_opera_stage_coordinates(1)
        >>> coords  # (12.008, 9.008, 'A1', 0, 1) - middle plane
    """
    # 96-well plate geometry (standard layout)
    WELL_SPACING_MM = 9.0  # Standard 96-well spacing
    A1_X_MM = 12.05  # A1 well center X position
    A1_Y_MM = 9.05   # A1 well center Y position
    FOV_STEP_MM = 0.042  # Effective FoV spacing with ~10% overlap
    WELLS_PER_ROW = 12  # 96-well plate has 12 columns
    NUM_ROWS = 8  # 96-well plate has 8 rows (A-H)
    
    # Parse tile structure: well_idx × 27 + fov_idx × 3 + z_idx
    well_idx = tile_idx // 27  # Which well (0-95)
    fov_idx = (tile_idx % 27) // 3  # Which FoV within well (0-8)
    z_idx = tile_idx % 3  # Which z-plane (0, 1, 2)
    
    # Filter: only process middle z-plane
    if z_idx != 1:
        return None
    
    # Calculate well position (row-major order: 8 rows × 12 columns)
    well_row = well_idx // WELLS_PER_ROW  # 0-7 (A-H)
    well_col = well_idx % WELLS_PER_ROW   # 0-11 (1-12)
    well_id = f"{chr(ord('A') + well_row)}{well_col + 1}"
    
    # Well center in stage coordinates
    well_center_x = A1_X_MM + well_col * WELL_SPACING_MM
    well_center_y = A1_Y_MM + well_row * WELL_SPACING_MM
    
    # FoV offset within well (3×3 grid, centered on well)
    # fov_idx mapping: 0=top-left, 1=top-center, 2=top-right,
    #                  3=mid-left, 4=center, 5=mid-right,
    #                  6=bot-left, 7=bot-center, 8=bot-right
    fov_row = fov_idx // 3  # 0, 1, 2 (top, middle, bottom)
    fov_col = fov_idx % 3   # 0, 1, 2 (left, center, right)
    
    # Offset from well center: -1, 0, +1 positions
    fov_offset_x = (fov_col - 1) * FOV_STEP_MM  # -0.042, 0, +0.042 mm
    fov_offset_y = (fov_row - 1) * FOV_STEP_MM
    
    # Final stage position
    stage_x = well_center_x + fov_offset_x
    stage_y = well_center_y + fov_offset_y
    
    return (stage_x, stage_y, well_id, fov_idx, z_idx)


# ============================================================================
# VALIDATION FUNCTIONS
# ============================================================================

def validate_import(
    canvas: ZarrCanvas,
    tiles_imported: int,
    expected_wells: int = 96,
    expected_fov_per_well: int = 9
) -> Dict:
    """
    Validate the imported dataset for completeness and correctness.
    
    Args:
        canvas: The ZarrCanvas that was populated
        tiles_imported: Number of tiles actually imported
        expected_wells: Expected number of wells (default: 96)
        expected_fov_per_well: Expected FoV per well (default: 9)
        
    Returns:
        Dict with validation results and statistics
    """
    logger.info("=" * 80)
    logger.info("VALIDATION REPORT")
    logger.info("=" * 80)
    
    validation_results = {
        'tiles_imported': tiles_imported,
        'expected_tiles': expected_wells * expected_fov_per_well,
        'match': False,
        'canvas_info': {}
    }
    
    expected_tiles = expected_wells * expected_fov_per_well
    
    # Check tile count
    if tiles_imported == expected_tiles:
        logger.info(f"✓ Tile count: {tiles_imported} / {expected_tiles} (PASS)")
        validation_results['match'] = True
    else:
        logger.warning(f"✗ Tile count: {tiles_imported} / {expected_tiles} (MISMATCH)")
        validation_results['match'] = False
    
    # Canvas statistics
    try:
        canvas_info = {
            'canvas_size_px': (canvas.canvas_width_px, canvas.canvas_height_px),
            'pixel_size_um': canvas.pixel_size_xy_um,
            'num_channels': len(canvas.channels),
            'channels': canvas.channels,
            'num_scales': canvas.num_scales,
            'chunk_size': canvas.chunk_size,
        }
        validation_results['canvas_info'] = canvas_info
        
        logger.info(f"\nCanvas Statistics:")
        logger.info(f"  Size: {canvas_info['canvas_size_px'][0]:,} × {canvas_info['canvas_size_px'][1]:,} pixels")
        logger.info(f"  Pixel size: {canvas_info['pixel_size_um']} µm/pixel")
        logger.info(f"  Channels: {canvas_info['num_channels']} - {', '.join(canvas_info['channels'])}")
        logger.info(f"  Scales: {canvas_info['num_scales']} levels")
        logger.info(f"  Chunk size: {canvas_info['chunk_size']} pixels")
        
    except Exception as e:
        logger.error(f"Could not retrieve canvas statistics: {e}")
    
    # Well coverage analysis
    logger.info(f"\nWell Coverage:")
    logger.info(f"  Expected wells: {expected_wells} (A1 to H12)")
    logger.info(f"  FoV per well: {expected_fov_per_well} (3×3 grid)")
    logger.info(f"  Total FoV imported: {tiles_imported}")
    
    # Coordinate range
    logger.info(f"\nExpected Coordinate Range:")
    logger.info(f"  X: 12.05 to {12.05 + 11 * 4.5:.2f} mm (12 columns × 4.5 mm spacing)")
    logger.info(f"  Y: 9.05 to {9.05 + 7 * 4.5:.2f} mm (8 rows × 4.5 mm spacing)")
    
    logger.info("=" * 80)
    
    return validation_results


# ============================================================================
# MAIN IMPORT FUNCTION
# ============================================================================

async def import_opera_dataset(
    opera_tif_path: Path,
    output_base_path: Path,
    experiment_name: str = "opera_import",
    channel_mapping: Optional[Dict[int, str]] = None,
    z_plane: int = 1,
    channels_to_import: Optional[List[int]] = None,
    save_preview: bool = True
) -> Dict:
    """
    Import Opera Phoenix dataset into coordinate-based Zarr canvas.
    
    Args:
        opera_tif_path: Path to plate1.ome.tif
        output_base_path: Base path for output Zarr storage
        experiment_name: Name for the experiment (default: "opera_import")
        channel_mapping: Dict mapping Opera channel idx to squid channel names
                        Default: {0: 'Fluorescence 405 nm Ex' (DAPI),
                                  1: 'Fluorescence 561 nm Ex' (ER),
                                  2: 'Fluorescence 638 nm Ex' (Virus),
                                  3: 'Fluorescence 488 nm Ex' (Target)}
        z_plane: Which z-plane to import (0, 1, or 2; default: 1 = middle)
        channels_to_import: List of channel indices to import (default: all)
        save_preview: Whether to save preview images (default: True)
        
    Returns:
        Dict with import statistics and results
    """
    start_time = time.time()
    
    # Default channel mapping (Opera channels → Squid channels)
    if channel_mapping is None:
        channel_mapping = {
            0: 'Fluorescence 405 nm Ex',  # DAPI (nucleus, blue, 405nm)
            1: 'Fluorescence 488 nm Ex',  # Target protein (green, 488nm)
            2: 'Fluorescence 638 nm Ex',  # SARS-CoV-2 Virus (red, 638nm)
            3: 'Fluorescence 561 nm Ex',  # ER (yellow/orange, 561nm)
        }
    
    if channels_to_import is None:
        channels_to_import = list(channel_mapping.keys())
    
    logger.info("=" * 80)
    logger.info("OPERA PHOENIX TO ZARR IMPORT")
    logger.info("=" * 80)
    logger.info(f"Input: {opera_tif_path}")
    logger.info(f"Output: {output_base_path / experiment_name}")
    logger.info(f"Z-plane: {z_plane} (0=top, 1=middle, 2=bottom)")
    logger.info(f"Channels to import: {channels_to_import}")
    logger.info(f"Channel mapping: {channel_mapping}")
    logger.info("=" * 80)
    
    # Step 1: Open Opera OME-TIFF as Zarr
    logger.info("\n[1/7] Opening Opera OME-TIFF...")
    try:
        with tifffile.TiffFile(opera_tif_path) as tif:
            if not hasattr(tif, 'aszarr'):
                raise ValueError("TIFF does not have Zarr storage")
            store = tif.aszarr()
            zgroup = zarr.open(store, mode="r")
            
            if "0" in zgroup:
                zarr_arr = zgroup["0"]
            else:
                raise ValueError("Could not find scale 0 in Zarr group")
            
            n_tiles, n_channels, tile_h, tile_w = zarr_arr.shape
            logger.info(f"  Zarr shape: {zarr_arr.shape}")
            logger.info(f"  Total tiles: {n_tiles}")
            logger.info(f"  Channels: {n_channels}")
            logger.info(f"  Tile size: {tile_h} × {tile_w} pixels")
            
            # Calculate expected tiles after filtering
            tiles_per_well = 9  # 9 FoV per well
            z_planes_per_fov = 3
            total_fov = n_tiles // z_planes_per_fov
            num_wells = total_fov // tiles_per_well
            tiles_after_z_filter = total_fov
            
            logger.info(f"  Wells in dataset: {num_wells}")
            logger.info(f"  FoV per well: {tiles_per_well}")
            logger.info(f"  Tiles after z-plane filtering: {tiles_after_z_filter}")
            
            # Step 2: Create ExperimentManager and Canvas
            logger.info("\n[2/7] Creating Zarr canvas...")
            pixel_size_xy_um = 0.0435  # 63X magnification, calculated from Opera specs
            
            stage_limits = {
                'x_positive': 120.0,
                'x_negative': 0.0,
                'y_positive': 80.0,
                'y_negative': 0.0
            }
            
            logger.info(f"  Pixel size: {pixel_size_xy_um} µm/pixel")
            
            # Create experiment manager - it will create canvas with ALL standard channels
            exp_manager = ExperimentManager(
                base_path=str(output_base_path),
                pixel_size_xy_um=pixel_size_xy_um,
                stage_limits=stage_limits
            )
            
            # Create experiment (canvas will have standard channel order)
            logger.info(f"  Creating experiment: {experiment_name}")
            exp_result = exp_manager.create_experiment(
                experiment_name=experiment_name,
                well_plate_type="96",
                initialize_all_wells=False
            )
            
            # Get canvas (has standard channel structure)
            canvas = exp_manager.get_canvas(initialize_new=False)
            
            # DO NOT override channels - keep standard Squid microscope channel structure
            # Standard order: BF(0), 405(1), 488(2), 638(3), 561(4), 730(5)
            logger.info(f"  Canvas channels (standard): {canvas.channels}")
            logger.info(f"  Canvas zarr indices: {canvas.zarr_index_to_channel}")
            
            logger.info(f"  Canvas created: {canvas.canvas_width_px} × {canvas.canvas_height_px} pixels")
            logger.info(f"  Multi-scale levels: {canvas.num_scales}")
            
            # Step 3: Start stitching
            logger.info("\n[3/7] Starting canvas stitching system...")
            await canvas.start_stitching()
            logger.info("  Stitching system ready")
            
            # Step 4: Import tiles
            logger.info(f"\n[4/7] Importing tiles (z-plane={z_plane})...")
            tiles_imported = 0
            tiles_skipped = 0
            last_log_time = time.time()
            last_well = None
            
            # Memory monitoring
            process = psutil.Process()
            initial_memory_mb = process.memory_info().rss / 1024 / 1024
            
            for tile_idx in range(n_tiles):
                # Calculate coordinates
                coords = get_opera_stage_coordinates(tile_idx)
                
                # Skip if not the selected z-plane
                if coords is None:
                    tiles_skipped += 1
                    continue
                
                stage_x, stage_y, well_id, fov_idx, tile_z_idx = coords
                
                # Verify we're processing the correct z-plane
                if tile_z_idx != z_plane:
                    tiles_skipped += 1
                    continue
                
                # Log progress per well
                if well_id != last_well:
                    logger.info(f"  Processing well {well_id}...")
                    last_well = well_id
                
                # Import each requested channel
                for opera_channel_idx in channels_to_import:
                    if opera_channel_idx >= n_channels:
                        logger.warning(f"    Channel {opera_channel_idx} not available in dataset (max: {n_channels-1})")
                        continue
                    
                    # Read tile from Opera Zarr
                    tile_data = zarr_arr[tile_idx, opera_channel_idx, :, :]
                    tile_data = np.asarray(tile_data)
                    
                    # CRITICAL: Normalize uint16 → uint8 following image_processing.py pattern
                    # Opera data is uint16 (typically 12-bit: 0-4095), canvas expects uint8 (0-255)
                    if tile_data.dtype != np.uint8:
                        # Normalize to 0-255 range based on actual data range per tile
                        data_min = tile_data.min()
                        data_max = tile_data.max()
                        
                        if data_max > data_min:
                            # Scale to full uint8 range
                            normalized = ((tile_data - data_min) / (data_max - data_min) * 255).astype(np.uint8)
                        else:
                            # Uniform tile (rare), convert directly
                            normalized = np.clip(tile_data // 256, 0, 255).astype(np.uint8)
                        
                        tile_data = normalized
                    
                    # Map Opera channel to Squid canvas zarr index (standard structure)
                    # Standard canvas order: BF(0), 405(1), 488(2), 638(3), 561(4), 730(5)
                    squid_channel_name = channel_mapping[opera_channel_idx]
                    canvas_zarr_idx = canvas.channel_to_zarr_index[squid_channel_name]
                    
                    # Add to canvas (synchronous for reliability)
                    canvas.add_image_sync(
                        image=tile_data,
                        x_mm=stage_x,
                        y_mm=stage_y,
                        channel_idx=canvas_zarr_idx,
                        z_idx=0,  # Canvas z_idx (not Opera z-plane)
                        timepoint=0
                    )
                
                tiles_imported += 1
                
                # Progress logging every 5 seconds
                current_time = time.time()
                if current_time - last_log_time > 5.0:
                    progress_pct = (tile_idx + 1) / n_tiles * 100
                    memory_mb = process.memory_info().rss / 1024 / 1024
                    memory_delta = memory_mb - initial_memory_mb
                    elapsed = current_time - start_time
                    rate = tiles_imported / elapsed if elapsed > 0 else 0
                    eta_seconds = (tiles_after_z_filter - tiles_imported) / rate if rate > 0 else 0
                    
                    logger.info(f"  Progress: {progress_pct:.1f}% | "
                              f"Imported: {tiles_imported}/{tiles_after_z_filter} | "
                              f"Rate: {rate:.1f} tiles/s | "
                              f"ETA: {eta_seconds/60:.1f} min | "
                              f"Memory: +{memory_delta:.0f} MB")
                    last_log_time = current_time
            
            logger.info(f"  Tile import complete: {tiles_imported} imported, {tiles_skipped} skipped")
            
            # Step 5: Wait for stitching to complete
            logger.info("\n[5/7] Waiting for canvas processing to complete...")
            await canvas.stop_stitching()
            logger.info("  Canvas processing complete")
            
            # Step 6: Save preview if requested (using safe method for high-res canvases)
            if save_preview:
                logger.info("\n[6/7] Saving preview images...")
                try:
                    _save_preview_safe(canvas, action_id="opera_import")
                except Exception as e:
                    logger.warning(f"  Could not save preview: {e}")
            else:
                logger.info("\n[6/7] Skipping preview generation")
            
            # Step 7: Validation
            logger.info("\n[7/7] Validating import...")
            validation_results = validate_import(canvas, tiles_imported, num_wells, tiles_per_well)
            
            # Final statistics
            elapsed_time = time.time() - start_time
            final_memory_mb = process.memory_info().rss / 1024 / 1024
            memory_used = final_memory_mb - initial_memory_mb
            
            logger.info("\n" + "=" * 80)
            logger.info("IMPORT COMPLETE")
            logger.info("=" * 80)
            logger.info(f"Total time: {elapsed_time/60:.1f} minutes")
            logger.info(f"Tiles imported: {tiles_imported}")
            logger.info(f"Processing rate: {tiles_imported/elapsed_time:.2f} tiles/second")
            logger.info(f"Memory used: {memory_used:.0f} MB")
            logger.info(f"Output location: {output_base_path / experiment_name}")
            logger.info("=" * 80)
            
            return {
                'success': True,
                'tiles_imported': tiles_imported,
                'tiles_skipped': tiles_skipped,
                'elapsed_time': elapsed_time,
                'memory_used_mb': memory_used,
                'validation': validation_results,
                'output_path': str(output_base_path / experiment_name)
            }
            
    except Exception as e:
        logger.error(f"Import failed: {e}")
        import traceback
        traceback.print_exc()
        return {
            'success': False,
            'error': str(e),
            'tiles_imported': 0
        }


# ============================================================================
# CLI INTERFACE
# ============================================================================

def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description='Import Opera Phoenix microscopy data into Zarr canvas',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage (import all channels, middle z-plane)
  python scripts/import_opera_to_zarr.py \\
      --opera-tif /path/to/plate1.ome.tif \\
      --output-dir /path/to/experiments \\
      --experiment-name opera_import
  
  # Import specific channels only
  python scripts/import_opera_to_zarr.py \\
      --opera-tif /path/to/plate1.ome.tif \\
      --output-dir /path/to/experiments \\
      --experiment-name opera_dapi_only \\
      --channels 0
  
  # Import different z-plane
  python scripts/import_opera_to_zarr.py \\
      --opera-tif /path/to/plate1.ome.tif \\
      --output-dir /path/to/experiments \\
      --experiment-name opera_top_plane \\
      --z-plane 0
        """
    )
    
    parser.add_argument(
        '--opera-tif',
        type=Path,
        required=True,
        help='Path to Opera Phoenix OME-TIFF file (plate1.ome.tif)'
    )
    
    parser.add_argument(
        '--output-dir',
        type=Path,
        required=True,
        help='Output directory for Zarr experiments'
    )
    
    parser.add_argument(
        '--experiment-name',
        type=str,
        default='opera_import',
        help='Name for the experiment (default: opera_import)'
    )
    
    parser.add_argument(
        '--channels',
        type=str,
        default=None,
        help='Comma-separated list of channel indices to import (e.g., "0,1,2,3"). Default: all channels'
    )
    
    parser.add_argument(
        '--z-plane',
        type=int,
        default=1,
        choices=[0, 1, 2],
        help='Which z-plane to import: 0=top, 1=middle (default), 2=bottom'
    )
    
    parser.add_argument(
        '--no-preview',
        action='store_true',
        help='Skip preview image generation'
    )
    
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Enable verbose logging'
    )
    
    args = parser.parse_args()
    
    # Set logging level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Validate inputs
    if not args.opera_tif.exists():
        logger.error(f"Opera TIFF file not found: {args.opera_tif}")
        return 1
    
    # Create output directory if needed
    args.output_dir.mkdir(parents=True, exist_ok=True)
    
    # Parse channels
    channels_to_import = None
    if args.channels:
        try:
            channels_to_import = [int(c.strip()) for c in args.channels.split(',')]
            logger.info(f"Importing channels: {channels_to_import}")
        except ValueError:
            logger.error(f"Invalid channel list: {args.channels}")
            return 1
    
    # Run import
    try:
        result = asyncio.run(import_opera_dataset(
            opera_tif_path=args.opera_tif,
            output_base_path=args.output_dir,
            experiment_name=args.experiment_name,
            z_plane=args.z_plane,
            channels_to_import=channels_to_import,
            save_preview=not args.no_preview
        ))
        
        if result['success']:
            logger.info("\n✓ Import succeeded")
            return 0
        else:
            logger.error(f"\n✗ Import failed: {result.get('error', 'Unknown error')}")
            return 1
            
    except KeyboardInterrupt:
        logger.warning("\nImport interrupted by user")
        return 1
    except Exception as e:
        logger.error(f"\nUnexpected error: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
