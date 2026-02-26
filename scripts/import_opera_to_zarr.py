#!/usr/bin/env python3
"""
Import Opera Phoenix microscopy dataset into coordinate-based Zarr canvas.

Converts Opera Phoenix OME-TIFF data (96 wells, 9 FoV/well, 3 z-planes) into
multi-scale OME-Zarr format compatible with squid microscope system.

Usage:
    python scripts/import_opera_to_zarr.py --plates 1
    python scripts/import_opera_to_zarr.py --plates 1 2 3 --verbose
    python scripts/import_opera_to_zarr.py --plates 1 --config custom_config.ini

Args:
    --data-dir      Directory with plate*.ome.tif files
    --output-dir    Output directory for zarr canvases
    --config        Squid config file for wellplate offsets
    --plates        Plate numbers to import (auto-discovers if omitted)
    --verbose       Enable debug logging
Dataset: Opera Phoenix 63X, 96-well plate geometry
"""

import argparse
import asyncio
from configparser import ConfigParser
import logging
import sys
import time
from pathlib import Path
from typing import Dict, Optional, Tuple

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

# 96-well plate geometry constants (matches WELLPLATE_FORMAT_96 in config.py)
_WELLPLATE_96_A1_X_MM = 14.3
_WELLPLATE_96_A1_Y_MM = 11.36
_WELLPLATE_96_WELL_SPACING_MM = 9.0


def read_wellplate_config(config_path: Path) -> Tuple[float, float]:
    """
    Read wellplate_offset_x_mm and wellplate_offset_y_mm from a squid config ini file.

    Uses the same values that move_to_well() applies so that imported tile coordinates
    match the stage positions the microscope will visit.

    Args:
        config_path: Path to a squid configuration .ini file

    Returns:
        (offset_x_mm, offset_y_mm)
    """
    cfp = ConfigParser()
    cfp.read(str(config_path))
    offset_x = float(cfp.get('GENERAL', 'wellplate_offset_x_mm', fallback='0'))
    offset_y = float(cfp.get('GENERAL', 'wellplate_offset_y_mm', fallback='0'))
    return offset_x, offset_y


def get_opera_stage_coordinates(
    tile_idx: int,
    a1_x_mm: float,
    a1_y_mm: float,
    well_spacing_mm: float = _WELLPLATE_96_WELL_SPACING_MM,
) -> Optional[Tuple[float, float, str, int, int]]:
    """
    Calculate stage coordinates for Opera Phoenix tile.

    The Opera Phoenix dataset has:
    - 96 wells in standard 8×12 layout (A1-H12)
    - 9 FoV per well (3x3 grid)
    - 3 z-planes per FoV (we use middle plane, z_idx=1)

    Tile organization: tile_idx = (well_idx * 27) + (fov_idx * 3) + z_idx

    Coordinate calculation mirrors move_to_well() in squid_controller.py:
        x_mm = A1_X_MM + (col - 1) * WELL_SPACING_MM + wellplate_offset_x
        y_mm = A1_Y_MM + row_index * WELL_SPACING_MM + wellplate_offset_y
    where a1_x_mm / a1_y_mm already include the wellplate offset.

    Args:
        tile_idx: Tile index (0 to 2591)
        a1_x_mm: Stage X coordinate of well A1 centre (base + offset from config)
        a1_y_mm: Stage Y coordinate of well A1 centre (base + offset from config)
        well_spacing_mm: Centre-to-centre well spacing (default 9.0 mm for 96-well)

    Returns:
        (x_mm, y_mm, well_id, fov_idx, z_idx) or None if not middle z-plane
    """
    FOV_STEP_MM = 0.042  # Effective FoV spacing with ~10% overlap
    WELLS_PER_ROW = 12   # 96-well plate has 12 columns

    # Parse tile structure: well_idx × 27 + fov_idx × 3 + z_idx
    well_idx = tile_idx // 27  # Which well (0-95)
    fov_idx = (tile_idx % 27) // 3  # Which FoV within well (0-8)
    z_idx = tile_idx % 3  # Which z-plane (0, 1, 2)

    # Filter: only process middle z-plane
    if z_idx != 1:
        return None

    # Skip H12 (well_idx=95) — always empty across all plates
    if well_idx == 95:
        return None

    # Calculate well position (row-major order: 8 rows × 12 columns)
    well_row = well_idx // WELLS_PER_ROW  # 0-7 (A-H)
    well_col = well_idx % WELLS_PER_ROW   # 0-11 (1-12)
    well_id = f"{chr(ord('A') + well_row)}{well_col + 1}"

    # Well centre in stage coordinates — same formula as move_to_well():
    #   x_mm = A1_X_MM + (column - 1) * WELL_SPACING_MM + offset
    # a1_x_mm already incorporates the offset, and well_col is 0-based (column-1)
    well_center_x = a1_x_mm + well_col * well_spacing_mm
    well_center_y = a1_y_mm + well_row * well_spacing_mm

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
    expected_fov_per_well: int = 9,
    a1_x_mm: float = _WELLPLATE_96_A1_X_MM,
    a1_y_mm: float = _WELLPLATE_96_A1_Y_MM,
    well_spacing_mm: float = _WELLPLATE_96_WELL_SPACING_MM,
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
    logger.info(f"  X: {a1_x_mm:.3f} to {a1_x_mm + 11 * well_spacing_mm:.3f} mm (12 columns × {well_spacing_mm} mm spacing)")
    logger.info(f"  Y: {a1_y_mm:.3f} to {a1_y_mm + 7 * well_spacing_mm:.3f} mm (8 rows × {well_spacing_mm} mm spacing)")
    
    logger.info("=" * 80)
    
    return validation_results


DEFAULT_DATA_DIR   = Path('/media/reef/harddisk/immunofluorescence_data/14315777')
DEFAULT_OUTPUT_DIR = Path('/mnt/shared_documents')
DEFAULT_CONFIG     = Path(__file__).parent.parent / 'squid_control/config/configuration_HCS_v2_example.ini'

# ============================================================================
# MAIN IMPORT FUNCTION
# ============================================================================

async def import_opera_dataset(
    plate_num: int,
    data_dir: Path = DEFAULT_DATA_DIR,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    channel_mapping: Optional[Dict[int, str]] = None,
    config_path: Optional[Path] = None,
) -> Dict:
    """
    Import one Opera Phoenix plate into a coordinate-based Zarr canvas.

    Output: {output_dir}/plate{plate_num}.zarr

    Tile stage coordinates mirror move_to_well() in squid_controller.py.
    Well H12 (well_idx=95) is always empty and skipped automatically.
    Only the middle z-plane (z_idx=1) is imported.
    """
    start_time = time.time()

    tif_path = data_dir / f'plate{plate_num}.ome.tif'
    if not tif_path.exists():
        return {'success': False, 'error': f'TIFF not found: {tif_path}', 'tiles_imported': 0}

    # Resolve config path
    if config_path is None:
        config_path = DEFAULT_CONFIG

    # Read wellplate offsets (same values used by move_to_well())
    offset_x_mm, offset_y_mm = read_wellplate_config(config_path)
    a1_x_mm = _WELLPLATE_96_A1_X_MM + offset_x_mm
    a1_y_mm = _WELLPLATE_96_A1_Y_MM + offset_y_mm
    logger.info(f"plate{plate_num}: config={config_path.name}, "
                f"offset=({offset_x_mm}, {offset_y_mm}), A1=({a1_x_mm:.3f}, {a1_y_mm:.3f}) mm")

    # Default channel mapping (Opera channels → Squid channel names)
    if channel_mapping is None:
        channel_mapping = {
            0: 'Fluorescence 405 nm Ex',   # DAPI
            1: 'Fluorescence 488 nm Ex',   # Target protein
            2: 'Fluorescence 638 nm Ex',   # SARS-CoV-2 virus
            3: 'Fluorescence 561 nm Ex',   # ER
        }

    channels_to_import = list(channel_mapping.keys())

    logger.info("=" * 60)
    logger.info(f"OPERA IMPORT  plate{plate_num}")
    logger.info(f"  Input:    {tif_path}")
    logger.info(f"  Output:   {output_dir}/plate{plate_num}.zarr")
    logger.info("=" * 60)

    try:
        with tifffile.TiffFile(tif_path) as tif:
            store = tif.aszarr()
            zgroup = zarr.open(store, mode="r")
            zarr_arr = zgroup["0"]
            n_tiles = zarr_arr.shape[0]
            tiles_per_well = 9
            num_wells = (n_tiles // 3) // tiles_per_well   # 3 z-planes
            fovs_expected = (num_wells - 1) * tiles_per_well  # -1 for empty H12
            logger.info(f"  TIFF shape: {zarr_arr.shape}  ({num_wells} wells × {tiles_per_well} FOV × 3 z)")

            # Canvas
            stage_limits = {
                'x_positive': 120.0, 'x_negative': 0.0,
                'y_positive': 80.0,  'y_negative': 0.0,
            }
            exp_manager = ExperimentManager(
                base_path=str(output_dir),
                pixel_size_xy_um=0.0435,
                stage_limits=stage_limits,
            )
            exp_manager.create_experiment(
                experiment_name=f'plate{plate_num}',
                well_plate_type='96',
                initialize_all_wells=False,
            )
            canvas = exp_manager.get_canvas(initialize_new=True)
            logger.info(f"  Canvas: {canvas.canvas_width_px} × {canvas.canvas_height_px} px, "
                        f"channels: {canvas.channels}")

            await canvas.start_stitching()

            tiles_imported = 0
            tiles_skipped = 0
            last_log_time = time.time()
            last_well = None

            for tile_idx in range(n_tiles):
                coords = get_opera_stage_coordinates(tile_idx, a1_x_mm, a1_y_mm)
                if coords is None:          # wrong z-plane or empty H12
                    tiles_skipped += 1
                    continue

                stage_x, stage_y, well_id, fov_idx, _ = coords

                if well_id != last_well:
                    logger.info(f"  Processing well {well_id}…")
                    last_well = well_id

                for opera_ch in channels_to_import:
                    tile_data = np.asarray(zarr_arr[tile_idx, opera_ch, :, :])

                    # Normalize uint16 → uint8
                    if tile_data.dtype != np.uint8:
                        d_min, d_max = tile_data.min(), tile_data.max()
                        if d_max > d_min:
                            tile_data = ((tile_data - d_min) / (d_max - d_min) * 255).astype(np.uint8)
                        else:
                            tile_data = np.clip(tile_data // 256, 0, 255).astype(np.uint8)

                    canvas_ch = canvas.channel_to_zarr_index[channel_mapping[opera_ch]]
                    canvas.add_image_sync(
                        image=tile_data, x_mm=stage_x, y_mm=stage_y,
                        channel_idx=canvas_ch, z_idx=0, timepoint=0,
                    )

                tiles_imported += 1

                now = time.time()
                if now - last_log_time > 5.0:
                    elapsed = now - start_time
                    rate = tiles_imported / elapsed if elapsed > 0 else 1
                    eta = (fovs_expected - tiles_imported) / rate
                    logger.info(f"  {tiles_imported}/{fovs_expected} FOVs | "
                                f"{elapsed:.0f}s elapsed | ETA ~{eta:.0f}s")
                    last_log_time = now

            await canvas.stop_stitching()

            elapsed_time = time.time() - start_time
            logger.info("=" * 60)
            logger.info(f"plate{plate_num}: done — {tiles_imported} FOVs in {elapsed_time/60:.1f} min")
            logger.info("=" * 60)

            return {
                'success': True,
                'plate': plate_num,
                'tiles_imported': tiles_imported,
                'tiles_skipped': tiles_skipped,
                'elapsed_time': elapsed_time,
                'output_path': str(output_dir / f'plate{plate_num}'),
            }

    except Exception as e:
        logger.error(f"plate{plate_num}: import failed: {e}")
        import traceback
        traceback.print_exc()
        return {'success': False, 'error': str(e), 'tiles_imported': 0}


# ============================================================================
# CLI INTERFACE
# ============================================================================

def main() -> int:
    parser = argparse.ArgumentParser(
        description='Import Opera Phoenix plates into Zarr canvases'
    )
    parser.add_argument(
        '--data-dir', type=Path, default=DEFAULT_DATA_DIR,
        help='Directory containing plate*.ome.tif files',
    )
    parser.add_argument(
        '--output-dir', type=Path, default=DEFAULT_OUTPUT_DIR,
        help='Output directory (creates plate{N}.zarr inside)',
    )
    parser.add_argument(
        '--config', type=Path, default=None,
        help='Squid .ini config file for wellplate offsets',
    )
    parser.add_argument(
        '--plates', type=int, nargs='+', default=None,
        help='Plate numbers to import (default: all found in --data-dir)',
    )
    parser.add_argument('--verbose', action='store_true')
    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # Auto-discover available plates
    available = sorted(
        int(p.with_suffix('').stem.replace('plate', ''))
        for p in args.data_dir.glob('plate*.ome.tif')
    )
    plates = args.plates if args.plates else available
    if not plates:
        logger.error(f'No plate*.ome.tif files found in {args.data_dir}')
        return 1

    logger.info(f'Plates to import: {plates}')
    args.output_dir.mkdir(parents=True, exist_ok=True)

    results = []
    for plate_num in plates:
        result = asyncio.run(import_opera_dataset(
            plate_num=plate_num,
            data_dir=args.data_dir,
            output_dir=args.output_dir,
            config_path=args.config,
        ))
        results.append(result)

    logger.info('=' * 60)
    logger.info('SUMMARY')
    for r in results:
        if r.get('success'):
            logger.info(f"  plate{r['plate']}: {r['tiles_imported']} FOVs "
                        f"in {r['elapsed_time']/60:.1f} min → {r['output_path']}")
        else:
            logger.info(f"  plate{r.get('plate', '?')}: FAILED — {r.get('error')}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
