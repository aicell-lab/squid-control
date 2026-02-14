#!/usr/bin/env python3
"""
Validate and visualize Opera Phoenix import results.

This script:
1. Validates the imported Zarr canvas
2. Creates preview images from different scales
3. Generates a spatial map showing tile locations
4. Reports statistics on import completeness

Usage:
    python scripts/validate_opera_import.py \
        --experiment-dir /mnt/shared_documents/opera_import \
        --output-dir ./validation_results
"""

import argparse
import sys
from pathlib import Path
import logging

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

import zarr
import numpy as np
from PIL import Image
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


def validate_zarr_structure(zarr_path: Path) -> dict:
    """
    Validate the Zarr canvas structure and report statistics.
    
    Returns dict with validation results.
    """
    logger.info("=" * 80)
    logger.info("ZARR STRUCTURE VALIDATION")
    logger.info("=" * 80)
    
    validation = {
        'valid': False,
        'path': str(zarr_path),
        'scales': [],
        'channels': 0,
        'timepoints': 0,
        'errors': []
    }
    
    if not zarr_path.exists():
        validation['errors'].append(f"Zarr path does not exist: {zarr_path}")
        return validation
    
    try:
        # Open zarr group
        zgroup = zarr.open(str(zarr_path), mode='r')
        logger.info(f"✓ Opened Zarr group: {zarr_path}")
        
        # Check each scale
        for scale_num in range(10):  # Check up to scale 9
            scale_key = str(scale_num)
            if scale_key in zgroup:
                arr = zgroup[scale_key]
                validation['scales'].append({
                    'scale': scale_num,
                    'shape': arr.shape,
                    'dtype': str(arr.dtype),
                    'chunks': arr.chunks
                })
                
                if scale_num == 0:
                    # Extract metadata from scale 0
                    validation['channels'] = arr.shape[1]
                    validation['timepoints'] = arr.shape[0]
                    validation['z_planes'] = arr.shape[2]
                    validation['height_px'] = arr.shape[3]
                    validation['width_px'] = arr.shape[4]
                
                logger.info(f"  Scale {scale_num}: shape={arr.shape}, dtype={arr.dtype}")
        
        if not validation['scales']:
            validation['errors'].append("No scales found in Zarr group")
            return validation
        
        # Check for non-empty data
        scale0 = zgroup['0']
        sample_data = scale0[0, 0, 0, :100, :100]
        data_range = (float(np.min(sample_data)), float(np.max(sample_data)))
        validation['data_range'] = data_range
        
        if data_range[0] == data_range[1]:
            validation['errors'].append("Canvas appears empty (no data variance)")
        else:
            logger.info(f"✓ Data present: range={data_range}")
        
        validation['valid'] = len(validation['errors']) == 0
        
    except Exception as e:
        validation['errors'].append(f"Error validating Zarr: {e}")
        logger.error(f"✗ Validation error: {e}")
    
    return validation


def create_scale_previews(zarr_path: Path, output_dir: Path, scales_to_preview: list = [3, 4, 5]):
    """
    Create preview images from different scales.
    Uses higher scales (3-5) to avoid OOM errors on high-resolution canvases.
    """
    logger.info("\n" + "=" * 80)
    logger.info("GENERATING SCALE PREVIEWS")
    logger.info("=" * 80)
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        zgroup = zarr.open(str(zarr_path), mode='r')
        
        for scale in scales_to_preview:
            scale_key = str(scale)
            if scale_key not in zgroup:
                logger.warning(f"  Scale {scale} not found, skipping")
                continue
            
            try:
                arr = zgroup[scale_key]
                height, width = arr.shape[3], arr.shape[4]
                logger.info(f"\nProcessing scale {scale}: {width} × {height} pixels")
                
                # Read first channel
                image_data = arr[0, 0, 0, :, :]
                image_data = np.asarray(image_data)
                
                # Check for data
                if image_data.max() == image_data.min():
                    logger.warning(f"  Scale {scale}: no data variance")
                    continue
                
                # Normalize to 0-255
                data_min = image_data.min()
                data_max = image_data.max()
                logger.info(f"  Data range: [{data_min}, {data_max}]")
                
                normalized = ((image_data - data_min) / (data_max - data_min) * 255).astype(np.uint8)
                
                # Save preview
                image = Image.fromarray(normalized)
                preview_path = output_dir / f"preview_scale{scale}.png"
                image.save(preview_path)
                logger.info(f"  ✓ Saved: {preview_path}")
                
                # Also create a thumbnail
                thumb_size = (1024, 1024)
                image.thumbnail(thumb_size, Image.Resampling.LANCZOS)
                thumb_path = output_dir / f"preview_scale{scale}_thumb.png"
                image.save(thumb_path)
                logger.info(f"  ✓ Saved thumbnail: {thumb_path}")
                
            except MemoryError as e:
                logger.error(f"  Scale {scale}: Memory error - {e}")
                continue
            except Exception as e:
                logger.error(f"  Scale {scale}: Error - {e}")
                continue
                
    except Exception as e:
        logger.error(f"Failed to generate previews: {e}")


def create_spatial_map(zarr_path: Path, output_dir: Path, validation: dict):
    """
    Create a spatial map showing where data exists in the canvas.
    Samples the canvas at regular intervals to show coverage.
    """
    logger.info("\n" + "=" * 80)
    logger.info("GENERATING SPATIAL COVERAGE MAP")
    logger.info("=" * 80)
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        zgroup = zarr.open(str(zarr_path), mode='r')
        
        # Use highest scale for fast sampling
        max_scale = len(validation['scales']) - 1
        arr = zgroup[str(max_scale)]
        
        logger.info(f"Using scale {max_scale} for spatial map")
        logger.info(f"  Shape: {arr.shape}")
        
        # Read the full array at this scale (should be small)
        data = arr[0, 0, 0, :, :]
        data = np.asarray(data)
        
        # Create figure
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
        
        # Plot 1: Actual data
        im1 = ax1.imshow(data, cmap='gray', interpolation='nearest')
        ax1.set_title(f'Canvas Data (Scale {max_scale})', fontsize=14, fontweight='bold')
        ax1.set_xlabel('X (pixels)', fontsize=12)
        ax1.set_ylabel('Y (pixels)', fontsize=12)
        plt.colorbar(im1, ax=ax1, label='Intensity')
        
        # Plot 2: Coverage map (binary: data vs no data)
        coverage = (data > 0).astype(np.uint8)
        coverage_percent = (coverage.sum() / coverage.size) * 100
        
        im2 = ax2.imshow(coverage, cmap='RdYlGn', interpolation='nearest', vmin=0, vmax=1)
        ax2.set_title(f'Coverage Map ({coverage_percent:.1f}% filled)', fontsize=14, fontweight='bold')
        ax2.set_xlabel('X (pixels)', fontsize=12)
        ax2.set_ylabel('Y (pixels)', fontsize=12)
        plt.colorbar(im2, ax=ax2, label='Has Data', ticks=[0, 1])
        
        # Add text info
        info_text = (
            f"Canvas: {validation['width_px']} × {validation['height_px']} px\n"
            f"Channels: {validation['channels']}\n"
            f"Timepoints: {validation['timepoints']}\n"
            f"Z-planes: {validation['z_planes']}\n"
            f"Scales: {len(validation['scales'])}"
        )
        fig.text(0.02, 0.98, info_text, transform=fig.transFigure,
                fontsize=10, verticalalignment='top', family='monospace',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        plt.tight_layout()
        
        # Save
        map_path = output_dir / 'spatial_coverage_map.png'
        plt.savefig(map_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        logger.info(f"✓ Saved spatial map: {map_path}")
        logger.info(f"  Coverage: {coverage_percent:.1f}% of canvas has data")
        
    except Exception as e:
        logger.error(f"Failed to create spatial map: {e}")


def generate_report(validation: dict, output_dir: Path):
    """Generate a text report of validation results."""
    logger.info("\n" + "=" * 80)
    logger.info("GENERATING VALIDATION REPORT")
    logger.info("=" * 80)
    
    report_path = output_dir / 'validation_report.txt'
    
    with open(report_path, 'w') as f:
        f.write("=" * 80 + "\n")
        f.write("OPERA PHOENIX IMPORT VALIDATION REPORT\n")
        f.write("=" * 80 + "\n\n")
        
        f.write(f"Zarr Path: {validation['path']}\n")
        f.write(f"Valid: {validation['valid']}\n\n")
        
        if validation['valid']:
            f.write(f"Canvas Dimensions: {validation['width_px']} × {validation['height_px']} pixels\n")
            f.write(f"Channels: {validation['channels']}\n")
            f.write(f"Timepoints: {validation['timepoints']}\n")
            f.write(f"Z-planes: {validation['z_planes']}\n")
            f.write(f"Number of scales: {len(validation['scales'])}\n")
            f.write(f"Data range: {validation.get('data_range', 'N/A')}\n\n")
            
            f.write("Scale Information:\n")
            f.write("-" * 80 + "\n")
            for scale_info in validation['scales']:
                f.write(f"  Scale {scale_info['scale']}:\n")
                f.write(f"    Shape: {scale_info['shape']}\n")
                f.write(f"    Dtype: {scale_info['dtype']}\n")
                f.write(f"    Chunks: {scale_info['chunks']}\n")
        
        if validation['errors']:
            f.write("\nErrors:\n")
            f.write("-" * 80 + "\n")
            for error in validation['errors']:
                f.write(f"  • {error}\n")
        
        f.write("\n" + "=" * 80 + "\n")
    
    logger.info(f"✓ Saved report: {report_path}")


def main():
    parser = argparse.ArgumentParser(description='Validate Opera Phoenix import')
    parser.add_argument('--experiment-dir', type=Path, required=True,
                      help='Path to experiment directory containing data.zarr')
    parser.add_argument('--output-dir', type=Path, default=Path('./validation_results'),
                      help='Output directory for validation results')
    parser.add_argument('--scales', type=int, nargs='+', default=[3, 4, 5],
                      help='Scales to generate previews for (default: 3 4 5)')
    
    args = parser.parse_args()
    
    # Find zarr path
    zarr_path = args.experiment_dir / 'data.zarr'
    
    if not zarr_path.exists():
        logger.error(f"Zarr path not found: {zarr_path}")
        return 1
    
    logger.info("=" * 80)
    logger.info("OPERA PHOENIX IMPORT VALIDATION")
    logger.info("=" * 80)
    logger.info(f"Experiment: {args.experiment_dir}")
    logger.info(f"Output: {args.output_dir}")
    logger.info("=" * 80)
    
    # Create output directory
    args.output_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. Validate structure
    validation = validate_zarr_structure(zarr_path)
    
    if not validation['valid']:
        logger.error("\n✗ Validation failed!")
        for error in validation['errors']:
            logger.error(f"  • {error}")
        return 1
    
    logger.info("\n✓ Validation passed!")
    
    # 2. Create previews
    create_scale_previews(zarr_path, args.output_dir, args.scales)
    
    # 3. Create spatial map
    create_spatial_map(zarr_path, args.output_dir, validation)
    
    # 4. Generate report
    generate_report(validation, args.output_dir)
    
    logger.info("\n" + "=" * 80)
    logger.info("VALIDATION COMPLETE")
    logger.info("=" * 80)
    logger.info(f"Results saved to: {args.output_dir}")
    
    return 0


if __name__ == '__main__':
    sys.exit(main())
