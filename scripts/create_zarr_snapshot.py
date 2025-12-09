#!/usr/bin/env python3
"""
Create a snapshot image from a Zarr dataset at a specific scale level.

Usage:
    python create_zarr_snapshot.py <zarr_path> [scale_level] [output_path] [channel_idx]

Example:
    python create_zarr_snapshot.py /data/offline_stitch_20251201-u2os-full-plate_2025-12-01_17-00-56.154975/data.zarr 4 snapshot_scale4.png
"""

import sys
import zarr
import numpy as np
from pathlib import Path
from PIL import Image
import argparse


def create_snapshot(zarr_path: str, scale_level: int = 4, output_path: str = None, 
                   channel_idx: int = 0, timepoint: int = 0, z_idx: int = 0):
    """
    Create a snapshot image from a Zarr dataset at a specific scale level.
    
    Args:
        zarr_path: Path to the Zarr dataset directory
        scale_level: Scale level (0=full, 1=1/4, 2=1/16, 3=1/64, 4=1/256, 5=1/1024)
        output_path: Output image path (default: snapshot_scale{scale_level}.png)
        channel_idx: Channel index to use (default: 0 for BF LED matrix full)
        timepoint: Timepoint index (default: 0)
        z_idx: Z-slice index (default: 0)
    """
    zarr_path = Path(zarr_path)
    
    # Open Zarr group
    print(f"Opening Zarr dataset at: {zarr_path}")
    root = zarr.open_group(str(zarr_path), mode='r')
    
    # Load metadata
    zattrs = dict(root.attrs)
    squid_canvas = zattrs.get('squid_canvas', {})
    channels = squid_canvas.get('channel_mapping', {})
    
    # Get available channels
    if channels:
        channel_names = list(channels.keys())
        print(f"Available channels: {channel_names}")
        if channel_idx < len(channel_names):
            print(f"Using channel {channel_idx}: {channel_names[channel_idx]}")
    else:
        print(f"Using channel index {channel_idx} (channel mapping not found in metadata)")
    
    # Check if scale exists
    scale_str = str(scale_level)
    if scale_str not in root:
        available_scales = sorted([k for k in root.keys() if k.isdigit()])
        print(f"Error: Scale {scale_level} not found. Available scales: {available_scales}")
        return False
    
    # Open scale array
    arr = root[scale_str]
    shape = arr.shape  # [T, C, Z, Y, X]
    print(f"Scale {scale_level} array shape: {shape} (T, C, Z, Y, X)")
    
    # Validate indices
    if timepoint >= shape[0]:
        print(f"Warning: Timepoint {timepoint} >= {shape[0]}, using timepoint 0")
        timepoint = 0
    
    if channel_idx >= shape[1]:
        print(f"Warning: Channel {channel_idx} >= {shape[1]}, using channel 0")
        channel_idx = 0
    
    if z_idx >= shape[2]:
        print(f"Warning: Z-slice {z_idx} >= {shape[2]}, using z-slice 0")
        z_idx = 0
    
    # Read full image at this scale
    print(f"Reading full image at scale {scale_level} (timepoint={timepoint}, channel={channel_idx}, z={z_idx})...")
    image = arr[timepoint, channel_idx, z_idx, :, :]
    
    # Convert to numpy array if needed
    if hasattr(image, 'compute'):
        image = image.compute()
    image = np.array(image)
    
    print(f"Image shape: {image.shape}, dtype: {image.dtype}")
    print(f"Image value range: [{image.min()}, {image.max()}]")
    
    # Normalize to 0-255 if needed
    if image.dtype != np.uint8:
        if image.max() > 255:
            # Scale down from uint16 or higher
            if image.dtype == np.uint16:
                image = (image / 256).astype(np.uint8)
            else:
                # Normalize float or other types
                if image.max() > image.min():
                    image = ((image - image.min()) / (image.max() - image.min()) * 255).astype(np.uint8)
                else:
                    image = np.zeros_like(image, dtype=np.uint8)
        else:
            image = image.astype(np.uint8)
    
    # Determine output path
    if output_path is None:
        zarr_name = zarr_path.name if zarr_path.is_dir() else zarr_path.stem
        output_path = f"snapshot_{zarr_name}_scale{scale_level}.png"
    
    output_path = Path(output_path)
    
    # Save as PNG
    print(f"Saving snapshot to: {output_path}")
    img_pil = Image.fromarray(image, mode='L')  # Grayscale
    img_pil.save(output_path, format='PNG')
    
    # Print summary
    scale_factor = 4 ** scale_level
    print(f"\n✅ Snapshot created successfully!")
    print(f"   Output: {output_path}")
    print(f"   Scale: {scale_level} (1/{scale_factor} resolution)")
    print(f"   Image size: {image.shape[1]}x{image.shape[0]} pixels")
    print(f"   File size: {output_path.stat().st_size / 1024 / 1024:.2f} MB")
    
    return True


def main():
    parser = argparse.ArgumentParser(
        description='Create a snapshot image from a Zarr dataset at a specific scale level',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Create snapshot at scale 4 (default)
  python create_zarr_snapshot.py /data/offline_stitch_20251201-u2os-full-plate_2025-12-01_17-00-56.154975/data.zarr

  # Specify scale level and output path
  python create_zarr_snapshot.py /data/offline_stitch_20251201-u2os-full-plate_2025-12-01_17-00-56.154975/data.zarr 4 snapshot.png

  # Use a specific channel
  python create_zarr_snapshot.py /data/offline_stitch_20251201-u2os-full-plate_2025-12-01_17-00-56.154975/data.zarr 4 snapshot.png 0
        """
    )
    
    parser.add_argument('zarr_path', type=str, 
                       help='Path to the Zarr dataset directory')
    parser.add_argument('scale_level', type=int, nargs='?', default=4,
                       help='Scale level (0=full, 1=1/4, 2=1/16, 3=1/64, 4=1/256, 5=1/1024) [default: 4]')
    parser.add_argument('output_path', type=str, nargs='?', default=None,
                       help='Output image path [default: snapshot_<name>_scale<level>.png]')
    parser.add_argument('channel_idx', type=int, nargs='?', default=0,
                       help='Channel index to use [default: 0]')
    parser.add_argument('--timepoint', type=int, default=0,
                       help='Timepoint index [default: 0]')
    parser.add_argument('--z', type=int, default=0, dest='z_idx',
                       help='Z-slice index [default: 0]')
    
    args = parser.parse_args()
    
    # Check if zarr path exists
    zarr_path = Path(args.zarr_path)
    if not zarr_path.exists():
        print(f"Error: Zarr path does not exist: {zarr_path}")
        return 1
    
    # Create snapshot
    success = create_snapshot(
        str(zarr_path),
        scale_level=args.scale_level,
        output_path=args.output_path,
        channel_idx=args.channel_idx,
        timepoint=args.timepoint,
        z_idx=args.z_idx
    )
    
    return 0 if success else 1


if __name__ == '__main__':
    sys.exit(main())

