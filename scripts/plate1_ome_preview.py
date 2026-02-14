#!/usr/bin/env python3
"""
Quick preview of plate1.ome.tif (Zarr-based) and plate1.ome.tif_offsets.json:
- Export a stitched overview screenshot from tiles
- Print and plot plate/offset information

Usage:
  python scripts/plate1_ome_preview.py
  python scripts/plate1_ome_preview.py --dir /media/reef/harddisk/immunofluorescence_data/14315777
  python scripts/plate1_ome_preview.py --tif path/to/plate1.ome.tif --offsets path/to/plate1.ome.tif_offsets.json
"""

import argparse
import json
from pathlib import Path

import numpy as np
import tifffile
import zarr

# Matplotlib: use non-interactive backend so it works headless
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def find_plate1_files(dir_path: Path) -> tuple[Path | None, Path | None]:
    """Return (path to plate1.ome.tif, path to plate1.ome.tif_offsets.json) if found."""
    tif = dir_path / "plate1.ome.tif"
    offsets = dir_path / "plate1.ome.tif_offsets.json"
    if tif.exists() and offsets.exists():
        return tif, offsets
    # Also check current dir
    tif_c = Path("plate1.ome.tif")
    off_c = Path("plate1.ome.tif_offsets.json")
    if tif_c.exists() and off_c.exists():
        return tif_c, off_c
    return None, None


def load_offsets(path: Path) -> dict | list:
    """Load and return the offsets JSON."""
    with open(path) as f:
        return json.load(f)


def create_tile_montage(
    tif_path: Path,
    max_tiles: int = 100,
    downsample: int = 4,
    grid_cols: int = 10,
) -> np.ndarray:
    """
    Create a montage (grid) of tiles from the Zarr array.
    Zarr shape: (T, C, Y, X) where T = number of tiles/timepoints.
    Returns a 2D numpy array showing a grid of tiles.
    """
    # Open OME-TIFF with tifffile and access Zarr store
    try:
        with tifffile.TiffFile(tif_path) as tif:
            if not hasattr(tif, 'aszarr'):
                raise ValueError("TIFF does not have Zarr storage")
            store = tif.aszarr()
            zgroup = zarr.open(store, mode="r")
    except Exception as e:
        raise ValueError(f"Could not open OME-TIFF as Zarr: {e}")

    # Find the image array (scale 0)
    if "0" in zgroup:
        zarr_arr = zgroup["0"]
    elif hasattr(zgroup, "shape"):
        zarr_arr = zgroup
    else:
        raise ValueError("Could not find image array in Zarr group")

    print(f"  Zarr shape: {zarr_arr.shape}, dtype: {zarr_arr.dtype}, chunks: {zarr_arr.chunks}")
    
    # Zarr shape: (T, C, Y, X) = (2592 tiles, 4 channels, 1080, 1080)
    n_tiles, n_channels, tile_h, tile_w = zarr_arr.shape
    n_tiles_to_show = min(n_tiles, max_tiles)
    
    print(f"  Creating montage of {n_tiles_to_show} tiles (from {n_tiles} total)")
    print(f"  Using channel 0, downsampling {downsample}x")
    
    # Calculate grid dimensions
    grid_rows = (n_tiles_to_show + grid_cols - 1) // grid_cols
    
    # Downsampled tile size
    ds_tile_h = tile_h // downsample
    ds_tile_w = tile_w // downsample
    
    # Create canvas
    canvas_h = grid_rows * ds_tile_h
    canvas_w = grid_cols * ds_tile_w
    canvas = np.zeros((canvas_h, canvas_w), dtype=np.uint16)
    
    # Fill canvas with tiles
    for idx in range(n_tiles_to_show):
        row = idx // grid_cols
        col = idx % grid_cols
        
        try:
            # Read tile: first channel (0), downsampled
            tile = zarr_arr[idx, 0, ::downsample, ::downsample]
            tile = np.squeeze(tile)
            
            # Place in canvas
            y_start = row * ds_tile_h
            x_start = col * ds_tile_w
            canvas[y_start:y_start + ds_tile_h, x_start:x_start + ds_tile_w] = tile
        except Exception as e:
            print(f"    Warning: Could not read tile {idx}: {e}")
            continue
    
    return canvas


def get_zarr_info(tif_path: Path) -> dict:
    """Extract Zarr metadata and shape info from OME-TIFF."""
    info = {"path": str(tif_path), "shape": None, "dtype": None, "chunks": None, "keys": []}
    try:
        with tifffile.TiffFile(tif_path) as tif:
            if hasattr(tif, 'aszarr'):
                store = tif.aszarr()
                zgroup = zarr.open(store, mode="r")
                if "0" in zgroup:
                    zarr_arr = zgroup["0"]
                    info["shape"] = zarr_arr.shape
                    info["dtype"] = str(zarr_arr.dtype)
                    info["chunks"] = zarr_arr.chunks
                info["keys"] = list(zgroup.keys())[:20]
            else:
                info["error"] = "No Zarr storage in TIFF"
    except Exception as e:
        info["error"] = str(e)
    return info


def plot_offsets_info(offsets_data: dict | list, out_path: Path) -> None:
    """
    Plot plate/offset information. Handles common structures:
    - List of dicts with x, y (or row, col, or position_x, position_y)
    - Dict with 'positions' or 'wells' or 'offsets' key
    """
    points = []  # list of (x, y) or (col, row)
    labels = []

    def collect_xy(items, x_key="x", y_key="y"):
        for i, item in enumerate(items):
            if not isinstance(item, dict):
                continue
            x = item.get("x") or item.get("position_x") or item.get("col")
            y = item.get("y") or item.get("position_y") or item.get("row")
            if x is not None and y is not None:
                points.append((float(x), float(y)))
                labels.append(item.get("well_id") or item.get("name") or str(i))

    if isinstance(offsets_data, list):
        collect_xy(offsets_data)
    elif isinstance(offsets_data, dict):
        for key in ("positions", "wells", "offsets", "tiles", "points"):
            if key in offsets_data and isinstance(offsets_data[key], list):
                collect_xy(offsets_data[key])
                break
        # If no list found, try dict values as list
        if not points and offsets_data:
            first_val = next(iter(offsets_data.values()), None)
            if isinstance(first_val, list):
                collect_xy(first_val)

    if not points:
        # Just print the JSON structure and create a simple text plot
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.text(0.5, 0.5, "No x/y positions found in offsets.\nStructure: " + json.dumps(
            {k: (v[:3] if isinstance(v, list) and len(v) > 3 else v) for k, v in (offsets_data.items() if isinstance(offsets_data, dict) else [])},
            indent=2,
        )[:500], transform=ax.transAxes, fontsize=8, verticalalignment="center", family="monospace")
        ax.axis("off")
        plt.tight_layout()
        plt.savefig(out_path, dpi=120, bbox_inches="tight")
        plt.close()
        return

    xs, ys = [p[0] for p in points], [p[1] for p in points]
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.scatter(xs, ys, s=50, alpha=0.7, c="tab:blue", edgecolors="black")
    for i, (x, y) in enumerate(zip(xs, ys)):
        ax.annotate(labels[i] if i < len(labels) else str(i), (x, y), xytext=(5, 5), textcoords="offset points", fontsize=6)
    ax.set_xlabel("X (or column)")
    ax.set_ylabel("Y (or row)")
    ax.set_title("Plate / offset positions")
    ax.grid(True, alpha=0.3)
    ax.set_aspect("equal")
    plt.tight_layout()
    plt.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Preview plate1.ome.tif and plot offsets info")
    parser.add_argument("--dir", type=Path, default=None, help="Directory containing plate1.ome.tif and _offsets.json")
    parser.add_argument("--tif", type=Path, default=None, help="Path to plate1.ome.tif")
    parser.add_argument("--offsets", type=Path, default=None, help="Path to plate1.ome.tif_offsets.json")
    parser.add_argument("--out", type=Path, default=None, help="Output directory for screenshot and plot (default: same as --dir or cwd)")
    parser.add_argument("--max-pixels", type=int, default=4096, help="Max width/height for screenshot (default 4096)")
    args = parser.parse_args()

    if args.tif and args.offsets:
        tif_path = args.tif.resolve()
        offsets_path = args.offsets.resolve()
    else:
        base = (args.dir or Path.cwd()).resolve()
        tif_path, offsets_path = find_plate1_files(base)
        if not tif_path or not offsets_path:
            print("Could not find plate1.ome.tif and plate1.ome.tif_offsets.json.")
            print("Use --dir <path> or --tif and --offsets.")
            return
        tif_path = tif_path.resolve()
        offsets_path = offsets_path.resolve()

    out_dir = (args.out or tif_path.parent).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    screenshot_path = out_dir / "plate1_screenshot.png"
    plot_path = out_dir / "plate1_offsets_plot.png"

    print("OME-TIFF:", tif_path)
    print("Offsets: ", offsets_path)
    print()

    # Zarr info
    zarr_info = get_zarr_info(tif_path)
    print("Zarr/OME-TIFF info:")
    print(f"  Shape: {zarr_info.get('shape')}, dtype: {zarr_info.get('dtype')}, chunks: {zarr_info.get('chunks')}")
    if zarr_info.get("keys"):
        print(f"  Zarr keys (first 20): {zarr_info['keys']}")
    if zarr_info.get("error"):
        print(f"  Error: {zarr_info['error']}")
    print()

    # Load offsets first (needed for stitching)
    offsets_data = load_offsets(offsets_path)
    print("Offsets JSON (structure):")
    if isinstance(offsets_data, list):
        print(f"  List of {len(offsets_data)} items. First: {json.dumps(offsets_data[0] if offsets_data else {}, indent=4)[:400]}")
    else:
        print(f"  Dict with {len(offsets_data)} keys.")
        for k, v in list(offsets_data.items())[:3]:
            print(f"  {k}: {type(v).__name__} = {str(v)[:150]}")
    print()

    # Screenshot (tile montage)
    print("Creating tile montage ...")
    try:
        arr = create_tile_montage(
            tif_path,
            max_tiles=100,  # Show first 100 tiles
            downsample=4,  # Downsample each tile 4x
            grid_cols=10,  # 10 tiles per row
        )
        # Normalize for display (0–255)
        if arr.dtype == np.uint16:
            arr = (arr.astype(np.float32) / max(arr.max(), 1) * 255).astype(np.uint8)
        elif arr.dtype != np.uint8:
            arr = np.clip(arr, 0, 255).astype(np.uint8)
        plt.imsave(screenshot_path, arr, cmap="gray")
        print(f"  Saved: {screenshot_path} (shape: {arr.shape})")
    except Exception as e:
        print(f"  Screenshot failed: {e}")
        import traceback
        traceback.print_exc()

    # Plot
    print("Plotting plate/offsets ...")
    plot_offsets_info(offsets_data, plot_path)
    print(f"  Saved: {plot_path}")
    print("\nDone.")


if __name__ == "__main__":
    main()
