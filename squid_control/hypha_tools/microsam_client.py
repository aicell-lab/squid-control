"""
microSAM segmentation client for automated cell segmentation.

This module provides helper functions to connect to the microSAM BioEngine service
and perform automated instance segmentation on microscopy images.
"""

import asyncio
import base64
import json
import logging
import threading
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import cv2
import numpy as np

logger = logging.getLogger(__name__)

# Global lock for thread-safe JSON file writes
_polygon_file_lock = threading.Lock()


def get_channel_color_hex(channel_name: str) -> str:
    """
    Get OME-Zarr color hex code for channel.
    Maps channel names to their standard visualization colors.
    
    Args:
        channel_name: Human-readable channel name
    
    Returns:
        color_hex: 6-digit hex color (e.g., "00FF00" for green)
    """
    from squid_control.control.config import ChannelMapper

    try:
        # Get color from centralized channel mapping
        return ChannelMapper.get_channel_color_by_name(channel_name)
    except ValueError:
        logger.warning(f"Unknown channel '{channel_name}', using default white color")
        return "FFFFFF"  # Default white


def hex_to_rgb_float(hex_color: str) -> tuple:
    """
    Convert hex color to RGB float tuple (0-1 range).
    
    Args:
        hex_color: 6-digit hex string (e.g., "00FF00")
    
    Returns:
        (r, g, b): RGB values as floats (0.0-1.0)
    """
    r = int(hex_color[0:2], 16) / 255.0
    g = int(hex_color[2:4], 16) / 255.0
    b = int(hex_color[4:6], 16) / 255.0
    return (r, g, b)


def merge_channels_with_colors(channel_images: List[np.ndarray],
                               channel_names: List[str]) -> np.ndarray:
    """
    Merge multiple channels to RGB using OME-Zarr color mapping.
    
    Args:
        channel_images: List of contrast-adjusted channel images (uint8)
        channel_names: List of channel names for color mapping
    
    Returns:
        rgb_image: Merged RGB image (H, W, 3) as uint8
    """
    if not channel_images:
        raise ValueError("No channel images provided")

    # Initialize RGB canvas
    h, w = channel_images[0].shape
    rgb_image = np.zeros((h, w, 3), dtype=np.float32)

    logger.info(f"Merging {len(channel_images)} channels with OME-Zarr color mapping")

    for image, channel_name in zip(channel_images, channel_names):
        # Get OME-Zarr color for this channel
        color_hex = get_channel_color_hex(channel_name)
        r, g, b = hex_to_rgb_float(color_hex)

        logger.debug(f"  Channel '{channel_name}': color={color_hex}, RGB=({r:.2f}, {g:.2f}, {b:.2f})")

        # Convert image to float (0-1)
        img_float = image.astype(np.float32) / 255.0

        # Add weighted channel to RGB components
        rgb_image[:, :, 0] += img_float * r
        rgb_image[:, :, 1] += img_float * g
        rgb_image[:, :, 2] += img_float * b

    # Clip to valid range and convert to uint8
    rgb_image = np.clip(rgb_image * 255, 0, 255).astype(np.uint8)

    logger.info(f"✅ Merged to RGB image: shape={rgb_image.shape}, dtype={rgb_image.dtype}")
    return rgb_image

def encode_image_to_png(image: np.ndarray) -> bytes:
    """
    Encode numpy array to PNG bytes for lossless transmission.
    
    Args:
        image: Image array in (H, W) for grayscale or (H, W, 3) for RGB, dtype uint8
    
    Returns:
        png_bytes: Compressed PNG image as bytes
    
    Raises:
        ValueError: If encoding fails
    """
    try:
        if len(image.shape) == 2:
            # Grayscale image
            success, encoded = cv2.imencode('.png', image)
        elif len(image.shape) == 3 and image.shape[2] == 3:
            # RGB image - convert RGB to BGR for OpenCV
            bgr_image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            success, encoded = cv2.imencode('.png', bgr_image)
        else:
            raise ValueError(f"Unsupported image shape for PNG encoding: {image.shape}")

        if not success:
            raise ValueError("Failed to encode image to PNG")

        png_bytes = encoded.tobytes()
        original_size = image.nbytes
        compressed_size = len(png_bytes)
        compression_ratio = original_size / compressed_size if compressed_size > 0 else 1.0

        logger.debug(f"PNG encoding: {original_size:,} bytes -> {compressed_size:,} bytes "
                    f"(ratio: {compression_ratio:.2f}x)")

        return png_bytes

    except Exception as e:
        logger.error(f"Error encoding image to PNG: {e}")
        raise ValueError(f"PNG encoding failed: {e}")


def apply_contrast_adjustment(image: np.ndarray, min_percentile: float, max_percentile: float) -> np.ndarray:
    """
    Apply simple percentile-based contrast adjustment to image.
    
    Threshold calculation:
    - min_threshold = min_percentile * 255 / 100 (e.g., 1.0% -> 2.55)
    - max_threshold = max_percentile * 255 / 100 (e.g., 99.0% -> 252.45)
    
    Then clip values to [min_threshold, max_threshold] and normalize to 0-255.
    
    Args:
        image: Input image array
        min_percentile: Lower percentile percentage (e.g., 1.0 for 1%)
        max_percentile: Upper percentile percentage (e.g., 99.0 for 99%)
    
    Returns:
        Contrast-adjusted image normalized to 0-255 range
    """
    logger.debug(f"Applying contrast adjustment: {min_percentile}%-{max_percentile}%")

    # Calculate thresholds as percentage of 255
    min_threshold = min_percentile * 255.0 / 100.0
    max_threshold = max_percentile * 255.0 / 100.0

    logger.debug(f"Thresholds: min={min_threshold:.2f}, max={max_threshold:.2f} (out of 255)")

    # Clip image values to threshold range
    clipped = np.clip(image, min_threshold, max_threshold)

    # Normalize clipped values to 0-255 range
    if max_threshold > min_threshold:
        normalized = ((clipped - min_threshold) / (max_threshold - min_threshold) * 255).astype(np.uint8)
        logger.debug(f"Contrast adjustment applied: {image.min()}-{image.max()} -> clipped to [{min_threshold:.2f}, {max_threshold:.2f}] -> normalized to 0-255")
    else:
        normalized = np.zeros_like(image, dtype=np.uint8)
        logger.warning("Image has uniform intensity - contrast adjustment resulted in zero range")

    return normalized


async def connect_to_microsam(server):
    """
    Connect to the microSAM BioEngine service via HTTP.
    
    Args:
        server: Connected Hypha server instance
    
    Returns:
        microsam_service: Connected microSAM service object
    
    Raises:
        Exception: If connection fails
    """
    logger.info("Connecting to microSAM service via HTTP...")

    try:
        microsam_service = await server.get_service("agent-lens/micro-sam")
        logger.info("✅ Connected to microSAM service via HTTP")
        return microsam_service

    except Exception as e:
        logger.error(f"❌ Failed to connect to microSAM service: {e}")
        raise Exception(f"Could not connect to microSAM service: {e}")


async def segment_image(microsam_service, image_array: np.ndarray,
                       min_contrast_percentile: float = 1.0,
                       max_contrast_percentile: float = 99.0,
                       resize: float = 0.5) -> np.ndarray:
    """
    Segment a single image using the microSAM service with contrast adjustment.
    
    Args:
        microsam_service: Connected microSAM service
        image_array: Image array in (H, W) for grayscale or (H, W, 3) for RGB
        min_contrast_percentile: Lower percentile for contrast (default: 1.0)
        max_contrast_percentile: Upper percentile for contrast (default: 99.0)
        resize: Resize factor before segmentation (default: 0.5 = 1/2 resolution).
                Use 1.0 for no resize, 0.25 for 1/4 resolution, etc.
    
    Returns:
        Binary segmentation mask as uint8 array (H, W): 0 = background, 255 = segmented pixels
    
    Note:
        For multi-channel inputs, use merge_channels_with_colors() first.
        Images are downsampled by resize factor, then upsampled back to original dimensions.
    """
    logger.info(f"Segmenting image: {image_array.shape}, contrast: {min_contrast_percentile}%-{max_contrast_percentile}%, resize: {resize}")

    # Apply contrast adjustment
    adjusted_image = apply_contrast_adjustment(
        image_array, min_contrast_percentile, max_contrast_percentile
    )

    # Store original dimensions and conditionally resize
    original_height, original_width = adjusted_image.shape[:2]
    if resize != 1.0:
        new_width = int(original_width * resize)
        new_height = int(original_height * resize)
        image_to_encode = cv2.resize(
            adjusted_image,
            (new_width, new_height),
            interpolation=cv2.INTER_AREA
        )
        logger.info(f"Resized to {new_width}x{new_height} (factor: {resize}) for segmentation")
    else:
        image_to_encode = adjusted_image

    # Compress to PNG
    png_bytes = encode_image_to_png(image_to_encode)
    compressed_size_mb = len(png_bytes) / (1024 * 1024)
    logger.info(f"Compressed to {compressed_size_mb:.2f} MB PNG")

    del image_to_encode

    # Encode PNG bytes to base64 for safe JSON transmission
    png_base64 = base64.b64encode(png_bytes).decode('utf-8')

    # Send to microSAM
    segmentation_result_b64 = await microsam_service.segment_all(
        image_or_embedding=png_base64,
        embedding=False
    )

    try:
        # Decode result
        if isinstance(segmentation_result_b64, str):
            segmentation_png_bytes = base64.b64decode(segmentation_result_b64)
            nparr = np.frombuffer(segmentation_png_bytes, np.uint8)
            segmentation_result = cv2.imdecode(nparr, cv2.IMREAD_GRAYSCALE)
            if segmentation_result is None:
                raise ValueError("Failed to decode segmentation result from PNG")
        else:
            segmentation_result = segmentation_result_b64

        # Normalize to binary mask
        if segmentation_result.dtype != np.uint8:
            segmentation_result = segmentation_result.astype(np.uint8)
        segmentation_result = np.where(segmentation_result > 0, 255, 0).astype(np.uint8)

        # Resize back to original dimensions if needed
        if resize != 1.0 and (segmentation_result.shape[:2] != (original_height, original_width)):
            segmentation_result = cv2.resize(
                segmentation_result,
                (original_width, original_height),
                interpolation=cv2.INTER_NEAREST
            )
            logger.info(f"Resized mask back to {original_width}x{original_height}")

        # Calculate statistics about the binary mask
        segmented_pixels = np.sum(segmentation_result == 255)
        total_pixels = segmentation_result.size
        segmented_percentage = (segmented_pixels / total_pixels) * 100.0

        # Count connected components as approximate object count
        # Use cv2.findContours to count distinct objects
        contours, _ = cv2.findContours(
            segmentation_result, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        num_objects = len(contours)

        logger.info(
            f"🔬 Binary mask stats: {segmented_pixels}/{total_pixels} pixels segmented "
            f"({segmented_percentage:.2f}%), ~{num_objects} objects detected"
        )

        return segmentation_result

    except Exception as e:
        logger.error(f"❌ Segmentation failed: {e}", exc_info=True)
        raise Exception(f"microSAM segmentation failed: {e}")


async def segment_well_region_grid_based(
    microsam_service,
    experiment_manager,
    source_experiment: str,
    well_row: str,
    well_column: int,
    channel_configs: List[Dict[str, Any]],
    scale_level: int,
    timepoint: int,
    well_plate_type: str,
    well_padding_mm: float,
    seg_canvas,
    segmentation_channel: str,
    channel_idx: int,
    tile_progress_callback: Optional[Callable[[str, int, int, int, int], None]] = None,
    well_index: Optional[int] = None,
    total_wells: Optional[int] = None
) -> Dict[str, Any]:
    """
    Segment well region by processing it as a 9x9 grid to reduce memory usage.
    Each grid tile is processed separately: load channels, merge, segment, and write to zarr.
    
    Args:
        microsam_service: Connected microSAM service
        experiment_manager: ExperimentManager instance
        source_experiment: Name of the source experiment
        well_row: Well row letter (e.g., 'A', 'B')
        well_column: Well column number (e.g., 1, 2)
        channel_configs: List of dicts with channel configurations
        scale_level: Pyramid scale level (0=full resolution)
        timepoint: Timepoint index
        well_plate_type: Well plate format ('96', '384', etc.)
        well_padding_mm: Well padding in millimeters
        seg_canvas: Segmentation canvas for writing results
        segmentation_channel: Channel name for segmentation masks
        channel_idx: Channel index in segmentation canvas
        tile_progress_callback: Optional callback(well_id, tile_idx, total_tiles, completed_wells, total_wells)
    
    Returns:
        Dictionary with processing statistics (tiles_processed, tiles_failed, etc.)
    """
    well_id = f"{well_row}{well_column}"
    logger.info(f"📍 Processing well {well_id} with {len(channel_configs)} channels using 9x9 grid")

    grid_size = 9
    total_tiles = grid_size * grid_size  # 81 tiles

    stats = {
        'well_id': well_id,
        'tiles_processed': 0,
        'tiles_failed': 0,
        'tiles_skipped': 0
    }

    try:
        # Get well canvas from source experiment
        canvas = experiment_manager.get_well_canvas(
            well_row, well_column, well_plate_type, well_padding_mm, source_experiment
        )

        # Get well center coordinates and canvas bounds
        well_info = canvas.get_well_info()
        center_x = well_info['well_info']['well_center_x_mm']
        center_y = well_info['well_info']['well_center_y_mm']
        well_diameter = well_info['well_info']['well_diameter_mm']
        canvas_width_mm = well_info['canvas_info']['canvas_width_mm']
        canvas_height_mm = well_info['canvas_info']['canvas_height_mm']

        # Calculate tile size (use minimum to ensure tiles fit)
        base_tile_size_mm = min(canvas_width_mm, canvas_height_mm, well_diameter) / grid_size
        # Add 5% overlap to tile size to ensure proper segmentation at boundaries
        tile_size_mm = base_tile_size_mm * 1.05

        logger.info(f"Well {well_id} - center: ({center_x:.2f}, {center_y:.2f})mm")
        logger.info(f"Canvas size: {canvas_width_mm:.2f}x{canvas_height_mm:.2f}mm")
        logger.info(f"Grid: {grid_size}x{grid_size} tiles, base tile size: {base_tile_size_mm:.2f}mm, with 5% overlap: {tile_size_mm:.2f}mm")

        # Process each grid tile
        tile_idx = 0
        for grid_i in range(grid_size):
            for grid_j in range(grid_size):
                tile_idx += 1

                # Yield to event loop to prevent WebSocket keepalive timeout
                await asyncio.sleep(0)

                # Calculate tile center coordinates in millimeters
                # Grid indices: 0-8, tiles should cover full canvas from -canvas_half to +canvas_half
                # Tile centers are spaced by base_tile_size (without overlap)
                # Each tile region extends 5% beyond its boundary to create overlap
                tile_center_x = center_x + (grid_i - (grid_size - 1) / 2.0) * base_tile_size_mm
                tile_center_y = center_y + (grid_j - (grid_size - 1) / 2.0) * base_tile_size_mm

                logger.debug(f"  Tile {tile_idx}/{total_tiles} ({grid_i},{grid_j}): center=({tile_center_x:.2f}, {tile_center_y:.2f})mm")

                # CRITICAL: Convert tile center to pixel coordinates ONCE at the beginning
                # This ensures exact alignment between read and write operations
                tile_center_x_px, tile_center_y_px = canvas.stage_to_pixel_coords(tile_center_x, tile_center_y, scale_level)

                # Calculate pixel bounds for the tile region
                # Use overlapped tile size (5% larger than base) to ensure proper segmentation at boundaries
                request_size = min(tile_size_mm, well_diameter)
                scale_factor = 4 ** scale_level
                request_size_px = int(request_size * 1000 / (canvas.pixel_size_xy_um * scale_factor))

                logger.debug(f"  Tile {tile_idx}: Pixel center=({tile_center_x_px}, {tile_center_y_px}), size={request_size_px}px")

                try:
                    # Load and process each channel for this tile using pixel-based read
                    processed_channels = []
                    channel_names = []
                    read_bounds = None  # Store exact pixel bounds from first channel read

                    for config in channel_configs:
                        channel_name = config['channel']
                        min_percentile = config.get('min_percentile', 1.0)
                        max_percentile = config.get('max_percentile', 99.0)
                        weight = config.get('weight', 1.0)

                        # Get channel index for pixel-based read
                        channel_idx_read = canvas.get_zarr_channel_index(channel_name)

                        # Get tile region using pixel-based read method for exact alignment
                        # Wrap in asyncio.to_thread to prevent blocking the event loop during Zarr I/O
                        channel_data, bounds = await asyncio.to_thread(
                            canvas.get_canvas_region_pixels,
                            tile_center_x_px, tile_center_y_px, request_size_px, request_size_px,
                            scale_level, channel_idx_read, timepoint
                        )

                        if channel_data is None:
                            logger.debug(f"    No data for channel {channel_name} in tile {tile_idx}, skipping")
                            continue

                        # Store exact bounds from first successful read (all channels should have same bounds)
                        if read_bounds is None:
                            read_bounds = bounds
                            logger.info(f"  Tile {tile_idx}: ✅ Read bounds captured - x=[{bounds['x_start']}:{bounds['x_end']}], "
                                       f"y=[{bounds['y_start']}:{bounds['y_end']}], shape={channel_data.shape}")

                        # Apply contrast adjustment
                        adjusted = apply_contrast_adjustment(channel_data, min_percentile, max_percentile)
                        del channel_data  # Free memory immediately

                        # Apply weight
                        if weight != 1.0:
                            adjusted = np.clip(adjusted * weight, 0, 255).astype(np.uint8)

                        processed_channels.append(adjusted)
                        channel_names.append(channel_name)

                    if not processed_channels:
                        logger.debug(f"  Tile {tile_idx} has no valid channel data, skipping")
                        stats['tiles_skipped'] += 1
                        continue

                    if read_bounds is None:
                        logger.warning(f"  Tile {tile_idx}: No valid read bounds obtained, skipping")
                        stats['tiles_skipped'] += 1
                        continue

                    # Merge channels using OME-Zarr colors
                    if len(processed_channels) == 1:
                        merged_image = processed_channels[0]
                    else:
                        merged_image = merge_channels_with_colors(processed_channels, channel_names)

                    # Free processed channels after merging
                    del processed_channels
                    del channel_names

                    # Check if image has enough content (>5% non-black pixels)
                    # Convert to grayscale if RGB for threshold check
                    if len(merged_image.shape) == 3:
                        gray_image = cv2.cvtColor(merged_image, cv2.COLOR_RGB2GRAY)
                    else:
                        gray_image = merged_image

                    # Count non-black pixels (threshold > 10 to account for noise)
                    non_black_threshold = 10
                    non_black_pixels = np.sum(gray_image > non_black_threshold)
                    total_pixels = gray_image.size
                    non_black_percentage = (non_black_pixels / total_pixels) * 100.0

                    if non_black_percentage < 5.0:
                        logger.debug(
                            f"  Tile {tile_idx}: Skipping segmentation - "
                            f"only {non_black_percentage:.2f}% non-black pixels (threshold: 5%)"
                        )
                        stats['tiles_skipped'] += 1
                        del merged_image
                        del gray_image
                        continue

                    # Segment this tile - segmentation should preserve exact dimensions
                    tile_segmentation = await segment_image(
                        microsam_service, merged_image,
                        min_contrast_percentile=0.0,
                        max_contrast_percentile=100.0  # No additional contrast
                    )

                    # Free gray_image if it was created separately
                    if len(merged_image.shape) == 3:
                        del gray_image

                    # Free merged image after segmentation
                    del merged_image

                    if tile_segmentation is None:
                        logger.warning(f"  Tile {tile_idx} segmentation returned None, skipping")
                        stats['tiles_failed'] += 1
                        continue

                    # CRITICAL: Ensure segmentation mask dimensions match read region exactly
                    # If segmentation changed dimensions, resize to match read bounds
                    seg_height, seg_width = tile_segmentation.shape[:2]
                    expected_height = read_bounds['y_end'] - read_bounds['y_start']
                    expected_width = read_bounds['x_end'] - read_bounds['x_start']

                    if seg_height != expected_height or seg_width != expected_width:
                        logger.debug(
                            f"  Tile {tile_idx}: Resizing segmentation from {seg_height}x{seg_width} "
                            f"to match read bounds {expected_height}x{expected_width}"
                        )
                        tile_segmentation = cv2.resize(
                            tile_segmentation,
                            (expected_width, expected_height),  # (width, height)
                            interpolation=cv2.INTER_NEAREST  # Preserve binary mask values
                        )

                    # CRITICAL: Merge overlapping masks instead of overwriting
                    # Read existing mask region using pixel-based read for exact alignment
                    # Wrap in asyncio.to_thread to prevent blocking the event loop during Zarr I/O
                    existing_mask, existing_bounds = await asyncio.to_thread(
                        seg_canvas.get_canvas_region_pixels,
                        tile_center_x_px, tile_center_y_px, request_size_px, request_size_px,
                        scale_level, channel_idx, timepoint
                    )

                    if existing_mask is not None and existing_mask.size > 0:
                        # Ensure existing mask matches the new mask dimensions
                        if existing_mask.shape != tile_segmentation.shape:
                            # Resize existing mask to match if dimensions differ
                            logger.debug(
                                f"  Tile {tile_idx}: Resizing existing mask from {existing_mask.shape} "
                                f"to match new mask {tile_segmentation.shape} for merging"
                            )
                            existing_mask = cv2.resize(
                                existing_mask,
                                (tile_segmentation.shape[1], tile_segmentation.shape[0]),
                                interpolation=cv2.INTER_NEAREST
                            )

                        # Merge masks using maximum (union operation for binary masks)
                        # max(0, 0) = 0 (background), max(0, 255) = 255 (segmented), max(255, 255) = 255
                        tile_segmentation = np.maximum(tile_segmentation, existing_mask).astype(np.uint8)

                        logger.debug(
                            f"  Tile {tile_idx}: Merged with existing mask "
                            f"({np.sum(existing_mask > 0)} existing segmented pixels)"
                        )

                    # Yield to event loop before zarr write (potentially slow operation)
                    await asyncio.sleep(0)

                    # Write merged tile segmentation mask using pixel-based write with EXACT same bounds
                    # This ensures perfect alignment - no preprocessing, no coordinate conversion errors
                    logger.info(f"  Tile {tile_idx}: Attempting to write segmentation mask (shape={tile_segmentation.shape}, "
                               f"dtype={tile_segmentation.dtype}) to pixel bounds x=[{read_bounds['x_start']}:{read_bounds['x_end']}], "
                               f"y=[{read_bounds['y_start']}:{read_bounds['y_end']}]")

                    seg_canvas.add_image_sync_pixels(
                        image=tile_segmentation,
                        x_start_px=read_bounds['x_start'],  # Exact pixel bounds from read operation
                        y_start_px=read_bounds['y_start'],  # Exact pixel bounds from read operation
                        channel_idx=channel_idx,
                        z_idx=0,
                        timepoint=timepoint,
                        scale=scale_level
                    )

                    logger.info(
                        f"  Tile {tile_idx}: ✅ Completed write of segmentation mask (shape={tile_segmentation.shape}) "
                        f"to pixel bounds x=[{read_bounds['x_start']}:{read_bounds['x_end']}], "
                        f"y=[{read_bounds['y_start']}:{read_bounds['y_end']}] "
                        f"- exact same bounds as source image read"
                    )

                    # Free segmentation result after writing
                    del tile_segmentation

                    stats['tiles_processed'] += 1

                    # Call tile progress callback
                    if tile_progress_callback and well_index is not None and total_wells is not None:
                        try:
                            tile_progress_callback(well_id, tile_idx, total_tiles, well_index + 1, total_wells)
                        except Exception as callback_error:
                            logger.debug(f"Tile progress callback error: {callback_error}")

                    # Log progress every 10 tiles
                    if tile_idx % 10 == 0:
                        logger.info(f"  Progress: {tile_idx}/{total_tiles} tiles processed ({stats['tiles_processed']} successful, {stats['tiles_failed']} failed)")

                except Exception as e:
                    logger.error(f"  Failed to process tile {tile_idx} ({grid_i},{grid_j}): {e}")
                    stats['tiles_failed'] += 1
                    continue

        logger.info(f"✅ Well {well_id} grid segmentation complete: {stats['tiles_processed']} tiles processed, {stats['tiles_failed']} failed, {stats['tiles_skipped']} skipped")

        # Launch background polygon extraction if well segmentation was successful
        if stats['tiles_processed'] > 0:
            try:
                # Get segmentation experiment name from seg_canvas fileset_name
                # fileset_name format: "well_A1_96"
                segmentation_experiment = str(seg_canvas.base_path.name)

                logger.info(f"🚀 Launching background polygon extraction for well {well_id}")

                # Launch in background thread to avoid blocking segmentation
                thread = threading.Thread(
                    target=process_well_polygons_background,
                    args=(experiment_manager, segmentation_experiment, well_row, well_column,
                          well_plate_type, well_padding_mm, 1, timepoint),
                    daemon=True  # Daemon thread won't block program exit
                )
                thread.start()

                logger.info(f"Background polygon extraction thread started for well {well_id}")

            except Exception as e:
                logger.error(f"Failed to launch polygon extraction thread for well {well_id}: {e}")

        return stats

    except Exception as e:
        logger.error(f"Failed to segment well {well_id}: {e}", exc_info=True)
        stats['tiles_failed'] = total_tiles - stats['tiles_processed']
        return stats


def pixel_to_well_relative_mm(pixel_coords: np.ndarray, canvas_info: Dict,
                               pixel_size_xy_um: float, scale: int = 1) -> np.ndarray:
    """
    Convert pixel coordinates to well-relative millimeter coordinates.
    
    Args:
        pixel_coords: Array of pixel coordinates (N, 2) in format [[x1, y1], [x2, y2], ...]
        canvas_info: Dictionary from WellZarrCanvas.get_well_info()
        pixel_size_xy_um: Pixel size in micrometers at scale 0
        scale: Scale level (1 = 1/4 resolution)
    
    Returns:
        Array of well-relative coordinates in mm (N, 2)
    """
    # Get canvas dimensions and padding info from canvas_info
    canvas_width_px = canvas_info['canvas_info']['canvas_width_px']
    canvas_height_px = canvas_info['canvas_info']['canvas_height_px']

    # Apply scale factor
    scale_factor = 4 ** scale
    scaled_pixel_size_um = pixel_size_xy_um * scale_factor

    # Calculate canvas center in pixels (at given scale)
    canvas_center_x_px = canvas_width_px / (2 * scale_factor)
    canvas_center_y_px = canvas_height_px / (2 * scale_factor)

    # Convert to well-relative coordinates (canvas center is well center = 0, 0)
    # Subtract canvas center to get relative pixel coords, then convert to mm
    relative_x_mm = (pixel_coords[:, 0] - canvas_center_x_px) * scaled_pixel_size_um / 1000.0
    relative_y_mm = (pixel_coords[:, 1] - canvas_center_y_px) * scaled_pixel_size_um / 1000.0

    return np.column_stack([relative_x_mm, relative_y_mm])


def extract_polygons_from_segmentation_mask(mask: np.ndarray, well_id: str,
                                            canvas_info: Dict, pixel_size_xy_um: float,
                                            scale: int = 1, min_area_px: int = 100) -> List[Dict[str, str]]:
    """
    Extract polygon contours from a binary segmentation mask.
    
    Args:
        mask: Binary segmentation mask (uint8, 0 or 255)
        well_id: Well identifier (e.g., "A1")
        canvas_info: Dictionary from WellZarrCanvas.get_well_info()
        pixel_size_xy_um: Pixel size in micrometers at scale 0
        scale: Scale level used (1 = 1/4 resolution)
        min_area_px: Minimum contour area in pixels to keep (filters noise)
    
    Returns:
        List of polygon dictionaries with format:
        [{"well_id": "A1", "polygon_wkt": "POLYGON((x1 y1, x2 y2, ...))"}]
    """
    if mask is None or mask.size == 0:
        logger.warning(f"Empty mask for well {well_id}")
        return []

    # Find contours using RETR_EXTERNAL to get only outer boundaries (no holes)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        logger.info(f"No contours found in well {well_id}")
        return []

    polygons = []

    for contour in contours:
        # Filter small contours (likely noise)
        area = cv2.contourArea(contour)
        if area < min_area_px:
            continue

        # Simplify polygon to reduce point count (memory optimization)
        # Epsilon = 0.001mm in pixels at current scale
        epsilon_mm = 0.001
        scale_factor = 4 ** scale
        epsilon_px = (epsilon_mm * 1000.0) / (pixel_size_xy_um * scale_factor)
        approx = cv2.approxPolyDP(contour, epsilon_px, closed=True)

        # Need at least 3 points for a polygon
        if len(approx) < 3:
            continue

        # Reshape from (N, 1, 2) to (N, 2)
        points = approx.reshape(-1, 2)

        # Convert pixel coordinates to well-relative millimeters
        points_mm = pixel_to_well_relative_mm(points, canvas_info, pixel_size_xy_um, scale)

        # Format as WKT POLYGON string with 5 decimal places
        # WKT format: POLYGON((x1 y1, x2 y2, x3 y3, ..., x1 y1))
        # Note: First and last points must be the same (closed polygon)
        point_strings = [f"{x:.5f} {y:.5f}" for x, y in points_mm]

        # Ensure polygon is closed (first point == last point)
        if not np.allclose(points_mm[0], points_mm[-1], atol=1e-6):
            point_strings.append(f"{points_mm[0][0]:.5f} {points_mm[0][1]:.5f}")

        wkt = f"POLYGON(({', '.join(point_strings)}))"

        polygons.append({
            "well_id": well_id,
            "polygon_wkt": wkt
        })

    logger.info(f"Extracted {len(polygons)} polygons from well {well_id} "
               f"(from {len(contours)} total contours, min_area={min_area_px}px)")

    return polygons


def append_polygons_to_json(json_path: Path, new_polygons: List[Dict[str, str]]):
    """
    Thread-safe append of polygons to JSON file.
    
    Args:
        json_path: Path to polygons.json file
        new_polygons: List of polygon dictionaries to append
    """
    if not new_polygons:
        return

    with _polygon_file_lock:
        try:
            # Read existing data
            if json_path.exists():
                try:
                    with open(json_path) as f:
                        data = json.load(f)
                except json.JSONDecodeError:
                    logger.warning(f"Corrupt JSON file {json_path}, recreating")
                    data = {"polygons": []}
            else:
                data = {"polygons": []}

            # Append new polygons
            data["polygons"].extend(new_polygons)

            # Write back
            with open(json_path, 'w') as f:
                json.dump(data, f, indent=2)

            logger.info(f"Appended {len(new_polygons)} polygons to {json_path} "
                       f"(total now: {len(data['polygons'])})")

        except Exception as e:
            logger.error(f"Failed to append polygons to JSON: {e}", exc_info=True)


def process_well_polygons_background(experiment_manager, segmentation_experiment: str,
                                     well_row: str, well_column: int, well_plate_type: str,
                                     well_padding_mm: float, scale: int = 1, timepoint: int = 0):
    """
    Background thread function to extract polygons from a completed well's segmentation mask.
    
    Args:
        experiment_manager: ExperimentManager instance
        segmentation_experiment: Name of segmentation experiment
        well_row: Well row (e.g., 'A')
        well_column: Well column (e.g., 1)
        well_plate_type: Well plate format
        well_padding_mm: Well padding in mm
        scale: Scale level to read from (default 1)
        timepoint: Timepoint index (default 0)
    """
    well_id = f"{well_row}{well_column}"

    try:
        logger.info(f"🔄 Starting background polygon extraction for well {well_id}")

        # Get segmentation canvas
        seg_canvas = experiment_manager.get_well_canvas(
            well_row, well_column, well_plate_type, well_padding_mm, segmentation_experiment
        )

        # Get well info for coordinate conversion
        well_info = seg_canvas.get_well_info()

        # Get segmentation channel (BF LED matrix full = index 0)
        segmentation_channel = "BF LED matrix full"
        channel_idx = seg_canvas.get_zarr_channel_index(segmentation_channel)

        # Read the full segmentation mask at scale 1
        with seg_canvas.zarr_lock:
            if scale not in seg_canvas.zarr_arrays:
                logger.error(f"Scale {scale} not available in segmentation canvas")
                return

            zarr_array = seg_canvas.zarr_arrays[scale]

            # Read full mask for this well: [timepoint, channel, z=0, :, :]
            mask = zarr_array[timepoint, channel_idx, 0, :, :]

        logger.info(f"Read segmentation mask for well {well_id}: shape={mask.shape}, "
                   f"dtype={mask.dtype}, non-zero pixels={np.sum(mask > 0)}")

        # Extract polygons from mask
        polygons = extract_polygons_from_segmentation_mask(
            mask=mask,
            well_id=well_id,
            canvas_info=well_info,
            pixel_size_xy_um=seg_canvas.pixel_size_xy_um,
            scale=scale,
            min_area_px=100  # Filter small noise
        )

        # Free memory
        del mask

        if not polygons:
            logger.info(f"No polygons extracted for well {well_id}")
            return

        # Append to JSON file
        experiment_path = experiment_manager.base_path / segmentation_experiment
        json_path = experiment_path / "polygons.json"

        append_polygons_to_json(json_path, polygons)

        logger.info(f"✅ Completed polygon extraction for well {well_id}: {len(polygons)} polygons")

    except Exception as e:
        logger.error(f"Failed to process polygons for well {well_id}: {e}", exc_info=True)


async def segment_experiment_wells(
    microsam_service,
    experiment_manager,
    source_experiment: str,
    wells_to_segment: List[str],
    channel_configs: List[Dict[str, Any]],
    scale_level: int,
    timepoint: int,
    well_plate_type: str,
    well_padding_mm: float,
    progress_callback: Optional[Callable[[str, int, int], None]] = None
) -> Dict[str, Any]:
    """
    Segment multiple wells from an experiment with multi-channel merging.
    
    Args:
        microsam_service: Connected microSAM service
        experiment_manager: ExperimentManager instance
        source_experiment: Name of the source experiment
        wells_to_segment: List of well identifiers (e.g., ['A1', 'B2'])
        channel_configs: List of channel configurations for merging
        scale_level: Pyramid scale level
        timepoint: Timepoint index
        well_plate_type: Well plate format
        well_padding_mm: Well padding in mm
        progress_callback: Optional callback function(well_id, completed, total)
    
    Returns:
        results: Dictionary with segmentation results and statistics
    """
    logger.info(f"🚀 Starting batch segmentation of {len(wells_to_segment)} wells")
    logger.info(f"Source experiment: '{source_experiment}'")
    logger.info(f"Channels: {len(channel_configs)} channel(s)")
    for config in channel_configs:
        logger.info(f"  {config['channel']}: {config.get('min_percentile', 1.0)}%-{config.get('max_percentile', 99.0)}%")

    # Create segmentation experiment name
    segmentation_experiment = f"{source_experiment}-segmentation"
    logger.info(f"Target segmentation experiment: '{segmentation_experiment}'")

    # Create segmentation experiment if it doesn't exist
    try:
        experiment_manager.create_experiment(segmentation_experiment)
        logger.info(f"✅ Created segmentation experiment: '{segmentation_experiment}'")
    except ValueError:
        # Experiment already exists
        logger.info(f"Segmentation experiment '{segmentation_experiment}' already exists, using existing")

    # Set segmentation experiment as active for writing
    experiment_manager.set_active_experiment(segmentation_experiment)

    # Initialize/clean up polygons.json file
    experiment_path = experiment_manager.base_path / segmentation_experiment
    json_path = experiment_path / "polygons.json"

    # Delete existing polygons.json if present
    if json_path.exists():
        try:
            json_path.unlink()
            logger.info(f"Deleted existing polygons.json from '{segmentation_experiment}'")
        except Exception as e:
            logger.warning(f"Failed to delete existing polygons.json: {e}")

    # Create empty polygons.json structure
    try:
        with open(json_path, 'w') as f:
            json.dump({"polygons": []}, f, indent=2)
        logger.info(f"Created empty polygons.json in '{segmentation_experiment}'")
    except Exception as create_err:
        logger.warning(f"Failed to create polygons.json: {create_err}")

    results = {
        'source_experiment': source_experiment,
        'segmentation_experiment': segmentation_experiment,
        'total_wells': len(wells_to_segment),
        'successful_wells': 0,
        'failed_wells': 0,
        'wells_processed': [],
        'wells_failed': [],
        'channel_configs': channel_configs  # Store channel configurations used
    }

    for idx, well_str in enumerate(wells_to_segment):
        # Parse well identifier (e.g., "A1" -> row='A', column=1)
        import re
        match = re.match(r'^([A-Z]+)(\d+)$', well_str.upper())
        if not match:
            logger.error(f"Invalid well identifier: {well_str}")
            results['failed_wells'] += 1
            results['wells_failed'].append(well_str)
            continue

        well_row = match.group(1)
        well_column = int(match.group(2))
        well_id = f"{well_row}{well_column}"

        logger.info(f"📊 Processing well {idx+1}/{len(wells_to_segment)}: {well_id}")

        try:
            # CRITICAL: Validate that source well exists and has data before creating segmentation canvas
            source_canvas_path = experiment_manager.base_path / source_experiment / f"well_{well_id}_{well_plate_type}.zarr"
            if not source_canvas_path.exists():
                logger.warning(f"⚠️  Source well canvas does not exist for well {well_id}: {source_canvas_path}")
                logger.warning(f"   Skipping well {well_id} - no source data available")
                results['failed_wells'] += 1
                results['wells_failed'].append(well_id)
                continue

            # Get source canvas to check if it has data
            try:
                source_canvas = experiment_manager.get_well_canvas(
                    well_row, well_column, well_plate_type, well_padding_mm, source_experiment
                )
            except Exception as e:
                logger.warning(f"⚠️  Failed to open source well canvas for well {well_id}: {e}")
                logger.warning(f"   Skipping well {well_id}")
                results['failed_wells'] += 1
                results['wells_failed'].append(well_id)
                continue

            # Check if source canvas has data by reading a sample region at scale 4 (high scale = fast check)
            # Scale 4 is very downsampled, so it's quick to read and gives a good indication if data exists
            try:
                # Check if scale 4 exists (some canvases may have fewer scales)
                max_scale = min(4, len(source_canvas.zarr_arrays) - 1) if hasattr(source_canvas, 'zarr_arrays') and source_canvas.zarr_arrays else 0
                
                # Get canvas dimensions at the selected scale
                if hasattr(source_canvas, 'zarr_arrays') and max_scale in source_canvas.zarr_arrays:
                    zarr_array = source_canvas.zarr_arrays[max_scale]
                    canvas_height, canvas_width = zarr_array.shape[3], zarr_array.shape[4]
                    
                    # Read a small region from the center at scale 4 (or highest available scale)
                    # Use a small region (e.g., 100x100 pixels) for fast checking
                    sample_size = max(50, min(100, canvas_width // 4, canvas_height // 4))  # At least 50px, max 100px
                    center_x_px = canvas_width // 2
                    center_y_px = canvas_height // 2
                    
                    # Read sample region asynchronously to avoid blocking
                    sample_region, _ = await asyncio.to_thread(
                        source_canvas.get_canvas_region_pixels,
                        center_x_px, center_y_px, sample_size, sample_size,
                        max_scale, 0, timepoint  # Use channel 0 (BF) and requested timepoint
                    )
                    
                    if sample_region is None:
                        logger.warning(f"⚠️  Could not read sample region from source well {well_id} - skipping")
                        results['failed_wells'] += 1
                        results['wells_failed'].append(well_id)
                        continue
                    
                    # Check if the sample region is empty (all zeros or mostly zeros)
                    # Consider empty if >95% of pixels are zero (or very close to zero, accounting for noise)
                    non_zero_pixels = np.sum(sample_region > 10)  # Threshold of 10 to account for noise
                    total_pixels = sample_region.size
                    non_zero_percentage = (non_zero_pixels / total_pixels) * 100.0
                    
                    if non_zero_percentage < 5.0:
                        logger.warning(f"⚠️  Source well {well_id} appears to be empty (only {non_zero_percentage:.2f}% non-zero pixels in sample region)")
                        logger.warning(f"   Skipping well {well_id} - no imaging data available")
                        results['failed_wells'] += 1
                        results['wells_failed'].append(well_id)
                        continue
                    
                    logger.info(f"✅ Source well {well_id} validated: {non_zero_percentage:.2f}% non-zero pixels in sample region (has data)")
                else:
                    # If zarr arrays not initialized or scale not available, check timepoints
                    if not hasattr(source_canvas, 'available_timepoints') or not source_canvas.available_timepoints:
                        logger.warning(f"⚠️  Source well {well_id} has no timepoints - skipping")
                        results['failed_wells'] += 1
                        results['wells_failed'].append(well_id)
                        continue
                    
                    if timepoint not in source_canvas.available_timepoints:
                        logger.warning(f"⚠️  Source well {well_id} does not have timepoint {timepoint} - available: {source_canvas.available_timepoints}")
                        results['failed_wells'] += 1
                        results['wells_failed'].append(well_id)
                        continue
                    
                    logger.info(f"✅ Source well {well_id} validated: {len(source_canvas.available_timepoints)} timepoint(s) available")
                    
            except Exception as e:
                logger.warning(f"⚠️  Failed to validate source well {well_id} data: {e}")
                logger.warning(f"   Skipping well {well_id}")
                results['failed_wells'] += 1
                results['wells_failed'].append(well_id)
                continue

            # Get or create well canvas in segmentation experiment (only if source is valid)
            seg_canvas = experiment_manager.get_well_canvas(
                well_row, well_column, well_plate_type, well_padding_mm, segmentation_experiment
            )

            # Use "BF LED matrix full" channel for segmentation masks to maintain OME-Zarr format consistency
            segmentation_channel = "BF LED matrix full"

            # Get channel index for brightfield channel (should be 0 in standard OME-Zarr format)
            if segmentation_channel not in seg_canvas.channel_to_zarr_index:
                logger.error(f"Channel '{segmentation_channel}' not found in segmentation canvas!")
                logger.error(f"Available channels: {list(seg_canvas.channel_to_zarr_index.keys())}")
                raise ValueError(f"Channel '{segmentation_channel}' not available in segmentation canvas")

            channel_idx = seg_canvas.channel_to_zarr_index[segmentation_channel]
            logger.info(f"Using channel '{segmentation_channel}' at index {channel_idx} for segmentation masks")

            # Create tile progress callback if main progress callback exists
            def make_tile_callback(well_idx, total_wells):
                def tile_callback(well_id_inner, tile_idx, total_tiles, completed_wells, total_wells_inner):
                    # Report progress every 10 tiles or at completion
                    if tile_idx % 10 == 0 or tile_idx == total_tiles:
                        if progress_callback:
                            progress_callback(well_id_inner, completed_wells, total_wells_inner)
                return tile_callback

            tile_callback = make_tile_callback(idx, len(wells_to_segment)) if progress_callback else None

            # Segment the well region using grid-based approach
            tile_stats = await segment_well_region_grid_based(
                microsam_service=microsam_service,
                experiment_manager=experiment_manager,
                source_experiment=source_experiment,
                well_row=well_row,
                well_column=well_column,
                channel_configs=channel_configs,
                scale_level=scale_level,
                timepoint=timepoint,
                well_plate_type=well_plate_type,
                well_padding_mm=well_padding_mm,
                seg_canvas=seg_canvas,
                segmentation_channel=segmentation_channel,
                channel_idx=channel_idx,
                tile_progress_callback=tile_callback,
                well_index=idx,
                total_wells=len(wells_to_segment)
            )

            # Check if segmentation was successful (at least some tiles processed)
            if tile_stats['tiles_processed'] == 0:
                logger.warning(f"No tiles processed for well {well_id} (failed: {tile_stats['tiles_failed']}, skipped: {tile_stats['tiles_skipped']})")
                results['failed_wells'] += 1
                results['wells_failed'].append(well_id)
                continue

            logger.info(f"✅ Saved segmentation for well {well_id} to '{segmentation_experiment}' "
                       f"({tile_stats['tiles_processed']} tiles processed, {tile_stats['tiles_failed']} failed, {tile_stats['tiles_skipped']} skipped)")

            results['successful_wells'] += 1
            results['wells_processed'].append(well_id)

            # Call progress callback
            if progress_callback:
                progress_callback(well_id, idx + 1, len(wells_to_segment))

        except Exception as e:
            logger.error(f"Failed to process well {well_id}: {e}", exc_info=True)
            results['failed_wells'] += 1
            results['wells_failed'].append(well_id)

    logger.info("🎉 Batch segmentation complete!")
    logger.info(f"  Successful: {results['successful_wells']}/{results['total_wells']}")
    logger.info(f"  Failed: {results['failed_wells']}/{results['total_wells']}")
    logger.info(f"  Channels used: {len(channel_configs)}")

    return results

