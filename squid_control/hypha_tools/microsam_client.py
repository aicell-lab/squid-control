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
                       max_contrast_percentile: float = 99.0) -> List[Dict[str, Any]]:
    """
    Segment a single image using the microSAM service with contrast adjustment.
    
    Args:
        microsam_service: Connected microSAM service
        image_array: Image array in (H, W) for grayscale or (H, W, 3) for RGB
        min_contrast_percentile: Lower percentile for contrast (default: 1.0)
        max_contrast_percentile: Upper percentile for contrast (default: 99.0)
    
    Returns:
        List of polygon objects with format:
        [{"id": 1, "polygons": [[[x1, y1], [x2, y2], ...]], "bbox": [x_min, y_min, x_max, y_max]}, ...]
        Polygon coordinates are in image pixel coordinates (top-left origin).
    
    Note:
        For multi-channel inputs, use merge_channels_with_colors() first.
        All processing uses scale1 data without resizing.
    """
    logger.info(f"Segmenting image: {image_array.shape}, contrast: {min_contrast_percentile}%-{max_contrast_percentile}%")

    # Apply contrast adjustment
    adjusted_image = apply_contrast_adjustment(
        image_array, min_contrast_percentile, max_contrast_percentile
    )

    # Compress to PNG
    png_bytes = encode_image_to_png(adjusted_image)
    compressed_size_mb = len(png_bytes) / (1024 * 1024)
    logger.info(f"Compressed to {compressed_size_mb:.2f} MB PNG")

    del adjusted_image

    # Encode PNG bytes to base64 for safe JSON transmission
    png_base64 = base64.b64encode(png_bytes).decode('utf-8')

    # Send to microSAM - now returns polygons directly
    segmentation_result = await microsam_service.segment_all(
        image_or_embedding=png_base64,
        embedding=False
    )

    try:
        # Parse JSON response - micro-SAM now returns list of polygon objects
        if isinstance(segmentation_result, str):
            polygon_objects = json.loads(segmentation_result)
        elif isinstance(segmentation_result, list):
            polygon_objects = segmentation_result
        else:
            raise ValueError(f"Unexpected segmentation result type: {type(segmentation_result)}")

        # Calculate statistics
        num_objects = len(polygon_objects)
        total_polygons = sum(len(obj.get('polygons', [])) for obj in polygon_objects)

        logger.info(
            f"🔬 Segmentation complete: {num_objects} object(s) detected, "
            f"{total_polygons} total polygon(s)"
        )
        
        if num_objects > 0:
            avg_polygons_per_object = total_polygons / num_objects
            logger.debug(f"   Average polygons per object: {avg_polygons_per_object:.2f}")

        return polygon_objects

    except json.JSONDecodeError as e:
        logger.error(f"❌ Failed to parse JSON response from micro-SAM: {e}")
        raise Exception(f"microSAM segmentation failed: Invalid JSON response: {e}")
    except Exception as e:
        logger.error(f"❌ Segmentation failed: {e}", exc_info=True)
        raise Exception(f"microSAM segmentation failed: {e}")


def convert_polygons_to_well_relative_mm(
    polygon_objects: List[Dict[str, Any]],
    tile_center_x: float,
    tile_center_y: float,
    image_width: int,
    image_height: int,
    pixel_size_xy_um: float,
    scale_level: int
) -> List[Dict[str, Any]]:
    """
    Convert polygon coordinates from image-relative pixels (top-left origin) to well-relative millimeters.
    
    Args:
        polygon_objects: List of polygon objects from segment_image() with format:
            [{"id": 1, "polygons": [[[x1, y1], [x2, y2], ...]], "bbox": [...]}, ...]
        tile_center_x: Tile center X coordinate in well-relative millimeters (relative to well center at 0,0)
        tile_center_y: Tile center Y coordinate in well-relative millimeters (relative to well center at 0,0)
        image_width: Image width in pixels
        image_height: Image height in pixels
        pixel_size_xy_um: Pixel size in micrometers at scale 0
        scale_level: Scale level (0=full resolution, 1=1/4x, etc.)
    
    Returns:
        List of polygon objects with coordinates converted to well-relative millimeters (relative to well center).
        Format: [{"id": 1, "polygons": [[[x_mm, y_mm], ...]], "bbox": [...]}, ...]
        
    Note:
        The tile_center coordinates must be well-relative (not absolute stage coordinates).
        If you have absolute stage coordinates, convert them first:
        tile_center_x_well_relative = tile_center_x_absolute - well_center_x_absolute
    """
    # Calculate scale factor and pixel size at current scale
    scale_factor = 4 ** scale_level
    scaled_pixel_size_um = pixel_size_xy_um * scale_factor
    pixel_size_mm = scaled_pixel_size_um / 1000.0
    
    # Image center in pixels (for offset calculation)
    image_center_x = image_width / 2.0
    image_center_y = image_height / 2.0
    
    converted_objects = []
    
    for obj in polygon_objects:
        converted_obj = {
            "id": obj.get("id", 0),
            "polygons": [],
            "bbox": obj.get("bbox", [])
        }
        
        # Convert all polygons for this object
        if "polygons" in obj:
            for polygon in obj["polygons"]:
                converted_polygon = []
                for point in polygon:
                    # Point is [x, y] in image pixel coordinates (top-left origin)
                    x_px, y_px = point[0], point[1]
                    
                    # Convert to offset from image center
                    offset_x_px = x_px - image_center_x
                    offset_y_px = y_px - image_center_y
                    
                    # Convert pixel offset to millimeters
                    offset_x_mm = offset_x_px * pixel_size_mm
                    offset_y_mm = offset_y_px * pixel_size_mm
                    
                    # Add tile center to get well-relative coordinates
                    well_relative_x = tile_center_x + offset_x_mm
                    well_relative_y = tile_center_y + offset_y_mm
                    
                    # Round to 8 decimal places for ~0.01 micrometer precision
                    # (0.01 μm = 0.00001 mm, so 8 decimals gives us 0.00000001 mm = 0.00001 μm precision)
                    converted_polygon.append([round(well_relative_x, 8), round(well_relative_y, 8)])
                
                converted_obj["polygons"].append(converted_polygon)
        
        # Convert bounding box if present (bbox format: [x_min, y_min, x_max, y_max])
        if "bbox" in obj and len(obj["bbox"]) == 4:
            bbox = obj["bbox"]
            # Convert bbox corners from image pixels to well-relative mm
            bbox_points = [
                [bbox[0], bbox[1]],  # x_min, y_min
                [bbox[2], bbox[1]],  # x_max, y_min
                [bbox[2], bbox[3]],  # x_max, y_max
                [bbox[0], bbox[3]]   # x_min, y_max
            ]
            
            converted_bbox_points = []
            for point in bbox_points:
                x_px, y_px = point[0], point[1]
                offset_x_px = x_px - image_center_x
                offset_y_px = y_px - image_center_y
                offset_x_mm = offset_x_px * pixel_size_mm
                offset_y_mm = offset_y_px * pixel_size_mm
                well_relative_x = tile_center_x + offset_x_mm
                well_relative_y = tile_center_y + offset_y_mm
                converted_bbox_points.append([round(well_relative_x, 8), round(well_relative_y, 8)])
            
            # Calculate new bbox from converted points with 8 decimal precision (~0.01 μm)
            x_coords = [p[0] for p in converted_bbox_points]
            y_coords = [p[1] for p in converted_bbox_points]
            converted_obj["bbox"] = [
                round(min(x_coords), 8), round(min(y_coords), 8),
                round(max(x_coords), 8), round(max(y_coords), 8)
            ]
        
        converted_objects.append(converted_obj)
    
    return converted_objects


def _bbox_iom(bbox1: List[float], bbox2: List[float]) -> float:
    """
    Calculate Intersection over Minimum (IoM) of two bounding boxes.
    
    IoM = intersection / min(area1, area2)
    
    This is better than IoU for detecting duplicates when objects have very different sizes.
    For example, if a small object is inside a large object, IoM = 1.0 (perfect match),
    whereas IoU would be very low.
    
    Args:
        bbox1: Bounding box [x_min, y_min, x_max, y_max]
        bbox2: Bounding box [x_min, y_min, x_max, y_max]
    
    Returns:
        IoM value between 0.0 and 1.0 (1.0 = smaller object is completely inside larger)
    """
    if len(bbox1) != 4 or len(bbox2) != 4:
        return 0.0
    
    x1_min, y1_min, x1_max, y1_max = bbox1
    x2_min, y2_min, x2_max, y2_max = bbox2
    
    # Calculate intersection
    x_overlap = max(0, min(x1_max, x2_max) - max(x1_min, x2_min))
    y_overlap = max(0, min(y1_max, y2_max) - max(y1_min, y2_min))
    intersection = x_overlap * y_overlap
    
    if intersection == 0:
        return 0.0
    
    # Calculate areas
    area1 = (x1_max - x1_min) * (y1_max - y1_min)
    area2 = (x2_max - x2_min) * (y2_max - y2_min)
    min_area = min(area1, area2)
    
    if min_area <= 0:
        return 0.0
    
    return intersection / min_area


def deduplicate_polygons(polygon_objects: List[Dict[str, Any]], iom_threshold: float = 0.7) -> List[Dict[str, Any]]:
    """
    Remove duplicate polygons from overlapping tiles using bounding box IoM (Intersection over Minimum) detection.
    
    Uses IoM instead of IoU to handle objects with very different sizes. IoM measures how much of the
    smaller object overlaps with the larger one, making it ideal for detecting duplicates when a small
    object is inside a large object (IoM = 1.0) or when objects have similar sizes (IoM ≈ IoU).
    
    Args:
        polygon_objects: List of polygon objects with format:
            [{"id": 1, "polygons": [[[x_mm, y_mm], ...]], "bbox": [x_min, y_min, x_max, y_max]}, ...]
        iom_threshold: Bounding box IoM threshold for considering polygons as duplicates (default: 0.7)
                       IoM = intersection / min(area1, area2)
                       A threshold of 0.7 means 70% of the smaller object must overlap to be considered a duplicate.
    
    Returns:
        Deduplicated list of polygon objects, keeping objects with larger bbox area when duplicates found.
    """
    if not polygon_objects:
        return []
    
    # Prepare objects with bbox area for sorting
    objects_with_area = []
    for obj_idx, obj in enumerate(polygon_objects):
        obj_id = obj.get("id", obj_idx)
        bbox = obj.get("bbox", [])
        
        # Calculate bbox area for sorting
        if len(bbox) == 4:
            bbox_area = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
        else:
            bbox_area = 0.0
        
        objects_with_area.append({
            "object_id": obj_id,
            "original_obj": obj,
            "bbox": bbox,
            "bbox_area": bbox_area
        })
    
    if not objects_with_area:
        return []
    
    # Sort by bbox area (largest first) to prefer keeping larger objects
    objects_with_area.sort(key=lambda x: x["bbox_area"], reverse=True)
    
    # Deduplicate: for each object, check if its bbox overlaps significantly with any kept object
    kept_objects = []
    
    for obj_data in objects_with_area:
        is_duplicate = False
        bbox = obj_data["bbox"]
        
        if len(bbox) != 4:
            # Invalid bbox, skip
            continue
        
        # Check against all kept objects
        for kept_obj_data in kept_objects:
            kept_bbox = kept_obj_data["bbox"]
            
            if len(kept_bbox) != 4:
                continue
            
            # Calculate bbox IoM (Intersection over Minimum)
            iom = _bbox_iom(bbox, kept_bbox)
            
            if iom > iom_threshold:
                # Duplicate found - keep the one with larger area (already sorted)
                is_duplicate = True
                break
        
        if not is_duplicate:
            kept_objects.append(obj_data)
    
    # Reconstruct polygon objects from kept objects
    deduplicated_objects = []
    for obj_data in kept_objects:
        original_obj = obj_data["original_obj"]
        deduplicated_objects.append(original_obj)
    
    total_polygons_before = sum(len(obj.get('polygons', [])) for obj in polygon_objects)
    total_polygons_after = sum(len(obj.get('polygons', [])) for obj in deduplicated_objects)
    
    logger.info(
        f"Deduplication (bbox IoM={iom_threshold}): {len(polygon_objects)} objects ({total_polygons_before} polygons) -> "
        f"{len(deduplicated_objects)} objects ({total_polygons_after} polygons)"
    )
    
    return deduplicated_objects


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
    tile_progress_callback: Optional[Callable[[str, int, int, int, int], None]] = None,
    well_index: Optional[int] = None,
    total_wells: Optional[int] = None
) -> Dict[str, Any]:
    """
    Segment well region by processing it as a 9x9 grid to reduce memory usage.
    Each grid tile is processed separately: load channels, merge, segment, and collect polygons.
    
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
        tile_progress_callback: Optional callback(well_id, tile_idx, total_tiles, completed_wells, total_wells)
    
    Returns:
        Dictionary with processing statistics and collected polygons:
        {
            'well_id': str,
            'tiles_processed': int,
            'tiles_failed': int,
            'tiles_skipped': int,
            'polygons': List[Dict] - deduplicated polygons in well-relative mm coordinates
        }
    """
    well_id = f"{well_row}{well_column}"
    logger.info(f"📍 Processing well {well_id} with {len(channel_configs)} channels using 9x9 grid")

    grid_size = 9
    total_tiles = grid_size * grid_size  # 81 tiles

    stats = {
        'well_id': well_id,
        'tiles_processed': 0,
        'tiles_failed': 0,
        'tiles_skipped': 0,
        'polygons': []  # Accumulate polygons from all tiles
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

                    # Segment this tile - now returns polygons directly
                    tile_polygon_objects = await segment_image(
                        microsam_service, merged_image,
                        min_contrast_percentile=0.0,
                        max_contrast_percentile=100.0  # No additional contrast
                    )

                    # Free gray_image if it was created separately
                    if len(merged_image.shape) == 3:
                        del gray_image

                    # Free merged image after segmentation
                    del merged_image

                    if not tile_polygon_objects:
                        logger.info(f"  Tile {tile_idx}: ⚠️  Segmentation returned no polygons, skipping tile")
                        stats['tiles_skipped'] += 1
                        continue

                    # Count total polygons across all objects
                    total_polygons_before_conversion = sum(len(obj.get('polygons', [])) for obj in tile_polygon_objects)
                    logger.info(f"  Tile {tile_idx}: Polygon objects contain {total_polygons_before_conversion} total polygon(s) across {len(tile_polygon_objects)} object(s)")

                    # Get image dimensions for coordinate conversion
                    image_height = read_bounds['y_end'] - read_bounds['y_start']
                    image_width = read_bounds['x_end'] - read_bounds['x_start']

                    # Convert absolute stage coordinates to well-relative coordinates
                    # (WellZarrCanvas uses absolute coords, but we want to store polygons as well-relative)
                    tile_center_x_well_relative = tile_center_x - center_x
                    tile_center_y_well_relative = tile_center_y - center_y
                    
                    logger.info(f"  Tile {tile_idx}: Converting polygon coordinates from image pixels to well-relative mm")
                    logger.info(f"  Tile {tile_idx}:   Image dimensions: {image_width}x{image_height}px")
                    logger.info(f"  Tile {tile_idx}:   Tile center (absolute): ({tile_center_x:.3f}, {tile_center_y:.3f})mm")
                    logger.info(f"  Tile {tile_idx}:   Tile center (well-relative): ({tile_center_x_well_relative:.3f}, {tile_center_y_well_relative:.3f})mm")
                    logger.info(f"  Tile {tile_idx}:   Well center (absolute): ({center_x:.3f}, {center_y:.3f})mm")
                    logger.info(f"  Tile {tile_idx}:   Pixel size: {canvas.pixel_size_xy_um}µm at scale0, scale_level={scale_level}")

                    # Convert polygon coordinates from image pixels to well-relative mm
                    converted_polygons = convert_polygons_to_well_relative_mm(
                        polygon_objects=tile_polygon_objects,
                        tile_center_x=tile_center_x_well_relative,
                        tile_center_y=tile_center_y_well_relative,
                        image_width=image_width,
                        image_height=image_height,
                        pixel_size_xy_um=canvas.pixel_size_xy_um,
                        scale_level=scale_level
                    )

                    # Accumulate polygons from this tile
                    total_polygons_after_conversion = sum(len(obj.get('polygons', [])) for obj in converted_polygons)
                    stats['polygons'].extend(converted_polygons)

                    logger.info(
                        f"  Tile {tile_idx}: ✅ Coordinate conversion complete - {len(converted_polygons)} polygon object(s) "
                        f"({total_polygons_after_conversion} total polygons) converted to well-relative mm coordinates"
                    )

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

        total_polygons_before_dedup = sum(len(obj.get('polygons', [])) for obj in stats['polygons'])
        logger.info(f"✅ Well {well_id} grid segmentation complete:")
        logger.info(f"   Tiles processed: {stats['tiles_processed']}/{total_tiles}")
        logger.info(f"   Tiles failed: {stats['tiles_failed']}/{total_tiles}")
        logger.info(f"   Tiles skipped: {stats['tiles_skipped']}/{total_tiles}")
        logger.info(f"   Polygon objects collected: {len(stats['polygons'])}")
        logger.info(f"   Total polygons collected: {total_polygons_before_dedup}")

        # Deduplicate polygons from overlapping tiles
        # Yield to event loop before deduplication to prevent blocking
        await asyncio.sleep(0)
        
        if stats['polygons']:
            logger.info(f"🔄 Deduplicating {len(stats['polygons'])} polygon objects from well {well_id} (bbox IoM threshold: 0.7)")
            logger.info(f"   Total polygons before deduplication: {total_polygons_before_dedup}")
            
            # Run deduplication in a thread to avoid blocking the event loop
            # This prevents WebSocket keepalive timeouts during long deduplication
            stats['polygons'] = await asyncio.to_thread(
                deduplicate_polygons, 
                stats['polygons'], 
                0.7  # bbox IoM threshold (Intersection over Minimum) - 70% overlap required
            )
            
            total_polygons_after_dedup = sum(len(obj.get('polygons', [])) for obj in stats['polygons'])
            logger.info(f"✅ Deduplication complete: {len(stats['polygons'])} unique polygon objects remaining")
            logger.info(f"   Total polygons after deduplication: {total_polygons_after_dedup}")
            logger.info(f"   Polygons removed: {total_polygons_before_dedup - total_polygons_after_dedup}")
        else:
            logger.warning(f"⚠️  No polygons collected from well {well_id}")

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




def _convert_polygon_objects_to_wkt(polygon_objects: List[Dict[str, Any]], well_id: str) -> List[Dict[str, str]]:
    """
    Convert polygon objects to WKT format for JSON storage.
    
    Args:
        polygon_objects: List of polygon objects with format:
            [{"id": 1, "polygons": [[[x_mm, y_mm], ...]], "bbox": [...]}, ...]
        well_id: Well identifier (e.g., "A1")
    
    Returns:
        List of polygon dictionaries with WKT format:
        [{"well_id": "A1", "polygon_wkt": "POLYGON((x1 y1, x2 y2, ...))"}, ...]
    """
    wkt_polygons = []
    
    for obj in polygon_objects:
        obj_id = obj.get("id", 0)
        if "polygons" in obj:
            for polygon in obj["polygons"]:
                if len(polygon) < 3:
                    continue  # Skip invalid polygons
                
                # Format as WKT POLYGON string with 5 decimal places
                # WKT format: POLYGON((x1 y1, x2 y2, x3 y3, ..., x1 y1))
                point_strings = [f"{x:.5f} {y:.5f}" for x, y in polygon]
                
                # Ensure polygon is closed (first point == last point)
                if not (len(polygon) > 0 and 
                        abs(polygon[0][0] - polygon[-1][0]) < 1e-6 and 
                        abs(polygon[0][1] - polygon[-1][1]) < 1e-6):
                    point_strings.append(f"{polygon[0][0]:.5f} {polygon[0][1]:.5f}")
                
                wkt = f"POLYGON(({', '.join(point_strings)}))"
                
                wkt_polygons.append({
                    "well_id": well_id,
                    "polygon_wkt": wkt,
                    "object_id": obj_id
                })
    
    return wkt_polygons


def append_polygons_to_json(json_path: Path, new_polygons: List[Dict[str, Any]]):
    """
    Thread-safe append of polygons to JSON file.
    
    Args:
        json_path: Path to polygons.json file
        new_polygons: List of polygon objects or WKT-formatted polygons to append.
                     If polygon objects (with "polygons" key), they will be converted to WKT format.
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

            # Convert polygon objects to WKT format if needed
            # Check if first polygon has "polygons" key (polygon object) or "polygon_wkt" key (WKT format)
            if new_polygons and "polygons" in new_polygons[0]:
                # Extract well_id from first polygon object (assume all have same well_id)
                # If well_id not in object, try to get it from the object structure
                well_id = new_polygons[0].get("well_id")
                if not well_id:
                    # Try to extract from nested structure or use default
                    well_id = "unknown"
                wkt_polygons = _convert_polygon_objects_to_wkt(new_polygons, well_id)
                data["polygons"].extend(wkt_polygons)
            else:
                # Already in WKT format
                data["polygons"].extend(new_polygons)

            # Write back
            with open(json_path, 'w') as f:
                json.dump(data, f, indent=2)

            logger.info(f"Appended {len(new_polygons)} polygons to {json_path} "
                       f"(total now: {len(data['polygons'])})")

        except Exception as e:
            logger.error(f"Failed to append polygons to JSON: {e}", exc_info=True)


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
    progress_callback: Optional[Callable[[str, int, int], None]] = None,
    enable_similarity_search: bool = True,
    similarity_search_callback: Optional[Callable[[str, int, int], None]] = None
) -> Dict[str, Any]:
    """
    Segment multiple wells from an experiment with multi-channel merging.
    
    Automatically processes segmentation results for similarity search if enabled.
    
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
        enable_similarity_search: Automatically process for similarity search (default: True)
        similarity_search_callback: Optional callback for similarity search progress(message, current, total)
    
    Returns:
        results: Dictionary with segmentation results and statistics, including:
            - Standard segmentation results
            - 'similarity_search_results': Dict with extraction/upload results (if enabled)
    """
    logger.info(f"🚀 Starting batch segmentation of {len(wells_to_segment)} wells")
    logger.info(f"Source experiment: '{source_experiment}'")
    logger.info(f"Channels: {len(channel_configs)} channel(s)")
    for config in channel_configs:
        logger.info(f"  {config['channel']}: {config.get('min_percentile', 1.0)}%-{config.get('max_percentile', 99.0)}%")

    # Store polygons.json directly in the source experiment folder (no separate segmentation experiment)
    experiment_path = experiment_manager.base_path / source_experiment
    json_path = experiment_path / "polygons.json"

    # Delete existing polygons.json if present (to start fresh)
    if json_path.exists():
        try:
            json_path.unlink()
            logger.info(f"Deleted existing polygons.json from '{source_experiment}'")
        except Exception as e:
            logger.warning(f"Failed to delete existing polygons.json: {e}")

    # Create empty polygons.json structure
    try:
        with open(json_path, 'w') as f:
            json.dump({"polygons": []}, f, indent=2)
        logger.info(f"Created empty polygons.json in '{source_experiment}'")
    except Exception as create_err:
        logger.warning(f"Failed to create polygons.json: {create_err}")

    results = {
        'source_experiment': source_experiment,
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

            # Create tile progress callback if main progress callback exists
            # Note: This callback is only called at intervals (every 10 tiles or completion)
            # to avoid spamming the progress callback. We don't call the main progress callback
            # from tile callback to avoid duplicate calls - only call it once at well completion.
            def make_tile_callback(well_idx, total_wells):
                def tile_callback(well_id_inner, tile_idx, total_tiles, completed_wells, total_wells_inner):
                    # Don't call main progress callback from tile callback
                    # Progress will be reported once at well completion instead
                    # This prevents duplicate/repeated progress callbacks
                    pass
                return tile_callback

            tile_callback = None  # Disable tile-level progress callbacks to avoid spam

            # Segment the well region using grid-based approach (now returns polygons directly)
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

            # Store polygons directly to JSON file
            if tile_stats.get('polygons'):
                total_polygons = sum(len(obj.get('polygons', [])) for obj in tile_stats['polygons'])
                logger.info(f"💾 Storing polygons for well {well_id}: {len(tile_stats['polygons'])} object(s), {total_polygons} total polygon(s)")
                
                # Add well_id to each polygon object for WKT conversion
                for polygon_obj in tile_stats['polygons']:
                    polygon_obj['well_id'] = well_id
                
                # Append polygons to JSON file (run in thread to avoid blocking)
                await asyncio.to_thread(append_polygons_to_json, json_path, tile_stats['polygons'])
                logger.info(f"✅ Saved {len(tile_stats['polygons'])} polygon object(s) ({total_polygons} total polygons) for well {well_id} to '{source_experiment}/polygons.json'")
            else:
                logger.warning(f"⚠️  No polygons collected for well {well_id} - nothing to store")

            logger.info(f"✅ Completed segmentation for well {well_id} "
                       f"({tile_stats['tiles_processed']} tiles processed, {tile_stats['tiles_failed']} failed, {tile_stats['tiles_skipped']} skipped)")

            results['successful_wells'] += 1
            results['wells_processed'].append(well_id)

            # Call progress callback once at well completion (not during tile processing)
            # Yield to event loop first to ensure any pending operations complete
            await asyncio.sleep(0)
            if progress_callback:
                try:
                    progress_callback(well_id, idx + 1, len(wells_to_segment))
                except Exception as callback_error:
                    logger.warning(f"Progress callback error after well {well_id} completion: {callback_error}")

        except Exception as e:
            logger.error(f"Failed to process well {well_id}: {e}", exc_info=True)
            results['failed_wells'] += 1
            results['wells_failed'].append(well_id)

    logger.info("🎉 Batch segmentation complete!")
    logger.info(f"  Successful: {results['successful_wells']}/{results['total_wells']}")
    logger.info(f"  Failed: {results['failed_wells']}/{results['total_wells']}")
    logger.info(f"  Channels used: {len(channel_configs)}")

    # Automatically process for similarity search if enabled
    if enable_similarity_search and results['successful_wells'] > 0:
        logger.info("🔄 Starting automatic similarity search processing...")
        try:
            similarity_results = await process_segmentation_for_similarity_search(
                experiment_manager=experiment_manager,
                source_experiment=source_experiment,
                channel_configs=channel_configs,
                progress_callback=similarity_search_callback,
                batch_size=64
            )
            results['similarity_search_results'] = similarity_results
            
            if similarity_results['success']:
                logger.info(f"✅ Similarity search processing complete: "
                          f"{similarity_results['uploaded_count']} cells uploaded to Weaviate")
            else:
                logger.warning(f"⚠️  Similarity search processing had issues: "
                             f"{similarity_results.get('failed_count', 0)} failed, "
                             f"errors: {similarity_results.get('errors', [])}")
        except Exception as e:
            logger.error(f"❌ Similarity search processing failed: {e}", exc_info=True)
            results['similarity_search_results'] = {
                'success': False,
                'error': str(e)
            }
    elif enable_similarity_search:
        logger.info("⚠️  Skipping similarity search processing: no successful wells")
        results['similarity_search_results'] = {
            'success': False,
            'error': 'No successful wells to process'
        }

    return results


async def process_segmentation_for_similarity_search(
    experiment_manager,
    source_experiment: str,
    channel_configs: List[Dict],
    progress_callback: Optional[Callable[[str, int, int], None]] = None,
    batch_size: int = 32,
    agent_lens_base_url: str = "https://hypha.aicell.io/agent-lens/apps/agent-lens"
) -> Dict[str, Any]:
    """
    Post-process segmentation results for similarity search.
    
    Orchestrates the complete pipeline:
    1. Load polygons from JSON file
    2. Extract cell images from zarr using polygons
    3. Generate embeddings using agent-lens
    4. Setup Weaviate application (delete old, create new)
    5. Batch upload to Weaviate
    
    Args:
        experiment_manager: ExperimentManager instance
        source_experiment: Source experiment name
        channel_configs: List of channel configurations (same as used for segmentation)
        progress_callback: Optional callback(status_message, current, total)
        batch_size: Batch size for embedding generation and upload (default: 32)
        agent_lens_base_url: Base URL for agent-lens service
    
    Returns:
        Dict with results:
            {
                'success': bool,
                'total_polygons': int,
                'extracted_count': int,
                'embedding_success_count': int,
                'uploaded_count': int,
                'failed_count': int,
                'errors': List[str]
            }
    """
    from datetime import datetime
    
    from squid_control.hypha_tools.cell_extractor import (
        extract_cell_image_from_experiment,
        generate_cell_preview
    )
    from squid_control.hypha_tools.embedding_generator import generate_embeddings_batch
    from squid_control.hypha_tools.weaviate_client import (
        batch_upload_to_weaviate,
        setup_weaviate_application
    )
    
    logger.info(f"🚀 Starting similarity search processing for experiment: {source_experiment}")
    
    results = {
        'success': False,
        'total_polygons': 0,
        'extracted_count': 0,
        'embedding_success_count': 0,
        'uploaded_count': 0,
        'failed_count': 0,
        'errors': []
    }
    
    try:
        # Step 1: Load polygons from JSON
        json_path = experiment_manager.base_path / source_experiment / "polygons.json"
        if not json_path.exists():
            error_msg = f"Polygons file not found: {json_path}"
            logger.error(error_msg)
            results['errors'].append(error_msg)
            return results
        
        logger.info(f"📂 Loading polygons from: {json_path}")
        with open(json_path, 'r') as f:
            data = json.load(f)
        
        # Extract polygons array from JSON structure
        if isinstance(data, dict) and 'polygons' in data:
            all_polygons = data['polygons']
        elif isinstance(data, list):
            # Legacy format: direct list
            all_polygons = data
        else:
            raise ValueError(f"Invalid polygons.json format: expected dict with 'polygons' key or list")
        
        results['total_polygons'] = len(all_polygons)
        logger.info(f"Loaded {results['total_polygons']} polygon objects")
        
        if results['total_polygons'] == 0:
            logger.warning("No polygons to process")
            results['success'] = True
            return results
        
        if progress_callback:
            progress_callback(f"Loaded {results['total_polygons']} polygons", 0, results['total_polygons'])
        
        # Step 2: Extract cell images and prepare data
        logger.info("🖼️  Extracting cell images from zarr...")
        extracted_images = []
        extracted_metadata = []
        
        for idx, polygon_obj in enumerate(all_polygons):
            try:
                polygon_wkt = polygon_obj.get('polygon_wkt')
                well_id = polygon_obj.get('well_id')
                
                if not polygon_wkt or not well_id:
                    logger.warning(f"Polygon {idx}: Missing polygon_wkt or well_id, skipping")
                    results['failed_count'] += 1
                    continue
                
                # Extract cell image (scale 0 = full resolution)
                cell_image, metadata = extract_cell_image_from_experiment(
                    experiment_manager=experiment_manager,
                    experiment_name=source_experiment,
                    well_id=well_id,
                    polygon_wkt=polygon_wkt,
                    channel_configs=channel_configs,
                    well_plate_type="96",  # Default, could be made configurable
                    timepoint=0,
                    scale=0
                )
                
                extracted_images.append(cell_image)
                extracted_metadata.append({
                    'polygon_obj': polygon_obj,
                    'well_id': well_id,
                    'extraction_metadata': metadata
                })
                results['extracted_count'] += 1
                
                if (idx + 1) % 100 == 0:
                    logger.info(f"  Extracted {idx + 1}/{results['total_polygons']} cell images")
                    if progress_callback:
                        progress_callback(f"Extracting images", idx + 1, results['total_polygons'])
                
            except Exception as e:
                logger.warning(f"Polygon {idx}: Failed to extract image: {e}")
                results['failed_count'] += 1
                results['errors'].append(f"Extract polygon {idx}: {str(e)}")
        
        logger.info(f"✅ Extracted {results['extracted_count']}/{results['total_polygons']} cell images")
        
        if results['extracted_count'] == 0:
            error_msg = "No cell images extracted successfully"
            logger.error(error_msg)
            results['errors'].append(error_msg)
            return results
        
        # Step 3: Generate embeddings
        logger.info(f"🧠 Generating embeddings for {results['extracted_count']} images...")
        if progress_callback:
            progress_callback(f"Generating embeddings", 0, results['extracted_count'])
        
        # Create progress callback wrapper for embedding generation
        def embedding_progress_callback(current, total):
            if progress_callback:
                progress_callback(f"Generating embeddings", current, total)
        
        embeddings = await generate_embeddings_batch(
            images=extracted_images,
            batch_size=batch_size,
            retry_attempts=2,
            base_url=agent_lens_base_url,
            progress_callback=embedding_progress_callback
        )
        
        results['embedding_success_count'] = sum(1 for e in embeddings if e is not None)
        logger.info(f"✅ Generated {results['embedding_success_count']}/{results['extracted_count']} embeddings")
        
        if results['embedding_success_count'] == 0:
            error_msg = "No embeddings generated successfully"
            logger.error(error_msg)
            results['errors'].append(error_msg)
            return results
        
        # Step 4: Setup Weaviate application
        logger.info("🔧 Setting up Weaviate application...")
        if progress_callback:
            progress_callback("Setting up Weaviate", 0, 1)
        
        setup_result = await setup_weaviate_application(
            application_id=source_experiment,
            description=f"Segmentation results from {source_experiment}",
            base_url=agent_lens_base_url
        )
        
        if not setup_result['success']:
            error_msg = f"Failed to setup Weaviate application: {setup_result.get('error')}"
            logger.error(error_msg)
            results['errors'].append(error_msg)
            return results
        
        logger.info("✅ Weaviate application ready")
        
        # Step 5: Prepare objects and upload to Weaviate
        logger.info("📤 Preparing objects for upload...")
        upload_objects = []
        timestamp = datetime.utcnow().isoformat() + "Z"
        
        for idx, (metadata, embedding, image) in enumerate(zip(extracted_metadata, embeddings, extracted_images)):
            if embedding is None:
                logger.warning(f"Cell {idx}: No embedding, skipping upload")
                continue
            
            polygon_obj = metadata['polygon_obj']
            well_id = metadata['well_id']
            
            # Generate annotation ID
            annotation_id = f"{source_experiment}_cell_{idx}"
            
            # Generate preview image
            try:
                preview_base64 = generate_cell_preview(image, size=50)
            except Exception as e:
                logger.warning(f"Cell {idx}: Failed to generate preview: {e}")
                preview_base64 = None
            
            # Prepare metadata
            upload_metadata = {
                'annotation_id': annotation_id,
                'well_id': well_id,
                'annotation_type': 'polygon',
                'timestamp': timestamp,
                'polygon_wkt': polygon_obj.get('polygon_wkt'),
                'source': 'segmentation',
                'bbox': polygon_obj.get('bbox'),
                'channel_info': [config['channel'] for config in channel_configs]
            }
            
            # Prepare upload object
            upload_obj = {
                'image_id': annotation_id,
                'description': f"Segmented cell from well {well_id}",
                'metadata': upload_metadata,
                'dataset_id': source_experiment,
                'vector': embedding
            }
            
            if preview_base64:
                upload_obj['preview_image'] = preview_base64
            
            upload_objects.append(upload_obj)
        
        logger.info(f"Prepared {len(upload_objects)} objects for upload")
        
        # Upload in batches
        logger.info(f"📤 Uploading {len(upload_objects)} objects to Weaviate...")
        uploaded_total = 0
        
        for batch_start in range(0, len(upload_objects), batch_size):
            batch_end = min(batch_start + batch_size, len(upload_objects))
            batch_objects = upload_objects[batch_start:batch_end]
            
            if progress_callback:
                progress_callback(f"Uploading batch", batch_start, len(upload_objects))
            
            upload_result = await batch_upload_to_weaviate(
                objects=batch_objects,
                application_id=source_experiment,
                base_url=agent_lens_base_url,
                retry_attempts=2
            )
            
            if upload_result['success']:
                uploaded_total += upload_result['uploaded_count']
                logger.info(f"  Uploaded batch {batch_start//batch_size + 1}: "
                          f"{upload_result['uploaded_count']} objects")
            else:
                error_msg = f"Batch {batch_start}-{batch_end-1} upload failed: {upload_result.get('error')}"
                logger.warning(error_msg)
                results['errors'].append(error_msg)
        
        results['uploaded_count'] = uploaded_total
        logger.info(f"✅ Uploaded {results['uploaded_count']}/{len(upload_objects)} objects to Weaviate")
        
        # Final status
        if results['uploaded_count'] > 0:
            results['success'] = True
            logger.info(f"🎉 Similarity search processing complete!")
            logger.info(f"  Total polygons: {results['total_polygons']}")
            logger.info(f"  Extracted: {results['extracted_count']}")
            logger.info(f"  Embeddings: {results['embedding_success_count']}")
            logger.info(f"  Uploaded: {results['uploaded_count']}")
            logger.info(f"  Failed: {results['failed_count']}")
        else:
            logger.error("No objects uploaded to Weaviate")
        
        if progress_callback:
            progress_callback("Complete", results['uploaded_count'], results['total_polygons'])
        
    except Exception as e:
        error_msg = f"Fatal error in similarity search processing: {e}"
        logger.error(error_msg, exc_info=True)
        results['errors'].append(error_msg)
    
    return results

