"""
Cell image extraction module for segmentation post-processing.

This module provides functions to extract individual cell images from OME-Zarr data
using polygon annotations, including WKT parsing, bounding box calculation, and preview generation.
"""

import base64
import io
import logging
import re
from typing import Dict, List, Tuple

import cv2
import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)


def parse_wkt_polygon(wkt_string: str) -> List[List[float]]:
    """
    Parse WKT polygon string to list of coordinates.
    
    Args:
        wkt_string: WKT format polygon string like "POLYGON((x1 y1, x2 y2, ...))"
    
    Returns:
        List of [x, y] coordinates in millimeters
    
    Raises:
        ValueError: If WKT string is invalid or cannot be parsed
    """
    if not wkt_string or not isinstance(wkt_string, str):
        raise ValueError("Invalid WKT string: must be non-empty string")
    
    # Remove "POLYGON((" prefix and "))" suffix
    match = re.match(r'POLYGON\s*\(\s*\((.*)\)\s*\)', wkt_string.strip(), re.IGNORECASE)
    if not match:
        raise ValueError(f"Invalid WKT POLYGON format: {wkt_string}")
    
    coords_string = match.group(1)
    
    # Parse coordinate pairs
    coordinates = []
    try:
        # Split by comma to get coordinate pairs
        pairs = coords_string.split(',')
        for pair in pairs:
            # Split by space to get x and y
            parts = pair.strip().split()
            if len(parts) >= 2:
                x, y = float(parts[0]), float(parts[1])
                coordinates.append([x, y])
    except (ValueError, IndexError) as e:
        raise ValueError(f"Failed to parse WKT coordinates: {e}")
    
    if len(coordinates) < 3:
        raise ValueError(f"Polygon must have at least 3 points, got {len(coordinates)}")
    
    return coordinates


def calculate_bbox_from_polygon(polygon_coords: List[List[float]]) -> Tuple[float, float, float, float]:
    """
    Calculate bounding box from polygon coordinates.
    
    Args:
        polygon_coords: List of [x, y] coordinates
    
    Returns:
        Tuple of (x_min, y_min, x_max, y_max) in millimeters
    """
    if not polygon_coords or len(polygon_coords) < 3:
        raise ValueError("Polygon must have at least 3 coordinates")
    
    x_coords = [p[0] for p in polygon_coords]
    y_coords = [p[1] for p in polygon_coords]
    
    return (min(x_coords), min(y_coords), max(x_coords), max(y_coords))


def mm_to_pixels(mm_coord: float, pixel_size_um: float, scale_factor: int = 1) -> int:
    """
    Convert millimeter coordinates to pixel coordinates.
    
    Args:
        mm_coord: Coordinate in millimeters
        pixel_size_um: Pixel size in micrometers at scale 0
        scale_factor: Scale factor (4^scale_level)
    
    Returns:
        Pixel coordinate
    """
    scaled_pixel_size_um = pixel_size_um * scale_factor
    return int((mm_coord * 1000.0) / scaled_pixel_size_um)


def extract_cell_image_from_experiment(
    experiment_manager,
    experiment_name: str,
    well_id: str,
    polygon_wkt: str,
    channel_configs: List[Dict],
    well_plate_type: str = "96",
    timepoint: int = 0,
    scale: int = 0
) -> Tuple[np.ndarray, Dict]:
    """
    Extract cell image from experiment using polygon bounding box.
    
    Uses the stitching infrastructure (similar to get_stitched_region) to properly
    handle multi-channel extraction, contrast adjustment, and merging.
    
    Args:
        experiment_manager: ExperimentManager instance
        experiment_name: Name of the experiment
        well_id: Well identifier (e.g., "A1", "B2")
        polygon_wkt: WKT format polygon string (well-relative mm coordinates)
        channel_configs: List of channel configurations with 'channel', 'min_percentile', 'max_percentile'
        well_plate_type: Well plate format (default: "96")
        timepoint: Timepoint index (default 0)
        scale: Scale level to extract from (default 0 for full resolution)
    
    Returns:
        Tuple of (masked_rgb_image, metadata_dict)
        - masked_rgb_image: RGB image with polygon mask applied (pixels outside polygon are black)
        - metadata_dict: Dictionary with extraction metadata (bbox, dimensions, etc.)
    
    Raises:
        ValueError: If polygon parsing or extraction fails
    """
    try:
        # Parse WKT polygon
        polygon_coords_mm = parse_wkt_polygon(polygon_wkt)
        
        # Calculate bounding box in mm (well-relative coordinates)
        bbox_mm = calculate_bbox_from_polygon(polygon_coords_mm)
        x_min_mm, y_min_mm, x_max_mm, y_max_mm = bbox_mm
        
        # Calculate center and size
        center_x_well_relative = (x_min_mm + x_max_mm) / 2.0
        center_y_well_relative = (y_min_mm + y_max_mm) / 2.0
        width_mm = x_max_mm - x_min_mm
        height_mm = y_max_mm - y_min_mm
        
        # Add 5% padding to ensure we capture cell edges
        padding_percent = 0.05
        width_mm_padded = width_mm * (1 + padding_percent)
        height_mm_padded = height_mm * (1 + padding_percent)
        
        # Get well canvas to access well center coordinates
        well_row, well_col = well_id[0], int(well_id[1:])
        canvas = experiment_manager.get_well_canvas(well_row, well_col, well_plate_type, experiment_name=experiment_name)
        
        if canvas is None:
            raise ValueError(f"Cannot get canvas for well {well_id}")
        
        # Convert well-relative coordinates to stage coordinates
        # Well center is at (0, 0) in well-relative coordinates
        center_x_stage = canvas.well_center_x + center_x_well_relative
        center_y_stage = canvas.well_center_y + center_y_well_relative
        
        logger.debug(f"Extracting cell: bbox_mm={bbox_mm}, center_stage=({center_x_stage:.3f}, {center_y_stage:.3f}), size=({width_mm_padded:.3f}, {height_mm_padded:.3f})")
        
        # Extract region for each channel
        processed_channels = []
        channel_names = []
        
        for config in channel_configs:
            channel_name = config['channel']
            min_percentile = config.get('min_percentile', 1.0)
            max_percentile = config.get('max_percentile', 99.0)
            
            # Get channel index
            try:
                channel_idx = canvas.get_zarr_channel_index(channel_name)
            except ValueError:
                logger.warning(f"Channel '{channel_name}' not found in canvas, skipping")
                continue
            
            # Convert stage coordinates to pixel coordinates for extraction
            center_x_px, center_y_px = canvas.stage_to_pixel_coords(center_x_stage, center_y_stage, scale)
            
            # Calculate pixel dimensions
            scale_factor = 4 ** scale
            width_px = int((width_mm_padded * 1000.0) / (canvas.pixel_size_xy_um * scale_factor))
            height_px = int((height_mm_padded * 1000.0) / (canvas.pixel_size_xy_um * scale_factor))
            
            # Extract region
            channel_data, bounds = canvas.get_canvas_region_pixels(
                center_x_px, center_y_px, width_px, height_px,
                scale, channel_idx, timepoint
            )
            
            if channel_data is None:
                logger.warning(f"No data for channel {channel_name}, skipping")
                continue
            
            # Apply contrast adjustment using percentiles
            from squid_control.hypha_tools.microsam_client import apply_contrast_adjustment
            adjusted = apply_contrast_adjustment(channel_data, min_percentile, max_percentile)
            
            processed_channels.append(adjusted)
            channel_names.append(channel_name)
        
        if not processed_channels:
            raise ValueError("No valid channel data extracted")
        
        # Merge channels using color mapping
        if len(processed_channels) == 1:
            merged_image = processed_channels[0]
            # Convert grayscale to RGB for consistency
            if len(merged_image.shape) == 2:
                merged_image = cv2.cvtColor(merged_image, cv2.COLOR_GRAY2RGB)
        else:
            from squid_control.hypha_tools.microsam_client import merge_channels_with_colors
            merged_image = merge_channels_with_colors(processed_channels, channel_names)
        
        # Create polygon mask
        # Convert polygon coordinates from well-relative mm to pixels relative to extracted image center
        polygon_coords_px = []
        # Get image dimensions
        image_height, image_width = merged_image.shape[:2]
        image_center_x = image_width / 2.0
        image_center_y = image_height / 2.0
        
        for x_mm, y_mm in polygon_coords_mm:
            # Convert well-relative mm to stage coordinates
            px_stage = canvas.well_center_x + x_mm
            py_stage = canvas.well_center_y + y_mm
            
            # Calculate offset from extraction center in mm
            offset_x_mm = px_stage - center_x_stage
            offset_y_mm = py_stage - center_y_stage
            
            # Convert mm offset to pixels
            scale_factor = 4 ** scale
            offset_x_px = (offset_x_mm * 1000.0) / (canvas.pixel_size_xy_um * scale_factor)
            offset_y_px = (offset_y_mm * 1000.0) / (canvas.pixel_size_xy_um * scale_factor)
            
            # Convert to image-relative coordinates (centered at image center)
            rel_x = image_center_x + offset_x_px
            rel_y = image_center_y + offset_y_px
            
            polygon_coords_px.append([rel_x, rel_y])
        
        # Create binary mask
        mask = np.zeros(merged_image.shape[:2], dtype=np.uint8)
        polygon_array = np.array(polygon_coords_px, dtype=np.int32)
        cv2.fillPoly(mask, [polygon_array], 255)
        
        # Apply mask to image
        if len(merged_image.shape) == 3:
            # RGB image
            masked_image = merged_image.copy()
            masked_image[mask == 0] = 0
        else:
            # Grayscale image
            masked_image = cv2.bitwise_and(merged_image, merged_image, mask=mask)
        
        # Ensure image is uint8 format (0-255 range) for proper preview generation
        if masked_image.dtype != np.uint8:
            # If float, assume 0-1 range and scale to 0-255
            if masked_image.dtype in [np.float32, np.float64]:
                masked_image = (masked_image * 255).astype(np.uint8)
            else:
                # For other types, clip to 0-255 range
                masked_image = np.clip(masked_image, 0, 255).astype(np.uint8)
        
        # Prepare metadata
        metadata = {
            'bbox_mm': bbox_mm,
            'bbox_mm_stage': (
                center_x_stage - width_mm_padded/2,
                center_y_stage - height_mm_padded/2,
                center_x_stage + width_mm_padded/2,
                center_y_stage + height_mm_padded/2
            ),
            'dimensions_px': (image_height, image_width),
            'scale': scale,
            'timepoint': timepoint,
            'channels': channel_names,
            'well_id': well_id
        }
        
        return masked_image, metadata
        
    except Exception as e:
        logger.error(f"Failed to extract cell image: {e}", exc_info=True)
        raise ValueError(f"Cell extraction failed: {e}")


def generate_cell_preview(image: np.ndarray, size: int = 50) -> str:
    """
    Create preview image and encode as base64 string.
    
    Args:
        image: Input image (RGB or grayscale numpy array)
        size: Target size for preview (default 50x50 pixels)
    
    Returns:
        Base64 encoded PNG string (without data URL prefix)
    """
    try:
        # Convert to PIL Image
        if len(image.shape) == 2:
            # Grayscale
            pil_image = Image.fromarray(image, mode='L')
        elif len(image.shape) == 3 and image.shape[2] == 3:
            # RGB
            pil_image = Image.fromarray(image, mode='RGB')
        else:
            raise ValueError(f"Unsupported image shape: {image.shape}")
        
        # Calculate aspect ratio
        width, height = pil_image.size
        aspect_ratio = width / height
        
        # Calculate new dimensions to fit in size x size while maintaining aspect ratio
        if aspect_ratio > 1:
            new_width = size
            new_height = int(size / aspect_ratio)
        else:
            new_width = int(size * aspect_ratio)
            new_height = size
        
        # Resize image
        resized = pil_image.resize((new_width, new_height), Image.Resampling.LANCZOS)
        
        # Create black canvas
        if pil_image.mode == 'L':
            canvas = Image.new('L', (size, size), 0)
        else:
            canvas = Image.new('RGB', (size, size), (0, 0, 0))
        
        # Paste resized image in center
        paste_x = (size - new_width) // 2
        paste_y = (size - new_height) // 2
        canvas.paste(resized, (paste_x, paste_y))
        
        # Convert to PNG bytes
        buffer = io.BytesIO()
        canvas.save(buffer, format='PNG', optimize=True)
        png_bytes = buffer.getvalue()
        
        # Encode as base64 (without data URL prefix)
        base64_string = base64.b64encode(png_bytes).decode('utf-8')
        
        return base64_string
        
    except Exception as e:
        logger.error(f"Failed to generate preview: {e}", exc_info=True)
        raise ValueError(f"Preview generation failed: {e}")

