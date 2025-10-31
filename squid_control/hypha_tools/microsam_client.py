"""
microSAM segmentation client for automated cell segmentation.

This module provides helper functions to connect to the microSAM BioEngine service
and perform automated instance segmentation on microscopy images.
"""

import asyncio
import base64
import logging
from typing import Any, Callable, Dict, List, Optional

import cv2
import numpy as np

logger = logging.getLogger(__name__)


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


def encode_image_to_jpeg(image: np.ndarray, quality: int = 95) -> bytes:
    """
    Encode numpy array to JPEG bytes for efficient transmission.
    
    Args:
        image: Image array in (H, W) for grayscale or (H, W, 3) for RGB, dtype uint8
        quality: JPEG quality (1-100, default 95 for high quality)
    
    Returns:
        jpeg_bytes: Compressed JPEG image as bytes
    
    Raises:
        ValueError: If encoding fails
    """
    try:
        if len(image.shape) == 2:
            # Grayscale image
            success, encoded = cv2.imencode('.jpg', image, 
                                          [cv2.IMWRITE_JPEG_QUALITY, quality])
        elif len(image.shape) == 3 and image.shape[2] == 3:
            # RGB image - convert RGB to BGR for OpenCV
            bgr_image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            success, encoded = cv2.imencode('.jpg', bgr_image,
                                           [cv2.IMWRITE_JPEG_QUALITY, quality])
        else:
            raise ValueError(f"Unsupported image shape for JPEG encoding: {image.shape}")
        
        if not success:
            raise ValueError("Failed to encode image to JPEG")
        
        jpeg_bytes = encoded.tobytes()
        original_size = image.nbytes
        compressed_size = len(jpeg_bytes)
        compression_ratio = original_size / compressed_size if compressed_size > 0 else 1.0
        
        logger.debug(f"JPEG encoding: {original_size:,} bytes -> {compressed_size:,} bytes "
                    f"(ratio: {compression_ratio:.2f}x, quality: {quality})")
        
        return jpeg_bytes
        
    except Exception as e:
        logger.error(f"Error encoding image to JPEG: {e}")
        raise ValueError(f"JPEG encoding failed: {e}")


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
                       jpeg_quality: int = 95) -> np.ndarray:
    """
    Segment a single image using the microSAM service with contrast adjustment.
    Images are compressed to JPEG before transmission to reduce payload size and prevent WebSocket blocking.
    
    Args:
        microsam_service: Connected microSAM service
        image_array: Image array in (H, W) for grayscale or (H, W, 3) for RGB
                     Should be contrast-adjusted if using direct input
        min_contrast_percentile: Lower percentile for contrast (default: 1.0)
        max_contrast_percentile: Upper percentile for contrast (default: 99.0)
        jpeg_quality: JPEG quality 1-100, higher = better quality but larger file (default: 95)
    
    Returns:
        segmentation: Binary segmentation mask as uint8 array with shape (H, W)
                     Binary mask: 0 = background, 255 = segmented pixels
    
    Raises:
        ValueError: If image shape is unsupported
        Exception: If segmentation fails
    
    Note:
        For multi-channel inputs, use merge_channels_with_colors() first to create RGB image.
        JPEG compression significantly reduces payload size (typically 10:1 ratio) and prevents WebSocket timeouts.
        The returned mask is a binary mask (uint8), not instance segmentation with unique IDs.
    """
    logger.info(f"Segmenting image with shape: {image_array.shape}, dtype: {image_array.dtype}")
    logger.info(f"Contrast adjustment: {min_contrast_percentile}%-{max_contrast_percentile}%")
    
    # Apply contrast adjustment
    adjusted_image = apply_contrast_adjustment(
        image_array, min_contrast_percentile, max_contrast_percentile
    )
    
    original_size_mb = adjusted_image.nbytes / (1024 * 1024)
    
    # Compress to JPEG for efficient transmission
    logger.info(f"Compressing image to JPEG (quality={jpeg_quality}) for transmission...")
    jpeg_bytes = encode_image_to_jpeg(adjusted_image, quality=jpeg_quality)
    compressed_size_mb = len(jpeg_bytes) / (1024 * 1024)
    compression_ratio = original_size_mb / compressed_size_mb if compressed_size_mb > 0 else 1.0
    
    logger.info(f"Image compression: {original_size_mb:.2f} MB -> {compressed_size_mb:.2f} MB "
               f"({compression_ratio:.1f}x reduction)")
    
    # Encode JPEG bytes to base64 for safe JSON transmission
    jpeg_base64 = base64.b64encode(jpeg_bytes).decode('utf-8')
    
    # Send JPEG-compressed image to microSAM
    # microSAM returns a JPEG base64 string containing a binary mask (uint8: 0 or 255)
    logger.info(f"Calling microSAM segment_all with JPEG-compressed image "
               f"({compressed_size_mb:.2f} MB)")
    
    segmentation_result_b64 = await microsam_service.segment_all(
        image_or_embedding=jpeg_base64,  # Send base64-encoded JPEG
        embedding=False
    )
    
    try:
        # Decode the returned JPEG base64 string back to numpy array
        # microSAM returns a JPEG base64 string containing a binary mask (uint8: 0 or 255)
        if isinstance(segmentation_result_b64, str):
            # Decode base64 to bytes
            segmentation_jpeg_bytes = base64.b64decode(segmentation_result_b64)
            # Decode JPEG bytes to numpy array (grayscale binary mask: 0 or 255)
            nparr = np.frombuffer(segmentation_jpeg_bytes, np.uint8)
            segmentation_result = cv2.imdecode(nparr, cv2.IMREAD_GRAYSCALE)
            
            if segmentation_result is None:
                raise ValueError("Failed to decode segmentation result from JPEG")
            
            logger.info(f"✅ Segmentation completed! Result shape: {segmentation_result.shape}, dtype: {segmentation_result.dtype}")
        else:
            # Fallback: if it's already a numpy array (shouldn't happen per server definition)
            segmentation_result = segmentation_result_b64
            logger.info(f"✅ Segmentation completed! Result shape: {segmentation_result.shape}, dtype: {segmentation_result.dtype}")
        
        # Ensure result is uint8 binary mask (0 or 255)
        # The mask should already be in this format from microSAM, but verify and normalize if needed
        if segmentation_result.dtype != np.uint8:
            segmentation_result = segmentation_result.astype(np.uint8)
        
        # Normalize to binary mask: values > 0 become 255, 0 stays 0
        # This handles any potential variations in the mask values
        segmentation_result = np.where(segmentation_result > 0, 255, 0).astype(np.uint8)
        
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
                
                # Calculate tile center coordinates
                # Grid indices: 0-8, tiles should cover full canvas from -canvas_half to +canvas_half
                # Tile centers are spaced by base_tile_size (without overlap)
                # Each tile region extends 5% beyond its boundary to create overlap
                tile_center_x = center_x + (grid_i - (grid_size - 1) / 2.0) * base_tile_size_mm
                tile_center_y = center_y + (grid_j - (grid_size - 1) / 2.0) * base_tile_size_mm
                
                logger.debug(f"  Tile {tile_idx}/{total_tiles} ({grid_i},{grid_j}): center=({tile_center_x:.2f}, {tile_center_y:.2f})mm")
                
                try:
                    # Load and process each channel for this tile
                    processed_channels = []
                    channel_names = []
                    
                    for config in channel_configs:
                        channel_name = config['channel']
                        min_percentile = config.get('min_percentile', 1.0)
                        max_percentile = config.get('max_percentile', 99.0)
                        weight = config.get('weight', 1.0)
                        
                        # Get tile region from this channel
                        # Use overlapped tile size (5% larger than base) to ensure proper segmentation at boundaries
                        request_size = min(tile_size_mm, well_diameter)
                        channel_data = canvas.get_canvas_region_by_channel_name(
                            tile_center_x, tile_center_y, request_size, request_size,
                            channel_name, scale=scale_level, timepoint=timepoint
                        )
                        
                        if channel_data is None:
                            logger.debug(f"    No data for channel {channel_name} in tile {tile_idx}, skipping")
                            continue
                        
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
                    
                    # Segment this tile
                    # Store original image dimensions to verify segmentation result matches
                    original_image_shape = merged_image.shape[:2]  # (height, width)
                    
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
                    
                    # Verify segmentation mask dimensions match source image
                    seg_shape = tile_segmentation.shape[:2]  # (height, width)
                    if seg_shape != original_image_shape:
                        logger.warning(
                            f"  Tile {tile_idx}: Segmentation mask size mismatch! "
                            f"Source: {original_image_shape}, Segmentation: {seg_shape}. "
                            f"Resizing segmentation to match source dimensions."
                        )
                        # Resize segmentation to match source image dimensions
                        tile_segmentation = cv2.resize(
                            tile_segmentation, 
                            (original_image_shape[1], original_image_shape[0]),  # (width, height)
                            interpolation=cv2.INTER_NEAREST  # Preserve label values
                        )
                    
                    # Write tile segmentation mask to segmentation canvas at the EXACT same location
                    # as the source image was read from (same tile_center_x, tile_center_y coordinates)
                    seg_canvas.add_image_sync(
                        image=tile_segmentation,
                        x_mm=tile_center_x,  # Same coordinate as source image read
                        y_mm=tile_center_y,  # Same coordinate as source image read
                        channel_idx=channel_idx,
                        z_idx=0,
                        timepoint=timepoint
                    )
                    
                    logger.debug(
                        f"  Tile {tile_idx}: Wrote segmentation mask (shape={tile_segmentation.shape}) "
                        f"to position ({tile_center_x:.2f}, {tile_center_y:.2f})mm "
                        f"- same location as source image"
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
        return stats
        
    except Exception as e:
        logger.error(f"Failed to segment well {well_id}: {e}", exc_info=True)
        stats['tiles_failed'] = total_tiles - stats['tiles_processed']
        return stats


async def segment_well_region(
    microsam_service,
    experiment_manager,
    source_experiment: str,
    well_row: str,
    well_column: int,
    channel_configs: List[Dict[str, Any]],
    scale_level: int,
    timepoint: int,
    well_plate_type: str,
    well_padding_mm: float
) -> Optional[np.ndarray]:
    """
    Get well region from multiple channels, merge with colors, and segment.
    
    Args:
        microsam_service: Connected microSAM service
        experiment_manager: ExperimentManager instance
        source_experiment: Name of the source experiment
        well_row: Well row letter (e.g., 'A', 'B')
        well_column: Well column number (e.g., 1, 2)
        channel_configs: List of dicts with:
            - 'channel': Channel name (str)
            - 'min_percentile': Lower contrast percentile (float, default 1.0)
            - 'max_percentile': Upper contrast percentile (float, default 99.0)
            - 'weight': Blend weight (float, default 1.0, optional)
        scale_level: Pyramid scale level (0=full resolution)
        timepoint: Timepoint index
        well_plate_type: Well plate format ('96', '384', etc.)
        well_padding_mm: Well padding in millimeters
    
    Returns:
        segmentation: Binary segmentation mask as uint8 array (0 = background, 255 = segmented pixels),
                     or None if well not found
    
    Example:
        channel_configs = [
            {"channel": "BF LED matrix full", "min_percentile": 1.0, "max_percentile": 99.0},
            {"channel": "Fluorescence 488 nm Ex", "min_percentile": 5.0, "max_percentile": 95.0}
        ]
    """
    well_id = f"{well_row}{well_column}"
    logger.info(f"📍 Processing well {well_id} with {len(channel_configs)} channels")
    
    try:
        # Get well canvas from source experiment
        canvas = experiment_manager.get_well_canvas(
            well_row, well_column, well_plate_type, well_padding_mm, source_experiment
        )
        
        # Get well center coordinates and diameter
        well_info = canvas.get_well_info()
        center_x = well_info['well_info']['well_center_x_mm']
        center_y = well_info['well_info']['well_center_y_mm']
        well_diameter = well_info['well_info']['well_diameter_mm']
        
        logger.info(f"Well {well_id} - center: ({center_x:.2f}, {center_y:.2f}), diameter: {well_diameter:.2f}mm")
        
        # Process each channel with its contrast settings
        processed_channels = []
        channel_names = []
        
        for config in channel_configs:
            channel_name = config['channel']
            min_percentile = config.get('min_percentile', 1.0)
            max_percentile = config.get('max_percentile', 99.0)
            weight = config.get('weight', 1.0)
            
            logger.info(f"  Channel: {channel_name}, contrast: {min_percentile}%-{max_percentile}%, weight: {weight}")
            
            # Get channel data
            channel_data = canvas.get_canvas_region_by_channel_name(
                center_x, center_y, well_diameter, well_diameter,
                channel_name, scale=scale_level, timepoint=timepoint
            )
            
            if channel_data is None:
                logger.warning(f"  No data for channel {channel_name}, skipping")
                continue
            
            # Apply contrast adjustment
            adjusted = apply_contrast_adjustment(channel_data, min_percentile, max_percentile)
            
            # Apply weight
            if weight != 1.0:
                adjusted = np.clip(adjusted * weight, 0, 255).astype(np.uint8)
            
            processed_channels.append(adjusted)
            channel_names.append(channel_name)
        
        if not processed_channels:
            logger.error(f"No valid channels found for well {well_id}")
            return None
        
        # Merge channels using OME-Zarr colors
        if len(processed_channels) == 1:
            # Single channel - use grayscale input (no contrast adjustment needed, already done)
            merged_image = processed_channels[0]
            logger.info(f"  Single channel mode: shape={merged_image.shape}, dtype={merged_image.dtype}")
        else:
            # Multi-channel - merge to RGB
            merged_image = merge_channels_with_colors(processed_channels, channel_names)
            logger.info(f"  Multi-channel merged: shape={merged_image.shape}, dtype={merged_image.dtype}")
        
        # Segment the merged image (no additional contrast adjustment needed)
        segmentation = await segment_image(microsam_service, merged_image, 
                                          min_contrast_percentile=0.0, 
                                          max_contrast_percentile=100.0)  # No additional contrast
        
        logger.info(f"✅ Well {well_id} segmentation complete: shape={segmentation.shape}")
        return segmentation
        
    except Exception as e:
        logger.error(f"Failed to segment well {well_id}: {e}", exc_info=True)
        return None


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
    except ValueError as e:
        # Experiment already exists
        logger.info(f"Segmentation experiment '{segmentation_experiment}' already exists, using existing")
    
    # Set segmentation experiment as active for writing
    experiment_manager.set_active_experiment(segmentation_experiment)
    
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
            # Get or create well canvas in segmentation experiment
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
    
    logger.info(f"🎉 Batch segmentation complete!")
    logger.info(f"  Successful: {results['successful_wells']}/{results['total_wells']}")
    logger.info(f"  Failed: {results['failed_wells']}/{results['total_wells']}")
    logger.info(f"  Channels used: {len(channel_configs)}")
    
    return results

