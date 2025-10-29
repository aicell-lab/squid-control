"""
microSAM segmentation client for automated cell segmentation.

This module provides helper functions to connect to the microSAM BioEngine service
and perform automated instance segmentation on microscopy images.
"""

import asyncio
import logging
import numpy as np
import cv2
import base64
from typing import Optional, List, Callable, Dict, Any

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
    Apply percentile-based contrast adjustment to image.
    
    Args:
        image: Input image array
        min_percentile: Lower percentile (e.g., 1.0 for 1st percentile)
        max_percentile: Upper percentile (e.g., 99.0 for 99th percentile)
    
    Returns:
        Contrast-adjusted image normalized to 0-255 range
    """
    logger.debug(f"Applying contrast adjustment: {min_percentile}%-{max_percentile}%")
    
    # Calculate percentile values
    p_min = np.percentile(image, min_percentile)
    p_max = np.percentile(image, max_percentile)
    
    logger.debug(f"Percentile values: min={p_min:.2f}, max={p_max:.2f}")
    
    # Clip and normalize
    clipped = np.clip(image, p_min, p_max)
    
    # Avoid division by zero
    if p_max > p_min:
        normalized = ((clipped - p_min) / (p_max - p_min) * 255).astype(np.uint8)
        logger.debug(f"Contrast adjustment applied: {image.min()}-{image.max()} -> 0-255")
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
        segmentation: Instance segmentation mask as uint16 array with shape (H, W)
                     Each cell has a unique integer ID (0=background, 1-N=objects)
    
    Raises:
        ValueError: If image shape is unsupported
        Exception: If segmentation fails
    
    Note:
        For multi-channel inputs, use merge_channels_with_colors() first to create RGB image.
        JPEG compression significantly reduces payload size (typically 10:1 ratio) and prevents WebSocket timeouts.
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
    # microSAM should decode the JPEG back to numpy array
    logger.info(f"Calling microSAM segment_all with JPEG-compressed image "
               f"({compressed_size_mb:.2f} MB)")
    
    segmentation_result = await microsam_service.segment_all(
        image_or_embedding=jpeg_base64,  # Send base64-encoded JPEG
        embedding=False,
        format='jpeg'  # Indicate format to microSAM
    )
    
    try:
        logger.info(f"✅ Segmentation completed! Result shape: {segmentation_result.shape}, dtype: {segmentation_result.dtype}")
        
        # Convert to uint16 for storage (supports up to 65,535 unique objects)
        if segmentation_result.dtype != np.uint16:
            # Check if we need to handle the conversion carefully
            max_label = segmentation_result.max()
            if max_label > 65535:
                logger.warning(f"Segmentation has {max_label} objects, truncating to uint16 range")
            segmentation_result = segmentation_result.astype(np.uint16)
            logger.debug(f"Converted segmentation to uint16")
        
        # Count unique objects (excluding background)
        unique_labels = np.unique(segmentation_result)
        num_objects = len(unique_labels) - 1 if 0 in unique_labels else len(unique_labels)
        logger.info(f"🔬 Detected {num_objects} objects in segmentation")
        
        return segmentation_result
        
    except Exception as e:
        logger.error(f"❌ Segmentation failed: {e}", exc_info=True)
        raise Exception(f"microSAM segmentation failed: {e}")


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
        segmentation: Segmentation mask as uint16 array, or None if well not found
    
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
            # Segment the well region with multi-channel support
            segmentation = await segment_well_region(
                microsam_service=microsam_service,
                experiment_manager=experiment_manager,
                source_experiment=source_experiment,
                well_row=well_row,
                well_column=well_column,
                channel_configs=channel_configs,
                scale_level=scale_level,
                timepoint=timepoint,
                well_plate_type=well_plate_type,
                well_padding_mm=well_padding_mm
            )
            
            if segmentation is None:
                logger.warning(f"Segmentation returned None for well {well_id}")
                results['failed_wells'] += 1
                results['wells_failed'].append(well_id)
                continue
            
            # Save segmentation to the segmentation experiment
            # Get or create well canvas in segmentation experiment
            seg_canvas = experiment_manager.get_well_canvas(
                well_row, well_column, well_plate_type, well_padding_mm, segmentation_experiment
            )
            
            # Use channel name "Segmentation" for the segmentation masks
            segmentation_channel = "Segmentation"
            
            # Get channel index for segmentation channel
            if segmentation_channel in seg_canvas.channel_to_zarr_index:
                channel_idx = seg_canvas.channel_to_zarr_index[segmentation_channel]
            else:
                # Add new channel if it doesn't exist
                channel_idx = len(seg_canvas.channels)
                seg_canvas.channels.append(segmentation_channel)
                seg_canvas.channel_to_zarr_index[segmentation_channel] = channel_idx
                logger.info(f"Added new channel '{segmentation_channel}' at index {channel_idx}")
            
            # Get well center for coordinate system
            well_info = seg_canvas.get_well_info()
            center_x = well_info['well_info']['well_center_x_mm']
            center_y = well_info['well_info']['well_center_y_mm']
            
            # Add segmentation mask to canvas
            # Note: add_image_sync expects absolute stage coordinates
            seg_canvas.add_image_sync(
                image=segmentation,
                x_mm=center_x,  # Use well center
                y_mm=center_y,
                channel_idx=channel_idx,
                z_idx=0,
                timepoint=timepoint
            )
            
            logger.info(f"✅ Saved segmentation for well {well_id} to '{segmentation_experiment}'")
            
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

