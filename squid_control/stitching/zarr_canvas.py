import os
import numpy as np
import zarr
import logging
import threading
import time
from pathlib import Path
from typing import Tuple, Optional, Dict, List
import asyncio
from concurrent.futures import ThreadPoolExecutor
import json
from PIL import Image
from datetime import datetime
import cv2

# Get the logger for this module
logger = logging.getLogger(__name__)

# Ensure the logger has the same level as the root logger
# This ensures our INFO messages are actually displayed
if not logger.handlers:
    # If no handlers are set up, inherit from the root logger
    logger.setLevel(logging.INFO)
    # Add a handler that matches the main service format
    handler = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.propagate = False  # Prevent double logging

class ZarrCanvas:
    """
    Manages an OME-Zarr canvas for live microscope image stitching.
    Creates and maintains a multi-scale pyramid structure for efficient viewing.
    """
    
    def __init__(self, base_path: str, pixel_size_xy_um: float, stage_limits: Dict[str, float], 
                 channels: List[str] = None, chunk_size: int = 256, rotation_angle_deg: float = 0.0,
                 initial_timepoints: int = 20, timepoint_expansion_chunk: int = 10, fileset_name: str = "live_stitching",
                 initialize_new: bool = False):
        """
        Initialize the Zarr canvas.
        
        Args:
            base_path: Base directory for zarr storage (from ZARR_PATH env variable)
            pixel_size_xy_um: Pixel size in micrometers
            stage_limits: Dictionary with x_positive, x_negative, y_positive, y_negative in mm
            channels: List of channel names (human-readable names)
            chunk_size: Size of chunks in pixels (default 256)
            rotation_angle_deg: Rotation angle for stitching in degrees (positive=clockwise, negative=counterclockwise)
            initial_timepoints: Number of timepoints to pre-allocate during initialization (default 20)
            timepoint_expansion_chunk: Number of timepoints to add when expansion is needed (default 10)
            fileset_name: Name of the zarr fileset (default 'live_stitching')
            initialize_new: If True, create a new fileset (deletes existing). If False, open existing if present.
        """
        self.base_path = Path(base_path)
        self.pixel_size_xy_um = pixel_size_xy_um
        self.stage_limits = stage_limits
        self.channels = channels or ['BF_LED_matrix_full']
        self.chunk_size = chunk_size
        self.rotation_angle_deg = rotation_angle_deg
        self.fileset_name = fileset_name
        self.zarr_path = self.base_path / f"{fileset_name}.zarr"
        
        # Create channel mapping: channel_name -> local_zarr_index
        self.channel_to_zarr_index = {channel_name: idx for idx, channel_name in enumerate(self.channels)}
        self.zarr_index_to_channel = {idx: channel_name for idx, channel_name in enumerate(self.channels)}
        
        logger.info(f"Channel mapping: {self.channel_to_zarr_index}")
        
        # Calculate canvas dimensions in pixels based on stage limits
        self.stage_width_mm = stage_limits['x_positive'] - stage_limits['x_negative']
        self.stage_height_mm = stage_limits['y_positive'] - stage_limits['y_negative']
        
        # Convert to pixels (with some padding)
        padding_factor = 1.1  # 10% padding
        self.canvas_width_px = int((self.stage_width_mm * 1000 / pixel_size_xy_um) * padding_factor)
        self.canvas_height_px = int((self.stage_height_mm * 1000 / pixel_size_xy_um) * padding_factor)
        
        # Make dimensions divisible by chunk_size
        self.canvas_width_px = ((self.canvas_width_px + chunk_size - 1) // chunk_size) * chunk_size
        self.canvas_height_px = ((self.canvas_height_px + chunk_size - 1) // chunk_size) * chunk_size
        
        # Number of pyramid levels (scale0 is full res, scale1 is 1/4, scale2 is 1/16, etc)
        self.num_scales = self._calculate_num_scales()
        
        # Thread pool for async zarr operations
        self.executor = ThreadPoolExecutor(max_workers=4)
        
        # Lock for thread-safe zarr access
        self.zarr_lock = threading.RLock()
        
        # Queue for frame stitching
        self.stitch_queue = asyncio.Queue(maxsize=100)
        self.stitching_task = None
        self.is_stitching = False
        
        # Track available timepoints
        self.available_timepoints = [0]  # Start with timepoint 0 as a list
        
        # Only initialize or open
        if initialize_new or not self.zarr_path.exists():
            self.initialize_canvas()
        else:
            self.open_existing_canvas()
        
        logger.info(f"ZarrCanvas initialized: {self.canvas_width_px}x{self.canvas_height_px} px, "
                    f"{self.num_scales} scales, chunk_size={chunk_size}")
    
    def _calculate_num_scales(self) -> int:
        """Calculate the number of pyramid levels needed."""
        min_size = 512  # Minimum size for lowest resolution
        num_scales = 1
        width, height = self.canvas_width_px, self.canvas_height_px
        
        while width > min_size and height > min_size:
            width //= 4
            height //= 4
            num_scales += 1
            
        return min(num_scales, 6)  # Cap at 6 levels
    
    def get_zarr_channel_index(self, channel_name: str) -> int:
        """
        Get the local zarr array index for a channel name.
        
        Args:
            channel_name: Human-readable channel name
            
        Returns:
            int: Local index in the zarr array (0, 1, 2, etc.)
            
        Raises:
            ValueError: If channel name is not found
        """
        if channel_name not in self.channel_to_zarr_index:
            raise ValueError(f"Channel '{channel_name}' not found in zarr canvas. Available channels: {list(self.channel_to_zarr_index.keys())}")
        return self.channel_to_zarr_index[channel_name]
    
    def get_channel_name_by_zarr_index(self, zarr_index: int) -> str:
        """
        Get the channel name for a local zarr array index.
        
        Args:
            zarr_index: Local index in the zarr array
            
        Returns:
            str: Human-readable channel name
            
        Raises:
            ValueError: If zarr index is not found
        """
        if zarr_index not in self.zarr_index_to_channel:
            raise ValueError(f"Zarr index {zarr_index} not found. Available indices: {list(self.zarr_index_to_channel.keys())}")
        return self.zarr_index_to_channel[zarr_index]
    
    def initialize_canvas(self):
        """Initialize the OME-Zarr structure with proper metadata."""
        logger.info(f"Initializing OME-Zarr canvas at {self.zarr_path}")
        
        try:
            # Ensure the parent directory exists
            self.base_path.mkdir(parents=True, exist_ok=True)
            
            # Remove existing zarr if it exists and is corrupted
            if self.zarr_path.exists():
                import shutil
                shutil.rmtree(self.zarr_path)
                logger.info(f"Removed existing zarr directory: {self.zarr_path}")
            
            # Create the root group
            store = zarr.DirectoryStore(str(self.zarr_path))
            root = zarr.open_group(store=store, mode='w')
            
            # Import ChannelMapper for better metadata
            from squid_control.control.config import ChannelMapper
            
            # Create enhanced channel metadata with proper colors and info
            omero_channels = []
            for ch in self.channels:
                try:
                    channel_info = ChannelMapper.get_channel_by_human_name(ch)
                    # Assign colors based on channel type
                    if channel_info.channel_id == 0:  # BF
                        color = "FFFFFF"  # White
                    elif channel_info.channel_id == 11:  # 405nm
                        color = "8000FF"  # Blue-violet
                    elif channel_info.channel_id == 12:  # 488nm
                        color = "00FF00"  # Green
                    elif channel_info.channel_id == 13:  # 638nm
                        color = "FF0000"  # Red
                    elif channel_info.channel_id == 14:  # 561nm
                        color = "FFFF00"  # Yellow
                    elif channel_info.channel_id == 15:  # 730nm
                        color = "FF00FF"  # Magenta
                    else:
                        color = "FFFFFF"  # Default white
                        
                    omero_channels.append({
                        "label": ch,
                        "color": color,
                        "active": True,
                        "window": {"start": 0, "end": 255},
                        "family": "linear",
                        "coefficient": 1.0
                    })
                except ValueError:
                    # Fallback for unknown channels
                    omero_channels.append({
                        "label": ch,
                        "color": "FFFFFF",
                        "active": True,
                        "window": {"start": 0, "end": 255},
                        "family": "linear",
                        "coefficient": 1.0
                    })
            
            # Create OME-Zarr metadata
            multiscales_metadata = {
                "multiscales": [{
                    "axes": [
                        {"name": "t", "type": "time", "unit": "second"},
                        {"name": "c", "type": "channel"},
                        {"name": "z", "type": "space", "unit": "micrometer"},
                        {"name": "y", "type": "space", "unit": "micrometer"},
                        {"name": "x", "type": "space", "unit": "micrometer"}
                    ],
                    "datasets": [],
                    "name": self.fileset_name,
                    "version": "0.4"
                }],
                "omero": {
                    "id": 1,
                    "name": f"Squid Microscope Live Stitching ({self.fileset_name})",
                    "channels": omero_channels,
                    "rdefs": {
                        "defaultT": 0,
                        "defaultZ": 0,
                        "model": "color"
                    }
                },
                "squid_canvas": {
                    "channel_mapping": self.channel_to_zarr_index,
                    "zarr_index_mapping": self.zarr_index_to_channel,
                    "rotation_angle_deg": self.rotation_angle_deg,
                    "pixel_size_xy_um": self.pixel_size_xy_um,
                    "stage_limits": self.stage_limits,
                    "available_timepoints": sorted(self.available_timepoints),
                    "num_timepoints": len(self.available_timepoints),
                    "version": "1.0",
                    "fileset_name": self.fileset_name
                }
            }
            
            # Create arrays for each scale level
            for scale in range(self.num_scales):
                scale_factor = 4 ** scale
                width = self.canvas_width_px // scale_factor
                height = self.canvas_height_px // scale_factor
                
                # Create the array (T, C, Z, Y, X)
                # For now: 1 timepoint, len(channels) channels, 1 z-slice
                array = root.create_dataset(
                    str(scale),
                    shape=(1, len(self.channels), 1, height, width),
                    chunks=(1, 1, 1, self.chunk_size, self.chunk_size),
                    dtype='uint8',
                    fill_value=0,
                    overwrite=True
                )
                
                # Add scale metadata
                scale_transform = self.pixel_size_xy_um * scale_factor
                dataset_meta = {
                    "path": str(scale),
                    "coordinateTransformations": [{
                        "type": "scale",
                        "scale": [1.0, 1.0, 1.0, scale_transform, scale_transform]
                    }]
                }
                multiscales_metadata["multiscales"][0]["datasets"].append(dataset_meta)
            
            # Write metadata
            root.attrs.update(multiscales_metadata)
            
            # Store references to arrays
            self.zarr_arrays = {}
            for scale in range(self.num_scales):
                self.zarr_arrays[scale] = root[str(scale)]
                
            logger.info(f"OME-Zarr canvas initialized successfully with {self.num_scales} scales")
            
        except Exception as e:
            logger.error(f"Failed to initialize OME-Zarr canvas: {e}")
            raise RuntimeError(f"Cannot initialize zarr canvas: {e}")
    
    def open_existing_canvas(self):
        """Open an existing OME-Zarr structure from disk without deleting data."""
        import zarr
        store = zarr.DirectoryStore(str(self.zarr_path))
        root = zarr.open_group(store=store, mode='r+')
        self.zarr_root = root
        # Load arrays for each scale
        self.zarr_arrays = {}
        for scale in range(self.num_scales):
            if str(scale) in root:
                self.zarr_arrays[scale] = root[str(scale)]
        # Try to load available timepoints from metadata
        if 'squid_canvas' in root.attrs and 'available_timepoints' in root.attrs['squid_canvas']:
            self.available_timepoints = list(root.attrs['squid_canvas']['available_timepoints'])
        else:
            self.available_timepoints = [0]
        logger.info(f"Opened existing Zarr canvas at {self.zarr_path}")
    
    def stage_to_pixel_coords(self, x_mm: float, y_mm: float, scale: int = 0) -> Tuple[int, int]:
        """
        Convert stage coordinates (mm) to pixel coordinates for a given scale.
        
        Args:
            x_mm: X position in millimeters
            y_mm: Y position in millimeters  
            scale: Scale level (0 = full resolution)
            
        Returns:
            Tuple of (x_pixel, y_pixel) coordinates
        """
        # Offset to make all coordinates positive
        x_offset_mm = -self.stage_limits['x_negative']
        y_offset_mm = -self.stage_limits['y_negative']
        
        # Convert to pixels at scale 0
        x_px = int((x_mm + x_offset_mm) * 1000 / self.pixel_size_xy_um)
        y_px = int((y_mm + y_offset_mm) * 1000 / self.pixel_size_xy_um)
        
        # Apply scale factor
        scale_factor = 4 ** scale
        x_px //= scale_factor
        y_px //= scale_factor
        
        return x_px, y_px
    
    def _rotate_and_crop_image(self, image: np.ndarray) -> np.ndarray:
        """
        Rotate an image by the configured angle and crop to 95% of the original size.
        
        Args:
            image: Input image array (2D)
            
        Returns:
            Rotated and cropped image array
        """
        if abs(self.rotation_angle_deg) < 0.001:  # No rotation needed
            return image
            
        height, width = image.shape[:2]
        
        # Calculate rotation matrix
        center = (width // 2, height // 2)
        rotation_matrix = cv2.getRotationMatrix2D(center, self.rotation_angle_deg, 1.0)
        
        # Perform rotation, positive angle means counterclockwise rotation
        rotated = cv2.warpAffine(image, rotation_matrix, (width, height), 
                                flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        
        # Crop to 95% of original size to remove black borders
        crop_factor = 0.96
        new_height = int(height * crop_factor)
        new_width = int(width * crop_factor)
        
        # Calculate crop bounds (center crop)
        y_start = (height - new_height) // 2
        y_end = y_start + new_height
        x_start = (width - new_width) // 2
        x_end = x_start + new_width
        
        cropped = rotated[y_start:y_end, x_start:x_end]
        
        logger.debug(f"Rotated image by {self.rotation_angle_deg}° and cropped from {width}x{height} to {new_width}x{new_height}")
        
        return cropped
    
    def add_image_sync(self, image: np.ndarray, x_mm: float, y_mm: float, 
                       channel_idx: int = 0, z_idx: int = 0):
        """
        Synchronously add an image to the canvas at the specified position.
        Updates all pyramid levels.
        
        Args:
            image: Image array (2D)
            x_mm: X position in millimeters
            y_mm: Y position in millimeters
            channel_idx: Local zarr channel index (0, 1, 2, etc.)
            z_idx: Z-slice index (default 0)
        """
        # Validate channel index
        if channel_idx >= len(self.channels):
            logger.error(f"Channel index {channel_idx} out of bounds. Available channels: {len(self.channels)} (indices 0-{len(self.channels)-1})")
            return
        
        if channel_idx < 0:
            logger.error(f"Channel index {channel_idx} cannot be negative")
            return
        
        # Apply rotation and cropping first
        processed_image = self._rotate_and_crop_image(image)
        
        with self.zarr_lock:
            for scale in range(self.num_scales):
                scale_factor = 4 ** scale
                
                # Get pixel coordinates for this scale
                x_px, y_px = self.stage_to_pixel_coords(x_mm, y_mm, scale)
                
                # Resize image if needed
                if scale > 0:
                    new_size = (processed_image.shape[1] // scale_factor, processed_image.shape[0] // scale_factor)
                    scaled_image = cv2.resize(processed_image, new_size, interpolation=cv2.INTER_AREA)
                else:
                    scaled_image = processed_image
                
                # Get the zarr array for this scale
                zarr_array = self.zarr_arrays[scale]
                
                # Double-check zarr array dimensions
                if channel_idx >= zarr_array.shape[1]:
                    logger.error(f"Channel index {channel_idx} exceeds zarr array channel dimension {zarr_array.shape[1]}")
                    continue
                
                # Calculate bounds
                y_start = max(0, y_px - scaled_image.shape[0] // 2)
                y_end = min(zarr_array.shape[3], y_start + scaled_image.shape[0])
                x_start = max(0, x_px - scaled_image.shape[1] // 2)
                x_end = min(zarr_array.shape[4], x_start + scaled_image.shape[1])
                
                # Crop image if it extends beyond canvas
                img_y_start = max(0, -y_px + scaled_image.shape[0] // 2)
                img_y_end = img_y_start + (y_end - y_start)
                img_x_start = max(0, -x_px + scaled_image.shape[1] // 2)
                img_x_end = img_x_start + (x_end - x_start)
                
                # Write to zarr
                if y_end > y_start and x_end > x_start:
                    try:
                        zarr_array[0, channel_idx, z_idx, y_start:y_end, x_start:x_end] = \
                            scaled_image[img_y_start:img_y_end, img_x_start:img_x_end]
                        logger.debug(f"Successfully wrote image to zarr at scale {scale}, channel {channel_idx}")
                    except IndexError as e:
                        logger.error(f"IndexError writing to zarr array at scale {scale}, channel {channel_idx}: {e}")
                        logger.error(f"Zarr array shape: {zarr_array.shape}, trying to access channel {channel_idx}")
                    except Exception as e:
                        logger.error(f"Error writing to zarr array at scale {scale}, channel {channel_idx}: {e}")
    
    def add_image_sync_quick(self, image: np.ndarray, x_mm: float, y_mm: float, 
                           channel_idx: int = 0, z_idx: int = 0):
        """
        Synchronously add an image to the canvas for quick scan mode.
        Only updates scales 1-5 (skips scale 0 for performance).
        The input image should already be at scale1 resolution.
        
        Args:
            image: Image array (2D) - should be at scale1 resolution (1/4 of original)
            x_mm: X position in millimeters
            y_mm: Y position in millimeters
            channel_idx: Local zarr channel index (0, 1, 2, etc.)
            z_idx: Z-slice index (default 0)
        """
        # Validate channel index
        if channel_idx >= len(self.channels):
            logger.error(f"Channel index {channel_idx} out of bounds. Available channels: {len(self.channels)} (indices 0-{len(self.channels)-1})")
            return
        
        if channel_idx < 0:
            logger.error(f"Channel index {channel_idx} cannot be negative")
            return
        
        # For quick scan, we skip rotation to reduce computation pressure
        # The image should already be rotated and flipped by the caller
        processed_image = image
        
        with self.zarr_lock:
            # Only process scales 1-5 (skip scale 0 for performance)
            for scale in range(1, min(self.num_scales, 6)):  # scales 1-5
                scale_factor = 4 ** scale
                
                # Get pixel coordinates for this scale
                x_px, y_px = self.stage_to_pixel_coords(x_mm, y_mm, scale)
                
                # Resize image - note that input image is already at scale1 resolution
                # So for scale1: use image as-is
                # For scale2: resize by 1/4, scale3: by 1/16, etc.
                if scale == 1:
                    scaled_image = processed_image  # Already at scale1 resolution
                else:
                    # Scale relative to scale1 (which the input image represents)
                    relative_scale_factor = 4 ** (scale - 1)
                    new_size = (processed_image.shape[1] // relative_scale_factor, 
                               processed_image.shape[0] // relative_scale_factor)
                    scaled_image = cv2.resize(processed_image, new_size, interpolation=cv2.INTER_AREA)
                
                # Get the zarr array for this scale
                zarr_array = self.zarr_arrays[scale]
                
                # Double-check zarr array dimensions
                if channel_idx >= zarr_array.shape[1]:
                    logger.error(f"Channel index {channel_idx} exceeds zarr array channel dimension {zarr_array.shape[1]}")
                    continue
                
                # Calculate bounds
                y_start = max(0, y_px - scaled_image.shape[0] // 2)
                y_end = min(zarr_array.shape[3], y_start + scaled_image.shape[0])
                x_start = max(0, x_px - scaled_image.shape[1] // 2)
                x_end = min(zarr_array.shape[4], x_start + scaled_image.shape[1])
                
                # Crop image if it extends beyond canvas
                img_y_start = max(0, -y_px + scaled_image.shape[0] // 2)
                img_y_end = img_y_start + (y_end - y_start)
                img_x_start = max(0, -x_px + scaled_image.shape[1] // 2)
                img_x_end = img_x_start + (x_end - x_start)
                
                # Write to zarr
                if y_end > y_start and x_end > x_start:
                    try:
                        zarr_array[0, channel_idx, z_idx, y_start:y_end, x_start:x_end] = \
                            scaled_image[img_y_start:img_y_end, img_x_start:img_x_end]
                        logger.debug(f"Successfully wrote image to zarr at scale {scale}, channel {channel_idx} (quick scan)")
                    except IndexError as e:
                        logger.error(f"IndexError writing to zarr array at scale {scale}, channel {channel_idx}: {e}")
                        logger.error(f"Zarr array shape: {zarr_array.shape}, trying to access channel {channel_idx}")
                    except Exception as e:
                        logger.error(f"Error writing to zarr array at scale {scale}, channel {channel_idx}: {e}")
    
    async def add_image_async(self, image: np.ndarray, x_mm: float, y_mm: float,
                              channel_idx: int = 0, z_idx: int = 0):
        """Add image to the stitching queue for asynchronous processing."""
        await self.stitch_queue.put({
            'image': image.copy(),
            'x_mm': x_mm,
            'y_mm': y_mm,
            'channel_idx': channel_idx,
            'z_idx': z_idx,
            'timestamp': time.time()
        })
    
    async def start_stitching(self):
        """Start the background stitching task."""
        if not self.is_stitching:
            self.is_stitching = True
            self.stitching_task = asyncio.create_task(self._stitching_loop())
            logger.info("Started background stitching task")
    
    async def stop_stitching(self):
        """Stop the background stitching task and process all remaining images in queue."""
        self.is_stitching = False
        
        # Process any remaining images in the queue
        logger.info("Processing remaining images in stitching queue before stopping...")
        remaining_count = 0
        
        while not self.stitch_queue.empty():
            try:
                frame_data = await asyncio.wait_for(
                    self.stitch_queue.get(), 
                    timeout=0.1  # Short timeout to avoid hanging
                )
                
                # Check if this is a quick scan
                is_quick_scan = frame_data.get('quick_scan', False)
                
                # Process in thread pool to avoid blocking
                loop = asyncio.get_event_loop()
                if is_quick_scan:
                    await loop.run_in_executor(
                        self.executor,
                        self.add_image_sync_quick,
                        frame_data['image'],
                        frame_data['x_mm'],
                        frame_data['y_mm'],
                        frame_data['channel_idx'],
                        frame_data['z_idx']
                    )
                else:
                    await loop.run_in_executor(
                        self.executor,
                        self.add_image_sync,
                        frame_data['image'],
                        frame_data['x_mm'],
                        frame_data['y_mm'],
                        frame_data['channel_idx'],
                        frame_data['z_idx']
                    )
                remaining_count += 1
                
            except asyncio.TimeoutError:
                break  # No more items in queue
            except Exception as e:
                logger.error(f"Error processing remaining image in queue: {e}")
        
        if remaining_count > 0:
            logger.info(f"Processed {remaining_count} remaining images from stitching queue")
        
        # Wait for the stitching task to complete
        if self.stitching_task:
            await self.stitching_task
        
        # CRITICAL: Wait for all thread pool operations to complete
        logger.info("Waiting for all zarr operations to complete...")
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, self._wait_for_zarr_operations_complete)
        
        logger.info("Stopped background stitching task")
    
    async def _stitching_loop(self):
        """Background loop that processes the stitching queue."""
        while self.is_stitching:
            try:
                # Get frame from queue with timeout
                frame_data = await asyncio.wait_for(
                    self.stitch_queue.get(), 
                    timeout=1.0
                )
                
                # Check if this is a quick scan (only updates scales 1-5)
                is_quick_scan = frame_data.get('quick_scan', False)
                
                # Process in thread pool to avoid blocking
                loop = asyncio.get_event_loop()
                if is_quick_scan:
                    # Use quick scan method that only updates scales 1-5
                    await loop.run_in_executor(
                        self.executor,
                        self.add_image_sync_quick,
                        frame_data['image'],
                        frame_data['x_mm'],
                        frame_data['y_mm'],
                        frame_data['channel_idx'],
                        frame_data['z_idx']
                    )
                else:
                    # Use normal method that updates all scales
                    await loop.run_in_executor(
                        self.executor,
                        self.add_image_sync,
                        frame_data['image'],
                        frame_data['x_mm'],
                        frame_data['y_mm'],
                        frame_data['channel_idx'],
                        frame_data['z_idx']
                    )
                
            except asyncio.TimeoutError:
                continue
            except Exception as e:
                logger.error(f"Error in stitching loop: {e}")
        
        # Process any final images that might have been added during the last iteration
        logger.debug("Stitching loop exited, checking for any final images in queue...")
        final_count = 0
        while not self.stitch_queue.empty():
            try:
                frame_data = await asyncio.wait_for(
                    self.stitch_queue.get(), 
                    timeout=0.1
                )
                
                # Check if this is a quick scan
                is_quick_scan = frame_data.get('quick_scan', False)
                
                # Process in thread pool to avoid blocking
                loop = asyncio.get_event_loop()
                if is_quick_scan:
                    await loop.run_in_executor(
                        self.executor,
                        self.add_image_sync_quick,
                        frame_data['image'],
                        frame_data['x_mm'],
                        frame_data['y_mm'],
                        frame_data['channel_idx'],
                        frame_data['z_idx']
                    )
                else:
                    await loop.run_in_executor(
                        self.executor,
                        self.add_image_sync,
                        frame_data['image'],
                        frame_data['x_mm'],
                        frame_data['y_mm'],
                        frame_data['channel_idx'],
                        frame_data['z_idx']
                    )
                final_count += 1
                
            except asyncio.TimeoutError:
                break
            except Exception as e:
                logger.error(f"Error processing final image in stitching loop: {e}")
        
        if final_count > 0:
            logger.info(f"Stitching loop processed {final_count} final images before exiting")
    
    def get_canvas_region(self, x_mm: float, y_mm: float, width_mm: float, height_mm: float,
                          scale: int = 0, channel_idx: int = 0) -> np.ndarray:
        """
        Get a region from the canvas by zarr channel index.
        
        Args:
            x_mm: Center X position in millimeters
            y_mm: Center Y position in millimeters
            width_mm: Width in millimeters
            height_mm: Height in millimeters
            scale: Scale level to retrieve from
            channel_idx: Local zarr channel index (0, 1, 2, etc.)
            
        Returns:
            Retrieved image region as numpy array
        """
        # Validate channel index
        if channel_idx >= len(self.channels) or channel_idx < 0:
            logger.error(f"Channel index {channel_idx} out of bounds. Available channels: {len(self.channels)} (indices 0-{len(self.channels)-1})")
            return None
        
        with self.zarr_lock:
            # Validate zarr arrays exist
            if not hasattr(self, 'zarr_arrays') or scale not in self.zarr_arrays:
                logger.error(f"Zarr arrays not initialized or scale {scale} not available")
                return None
                
            zarr_array = self.zarr_arrays[scale]
            
            # Double-check zarr array dimensions
            if channel_idx >= zarr_array.shape[1]:
                logger.error(f"Channel index {channel_idx} exceeds zarr array channel dimension {zarr_array.shape[1]}")
                return None
            
            # Convert to pixel coordinates
            center_x_px, center_y_px = self.stage_to_pixel_coords(x_mm, y_mm, scale)
            
            scale_factor = 4 ** scale
            width_px = int(width_mm * 1000 / (self.pixel_size_xy_um * scale_factor))
            height_px = int(height_mm * 1000 / (self.pixel_size_xy_um * scale_factor))
            
            # Calculate bounds
            x_start = max(0, center_x_px - width_px // 2)
            x_end = min(zarr_array.shape[4], x_start + width_px)
            y_start = max(0, center_y_px - height_px // 2)
            y_end = min(zarr_array.shape[3], y_start + height_px)
            
            # Read from zarr
            try:
                region = zarr_array[0, channel_idx, 0, y_start:y_end, x_start:x_end]
                logger.debug(f"Successfully retrieved region from zarr at scale {scale}, channel {channel_idx}")
                return region
            except IndexError as e:
                logger.error(f"IndexError reading from zarr array at scale {scale}, channel {channel_idx}: {e}")
                logger.error(f"Zarr array shape: {zarr_array.shape}, trying to access channel {channel_idx}")
                return None
            except Exception as e:
                logger.error(f"Error reading from zarr array at scale {scale}, channel {channel_idx}: {e}")
                return None
    
    def get_canvas_region_by_channel_name(self, x_mm: float, y_mm: float, width_mm: float, height_mm: float,
                                         channel_name: str, scale: int = 0) -> np.ndarray:
        """
        Get a region from the canvas by channel name.
        
        Args:
            x_mm: Center X position in millimeters
            y_mm: Center Y position in millimeters
            width_mm: Width in millimeters
            height_mm: Height in millimeters
            channel_name: Human-readable channel name
            scale: Scale level to retrieve from
            
        Returns:
            Retrieved image region as numpy array
        """
        # Get the local zarr index for this channel
        try:
            channel_idx = self.get_zarr_channel_index(channel_name)
        except ValueError as e:
            logger.error(f"Channel not found: {e}")
            return None
            
        return self.get_canvas_region(x_mm, y_mm, width_mm, height_mm, scale, channel_idx)
    
    def close(self):
        """Clean up resources."""
        self.executor.shutdown(wait=True)
        logger.info("ZarrCanvas closed")
    
    def save_preview(self, action_ID: str = "canvas_preview"):
        """Save a preview image of the canvas at different scales."""
        try:
            preview_dir = self.base_path / "previews"
            preview_dir.mkdir(exist_ok=True)
            
            for scale in range(min(2, self.num_scales)):  # Save first 2 scales
                if scale in self.zarr_arrays:
                    # Get the first channel (usually brightfield)
                    array = self.zarr_arrays[scale]
                    if array.shape[1] > 0:  # Check if we have channels
                        # Get the image data (T=0, C=0, Z=0, :, :)
                        image_data = array[0, 0, 0, :, :]
                        
                        # Convert to PIL Image and save
                        if image_data.max() > image_data.min():  # Only save if there's actual data
                            # Normalize to 0-255
                            normalized = ((image_data - image_data.min()) / 
                                        (image_data.max() - image_data.min()) * 255).astype(np.uint8)
                            image = Image.fromarray(normalized)
                            preview_path = preview_dir / f"{action_ID}_scale{scale}.png"
                            image.save(preview_path)
                            logger.info(f"Saved preview: {preview_path}")
                        
        except Exception as e:
            logger.warning(f"Failed to save preview: {e}")
    
    def _flush_and_sync_zarr_arrays(self):
        """
        Flush and synchronize all zarr arrays to ensure all data is written to disk.
        This is critical before ZIP export to prevent race conditions.
        """
        try:
            with self.zarr_lock:
                if hasattr(self, 'zarr_arrays'):
                    for scale, zarr_array in self.zarr_arrays.items():
                        try:
                            # Flush any pending writes to disk
                            if hasattr(zarr_array, 'flush'):
                                zarr_array.flush()
                            # Sync the underlying store
                            if hasattr(zarr_array.store, 'sync'):
                                zarr_array.store.sync()
                            logger.debug(f"Flushed and synced zarr array scale {scale}")
                        except Exception as e:
                            logger.warning(f"Error flushing zarr array scale {scale}: {e}")
                
                # Also shutdown and recreate the thread pool to ensure all tasks are complete
                if hasattr(self, 'executor'):
                    self.executor.shutdown(wait=True)
                    self.executor = ThreadPoolExecutor(max_workers=4)
                    logger.info("Thread pool shutdown and recreated to ensure all zarr operations complete")
                
                # Give the filesystem a moment to complete any pending I/O
                import time
                time.sleep(0.1)
                
                logger.info("All zarr arrays flushed and synchronized")
                
        except Exception as e:
            logger.error(f"Error during zarr flush and sync: {e}")
            raise RuntimeError(f"Failed to flush zarr arrays: {e}")

    def export_as_zip(self) -> bytes:
        """
        Export the entire zarr canvas as a zip file for upload to artifact manager.
        
        Returns:
            bytes: The zip file content containing the entire zarr directory structure
        """
        import zipfile
        import io
        
        zip_buffer = io.BytesIO()
        
        try:
            # CRITICAL: Ensure all zarr operations are complete before ZIP export
            logger.info("Preparing zarr canvas for ZIP export...")
            self._flush_and_sync_zarr_arrays()
            
            # Create ZIP file with ZIP64 support for large archives
            # Use allowZip64=True to ensure ZIP64 format is used for large files
            zip_kwargs = {
                'mode': 'w',
                'compression': zipfile.ZIP_STORED,
                'compresslevel': None,
                'allowZip64': True  # Allow ZIP64 format for large files
            }
            
            # Create ZIP file with ZIP64 support
            with zipfile.ZipFile(zip_buffer, **zip_kwargs) as zip_file:
                logger.info("Creating ZIP archive with ZIP64 support...")
                
                # First, count total files for progress tracking
                total_files = 0
                for root, dirs, files in os.walk(self.zarr_path):
                    total_files += len(files)
                
                logger.info(f"Found {total_files} files to add to ZIP archive")
                
                # Walk through the entire zarr directory
                processed_files = 0
                for root, dirs, files in os.walk(self.zarr_path):
                    for file in files:
                        file_path = Path(root) / file
                        
                        # Verify the file is completely written and not currently being modified
                        try:
                            # Check if file exists and is readable
                            if not file_path.exists():
                                logger.warning(f"Skipping non-existent file: {file_path}")
                                continue
                            
                            # Try to open the file to ensure it's not locked
                            with open(file_path, 'rb') as f:
                                # Read a small portion to verify file integrity
                                f.read(1)
                                f.seek(0)
                            
                        except (IOError, OSError) as e:
                            logger.warning(f"Skipping file that cannot be read: {file_path} - {e}")
                            continue
                        
                        # Create relative path within the zarr directory
                        relative_path = file_path.relative_to(self.zarr_path)
                        # Build the archive name with fixed "data.zarr" prefix
                        if relative_path.parts:
                            # Join "data.zarr" with the relative path using forward slashes (ZIP standard)
                            arcname = "data.zarr/" + "/".join(relative_path.parts)
                        else:
                            # This shouldn't happen, but handle edge case
                            arcname = "data.zarr/" + file_path.name
                        
                        # Add file to ZIP with error handling
                        try:
                            zip_file.write(file_path, arcname=arcname)
                            processed_files += 1
                            
                            # Log progress every 1000 files
                            if processed_files % 1000 == 0:
                                logger.info(f"ZIP progress: {processed_files}/{total_files} files processed")
                                
                        except Exception as e:
                            logger.error(f"Failed to add file to ZIP: {file_path} - {e}")
                            # Continue with other files rather than failing completely
                            continue
                
                # Add a metadata file with canvas information
                metadata = {
                    "canvas_info": {
                        "pixel_size_xy_um": self.pixel_size_xy_um,
                        "rotation_angle_deg": self.rotation_angle_deg,
                        "stage_limits": self.stage_limits,
                        "channels": self.channels,
                        "num_scales": self.num_scales,
                        "canvas_size_px": {
                            "width": self.canvas_width_px,
                            "height": self.canvas_height_px
                        },
                        "export_timestamp": time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime()),
                        "squid_canvas_version": "1.0"
                    }
                }
                
                import json
                metadata_json = json.dumps(metadata, indent=2)
                zip_file.writestr("squid_canvas_metadata.json", metadata_json)
                logger.info("Added metadata file to ZIP archive")
                logger.info(f"ZIP creation completed: {processed_files} files processed")
                
            zip_buffer.seek(0)
            zip_content = zip_buffer.getvalue()
            zip_size_mb = len(zip_content) / (1024 * 1024)
            
            # Count files in the ZIP for logging
            import zipfile
            import io
            zip_buffer_for_count = io.BytesIO(zip_content)
            with zipfile.ZipFile(zip_buffer_for_count, 'r') as zf:
                file_count = len(zf.namelist())
            
            # Validate ZIP file integrity before returning
            self._validate_zip_file(zip_content)
            
            logger.info(f"ZIP archive created successfully: {file_count} files, {zip_size_mb:.2f} MB")
            return zip_content
            
        except Exception as e:
            logger.error(f"Failed to export zarr canvas as zip: {e}")
            raise RuntimeError(f"Cannot export zarr canvas: {e}")
        finally:
            zip_buffer.close()

    def _validate_zip_file(self, zip_content: bytes) -> None:
        """
        Validate ZIP file integrity to catch issues before upload.
        
        Args:
            zip_content (bytes): The ZIP file content to validate
            
        Raises:
            RuntimeError: If ZIP file is invalid or corrupted
        """
        try:
            import zipfile
            import io
            
            # Test if the ZIP file can be opened and read
            zip_buffer = io.BytesIO(zip_content)
            with zipfile.ZipFile(zip_buffer, 'r') as zip_file:
                # Test the ZIP file structure
                file_list = zip_file.namelist()
                if not file_list:
                    raise RuntimeError("ZIP file is empty")
                
                # Check if ZIP64 format is being used for large files
                zip_size_mb = len(zip_content) / (1024 * 1024)
                if zip_size_mb > 200:
                    # For large files, verify ZIP64 compatibility
                    try:
                        # Try to access the central directory
                        central_dir_info = zip_file.infolist()
                        if not central_dir_info:
                            raise RuntimeError("Cannot access central directory in large ZIP file")
                        
                        # Check for ZIP64 indicators in the central directory
                        zip64_indicators = []
                        for info in central_dir_info[:5]:  # Check first 5 files
                            # Check for large file sizes that would require ZIP64
                            if info.file_size >= 0xFFFFFFFF or info.compress_size >= 0xFFFFFFFF:
                                zip64_indicators.append("large_files")
                                break
                        
                        # Check total archive size vs ZIP32 limits
                        if zip_size_mb * 1024 * 1024 >= 0xFFFFFFFF:  # 4GB limit
                            zip64_indicators.append("archive_size")
                        
                        # Check file count vs ZIP32 limits
                        if len(file_list) >= 0xFFFF:  # 65535 file limit
                            zip64_indicators.append("file_count")
                        
                        if zip64_indicators:
                            logger.info(f"ZIP64 format validation passed for {zip_size_mb:.2f} MB file (indicators: {', '.join(zip64_indicators)})")
                        else:
                            logger.info(f"ZIP64 format validation passed for {zip_size_mb:.2f} MB file")
                            
                    except Exception as e:
                        raise RuntimeError(f"ZIP64 format validation failed: {e}")
                
                # Test a few files to ensure they can be read
                test_count = min(5, len(file_list))
                for i in range(test_count):
                    try:
                        with zip_file.open(file_list[i]) as f:
                            # Read a small chunk to verify file integrity
                            f.read(1024)
                    except Exception as e:
                        raise RuntimeError(f"Cannot read file {file_list[i]} from ZIP: {e}")
                
                # Additional validation for central directory integrity
                try:
                    # Test that we can get file info for all files
                    for info in zip_file.infolist()[:10]:  # Test first 10 files
                        if info.file_size > 0:  # Only test non-empty files
                            with zip_file.open(info) as f:
                                f.read(1)  # Read just one byte to verify access
                except Exception as e:
                    raise RuntimeError(f"Central directory validation failed: {e}")
                
                logger.info(f"ZIP file validation passed: {len(file_list)} files, {zip_size_mb:.2f} MB")
                
        except zipfile.BadZipFile as e:
            raise RuntimeError(f"Invalid ZIP file format: {e}")
        except Exception as e:
            raise RuntimeError(f"ZIP file validation failed: {e}")

    def get_export_info(self) -> dict:
        """
        Get information about the current canvas for export planning.
        
        Returns:
            dict: Information about canvas size, data, and export feasibility
        """
        try:
            # Calculate actual disk usage instead of theoretical array size
            total_size_bytes = 0
            data_arrays = 0
            file_count = 0
            
            if preview_region.size == 0:
                logger.warning('Preview region is empty, skipping preview generation')
                return None
            
            # Convert to PIL Image
            from PIL import Image
            from datetime import datetime
            
            # Ensure data is uint8
            if preview_region.dtype != np.uint8:
                # Scale to 8-bit
                if preview_region.max() > 0:
                    preview_region = (preview_region / preview_region.max() * 255).astype(np.uint8)
                else:
                    preview_region = preview_region.astype(np.uint8)
            
            # Create PIL image
            preview_img = Image.fromarray(preview_region)
            
            return {
                "canvas_path": str(self.zarr_path),
                "total_size_bytes": total_size_bytes,
                "total_size_mb": total_size_bytes / (1024 * 1024),
                "estimated_zip_size_mb": estimated_zip_size_mb,
                "file_count": file_count,
                "num_scales": self.num_scales,
                "num_channels": len(self.channels),
                "channels": self.channels,
                "arrays_with_data": data_arrays,
                "canvas_dimensions": {
                    "width_px": self.canvas_width_px,
                    "height_px": self.canvas_height_px,
                    "pixel_size_um": self.pixel_size_xy_um
                },
                "export_feasible": True  # Removed arbitrary size limit - let S3 handle large files
            }
            
        except Exception as e:
            logger.error(f"Failed to get export info: {e}")
            return {
                "error": str(e),
                "export_feasible": False
            } 

    def _wait_for_zarr_operations_complete(self):
        """
        Wait for all zarr operations to complete and ensure filesystem sync.
        This prevents race conditions with ZIP export.
        """
        import time
        
        with self.zarr_lock:
            # Shutdown thread pool and wait for all tasks to complete
            if hasattr(self, 'executor'):
                self.executor.shutdown(wait=True)
                self.executor = ThreadPoolExecutor(max_workers=4)
                logger.debug("Thread pool shutdown and recreated after stitching")
            
            # Flush all zarr arrays to ensure data is written
            if hasattr(self, 'zarr_arrays'):
                for scale, zarr_array in self.zarr_arrays.items():
                    try:
                        if hasattr(zarr_array, 'flush'):
                            zarr_array.flush()
                        if hasattr(zarr_array.store, 'sync'):
                            zarr_array.store.sync()
                    except Exception as e:
                        logger.warning(f"Error flushing zarr array scale {scale}: {e}")
            
            # Small delay to ensure filesystem operations complete
            time.sleep(0.2)
            
        logger.info("All zarr operations completed and synchronized") 
