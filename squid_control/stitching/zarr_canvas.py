import asyncio
import json
import logging
import os
import shutil
import tempfile
import threading
import time
import zipfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np
import zarr
from PIL import Image

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

class WellZarrCanvasBase:
    """
    Base class for well-specific zarr canvas functionality.
    Contains the core stitching and zarr management functionality without single-canvas assumptions.
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
        self.channels = channels or ['BF LED matrix full']
        self.chunk_size = chunk_size
        self.rotation_angle_deg = rotation_angle_deg
        self.fileset_name = fileset_name
        self.zarr_path = self.base_path / f"{fileset_name}.zarr"

        # Timepoint allocation strategy
        self.initial_timepoints = max(1, initial_timepoints)  # Ensure at least 1
        self.timepoint_expansion_chunk = max(1, timepoint_expansion_chunk)  # Ensure at least 1

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

        # Two-queue architecture for parallel processing:
        # 1. Preprocessing queue: receives raw images for CPU-intensive work (rotation, resizing)
        # 2. Write queue: receives preprocessed data for serialized zarr writes
        self.preprocessing_queue = asyncio.Queue(maxsize=500)  # Handle burst from camera
        self.zarr_write_queue = asyncio.Queue(maxsize=100)     # Preprocessed data ready to write

        self.stitching_task = None  # Preprocessing task
        self.writer_task = None     # Zarr writer task
        self.is_stitching = False   # Flag for preprocessing loop
        self.is_writing = False     # Flag for writer loop

        # Track available timepoints
        self.available_timepoints = [0]  # Start with timepoint 0 as a list

        # Only initialize or open
        if initialize_new or not self.zarr_path.exists():
            self.initialize_canvas()
        else:
            self.open_existing_canvas()

        logger.info(f"ZarrCanvas initialized: {self.canvas_width_px}x{self.canvas_height_px} px, "
                    f"{self.num_scales} scales, chunk_size={chunk_size}, "
                    f"initial_timepoints={self.initial_timepoints}, expansion_chunk={self.timepoint_expansion_chunk}")

    def _calculate_num_scales(self) -> int:
        """Calculate the number of pyramid levels needed."""
        min_size = 64  # Minimum size for lowest resolution
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

    def get_available_timepoints(self) -> List[int]:
        """
        Get a list of available timepoints in the zarr array.
        
        Returns:
            List[int]: Sorted list of available timepoint indices
        """
        with self.zarr_lock:
            return sorted(self.available_timepoints)

    def create_timepoint(self, timepoint: int):
        """
        Create a new timepoint in the zarr array.
        This is now a lightweight operation that just adds to the available list.
        Zarr array expansion happens lazily when actually writing data.
        
        Args:
            timepoint: The timepoint index to create
            
        Raises:
            ValueError: If timepoint already exists or is negative
        """
        if timepoint < 0:
            raise ValueError(f"Timepoint must be non-negative, got {timepoint}")

        with self.zarr_lock:
            if timepoint in self.available_timepoints:
                logger.info(f"Timepoint {timepoint} already exists")
                return

            logger.info(f"Creating new timepoint {timepoint} (lightweight)")

            # Simply add to available timepoints - zarr arrays will be expanded when needed
            self.available_timepoints.append(timepoint)
            self.available_timepoints.sort()  # Keep sorted for consistency

            # Update metadata
            self._update_timepoint_metadata()

    def pre_allocate_timepoints(self, max_timepoint: int):
        """
        Pre-allocate zarr arrays to accommodate timepoints up to max_timepoint.
        This is useful for time-lapse experiments where you know the number of timepoints in advance.
        Performing this operation early avoids delays during scanning.
        
        Args:
            max_timepoint: The maximum timepoint index to pre-allocate for
            
        Raises:
            ValueError: If max_timepoint is negative
        """
        if max_timepoint < 0:
            raise ValueError(f"Max timepoint must be non-negative, got {max_timepoint}")

        with self.zarr_lock:
            logger.info(f"Pre-allocating zarr arrays for timepoints up to {max_timepoint}")
            start_time = time.time()

            # Check if any arrays need expansion
            expansion_needed = False
            for scale in range(self.num_scales):
                if scale in self.zarr_arrays:
                    zarr_array = self.zarr_arrays[scale]
                    if max_timepoint >= zarr_array.shape[0]:
                        expansion_needed = True
                        break

            if not expansion_needed:
                logger.info(f"Zarr arrays already accommodate timepoint {max_timepoint}")
                return

            # Expand all arrays to accommodate max_timepoint
            self._ensure_timepoint_exists_in_zarr(max_timepoint)

            elapsed_time = time.time() - start_time
            logger.info(f"Pre-allocation completed in {elapsed_time:.2f} seconds")

    def remove_timepoint(self, timepoint: int):
        """
        Remove a timepoint from the zarr array by deleting its chunk files.
        
        Args:
            timepoint: The timepoint index to remove
            
        Raises:
            ValueError: If timepoint doesn't exist or is the last remaining timepoint
        """
        with self.zarr_lock:
            if timepoint not in self.available_timepoints:
                raise ValueError(f"Timepoint {timepoint} does not exist")

            if len(self.available_timepoints) == 1:
                raise ValueError("Cannot remove the last timepoint")

            logger.info(f"Removing timepoint {timepoint} and deleting chunk files")

            # Delete chunk files for this timepoint
            self._delete_timepoint_chunks(timepoint)

            # Remove from available timepoints list
            self.available_timepoints.remove(timepoint)

            # Update metadata
            self._update_timepoint_metadata()

    def clear_timepoint(self, timepoint: int):
        """
        Clear all data from a specific timepoint by deleting its chunk files.
        
        Args:
            timepoint: The timepoint index to clear
            
        Raises:
            ValueError: If timepoint doesn't exist
        """
        with self.zarr_lock:
            if timepoint not in self.available_timepoints:
                raise ValueError(f"Timepoint {timepoint} does not exist")

            logger.info(f"Clearing data from timepoint {timepoint} by deleting chunk files")

            # Delete chunk files for this timepoint
            self._delete_timepoint_chunks(timepoint)

    def _delete_timepoint_chunks(self, timepoint: int):
        """
        Delete all chunk files for a specific timepoint across all scales.
        This is much more efficient than zeroing out data.
        
        Args:
            timepoint: The timepoint index to delete chunks for
        """
        try:
            # For each scale, find and delete chunk files containing this timepoint
            for scale in range(self.num_scales):
                scale_path = self.zarr_path / str(scale)
                if not scale_path.exists():
                    continue

                # Zarr stores chunks in directories, timepoint is the first dimension
                # Chunk filename format: "t.c.z.y.x" where t is timepoint
                deleted_count = 0

                try:
                    # Look for chunk files that start with this timepoint
                    for chunk_file in scale_path.iterdir():
                        if chunk_file.is_file() and chunk_file.name.startswith(f"{timepoint}."):
                            try:
                                chunk_file.unlink()  # Delete the file
                                deleted_count += 1
                            except OSError as e:
                                logger.warning(f"Could not delete chunk file {chunk_file}: {e}")

                except OSError as e:
                    logger.warning(f"Could not access scale directory {scale_path}: {e}")

                if deleted_count > 0:
                    logger.debug(f"Deleted {deleted_count} chunk files for timepoint {timepoint} at scale {scale}")

        except Exception as e:
            logger.error(f"Error deleting timepoint chunks: {e}")

    def _ensure_timepoint_exists_in_zarr(self, timepoint: int):
        """
        Ensure that the zarr arrays are large enough to accommodate the given timepoint.
        This is called lazily only when actually writing data.
        Expands arrays in chunks to minimize expensive resize operations.
        
        Args:
            timepoint: The timepoint index that needs to exist in zarr
        """
        # Check if we need to expand any zarr arrays
        for scale in range(self.num_scales):
            if scale in self.zarr_arrays:
                zarr_array = self.zarr_arrays[scale]
                current_shape = zarr_array.shape

                # If the timepoint is beyond current array size, resize in chunks
                if timepoint >= current_shape[0]:
                    # Calculate new size with expansion chunk strategy
                    # Round up to the next chunk boundary to minimize future resizes
                    required_size = timepoint + 1
                    chunks_needed = (required_size + self.timepoint_expansion_chunk - 1) // self.timepoint_expansion_chunk
                    new_timepoint_count = chunks_needed * self.timepoint_expansion_chunk

                    new_shape = list(current_shape)
                    new_shape[0] = new_timepoint_count

                    # Resize the array with chunk-based expansion
                    logger.info(f"Expanding zarr scale {scale} from {current_shape[0]} to {new_timepoint_count} timepoints "
                               f"(required: {required_size}, chunk_size: {self.timepoint_expansion_chunk})")
                    start_time = time.time()
                    zarr_array.resize(new_shape)
                    elapsed_time = time.time() - start_time
                    logger.info(f"Zarr scale {scale} resize completed in {elapsed_time:.2f} seconds")

    def _update_timepoint_metadata(self):
        """Update the OME-Zarr metadata to reflect current timepoints."""
        if hasattr(self, 'zarr_root'):
            root = self.zarr_root
            if 'omero' in root.attrs:
                if self.available_timepoints:
                    root.attrs['omero']['rdefs']['defaultT'] = min(self.available_timepoints)

            # Update custom metadata
            if 'squid_canvas' in root.attrs:
                root.attrs['squid_canvas']['available_timepoints'] = sorted(self.available_timepoints)
                root.attrs['squid_canvas']['num_timepoints'] = len(self.available_timepoints)

    def _update_channel_activation(self, channel_idx: int, active: bool = True):
        """
        Update the activation status of a channel in the OME-Zarr metadata.
        
        Args:
            channel_idx: Local zarr channel index (0, 1, 2, etc.)
            active: Whether the channel should be marked as active
        """
        if not hasattr(self, 'zarr_root'):
            return

        try:
            root = self.zarr_root
            if 'omero' in root.attrs and 'channels' in root.attrs['omero']:
                channels = root.attrs['omero']['channels']
                if 0 <= channel_idx < len(channels):
                    # Update the channel activation status
                    channels[channel_idx]['active'] = active

                    # CRITICAL: Save the updated metadata back to the zarr file
                    # This is essential because zarr attributes are not automatically persisted
                    root.attrs['omero']['channels'] = channels

                    # Force sync to ensure attributes are written to disk
                    if hasattr(root.store, 'flush'):
                        root.store.flush()
                        logger.debug(f"Flushed zarr store after updating channel {channel_idx}")

                    logger.debug(f"Updated channel {channel_idx} activation status to {active} and saved to zarr")
                else:
                    logger.warning(f"Channel index {channel_idx} out of bounds for metadata update")
        except Exception as e:
            logger.warning(f"Failed to update channel activation status: {e}")

    def _ensure_channel_activated(self, channel_idx: int):
        """
        Simple channel activation: check if channel is already active, if not, activate it.
        
        Args:
            channel_idx: Local zarr channel index (0, 1, 2, etc.)
        """
        if not hasattr(self, 'zarr_root'):
            return

        try:
            root = self.zarr_root
            if 'omero' in root.attrs and 'channels' in root.attrs['omero']:
                channels = root.attrs['omero']['channels']
                if 0 <= channel_idx < len(channels):
                    # Check if channel is already active
                    if not channels[channel_idx]['active']:
                        # Channel is inactive, activate it
                        self._update_channel_activation(channel_idx, active=True)
                        logger.info(f"Activated channel {channel_idx} (was inactive)")
                    else:
                        logger.debug(f"Channel {channel_idx} already active")
                else:
                    logger.warning(f"Channel index {channel_idx} out of bounds for activation check")
        except Exception as e:
            logger.warning(f"Failed to check/ensure channel activation: {e}")
            # Fallback: try to activate anyway
            self._update_channel_activation(channel_idx, active=True)

    def activate_channels_with_data(self):
        """
        Simple post-stitching method: check highest available scale and activate channels that have data.
        This directly reads and writes the .zattrs file to bypass zarr caching issues.
        """
        if not hasattr(self, 'zarr_path'):
            logger.warning("Cannot activate channels: zarr_path not available")
            return

        try:
            import json

            logger.info("Checking for channels with data and activating them...")

            # Read .zattrs file directly
            zattrs_path = self.zarr_path / '.zattrs'
            if not zattrs_path.exists():
                logger.warning(f"No .zattrs file found at {zattrs_path}")
                return

            # Load current attributes
            with open(zattrs_path) as f:
                attrs = json.load(f)

            # Find the highest scale that exists (prefer highest scales for memory efficiency)
            scales_to_check = [5, 4, 3, 2, 1, 0]
            scale_used = None

            for scale in scales_to_check:
                scale_path = self.zarr_path / str(scale)
                if scale_path.exists():
                    scale_used = scale
                    logger.info(f"Using scale {scale} for channel activation check")
                    break

            if scale_used is None:
                logger.warning("No scale directories found for channel activation")
                return

            # Simple approach: list all chunk files for the highest scale and extract channel indices
            channels_with_data = set()
            scale_path = self.zarr_path / str(scale_used)

            logger.info(f"Scanning chunk files in {scale_path} to find channels with data...")

            # Look for all chunk files in the scale directory
            for chunk_file in scale_path.glob('*'):
                if chunk_file.is_file() and '.' in chunk_file.name:
                    # Parse chunk coordinates from filename: t.c.z.y.x
                    parts = chunk_file.name.split('.')
                    if len(parts) >= 5:  # t.c.z.y.x format
                        try:
                            chunk_channel = int(parts[1])  # c dimension
                            channels_with_data.add(chunk_channel)
                            logger.debug(f"Found chunk for channel {chunk_channel}: {chunk_file.name}")
                        except (ValueError, IndexError):
                            continue

            # Convert set to sorted list
            channels_with_data = sorted(list(channels_with_data))
            logger.info(f"Found data in channels: {channels_with_data}")

            # Update the attributes directly
            if 'omero' in attrs and 'channels' in attrs['omero']:
                channels = attrs['omero']['channels']

                # Activate channels that have data
                for channel_idx in channels_with_data:
                    if 0 <= channel_idx < len(channels):
                        channels[channel_idx]['active'] = True
                        logger.info(f"Activated channel {channel_idx}")

                # Write back the updated attributes
                with open(zattrs_path, 'w') as f:
                    json.dump(attrs, f, indent=4)

                logger.info(f"Successfully updated .zattrs file with {len(channels_with_data)} active channels")

                # Log final result
                active_channels = [i for i, ch in enumerate(channels) if ch['active']]
                logger.info(f"Final active channels: {active_channels}")
            else:
                logger.warning("No omero.channels found in .zattrs file")

        except Exception as e:
            logger.error(f"Error in activate_channels_with_data: {e}")
            import traceback
            traceback.print_exc()

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
            self.zarr_root = root  # Store reference for metadata updates

            # Import ChannelMapper for better metadata
            from squid_control.control.config import ChannelMapper

            # Create enhanced channel metadata with proper colors and info
            # Initially all channels are inactive until data is written
            omero_channels = []
            for ch in self.channels:
                try:
                    channel_info = ChannelMapper.get_channel_by_human_name(ch)
                    # Get color from centralized channel mapping
                    color = channel_info.color

                    omero_channels.append({
                        "label": ch,
                        "color": color,
                        "active": False,  # Start as inactive until data is written
                        "window": {"start": 0, "end": 255},
                        "family": "linear",
                        "coefficient": 1.0
                    })
                except ValueError:
                    # Fallback for unknown channels
                    omero_channels.append({
                        "label": ch,
                        "color": "FFFFFF",
                        "active": False,  # Start as inactive until data is written
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
                # Pre-allocate initial timepoints to avoid frequent resizing
                # Use no compression for direct access and fastest performance
                array = root.create_dataset(
                    str(scale),
                    shape=(self.initial_timepoints, len(self.channels), 1, height, width),
                    chunks=(1, 1, 1, self.chunk_size, self.chunk_size),
                    dtype='uint8',
                    fill_value=0,
                    overwrite=True,
                    compressor=None  # No compression for raw data access
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
        # Debug logging for coordinate conversion (only in debug mode)
        if logger.level <= 10:  # DEBUG level
            logger.debug(f"COORD_CONVERSION: Input coordinates ({x_mm:.2f}, {y_mm:.2f}) mm, scale {scale}")
            logger.debug(f"COORD_CONVERSION: Stage limits: {self.stage_limits}")
            logger.debug(f"COORD_CONVERSION: Canvas size: {self.canvas_width_px}x{self.canvas_height_px} px")
            logger.debug(f"COORD_CONVERSION: Pixel size: {self.pixel_size_xy_um} um")

        # Offset to make all coordinates positive
        x_offset_mm = -self.stage_limits['x_negative']
        y_offset_mm = -self.stage_limits['y_negative']

        # Convert to pixels at scale 0 (without padding)
        x_px_no_padding = (x_mm + x_offset_mm) * 1000 / self.pixel_size_xy_um
        y_px_no_padding = (y_mm + y_offset_mm) * 1000 / self.pixel_size_xy_um

        # Account for 10% padding by centering in the padded canvas
        # The canvas is 1.1x larger, so we need to add 5% margin on each side
        padding_factor = 1.1
        x_padding_px = (self.canvas_width_px - (self.stage_width_mm * 1000 / self.pixel_size_xy_um)) / 2
        y_padding_px = (self.canvas_height_px - (self.stage_height_mm * 1000 / self.pixel_size_xy_um)) / 2

        # Add padding offset to center the image in the padded canvas
        x_px = int(x_px_no_padding + x_padding_px)
        y_px = int(y_px_no_padding + y_padding_px)

        # Apply scale factor
        scale_factor = 4 ** scale
        x_px //= scale_factor
        y_px //= scale_factor

        if logger.level <= 10:  # DEBUG level
            logger.debug(f"COORD_CONVERSION: Final pixel coordinates: ({x_px}, {y_px}) for scale {scale}")

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

        # Crop to 97% of original size to remove black borders
        crop_factor = 0.97
        image_size = min(int(height * crop_factor), int(width * crop_factor))

        # Calculate crop bounds (center crop)
        y_start = (height - image_size) // 2
        y_end = y_start + image_size
        x_start = (width - image_size) // 2
        x_end = x_start + image_size

        cropped = rotated[y_start:y_end, x_start:x_end]

        logger.debug(f"Rotated image by {self.rotation_angle_deg}° and cropped from {width}x{height} to {image_size}x{image_size}")

        return cropped

    def _preprocess_image_for_stitching(self, image: np.ndarray, x_mm: float, y_mm: float,
                                        channel_idx: int = 0, z_idx: int = 0, timepoint: int = 0,
                                        is_quick_scan: bool = False):
        """
        CPU-intensive preprocessing: rotation, resizing, dtype conversion.
        This runs in parallel in the thread pool without holding zarr_lock.
        
        Args:
            image: Image array (2D)
            x_mm: X position in millimeters
            y_mm: Y position in millimeters
            channel_idx: Local zarr channel index (0, 1, 2, etc.)
            z_idx: Z-slice index (default 0)
            timepoint: Timepoint index (default 0)
            is_quick_scan: If True, only process scales 1-5 (skip scale 0)
            
        Returns:
            Dict with preprocessed data ready for zarr writing, or None if validation fails
        """
        # Validate channel index (no lock needed for read-only check)
        if channel_idx >= len(self.channels):
            logger.error(f"Channel index {channel_idx} out of bounds. Available channels: {len(self.channels)} (indices 0-{len(self.channels)-1})")
            return None

        if channel_idx < 0:
            logger.error(f"Channel index {channel_idx} cannot be negative")
            return None

        # Apply rotation and cropping (CPU work)
        processed_image = self._rotate_and_crop_image(image)

        # Prepare preprocessed data for all scales
        preprocessed_scales = []

        # Determine scale range based on scan mode
        start_scale = 1 if is_quick_scan else 0
        end_scale = min(self.num_scales, 6) if is_quick_scan else self.num_scales

        for scale in range(start_scale, end_scale):
            scale_factor = 4 ** scale

            # Get pixel coordinates for this scale
            x_px, y_px = self.stage_to_pixel_coords(x_mm, y_mm, scale)

            # Resize image if needed (CPU work)
            if is_quick_scan and scale == 1:
                # For quick scan, input is already at scale1 resolution
                scaled_image = processed_image
            elif scale > 0:
                if is_quick_scan:
                    # Scale relative to scale1 for quick scan
                    relative_scale_factor = 4 ** (scale - 1)
                    new_size = (processed_image.shape[1] // relative_scale_factor,
                               processed_image.shape[0] // relative_scale_factor)
                else:
                    # Normal scaling from original
                    new_size = (processed_image.shape[1] // scale_factor,
                               processed_image.shape[0] // scale_factor)
                scaled_image = cv2.resize(processed_image, new_size, interpolation=cv2.INTER_AREA)
            else:
                scaled_image = processed_image

            # Store preprocessed data for this scale
            preprocessed_scales.append({
                'scale': scale,
                'scaled_image': scaled_image,
                'x_px': x_px,
                'y_px': y_px,
                'x_mm': x_mm,
                'y_mm': y_mm
            })

        return {
            'preprocessed_scales': preprocessed_scales,
            'channel_idx': channel_idx,
            'z_idx': z_idx,
            'timepoint': timepoint,
            'is_quick_scan': is_quick_scan
        }

    def _write_preprocessed_to_zarr(self, preprocessed_data):
        """
        Write preprocessed image data to zarr arrays.
        This is I/O work that must be serialized. Runs in single writer thread.
        
        Args:
            preprocessed_data: Dict returned from _preprocess_image_for_stitching()
        """
        if preprocessed_data is None:
            return

        channel_idx = preprocessed_data['channel_idx']
        z_idx = preprocessed_data['z_idx']
        timepoint = preprocessed_data['timepoint']
        is_quick_scan = preprocessed_data['is_quick_scan']

        # Ensure timepoint exists in our tracking list
        if timepoint not in self.available_timepoints:
            with self.zarr_lock:
                if timepoint not in self.available_timepoints:
                    self.available_timepoints.append(timepoint)
                    self.available_timepoints.sort()
                    self._update_timepoint_metadata()

        with self.zarr_lock:
            # Ensure zarr arrays are sized correctly for this timepoint (lazy expansion)
            self._ensure_timepoint_exists_in_zarr(timepoint)

            # Write each preprocessed scale to zarr
            for scale_data in preprocessed_data['preprocessed_scales']:
                scale = scale_data['scale']
                scaled_image = scale_data['scaled_image']
                x_px = scale_data['x_px']
                y_px = scale_data['y_px']

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

                # CRITICAL: Always validate bounds before writing to zarr arrays
                if y_end > y_start and x_end > x_start and img_y_end > img_y_start and img_x_end > img_x_start:
                    # Additional validation to ensure image slice is within bounds
                    img_y_end = min(img_y_end, scaled_image.shape[0])
                    img_x_end = min(img_x_end, scaled_image.shape[1])

                    # Final check that we still have valid bounds after clamping
                    if img_y_end > img_y_start and img_x_end > img_x_start:
                        try:
                            # Ensure image is uint8 before writing to zarr
                            image_to_write = scaled_image[img_y_start:img_y_end, img_x_start:img_x_end]

                            if image_to_write.dtype != np.uint8:
                                # Convert to uint8 if needed
                                if image_to_write.dtype == np.uint16:
                                    image_to_write = (image_to_write / 256).astype(np.uint8)
                                elif image_to_write.dtype in [np.float32, np.float64]:
                                    # Normalize float data to 0-255
                                    if image_to_write.max() > image_to_write.min():
                                        image_to_write = ((image_to_write - image_to_write.min()) /
                                                        (image_to_write.max() - image_to_write.min()) * 255).astype(np.uint8)
                                    else:
                                        image_to_write = np.zeros_like(image_to_write, dtype=np.uint8)
                                else:
                                    image_to_write = image_to_write.astype(np.uint8)

                            # Double-check the final data type
                            if image_to_write.dtype != np.uint8:
                                image_to_write = image_to_write.astype(np.uint8)

                            zarr_array[timepoint, channel_idx, z_idx, y_start:y_end, x_start:x_end] = image_to_write

                        except IndexError as e:
                            logger.error(f"ZARR_WRITE: IndexError writing to zarr array at scale {scale}, channel {channel_idx}, timepoint {timepoint}: {e}")
                            logger.error(f"ZARR_WRITE: Zarr array shape: {zarr_array.shape}, trying to access timepoint {timepoint}")
                        except Exception as e:
                            logger.error(f"ZARR_WRITE: Error writing to zarr array at scale {scale}, channel {channel_idx}, timepoint {timepoint}: {e}")

    def add_image_sync(self, image: np.ndarray, x_mm: float, y_mm: float,
                       channel_idx: int = 0, z_idx: int = 0, timepoint: int = 0):
        """
        Synchronously add an image to the canvas at the specified position and timepoint.
        Updates all pyramid levels. Now uses 2-step preprocessing + writing for consistency.
        
        Args:
            image: Image array (2D)
            x_mm: X position in millimeters
            y_mm: Y position in millimeters
            channel_idx: Local zarr channel index (0, 1, 2, etc.)
            z_idx: Z-slice index (default 0)
            timepoint: Timepoint index (default 0)
        """
        # Step 1: Preprocess (CPU work)
        preprocessed = self._preprocess_image_for_stitching(
            image, x_mm, y_mm, channel_idx, z_idx, timepoint, is_quick_scan=False
        )

        # Step 2: Write to zarr (I/O work)
        if preprocessed is not None:
            self._write_preprocessed_to_zarr(preprocessed)

    def add_image_sync_quick(self, image: np.ndarray, x_mm: float, y_mm: float,
                           channel_idx: int = 0, z_idx: int = 0, timepoint: int = 0):
        """
        Synchronously add an image to the canvas for quick scan mode.
        Only updates scales 1-5 (skips scale 0 for performance).
        The input image should already be at scale1 resolution.
        Now uses 2-step preprocessing + writing for consistency.
        
        Args:
            image: Image array (2D) - should be at scale1 resolution (1/4 of original)
            x_mm: X position in millimeters
            y_mm: Y position in millimeters
            channel_idx: Local zarr channel index (0, 1, 2, etc.)
            z_idx: Z-slice index (default 0)
            timepoint: Timepoint index (default 0)
        """
        # Step 1: Preprocess (CPU work)
        preprocessed = self._preprocess_image_for_stitching(
            image, x_mm, y_mm, channel_idx, z_idx, timepoint, is_quick_scan=True
        )

        # Step 2: Write to zarr (I/O work)
        if preprocessed is not None:
            self._write_preprocessed_to_zarr(preprocessed)

    async def add_image_async(self, image: np.ndarray, x_mm: float, y_mm: float,
                              channel_idx: int = 0, z_idx: int = 0, timepoint: int = 0,
                              quick_scan: bool = False):
        """
        Add image to the preprocessing queue for asynchronous processing.
        
        Args:
            image: Image array (2D)
            x_mm: X position in millimeters
            y_mm: Y position in millimeters
            channel_idx: Local zarr channel index
            z_idx: Z-slice index
            timepoint: Timepoint index
            quick_scan: If True, use quick scan mode (scales 1-5 only)
        """
        await self.preprocessing_queue.put({
            'image': image.copy(),
            'x_mm': x_mm,
            'y_mm': y_mm,
            'channel_idx': channel_idx,
            'z_idx': z_idx,
            'timepoint': timepoint,
            'quick_scan': quick_scan,
            'timestamp': time.time()
        })

    async def _preprocess_and_queue_write(self, frame_data):
        """
        Preprocess image in thread pool (parallel), then queue for writing (serialized).
        
        Args:
            frame_data: Dict with image, coordinates, and metadata
        """
        loop = asyncio.get_event_loop()

        # CPU work in thread pool (runs in parallel with other preprocessing tasks)
        preprocessed = await loop.run_in_executor(
            self.executor,
            self._preprocess_image_for_stitching,
            frame_data['image'],
            frame_data['x_mm'],
            frame_data['y_mm'],
            frame_data['channel_idx'],
            frame_data['z_idx'],
            frame_data['timepoint'],
            frame_data.get('quick_scan', False)
        )

        # Queue for serialized writing (only if preprocessing succeeded)
        if preprocessed is not None:
            await self.zarr_write_queue.put(preprocessed)

    async def _zarr_writer_loop(self):
        """
        Dedicated writer task that serializes zarr writes.
        Runs in single task to prevent file write conflicts.
        """
        while self.is_writing:
            try:
                # Get preprocessed data from write queue
                preprocessed_data = await asyncio.wait_for(
                    self.zarr_write_queue.get(),
                    timeout=1.0
                )

                # Write to zarr (I/O work, single-threaded)
                loop = asyncio.get_event_loop()
                await loop.run_in_executor(
                    None,  # Use default executor for I/O
                    self._write_preprocessed_to_zarr,
                    preprocessed_data
                )

            except asyncio.TimeoutError:
                continue  # No data to write, keep waiting
            except Exception as e:
                logger.error(f"Error in zarr writer loop: {e}")
                import traceback
                traceback.print_exc()

        # Process any remaining writes in queue before exiting
        logger.info("Draining remaining writes from zarr write queue...")
        remaining_count = 0
        while not self.zarr_write_queue.empty():
            try:
                preprocessed_data = await asyncio.wait_for(
                    self.zarr_write_queue.get(),
                    timeout=0.1
                )
                loop = asyncio.get_event_loop()
                await loop.run_in_executor(
                    None,
                    self._write_preprocessed_to_zarr,
                    preprocessed_data
                )
                remaining_count += 1
            except asyncio.TimeoutError:
                break
            except Exception as e:
                logger.error(f"Error writing remaining data: {e}")

        if remaining_count > 0:
            logger.info(f"Zarr writer loop processed {remaining_count} remaining writes")

    async def start_stitching(self):
        """Start the background stitching tasks (preprocessing + writing)."""
        if not self.is_stitching:
            self.is_stitching = True
            self.is_writing = True

            # Start preprocessing loop
            self.stitching_task = asyncio.create_task(self._stitching_loop())
            # Start writer loop
            self.writer_task = asyncio.create_task(self._zarr_writer_loop())

            logger.info("Started background stitching (preprocessing + writer tasks)")

    async def stop_stitching(self):
        """
        Stop the background stitching tasks with proper cleanup order.
        Order: Stop preprocessing → Drain preprocessing queue → Stop writer → Drain write queue
        """
        logger.info("Stopping stitching pipeline...")

        # Step 1: Stop preprocessing loop (no new images will be preprocessed)
        self.is_stitching = False
        logger.info("Stopped preprocessing loop, draining preprocessing queue...")

        # Step 2: Wait for preprocessing loop to finish
        if self.stitching_task:
            await self.stitching_task
            logger.info("Preprocessing loop completed")

        # Step 3: Process any remaining images in preprocessing queue
        preprocess_count = 0
        while not self.preprocessing_queue.empty():
            try:
                frame_data = await asyncio.wait_for(
                    self.preprocessing_queue.get(),
                    timeout=0.1
                )
                # Preprocess and queue for writing
                await self._preprocess_and_queue_write(frame_data)
                preprocess_count += 1
            except asyncio.TimeoutError:
                break
            except Exception as e:
                logger.error(f"Error processing remaining frame in preprocessing queue: {e}")

        if preprocess_count > 0:
            logger.info(f"Processed {preprocess_count} remaining frames from preprocessing queue")

        # Step 4: Wait for write queue to drain (all preprocessed data gets written)
        logger.info(f"Waiting for write queue to drain ({self.zarr_write_queue.qsize()} items remaining)...")
        while not self.zarr_write_queue.empty():
            await asyncio.sleep(0.1)  # Give writer time to process
            # Safety timeout: if queue isn't draining, break after reasonable time
            if self.zarr_write_queue.qsize() > 0:
                await asyncio.sleep(0.1)

        # Step 5: Stop writer loop (no more writes will be processed)
        self.is_writing = False
        logger.info("Stopping writer loop...")

        # Step 6: Wait for writer task to complete
        if self.writer_task:
            await self.writer_task
            logger.info("Writer loop completed")

        # Step 7: CRITICAL - Wait for all thread pool operations to complete
        logger.info("Waiting for all zarr operations to complete...")
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, self._wait_for_zarr_operations_complete)

        logger.info("Stitching pipeline stopped successfully")

    async def _stitching_loop(self):
        """
        Background preprocessing loop that spawns multiple concurrent preprocessing tasks.
        This enables parallel CPU work across multiple threads.
        """
        active_tasks = set()
        max_concurrent = 4  # Match thread pool size for optimal parallelization

        logger.info(f"Preprocessing loop started (max {max_concurrent} concurrent tasks)")

        while self.is_stitching or not self.preprocessing_queue.empty():
            try:
                # Launch new preprocessing tasks up to max_concurrent limit
                while len(active_tasks) < max_concurrent and not self.preprocessing_queue.empty():
                    try:
                        # Get frame from preprocessing queue (non-blocking)
                        frame_data = self.preprocessing_queue.get_nowait()

                        # Launch preprocessing task (runs in parallel)
                        task = asyncio.create_task(self._preprocess_and_queue_write(frame_data))
                        active_tasks.add(task)

                        # Remove task from set when done
                        task.add_done_callback(active_tasks.discard)

                        logger.debug(f"Launched preprocessing task ({len(active_tasks)} active)")

                    except asyncio.QueueEmpty:
                        break  # No more items in queue
                    except Exception as e:
                        logger.error(f"Error launching preprocessing task: {e}")

                # Small delay to prevent busy loop
                await asyncio.sleep(0.01)

                # If no active tasks and queue is empty, wait a bit longer
                if len(active_tasks) == 0 and self.preprocessing_queue.empty():
                    await asyncio.sleep(0.1)

            except Exception as e:
                logger.error(f"Error in preprocessing loop: {e}")
                import traceback
                traceback.print_exc()

        # Wait for all active preprocessing tasks to complete
        if active_tasks:
            logger.info(f"Waiting for {len(active_tasks)} active preprocessing tasks to complete...")
            await asyncio.gather(*active_tasks, return_exceptions=True)
            logger.info("All preprocessing tasks completed")

        logger.info("Preprocessing loop exited")

    def get_canvas_region_pixels(self, center_x_px: int, center_y_px: int, width_px: int, height_px: int,
                                 scale: int = 0, channel_idx: int = 0, timepoint: int = 0) -> Tuple[np.ndarray, Dict]:
        """
        Get a region from the canvas using pixel coordinates directly.
        Returns both the region array and exact pixel bounds used for reading.
        No preprocessing/transformation is applied.
        
        Args:
            center_x_px: Center X position in pixels (at the specified scale)
            center_y_px: Center Y position in pixels (at the specified scale)
            width_px: Width in pixels (at the specified scale)
            height_px: Height in pixels (at the specified scale)
            scale: Scale level to retrieve from
            channel_idx: Local zarr channel index (0, 1, 2, etc.)
            timepoint: Timepoint index (default 0)
            
        Returns:
            Tuple of (region_array, bounds_dict) where bounds_dict contains:
                - 'x_start': Exact pixel x start coordinate
                - 'x_end': Exact pixel x end coordinate
                - 'y_start': Exact pixel y start coordinate
                'y_end': Exact pixel y end coordinate
            Returns (None, None) if read fails
        """
        # Validate channel index
        if channel_idx >= len(self.channels) or channel_idx < 0:
            logger.error(f"Channel index {channel_idx} out of bounds. Available channels: {len(self.channels)} (indices 0-{len(self.channels)-1})")
            return None, None

        # Validate timepoint
        if timepoint not in self.available_timepoints:
            logger.error(f"Timepoint {timepoint} not available. Available timepoints: {sorted(self.available_timepoints)}")
            return None, None

        with self.zarr_lock:
            # Validate zarr arrays exist
            if not hasattr(self, 'zarr_arrays') or scale not in self.zarr_arrays:
                logger.error(f"Zarr arrays not initialized or scale {scale} not available")
                return None, None

            zarr_array = self.zarr_arrays[scale]

            # Check if timepoint exists in zarr array
            if timepoint >= zarr_array.shape[0]:
                logger.warning(f"Timepoint {timepoint} not yet written to zarr array (shape: {zarr_array.shape})")
                # Return zeros of the expected size
                return np.zeros((height_px, width_px), dtype=zarr_array.dtype), {
                    'x_start': max(0, center_x_px - width_px // 2),
                    'x_end': max(0, center_x_px - width_px // 2) + width_px,
                    'y_start': max(0, center_y_px - height_px // 2),
                    'y_end': max(0, center_y_px - height_px // 2) + height_px
                }

            # Double-check zarr array dimensions
            if channel_idx >= zarr_array.shape[1]:
                logger.error(f"Channel index {channel_idx} exceeds zarr array channel dimension {zarr_array.shape[1]}")
                return None, None

            # Calculate exact bounds
            x_start = max(0, center_x_px - width_px // 2)
            x_end = min(zarr_array.shape[4], x_start + width_px)
            y_start = max(0, center_y_px - height_px // 2)
            y_end = min(zarr_array.shape[3], y_start + height_px)

            # Read from zarr
            try:
                region = zarr_array[timepoint, channel_idx, 0, y_start:y_end, x_start:x_end]
                bounds = {
                    'x_start': x_start,
                    'x_end': x_end,
                    'y_start': y_start,
                    'y_end': y_end
                }
                logger.debug(f"Successfully retrieved region from zarr at scale {scale}, channel {channel_idx}, timepoint {timepoint} "
                           f"with bounds x=[{x_start}:{x_end}], y=[{y_start}:{y_end}]")
                return region, bounds
            except IndexError as e:
                logger.error(f"IndexError reading from zarr array at scale {scale}, channel {channel_idx}, timepoint {timepoint}: {e}")
                logger.error(f"Zarr array shape: {zarr_array.shape}, trying to access timepoint {timepoint}")
                return None, None
            except Exception as e:
                logger.error(f"Error reading from zarr array at scale {scale}, channel {channel_idx}, timepoint {timepoint}: {e}")
                return None, None

    def add_image_sync_pixels(self, image: np.ndarray, x_start_px: int, y_start_px: int,
                              channel_idx: int = 0, z_idx: int = 0, timepoint: int = 0,
                              scale: int = 0):
        """
        Write image directly to zarr using pixel coordinates without any preprocessing.
        Used for segmentation masks that must align exactly with source data.
        No rotation, cropping, or resizing is applied.
        Writes to all pyramid scales for consistency with regular stitching.
        
        Args:
            image: Image array (2D, uint8) - must match exact dimensions for the region at scale 0
            x_start_px: X start pixel coordinate (at scale 0)
            y_start_px: Y start pixel coordinate (at scale 0)
            channel_idx: Local zarr channel index (0, 1, 2, etc.)
            z_idx: Z-slice index (default 0)
            timepoint: Timepoint index (default 0)
            scale: Scale level for the input coordinates (default 0) - used as reference scale
        """
        # Validate channel index
        if channel_idx >= len(self.channels) or channel_idx < 0:
            logger.error(f"ZARR_WRITE_PIXELS: Channel index {channel_idx} out of bounds. Available channels: {len(self.channels)} (indices 0-{len(self.channels)-1})")
            return

        # Validate image input
        if image is None or image.size == 0:
            logger.error("ZARR_WRITE_PIXELS: Invalid image input - image is None or empty")
            return

        logger.info(f"ZARR_WRITE_PIXELS: Writing image shape={image.shape}, dtype={image.dtype} to all scales, "
                   f"channel={channel_idx}, timepoint={timepoint}, pixel bounds at scale {scale} x=[{x_start_px}:?], y=[{y_start_px}:?]")

        # Validate timepoint
        if timepoint not in self.available_timepoints:
            with self.zarr_lock:
                if timepoint not in self.available_timepoints:
                    self.available_timepoints.append(timepoint)
                    self.available_timepoints.sort()
                    self._update_timepoint_metadata()
                    logger.info(f"ZARR_WRITE_PIXELS: Created new timepoint {timepoint}")

        with self.zarr_lock:
            # Ensure zarr arrays are sized correctly for this timepoint (lazy expansion)
            self._ensure_timepoint_exists_in_zarr(timepoint)

            # Ensure image is uint8 before processing
            image_base = image.copy()
            if image_base.dtype != np.uint8:
                logger.info(f"ZARR_WRITE_PIXELS: Converting image from {image_base.dtype} to uint8")
                if image_base.dtype == np.uint16:
                    image_base = (image_base / 256).astype(np.uint8)
                elif image_base.dtype in [np.float32, np.float64]:
                    if image_base.max() > image_base.min():
                        image_base = ((image_base - image_base.min()) /
                                    (image_base.max() - image_base.min()) * 255).astype(np.uint8)
                    else:
                        image_base = np.zeros_like(image_base, dtype=np.uint8)
                else:
                    image_base = image_base.astype(np.uint8)

            # Write to all pyramid scales
            scales_written = 0
            for target_scale in range(self.num_scales):
                try:
                    # Validate zarr arrays exist for this scale
                    if not hasattr(self, 'zarr_arrays') or target_scale not in self.zarr_arrays:
                        logger.error(f"ZARR_WRITE_PIXELS: Zarr arrays not initialized or scale {target_scale} not available")
                        continue

                    zarr_array = self.zarr_arrays[target_scale]

                    # Double-check zarr array dimensions
                    if channel_idx >= zarr_array.shape[1]:
                        logger.error(f"ZARR_WRITE_PIXELS: Channel index {channel_idx} exceeds zarr array channel dimension {zarr_array.shape[1]}")
                        continue

                    # Calculate scale factor from reference scale to target scale
                    # If input is at scale 0 and we're writing to scale 1, we need to downsample by 4
                    scale_factor_from_ref = 4 ** (target_scale - scale)

                    # Convert pixel coordinates for this scale
                    x_start_px_scaled = x_start_px // scale_factor_from_ref if scale_factor_from_ref > 1 else x_start_px
                    y_start_px_scaled = y_start_px // scale_factor_from_ref if scale_factor_from_ref > 1 else y_start_px

                    # Downsample image for this scale if needed
                    if target_scale == scale:
                        # Same scale: use original image
                        scaled_image = image_base
                    elif target_scale > scale:
                        # Downscale: downsample image
                        scale_factor_down = 4 ** (target_scale - scale)
                        new_width = image_base.shape[1] // scale_factor_down
                        new_height = image_base.shape[0] // scale_factor_down
                        scaled_image = cv2.resize(image_base, (new_width, new_height), interpolation=cv2.INTER_NEAREST)
                    else:
                        # Upscale: this shouldn't happen but handle it
                        scale_factor_up = 4 ** (scale - target_scale)
                        new_width = image_base.shape[1] * scale_factor_up
                        new_height = image_base.shape[0] * scale_factor_up
                        scaled_image = cv2.resize(image_base, (new_width, new_height), interpolation=cv2.INTER_NEAREST)

                    # Calculate exact bounds for this scale
                    height, width = scaled_image.shape[:2]
                    x_end_px_scaled = min(zarr_array.shape[4], x_start_px_scaled + width)
                    y_end_px_scaled = min(zarr_array.shape[3], y_start_px_scaled + height)

                    logger.info(f"ZARR_WRITE_PIXELS: Scale {target_scale} - zarr array shape={zarr_array.shape}, "
                               f"image size={width}x{height}, calculated bounds x=[{x_start_px_scaled}:{x_end_px_scaled}], "
                               f"y=[{y_start_px_scaled}:{y_end_px_scaled}]")

                    # Crop image if it extends beyond canvas bounds
                    img_width_actual = x_end_px_scaled - x_start_px_scaled
                    img_height_actual = y_end_px_scaled - y_start_px_scaled

                    # CRITICAL: Always validate bounds before writing to zarr arrays
                    if y_end_px_scaled > y_start_px_scaled and x_end_px_scaled > x_start_px_scaled and img_height_actual > 0 and img_width_actual > 0:
                        try:
                            # Crop image to actual bounds
                            image_to_write = scaled_image[:img_height_actual, :img_width_actual]

                            # Ensure image is uint8
                            if image_to_write.dtype != np.uint8:
                                image_to_write = image_to_write.astype(np.uint8)

                            # Write directly to zarr at exact pixel coordinates
                            logger.info(f"ZARR_WRITE_PIXELS: Writing to scale {target_scale} - zarr array[{timepoint}, {channel_idx}, {z_idx}, "
                                       f"{y_start_px_scaled}:{y_end_px_scaled}, {x_start_px_scaled}:{x_end_px_scaled}] "
                                       f"with image shape={image_to_write.shape}")
                            zarr_array[timepoint, channel_idx, z_idx, y_start_px_scaled:y_end_px_scaled, x_start_px_scaled:x_end_px_scaled] = image_to_write

                            scales_written += 1
                        except IndexError as e:
                            logger.error(f"ZARR_WRITE_PIXELS: ❌ IndexError writing to scale {target_scale}, channel {channel_idx}, timepoint {timepoint}: {e}")
                            logger.error(f"ZARR_WRITE_PIXELS: Zarr array shape: {zarr_array.shape}, trying to access "
                                       f"[{timepoint}, {channel_idx}, {z_idx}, {y_start_px_scaled}:{y_end_px_scaled}, {x_start_px_scaled}:{x_end_px_scaled}]")
                            import traceback
                            traceback.print_exc()
                        except Exception as e:
                            logger.error(f"ZARR_WRITE_PIXELS: ❌ Error writing to scale {target_scale}, channel {channel_idx}, timepoint {timepoint}: {e}")
                            import traceback
                            traceback.print_exc()
                    else:
                        logger.error(f"ZARR_WRITE_PIXELS: ❌ Invalid bounds for scale {target_scale} - cannot write: "
                                   f"x_end={x_end_px_scaled} <= x_start={x_start_px_scaled} or "
                                   f"y_end={y_end_px_scaled} <= y_start={y_start_px_scaled} or "
                                   f"img_width_actual={img_width_actual} <= 0 or "
                                   f"img_height_actual={img_height_actual} <= 0")
                        logger.error(f"ZARR_WRITE_PIXELS: Scale {target_scale} - image shape={scaled_image.shape}, "
                                   f"zarr array shape={zarr_array.shape}, "
                                   f"requested bounds x=[{x_start_px_scaled}:{x_start_px_scaled + width}], "
                                   f"y=[{y_start_px_scaled}:{y_start_px_scaled + height}]")

                except Exception as e:
                    logger.error(f"ZARR_WRITE_PIXELS: ❌ Error processing scale {target_scale}: {e}")
                    import traceback
                    traceback.print_exc()

            # Activate channel after successful writes (only once, not per scale)
            if scales_written > 0:
                self._ensure_channel_activated(channel_idx)
            else:
                logger.error(f"ZARR_WRITE_PIXELS: ❌ Failed to write to any scales for channel {channel_idx}, timepoint {timepoint}")

    def close(self):
        """Close the canvas and clean up resources."""
        if hasattr(self, 'zarr_array') and self.zarr_array is not None:
            self.zarr_array = None
        logger.info(f"Closed well canvas: {self.fileset_name}")

    def export_to_zip(self, zip_path):
        """
        Export the well canvas to a ZIP file.
        
        Args:
            zip_path (str): Path to the output ZIP file
        """

        try:
            # Check if the zarr path exists
            if not self.zarr_path.exists():
                logger.warning(f"Zarr path does not exist: {self.zarr_path}")
                return

            # Create the ZIP file directly from the existing zarr data
            with zipfile.ZipFile(zip_path, 'w', allowZip64=True, compression=zipfile.ZIP_STORED) as zf:
                # Walk through the zarr directory and add all files
                for root, dirs, files in os.walk(self.zarr_path):
                    for file in files:
                        file_path = os.path.join(root, file)
                        # Calculate relative path for the ZIP
                        relative_path = os.path.relpath(file_path, self.zarr_path.parent)
                        # Use forward slashes for ZIP paths and ensure it starts with "data.zarr/"
                        arcname = "data.zarr/" + relative_path.replace(os.sep, '/').split('/', 1)[-1]
                        zf.write(file_path, arcname)

            logger.info(f"Exported well canvas to ZIP: {zip_path}")

        except Exception as e:
            logger.error(f"Failed to export well canvas to ZIP: {e}")
            raise

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
                time.sleep(0.1)

                logger.info("All zarr arrays flushed and synchronized")

        except Exception as e:
            logger.error(f"Error during zarr flush and sync: {e}")
            raise RuntimeError(f"Failed to flush zarr arrays: {e}")

    def export_as_zip_file(self) -> str:
        """
        Export the entire zarr canvas as a zip file to a temporary file.
        Uses robust ZIP64 creation that's compatible with S3 ZIP parsers.
        Avoids memory corruption by writing directly to file.
        
        Returns:
            str: Path to the temporary ZIP file (caller must clean up)
        """
        import os
        import zipfile

        # Create temporary file for ZIP creation in ZARR_PATH to avoid memory issues
        # Ensure base_path exists
        self.base_path.mkdir(parents=True, exist_ok=True)
        temp_fd, temp_path = tempfile.mkstemp(suffix='.zip', prefix='zarr_export_', dir=str(self.base_path))

        try:
            # Close file descriptor immediately to avoid issues
            os.close(temp_fd)
            temp_fd = None  # Mark as closed

            # CRITICAL: Ensure all zarr operations are complete before ZIP export
            logger.info("Preparing zarr canvas for ZIP export...")
            self._flush_and_sync_zarr_arrays()

            # Force ZIP64 format explicitly for compatibility with S3 parser
            # Use minimal compression for reliability with many small files
            zip_kwargs = {
                'mode': 'w',
                'compression': zipfile.ZIP_STORED,  # No compression for reliability
                'allowZip64': True,
                'strict_timestamps': False  # Handle timestamp edge cases
            }

            # Create ZIP file with explicit ZIP64 support
            with zipfile.ZipFile(temp_path, **zip_kwargs) as zip_file:
                logger.info("Creating ZIP archive with explicit ZIP64 support...")

                # Build file list first to validate and count
                files_to_add = []
                total_size = 0

                for root, dirs, files in os.walk(self.zarr_path):
                    for file in files:
                        file_path = Path(root) / file

                        # Skip files that don't exist or can't be read
                        if not file_path.exists() or not file_path.is_file():
                            logger.warning(f"Skipping non-existent or non-file: {file_path}")
                            continue

                        try:
                            # Verify file is readable and get size
                            file_size = file_path.stat().st_size
                            total_size += file_size

                            # Create relative path for ZIP archive
                            relative_path = file_path.relative_to(self.zarr_path)
                            # Use forward slashes for ZIP compatibility (standard requirement)
                            arcname = "data.zarr/" + str(relative_path).replace(os.sep, '/')

                            files_to_add.append((file_path, arcname, file_size))

                        except OSError as e:
                            logger.warning(f"Skipping unreadable file {file_path}: {e}")
                            continue

                logger.info(f"Validated {len(files_to_add)} files for ZIP archive (total: {total_size / (1024*1024):.1f} MB)")

                # Check if we need ZIP64 format (more than 65535 files or 4GB total)
                needs_zip64 = len(files_to_add) >= 65535 or total_size >= (4 * 1024 * 1024 * 1024)
                if needs_zip64:
                    logger.info(f"ZIP64 format required: {len(files_to_add)} files, {total_size / (1024*1024):.1f} MB")

                # Add files to ZIP in sorted order for consistent central directory
                files_to_add.sort(key=lambda x: x[1])  # Sort by arcname

                processed_files = 0
                for file_path, arcname, file_size in files_to_add:
                    try:
                        # Add file with explicit error handling
                        zip_file.write(file_path, arcname=arcname)
                        processed_files += 1

                        # Progress logging every 1000 files
                        if processed_files % 1000 == 0:
                            logger.info(f"ZIP progress: {processed_files}/{len(files_to_add)} files processed")

                    except Exception as e:
                        logger.error(f"Failed to add file to ZIP: {file_path} -> {arcname}: {e}")
                        continue

                # Add metadata with proper JSON formatting
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
                        "squid_canvas_version": "1.0",
                        "zip_format": "ZIP64" if needs_zip64 else "standard"
                    }
                }

                metadata_json = json.dumps(metadata, indent=2, ensure_ascii=False)
                zip_file.writestr("squid_canvas_metadata.json", metadata_json.encode('utf-8'))
                processed_files += 1

                logger.info(f"ZIP creation completed: {processed_files} files processed")

            # Get file size for validation
            zip_size_mb = os.path.getsize(temp_path) / (1024 * 1024)

            # Enhanced ZIP validation specifically for S3 compatibility
            with open(temp_path, 'rb') as f:
                zip_content_for_validation = f.read()
            self._validate_zip_structure_for_s3(zip_content_for_validation)

            logger.info(f"ZIP export successful: {zip_size_mb:.2f} MB, {processed_files} files")
            return temp_path

        except Exception as e:
            logger.error(f"Failed to export zarr canvas as zip: {e}")
            # Clean up temp file on error
            try:
                if 'temp_path' in locals() and os.path.exists(temp_path):
                    os.unlink(temp_path)
            except Exception:
                pass
            raise RuntimeError(f"Cannot export zarr canvas: {e}")
        finally:
            # Clean up file descriptor if still open
            if temp_fd is not None:
                try:
                    os.close(temp_fd)
                except Exception:
                    pass  # Ignore errors closing fd

    def _validate_zip_structure_for_s3(self, zip_content: bytes) -> None:
        """
        Validate ZIP file structure specifically for S3 ZIP parser compatibility.
        Checks for proper central directory structure and ZIP64 format compliance.
        
        Args:
            zip_content (bytes): The ZIP file content to validate
            
        Raises:
            RuntimeError: If ZIP file structure is incompatible with S3 parser
        """
        try:
            import io
            import zipfile

            # Basic ZIP file validation
            zip_buffer = io.BytesIO(zip_content)
            with zipfile.ZipFile(zip_buffer, 'r') as zip_file:
                file_list = zip_file.namelist()
                if not file_list:
                    raise RuntimeError("ZIP file is empty")

                zip_size_mb = len(zip_content) / (1024 * 1024)
                file_count = len(file_list)

                logger.info(f"Basic ZIP validation passed: {file_count} files, {zip_size_mb:.2f} MB")

                # Check for ZIP64 indicators (critical for S3 parser)
                is_zip64 = file_count >= 65535 or zip_size_mb >= 4000

                if is_zip64:
                    # For ZIP64 files, check that central directory can be found
                    # This mimics what the S3 ZIP parser does
                    logger.info("Validating ZIP64 central directory structure...")

                    # Look for ZIP64 signatures in the file
                    zip64_eocd_locator = b"PK\x06\x07"  # ZIP64 End of Central Directory Locator
                    zip64_eocd = b"PK\x06\x06"          # ZIP64 End of Central Directory
                    standard_eocd = b"PK\x05\x06"       # Standard End of Central Directory

                    # Check the last 128KB for these signatures (like S3 parser does)
                    tail_size = min(128 * 1024, len(zip_content))
                    tail_data = zip_content[-tail_size:]

                    has_zip64_locator = zip64_eocd_locator in tail_data
                    has_zip64_eocd = zip64_eocd in tail_data
                    has_standard_eocd = standard_eocd in tail_data

                    logger.info(f"ZIP64 structure check: locator={has_zip64_locator}, eocd={has_zip64_eocd}, standard_eocd={has_standard_eocd}")

                    # ZIP64 files should have proper directory structures
                    if not (has_zip64_locator and has_standard_eocd):
                        logger.warning("ZIP64 format validation issues detected")

                    # Verify we can read file info (this is what S3 parser tries to do)
                    test_files = min(10, len(file_list))
                    for i in range(test_files):
                        try:
                            info = zip_file.getinfo(file_list[i])
                            # Try to access file info that S3 parser needs
                            _ = info.filename
                            _ = info.file_size
                            _ = info.compress_size
                            _ = info.date_time
                        except Exception as e:
                            logger.warning(f"File info access issue for {file_list[i]}: {e}")

                # Test random file access (S3 parser does this)
                test_count = min(5, len(file_list))
                for i in range(0, len(file_list), max(1, len(file_list) // test_count)):
                    try:
                        with zip_file.open(file_list[i]) as f:
                            # Read just 1 byte to verify file can be opened
                            f.read(1)
                    except Exception as e:
                        logger.warning(f"File access test failed for {file_list[i]}: {e}")

                logger.info("S3-compatible ZIP validation completed successfully")

        except zipfile.BadZipFile as e:
            logger.error(f"Invalid ZIP file format: {e}")
            raise RuntimeError(f"Invalid ZIP file format: {e}")
        except Exception as e:
            logger.error(f"ZIP validation failed: {e}")
            raise RuntimeError(f"ZIP validation failed: {e}")

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

            # Get actual file size on disk by walking the zarr directory
            if self.zarr_path.exists():
                try:
                    for file_path in self.zarr_path.rglob('*'):
                        if file_path.is_file():
                            try:
                                size = file_path.stat().st_size
                                total_size_bytes += size
                                file_count += 1
                            except (OSError, PermissionError) as e:
                                logger.warning(f"Could not read size of {file_path}: {e}")
                except Exception as e:
                    logger.error(f"Error walking zarr directory {self.zarr_path}: {e}")
                    # Fallback: try to get directory size using os.path.getsize
                    try:
                        import os
                        total_size_bytes = sum(os.path.getsize(os.path.join(dirpath, filename))
                                             for dirpath, dirnames, filenames in os.walk(self.zarr_path)
                                             for filename in filenames)
                    except Exception as fallback_e:
                        logger.error(f"Fallback size calculation also failed: {fallback_e}")
                        total_size_bytes = 0
            else:
                logger.warning(f"Zarr path does not exist: {self.zarr_path}")

            # Check which arrays have actual data
            for scale in range(self.num_scales):
                if scale in self.zarr_arrays:
                    array = self.zarr_arrays[scale]

                    # Check if array has any data (non-zero values)
                    if array.size > 0:
                        try:
                            # Sample a small region to check for data
                            sample_size = min(100, array.shape[3], array.shape[4])
                            sample = array[0, 0, 0, :sample_size, :sample_size]
                            if sample.max() > 0:
                                data_arrays += 1
                        except Exception as e:
                            logger.warning(f"Could not sample array at scale {scale}: {e}")

            # For empty arrays, estimate zip size based on actual disk usage
            # Zarr metadata and empty arrays compress very well
            if data_arrays == 0:
                # Empty zarr structures are mostly metadata, compress to ~10% of disk size
                estimated_zip_size_mb = (total_size_bytes * 0.1) / (1024 * 1024)
            else:
                # Arrays with data compress moderately (20-40% depending on content)
                estimated_zip_size_mb = (total_size_bytes * 0.3) / (1024 * 1024)

            logger.info(f"Export info: {total_size_bytes / (1024*1024):.1f} MB on disk ({file_count} files), "
                       f"{data_arrays} arrays with data, estimated zip: {estimated_zip_size_mb:.1f} MB")

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

class WellZarrCanvas(WellZarrCanvasBase):
    """
    Well-specific zarr canvas for individual well imaging with well-center-relative coordinates.
    
    This class extends WellZarrCanvasBase to provide well-specific functionality:
    - Well-center-relative coordinate system (0,0 at well center)
    - Automatic well center calculation from well plate formats
    - Canvas size based on well diameter + configurable padding
    - Well-specific fileset naming (well_{row}{column}_{well_plate_type})
    """

    def __init__(self, well_row: str, well_column: int, well_plate_type: str = '96',
                 padding_mm: float = 1.0, base_path: str = None,
                 pixel_size_xy_um: float = 0.333, channels: List[str] = None, **kwargs):
        """
        Initialize well-specific canvas.
        
        Args:
            well_row: Well row (e.g., 'A', 'B')
            well_column: Well column (e.g., 1, 2, 3)
            well_plate_type: Well plate type ('6', '12', '24', '96', '384')
            padding_mm: Padding around well in mm (default 2.0)
            base_path: Base directory for zarr storage
            pixel_size_xy_um: Pixel size in micrometers
            channels: List of channel names
            **kwargs: Additional arguments passed to ZarrCanvas
        """
        # Import well plate format classes
        from squid_control.control.config import (
            CONFIG,
        )

        # Get well plate format
        self.wellplate_format = self._get_wellplate_format(well_plate_type)

        # Store well information
        self.well_row = well_row
        self.well_column = well_column
        self.well_plate_type = well_plate_type
        self.padding_mm = padding_mm

        # Calculate well center coordinates (absolute stage coordinates)
        if hasattr(CONFIG, 'WELLPLATE_OFFSET_X_MM') and hasattr(CONFIG, 'WELLPLATE_OFFSET_Y_MM'):
            # Use offsets if available (hardware mode)
            x_offset = CONFIG.WELLPLATE_OFFSET_X_MM
            y_offset = CONFIG.WELLPLATE_OFFSET_Y_MM
        else:
            # No offsets (simulation mode)
            x_offset = 0
            y_offset = 0

        self.well_center_x = (self.wellplate_format.A1_X_MM + x_offset +
                             (well_column - 1) * self.wellplate_format.WELL_SPACING_MM)
        self.well_center_y = (self.wellplate_format.A1_Y_MM + y_offset +
                             (ord(well_row) - ord('A')) * self.wellplate_format.WELL_SPACING_MM)

        # Calculate canvas size (well diameter + padding)
        canvas_size_mm = self.wellplate_format.WELL_SIZE_MM + (2 * padding_mm)

        # Define well-relative stage limits (centered around 0,0)
        stage_limits = {
            'x_positive': canvas_size_mm / 2,
            'x_negative': -canvas_size_mm / 2,
            'y_positive': canvas_size_mm / 2,
            'y_negative': -canvas_size_mm / 2,
            'z_positive': 6
        }

        # Create well-specific fileset name
        fileset_name = f"well_{well_row}{well_column}_{well_plate_type}"

        # Initialize parent ZarrCanvas with well-specific parameters
        super().__init__(
            base_path=base_path,
            pixel_size_xy_um=pixel_size_xy_um,
            stage_limits=stage_limits,
            channels=channels,
            fileset_name=fileset_name,
            **kwargs
        )

        logger.info(f"WellZarrCanvas initialized for well {well_row}{well_column} ({well_plate_type})")
        logger.info(f"Well center: ({self.well_center_x:.2f}, {self.well_center_y:.2f}) mm")
        logger.info(f"Canvas size: {canvas_size_mm:.2f} mm, padding: {padding_mm:.2f} mm")

    def _get_wellplate_format(self, well_plate_type: str):
        """Get well plate format configuration."""
        from squid_control.control.config import (
            WELLPLATE_FORMAT_6,
            WELLPLATE_FORMAT_12,
            WELLPLATE_FORMAT_24,
            WELLPLATE_FORMAT_96,
            WELLPLATE_FORMAT_384,
        )

        if well_plate_type == '6':
            return WELLPLATE_FORMAT_6
        elif well_plate_type == '12':
            return WELLPLATE_FORMAT_12
        elif well_plate_type == '24':
            return WELLPLATE_FORMAT_24
        elif well_plate_type == '96':
            return WELLPLATE_FORMAT_96
        elif well_plate_type == '384':
            return WELLPLATE_FORMAT_384
        else:
            return WELLPLATE_FORMAT_96  # Default

    def stage_to_pixel_coords(self, x_mm: float, y_mm: float, scale: int = 0) -> Tuple[int, int]:
        """
        Convert absolute stage coordinates to well-relative pixel coordinates.
        
        Args:
            x_mm: Absolute X position in mm
            y_mm: Absolute Y position in mm
            scale: Scale level
            
        Returns:
            Tuple of (x_pixel, y_pixel) coordinates relative to well center
        """
        # Convert absolute coordinates to well-relative coordinates
        well_relative_x = x_mm - self.well_center_x
        well_relative_y = y_mm - self.well_center_y

        # Use parent's coordinate conversion with well-relative coordinates
        return super().stage_to_pixel_coords(well_relative_x, well_relative_y, scale)

    def get_well_info(self) -> dict:
        """
        Get comprehensive information about this well canvas.
        
        Returns:
            dict: Well information including coordinates, size, and metadata
        """
        return {
            "well_info": {
                "row": self.well_row,
                "column": self.well_column,
                "well_id": f"{self.well_row}{self.well_column}",
                "well_plate_type": self.well_plate_type,
                "well_center_x_mm": self.well_center_x,
                "well_center_y_mm": self.well_center_y,
                "well_diameter_mm": self.wellplate_format.WELL_SIZE_MM,
                "well_spacing_mm": self.wellplate_format.WELL_SPACING_MM,
                "padding_mm": self.padding_mm
            },
            "canvas_info": {
                "canvas_width_mm": self.stage_limits['x_positive'] - self.stage_limits['x_negative'],
                "canvas_height_mm": self.stage_limits['y_positive'] - self.stage_limits['y_negative'],
                "coordinate_system": "well_relative",
                "origin": "well_center",
                "canvas_width_px": self.canvas_width_px,
                "canvas_height_px": self.canvas_height_px,
                "pixel_size_xy_um": self.pixel_size_xy_um
            }
        }


class ExperimentManager:
    """
    Manages experiment folders containing well-specific zarr canvases.
    
    Each experiment is a folder containing multiple well canvases:
    ZARR_PATH/experiment_name/A1_96.zarr, A2_96.zarr, etc.
    
    This replaces the single-canvas system with a well-separated approach.
    """

    def __init__(self, base_path: str, pixel_size_xy_um: float):
        """
        Initialize the experiment manager.
        
        Args:
            base_path: Base directory for zarr storage (from ZARR_PATH env variable)
            pixel_size_xy_um: Pixel size in micrometers
        """
        self.base_path = Path(base_path)
        self.pixel_size_xy_um = pixel_size_xy_um
        self.current_experiment = None  # Current experiment name
        self.well_canvases = {}  # {well_id: WellZarrCanvas} for current experiment

        # Ensure base directory exists
        self.base_path.mkdir(parents=True, exist_ok=True)

        # Set 'default' as the default experiment
        self._ensure_default_experiment()

        logger.info(f"ExperimentManager initialized at {self.base_path}")

    def _ensure_default_experiment(self):
        """
        Ensure that a 'default' experiment exists and is set as the current experiment.
        Creates the experiment if it doesn't exist.
        """
        default_experiment_name = 'default'
        default_experiment_path = self.base_path / default_experiment_name

        # Create default experiment if it doesn't exist
        if not default_experiment_path.exists():
            default_experiment_path.mkdir(parents=True, exist_ok=True)
            logger.info(f"Created default experiment '{default_experiment_name}'")

        # Set as current experiment
        self.current_experiment = default_experiment_name
        logger.info(f"Set '{default_experiment_name}' as default experiment")

    @property
    def current_experiment_name(self) -> str:
        """Get the current experiment name."""
        return self.current_experiment

    def create_experiment(self, experiment_name: str, well_plate_type: str = '96',
                         well_padding_mm: float = 1.0, initialize_all_wells: bool = False):
        """
        Create a new experiment folder and optionally initialize all well canvases.
        
        Args:
            experiment_name: Name of the experiment
            well_plate_type: Well plate type ('6', '12', '24', '96', '384')
            well_padding_mm: Padding around each well in mm
            initialize_all_wells: If True, create canvases for all wells in the plate
            
        Returns:
            dict: Information about the created experiment
        """
        experiment_path = self.base_path / experiment_name

        if experiment_path.exists():
            raise ValueError(f"Experiment '{experiment_name}' already exists")

        # Create experiment directory
        experiment_path.mkdir(parents=True, exist_ok=True)

        # Set as current experiment
        self.current_experiment = experiment_name
        self.well_canvases = {}

        logger.info(f"Created experiment '{experiment_name}' at {experiment_path}")

        # Optionally initialize all wells
        initialized_wells = []
        if initialize_all_wells:
            well_positions = self._get_all_well_positions(well_plate_type)
            for well_row, well_column in well_positions:
                try:
                    canvas = self.get_well_canvas(well_row, well_column, well_plate_type, well_padding_mm)
                    initialized_wells.append(f"{well_row}{well_column}")
                except Exception as e:
                    logger.warning(f"Failed to initialize well {well_row}{well_column}: {e}")

        return {
            "experiment_name": experiment_name,
            "experiment_path": str(experiment_path),
            "well_plate_type": well_plate_type,
            "initialized_wells": initialized_wells,
            "total_wells": len(initialized_wells) if initialize_all_wells else 0
        }

    def set_active_experiment(self, experiment_name: str):
        """
        Set the active experiment.
        
        Args:
            experiment_name: Name of the experiment to activate
            
        Returns:
            dict: Information about the activated experiment
        """
        experiment_path = self.base_path / experiment_name

        if not experiment_path.exists():
            raise ValueError(f"Experiment '{experiment_name}' not found")

        # Close current well canvases
        for canvas in self.well_canvases.values():
            canvas.close()

        # Set new experiment
        self.current_experiment = experiment_name
        self.well_canvases = {}

        logger.info(f"Set active experiment to '{experiment_name}'")

        return {
            "experiment_name": experiment_name,
            "experiment_path": str(experiment_path),
            "message": f"Activated experiment '{experiment_name}'"
        }

    def list_experiments(self):
        """
        List all available experiments.
        
        Returns:
            dict: List of experiments and their information
        """
        experiments = []

        try:
            for item in self.base_path.iterdir():
                if item.is_dir():
                    # Count well canvases in this experiment
                    well_count = len([f for f in item.iterdir() if f.is_dir() and f.suffix == '.zarr'])

                    experiments.append({
                        "name": item.name,
                        "path": str(item),
                        "is_active": item.name == self.current_experiment,
                        "well_count": well_count
                    })
        except Exception as e:
            logger.error(f"Error listing experiments: {e}")

        return {
            "experiments": experiments,
            "active_experiment": self.current_experiment,
            "total_count": len(experiments)
        }

    def remove_experiment(self, experiment_name: str):
        """
        Remove an experiment and all its well canvases.
        
        Args:
            experiment_name: Name of the experiment to remove
            
        Returns:
            dict: Information about the removed experiment
        """
        if experiment_name == self.current_experiment:
            raise ValueError(f"Cannot remove active experiment '{experiment_name}'. Please switch to another experiment first.")

        experiment_path = self.base_path / experiment_name

        if not experiment_path.exists():
            raise ValueError(f"Experiment '{experiment_name}' not found")

        # Remove experiment directory and all contents
        shutil.rmtree(experiment_path)

        logger.info(f"Removed experiment '{experiment_name}'")

        return {
            "experiment_name": experiment_name,
            "message": f"Removed experiment '{experiment_name}'"
        }

    def reset_experiment(self, experiment_name: str = None):
        """
        Reset an experiment by removing all well canvases but keeping the folder.
        
        Args:
            experiment_name: Name of the experiment to reset (default: current experiment)
            
        Returns:
            dict: Information about the reset experiment
        """
        if experiment_name is None:
            experiment_name = self.current_experiment

        if experiment_name is None:
            raise ValueError("No experiment specified and no active experiment")

        experiment_path = self.base_path / experiment_name

        if not experiment_path.exists():
            raise ValueError(f"Experiment '{experiment_name}' not found")

        # Close well canvases if this is the active experiment
        if experiment_name == self.current_experiment:
            for canvas in self.well_canvases.values():
                canvas.close()
            self.well_canvases = {}

        # Remove all .zarr directories in the experiment folder
        removed_count = 0
        for item in experiment_path.iterdir():
            if item.is_dir() and item.suffix == '.zarr':
                import shutil
                shutil.rmtree(item)
                removed_count += 1

        # If this is the active experiment, also deactivate all channels in active canvases
        if experiment_name == self.current_experiment:
            for canvas in self.well_canvases.values():
                try:
                    canvas.deactivate_all_channels()
                except Exception as e:
                    logger.warning(f"Failed to deactivate channels in canvas: {e}")

        logger.info(f"Reset experiment '{experiment_name}', removed {removed_count} well canvases")

        return {
            "experiment_name": experiment_name,
            "removed_wells": removed_count,
            "message": f"Reset experiment '{experiment_name}'"
        }

    def get_well_canvas(self, well_row: str, well_column: int, well_plate_type: str = '96',
                       padding_mm: float = 1.0, experiment_name: str = None):
        """
        Get or create a well canvas for the specified experiment.
        
        Args:
            well_row: Well row (e.g., 'A', 'B')
            well_column: Well column (e.g., 1, 2, 3)
            well_plate_type: Well plate type ('6', '12', '24', '96', '384')
            padding_mm: Padding around well in mm
            experiment_name: Name of the experiment to get canvas from (default: None uses current experiment)
            
        Returns:
            WellZarrCanvas: The well-specific canvas
        """
        target_experiment = experiment_name if experiment_name is not None else self.current_experiment
        if target_experiment is None:
            raise RuntimeError("No experiment specified and no active experiment. Create or set an experiment first.")

        well_id = f"{well_row}{well_column}_{well_plate_type}"

        # If requesting from current experiment, use cached canvases
        if target_experiment == self.current_experiment and well_id in self.well_canvases:
            return self.well_canvases[well_id]

        # For different experiments or new canvases, create/load directly
        experiment_path = self.base_path / target_experiment

        # Ensure experiment directory exists
        if not experiment_path.exists():
            raise ValueError(f"Experiment '{target_experiment}' does not exist")

        from squid_control.control.config import CONFIG, ChannelMapper
        all_channels = ChannelMapper.get_all_human_names()

        canvas = WellZarrCanvas(
            well_row=well_row,
            well_column=well_column,
            well_plate_type=well_plate_type,
            padding_mm=padding_mm,
            base_path=str(experiment_path),  # Use experiment folder as base
            pixel_size_xy_um=self.pixel_size_xy_um,
            channels=all_channels,
            rotation_angle_deg=CONFIG.STITCHING_ROTATION_ANGLE_DEG,
            initial_timepoints=20,
            timepoint_expansion_chunk=10
        )

        # Only cache if it's for the current experiment
        if target_experiment == self.current_experiment:
            self.well_canvases[well_id] = canvas
            logger.info(f"Created and cached well canvas {well_row}{well_column} for current experiment '{self.current_experiment}'")
        else:
            logger.info(f"Created well canvas {well_row}{well_column} for experiment '{target_experiment}' (not cached)")

        return canvas

    def list_well_canvases(self):
        """
        List all well canvases in the current experiment.
        
        Returns:
            dict: Information about well canvases
        """
        if self.current_experiment is None:
            return {
                "well_canvases": [],
                "experiment_name": None,
                "total_count": 0
            }

        canvases = []

        # List active canvases
        for well_id, canvas in self.well_canvases.items():
            well_info = canvas.get_well_info()
            canvases.append({
                "well_id": well_id,
                "well_row": canvas.well_row,
                "well_column": canvas.well_column,
                "well_plate_type": canvas.well_plate_type,
                "canvas_path": str(canvas.zarr_path),
                "well_center_x_mm": canvas.well_center_x,
                "well_center_y_mm": canvas.well_center_y,
                "padding_mm": canvas.padding_mm,
                "channels": len(canvas.channels),
                "timepoints": len(canvas.available_timepoints),
                "status": "active"
            })

        # List canvases on disk (in experiment folder)
        experiment_path = self.base_path / self.current_experiment
        for item in experiment_path.iterdir():
            if item.is_dir() and item.suffix == '.zarr':
                well_name = item.stem  # e.g., "well_A1_96"
                if well_name not in [c["well_id"] for c in canvases]:
                    canvases.append({
                        "well_id": well_name,
                        "canvas_path": str(item),
                        "status": "on_disk"
                    })

        return {
            "well_canvases": canvases,
            "experiment_name": self.current_experiment,
            "total_count": len(canvases)
        }

    def get_experiment_info(self, experiment_name: str = None):
        """
        Get detailed information about an experiment.
        
        Args:
            experiment_name: Name of the experiment (default: current experiment)
            
        Returns:
            dict: Detailed experiment information including OME-Zarr metadata
        """
        if experiment_name is None:
            experiment_name = self.current_experiment

        if experiment_name is None:
            raise ValueError("No experiment specified and no active experiment")

        experiment_path = self.base_path / experiment_name

        if not experiment_path.exists():
            raise ValueError(f"Experiment '{experiment_name}' not found")

        # Count well canvases and collect OME-Zarr metadata
        well_canvases = []
        total_size_bytes = 0
        omero_metadata = None

        for item in experiment_path.iterdir():
            if item.is_dir() and item.suffix == '.zarr':
                try:
                    # Calculate size
                    size_bytes = sum(f.stat().st_size for f in item.rglob('*') if f.is_file())
                    total_size_bytes += size_bytes

                    # Try to read OME-Zarr metadata from the first well canvas
                    if omero_metadata is None:
                        try:
                            zarr_path = item / '.zattrs'
                            if zarr_path.exists():
                                with open(zarr_path) as f:
                                    attrs = json.load(f)

                                # Extract OME-Zarr metadata
                                if 'omero' in attrs:
                                    omero_metadata = attrs['omero']
                                    logger.debug(f"Found OME-Zarr metadata in {item.name}")
                        except Exception as e:
                            logger.debug(f"Could not read OME-Zarr metadata from {item}: {e}")

                    well_canvases.append({
                        "name": item.stem,
                        "path": str(item),
                        "size_bytes": size_bytes,
                        "size_mb": size_bytes / (1024 * 1024)
                    })
                except Exception as e:
                    logger.warning(f"Error getting info for {item}: {e}")

        # Prepare the result dictionary
        result = {
            "experiment_name": experiment_name,
            "experiment_path": str(experiment_path),
            "is_active": experiment_name == self.current_experiment,
            "well_canvases": well_canvases,
            "total_wells": len(well_canvases),
            "total_size_bytes": total_size_bytes,
            "total_size_mb": total_size_bytes / (1024 * 1024)
        }

        # Add OME-Zarr metadata if available
        if omero_metadata is not None:
            result["omero"] = omero_metadata
        else:
            # Provide default OME-Zarr structure if no metadata found
            result["omero"] = {
                "channels": [
                    {
                        "active": True,
                        "coefficient": 1.0,
                        "color": "FFFFFF",
                        "family": "linear",
                        "label": "BF LED matrix full",
                        "window": {
                            "end": 255,
                            "start": 0
                        }
                    },
                    {
                        "active": True,
                        "coefficient": 1.0,
                        "color": "0000FF",
                        "family": "linear",
                        "label": "Fluorescence 405 nm Ex",
                        "window": {
                            "end": 255,
                            "start": 0
                        }
                    },
                    {
                        "active": True,
                        "coefficient": 1.0,
                        "color": "00FF00",
                        "family": "linear",
                        "label": "Fluorescence 488 nm Ex",
                        "window": {
                            "end": 255,
                            "start": 0
                        }
                    },
                    {
                        "active": True,
                        "coefficient": 1.0,
                        "color": "FF00FF",
                        "family": "linear",
                        "label": "Fluorescence 638 nm Ex",
                        "window": {
                            "end": 255,
                            "start": 0
                        }
                    },
                    {
                        "active": True,
                        "coefficient": 1.0,
                        "color": "FF0000",
                        "family": "linear",
                        "label": "Fluorescence 561 nm Ex",
                        "window": {
                            "end": 255,
                            "start": 0
                        }
                    },
                    {
                        "active": True,
                        "coefficient": 1.0,
                        "color": "00FFFF",
                        "family": "linear",
                        "label": "Fluorescence 730 nm Ex",
                        "window": {
                            "end": 255,
                            "start": 0
                        }
                    }
                ],
                "id": 1,
                "name": f"Squid Microscope Live Stitching ({experiment_name})",
                "rdefs": {
                    "defaultT": 0,
                    "defaultZ": 0,
                    "model": "color"
                }
            }

        return result

    def _get_all_well_positions(self, well_plate_type: str):
        """Get all well positions for a given plate type."""

        if well_plate_type == '6':
            max_rows, max_cols = 2, 3  # A-B, 1-3
        elif well_plate_type == '12':
            max_rows, max_cols = 3, 4  # A-C, 1-4
        elif well_plate_type == '24':
            max_rows, max_cols = 4, 6  # A-D, 1-6
        elif well_plate_type == '96':
            max_rows, max_cols = 8, 12  # A-H, 1-12
        elif well_plate_type == '384':
            max_rows, max_cols = 16, 24  # A-P, 1-24
        else:
            max_rows, max_cols = 8, 12  # Default to 96-well

        positions = []
        for row_idx in range(max_rows):
            for col_idx in range(max_cols):
                row_letter = chr(ord('A') + row_idx)
                col_number = col_idx + 1
                positions.append((row_letter, col_number))

        return positions

    def close(self):
        """Close all well canvases and clean up resources."""
        for canvas in self.well_canvases.values():
            canvas.close()
        self.well_canvases = {}
        logger.info("ExperimentManager closed")


# Alias for backward compatibility
ZarrCanvas = WellZarrCanvasBase
