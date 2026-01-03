"""
Illumination calibration manager with dynamic channel discovery.

This module manages flat-field calibrations for all microscopy channels,
automatically discovering available flat images and computing/caching shading fields.
"""

import logging
from pathlib import Path
from typing import Dict, Optional

import cv2
import numpy as np

from . import calculate_shading_field

logger = logging.getLogger(__name__)


class IlluminationCalibrationManager:
    """
    Manages flat-field illumination calibration with dynamic channel discovery.
    
    Features:
    - Automatically scans flat_images directory for available flat images
    - Converts filenames to channel names (flat_BF_LED_matrix_full.bmp -> 'BF LED matrix full')
    - Calculates or loads cached shading fields
    - Cache invalidation based on file modification time
    - Graceful error handling (never crashes microscope startup)
    
    File naming convention:
        flat_{channel_name}.{ext}
        Underscores in filename are converted to spaces in channel name.
    
    Examples:
        flat_BF_LED_matrix_full.bmp -> 'BF LED matrix full'
        flat_Fluorescence_488_nm_Ex.bmp -> 'Fluorescence 488 nm Ex'
        flat_Fluorescence_561_nm_Ex.bmp -> 'Fluorescence 561 nm Ex'
    """
    
    def __init__(self, flat_images_dir: str, cache_dir: str, sigma: float = 70):
        """
        Initialize calibration manager.
        
        Args:
            flat_images_dir: Directory containing flat images (e.g., flat_BF_LED_matrix_full.bmp)
            cache_dir: Directory for caching calculated shading fields (.npy files)
            sigma: Gaussian blur sigma for shading field calculation (default 70 pixels)
        """
        self.flat_images_dir = Path(flat_images_dir)
        self.cache_dir = Path(cache_dir)
        self.sigma = sigma
        
        # Storage for calculated shading fields: {channel_name: shading_array}
        self.shading_fields: Dict[str, np.ndarray] = {}
        
        # Ensure directories exist
        self.flat_images_dir.mkdir(parents=True, exist_ok=True)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        logger.debug(f"IlluminationCalibrationManager initialized:")
        logger.debug(f"  Flat images dir: {self.flat_images_dir}")
        logger.debug(f"  Cache dir: {self.cache_dir}")
        logger.debug(f"  Gaussian sigma: {self.sigma}")
    
    def _discover_flat_images(self) -> Dict[str, Path]:
        """
        Dynamically discover flat images in flat_images directory.
        
        Returns:
            Dict mapping channel names to flat image file paths
            
        File naming convention: flat_{channel_name}.{ext}
        Example: flat_BF_LED_matrix_full.bmp -> 'BF LED matrix full'
        
        Supported image formats: .bmp, .png, .jpg, .tiff, etc. (anything OpenCV can read)
        """
        channel_to_file = {}
        
        if not self.flat_images_dir.exists():
            logger.warning(f"Flat images directory does not exist: {self.flat_images_dir}")
            return channel_to_file
        
        # Scan for flat image files
        for file_path in self.flat_images_dir.iterdir():
            if file_path.is_file() and file_path.name.startswith('flat_'):
                # Extract channel name from filename
                # Format: flat_{channel_name}.{ext}
                filename = file_path.stem  # Remove extension
                if filename.startswith('flat_'):
                    channel_name = filename[5:]  # Remove 'flat_' prefix
                    # Convert underscores to spaces for channel name
                    channel_name = channel_name.replace('_', ' ')
                    
                    channel_to_file[channel_name] = file_path
                    logger.debug(f"Discovered flat image for channel '{channel_name}': {file_path.name}")
        
        if channel_to_file:
            logger.info(f"Discovered {len(channel_to_file)} flat images for channels: {list(channel_to_file.keys())}")
        else:
            logger.info("No flat images found in directory")
        
        return channel_to_file
    
    def _get_cache_path(self, channel_name: str) -> Path:
        """
        Get cache file path for a channel.
        
        Args:
            channel_name: Channel name (e.g., 'BF LED matrix full')
        
        Returns:
            Path to cache file (e.g., 'shading_field_BF_LED_matrix_full.npy')
        """
        # Sanitize channel name for filename (spaces to underscores)
        safe_name = channel_name.replace(' ', '_')
        return self.cache_dir / f"shading_field_{safe_name}.npy"
    
    def _should_recalculate(self, flat_image_path: Path, cache_path: Path) -> bool:
        """
        Check if shading field needs recalculation.
        
        Recalculation is needed if:
        - Cache doesn't exist
        - Flat image is newer than cache (user updated the flat image)
        
        Args:
            flat_image_path: Path to flat image file
            cache_path: Path to cache file
        
        Returns:
            True if recalculation needed, False if cache is valid
        """
        if not cache_path.exists():
            return True
        
        # Check if flat image is newer than cache
        flat_mtime = flat_image_path.stat().st_mtime
        cache_mtime = cache_path.stat().st_mtime
        
        return flat_mtime > cache_mtime
    
    def load_or_calculate_shading_fields(self):
        """
        Discover flat images and load/calculate shading fields for all available channels.
        
        This method:
        1. Scans flat_images directory for flat image files
        2. For each channel, checks if cached shading exists and is up-to-date
        3. Loads from cache if available, otherwise calculates and caches
        4. Stores all shading fields in self.shading_fields dict
        
        Error handling:
        - Missing/corrupted flat images are logged as errors and skipped
        - Calculation failures are logged but don't stop other channels
        - This method never raises exceptions (graceful degradation)
        """
        # Discover available flat images
        channel_to_file = self._discover_flat_images()
        
        if not channel_to_file:
            logger.info("No flat images found - flat-field correction will not be available")
            return
        
        # Process each channel
        successful_count = 0
        failed_count = 0
        
        for channel_name, flat_image_path in channel_to_file.items():
            cache_path = self._get_cache_path(channel_name)
            
            try:
                # Check if we need to recalculate
                if self._should_recalculate(flat_image_path, cache_path):
                    logger.info(f"Calculating shading field for '{channel_name}' from {flat_image_path.name}")
                    
                    # Load flat image
                    flat_image = cv2.imread(str(flat_image_path), cv2.IMREAD_GRAYSCALE)
                    if flat_image is None:
                        logger.error(f"Failed to load flat image: {flat_image_path}")
                        failed_count += 1
                        continue
                    
                    # Calculate shading field
                    shading_field = calculate_shading_field(flat_image, self.sigma)
                    
                    # Save to cache
                    np.save(cache_path, shading_field)
                    logger.info(f"Saved shading field cache: {cache_path.name}")
                    
                    self.shading_fields[channel_name] = shading_field
                    successful_count += 1
                else:
                    # Load from cache
                    logger.info(f"Loading cached shading field for '{channel_name}'")
                    shading_field = np.load(cache_path)
                    self.shading_fields[channel_name] = shading_field
                    successful_count += 1
                
                logger.info(f"Shading field ready for '{channel_name}': "
                          f"shape={shading_field.shape}, mean={shading_field.mean():.3f}, "
                          f"min={shading_field.min():.3f}, max={shading_field.max():.3f}")
                
            except Exception as e:
                logger.error(f"Failed to process shading field for '{channel_name}': {e}", exc_info=True)
                failed_count += 1
                continue
        
        # Summary log
        if successful_count > 0:
            logger.info(f"✓ Illumination calibration ready for {successful_count} channels: {list(self.shading_fields.keys())}")
        if failed_count > 0:
            logger.warning(f"⚠ Failed to load calibration for {failed_count} channels")
    
    def get_shading_field(self, channel_name: str) -> Optional[np.ndarray]:
        """
        Get shading field for a specific channel.
        
        Args:
            channel_name: Channel name (e.g., 'BF LED matrix full')
        
        Returns:
            Shading field array (float32, mean=1.0) or None if not available
        """
        return self.shading_fields.get(channel_name)
    
    def has_calibration(self, channel_name: str) -> bool:
        """
        Check if calibration exists for a channel.
        
        Args:
            channel_name: Channel name (e.g., 'BF LED matrix full')
        
        Returns:
            True if calibration is available, False otherwise
        """
        return channel_name in self.shading_fields
    
    def get_available_channels(self) -> list:
        """
        Get list of channels with calibration.
        
        Returns:
            List of channel names that have shading fields available
        """
        return list(self.shading_fields.keys())
    
    def get_calibration_info(self) -> Dict:
        """
        Get information about loaded calibrations.
        
        Returns:
            Dict with calibration statistics and status
        """
        return {
            'num_channels': len(self.shading_fields),
            'channels': list(self.shading_fields.keys()),
            'flat_images_dir': str(self.flat_images_dir),
            'cache_dir': str(self.cache_dir),
            'sigma': self.sigma,
        }





