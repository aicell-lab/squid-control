"""
Video frame acquisition utilities for the Squid microscope control system.

This module provides classes and functions for video buffering, frame processing,
and video streaming functionality.
"""

import logging
import threading
import time
from collections import deque

import cv2
import numpy as np

logger = logging.getLogger(__name__)


class VideoBuffer:
    """
    Video buffer to store and manage compressed microscope frames for smooth video streaming
    """
    def __init__(self, max_size=5):
        self.max_size = max_size
        self.buffer = deque(maxlen=max_size)
        self.lock = threading.Lock()
        self.last_frame_data = None  # Store compressed frame data
        self.last_metadata = None  # Store metadata for last frame
        self.frame_timestamp = 0

    def put_frame(self, frame_data, metadata=None):
        """Add a compressed frame and its metadata to the buffer

        Args:
            frame_data: dict with compressed frame info from _encode_frame_jpeg()
            metadata: dict with frame metadata including stage position and timestamp
        """
        with self.lock:
            self.buffer.append({
                'frame_data': frame_data,
                'metadata': metadata,
                'timestamp': time.time()
            })
            self.last_frame_data = frame_data
            self.last_metadata = metadata
            self.frame_timestamp = time.time()

    def get_frame_data(self):
        """Get the most recent compressed frame data and metadata from buffer

        Returns:
            tuple: (frame_data, metadata) or (None, None) if no frame available
        """
        with self.lock:
            if self.buffer:
                buffer_entry = self.buffer[-1]
                return buffer_entry['frame_data'], buffer_entry.get('metadata')
            elif self.last_frame_data is not None:
                return self.last_frame_data, self.last_metadata
            else:
                return None, None

    def get_frame(self):
        """Get the most recent decompressed frame from buffer (for backward compatibility)"""
        frame_data, _ = self.get_frame_data()  # Ignore metadata for backward compatibility
        if frame_data is None:
            return None

        # Decode JPEG back to numpy array
        try:
            if frame_data['format'] == 'jpeg':
                # Decode JPEG data
                nparr = np.frombuffer(frame_data['data'], np.uint8)
                bgr_frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                if bgr_frame is not None:
                    # Convert BGR back to RGB
                    return cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)
            elif frame_data['format'] == 'raw':
                # Raw numpy data
                return np.frombuffer(frame_data['data'], dtype=np.uint8).reshape((-1, 750, 3))
        except Exception as e:
            logger.error(f"Error decoding frame: {e}")

        return None

    def get_frame_age(self):
        """Get the age of the most recent frame in seconds"""
        with self.lock:
            if self.frame_timestamp > 0:
                return time.time() - self.frame_timestamp
            else:
                return float('inf')

    def clear(self):
        """Clear the buffer"""
        with self.lock:
            self.buffer.clear()
            self.last_frame_data = None
            self.last_metadata = None
            self.frame_timestamp = 0


class VideoFrameProcessor:
    """
    Utility class for processing video frames for streaming
    """
    
    @staticmethod
    def process_raw_frame(raw_frame, frame_width=750, frame_height=750, 
                         video_contrast_min=0, video_contrast_max=None):
        """Process raw frame for video streaming - OPTIMIZED"""
        try:
            # Import CONFIG here to avoid circular imports
            from squid_control.control.config import CONFIG
            
            # OPTIMIZATION 1: Crop FIRST, then resize to reduce data for all subsequent operations
            crop_height = CONFIG.ACQUISITION.CROP_HEIGHT
            crop_width = CONFIG.ACQUISITION.CROP_WIDTH
            height, width = raw_frame.shape[:2]  # Support both grayscale and color images
            start_x = width // 2 - crop_width // 2
            start_y = height // 2 - crop_height // 2

            # Ensure crop coordinates are within bounds
            start_x = max(0, start_x)
            start_y = max(0, start_y)
            end_x = min(width, start_x + crop_width)
            end_y = min(height, start_y + crop_height)

            cropped_frame = raw_frame[start_y:end_y, start_x:end_x]

            # Now resize the cropped frame to target dimensions
            if cropped_frame.shape[:2] != (frame_height, frame_width):
                # Use INTER_AREA for downsampling (faster than INTER_LINEAR)
                processed_frame = cv2.resize(cropped_frame, (frame_width, frame_height), interpolation=cv2.INTER_AREA)
            else:
                processed_frame = cropped_frame.copy()

            # Calculate gray level statistics on original frame BEFORE min/max adjustments
            gray_level_stats = VideoFrameProcessor._calculate_gray_level_statistics(processed_frame)

            # OPTIMIZATION 2: Robust contrast adjustment (fixed)
            min_val = video_contrast_min
            max_val = video_contrast_max

            if max_val is None:
                if processed_frame.dtype == np.uint16:
                    max_val = 65535
                else:
                    max_val = 255

            # OPTIMIZATION 3: Improved contrast scaling with proper range handling
            if max_val > min_val:
                # Clip values to the specified range
                processed_frame = np.clip(processed_frame, min_val, max_val)

                # Scale to 0-255 range using float for precision, then convert to uint8
                if max_val > min_val:
                    # Use float32 for accurate scaling, then convert to uint8
                    processed_frame = ((processed_frame.astype(np.float32) - min_val) / (max_val - min_val) * 255).astype(np.uint8)
                else:
                    # Edge case: min_val == max_val
                    processed_frame = np.full_like(processed_frame, 127, dtype=np.uint8)
            else:
                # Edge case: max_val <= min_val, return mid-gray
                height, width = processed_frame.shape[:2]
                processed_frame = np.full((height, width), 127, dtype=np.uint8)

            # Ensure we have uint8 output
            if processed_frame.dtype != np.uint8:
                processed_frame = processed_frame.astype(np.uint8)

            # OPTIMIZATION 4: Fast color space conversion
            if len(processed_frame.shape) == 2:
                # Direct array manipulation is faster than cv2.cvtColor for grayscale->RGB
                processed_frame = np.stack([processed_frame] * 3, axis=2)
            elif processed_frame.shape[2] == 1:
                processed_frame = np.repeat(processed_frame, 3, axis=2)

            return processed_frame, gray_level_stats

        except Exception as e:
            logger.error(f"Error processing frame: {e}")
            placeholder_frame = VideoFrameProcessor._create_placeholder_frame(frame_width, frame_height, f"Processing Error: {str(e)}")
            placeholder_stats = VideoFrameProcessor._calculate_gray_level_statistics(placeholder_frame)
            return placeholder_frame, placeholder_stats

    @staticmethod
    def _create_placeholder_frame(width, height, message="No Frame Available"):
        """Create a placeholder frame with error message"""
        placeholder_img = np.zeros((height, width, 3), dtype=np.uint8)
        cv2.putText(placeholder_img, message, (10, height//2),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (128, 128, 128), 2)
        return placeholder_img

    @staticmethod
    def _calculate_gray_level_statistics(rgb_frame):
        """Calculate comprehensive gray level statistics for microscope analysis"""
        try:
            # Convert RGB to grayscale for analysis (standard luminance formula)
            if len(rgb_frame.shape) == 3:
                # RGB to grayscale: Y = 0.299*R + 0.587*G + 0.114*B
                gray_frame = np.dot(rgb_frame[...,:3], [0.299, 0.587, 0.114])
            else:
                gray_frame = rgb_frame

            # Ensure we have a valid grayscale image
            if gray_frame.size == 0:
                return None

            # Convert to 0-100% range for analysis
            gray_normalized = (gray_frame / 255.0) * 100.0

            # Calculate comprehensive statistics
            stats = {
                'mean_percent': float(np.mean(gray_normalized)),
                'std_percent': float(np.std(gray_normalized)),
                'min_percent': float(np.min(gray_normalized)),
                'max_percent': float(np.max(gray_normalized)),
                'median_percent': float(np.median(gray_normalized)),
                'percentiles': {
                    'p5': float(np.percentile(gray_normalized, 5)),
                    'p25': float(np.percentile(gray_normalized, 25)),
                    'p75': float(np.percentile(gray_normalized, 75)),
                    'p95': float(np.percentile(gray_normalized, 95))
                },
                'histogram': {
                    'bins': 20,  # 20 bins for 0-100% range (5% per bin)
                    'counts': [],
                    'bin_edges': []
                }
            }

            # Calculate histogram (20 bins from 0-100%)
            hist_counts, bin_edges = np.histogram(gray_normalized, bins=20, range=(0, 100))
            stats['histogram']['counts'] = hist_counts.tolist()
            stats['histogram']['bin_edges'] = bin_edges.tolist()

            # Additional microscope-specific metrics
            stats['dynamic_range_percent'] = stats['max_percent'] - stats['min_percent']
            stats['contrast_ratio'] = stats['std_percent'] / stats['mean_percent'] if stats['mean_percent'] > 0 else 0

            # Exposure quality indicators
            stats['exposure_quality'] = {
                'underexposed_pixels_percent': float(np.sum(gray_normalized < 5) / gray_normalized.size * 100),
                'overexposed_pixels_percent': float(np.sum(gray_normalized > 95) / gray_normalized.size * 100),
                'well_exposed_pixels_percent': float(np.sum((gray_normalized >= 5) & (gray_normalized <= 95)) / gray_normalized.size * 100)
            }

            return stats

        except Exception as e:
            logger.warning(f"Error calculating gray level statistics: {e}")
            return None

    @staticmethod
    def decode_frame_jpeg(frame_data):
        """
        Decode compressed frame data back to numpy array
        
        Args:
            frame_data: dict from _encode_frame_jpeg() or get_video_frame()
        
        Returns:
            numpy array: RGB image data
        """
        try:
            if frame_data['format'] == 'jpeg':
                # Decode JPEG data
                nparr = np.frombuffer(frame_data['data'], np.uint8)
                bgr_frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                if bgr_frame is not None:
                    # Convert BGR back to RGB
                    return cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)
            elif frame_data['format'] == 'raw':
                # Raw numpy data
                height = frame_data.get('height', 750)
                width = frame_data.get('width', 750)
                return np.frombuffer(frame_data['data'], dtype=np.uint8).reshape((height, width, 3))
        except Exception as e:
            logger.error(f"Error decoding frame: {e}")

        # Return placeholder on error
        width = frame_data.get('width', 750)
        height = frame_data.get('height', 750)
        return VideoFrameProcessor._create_placeholder_frame(width, height, "Decode Error")

    @staticmethod
    def encode_frame_jpeg(frame, quality=85):
        """
        Encode frame to JPEG format for efficient network transmission
        
        Args:
            frame: RGB numpy array
            quality: JPEG quality (1-100, higher = better quality, larger size)
        
        Returns:
            dict: {
                'format': 'jpeg',
                'data': bytes,
                'size_bytes': int,
                'compression_ratio': float
            }
        """
        try:
            # Convert RGB to BGR for OpenCV JPEG encoding
            if len(frame.shape) == 3 and frame.shape[2] == 3:
                bgr_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            else:
                bgr_frame = frame

            # Encode to JPEG with specified quality
            encode_params = [cv2.IMWRITE_JPEG_QUALITY, quality]
            success, encoded_img = cv2.imencode('.jpg', bgr_frame, encode_params)

            if not success:
                raise ValueError("Failed to encode frame to JPEG")

            # Calculate compression statistics
            original_size = frame.nbytes
            compressed_size = len(encoded_img)
            compression_ratio = original_size / compressed_size if compressed_size > 0 else 1.0

            return {
                'format': 'jpeg',
                'data': encoded_img.tobytes(),
                'size_bytes': compressed_size,
                'compression_ratio': compression_ratio,
                'original_size': original_size
            }

        except Exception as e:
            logger.error(f"Error encoding frame to JPEG: {e}")
            # Return uncompressed as fallback
            raise e
