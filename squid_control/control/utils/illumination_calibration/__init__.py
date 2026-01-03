"""
Flat-field illumination calibration utilities for microscopy image correction.

This module provides functions to:
1. Calculate normalized shading fields from flat-field images
2. Apply flat-field correction to microscopy images
3. Resize shading fields to match different image dimensions

The correction removes uneven illumination (e.g., brighter center) using the formula:
    I_corrected(x,y) = I(x,y) / S(x,y)
where S(x,y) is a normalized shading field (mean=1.0).
"""

import cv2
import numpy as np


def calculate_shading_field(flat_image: np.ndarray, sigma: float = 70) -> np.ndarray:
    """
    Extract normalized shading field from flat image using Gaussian blur.
    
    This function:
    1. Applies heavy Gaussian blur to extract the illumination pattern
    2. Normalizes the result to mean=1.0 to remove absolute intensity dependence
    
    Args:
        flat_image: Flat-field image (grayscale, uint8 or uint16)
        sigma: Gaussian blur sigma in pixels (default 70). Larger values = more smoothing.
               Typical range: 50-100 pixels for microscopy images.
    
    Returns:
        Normalized shading field (float32, mean=1.0)
    
    Example:
        >>> flat_img = cv2.imread('flat_field.bmp', cv2.IMREAD_GRAYSCALE)
        >>> shading = calculate_shading_field(flat_img, sigma=70)
        >>> print(f"Shading mean: {shading.mean():.3f}")  # Should be ~1.0
    """
    # Convert to float for processing
    flat_float = flat_image.astype(np.float32)
    
    # Calculate kernel size from sigma (6*sigma covers ~99.7% of Gaussian)
    kernel_size = int(6 * sigma + 1)
    if kernel_size % 2 == 0:
        kernel_size += 1  # Must be odd
    
    # Apply Gaussian blur to extract smooth background illumination pattern
    background = cv2.GaussianBlur(flat_float, (kernel_size, kernel_size), sigma)
    
    # Normalize to mean=1.0
    # This makes the shading field independent of absolute intensity
    mean_intensity = np.mean(background)
    if mean_intensity > 0:
        shading_field = background / mean_intensity
    else:
        # Fallback for completely black image
        shading_field = np.ones_like(background, dtype=np.float32)
    
    return shading_field


def apply_flat_field_correction(image: np.ndarray, shading_field: np.ndarray) -> np.ndarray:
    """
    Apply flat-field correction to image using division.
    
    Formula: I_corrected(x,y) = I(x,y) / S(x,y)
    
    This normalizes the illumination across the image, making the center and edges
    have similar intensity for uniform samples.
    
    Args:
        image: Input image (uint8 or uint16)
        shading_field: Normalized shading field (float32, same shape as image)
    
    Returns:
        Corrected image (same dtype as input)
    
    Raises:
        ValueError: If shading field shape doesn't match image shape
    
    Example:
        >>> img = cv2.imread('microscopy_image.tif', cv2.IMREAD_GRAYSCALE)
        >>> shading = load_shading_field('shading_BF.npy')
        >>> corrected = apply_flat_field_correction(img, shading)
    """
    # Ensure shading field matches image dimensions
    if shading_field.shape != image.shape:
        raise ValueError(
            f"Shading field shape {shading_field.shape} must match image shape {image.shape}. "
            f"Use resize_shading_field() to resize first."
        )
    
    # Convert to float for division
    image_float = image.astype(np.float32)
    
    # Apply correction: I_corrected = I / S
    # Add small epsilon to avoid division by zero (though shading should never be zero)
    corrected = image_float / (shading_field + 1e-6)
    
    # Clip to valid range and convert back to original dtype
    if image.dtype == np.uint8:
        corrected = np.clip(corrected, 0, 255).astype(np.uint8)
    elif image.dtype == np.uint16:
        corrected = np.clip(corrected, 0, 65535).astype(np.uint16)
    else:
        # For other dtypes, just cast (e.g., float32)
        corrected = corrected.astype(image.dtype)
    
    return corrected


def resize_shading_field(shading_field: np.ndarray, target_shape: tuple) -> np.ndarray:
    """
    Resize shading field to match target image dimensions.
    
    This is needed when the flat-field image was captured at a different resolution
    than the images being corrected (e.g., due to cropping, binning, or different scales).
    
    After resizing, the field is re-normalized to mean=1.0 to maintain the correction
    property.
    
    Args:
        shading_field: Original shading field (float32)
        target_shape: Target (height, width) tuple
    
    Returns:
        Resized and re-normalized shading field (float32, mean=1.0)
    
    Example:
        >>> shading = np.load('shading_field.npy')  # 3000x3000
        >>> img_cropped = image[500:1500, 500:1500]  # 1000x1000
        >>> shading_resized = resize_shading_field(shading, img_cropped.shape)
        >>> corrected = apply_flat_field_correction(img_cropped, shading_resized)
    """
    # Resize using bilinear interpolation (smooth for illumination gradients)
    resized = cv2.resize(
        shading_field, 
        (target_shape[1], target_shape[0]),  # cv2.resize expects (width, height)
        interpolation=cv2.INTER_LINEAR
    )
    
    # Re-normalize to mean=1.0 after resize
    # This is important because interpolation can slightly change the mean
    mean_val = np.mean(resized)
    if mean_val > 0:
        resized = resized / mean_val
    
    return resized.astype(np.float32)


# Export public API
__all__ = [
    'calculate_shading_field',
    'apply_flat_field_correction',
    'resize_shading_field',
]





