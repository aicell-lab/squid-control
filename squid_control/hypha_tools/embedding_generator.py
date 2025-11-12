"""
Embedding generation module for cell images using agent-lens CLIP model.

This module provides batch embedding generation functionality using the agent-lens
service API for efficient processing of multiple cell images.
"""

import asyncio
import io
import logging
from typing import Callable, List, Optional, Tuple

import aiohttp
import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)

# Agent-lens service configuration
AGENT_LENS_BASE_URL = "https://hypha.aicell.io/agent-lens/apps/agent-lens"
EMBEDDING_BATCH_ENDPOINT = f"{AGENT_LENS_BASE_URL}/embedding/image-batch"


def numpy_to_png_bytes(image: np.ndarray) -> bytes:
    """
    Convert numpy array to PNG bytes.
    
    Args:
        image: Numpy array (grayscale or RGB)
    
    Returns:
        PNG encoded bytes
    """
    # Convert to PIL Image
    if len(image.shape) == 2:
        # Grayscale
        pil_image = Image.fromarray(image, mode='L')
    elif len(image.shape) == 3 and image.shape[2] == 3:
        # RGB
        pil_image = Image.fromarray(image, mode='RGB')
    else:
        raise ValueError(f"Unsupported image shape: {image.shape}")
    
    # Convert to PNG bytes
    buffer = io.BytesIO()
    pil_image.save(buffer, format='PNG', optimize=True)
    return buffer.getvalue()


async def generate_embeddings_batch(
    images: List[np.ndarray],
    batch_size: int = 64,
    retry_attempts: int = 2,
    base_url: str = AGENT_LENS_BASE_URL,
    progress_callback: Optional[Callable[[int, int], None]] = None
) -> List[Optional[List[float]]]:
    """
    Generate CLIP embeddings for a batch of images using agent-lens service.
    
    This function processes images in batches to optimize network usage and GPU utilization.
    Failed individual images return None in the results list.
    
    Args:
        images: List of numpy arrays (grayscale or RGB images)
        batch_size: Maximum number of images per API call (default: 64)
        retry_attempts: Number of retry attempts on failure (default: 2)
        base_url: Base URL for agent-lens service
    
    Returns:
        List of embedding vectors (List[float]) or None for failed images.
        Length matches input images list.
    
    Raises:
        ValueError: If images list is empty
    """
    if not images:
        raise ValueError("Images list cannot be empty")
    
    endpoint = f"{base_url}/embedding/image-batch"
    results: List[Optional[List[float]]] = [None] * len(images)
    
    # Process in batches
    for batch_start in range(0, len(images), batch_size):
        batch_end = min(batch_start + batch_size, len(images))
        batch_images = images[batch_start:batch_end]
        batch_indices = list(range(batch_start, batch_end))
        
        batch_num = batch_start//batch_size + 1
        total_batches = (len(images) + batch_size - 1) // batch_size
        logger.info(f"Processing embedding batch {batch_num}/{total_batches}: "
                   f"images {batch_start}-{batch_end-1} ({len(batch_images)} images)")
        
        # Retry logic for this batch
        last_error = None
        for attempt in range(retry_attempts + 1):
            try:
                # Prepare multipart form data
                form_data = aiohttp.FormData()
                
                for idx, img in enumerate(batch_images):
                    png_bytes = numpy_to_png_bytes(img)
                    form_data.add_field(
                        'images',
                        png_bytes,
                        filename=f'cell_{batch_start + idx}.png',
                        content_type='image/png'
                    )
                
                # Make API request
                async with aiohttp.ClientSession() as session:
                    async with session.post(
                        endpoint,
                        data=form_data,
                        timeout=aiohttp.ClientTimeout(total=120)
                    ) as response:
                        if response.status == 200:
                            result = await response.json()
                            
                            if not result.get('success', False):
                                raise ValueError(f"API returned success=False")
                            
                            # Extract embeddings from results
                            api_results = result.get('results', [])
                            if len(api_results) != len(batch_images):
                                logger.warning(f"Expected {len(batch_images)} results, got {len(api_results)}")
                            
                            batch_success_count = 0
                            for i, api_result in enumerate(api_results):
                                global_idx = batch_start + i
                                if api_result is not None and api_result.get('success', False):
                                    embedding = api_result.get('embedding')
                                    if embedding:
                                        results[global_idx] = embedding
                                        batch_success_count += 1
                                    else:
                                        logger.warning(f"Image {global_idx}: No embedding in result")
                                else:
                                    logger.warning(f"Image {global_idx}: Failed to generate embedding")
                            
                            # Update progress callback
                            if progress_callback:
                                progress_callback(batch_end, len(images))
                            
                            logger.info(f"✅ Batch {batch_num}/{total_batches} complete: {batch_success_count}/{len(batch_images)} embeddings generated")
                            
                            # Success - break retry loop
                            break
                        else:
                            error_text = await response.text()
                            last_error = f"HTTP {response.status}: {error_text}"
                            logger.warning(f"Batch {batch_start}-{batch_end-1} attempt {attempt + 1} failed: {last_error}")
                            
                            # Don't retry on 4xx errors (client errors)
                            if 400 <= response.status < 500:
                                break
                            
            except Exception as e:
                last_error = str(e)
                logger.warning(f"Batch {batch_start}-{batch_end-1} attempt {attempt + 1} exception: {e}")
        
        else:
            # All retries failed for this batch
            logger.error(f"Failed to generate embeddings for batch {batch_start}-{batch_end-1} "
                        f"after {retry_attempts + 1} attempts: {last_error}")
            # Results remain None for this batch
    
    return results


async def generate_embeddings_with_fallback(
    images: List[np.ndarray],
    batch_size: int = 64,
    retry_attempts: int = 2,
    base_url: str = AGENT_LENS_BASE_URL
) -> Tuple[List[Optional[List[float]]], int, int]:
    """
    Generate embeddings with comprehensive error handling and statistics.
    
    Args:
        images: List of numpy arrays
        batch_size: Batch size for API calls
        retry_attempts: Retry attempts per batch
        base_url: Agent-lens base URL
    
    Returns:
        Tuple of (embeddings_list, success_count, failure_count)
    """
    try:
        embeddings = await generate_embeddings_batch(images, batch_size, retry_attempts, base_url)
        
        success_count = sum(1 for e in embeddings if e is not None)
        failure_count = len(embeddings) - success_count
        
        logger.info(f"Embedding generation complete: {success_count}/{len(images)} successful, "
                   f"{failure_count} failed")
        
        return embeddings, success_count, failure_count
        
    except Exception as e:
        logger.error(f"Fatal error in embedding generation: {e}", exc_info=True)
        return [None] * len(images), 0, len(images)


async def generate_single_embedding(
    image: np.ndarray,
    retry_attempts: int = 2,
    base_url: str = AGENT_LENS_BASE_URL
) -> Optional[List[float]]:
    """
    Generate embedding for a single image (convenience function).
    
    Args:
        image: Numpy array (grayscale or RGB)
        retry_attempts: Number of retry attempts
        base_url: Agent-lens base URL
    
    Returns:
        Embedding vector or None if failed
    """
    results = await generate_embeddings_batch([image], batch_size=1, retry_attempts=retry_attempts, base_url=base_url)
    return results[0] if results else None

