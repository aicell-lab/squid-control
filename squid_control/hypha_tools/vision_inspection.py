"""
Vision inspection tool for microscope images using GPT-4 vision model.
"""
import logging
from typing import Any, Dict, List

from pydantic import BaseModel, Field

from .chatbot.aask import aask

logger = logging.getLogger(__name__)


class ImageInfo(BaseModel):
    """Image information."""
    url: str = Field(..., description="The URL of the image.")
    title: str | None = Field(None, description="The title of the image.")


async def inspect_images(
    images: List[Dict[str, Any]],
    query: str,
    context_description: str,
    check_permission=None,
    context=None
) -> str:
    """
    Inspect images using GPT-4's vision model (GPT-4o) for analysis and description.
    
    Args:
        images: List of dictionaries, each containing 'http_url' (required) and optionally 'title'
        query: User query about the images for GPT-4 vision model analysis
        context_description: Context description for the visual inspection task
        check_permission: Optional permission check function
        context: Optional request context for authentication
    
    Returns:
        String response from the vision model containing image analysis based on the query.
    
    Notes:
        All image URLs must be HTTP/HTTPS accessible. The method validates URLs and processes 
        images through the GPT-4 vision API.
    """
    try:
        # Check authentication if permission check function is provided
        if check_permission and context and not check_permission(context.get("user", {})):
            raise Exception("User not authorized to access this service")

        image_infos = [
            ImageInfo(url=image_dict['http_url'], title=image_dict.get('title'))
            for image_dict in images
        ]
        for image_info_obj in image_infos:
            if not image_info_obj.url.startswith("http"):
                raise ValueError("Image URL must start with http or https.")

        logger.info(f"Inspecting {len(image_infos)} image(s) with GPT-4 vision model. Query: {query[:100]}")
        response = await aask(image_infos, [context_description, query])
        logger.info("Image inspection completed successfully")
        return response
    except Exception as e:
        logger.error(f"Failed to inspect images: {e}")
        raise e

