"""
Vision inspection tool for microscope images using GPT-5.1 vision model.
"""
import base64
import logging
import os
from io import BytesIO
from typing import Any, Dict, List

import dotenv
import httpx
import matplotlib.pyplot as plt
from openai import AsyncOpenAI
from PIL import Image
from pydantic import BaseModel, Field

# Load environment variables from .env file
dotenv.load_dotenv()

logger = logging.getLogger(__name__)


class ImageInfo(BaseModel):
    """Image information."""
    url: str = Field(..., description="The URL of the image.")
    title: str | None = Field(None, description="The title of the image.")


async def aask(images, messages, max_tokens=1024):
    """
    Ask GPT-5.1 vision model about images.
    
    Args:
        images: List of ImageInfo objects containing image URLs and titles
        messages: List of string messages to send to the model
        max_tokens: Maximum tokens for the response
    
    Returns:
        String response from the vision model
    """
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY environment variable not set.")
    aclient = AsyncOpenAI(api_key=api_key)
    user_message = []
    # download the images and save it into a list of PIL image objects
    img_objs = []
    for image in images:
        async with httpx.AsyncClient() as client:
            response = await client.get(image.url)
            response.raise_for_status()
        try:
            img = Image.open(BytesIO(response.content))
        except Exception as e:
            raise ValueError(
                f"Failed to read image {image.title or ''} from {image.url}. Error: {e}"
            ) from e
        img_objs.append(img)

    if len(img_objs) == 1:
        # plot the image with matplotlib
        plt.imshow(img_objs[0])
        if images[0].title:
            plt.title(images[0].title)
        fig = plt.gcf()
    else:
        # plot them in subplots with matplotlib in a row
        fig, ax = plt.subplots(1, len(img_objs), figsize=(15, 5))
        for i, img in enumerate(img_objs):
            ax[i].imshow(img)
            if images[i].title:
                ax[i].set_title(images[i].title)
    # save the plot to a buffer as png format and convert to base64
    buffer = BytesIO()
    fig.tight_layout()
    # if the image size (width or height) is smaller than 512, use the original size and aspect ratio
    # otherwise set the maximun width of the image to n*512 pixels, where n is the number of images; the maximum total width is 1024 pixels
    fig_width = min(1024, len(img_objs) * 512, fig.get_figwidth() * fig.dpi)
    # make sure the pixel size (not inches)
    fig.set_size_inches(fig_width / fig.dpi, fig.get_figheight(), forward=True)

    # save fig
    fig.savefig(buffer, format="png")
    buffer.seek(0)
    base64_image = base64.b64encode(buffer.read()).decode("utf-8")
    # append the image to the user message
    user_message.append(
        {
            "type": "image_url",
            "image_url": {"url": f"data:image/png;base64,{base64_image}"},
        }
    )

    for message in messages:
        assert isinstance(message, str), "Message must be a string."
        user_message.append({"type": "text", "text": message})

    response = await aclient.chat.completions.create(
        model="gpt-5.1",
        messages=[
            {
                "role": "system",
                "content": "You are a helpful AI assistant that help user to inspect the provided images visually based on the context, make insightful comments and answer questions about the provided images.",
            },
            {"role": "user", "content": user_message},
        ],
        max_completion_tokens=max_tokens,
    )
    return response.choices[0].message.content


async def inspect_images(
    images: List[Dict[str, Any]],
    query: str,
    context_description: str,
    check_permission=None,
    context=None
) -> str:
    """
    Inspect images using GPT-5.1 vision model for analysis and description.
    
    Args:
        images: List of dictionaries, each containing 'http_url' (required) and optionally 'title'
        query: User query about the images for GPT-5.1 vision model analysis
        context_description: Context description for the visual inspection task
        check_permission: Optional permission check function
        context: Optional request context for authentication
    
    Returns:
        String response from the vision model containing image analysis based on the query.
    
    Notes:
        All image URLs must be HTTP/HTTPS accessible. The method validates URLs and processes 
        images through the GPT-5.1 vision API.
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

        logger.info(f"Inspecting {len(image_infos)} image(s) with GPT-5.1 vision model. Query: {query[:100]}")
        response = await aask(image_infos, [context_description, query])
        logger.info("Image inspection completed successfully")
        return response
    except Exception as e:
        logger.error(f"Failed to inspect images: {e}")
        raise e

