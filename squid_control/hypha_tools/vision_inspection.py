"""
Vision inspection tool for microscope images using GPT-5.1 vision model.
"""
import base64
import logging
import os
from io import BytesIO

import dotenv
import matplotlib.pyplot as plt
from openai import AsyncOpenAI
from PIL import Image
from pydantic import BaseModel, Field

# Load environment variables from .env file
dotenv.load_dotenv()

logger = logging.getLogger(__name__)


class ImageInfo(BaseModel):
    """Image information."""
    base64_data: str = Field(..., description="Base64-encoded PNG image data (without data URL prefix).")
    title: str | None = Field(None, description="The title of the image.")


async def aask(images, messages, max_tokens=1024):
    """
    Ask GPT-5.1 vision model about images.
    
    Args:
        images: List of ImageInfo objects containing base64-encoded PNG images and titles
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
    # decode base64 images and save them into a list of PIL image objects
    img_objs = []
    for image in images:
        try:
            # Decode base64 string to bytes
            image_bytes = base64.b64decode(image.base64_data)
            img = Image.open(BytesIO(image_bytes))
        except Exception as e:
            raise ValueError(
                f"Failed to decode base64 image {image.title or ''}. Error: {e}"
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
    image: str,
    query: str,
    context_description: str,
    check_permission=None,
    context=None,
    title: str | None = None
) -> str:
    """
    Inspect an image using GPT-5.1 vision model for analysis and description.
    
    Args:
        image: Base64-encoded PNG image data (with or without data URL prefix)
        query: User query about the image for GPT-5.1 vision model analysis
        context_description: Context description for the visual inspection task
        check_permission: Optional permission check function
        context: Optional request context for authentication
        title: Optional title for the image
    
    Returns:
        String response from the vision model containing image analysis based on the query.
    
    Notes:
        The image must be base64-encoded PNG format. The method validates base64 data and processes 
        the image through the GPT-5.1 vision API.
    """
    try:
        # Check authentication if permission check function is provided
        if check_permission and context and not check_permission(context.get("user", {})):
            raise Exception("User not authorized to access this service")

        base64_data = image
        
        # Remove data URL prefix if present (e.g., "data:image/png;base64,")
        if base64_data.startswith("data:image"):
            base64_data = base64_data.split(",", 1)[1]
        
        # Validate base64 format by attempting to decode
        try:
            base64.b64decode(base64_data, validate=True)
        except Exception as e:
            raise ValueError(f"Invalid base64 data: {e}")
        
        image_info = ImageInfo(base64_data=base64_data, title=title)

        logger.info(f"Inspecting image with GPT-5.1 vision model. Query: {query[:100]}")
        # aask expects a list of images, so we pass a single-item list
        response = await aask([image_info], [context_description, query])
        logger.info("Image inspection completed successfully")
        return response
    except Exception as e:
        logger.error(f"Failed to inspect image: {e}")
        raise e

