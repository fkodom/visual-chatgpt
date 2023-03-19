import base64
import io

from PIL import Image

# TODO: Remove these in favor of 'innovation_utilities' when moving to a separate repo.


def base64_to_image(string: str) -> Image:
    """Convert a base64 string to a PIL image."""
    image_buffer = io.BytesIO(base64.b64decode(string))
    return Image.open(image_buffer)


def image_to_base64(image: Image.Image) -> str:
    """Convert a PIL image to a base64 string."""
    image_buffer = io.BytesIO()
    image.save(image_buffer, format="PNG")
    image_bytes = image_buffer.getvalue()
    return base64.b64encode(image_bytes).decode("utf-8")
