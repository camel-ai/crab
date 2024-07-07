import base64
from io import BytesIO

from PIL import Image

from crab.core import action


@action
def save_base64_image(image: str, path: str = "image.png") -> None:
    image = Image.open(BytesIO(base64.b64decode(image)))
    image.save(path)
