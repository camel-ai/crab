# =========== Copyright 2024 @ CAMEL-AI.org. All Rights Reserved. ===========
# Licensed under the Apache License, Version 2.0 (the “License”);
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an “AS IS” BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# =========== Copyright 2024 @ CAMEL-AI.org. All Rights Reserved. ===========
import base64
from io import BytesIO
from typing import Tuple, Optional

from PIL import Image

from crab.core import action


def resize_image(image: Image.Image, ratio: float) -> Image.Image:
    """
    Resize the given image by the specified ratio while maintaining aspect ratio.

    :param image: The original image.
    :param ratio: The ratio to resize the image. Must be between 0 and 1 (exclusive).
    :return: The resized image.
    """
    if not (0 < ratio <= 1):
        raise ValueError("Ratio must be between 0 and 1 (exclusive).")

    new_width = int(image.width * ratio)
    new_height = int(image.height * ratio)

    return image.resize((new_width, new_height), Image.LANCZOS)  # 使用 LANCZOS 替代 ANTIALIAS

def compress_image(image: Image.Image, quality: int) -> BytesIO:
    """
    Compress the given image to the specified quality.

    :param image: The original image.
    :param quality: The quality level (1-100).
    :return: A BytesIO object containing the compressed image.
    """
    output = BytesIO()
    image.save(output, format="JPEG", quality=quality)
    output.seek(0)
    return output

@action
def save_base64_image(image: str, path: str = "image.png", ratio: Optional[float] = None, quality: Optional[int] = None) -> None:
    """
    Save a base64 encoded image to a file, optionally resizing and compressing it.

    :param image: The base64 encoded image.
    :param path: The file path to save the image.
    :param ratio: The ratio to resize the image. Must be between 0 and 1 (exclusive). If None, the original size is used.
    :param quality: The quality level (1-100) for image compression. If None, no compression is applied.
    """
    # Decode base64 string
    image_data = base64.b64decode(image)
    image = Image.open(BytesIO(image_data))
    
    if ratio:
        image = resize_image(image, ratio)
    
    # Determine the output format based on the file extension
    format = path.split('.')[-1].upper()
    if format == 'JPG':
        format = 'JPEG'
    
    # Convert to RGB if saving as JPEG
    if format == 'JPEG':
        image = image.convert('RGB')
    
    # Save the image
    if quality:
        image.save(path, format=format, quality=quality)
    else:
        image.save(path, format=format)
