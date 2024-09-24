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


def resize_image(image: Image.Image, size: Tuple[int, int]) -> Image.Image:
    return image.resize(size, Image.ANTIALIAS)

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
def save_base64_image(image: str, path: str = "image.png", size: Optional[Tuple[int, int]] = None, quality: Optional[int] = None) -> None:
    """
    Save a base64 encoded image to a file, optionally resizing and compressing it.

    :param image: The base64 encoded image.
    :param path: The file path to save the image.
    :param size: A tuple (width, height) specifying the new size. If None, the original size is used.
    :param quality: The quality level (1-100) for image compression. If None, no compression is applied.
    """
    image = Image.open(BytesIO(base64.b64decode(image)))
    
    if size:
        image = resize_image(image, size)
    
    if quality:
        image = Image.open(compress_image(image, quality))
    
    image.save(path)
