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
from PIL import Image
from pydantic import Field
from crab.core.decorators import action
from crab.utils.common import base64_to_image

@action
def save_image(image: str = Field(..., description="Base64 encoded image string"), path: str = Field(..., description="Path to save the image")):
    """Save a base64 encoded image to a file."""
    img = base64_to_image(image)
    img.save(path)
    return f"Image saved to {path}"
