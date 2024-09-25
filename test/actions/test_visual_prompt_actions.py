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
from pathlib import Path

import pytest
import requests
from PIL import Image

from crab.actions.visual_prompt_actions import (
    get_groundingdino_boxes,
    groundingdino_easyocr,
)
from crab.utils import image_to_base64


@pytest.mark.skip(reason="Too slow")
def test_get_groundingdino_boxes_single_image():
    url = "http://images.cocodataset.org/val2017/000000039769.jpg"
    image = Image.open(requests.get(url, stream=True).raw)
    text = "a cat."

    box_threshold = 0.4
    text_threshold = 0.3
    result = get_groundingdino_boxes(image, text, box_threshold, text_threshold)
    assert len(result) == 1
    assert len(result[0]) > 0
    assert len(result[0][0]) == 2


@pytest.mark.skip(reason="Too slow")
def test_get_groundingdino_boxes_multi_image():
    url1 = "http://images.cocodataset.org/val2017/000000039769.jpg"
    url2 = "https://farm5.staticflickr.com/4005/4666183752_c5b79faa17_z.jpg"
    image1 = Image.open(requests.get(url1, stream=True).raw)
    image2 = Image.open(requests.get(url2, stream=True).raw)
    text = "a cat. a car."

    box_threshold = 0.4
    text_threshold = 0.3
    result = get_groundingdino_boxes(
        [image1, image2], text, box_threshold, text_threshold
    )
    assert len(result) == 2
    assert len(result[0]) > 0
    assert len(result[1]) > 0
    assert len(result[0][0]) == 2


@pytest.mark.skip(reason="Too slow")
@pytest.mark.parametrize(
    "image_name", ["ubuntu_screenshot.png", "android_screenshot.png"]
)
def test_groundingdino_easy_ocr(image_name: str):
    class A:
        pass

    temp = A()

    test_dir = Path(__file__).parent.parent
    image_path = test_dir / "_assets" / image_name
    image = Image.open(image_path)
    image_base64 = image_to_base64(image)
    visual_prompt = groundingdino_easyocr(font_size=40).set_kept_param(env=temp)
    result_image, boxes = visual_prompt.run(input_base64_image=image_base64)
    assert result_image != image_base64
    assert boxes
