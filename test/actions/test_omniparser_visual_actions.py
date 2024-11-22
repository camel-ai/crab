# =========== Copyright 2024 @ CAMEL-AI.org. All Rights Reserved. ===========
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# =========== Copyright 2024 @ CAMEL-AI.org. All Rights Reserved. ===========
import os
import logging
from pathlib import Path

import pytest
from PIL import Image

from crab.actions.omniparser_visual_actions import (
    detect_gui_elements,
    detect_and_annotate_gui_elements,
    get_ocr_boxes,
    get_elements_prompt,
)
from crab.utils import image_to_base64

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configure environment variables for PaddleOCR
os.environ['FLAGS_mark_not_exist_as_used'] = 'true'
os.environ['FLAGS_minloglevel'] = '2'
os.environ['MKLDNN_CACHE_CAPACITY'] = '0'
os.environ['FLAGS_allocator_strategy'] = 'naive_best_fit'

@pytest.fixture
def test_image_path():
    """Fixture to provide test image path."""
    test_dir = Path(__file__).parent.parent
    return test_dir / "_assets" / "ubuntu_screenshot.png"

@pytest.fixture
def test_image(test_image_path):
    """Fixture to provide test image as PIL Image."""
    return Image.open(test_image_path)

@pytest.fixture
def test_image_base64(test_image):
    """Fixture to provide test image as base64 string."""
    return image_to_base64(test_image)

@pytest.fixture
def mock_env():
    """Fixture to provide mock environment object."""
    class MockEnv:
        def __init__(self):
            self.data = {}
    return MockEnv()

def test_detect_gui_elements(test_image):
    """Test basic GUI element detection functionality."""
    # Test with PIL Image input
    results = detect_gui_elements(test_image)
    assert isinstance(results, list)
    assert len(results) == 1
    assert isinstance(results[0], list)
    
    # Test with base64 input
    base64_image = image_to_base64(test_image)
    results = detect_gui_elements(base64_image)
    assert isinstance(results, list)
    assert len(results) == 1
    
    # Verify box format
    for detection in results[0]:
        assert len(detection) == 2
        box, description = detection
        assert len(box) == 4
        assert isinstance(description, str)

def test_get_ocr_boxes(test_image):
    """Test OCR box detection with both PaddleOCR and EasyOCR."""
    # Test with EasyOCR only since PaddleOCR requires specific environment setup
    boxes = get_ocr_boxes(test_image, use_paddle=False)
    assert isinstance(boxes, list)
    for box, text in boxes:
        assert len(box) == 4
        assert isinstance(text, str)

@pytest.mark.parametrize(
    "image_name", ["ubuntu_screenshot.png", "android_screenshot.png"]
)
def test_detect_and_annotate_gui_elements(image_name, mock_env):
    """Test GUI element detection and annotation with different screenshots."""
    logger = logging.getLogger(__name__)
    logger.info(f"\n\n=== Starting test for {image_name} ===")
    
    test_dir = Path(__file__).parent.parent
    image_path = test_dir / "_assets" / image_name
    logger.info(f"Loading test image from: {image_path}")
    
    image = Image.open(image_path)
    logger.info(f"Original image size: {image.size}")
    
    image_base64 = image_to_base64(image)
    logger.info(f"Converted image to base64 (length: {len(image_base64)})")

    # Create action instance and run it
    action = detect_and_annotate_gui_elements.set_kept_param(env=mock_env)
    logger.info("Created action instance")
    
    result = action.run(
        image=image_base64,
        font_size=12,
        use_paddle=False  # Use EasyOCR for testing
    )
    logger.info("Action completed")

    # Validate result structure
    assert isinstance(result, dict), f"Expected dict result, got {type(result)}"
    assert "image" in result, "Result missing 'image' key"
    assert "boxes" in result, "Result missing 'boxes' key"
    assert isinstance(result["boxes"], list), f"Expected list of boxes, got {type(result['boxes'])}"
    
    # Log result details
    logger.info(f"Result contains {len(result['boxes'])} boxes")
    logger.info(f"Result image base64 length: {len(result['image'])}")
    
    # Check if images are different
    if result["image"] == image_base64:
        logger.error("Result image is identical to input image!")
        logger.error("First 100 chars of input image: " + image_base64[:100])
        logger.error("First 100 chars of result image: " + result["image"][:100])
    
    assert result["image"] != image_base64  # Should be different due to annotations

    # Validate box format
    for box, label in result["boxes"]:
        logger.debug(f"Box: {box}, Label: {label}")
        assert len(box) == 4, f"Invalid box format: {box}"
        assert isinstance(label, str), f"Invalid label type: {type(label)}"

@pytest.mark.parametrize(
    "image_name", ["ubuntu_screenshot.png", "android_screenshot.png"]
)
def test_get_elements_prompt(image_name, mock_env):
    """Test generation of element prompts from detected GUI elements."""
    test_dir = Path(__file__).parent.parent
    image_path = test_dir / "_assets" / image_name
    image = Image.open(image_path)
    image_base64 = image_to_base64(image)

    # First detect and annotate elements
    action = detect_and_annotate_gui_elements.set_kept_param(env=mock_env)
    detect_result = action.run(
        image=image_base64,
        font_size=12,
        use_paddle=False  # Use EasyOCR for testing
    )
    
    # Then generate prompt
    action = get_elements_prompt.set_kept_param(env=mock_env)
    result = action.run(
        image=detect_result["image"],
        env=mock_env
    )
    
    assert isinstance(result, tuple)
    assert len(result) == 2
    assert isinstance(result[0], str)  # base64 image
    assert isinstance(result[1], str)  # prompt text
    assert len(result[1]) > 0  # prompt should not be empty

def test_error_handling():
    """Test error handling for invalid inputs."""
    # Test with invalid image data
    results = detect_gui_elements("invalid_base64_string")
    assert results == [[]]  # Should return empty list for invalid input

    # Test with empty image
    empty_image = Image.new('RGB', (1, 1))
    results = detect_gui_elements(empty_image)
    assert len(results[0]) == 0  # Should return empty detection list
