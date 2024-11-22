"""Compare the capabilities of old and new visual action implementations."""

import logging
import time
from pathlib import Path
import pytest
from PIL import Image

from crab.utils.common import image_to_base64
from crab.actions.visual_prompt_actions import (
    groundingdino_easyocr,
    get_elements_prompt as get_elements_prompt_old,
)
from crab.actions.omniparser_visual_actions import (
    detect_and_annotate_gui_elements,
    get_elements_prompt as get_elements_prompt_new,
)

logger = logging.getLogger(__name__)

# Test images
TEST_IMAGES = ["ubuntu_screenshot.png", "android_screenshot.png"]

@pytest.fixture(scope="module")
def test_images():
    """Fixture to provide test images."""
    test_dir = Path(__file__).parent.parent
    images = {}
    for image_name in TEST_IMAGES:
        image_path = test_dir / "_assets" / image_name
        image = Image.open(image_path)
        images[image_name] = image_to_base64(image)
    return images

@pytest.fixture(scope="function")
def mock_env():
    """Fixture to provide mock environment object."""
    class MockEnv:
        def __init__(self):
            self.element_label_map = []
            self.element_position_map = []
    return MockEnv()

def test_detection_speed(test_images, mock_env):
    """Compare detection speed between old and new implementations."""
    for image_name, image_base64 in test_images.items():
        logger.info(f"\nTesting detection speed for {image_name}")
        
        # Test old implementation
        start_time = time.time()
        old_result = groundingdino_easyocr(image_base64, font_size=12, env=mock_env)
        old_time = time.time() - start_time
        logger.info(f"Old implementation time: {old_time:.2f}s")
        
        # Test new implementation
        start_time = time.time()
        action = detect_and_annotate_gui_elements.set_kept_param(env=mock_env)
        new_result = action.run(
            image=image_base64,
            font_size=12,
            use_paddle=False  # Use EasyOCR for fair comparison
        )
        new_time = time.time() - start_time
        logger.info(f"New implementation time: {new_time:.2f}s")
        
        # Log timing details
        logger.info(f"Old implementation time: {old_time:.2f}s")
        logger.info(f"New implementation time: {new_time:.2f}s")
        logger.info(f"Speed difference: {old_time - new_time:.2f}s")

def test_detection_quality(test_images, mock_env):
    """Compare detection quality between old and new implementations."""
    for image_name, image_base64 in test_images.items():
        logger.info(f"\nTesting detection quality for {image_name}")
        
        # Get results from both implementations
        old_result = groundingdino_easyocr(image_base64, font_size=12, env=mock_env)
        
        action = detect_and_annotate_gui_elements.set_kept_param(env=mock_env)
        new_result = action.run(
            image=image_base64,
            font_size=12,
            use_paddle=False
        )
        
        # Compare number of detected elements
        old_boxes = mock_env.element_position_map
        new_boxes = new_result["boxes"]
        logger.info(f"Old implementation detected {len(old_boxes)} elements")
        logger.info(f"New implementation detected {len(new_boxes)} elements")
        
        # Assert minimum detection count
        assert len(old_boxes) > 0, "Old implementation should detect some elements"
        assert len(new_boxes) > 0, "New implementation should detect some elements"

def test_prompt_generation(test_images, mock_env):
    """Compare prompt generation between old and new implementations."""
    for image_name, image_base64 in test_images.items():
        logger.info(f"\n{'='*80}")
        logger.info(f"Testing prompt generation for {image_name}")
        logger.info(f"{'='*80}")
        
        # Get results from both implementations
        old_result_action = groundingdino_easyocr(image_base64, font_size=12, env=mock_env)
        old_result_action = old_result_action.set_kept_param(env=mock_env)
        old_result = old_result_action.run()
        
        old_prompt_action = get_elements_prompt_old(old_result)
        old_prompt_action = old_prompt_action.set_kept_param(env=mock_env)
        old_prompt = old_prompt_action.run()
        
        action = detect_and_annotate_gui_elements.set_kept_param(env=mock_env)
        new_result = action.run(
            image=image_base64,
            font_size=12,
            use_paddle=False
        )
        new_prompt_action = get_elements_prompt_new(new_result["image"])
        new_prompt_action = new_prompt_action.set_kept_param(env=mock_env)
        new_prompt = new_prompt_action.run()
        
        # Check that prompts are non-empty strings
        assert isinstance(old_prompt, tuple), f"Old prompt is not a tuple: {type(old_prompt)}"
        assert isinstance(old_prompt[1], str), f"Old prompt text is not a string: {type(old_prompt[1])}"
        assert isinstance(new_prompt[1], str), f"New prompt text is not a string: {type(new_prompt[1])}"
        assert len(old_prompt[1]) > 0, "Old prompt is empty"
        assert len(new_prompt[1]) > 0, "New prompt is empty"
        
        # Log detailed comparison
        logger.info("\nOld Implementation (groundingdino_easyocr):")
        logger.info("-" * 50)
        logger.info("Full prompt:")
        for line in old_prompt[1].split('\n'):
            logger.info(line)
        
        logger.info("\nNew Implementation (detect_and_annotate_gui_elements):")
        logger.info("-" * 50)
        logger.info("Full prompt:")
        for line in new_prompt[1].split('\n'):
            logger.info(line)
        
        # Compare number of detected elements
        old_element_count = len([line for line in old_prompt[1].split('\n') if line.strip()])
        new_element_count = len([line for line in new_prompt[1].split('\n') if line.strip()])
        
        logger.info("\nComparison Summary:")
        logger.info("-" * 50)
        logger.info(f"Old implementation detected {old_element_count} elements")
        logger.info(f"New implementation detected {new_element_count} elements")
        logger.info(f"Old prompt length: {len(old_prompt[1])} characters")
        logger.info(f"New prompt length: {len(new_prompt[1])} characters")

def test_ocr_capabilities(test_images, mock_env):
    """Compare OCR capabilities between old and new implementations."""
    for image_name, image_base64 in test_images.items():
        logger.info(f"\nTesting OCR capabilities for {image_name}")
        
        # Test old implementation (EasyOCR only)
        old_result = groundingdino_easyocr(image_base64, font_size=12, env=mock_env)
        old_text_elements = [label for label in mock_env.element_label_map if isinstance(label, str)]
        
        # Test new implementation with EasyOCR
        action = detect_and_annotate_gui_elements.set_kept_param(env=mock_env)
        new_result_easyocr = action.run(
            image=image_base64,
            font_size=12,
            use_paddle=False
        )
        easyocr_text_elements = [label for box, label in new_result_easyocr["boxes"] if isinstance(label, str)]
        
        # Test new implementation with PaddleOCR
        new_result_paddle = action.run(
            image=image_base64,
            font_size=12,
            use_paddle=True
        )
        paddle_text_elements = [label for box, label in new_result_paddle["boxes"] if isinstance(label, str)]
        
        # Log detection details
        logger.info(f"Old implementation found {len(old_text_elements)} text elements:")
        for text in old_text_elements[:5]:
            logger.info(f"  - {text}")
        
        logger.info(f"New implementation (EasyOCR) found {len(easyocr_text_elements)} text elements:")
        for text in easyocr_text_elements[:5]:
            logger.info(f"  - {text}")
        
        logger.info(f"New implementation (PaddleOCR) found {len(paddle_text_elements)} text elements:")
        for text in paddle_text_elements[:5]:
            logger.info(f"  - {text}")
        
        # Assert that both implementations find text elements
        assert len(old_text_elements) > 0, "Old implementation should find text elements"
        assert len(easyocr_text_elements) > 0, "New implementation with EasyOCR should find text elements"
        assert len(paddle_text_elements) > 0, "New implementation with PaddleOCR should find text elements"
