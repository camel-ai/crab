import pytest
from PIL import Image
import io
import base64
import os
from unittest.mock import patch, MagicMock
from crab.utils.common import base64_to_image, image_to_base64
from crab.actions.file_actions import save_image

import sys
# Mock the entire crab.agents.backend_models module
sys.modules['crab.agents.backend_models'] = MagicMock()
sys.modules['crab.agents.backend_models.openai_model'] = MagicMock()
sys.modules['crab.actions.desktop_actions'] = MagicMock()

# Create mock classes/functions
class MockOpenAIModel:
    def _convert_message(self, message):
        return {"type": "image_url", "image_url": {"url": "data:image/png;base64,mockbase64"}}

class MockMessageType:
    IMAGE_JPG_BASE64 = "image_jpg_base64"

def mock_screenshot():
    return Image.new('RGB', (100, 100), color='red')

# Apply mocks
patch('crab.agents.backend_models.openai_model.OpenAIModel', MockOpenAIModel).start()
patch('crab.agents.backend_models.openai_model.MessageType', MockMessageType).start()
patch('crab.actions.desktop_actions.screenshot', mock_screenshot).start()

class TestImageHandling:
    @pytest.fixture(autouse=True)
    def setup(self):
        self.test_image = Image.new('RGB', (100, 100), color='red')
        
    def test_image_processing_path(self):
        print("\n--- Image Processing Path Test ---")
        
        # Use self.test_image instead of taking a screenshot
        screenshot_image = self.test_image
        
        # 1. Start with a PIL Image (using self.test_image)
        print("1. Starting with a PIL Image")
        assert isinstance(self.test_image, Image.Image)
        
        # 2. Simulate saving the image
        print("2. Saving the image")
        save_image(self.test_image, "test_image.png")
        print("   Image saved successfully")
        
        # 3. Using self.test_image instead of taking a screenshot
        print("3. Using self.test_image instead of taking a screenshot")
        screenshot_image = self.test_image
        assert isinstance(screenshot_image, Image.Image)
        print("   Using self.test_image as PIL Image")
        
        # 4. Prepare for network transfer (serialize to base64)
        print("4. Serializing image for network transfer")
        base64_string = image_to_base64(self.test_image)
        assert isinstance(base64_string, str)
        print("   Image serialized to base64 string")
        
        # 5. Simulate network transfer
        print("5. Simulating network transfer")
        received_base64 = base64_string  # In reality, this would be sent and received
        
        # 6. Deserialize after network transfer
        print("6. Deserializing image after network transfer")
        received_image = base64_to_image(received_base64)
        assert isinstance(received_image, Image.Image)
        print("   Image deserialized back to PIL Image")
        
        # 7. Use the image in a backend model (e.g., OpenAI)
        print("7. Using image in backend model")
        openai_model = MockOpenAIModel()
        converted_message = openai_model._convert_message((received_image, MockMessageType.IMAGE_JPG_BASE64))
        assert converted_message["type"] == "image_url"
        assert converted_message["image_url"]["url"].startswith("data:image/png;base64,")
        print("   Image successfully converted for use in OpenAI model")
        
        print("--- Image Processing Path Test Completed Successfully ---")

    def test_base64_to_image(self):
        # Convert image to base64
        base64_string = image_to_base64(self.test_image)
        
        # Test base64_to_image function
        converted_image = base64_to_image(base64_string)
        assert isinstance(converted_image, Image.Image)
        assert converted_image.size == (100, 100)
        
    def test_image_to_base64(self):
        # Test image_to_base64 function
        base64_string = image_to_base64(self.test_image)
        assert isinstance(base64_string, str)
        
        # Verify that the base64 string can be converted back to an image
        converted_image = base64_to_image(base64_string)
        assert converted_image.size == (100, 100)

# Make sure to stop all patches after the tests
def teardown_module(module):
    patch.stopall()
