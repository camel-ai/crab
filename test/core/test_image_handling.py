import pytest
from typing import Any
from PIL import Image
import io
import base64
from crab.utils.common import base64_to_image, image_to_base64
from crab.actions.file_actions import save_image  # Comment out this import
from PIL import Image
from pydantic import GetCoreSchemaHandler
from pydantic_core import core_schema

class PILImageType(Image.Image):
    @classmethod
    def __get_pydantic_core_schema__(
        cls, source_type: Any, handler: GetCoreSchemaHandler
    ) -> core_schema.CoreSchema:
        return core_schema.any_schema()

class TestImageHandling:
    @pytest.fixture(autouse=True)
    def setup(self):
        # Create a sample image for testing
        self.test_image: PILImageType = Image.new('RGB', (100, 100), color='red')
        self.test_image_base64 = image_to_base64(self.test_image)
        
    def test_base64_to_image(self):
        # Convert image to base64
        buffered = io.BytesIO()
        self.test_image.save(buffered, format="PNG")
        img_str = base64.b64encode(buffered.getvalue()).decode()
        
        # Test base64_to_image function
        converted_image = base64_to_image(img_str)
        assert isinstance(converted_image, Image.Image)
        assert converted_image.size == (100, 100)
        
    def test_image_to_base64(self):
        # Test image_to_base64 function
        base64_string = image_to_base64(self.test_image)
        assert isinstance(base64_string, str)
        
        # Verify that the base64 string can be converted back to an image
        converted_image = base64_to_image(base64_string)
        assert converted_image.size == (100, 100)
        
    def test_save_image(self, tmp_path):
        # Test save_image function
        test_path = tmp_path / "test_image.png"
        save_image(self.test_image_base64, str(test_path))
        
        # Verify that the image was saved correctly
        saved_image = Image.open(test_path)
        assert saved_image.size == (100, 100)
