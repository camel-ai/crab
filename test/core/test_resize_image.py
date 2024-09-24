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

import unittest
from io import BytesIO
from PIL import Image
import base64
import os
import traceback

from crab.actions.file_actions import save_base64_image
from crab.core.models import Action

class TestSaveBase64Image(unittest.TestCase):

    def setUp(self):
        # Create a sample image for testing
        self.image = Image.new('RGB', (1000, 500), color='red')
        buffered = BytesIO()
        self.image.save(buffered, format="PNG")
        self.image_base64 = base64.b64encode(buffered.getvalue()).decode('utf-8')

    def tearDown(self):
        # Clean up the files created during tests
        files = ["test_image.png", "test_resized_image.png", "test_compressed_image.jpg", "test_resized_compressed_image.jpg"]
        for file in files:
            if os.path.exists(file):
                os.remove(file)

    def run_test_with_error_logging(self, test_method):
        try:
            print(f"Current working directory: {os.getcwd()}")  # 打印当前工作目录
            test_method()
        except Exception as e:
            print(f"Error in {test_method.__name__}:")
            print(traceback.format_exc())
            raise  # Re-raise the exception to make the test fail

    def _test_save_image_without_resize_or_compression(self):
        try:
            print("Calling save_base64_image...")
            action_obj = save_base64_image(self.image_base64, path="test_image.png")
            print(f"save_base64_image returned: {action_obj}")
            
            # 执行 Action 对象
            if isinstance(action_obj, Action):
                result = action_obj.run()
                print(f"Action execution result: {result}")
            else:
                print("Unexpected return type from save_base64_image")
            
            # 直接调用未装饰的函数
            print("Calling undecorated save_base64_image...")
            save_base64_image.__wrapped__(self.image_base64, path="test_image_undecorated.png")
            print("Undecorated function called successfully")
        except Exception as e:
            print(f"Exception: {e}")
            raise
        
        print(f"Checking if file exists: {os.path.abspath('test_image.png')}")
        if os.path.exists("test_image.png"):
            print("File exists")
            file_size = os.path.getsize("test_image.png")
            print(f"File size: {file_size} bytes")
        else:
            print("File does not exist")
        
        print(f"Checking if undecorated file exists: {os.path.abspath('test_image_undecorated.png')}")
        if os.path.exists("test_image_undecorated.png"):
            print("Undecorated file exists")
            file_size = os.path.getsize("test_image_undecorated.png")
            print(f"Undecorated file size: {file_size} bytes")
        else:
            print("Undecorated file does not exist")
        
        self.assertTrue(os.path.exists("test_image.png"), "File test_image.png was not created")
        self.assertTrue(os.path.exists("test_image_undecorated.png"), "File test_image_undecorated.png was not created")
        saved_image = Image.open("test_image.png")
        self.assertEqual(saved_image.size, (1000, 500))
        self.assertEqual(saved_image.format, "PNG")

    def test_save_image_with_resize(self):
        self.run_test_with_error_logging(self._test_save_image_with_resize)

    def _test_save_image_with_resize(self):
        action_obj = save_base64_image(self.image_base64, path="test_resized_image.png", ratio=0.5)
        if isinstance(action_obj, Action):
            action_obj.run()
        self.assertTrue(os.path.exists("test_resized_image.png"), "File test_resized_image.png was not created")
        with Image.open("test_resized_image.png") as saved_image:
            self.assertEqual(saved_image.size, (500, 250))
            self.assertEqual(saved_image.format, "PNG")

    def test_save_image_with_compression(self):
        self.run_test_with_error_logging(self._test_save_image_with_compression)

    def _test_save_image_with_compression(self):
        action_obj = save_base64_image(self.image_base64, path="test_compressed_image.jpg", quality=50)
        if isinstance(action_obj, Action):
            action_obj.run()
        self.assertTrue(os.path.exists("test_compressed_image.jpg"), "File test_compressed_image.jpg was not created")
        with Image.open("test_compressed_image.jpg") as saved_image:
            self.assertEqual(saved_image.format, "JPEG")

    def test_save_image_with_resize_and_compression(self):
        self.run_test_with_error_logging(self._test_save_image_with_resize_and_compression)

    def _test_save_image_with_resize_and_compression(self):
        action_obj = save_base64_image(self.image_base64, path="test_resized_compressed_image.jpg", ratio=0.5, quality=50)
        if isinstance(action_obj, Action):
            action_obj.run()
        self.assertTrue(os.path.exists("test_resized_compressed_image.jpg"), "File test_resized_compressed_image.jpg was not created")
        with Image.open("test_resized_compressed_image.jpg") as saved_image:
            self.assertEqual(saved_image.size, (500, 250))
            self.assertEqual(saved_image.format, "JPEG")

if __name__ == '__main__':
    unittest.main()
