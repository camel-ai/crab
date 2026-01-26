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

import unittest
from io import BytesIO
from PIL import Image
import base64
import os
import traceback

from crab.actions.file_actions import save_base64_image
from crab.core.models import Action

class TestSaveBase64Image(unittest.TestCase):

    # 设置类变量
    RESIZE_RATIO = 0.2
    COMPRESSION_QUALITY = 20

    def setUp(self):
        # 创建 test_images 文件夹（如果不存在）
        self.test_images_dir = "test_images"
        os.makedirs(self.test_images_dir, exist_ok=True)

        # 使用一个具体的图片文件
        image_path = "test_images/test_image.png"  # 请替换为实际的图片路径
        with open(image_path, "rb") as image_file:
            self.image_base64 = base64.b64encode(image_file.read()).decode('utf-8')

    def tearDown(self):
        # 不删除生成的文件，以便查看结果
        pass

    def run_test_with_error_logging(self, test_method):
        try:
            print(f"Current working directory: {os.getcwd()}")
            test_method()
        except Exception as e:
            print(f"Error in {test_method.__name__}:")
            print(traceback.format_exc())
            raise

    def _test_save_image_without_resize_or_compression(self):
        output_path = os.path.join(self.test_images_dir, "test_image_original.png")
        try:
            print("Calling save_base64_image...")
            action_obj = save_base64_image(self.image_base64, path=output_path)
            print(f"save_base64_image returned: {action_obj}")
            
            if isinstance(action_obj, Action):
                result = action_obj.run()
                print(f"Action execution result: {result}")
            else:
                print("Unexpected return type from save_base64_image")
        except Exception as e:
            print(f"Exception: {e}")
            raise
        
        print(f"Checking if file exists: {os.path.abspath(output_path)}")
        if os.path.exists(output_path):
            print("File exists")
            file_size = os.path.getsize(output_path)
            print(f"File size: {file_size} bytes")
        else:
            print("File does not exist")
        
        self.assertTrue(os.path.exists(output_path), f"File {output_path} was not created")
        with Image.open(output_path) as saved_image:
            print(f"Original image size: {saved_image.size}")
            print(f"Original image format: {saved_image.format}")

    def _test_save_image_with_resize(self):
        output_path = os.path.join(self.test_images_dir, "test_resized_image.png")
        action_obj = save_base64_image(self.image_base64, path=output_path, ratio=self.RESIZE_RATIO)
        if isinstance(action_obj, Action):
            action_obj.run()
        self.assertTrue(os.path.exists(output_path), f"File {output_path} was not created")
        with Image.open(output_path) as saved_image:
            print(f"Resized image size: {saved_image.size}")
            print(f"Resized image format: {saved_image.format}")

    def _test_save_image_with_compression(self):
        output_path = os.path.join(self.test_images_dir, "test_compressed_image.jpg")
        action_obj = save_base64_image(self.image_base64, path=output_path, quality=self.COMPRESSION_QUALITY)
        if isinstance(action_obj, Action):
            action_obj.run()
        self.assertTrue(os.path.exists(output_path), f"File {output_path} was not created")
        with Image.open(output_path) as saved_image:
            print(f"Compressed image size: {saved_image.size}")
            print(f"Compressed image format: {saved_image.format}")

    def _test_save_image_with_resize_and_compression(self):
        output_path = os.path.join(self.test_images_dir, "test_resized_compressed_image.jpg")
        action_obj = save_base64_image(self.image_base64, path=output_path, ratio=self.RESIZE_RATIO, quality=self.COMPRESSION_QUALITY)
        if isinstance(action_obj, Action):
            action_obj.run()
        self.assertTrue(os.path.exists(output_path), f"File {output_path} was not created")
        with Image.open(output_path) as saved_image:
            print(f"Resized and compressed image size: {saved_image.size}")
            print(f"Resized and compressed image format: {saved_image.format}")

    def test_save_image_without_resize_or_compression(self):
        self.run_test_with_error_logging(self._test_save_image_without_resize_or_compression)

    def test_save_image_with_resize(self):
        self.run_test_with_error_logging(self._test_save_image_with_resize)

    def test_save_image_with_compression(self):
        self.run_test_with_error_logging(self._test_save_image_with_compression)

    def test_save_image_with_resize_and_compression(self):
        self.run_test_with_error_logging(self._test_save_image_with_resize_and_compression)

if __name__ == '__main__':
    unittest.main()
