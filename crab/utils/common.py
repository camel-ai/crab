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
from typing import Callable
import logging
import gc

import dill
from PIL import Image

logger = logging.getLogger(__name__)

def base64_to_image(encoded: str) -> Image.Image:
    try:
        logger.info("Converting base64 to image")
        start_size = len(encoded)
        logger.info(f"Input base64 size: {start_size / 1024 / 1024:.2f} MB")
        
        # Decode base64
        img_data = base64.b64decode(encoded)
        logger.info(f"Decoded data size: {len(img_data) / 1024 / 1024:.2f} MB")
        
        # Create image from bytes
        img = Image.open(BytesIO(img_data))
        logger.info(f"Created image: {img.size}x{img.mode}")
        
        # Force load image data
        img.load()
        logger.info("Image data loaded into memory")
        
        # Clean up
        del img_data
        gc.collect()
        
        return img
    except Exception as e:
        logger.error(f"Error converting base64 to image: {str(e)}")
        raise


def image_to_base64(image: Image.Image) -> str:
    try:
        logger.info(f"Converting image to base64: {image.size}x{image.mode}")
        
        # Use BytesIO to save image
        img_byte_arr = BytesIO()
        image.save(img_byte_arr, format="PNG", optimize=True)
        img_data = img_byte_arr.getvalue()
        logger.info(f"PNG size: {len(img_data) / 1024 / 1024:.2f} MB")
        
        # Convert to base64
        encoded = base64.b64encode(img_data).decode("utf-8")
        logger.info(f"Base64 size: {len(encoded) / 1024 / 1024:.2f} MB")
        
        # Clean up
        img_byte_arr.close()
        del img_data
        gc.collect()
        
        return encoded
    except Exception as e:
        logger.error(f"Error converting image to base64: {str(e)}")
        raise


def callable_to_base64(func: Callable) -> str:
    return base64.b64encode(dill.dumps(func, recurse=True)).decode("utf-8")


def base64_to_callable(encoded: str) -> Callable:
    return dill.loads(base64.b64decode(encoded))


def json_expand_refs(schema: dict | list, defs: dict | None = None):
    """Recursively expand `$ref` and `allOf` in the JSON.

    This function walks through the schema object, replacing any `$ref` with its
    corresponding definition found in `$defs`. It also expands subschemas defined in
    `allOf` by merging their resolved definitions into a single schema.

    Args:
        schema: The JSON schema (or sub-schema).
        defs: The collection of definitions for `$ref` expansion. If None, it will look
            for `$defs` at the root of the schema.

    Returns:
        The schema with all `$ref` and `allOf` expanded.

    Raises:
        ValueError: If a reference cannot be resolved with the provided `$defs`.
    """
    # If defs is None, it means we're at the root of the schema
    if defs is None:
        defs = schema.pop("$defs", {})

    if isinstance(schema, dict):
        # Process `$ref` by replacing it with the referenced definition
        if "$ref" in schema:
            ref_path = schema["$ref"].split("/")
            ref_name = ref_path[-1]
            if ref_name in defs:
                return json_expand_refs(defs[ref_name], defs)
            else:
                raise ValueError(f"Reference {schema['$ref']} not found in $defs.")

        # Process `allOf` by combining all subschemas
        elif "allOf" in schema:
            combined_schema = {}
            for subschema in schema["allOf"]:
                expanded_subschema = json_expand_refs(subschema, defs)
                # Merge the expanded subschema into the combined_schema
                for key, value in expanded_subschema.items():
                    combined_schema[key] = value
            return combined_schema

        # Recursively process all keys in the dictionary
        else:
            return {key: json_expand_refs(value, defs) for key, value in schema.items()}

    elif isinstance(schema, list):
        # Recursively process each item in the list
        return [json_expand_refs(item, defs) for item in schema]

    # If it's neither a dict nor a list, return it as is (e.g., int, str)
    return schema
