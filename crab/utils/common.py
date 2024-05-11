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

import dill
from PIL import Image


def base64_to_image(encoded: str) -> Image.Image:
    return Image.open(BytesIO(base64.b64decode(encoded)))


def image_to_base64(image: Image.Image) -> str:
    img_byte_arr = BytesIO()
    image.save(img_byte_arr, format="png")
    return base64.b64encode(img_byte_arr.getvalue()).decode("utf-8")


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
