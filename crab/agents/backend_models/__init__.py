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
# ruff: noqa: F401
from typing import Any, Literal

from pydantic import BaseModel

from crab.core.backend_model import BackendModel

from .camel_model import CamelModel
from .claude_model import ClaudeModel
from .gemini_model import GeminiModel
from .openai_model import OpenAIModel
from .vllm_model import VLLMModel


class BackendModelConfig(BaseModel):
    model_class: Literal["openai", "claude", "gemini", "camel", "vllm"]
    model_name: str
    history_messages_len: int = 0
    parameters: dict[str, Any] = {}
    tool_call_required: bool = False
    base_url: str | None = None  # Only used in OpenAIModel and VLLMModel currently
    api_key: str | None = None  # Only used in OpenAIModel and VLLMModel currently


def create_backend_model(model_config: BackendModelConfig) -> BackendModel:
    match model_config.model_class:
        case "claude":
            if model_config.base_url is not None or model_config.api_key is not None:
                raise Warning(
                    "base_url and api_key are not supported for ClaudeModel currently."
                )
            return ClaudeModel(
                model=model_config.model_name,
                parameters=model_config.parameters,
                history_messages_len=model_config.history_messages_len,
            )
        case "gemini":
            if model_config.base_url is not None or model_config.api_key is not None:
                raise Warning(
                    "base_url and api_key are not supported for GeminiModel currently."
                )
            return GeminiModel(
                model=model_config.model_name,
                parameters=model_config.parameters,
                history_messages_len=model_config.history_messages_len,
            )
        case "openai":
            return OpenAIModel(
                model=model_config.model_name,
                parameters=model_config.parameters,
                history_messages_len=model_config.history_messages_len,
                base_url=model_config.base_url,
                api_key=model_config.api_key,
            )
        case "vllm":
            return VLLMModel(
                model=model_config.model_name,
                parameters=model_config.parameters,
                history_messages_len=model_config.history_messages_len,
                base_url=model_config.base_url,
                api_key=model_config.api_key,
            )
        case "camel":
            raise NotImplementedError("Cannot support camel model currently.")
        case _:
            raise ValueError(f"Unsupported model name: {model_config.model_name}")
