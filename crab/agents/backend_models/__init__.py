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
from .camel_rag_model import CamelRAGModel
from .claude_model import ClaudeModel
from .gemini_model import GeminiModel
from .openai_model import OpenAIModel, OpenAIModelJSON, SGlangOpenAIModelJSON


class BackendModelConfig(BaseModel):
    model_class: Literal["openai", "claude", "gemini", "camel", "sglang"]
    """Specify the model class to be used. Different model classese use different
    APIs.
    """

    model_name: str
    """Specify the model name to be used. This value is directly passed to the API, 
    check model provider API documentation for more details.
    """

    model_platform: str | None = None
    """Required for CamelModel. Otherwise, it is ignored. Please check CAMEL
    documentation for more details.
    """

    history_messages_len: int = 0
    """Number of rounds of previous messages to be used in the model input. 0 means no
    history.
    """

    parameters: dict[str, Any] = {}
    """Additional parameters to be passed to the model."""

    json_structre_output: bool = False
    """If True, the model generate action through JSON without using "tool call" or
    "function call". SGLang model only supports JSON output. OpenAI model supports both.
    Other models do not support JSON output.
    """

    tool_call_required: bool = True
    """Specify if the model enforce each round to generate tool/function calls."""

    base_url: str | None = None
    """Specify the base URL of the API. Only used in OpenAI and SGLang currently."""

    api_key: str | None = None
    """Specify the API key to be used. Only used in OpenAI and SGLang currently."""


def create_backend_model(model_config: BackendModelConfig) -> BackendModel:
    match model_config.model_class:
        case "claude":
            if model_config.base_url is not None or model_config.api_key is not None:
                raise Warning(
                    "base_url and api_key are not supported for ClaudeModel currently."
                )
            if model_config.json_structre_output:
                raise Warning(
                    "json_structre_output is not supported for ClaudeModel currently."
                )
            return ClaudeModel(
                model=model_config.model_name,
                parameters=model_config.parameters,
                history_messages_len=model_config.history_messages_len,
                tool_call_required=model_config.tool_call_required,
            )
        case "gemini":
            if model_config.base_url is not None or model_config.api_key is not None:
                raise Warning(
                    "base_url and api_key are not supported for GeminiModel currently."
                )
            if model_config.json_structre_output:
                raise Warning(
                    "json_structre_output is not supported for GeminiModel currently."
                )
            return GeminiModel(
                model=model_config.model_name,
                parameters=model_config.parameters,
                history_messages_len=model_config.history_messages_len,
                tool_call_required=model_config.tool_call_required,
            )
        case "openai":
            if not model_config.json_structre_output:
                return OpenAIModel(
                    model=model_config.model_name,
                    parameters=model_config.parameters,
                    history_messages_len=model_config.history_messages_len,
                    base_url=model_config.base_url,
                    api_key=model_config.api_key,
                    tool_call_required=model_config.tool_call_required,
                )
            else:
                return OpenAIModelJSON(
                    model=model_config.model_name,
                    parameters=model_config.parameters,
                    history_messages_len=model_config.history_messages_len,
                    base_url=model_config.base_url,
                    api_key=model_config.api_key,
                )
        case "sglang":
            return SGlangOpenAIModelJSON(
                model=model_config.model_name,
                parameters=model_config.parameters,
                history_messages_len=model_config.history_messages_len,
                base_url=model_config.base_url,
                api_key=model_config.api_key,
            )
        case "camel":
            return CamelModel(
                model=model_config.model_name,
                model_platform=model_config.model_platform,
                parameters=model_config.parameters,
                history_messages_len=model_config.history_messages_len,
                tool_call_required=model_config.tool_call_required,
            )
        case _:
            raise ValueError(f"Unsupported model name: {model_config.model_name}")
