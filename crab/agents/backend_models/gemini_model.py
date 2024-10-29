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
import os
from typing import Any

from PIL.Image import Image
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_fixed

from crab import Action, ActionOutput, BackendModel, BackendOutput, Message, MessageType
from crab.utils.common import base64_to_image, json_expand_refs

try:
    import google.generativeai as genai
    from google.ai.generativelanguage_v1beta import (
        Content,
        FunctionDeclaration,
        Part,
        Tool,
    )
    from google.api_core.exceptions import ResourceExhausted
    from google.generativeai.types import content_types

    gemini_model_enable = True
except ImportError:
    gemini_model_enable = False


class GeminiModel(BackendModel):
    def __init__(
        self,
        model: str,
        parameters: dict[str, Any] | None = None,
        history_messages_len: int = 0,
        tool_call_required: bool = True,
    ) -> None:
        if gemini_model_enable is False:
            raise ImportError("Please install google.generativeai to use GeminiModel")

        self.model = model
        self.parameters = parameters if parameters is not None else {}
        self.history_messages_len = history_messages_len
        assert self.history_messages_len >= 0
        genai.configure(api_key=os.environ["GEMINI_API_KEY"])
        self.client = genai
        self.tool_call_required = tool_call_required
        self.system_message: str = "You are a helpful assistant."
        self.action_space: list[Action] | None = None
        self.action_schema: list[Tool] | None = None
        self.token_usage: int = 0
        self.chat_history: list[list[dict]] = []
        self.support_tool_call = True

    def reset(self, system_message: str, action_space: list[Action] | None) -> None:
        self.system_message = system_message
        self.action_space = action_space
        self.action_schema = _convert_action_to_schema(self.action_space)
        self.token_usage = 0
        self.chat_history = []

    def chat(self, message: list[Message] | Message) -> BackendOutput:
        if isinstance(message, tuple):
            message = [message]
        request = self._fetch_from_memory()
        new_message = self._construct_new_message(message)
        request.append(new_message)
        response_message = self._call_api(request)
        self._record_message(new_message, response_message)
        return self._generate_backend_output(response_message)

    def _construct_new_message(self, message: list[Message]) -> dict[str, Any]:
        parts: list[str | Image] = []
        for content, msg_type in message:
            match msg_type:
                case MessageType.TEXT:
                    parts.append(content)
                case MessageType.IMAGE_JPG_BASE64:
                    parts.append(base64_to_image(content))
        return {
            "role": "user",
            "parts": parts,
        }

    def _generate_backend_output(self, response_message: Content) -> BackendOutput:
        tool_calls: list[ActionOutput] = []
        for part in response_message.parts:
            if "function_call" in Part.to_dict(part):
                call = Part.to_dict(part)["function_call"]
                tool_calls.append(
                    ActionOutput(
                        name=call["name"],
                        arguments=call["args"],
                    )
                )

        return BackendOutput(
            message=response_message.parts[0].text or None,
            action_list=tool_calls or None,
        )

    def _fetch_from_memory(self) -> list[dict]:
        request: list[dict] = []
        if self.history_messages_len > 0:
            fetch_history_len = min(self.history_messages_len, len(self.chat_history))
            for history_message in self.chat_history[-fetch_history_len:]:
                request = request + history_message
        return request

    def get_token_usage(self):
        return self.token_usage

    def _record_message(
        self, new_message: dict[str, Any], response_message: Content
    ) -> None:
        self.chat_history.append([new_message])
        self.chat_history[-1].append(
            {"role": response_message.role, "parts": response_message.parts}
        )

    @retry(
        wait=wait_fixed(10),
        stop=stop_after_attempt(7),
        retry=retry_if_exception_type(ResourceExhausted),
    )
    def _call_api(self, request_messages: list) -> Content:
        if self.action_schema is not None:
            tool_config = content_types.to_tool_config(
                {
                    "function_calling_config": {
                        "mode": "ANY" if self.tool_call_required else "AUTO"
                    }
                }
            )
            response = self.client.GenerativeModel(
                self.model, system_instruction=self.system_message
            ).generate_content(
                contents=request_messages,
                tools=self.action_schema,
                tool_config=tool_config,
                # **self.parameters, # TODO(Tianqi): Fix this line in the future
            )
        else:
            response = self.client.GenerativeModel(
                self.model, system_instruction=self.system_message
            ).generate_content(
                contents=request_messages,
                # **self.parameters, # TODO(Tianqi): Fix this line in the future
            )

        self.token_usage += response.candidates[0].token_count
        return response.candidates[0].content


def _convert_action_to_schema(action_space: list[Action] | None) -> list[Tool] | None:
    if action_space is None:
        return None
    actions = [
        Tool(
            function_declarations=[
                _action_to_func_dec(action) for action in action_space
            ]
        )
    ]
    return actions


def _clear_schema(schema_dict: dict) -> None:
    schema_dict.pop("title", None)
    p_type = schema_dict.pop("type", None)
    for prop in schema_dict.get("properties", {}).values():
        _clear_schema(prop)
    if p_type is not None:
        schema_dict["type_"] = p_type.upper()
    if "items" in schema_dict:
        _clear_schema(schema_dict["items"])


def _action_to_func_dec(action: Action) -> FunctionDeclaration:
    "Converts crab Action to google FunctionDeclaration"
    p_schema = action.parameters.model_json_schema()
    if "$defs" in p_schema:
        p_schema = json_expand_refs(p_schema)
    _clear_schema(p_schema)
    if not p_schema["properties"]:
        return FunctionDeclaration(
            name=action.name,
            description=action.description,
        )
    return FunctionDeclaration(
        name=action.name,
        description=action.description,
        parameters=p_schema,
    )
