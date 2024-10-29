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
from copy import deepcopy
from typing import Any

from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_fixed

from crab import Action, ActionOutput, BackendModel, BackendOutput, Message, MessageType

try:
    import anthropic
    from anthropic.types import TextBlock, ToolUseBlock

    anthropic_model_enable = True
except ImportError:
    anthropic_model_enable = False


class ClaudeModel(BackendModel):
    def __init__(
        self,
        model: str,
        parameters: dict[str, Any] | None = None,
        history_messages_len: int = 0,
        tool_call_required: bool = True,
    ) -> None:
        if anthropic_model_enable is False:
            raise ImportError("Please install anthropic to use ClaudeModel")
        self.model = model
        self.parameters = parameters if parameters is not None else {}
        self.history_messages_len = history_messages_len

        assert self.history_messages_len >= 0

        self.client = anthropic.Anthropic()
        self.tool_call_required: bool = tool_call_required
        self.system_message: str = "You are a helpful assistant."
        self.action_space: list[Action] | None = None
        self.action_schema: list[dict] | None = None
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
        parts: list[dict] = []
        for content, msg_type in message:
            match msg_type:
                case MessageType.TEXT:
                    parts.append(
                        {
                            "type": "text",
                            "text": content,
                        }
                    )
                case MessageType.IMAGE_JPG_BASE64:
                    parts.append(
                        {
                            "type": "image",
                            "source": {
                                "data": content,
                                "type": "base64",
                                "media_type": "image/png",
                            },
                        }
                    )
        return {
            "role": "user",
            "content": parts,
        }

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
        self, new_message: dict, response_message: anthropic.types.Message
    ) -> None:
        self.chat_history.append([new_message])
        self.chat_history[-1].append(
            {"role": response_message.role, "content": response_message.content}
        )

        if self.action_schema:
            tool_calls = response_message.content
            tool_content = []
            for call in tool_calls:
                if isinstance(call, ToolUseBlock):
                    tool_content.append(
                        {
                            "type": "tool_result",
                            "tool_use_id": call.id,
                            "content": "success",
                        }
                    )
            self.chat_history[-1].append(
                {
                    "role": "user",
                    "content": tool_content,
                }
            )

    @retry(
        wait=wait_fixed(10),
        stop=stop_after_attempt(7),
        retry=retry_if_exception_type(
            (
                anthropic.APITimeoutError,
                anthropic.APIConnectionError,
                anthropic.InternalServerError,
            )
        ),
    )
    def _call_api(self, request_messages: list[dict]) -> anthropic.types.Message:
        request_messages = _merge_request(request_messages)
        if self.action_schema is not None:
            response = self.client.messages.create(
                system=self.system_message,  # <-- system prompt
                messages=request_messages,  # type: ignore
                model=self.model,
                max_tokens=4096,
                tools=self.action_schema,
                tool_choice={"type": "any" if self.tool_call_required else "auto"},
                **self.parameters,
            )
        else:
            response = self.client.messages.create(
                system=self.system_message,  # <-- system prompt
                messages=request_messages,  # type: ignore
                model=self.model,
                max_tokens=4096,
                **self.parameters,
            )

        self.token_usage += response.usage.input_tokens + response.usage.output_tokens
        return response

    def _generate_backend_output(
        self, response_message: anthropic.types.Message
    ) -> BackendOutput:
        message = ""
        action_list = []
        for block in response_message.content:
            if isinstance(block, TextBlock):
                message += block.text
            elif isinstance(block, ToolUseBlock):
                action_list.append(
                    ActionOutput(
                        name=block.name,
                        arguments=block.input,  # type: ignore
                    )
                )
        if not action_list:
            return BackendOutput(message=message, action_list=None)
        else:
            return BackendOutput(
                message=message,
                action_list=action_list,
            )


def _merge_request(request: list[dict]) -> list[dict]:
    merge_request = [deepcopy(request[0])]
    for idx in range(1, len(request)):
        if request[idx]["role"] == merge_request[-1]["role"]:
            merge_request[-1]["content"].extend(request[idx]["content"])
        else:
            merge_request.append(deepcopy(request[idx]))

    return merge_request


def _convert_action_to_schema(action_space):
    if action_space is None:
        return None
    actions = []
    for action in action_space:
        new_action = action.to_openai_json_schema()
        new_action["input_schema"] = new_action.pop("parameters")
        if "returns" in new_action:
            new_action.pop("returns")
        if "title" in new_action:
            new_action.pop("title")
        if "type" in new_action:
            new_action["input_schema"]["type"] = new_action.pop("type")
        if "required" in new_action:
            new_action["input_schema"]["required"] = new_action.pop("required")

        actions.append(new_action)
    return actions
