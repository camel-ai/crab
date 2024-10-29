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
import json
from typing import Any

from crab import Action, ActionOutput, BackendModel, BackendOutput, Message, MessageType
from crab.agents.utils import extract_text_and_code_prompts

try:
    import openai
    from openai.types.chat import ChatCompletionMessage

    openai_model_enable = True
except ImportError:
    openai_model_enable = False


class OpenAIModel(BackendModel):
    def __init__(
        self,
        model: str,
        parameters: dict[str, Any] | None = None,
        history_messages_len: int = 0,
        tool_call_required: bool = True,
        base_url: str | None = None,
        api_key: str | None = None,
    ) -> None:
        if not openai_model_enable:
            raise ImportError("Please install openai to use OpenAIModel")

        self.model = model
        self.parameters = parameters if parameters is not None else {}
        self.history_messages_len = history_messages_len

        assert self.history_messages_len >= 0

        self.client = openai.OpenAI(api_key=api_key, base_url=base_url)
        self.tool_call_required: bool = tool_call_required
        self.system_message: str = "You are a helpful assistant."
        self.openai_system_message = {
            "role": "system",
            "content": self.system_message,
        }
        self.action_space: list[Action] | None = None
        self.action_schema: list[dict] | None = None
        self.token_usage: int = 0
        self.chat_history: list[list[ChatCompletionMessage | dict]] = []
        self.support_tool_call = True

    def reset(self, system_message: str, action_space: list[Action] | None) -> None:
        self.system_message = system_message
        self.openai_system_message = {
            "role": "system",
            "content": system_message,
        }
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

    def get_token_usage(self):
        return self.token_usage

    def _record_message(
        self, new_message: dict, response_message: ChatCompletionMessage
    ) -> None:
        self.chat_history.append([new_message])
        self.chat_history[-1].append(response_message)

        if self.action_schema and response_message.tool_calls is not None:
            for tool_call in response_message.tool_calls:
                self.chat_history[-1].append(
                    {
                        "tool_call_id": tool_call.id,
                        "role": "tool",
                        "name": tool_call.function.name,
                        "content": "success",
                    }
                )  # extend conversation with function response

    def _call_api(
        self, request_messages: list[ChatCompletionMessage | dict]
    ) -> ChatCompletionMessage:
        if self.action_schema is not None:
            response = self.client.chat.completions.create(
                messages=request_messages,  # type: ignore
                model=self.model,
                tools=self.action_schema,
                tool_choice="required" if self.tool_call_required else "auto",
                **self.parameters,
            )
        else:
            response = self.client.chat.completions.create(
                messages=request_messages,  # type: ignore
                model=self.model,
                **self.parameters,
            )

        self.token_usage += response.usage.total_tokens
        return response.choices[0].message

    def _fetch_from_memory(self) -> list[ChatCompletionMessage | dict]:
        request: list[ChatCompletionMessage | dict] = [self.openai_system_message]
        if self.history_messages_len > 0:
            fetch_history_len = min(self.history_messages_len, len(self.chat_history))
            for history_message in self.chat_history[-fetch_history_len:]:
                request = request + history_message
        return request

    def _construct_new_message(self, message: list[Message]) -> dict[str, Any]:
        new_message_content: list[dict[str, Any]] = []
        for content, msg_type in message:
            match msg_type:
                case MessageType.TEXT:
                    new_message_content.append(
                        {
                            "type": "text",
                            "text": content,
                        }
                    )
                case MessageType.IMAGE_JPG_BASE64:
                    new_message_content.append(
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{content}",
                                "detail": "high",
                            },
                        }
                    )

        return {"role": "user", "content": new_message_content}

    def _generate_backend_output(
        self, response_message: ChatCompletionMessage
    ) -> BackendOutput:
        if response_message.tool_calls is None:
            return BackendOutput(message=response_message.content, action_list=None)
        action_list = [
            ActionOutput(
                name=call.function.name,
                arguments=json.loads(call.function.arguments),
            )
            for call in response_message.tool_calls
        ]
        return BackendOutput(
            message=response_message.content,
            action_list=action_list,
        )


def _convert_action_to_schema(
    action_space: list[Action] | None,
) -> list[dict] | None:
    if action_space is None:
        return None
    actions = []
    for action in action_space:
        new_action = action.to_openai_json_schema()
        actions.append({"type": "function", "function": new_action})
    return actions


class OpenAIModelJSON(OpenAIModel):
    def __init__(
        self,
        model: str,
        parameters: dict[str, Any] = dict(),
        history_messages_len: int = 0,
        base_url: str | None = None,
        api_key: str | None = None,
    ) -> None:
        super().__init__(
            model,
            parameters,
            history_messages_len,
            False,
            base_url,
            api_key,
        )
        self.support_tool_call = False

    def reset(self, system_message: str, action_space: list[Action] | None) -> None:
        super().reset(system_message, action_space)
        self.action_schema = None

    def _record_message(
        self, new_message: dict, response_message: ChatCompletionMessage
    ) -> None:
        self.chat_history.append([new_message])
        self.chat_history[-1].append(
            {"role": "assistant", "content": response_message.content}
        )

    def _generate_backend_output(
        self, response_message: ChatCompletionMessage
    ) -> BackendOutput:
        content = response_message.content
        text_list, code_list = extract_text_and_code_prompts(content)

        action_list = []
        try:
            for code_block in code_list:
                action_object = json.loads(code_block)
                action_list.append(
                    ActionOutput(
                        name=action_object["name"], arguments=action_object["arguments"]
                    )
                )
        except json.JSONDecodeError as e:
            raise RuntimeError(f"Failed to parse code block: {code_block}") from e
        except KeyError as e:
            raise RuntimeError(f"Received invalid action format: {code_block}") from e

        return BackendOutput(
            message="".join(text_list),
            action_list=action_list,
        )


class SGlangOpenAIModelJSON(OpenAIModelJSON):
    def _construct_new_message(self, message: list[Message]) -> dict[str, Any]:
        new_message_content: list[dict[str, Any]] = []
        image_count = 0
        for _, msg_type in message:
            if msg_type == MessageType.IMAGE_JPG_BASE64:
                image_count += 1
        for content, msg_type in message:
            match msg_type:
                case MessageType.TEXT:
                    new_message_content.append(
                        {
                            "type": "text",
                            "text": content,
                        }
                    )
                case MessageType.IMAGE_JPG_BASE64:
                    image_content = {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/png;base64,{content}",
                            "detail": "high",
                        },
                    }
                    if image_count > 1:
                        image_content["modalities"] = "multi-images"
                    new_message_content.append(image_content)

        return {"role": "user", "content": new_message_content}
