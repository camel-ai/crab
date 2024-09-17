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
        parameters: dict[str, Any] = dict(),
        history_messages_len: int = 0,
    ) -> None:
        if not openai_model_enable:
            raise ImportError("Please install openai to use OpenAIModel")
        super().__init__(
            model,
            parameters,
            history_messages_len,
        )
        self.client = openai.OpenAI()

    def reset(self, system_message: str, action_space: list[Action] | None) -> None:
        self.system_message = system_message
        self.openai_system_message = {
            "role": "system",
            "content": system_message,
        }
        self.action_space = action_space
        self.action_schema = self._convert_action_to_schema(self.action_space)
        self.token_usage = 0
        self.chat_history = []

    def chat(self, message: list[Message] | Message) -> BackendOutput:
        if isinstance(message, tuple):
            message = [message]
        request = self.fetch_from_memory()
        new_message = self.construct_new_message(message)
        request.append(new_message)
        response_message = self.call_api(request)
        self.record_message(new_message, response_message)
        return self.generate_backend_output(response_message)

    def get_token_usage(self):
        return self.token_usage

    def record_message(self, new_message: dict, response_message: dict) -> None:
        self.chat_history.append([new_message])
        self.chat_history[-1].append(response_message)

        if self.action_schema:
            tool_calls = response_message.tool_calls
            for tool_call in tool_calls:
                self.chat_history[-1].append(
                    {
                        "tool_call_id": tool_call.id,
                        "role": "tool",
                        "name": tool_call.function.name,
                        "content": "",
                    }
                )  # extend conversation with function response

    def call_api(self, request_messages: list) -> ChatCompletionMessage:
        if self.action_schema is not None:
            response = self.client.chat.completions.create(
                messages=request_messages,  # type: ignore
                model=self.model,
                tools=self.action_schema,
                tool_choice="required",
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

    def fetch_from_memory(self) -> list[dict]:
        request = [self.openai_system_message]
        if self.history_messages_len > 0:
            fetch_hisotry_len = min(self.history_messages_len, len(self.chat_history))
            for history_message in self.chat_history[-fetch_hisotry_len:]:
                request = request + history_message
        return request

    def construct_new_message(self, message: list[Message]) -> list[dict]:
        new_message_content = []
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

    def generate_backend_output(
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

    @staticmethod
    def _convert_action_to_schema(action_space):
        if action_space is None:
            return None
        actions = []
        for action in action_space:
            new_action = action.to_openai_json_schema()
            actions.append({"type": "function", "function": new_action})
        return actions
