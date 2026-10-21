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
from time import sleep

from crab import Action, ActionOutput, BackendModel, BackendOutput, MessageType

try:
    from zhipuai import ZhipuAI
    glm_model_enable = True
except ImportError:
    glm_model_enable = False

class GLMModel(BackendModel):
    def __init__(
        self,
        model: str,
        parameters: dict[str, Any] = dict(),
        history_messages_len: int = 0,
    ) -> None:
        if not glm_model_enable:
            raise ImportError("Please install zhipuai to use GLMModel")
        super().__init__(
            model,
            parameters,
            history_messages_len,
        )
        self.client = ZhipuAI()

    def reset(self, system_message: str, action_space: list[Action] | None) -> None:
        self.system_message = system_message
        self.glm_system_message = {
            "role": "system",
            "content": system_message,
        }
        self.action_space = action_space
        self.action_schema = self._convert_action_to_schema(self.action_space)
        self.token_usage = 0
        self.chat_history = []

    def chat(self, message: tuple[str, MessageType]):
        request_messages = self._convert_to_request_messages(message)
        response = self.call_api(request_messages)
        
        assistant_message = response.choices[0].message
        action_list = self._convert_tool_calls_to_action_list(assistant_message)
        
        output = ChatOutput(
            message=assistant_message.content if not action_list else None,
            action_list=action_list,
        )
        
        self.record_message(request_messages[-1], assistant_message)
        return output

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
                )

    def call_api(self, request_messages: list):
        while True:
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=request_messages,
                    **self.parameters,
                )
            except Exception as e:
                print(f"API call failed: {str(e)}. Retrying in 10 seconds...")
                sleep(10)
            else:
                break

        self.token_usage += response.usage.total_tokens
        return response

    @staticmethod
    def _convert_action_to_schema(action_space: list[Action] | None):
        if action_space is None:
            return None
        
        tools = []
        for action in action_space:
            tool = {
                "type": "function",
                "function": {
                    "name": action.name,
                    "description": action.description,
                    "parameters": {
                        "type": "object",
                        "properties": {},
                        "required": [],
                    },
                },
            }
            for param in action.parameters:
                tool["function"]["parameters"]["properties"][param.name] = {
                    "type": param.type,
                    "description": param.description,
                }
                if param.required:
                    tool["function"]["parameters"]["required"].append(param.name)
            tools.append(tool)
        return tools

    @staticmethod
    def _convert_tool_calls_to_action_list(self, message):
        if not message.content or not message.content.startswith("arguments="):
            return None
        
        action_list = []
        parts = message.content.split(", name=")
        arguments = json.loads(parts[0].replace("arguments=", "").strip("'"))
        name = parts[1].strip("'")
        action_output = ActionOutput(
            name=name,
            args=arguments,
        )
        action_list.append(action_output)
        return action_list

    @staticmethod
    def _convert_message(message: tuple[str, MessageType]):
        content, message_type = message
        if message_type == MessageType.TEXT:
            return {"type": "text", "text": content}
        elif message_type == MessageType.IMAGE_URL:
            return {"type": "image_url", "image_url": {"url": content}}
        else:
            raise ValueError(f"Unsupported message type: {message_type}")