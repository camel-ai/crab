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

from openai.types.chat import ChatCompletionMessageToolCall
from PIL import Image

from crab import Action, ActionOutput, BackendModel, BackendOutput, MessageType
from crab.utils.common import base64_to_image

try:
    from camel.agents import ChatAgent
    from camel.messages import BaseMessage
    from camel.models import ModelFactory
    from camel.toolkits import OpenAIFunction
    from camel.types.enums import ModelPlatformType, ModelType

    CAMEL_ENABLED = True
except ImportError:
    CAMEL_ENABLED = False


def _get_model_platform_type(model_platform_name: str) -> "ModelPlatformType":
    try:
        return ModelPlatformType(model_platform_name)
    except ValueError:
        all_models = [platform.value for platform in ModelPlatformType]
        raise ValueError(
            f"Model {model_platform_name} not found. Supported models are {all_models}"
        )


def _get_model_type(model_name: str) -> "str | ModelType":
    try:
        return ModelType(model_name)
    except ValueError:
        return model_name


def _convert_action_to_schema(
    action_space: list[Action] | None,
) -> "list[OpenAIFunction] | None":
    if action_space is None:
        return None
    schema_list = []
    for action in action_space:
        new_action = action.to_openai_json_schema()
        schema = {"type": "function", "function": new_action}
        schema_list.append(OpenAIFunction(action.entry, schema))
    return schema_list


def _convert_tool_calls_to_action_list(
    tool_calls: list[ChatCompletionMessageToolCall] | None,
) -> list[ActionOutput] | None:
    if tool_calls is None:
        return None

    return [
        ActionOutput(
            name=call.function.name,
            arguments=json.loads(call.function.arguments),
        )
        for call in tool_calls
    ]


class CamelModel(BackendModel):
    def __init__(
        self,
        model: str,
        model_platform: str,
        parameters: dict[str, Any] | None = None,
        history_messages_len: int = 0,
        tool_call_required: bool = True,
    ) -> None:
        if not CAMEL_ENABLED:
            raise ImportError("Please install camel-ai to use CamelModel")
        self.model = model
        self.parameters = parameters if parameters is not None else {}
        self.history_messages_len = history_messages_len

        self.model_type = _get_model_type(model)
        self.model_platform_type = _get_model_platform_type(model_platform)
        self.client: ChatAgent | None = None
        self.token_usage = 0
        self.tool_call_required = tool_call_required
        self.history_messages_len = history_messages_len

    def get_token_usage(self) -> int:
        return self.token_usage

    def reset(self, system_message: str, action_space: list[Action] | None) -> None:
        action_schema = _convert_action_to_schema(action_space)
        config = self.parameters.copy()
        if action_schema is not None:
            config["tool_choice"] = "required" if self.tool_call_required else "auto"
            config["tools"] = [
                schema.get_openai_tool_schema() for schema in action_schema
            ]

        backend_model = ModelFactory.create(
            self.model_platform_type,
            self.model_type,
            model_config_dict=config,
        )
        sysmsg = BaseMessage.make_assistant_message(
            role_name="Assistant",
            content=system_message,
        )
        self.client = ChatAgent(
            model=backend_model,
            system_message=sysmsg,
            external_tools=action_schema,
            message_window_size=self.history_messages_len,
        )
        self.token_usage = 0

    def chat(self, messages: list[tuple[str, MessageType]]) -> BackendOutput:
        # TODO: handle multiple text messages after message refactoring
        image_list: list[Image.Image] = []
        content = ""
        for message in messages:
            if message[1] == MessageType.IMAGE_JPG_BASE64:
                image = base64_to_image(message[0])
                image_list.append(image)
            else:
                content = message[0]
        usermsg = BaseMessage.make_user_message(
            role_name="User",
            content=content,
            image_list=image_list,
        )
        response = self.client.step(usermsg)
        self.token_usage += response.info["usage"]["total_tokens"]
        tool_call_request = response.info.get("external_tool_request")

        return BackendOutput(
            message=response.msg.content,
            action_list=_convert_tool_calls_to_action_list([tool_call_request]),
        )
