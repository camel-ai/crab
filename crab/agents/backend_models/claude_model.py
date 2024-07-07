from copy import deepcopy
from time import sleep
from typing import Any

import anthropic
from anthropic.types import TextBlock
from anthropic.types.beta.tools import ToolUseBlock

from crab import Action, ActionOutput, BackendModel, BackendOutput, MessageType


class ClaudeModel(BackendModel):
    def __init__(
        self,
        model: str,
        parameters: dict[str, Any],
        history_messages_len: int = 0,
    ) -> None:
        super().__init__(
            model,
            parameters,
            history_messages_len,
        )
        self.client = anthropic.Anthropic()

    def reset(self, system_message: str, action_space: list[Action] | None) -> None:
        self.system_message = system_message
        self.action_space = action_space
        self.action_schema = self._convert_action_to_schema(self.action_space)
        self.token_usage = 0
        self.chat_history = []

    def chat(self, message: list[tuple[str, MessageType]]) -> BackendOutput:
        # Initialize chat history
        request = []
        if self.history_messages_len > 0 and len(self.chat_history) > 0:
            for history_message in self.chat_history[-self.history_messages_len :]:
                request = request + history_message

        if not isinstance(message, list):
            message = [message]

        new_message = {
            "role": "user",
            "content": [self._convert_message(part) for part in message],
        }
        request.append(new_message)
        request = self._merge_request(request)

        response = self.call_api(request)
        response_message = response
        self.record_message(new_message, response_message)

        return self._format_response(response_message.content)

    def get_token_usage(self):
        return self.token_usage

    def record_message(self, new_message: dict, response_message: dict) -> None:
        self.chat_history.append([new_message])
        self.chat_history[-1].append(
            {"role": response_message.role, "content": response_message.content}
        )

        if self.action_schema:
            tool_calls = response_message.content
            self.chat_history[-1].append(
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": call.id,
                            "content": "success",
                        }
                        for call in tool_calls
                    ],
                }
            )

    def call_api(self, request_messages: list):
        while True:
            try:
                if self.action_schema is not None:
                    response = self.client.beta.tools.messages.create(
                        system=self.system_message,  # <-- system prompt
                        messages=request_messages,  # type: ignore
                        model=self.model,
                        tools=self.action_schema,
                        tool_choice={"type": "any"},
                        **self.parameters,
                    )
                else:
                    response = self.client.messages.create(
                        system=self.system_message,  # <-- system prompt
                        messages=request_messages,  # type: ignore
                        model=self.model,
                        **self.parameters,
                    )
            except anthropic.RateLimitError:
                print("Rate Limit Error: Please waiting...")
                sleep(10)
            except anthropic.APIStatusError:
                print(len(request_messages))
                raise
            else:
                break

        self.token_usage += response.usage.input_tokens + response.usage.output_tokens
        return response

    @staticmethod
    def _convert_message(message: tuple[str, MessageType]):
        match message[1]:
            case MessageType.TEXT:
                return {
                    "type": "text",
                    "text": message[0],
                }
            case MessageType.IMAGE_JPG_BASE64:
                return {
                    "type": "image",
                    "source": {
                        "data": message[0],
                        "type": "base64",
                        "media_type": "image/png",
                    },
                }

    @staticmethod
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

    @staticmethod
    def _convert_tool_calls_to_action_list(tool_calls) -> list[ActionOutput]:
        if tool_calls is None:
            return tool_calls
        return [
            ActionOutput(
                name=call.name,
                arguments=call.input,
            )
            for call in tool_calls
        ]

    @staticmethod
    def _merge_request(request: list[dict]):
        merge_request = [deepcopy(request[0])]
        for idx in range(1, len(request)):
            if request[idx]["role"] == merge_request[-1]["role"]:
                merge_request[-1]["content"].extend(request[idx]["content"])
            else:
                merge_request.append(deepcopy(request[idx]))

        return merge_request

    @classmethod
    def _format_response(cls, content: list):
        message = None
        action_list = []
        for block in content:
            if isinstance(block, TextBlock):
                message = block.text
            elif isinstance(block, ToolUseBlock):
                action_list.append(block)
        if not action_list:
            return BackendOutput(message=message, action_list=None)
        else:
            return BackendOutput(
                message=message,
                action_list=cls._convert_tool_calls_to_action_list(action_list),
            )
