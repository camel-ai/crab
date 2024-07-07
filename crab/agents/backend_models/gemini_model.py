import os
from time import sleep
from typing import Any

import google.generativeai as genai
from google.ai.generativelanguage_v1beta import Part, Tool
from google.api_core.exceptions import ResourceExhausted
from google.generativeai.types import content_types

from crab import Action, ActionOutput, BackendModel, BackendOutput, MessageType
from crab.utils.agent.gemini_utils import _action_to_funcdec_policy
from crab.utils.common import base64_to_image


class GeminiModel(BackendModel):
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
        genai.configure(api_key=os.environ["GEMINI_API_KEY"])
        self.client = genai

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
            "parts": [self._convert_message(part) for part in message],
        }
        request.append(new_message)

        response = self.call_api(request)
        response_message = response.candidates[0].content
        self.record_message(new_message, response_message)

        tool_calls = [
            Part.to_dict(part)["function_call"]
            for part in response.parts
            if "function_call" in Part.to_dict(part)
        ]

        return BackendOutput(
            message=response_message.parts[0].text or None,
            action_list=self._convert_tool_calls_to_action_list(tool_calls),
        )

    def get_token_usage(self):
        return self.token_usage

    def record_message(self, new_message: dict, response_message: dict) -> None:
        self.chat_history.append([new_message])
        self.chat_history[-1].append(
            {"role": response_message.role, "parts": response_message.parts}
        )

    def call_api(self, request_messages: list):
        while True:
            try:
                if self.action_schema is not None:
                    tool_config = content_types.to_tool_config(
                        {"function_calling_config": {"mode": "ANY"}}
                    )
                    response = self.client.GenerativeModel(
                        self.model, system_instruction=self.system_message
                    ).generate_content(
                        contents=request_messages,
                        tools=self.action_schema,
                        tool_config=tool_config,
                        # **self.parameters,
                    )
                else:
                    response = self.client.GenerativeModel(
                        self.model, system_instruction=self.system_message
                    ).generate_content(
                        contents=request_messages,
                        # **self.parameters,
                    )
            except ResourceExhausted:
                print(
                    "ResourceExhausted: 429 Resource has been exhausted. Please waiting..."
                )
                sleep(10)
            else:
                break

        self.token_usage += response.candidates[0].token_count
        return response

    @staticmethod
    def _convert_message(message: tuple[str, MessageType]):
        match message[1]:
            case MessageType.TEXT:
                return message[0]
            case MessageType.IMAGE_JPG_BASE64:
                return base64_to_image(message[0])

    @staticmethod
    def _convert_action_to_schema(action_space):
        if action_space is None:
            return None
        actions = []
        for action in action_space:
            actions.append(
                Tool(function_declarations=[_action_to_funcdec_policy(action)])
            )
        return actions

    @staticmethod
    def _convert_tool_calls_to_action_list(tool_calls) -> list[ActionOutput]:
        if tool_calls:
            return [
                ActionOutput(
                    name=call["name"],
                    arguments=call["args"],
                )
                for call in tool_calls
            ]
        else:
            return None
