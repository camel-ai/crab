import json
from typing import Any

import openai

from crab import Action

from .base_agent import BaseAgent


class OpenAIAgent(BaseAgent):
    def __init__(
        self,
        description: str,
        action_space: dict[str, list[Action]],
        multienv=False,
        model: str = "gpt-4-turbo",
        max_tokens: int = 3000,
        history_messages_len: int = 0,
    ):
        super().__init__(
            description, model, action_space, multienv, max_tokens, history_messages_len
        )

        self._convert_action_to_schema(self.action_space)

        if self.multienv:
            self.system_message = {
                "role": "system",
                "content": self.multi_env_system_prompt.format(
                    description=description,
                    action_descriptions=self._generate_action_prompt(self.actions),
                ),
            }
        else:
            self.system_message = {
                "role": "system",
                "content": self.signle_env_system_prompt.format(
                    description=description,
                    action_descriptions=self._generate_action_prompt(self.actions),
                ),
            }

        self.client = openai.OpenAI()

    def chat(self, contents: list[tuple[str, int]]) -> dict[str, Any]:
        new_message = {
            "role": "user",
            "content": [
                self._to_message(content, modality) for content, modality in contents
            ],
        }

        request = [self.system_message]
        # Add chat_history
        if self.history_messages_len > 0 and len(self.chat_history) > 0:
            for message in self.chat_history[-self.history_messages_len :]:
                request = request + message

        request.append(new_message)
        self.chat_history.append([new_message])

        response = self.client.chat.completions.create(
            messages=request,  # type: ignore
            model=self.model,
            max_tokens=self.max_tokens,
            tools=[{"type": "function", "function": action} for action in self.actions],
        )

        if response.usage is not None:
            self.token_usage += response.usage.total_tokens
        response_message = response.choices[0].message

        tool_calls = response_message.tool_calls
        if tool_calls is None:
            print("\nRequest: ", request)
            print("\033[94m" f"Agent Reponse: {response_message}" "\033[0m")
            raise ValueError("For each step agent should take at least one action. ")

        self.chat_history[-1].append(response_message)
        for tool_call in tool_calls:
            self.chat_history[-1].append(
                {
                    "tool_call_id": tool_call.id,
                    "role": "tool",
                    "name": tool_call.function.name,
                    "content": "",
                }
            )  # extend conversation with function response

        if self.multienv:
            return {
                "content": response_message,
                "action_list": [
                    (
                        (
                            call.function.name.split("__in__")[0],
                            json.loads(call.function.arguments),
                        ),
                        call.function.name.split("__in__")[1],
                    )
                    for call in tool_calls
                ],
            }
        else:
            return {
                "content": response_message.content,
                "action_list": [
                    (
                        call.function.name.split("__in__")[0],
                        json.loads(call.function.arguments),
                    )
                    for call in tool_calls
                ],
            }

    def _to_message(self, content: str, modality: int):
        match modality:
            case 0:
                return {
                    "type": "text",
                    "text": content,
                }
            case 1:
                return {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{content}",
                        "detail": "high",
                    },
                }

    def _convert_action_to_schema(self, action_space):
        self.actions = []
        for env in action_space:
            for action in action_space[env]:
                new_action = action.to_openai_json_schema()
                new_action["name"] = new_action["name"] + "__in__" + env
                new_action["description"] = "In {} environment, {}".format(
                    env, new_action["description"]
                )
                self.actions.append(new_action)
