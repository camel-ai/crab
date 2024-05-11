from abc import ABC, abstractmethod
from typing import Any, List, Tuple

from crab import Action

signle_env_system_prompt = """You are a helpful assistant. Now you have to do a task as
described below: 

**"{description}."**

You should never forget this task and always perform actions to achieve this task. A
unit operation you can perform is called Action. You have a limited action space as
function calls:
{action_descriptions}
You may receive a screenshot of the current system. You may receive a screenshot of a
smartphone app. The interactive UI elements on the screenshot are labeled with numeric
tags starting from 1. 

In each step, You MUST explain what do you see from the current observation and
the plan of the next action, then use a provided action in each step to
achieve the task. You should state what action to take and what the parameters should
be. Your answer MUST be a least one function call. You SHOULD NEVER ask me to do
anything for you. Always do them by yourself using function call.
"""

multi_env_system_prompt = """You are a helpful assistant. Now you have to do a task as
described below: {description} A unit operation you can perform is called action in a
given environment. For each environment, you are given a limited action space as
function calls:
{action_descriptions}
You may receive a screenshot of the current system. The interactive UI elements on the
screenshot are labeled with numeric tags starting from 1. For each step, You must state
what actions to take, what the parameters are, and you MUST provide in which environment
to perform these actions.  Your answer must be a least one function call. please do not
output any other infomation.  You must make sure all function calls get their required
parameters."""


class BaseAgent(ABC):
    @abstractmethod
    def __init__(
        self,
        description: str,
        model: str,
        action_space: dict[list[Action]],
        multienv: bool = False,
        max_tokens: int = 300,
        history_messages_len: int = 0,
    ) -> None:
        # action_descriptions = ".\n".join([json.dumps(action) for action in actions])

        self.action_space = action_space
        self.multienv = multienv
        self.model = model
        self.max_tokens = max_tokens
        self.history_messages_len = history_messages_len
        assert self.history_messages_len >= 0

        self.signle_env_system_prompt = signle_env_system_prompt
        self.multi_env_system_prompt = multi_env_system_prompt

        self.chat_history = []
        self.token_usage = 0

    @abstractmethod
    def chat(self, contents: List[Tuple[str, int]]) -> dict[str, Any]:
        pass

    @abstractmethod
    def _to_message(self, content: str, modality: int):
        pass

    @abstractmethod
    def _convert_action_to_schema(self, action_space):
        pass

    def _generate_action_prompt(self, actions: list[dict]):
        result = ""
        for action in actions:
            result += f"[{action['name']}: {action['description']}]\n"
        return result

    def get_token_usage(self):
        return self.token_usage
