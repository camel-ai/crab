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
from abc import ABC, abstractmethod
from typing import Any

import gymnasium as gym
from openai.types.chat.chat_completion_tool_param import ChatCompletionToolParam


class Environment(gym.Env, ABC):
    """The base environment class for agents to interact with in the CRAB framework.

    Crab Environment is a subclass of `gymnasium.Env` and is designed to be a base class
    for all environments in the CRAB.  Your must implement two functions
    `get_action_schema` and `convert_tool_call_to_action` to make the environment
    compatible with OpenAI tool use API.
    """

    @abstractmethod
    def get_description(self) -> str:
        """Get the description of the environment, which can be used as a part of the
        agent prompt.

        Returns:
            A string description of the environment.
        """

    @abstractmethod
    def get_action_schema(self) -> list[ChatCompletionToolParam]:
        """Get the tool schema for the action space of the environment.

        The schema provides detailed descriptions of the whole actions space and their
        parameters that represent all the possible actions in the tool calling format,
        which can be directly used in the OpenAI API. It should be comprehensive and do
        not produce any misunderstanding for a human user.

        Returns:
            A list of tool schema.
        """
        ...

    @abstractmethod
    def convert_tool_call_to_action(self, tool_name: str, parameters: dict) -> Any:
        """Convert a tool call to the actual action space in the environment.

        Args:
            tool_name: The name of the tool.
            parameters: The parameters of the tool call.
        """
        ...
