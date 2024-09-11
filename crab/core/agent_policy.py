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

from .models import Action, ActionOutput, MessageType


class AgentPolicy(ABC):
    @abstractmethod
    def chat(
        self,
        observation: dict[str, list[tuple[str, MessageType]]],
    ) -> list[ActionOutput]: ...

    @abstractmethod
    def reset(
        self,
        task_description: str,
        action_spaces: dict[str, list[Action]],
        env_descriptions: dict[str, str],
    ) -> None: ...

    @abstractmethod
    def get_token_usage(self): ...

    @abstractmethod
    def get_backend_model_name(self) -> str: ...

    @staticmethod
    def combine_multi_env_action_space(
        action_space: dict[str, list[Action]] | None,
    ) -> list[Action]:
        """Combine multi-env action space together to fit in a single agent."""
        result = []
        if action_space is None:
            return result
        for env in action_space:
            for action in action_space[env]:
                new_action = action.model_copy()
                new_action.name = new_action.name + "__in__" + env
                new_action.description = (
                    f"In {env} environment, " + new_action.description
                )
                result.append(new_action)
        return result

    @staticmethod
    def decode_combined_action(
        output_actions: list[ActionOutput],
    ) -> list[ActionOutput]:
        """Decode combined action output to action output with the corresponding
        environment.
        """
        result = []
        for output in output_actions:
            name_env = output.name.split("__in__")
            if len(name_env) != 2:
                raise RuntimeError(
                    'The decoded action name should contain the splitter "__in__".'
                )
            new_output = output.model_copy()
            new_output.name = name_env[0]
            new_output.env = name_env[1]
            result.append(new_output)
        return result

    @staticmethod
    def generate_action_prompt(actions: list[Action] | None):
        if actions is None:
            return None
        result = ""
        for action in actions:
            result += f"[{action.name}: {action.description}]\n"
        return result
