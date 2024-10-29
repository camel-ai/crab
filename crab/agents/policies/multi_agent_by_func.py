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
from crab.agents.backend_models import BackendModelConfig, create_backend_model
from crab.agents.utils import (
    combine_multi_env_action_space,
    decode_combined_action,
    generate_action_prompt,
)
from crab.core import Action, ActionOutput
from crab.core.agent_policy import AgentPolicy
from crab.core.backend_model import MessageType


class MultiAgentByFuncPolicy(AgentPolicy):
    _system_prompt = """You are a helpful assistant. Now you have to do a task as
    described below: {task_description}. And this is the description of each given
    environment: {env_description}. A unit operation you can perform is called action in
    a given environment. For each environment, you are given a limited action space as
    function calls:
    {action_descriptions}
    You may receive a screenshot of the current system. The interactive UI elements on
    the screenshot are labeled with numeric tags starting from 1. For each step, You
    must state what actions to take, what the parameters are, and you MUST provide in
    which environment to perform these actions. """

    _tool_prompt = """You are a helpful assistant in generating function calls. I will
    give you a detailed description of what actions to take next, you should translate
    it into function calls. please do not output any other information.
    """

    def __init__(
        self,
        main_agent_model_backend: BackendModelConfig,
        tool_agent_model_backend: BackendModelConfig,
    ):
        self.main_agent_model_backend = create_backend_model(main_agent_model_backend)
        self.tool_agent_model_backend = create_backend_model(tool_agent_model_backend)
        self.reset(task_description="", action_spaces=None, env_descriptions={})

    def reset(
        self,
        task_description: str,
        action_spaces: dict[str, list[Action]],
        env_descriptions: dict[str, str],
    ) -> list[ActionOutput]:
        self.task_description = task_description
        self.action_space = combine_multi_env_action_space(action_spaces)

        main_agent_system_message = self._system_prompt.format(
            task_description=task_description,
            action_descriptions=generate_action_prompt(self.action_space),
            env_description=str(env_descriptions),
        )
        self.main_agent_model_backend.reset(main_agent_system_message, None)
        self.tool_agent_model_backend.reset(self._tool_prompt, self.action_space)

    def get_token_usage(self):
        return (
            self.main_agent_model_backend.get_token_usage()
            + self.tool_agent_model_backend.get_token_usage()
        )

    def get_backend_model_name(self):
        return (
            self.main_agent_model_backend.__class__.__name__
            + "_"
            + self.main_agent_model_backend.model
        )

    def chat(
        self,
        observation: dict[str, list[tuple[str, MessageType]]],
    ) -> list[ActionOutput]:
        prompt = []
        for env in observation:
            prompt.extend(observation[env])
        prompt.append(
            (
                f"Your target: {self.task_description}\nTell me the next action.",
                MessageType.TEXT,
            )
        )
        output = self.main_agent_model_backend.chat(prompt)
        tool_output = self.tool_agent_model_backend.chat(
            (output.message, MessageType.TEXT)
        )
        return decode_combined_action(tool_output.action_list)
