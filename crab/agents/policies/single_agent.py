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
from crab import Action, ActionOutput
from crab.agents.backend_models import BackendModelConfig, create_backend_model
from crab.agents.utils import (
    combine_multi_env_action_space,
    decode_combined_action,
    generate_action_prompt,
)
from crab.core.agent_policy import AgentPolicy
from crab.core.backend_model import (
    MessageType,
)
from crab.utils.measure import timed


class SingleAgentPolicy(AgentPolicy):
    _system_prompt_with_function_call = """\
    You are a helpful assistant. Now you have to do a task as described below: 

    **"{task_description}."**

    You should never forget this task and always perform actions to achieve this task. 
    And this is the description of each given environment: {env_description}. A
    unit operation you can perform is called Action. You have a limited action space as
    function calls:
    {action_descriptions}
    You may receive a screenshot of the current system. You may receive a screenshot of
    a smartphone app. The interactive UI elements on the screenshot are labeled with
    numeric tags starting from 1. 

    In each step, You MUST explain what do you see from the current observation and the
    plan of the next action, then use a provided action in each step to achieve the
    task. You should state what action to take and what the parameters should be. Your
    answer MUST be a least one function call. You SHOULD NEVER ask me to do anything for
    you. Always do them by yourself using function calls.
    """

    _system_prompt_no_function_call = """\
    You are a helpful assistant. Now you have to do a task as described below: 

    **"{task_description}."**

    You should never forget this task and always perform actions to achieve this task. 
    And this is the description of each given environment: {env_description}. You will
    receive screenshots of the environments. The interactive UI elements on the
    screenshot are labeled with numeric tags starting from 1. 

    A unit operation you can perform is called Action. You have a limited action space
    as function calls: {action_descriptions}. You should generate JSON code blocks to
    execute the actions. Each code block MUST contains only one json object, i.e. one
    action. You can output multiple code blocks to execute multiple actions in a single
    step. You must follow the JSON format below to output the action. 
    ```json
    {{"name": "action_name", "arguments": {{"arg1": "value1", "arg2": "value2"}}}}
    ```
    or if not arguments needed:
    ```json
    {{"name": "action_name", "arguments": {{}}}}
    ```

    In each step, You MUST explain what do you see from the current observation and the
    plan of the next action, then use a provided action in each step to achieve the
    task. You should state what action to take and what the parameters should be. Your
    answer MUST contain at least one code block. You SHOULD NEVER ask me to do anything
    for you. Always do them by yourself.
    """

    def __init__(
        self,
        model_backend: BackendModelConfig,
    ):
        self.model_backend = create_backend_model(model_backend)
        if self.model_backend.support_tool_call:
            self.system_prompt = self._system_prompt_with_function_call
        else:
            self.system_prompt = self._system_prompt_no_function_call
        self.reset(task_description="", action_spaces=None, env_descriptions={})

    def reset(
        self,
        task_description: str,
        action_spaces: dict[str, list[Action]],
        env_descriptions: dict[str, str],
    ) -> list:
        self.task_description = task_description
        self.action_space = combine_multi_env_action_space(action_spaces)
        system_message = self.system_prompt.format(
            task_description=task_description,
            action_descriptions=generate_action_prompt(
                self.action_space,
                expand=not self.model_backend.support_tool_call,
            ),
            env_description=str(env_descriptions),
        )
        self.model_backend.reset(system_message, self.action_space)

    def get_token_usage(self):
        return self.model_backend.get_token_usage()

    def get_backend_model_name(self):
        return self.model_backend.__class__.__name__

    @timed
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
        output = self.model_backend.chat(prompt)
        return decode_combined_action(output.action_list)
