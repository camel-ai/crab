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
from crab.agents.utils import generate_action_prompt
from crab.core.agent_policy import AgentPolicy
from crab.core.backend_model import (
    BackendModel,
    MessageType,
)


class MultiAgentByEnvPolicy(AgentPolicy):
    _main_agent_prompt = """You are a main agent, and your goal is to plan and
    give instructions to sub-agents in each environment to complete the final task. Now
    you have to do a task as described below: {task_description}.  The description of
    each given environment: {env_description}.  For each step, you are required to
    provide high-level instructions detailing the next actions to be taken.
    Additionally, you must specify which sub-agent in the designated environment should
    execute these instructions. If a sub-agent is not needed for a particular step, you
    may instruct it to skip that step."""

    _env_agent_prompt = """You are a sub-agent responsible for the {environment}
    environment.  The description of the {environment} environment is:
    {env_description}.  Your goal is to assist the main agent in completing the final
    task by performing actions in the {environment} environment according to the
    instructions from the main agent. The final task is described below:
    {task_description}. A unit operation you can perform is called action in a given
    environment. You can only execute action in the {environment} environment. For the
    {environment} environment, you are given a limited action space as function calls:
    {action_descriptions}
    The interactive UI elements on the screenshot are labeled with numeric tags starting
    from 1. For each step, You will receive an instruction telling you what you need to
    do next. After analyzing the instruction you received and the current {environment}
    system, if you think you don't need to do anything in the current {environment}
    system, you should choose SKIP action. Otherwise, you must state what actions to
    take, what the parameters are, and you MUST provide in which environment to perform
    these actions. Your answer must be function calls. Please do not output any other
    information. You must make sure all function calls get their required parameters."""

    _root_agent_prompt = """You are a sub-agent responsible for the crab benchmark root
    environment. Your goal is to assist the main agent in completing the whole task:
    "{task_description}". You can only complete the task or submit the result when the
    main agent tells you the whole task has been completed. Otherwise, you can only call
    SKIP.  """

    def __init__(
        self,
        main_agent_model_backend: BackendModelConfig,
        env_agent_model_backend: BackendModelConfig,
    ):
        self.main_agent_model_backend = create_backend_model(main_agent_model_backend)
        self.env_agent_model_backend_config = env_agent_model_backend
        self.reset(task_description="", action_spaces={}, env_descriptions={})

    def reset(
        self,
        task_description: str,
        action_spaces: dict[str, list[Action]],
        env_descriptions: dict[str, str],
    ) -> list:
        self.task_description = task_description
        main_agent_system_message = self._main_agent_prompt.format(
            task_description=task_description,
            env_description=str(env_descriptions),
        )
        self.main_agent_model_backend.reset(main_agent_system_message, None)

        root_agent_system_message = self._root_agent_prompt.format(
            task_description=task_description
        )
        self.env_agent_model_backends: dict[str, BackendModel] = {}
        for env in action_spaces:
            backend = create_backend_model(self.env_agent_model_backend_config)
            if env == "root":
                backend.reset(root_agent_system_message, action_spaces[env])
            else:
                backend.require_tool = True
                env_agent_system_message = self._env_agent_prompt.format(
                    task_description=task_description,
                    environment=env,
                    env_description=env_descriptions[env],
                    action_descriptions=generate_action_prompt(action_spaces[env]),
                )
                backend.reset(env_agent_system_message, action_spaces[env])
            self.env_agent_model_backends[env] = backend

    def get_token_usage(self):
        result = 0
        result += self.main_agent_model_backend.get_token_usage()
        for env_agent in self.env_agent_model_backends.values():
            result += env_agent.get_token_usage()
        return result

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
        main_prompt = []
        for env in observation:
            main_prompt.extend(observation[env])
        main_prompt.append(
            (
                (
                    f"Your target: {self.task_description}\n"
                    "Tell me the next step in each environment."
                ),
                MessageType.TEXT,
            )
        )
        output = self.main_agent_model_backend.chat(main_prompt)
        main_agent_message = (
            f"The instruction from main agent for this step: {output.message}"
        )

        tool_calls = []
        for env in self.env_agent_model_backends:
            backend = self.env_agent_model_backends[env]
            if env in observation:
                output = backend.chat(
                    observation[env] + [(main_agent_message, MessageType.TEXT)]
                )
            else:
                output = backend.chat((main_agent_message, MessageType.TEXT))
            for action in output.action_list:
                action.env = env
            tool_calls.extend(output.action_list)
        return tool_calls
