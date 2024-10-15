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
import traceback
from time import sleep
from typing import Any

from crab.core.graph_evaluator import GraphEvaluator
from crab.utils.measure import timed

from .environment import Environment, create_environment
from .exceptions import TaskNotFound
from .models import Action, BenchmarkConfig, ClosedAction, MessageType, StepResult, Task


class Benchmark:
    """The crab benchmark controller managing environments and agent evaluation.

    The class manages multiple environments together and provide the simple API by
    :meth:`step`, :meth:`observe` and :meth:`reset` for language model agents to perform
    tasks in multiple environments.

    This class introduces a "root" environment with no action or observation
    capabilities, intended as a utility for evaluations not directly tied to a specific
    environment.

    This class operates in two distinct modes: "multi-environment" and
    "single-environment".  In multi-environment mode, observations and action results
    are separated by environment, returned as a dictionary. While in single-environment
    mode, all observations and action outcomes are merged under the "root" environment,
    with actions being appropriately routed to their respective environments.

    """

    def __init__(
        self,
        name: str,
        tasks: list[Task],
        environments: list[Environment],
        default_env: str | None = None,
        multienv: bool = False,
        prompting_tools: dict[str, dict[str, Action]] = {},
        root_action_space: list[Action] = [],
        step_limit: int = 30,
        common_setup: list[ClosedAction] = [],
    ) -> None:
        """Initializes the instance.

        Args:
            name: Identifier for the benchmark.
            tasks: Tasks to be executed within the benchmark.
            environments: Environments in which the benchmark is conducted.
            default_env: The default environment name, applied when actions do not
                specify an environment. Defaults to "root" in the multi-environment mode
                and to the environment in the single environment mode.
            multienv: Indicates whether to enable multi-environment mode. Defaults to
                :obj:`False`.
            prompting_tools: Prompting tools applied in :meth:`observe_with_prompt`. The
                first level keys are environment names, the second level keys are
                observation action names. Defaults to empty.
            root_action_space: The action space executed in the root environment.
        """
        self.name = name
        self.tasks = tasks
        self.multienv = multienv
        self.prompting_tools = prompting_tools
        self.step_limit = step_limit
        self.common_setup = common_setup

        if isinstance(environments, Environment):
            environments = [environments]
        self.root_env = Environment(
            name="root",
            action_space=root_action_space,
            observation_space=[],
            description="The crab benchmark root. You can submit your answer or "
            "complete the task using this environment.",
        )
        self.root_env.contained_envs = {env.name: env for env in environments}  # A hack
        environments.append(self.root_env)
        self.environment_map: dict[str, Environment] = {
            env.name: env for env in environments
        }

        # if not multienv, combine all environments action space together
        if not self.multienv:
            # action_map is used only by "agent", specifically `step` and
            # `export_action_space` functions
            self._verify_spaces()
            self._generate_action_map()

        # default_env is used for predefined actions without env_name or like
        # evaluators setups, teardowns, and so on.
        if default_env is None:
            if not multienv and len(environments) == 2:
                self.default_env = environments[0].name
            else:
                self.default_env = self.root_env.name
        else:
            self.default_env = default_env

        self.current_task: Task | None = None
        self.current_evaluator: GraphEvaluator | None = None
        self.step_cnt = 0

    def start_task(self, task_id: str) -> tuple[Task, dict[str, list[Action]]]:
        """Initializes and starts a specified task.

        Args:
            task_id: The ID of the task to start.

        Returns:
            A tuple (task, action_space), where task is the started task object, and
            action_sapce is a dict mapping action names to the corresponding action
            object.
        """
        if self.current_task is not None:
            raise RuntimeError("Another task is running")
        self.current_task = self._get_task_by_id(task_id)

        # reset all environments
        self._reset_environments()

        for action in self.common_setup:
            self._take_env_action(action)

        # select environment by Action.env_name
        for action in self.current_task.setup:
            self._take_env_action(action)

        for task_action in self.current_task.extra_action:
            self._set_env_action(task_action)

        # reset evaluator
        self.current_evaluator = GraphEvaluator(self.current_task.evaluator)
        # put submit action to corresponding env space
        # For now, only the last node can be the submit task

        self.step_cnt = 0
        return self.current_task, self.export_action_space()

    def close_task(self) -> None:
        """Cleans up after a task is completed."""
        if self.current_evaluator is None or self.current_task is None:
            raise RuntimeError("There is no started task.")
        for action in self.current_task.teardown:
            self._take_env_action(action)
        self.current_task = None

    def get_env_descriptions(self) -> dict[str, str]:
        """Get environment descriptions as a dict structure."""
        return {
            name: self.environment_map[name].description
            for name in self.environment_map
        }

    def observe(self) -> dict[str, dict[str, Any]]:
        """Collects observations from all environments.

        Returns:
            A dict-of-dict with observations from each environment. The first level keys
            are environment names, the second level keys are observation action names.
        """
        env_obs = {env.name: env.observe() for env in self.environment_map.values()}
        if self.multienv:
            return env_obs
        return self._merge_dicts(env_obs)

    @timed
    def observe_with_prompt(
        self,
    ) -> tuple[dict[str, dict[str, Any]], dict[str, tuple[str, MessageType]]]:
        """Collects observations and applies prompting tools.

        Returns:
            A tuple (observations, prompts), where "observations" and "prompts" are
            observations from each environment and the result of applying prompting
            tools on them. The first level keys are environment names, the second level
            keys are observation action names. Notice that some dicts can be empty if
            its prompting tool wasn't set.
        """
        observations = {}
        prompts = {}
        for env_name, env in self.environment_map.items():
            if env_name in self.prompting_tools:
                tools = self.prompting_tools[env_name]
            else:
                tools = {}
            observations[env_name], prompts[env_name] = env.observe_with_prompt(tools)
        if self.multienv:
            return observations, prompts
        return self._merge_dicts(observations), self._merge_dicts(prompts)

    def evaluate(self):
        self.current_evaluator.step(self.environment_map, self.default_env)
        return self.current_evaluator.stat()

    @timed
    def step(
        self,
        action: str,
        parameters: dict[str, Any] = {},
        env_name: str | None = None,
    ) -> StepResult:
        """Executes a step in the benchmark by performing an action.

        Args:
            action: The action to execute.
            parameters: Parameters for the action.
            env_name: The name of the environment.

        Returns:
            The result of the step including observations and evaluation metrics. Notice
            that the `truncated` field in the result is not meaningful for now.
        """
        terminated = False
        info = {}
        if self.current_evaluator is None or self.current_task is None:
            raise RuntimeError("There is no started task.")

        if action == "complete":
            terminated = True
            info["terminate_reason"] = "agent_complete"
            return StepResult(
                truncated=False,
                terminated=True,
                action_returns=None,
                evaluation_results=self.current_evaluator.stat(),
                info=info,
            )

        try:
            environment = self._get_env(env_name=env_name, action_name=action)
        except Exception:
            print(traceback.format_exc())
            terminated = True
            info["terminate_reason"] = "action_format_error"
            info["exception_detail"] = traceback.format_exc()
            environment.reset()
            self.close_task()
            return StepResult(
                truncated=False,
                terminated=True,
                action_returns=None,
                evaluation_results=self.current_evaluator.stat(),
                info=info,
            )
        try:
            action_returns = environment.step(action, parameters)
        except Exception:
            print(traceback.format_exc())
            terminated = True
            info["terminate_reason"] = "env_exception"
            info["exception_detail"] = traceback.format_exc()
            environment.reset()
            self.close_task()
            return StepResult(
                truncated=False,
                terminated=True,
                action_returns=None,
                evaluation_results=self.current_evaluator.stat(),
                info=info,
            )

        try:
            evaluation_results = self.evaluate()
        except Exception:
            print(traceback.format_exc())
            terminated = True
            info["terminate_reason"] = "evaluator_exception"
            info["exception_detail"] = traceback.format_exc()
            environment.reset()
            self.close_task()
            return StepResult(
                truncated=False,
                terminated=True,
                action_returns=action_returns,
                evaluation_results=self.current_evaluator.stat(),
                info=info,
            )

        self.step_cnt += 1
        if self.current_evaluator.is_complete():
            terminated = True
            info["terminate_reason"] = "success"
        if self.step_cnt >= self.step_limit:
            terminated = True
            info["terminate_reason"] = "reach_max_step"
        if terminated:
            environment.reset()
            self.close_task()
        return StepResult(
            truncated=False,
            terminated=terminated,
            action_returns=action_returns,
            evaluation_results=evaluation_results,
            info=info,
        )

    def reset(self) -> None:
        """Resets all environments and the current task."""
        self.current_evaluator = None
        self._reset_environments()

    def human_evaluation(self, task_id: str) -> None:
        task, _ = self.start_task(task_id)
        print(task.description)

        self.current_evaluator.human_mode = True

        evaluation_results = self.evaluate()
        print(evaluation_results, end="")
        while evaluation_results["completeness"] != 1.0:
            sleep(2)
            evaluation_results = self.evaluate()
            print("\r" + str(evaluation_results), end="")
        self.close_task()

    def export_action_space(self) -> dict[str, list[Action]]:
        """Returns the action spaces from all environments.

        Returns:
            A dict of action lists for each environment, keyed by environment name.
        """
        result = {env.name: env.action_space for env in self.environment_map.values()}
        if self.multienv:
            return result
        return self._merge_lists(result)

    def _verify_spaces(self) -> None:
        """Make sure all actions and observations are unique."""
        observation_name_set = set()
        action_name_set = set()
        for env in self.environment_map.values():
            for action in env.action_space:
                if action.name in action_name_set:
                    raise ValueError(
                        "Dulplicated action names are not allowed in single "
                        "environment benchmark."
                    )
                action_name_set.add(action.name)
            for observation in env.observation_space:
                if observation.name in observation_name_set:
                    raise ValueError(
                        "Dulplicated observation names are not allowed in the "
                        "single environment benchmark."
                    )
                observation_name_set.add(observation.name)

    def _generate_action_map(self) -> None:
        self.action_map: dict[str, Environment] = {}
        for env in self.environment_map.values():
            for action in env.action_space:
                self.action_map[action.name] = env

    def _get_env(
        self, env_name: str | None = None, action_name: str | None = None
    ) -> Environment:
        # env_name exists just return it
        if env_name is not None:
            return self.environment_map[env_name]
        # or in multienv use default env, in singlenev use action_name mapping
        if action_name is not None and not self.multienv:
            return self.action_map[action_name]
        return self.environment_map[self.default_env]

    def _take_env_action(self, action: Action) -> Any:
        if action.env_name is None:
            env = self.environment_map[self.default_env]
        else:
            env = self.environment_map[action.env_name]
        return env.take_action(action)

    def _set_env_action(self, action: Action) -> None:
        if action.env_name is None:
            env = self.environment_map[self.default_env]
        else:
            env = self.environment_map[action.env_name]
        env.set_action(action)
        if not self.multienv:
            self.action_map[action.name] = env

    def _reset_environments(self):
        for env in self.environment_map.values():
            env.reset()
        if not self.multienv:
            self._generate_action_map()

    def _get_task_by_id(self, task_id: str) -> Task:
        result = [task for task in self.tasks if task_id == task.id]
        if len(result) == 0:  # Doesn't find the task
            raise TaskNotFound(f"No such task: {task_id}")
        return result[0]

    def _merge_dicts(
        self, env_dict: dict[str, dict[str, Any]]
    ) -> dict[str, dict[str, Any]]:
        "In single environment mode, merge aciton_space/observation_space in root."
        result = {}
        for dict_value in env_dict.values():
            result.update(dict_value)
        return {self.default_env: result}

    def _merge_lists(self, env_dict: dict[str, list]) -> dict[str, list]:
        "In single environment mode, merge aciton_space/observation_space in root."
        result = []
        for dict_value in env_dict.values():
            result.extend(dict_value)
        return {self.default_env: result}


def create_benchmark(config: BenchmarkConfig) -> Benchmark:
    """Creates a benchmark by BenchmarkConfig"""
    if isinstance(config, BenchmarkConfig):
        environments = [
            create_environment(env_config) for env_config in config.environments
        ]
        parameters = dict(config)
        parameters["environments"] = environments
        return Benchmark(**parameters)
    else:
        raise ValueError("Unsupport benchmark config type.")
