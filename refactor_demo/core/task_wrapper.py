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
from typing import Any

import gymnasium as gym
from gymnasium import Wrapper
from gymnasium.core import ActType, ObsType, WrapperObsType
from gymnasium.spaces import Dict, Space, Text, Tuple


class TaskWrapper(Wrapper[WrapperObsType, ActType, ObsType, ActType]):
    def __init__(
        self,
        env: gym.Env[ObsType, ActType],
        task: Task,
        *,
        dict_task_key: str = "task",
    ):
        super().__init__(env)
        self.env = env
        self.task = task

        task_space = Text(500)

        # Observation space in different situations
        if isinstance(env.observation_space, Dict):
            assert dict_task_key not in env.observation_space.keys()
            observation_space = Dict(
                {dict_task_key: task_space, **env.observation_space.spaces}
            )
            self._append_data_func = lambda obs, task: {dict_task_key: task, **obs}
        elif isinstance(env.observation_space, Tuple):
            observation_space = Tuple(env.observation_space.spaces + (task_space,))
            self._append_data_func = lambda obs, task: obs + (task,)
        else:
            observation_space = Dict(obs=env.observation_space, task=task_space)
            self._append_data_func = lambda obs, task: {"obs": obs, "task": task}

        self.observation_space: gym.Space[WrapperObsType] = observation_space

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[Dict, dict[str, Any]]:
        """Modifies the :attr:`env` after calling :meth:`reset`, returning a modified
        observation using :meth:`self.observation`."""
        obs, info = self.env.reset(seed=seed, options=options)
        return self.observation(obs), info

    def step(
        self, action: ActType
    ) -> tuple[WrapperObsType, float, bool, bool, dict[str, Any]]:
        observation, reward, terminal, truncated, info = self.step(action)
        reward = self.task.evaluate(self.env)
        return self.observation(observation), reward, terminal, truncated, info

    def observation(self, observation: ObsType):
        """Returns a modified observation.

        Args:
            observation: The :attr:`env` observation

        Returns:
            The modified observation
        """
        return self._append_data_func(observation, self.task.description)
