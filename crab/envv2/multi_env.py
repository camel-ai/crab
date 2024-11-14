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
from gymnasium import spaces

from crab.corev2.environment import CrabEnvironment


class MultiEnv(gym.Env[dict[str, Any], tuple[int, Any]]):
    def __init__(self, envs: dict[str, CrabEnvironment]):
        """
        Initialize the MultiEnv environment.

        Args:
            envs (list): A list of gymnasium environments to integrate.
        """
        super().__init__()

        # Store the environments
        self.envs = envs
        # Create action space using OneOf with the action spaces of each environment
        self.action_space = spaces.OneOf([env.action_space for env in envs])

        # Create observation space as a Dict space containing each environment's
        # observation space
        self.observation_space = spaces.Dict(
            {name: env.observation_space for name, env in envs.items()}
        )

        self._idx_to_env_name = {
            idx: env_name for idx, env_name in enumerate(envs.keys())
        }
        self._env_name_to_index = {
            env_name: idx for idx, env_name in enumerate(envs.keys())
        }

        self._saved_observations = {key: None for key in envs.keys()}
        self._saved_infos = {key: None for key in envs.keys()}
        self._saved_dones = {key: False for key in envs.keys()}

    def reset(self):
        """
        Reset all environments and return initial observations.

        Returns:
            dict: A dictionary with initial observations from each environment.
        """
        observations = {}
        infos = {}
        for name, env in self.envs.items():
            observations[name], infos[name] = env.reset()
        self._saved_observations = observations
        self._saved_infos = infos
        return observations, infos

    def step(self, action: tuple[int, Any]):
        """
        Take a step in the selected environment based on the action.

        Args:
            action: The index of the environment to take a step in.

        Returns:
            tuple: A tuple containing the observations, rewards, done flags, and info.
        """
        env_idx, actual_action = action
        env_name = self._idx_to_env_name[env_idx]
        assert (
            0 <= env_idx < len(self.envs)
        ), "Invalid action for environment selection."
        assert self.action_space[env_idx].contains(
            actual_action
        ), f"{actual_action!r} ({type(actual_action)}) invalid in {env_name}"

        env = self.envs[env_idx]

        reward = 0  # No reward in bare MultiEnv

        # Perform a step in the selected environment
        obs, reward, done, truncated, info = env.step(actual_action)

        # Populate results for the selected environment
        self._saved_observations[env_name] = obs
        self._saved_dones[env_name] = done
        self._saved_infos[env_name] = info

        # Set done if all environments are done
        all_done = all(self._saved_dones.values())

        return self._saved_observations, reward, all_done, truncated, self._saved_infos

    def render(self, mode="human"):
        """
        Render all environments (optional implementation).
        """
        for i, env in enumerate(self.envs):
            env.render(mode=mode)

    def close(self):
        """
        Close all environments.
        """
        for env in self.envs:
            env.close()
