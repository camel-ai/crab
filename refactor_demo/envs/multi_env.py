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
import gymnasium as gym
import numpy as np
from gymnasium import spaces


class MultiEnv(gym.Env):
    def __init__(self, envs):
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

        # Create observation space as a Dict space containing each environment's observation space
        self.observation_space = spaces.Dict(
            {f"env_{i}": env.observation_space for i, env in enumerate(envs)}
        )

    def reset(self):
        """
        Reset all environments and return initial observations.

        Returns:
            dict: A dictionary with initial observations from each environment.
        """
        observations = {}
        for i, env in enumerate(self.envs):
            observations[f"env_{i}"], _ = env.reset()
        return observations

    def step(self, action):
        """
        Take a step in the selected environment based on the action.

        Args:
            action (int): The index of the environment to take a step in.

        Returns:
            tuple: A tuple containing the observations, rewards, done flags, and info.
        """
        assert 0 <= action < len(self.envs), "Invalid action for environment selection."

        # Initialize dictionaries to store results
        observations = {}
        rewards = {}
        dones = {}
        infos = {}

        # Perform a step in the selected environment
        obs, reward, done, truncated, info = self.envs[action].step(action)

        # Populate results for the selected environment
        observations[f"env_{action}"] = obs
        rewards[f"env_{action}"] = reward
        dones[f"env_{action}"] = done
        infos[f"env_{action}"] = info

        # For other environments, simply pass their previous observations
        for i, env in enumerate(self.envs):
            if i != action:
                observations[f"env_{i}"] = (
                    None  # No new observation for non-acting environments
                )
                rewards[f"env_{i}"] = 0
                dones[f"env_{i}"] = False
                infos[f"env_{i}"] = {}

        # Set done if all environments are done
        all_done = all(dones.values())

        return observations, rewards, all_done, infos

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
