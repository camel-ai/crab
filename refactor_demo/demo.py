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
from typing import Any, Iterator

import gymnasium as gym
from pydantic import BaseModel


class Action:
    def __init__(
        self,
        name: str,
        description: str,
        parameters: type[BaseModel],
    ):
        pass

    def run(self, parameters: dict[str, Any], env: gym.Env | None):
        pass


class ActionSpace(BaseModel):
    description: str
    action_schema: dict[str, Any]

    def to_prompt(self) -> str:
        pass

    def to_gym_space(self) -> gym.Space:
        pass


class MultiEnvironment(gym.Env):
    def __init__(self):
        super().__init__()


class ActionExecutor:
    pass


class UbuntuEnvironment(gym.Env):
    def __init__(
        self,
        name: str,
        action_space: Iterator[Action],
        description: str = "",
        action_executor: ActionExecutor | None = None,
        **kwargs: Any,
    ) -> None:
        pass

    def _get_obs(self) -> gym.Space:
        pass

    def reset(self, *, seed=None, options=None):
        return super().reset(seed=seed, options=options)

    def step(self, action):
        return super().step(action)

    def render(self):
        return super().render()
