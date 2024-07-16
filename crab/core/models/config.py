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

from pydantic import BaseModel

from .action import Action, ClosedAction
from .task import Task


class EnvironmentConfig(BaseModel):
    name: str
    action_space: list[Action]
    observation_space: list[ClosedAction]
    description: str = ""
    reset: Action | None = None
    remote_url: str | None = None
    extra_attributes: dict[str, Any] = {}


class VMEnvironmentConfig(BaseModel):
    inside_environment: EnvironmentConfig
    remote_url: str = "http://192.168.0.0:8000"


class BenchmarkConfig(BaseModel):
    name: str
    tasks: list[Task]
    environments: list[EnvironmentConfig]
    default_env: str | None = None
    multienv: bool = False
    prompting_tools: dict[str, dict[str, Action]] = {}
    root_action_space: list[Action] = []
    step_limit: int = 30
    common_setup: list[ClosedAction] = []
