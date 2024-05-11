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
from crab.core import Environment, EnvironmentConfig, action


@action
def set_state(value: bool, env: Environment) -> None:
    """
    Set system state to the given value.

    Args:
        value (bool): The given value to set the system state.
    """
    env.state = value


@action
def current_state(env: Environment) -> bool:
    """
    Get current system state.
    """
    return env.state


template_environment_config = EnvironmentConfig(
    name="template_env",
    action_space=[set_state],
    observation_space=[current_state],
    description="A test environment",
    info=None,
    reset=set_state(False),
)
