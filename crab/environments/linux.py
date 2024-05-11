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
from crab.actions.desktop_actions import (
    click,
    key_press,
    screenshot,
    search_application,
    write_text,
)
from crab.core import EnvironmentConfig

UBUNTU_2204 = EnvironmentConfig(
    name="ubuntu2204",
    action_space=[click, key_press, write_text, search_application],
    observation_space=[screenshot],
    description="A Ubuntu 22.04 desktop environment with a single display.",
)
