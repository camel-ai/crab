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
from crab import EnvironmentConfig
from crab.actions.android_actions import (
    key_press,
    open_application_panel,
    screenshot,
    setup,
    swipe,
    tap,
    write_text,
)

android_env = EnvironmentConfig(
    name="android",
    action_space=[tap, key_press, write_text, swipe, open_application_panel],
    observation_space=[screenshot],
    description="An Android device",
    extra_attributes={"device": None},
    reset=setup,
)
