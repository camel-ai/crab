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
from crab import action
from crab.actions.crab_actions import complete, get_element_position
from crab.actions.desktop_actions import (
    click_position,
    key_press,
    press_hotkey,
    right_click,
    screenshot,
    write_text,
)
from crab.core import EnvironmentConfig


@action(local=True)
def click(element: int, env) -> None:
    """
    Click an UI element shown on the desktop screen. A simple use case can be
    click(5), which clicks the UI element labeled with the number 5.

    Args:
        element: A numeric tag assigned to an UI element shown on the screenshot.
    """
    x, y = get_element_position(element, env)
    env._action_endpoint(click_position, {"x": round(x / 2), "y": round(y / 2)})


mac_env = EnvironmentConfig(
    name="macos",
    action_space=[click, key_press, write_text, press_hotkey, right_click, complete],
    observation_space=[screenshot],
    description="A Macbook laptop environment with a single display.",
)
