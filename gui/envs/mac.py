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
    press_hotkey,
    right_click,
    screenshot,
    search_application,
    write_text,
)
from crab.core import EnvironmentConfig

MAC_ENV = EnvironmentConfig(
    name="mac",
    action_space=[
        click,
        key_press,
        write_text,
        press_hotkey,
        search_application,
        right_click,
    ],
    observation_space=[screenshot],
    description=(
        "A MacOS desktop operating system. The interface displays a current screenshot "
        "at each step and primarily supports interaction via mouse and keyboard. You "
        "are encouraged to use keyboard shortcuts and searching functionality to open"
        " applications in the system. This device includes system-related applications "
        "like Terminal, Finder, TextEdit, Settings, etc. For communication, Slack is "
        "available. It also features Safari as the web browser. The Google account is "
        "already logged in on Safari, synchronized with the same account used in other "
        "environments."
    ),
    remote_url="http://127.0.0.1:8080",
)
