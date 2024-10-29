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
    double_click,
    key_press,
    press_hotkey,
    right_click,
    screenshot,
    search_application,
    write_text,
)
from crab.core import EnvironmentConfig

UBUNTU_ENV = EnvironmentConfig(
    name="ubuntu",
    action_space=[
        click,
        key_press,
        write_text,
        press_hotkey,
        search_application,
        right_click,
        double_click,
    ],
    observation_space=[screenshot],
    description="""An Ubuntu 22.04 Linux desktop operating system. The interface \
displays a current screenshot at each step and primarily supports interaction \
via mouse and keyboard. You must use searching functionality to open any \
application in the system. This device includes system-related applications \
including Terminal, Files, Text Editor, Vim, and Settings. It also features \
Firefox as the web browser, and the LibreOffice suite—Writer, Calc, and \
Impress. For communication, Slack is available. The Google account is \
pre-logged in on Firefox, synchronized with the same account used in the \
Android environment.""",
)
