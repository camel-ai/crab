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
    long_tap,
    open_app_drawer,
    screenshot,
    setup,
    swipe,
    tap,
    write_text,
)

ANDROID_ENV = EnvironmentConfig(
    name="android",
    action_space=[tap, key_press, long_tap, write_text, swipe, open_app_drawer],
    observation_space=[screenshot],
    description="""A Google Pixel smartphone runs on the Android operating system. \
The interface displays a current screenshot at each step and primarily \
supports interaction through tapping and typing. This device offers a suite \
of standard applications including Phone, Photos, Camera, Chrome, and \
Calendar, among others. Access the app drawer to view all installed \
applications on the device. The Google account is pre-logged in, synchronized \
with the same account used in the Ubuntu environment.""",
    extra_attributes={"device": None},
    reset=setup,
)
