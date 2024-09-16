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
import base64
import subprocess
from enum import Enum
from time import sleep

from crab import action

from .crab_actions import get_element_position


def execute_adb(adb_command: str, env=None):
    if env.device is None:
        adb_command = "adb " + adb_command
    else:
        adb_command = f"adb -s {env.device} " + adb_command
    result = subprocess.run(
        adb_command,
        shell=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    if result.returncode == 0:
        return result.stdout.strip()
    print(f"Command execution failed: {adb_command}")
    print(result.stderr)
    return "ERROR"


def get_device_size(env):
    adb_command = "shell wm size"
    result = execute_adb(adb_command, env)
    if result != "ERROR":
        return map(int, result.split(": ")[1].split("x"))
    return 0, 0


_DURATION = 1.5


@action
def setup(env) -> None:
    env.width, env.height = get_device_size(env)


@action
def screenshot(env) -> str:
    """
    Get the current screenshot of phone screen.
    """
    if env.device is not None:
        command = f"adb -s {env.device} exec-out screencap -p"
    else:
        command = "adb exec-out screencap -p"
    result = subprocess.run(
        command,
        shell=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    return base64.b64encode(result.stdout).decode("utf-8")


@action
def tap(element: int, env) -> None:
    """
    Tap an UI element shown on the smartphone screen. A simple use case can be tap(5),
    which taps the UI element labeled with the number 5.

    Args:
        element: A numeric tag assigned to an UI element shown on the smartphone screen.
    """
    x, y = get_element_position(element, env)
    execute_adb(f"shell input tap {x} {y}", env)
    sleep(_DURATION)


@action
def long_tap(element: int, env) -> None:
    """
    Press and hold a UI element on the smartphone screen for 1 second, typically to
    access additional menu options. For example, the command long_tap(5) simulates a
    long press on the UI element labeled with the number 5.

    Args:
        element: A numeric tag assigned to an UI element shown on the smartphone screen.
    """
    x, y = get_element_position(element, env)
    adb_command = f"shell input swipe {x} {y} {x} {y} 1000"
    execute_adb(adb_command, env)
    sleep(_DURATION)


class SwipeDirection(str, Enum):
    RIGHT = "right"
    LEFT = "left"
    UP = "up"
    DOWN = "down"


class SwipeDist(str, Enum):
    SHORT = "short"
    MEDIUM = "medium"
    LONG = "long"


@action
def swipe(element: int, direction: SwipeDirection, dist: SwipeDist, env) -> None:
    """
    This function is used to swipe an UI element shown on the smartphone screen, usually
    a scroll view or a slide bar. You should choose the appropriate direction and
    distance option according to your need. A simple use case can be swipe(21, "up",
    "medium"), which swipes up the UI element labeled with the number 21 for a medium
    distance.

    Args:
        element: is a numeric tag assigned to an UI element shown on the smartphone
            screen.
        direction: is a string that represents the swipe direction
        dist: determines the distance of the swipe.
    """
    x, y = get_element_position(element, env)
    unit_dist = int(env.width / 10)
    if dist == "long":
        unit_dist *= 3
    elif dist == "medium":
        unit_dist *= 2
    if direction == "up":
        offset = 0, -2 * unit_dist
    elif direction == "down":
        offset = 0, 2 * unit_dist
    elif direction == "left":
        offset = -1 * unit_dist, 0
    elif direction == "right":
        offset = unit_dist, 0
    else:
        return "ERROR"
    adb_command = f"shell input swipe {x} {y} {x + offset[0]} {y + offset[1]} 200"
    execute_adb(adb_command, env)
    sleep(_DURATION)


@action
def open_app_drawer(env) -> None:
    """Open app drawer to list all the installed applications in this phone. For
    exmaple: you want to open "Messages" application, but you don't know where to find
    it, you can call "open_app_drawer()" and you will see all the installed applications
    through screenshot.
    """
    execute_adb("shell input keyevent KEYCODE_HOME", env)
    sleep(0.5)
    execute_adb("shell input swipe 800 2000 800 100 500", env)
    sleep(_DURATION)


class AndroidKey(str, Enum):
    HOME = "home"
    BACK = "back"


@action
def key_press(key: AndroidKey, env):
    """
    Press Android keys. press("home") to go back to main screen. press("back") to return
    to the preivous page.

    Args:
        key (str): The pressed key.
    """
    if key == AndroidKey.HOME:
        adb_command = "shell input keyevent KEYCODE_HOME"
    elif key == AndroidKey.BACK:
        adb_command = "shell input keyevent KEYCODE_BACK"
    else:
        raise ValueError("Unsupported key")
    execute_adb(adb_command, env)
    sleep(_DURATION)


@action
def write_text(text: str, env) -> None:
    """
    Typing the specified text.

    Args:
        text (str): The text to be typed.
    """
    text = text.replace(" ", "%s")
    text = text.replace("'", "")
    adb_command = f"shell input text {text}"
    execute_adb(adb_command, env)
    sleep(_DURATION)


@action
def stop_all_apps(env) -> None:
    """
    Stop all running apps.
    """
    execute_adb("shell input keyevent KEYCODE_HOME", env)
    execute_adb("shell input keyevent KEYCODE_APP_SWITCH", env)
    sleep(0.5)
    command = (
        f"shell input swipe 100 {env.height / 2} {env.width - 100} {env.height / 2} 200"
    )
    execute_adb(command, env)
    sleep(0.5)
    execute_adb("shell input tap 300 1400", env)
    sleep(_DURATION)
