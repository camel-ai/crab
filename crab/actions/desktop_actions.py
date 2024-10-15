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
import time
from enum import Enum

import pyautogui
from mss import mss, tools

from crab import action

from .crab_actions import get_element_position

DURATION = 0.8
DELAY = 1.0


@action
def click_position(x: int, y: int) -> None:
    """
    click on the current desktop screen.

    Args:
        x: The X coordinate, as a floating-point number in the range [0.0, 1.0].
        y: The Y coordinate, as a floating-point number in the range [0.0, 1.0].
    """
    pyautogui.click(x, y, duration=DURATION)
    time.sleep(DELAY)


@action(local=True)
def click(element: int, env) -> None:
    """
    Click an UI element shown on the desktop screen. A simple use case can be
    click(5), which clicks the UI element labeled with the number 5.

    Args:
        element: A numeric tag assigned to an UI element shown on the screenshot.
    """
    x, y = get_element_position(element, env)
    env._action_endpoint(click_position, {"x": x, "y": y})


@action
def right_click_position(x: int, y: int) -> None:
    """
    right-click on the current desktop screen.

    Args:
        x: The X coordinate, as a floating-point number in the range [0.0, 1.0].
        y: The Y coordinate, as a floating-point number in the range [0.0, 1.0].
    """
    pyautogui.click(x, y, duration=DURATION, button="right")


@action(local=True)
def right_click(element: int, env) -> None:
    """
    Right-click an UI element shown on the desktop screen using the mouse, which is
    usually used for opening the menu of the element. A simple use case can be
    right_click(5), which right-clicks the UI element labeled with the number 5 to open
    up menu on it.

    Args:
        element: A numeric tag assigned to an UI element shown on the screenshot.
    """
    x, y = get_element_position(element, env)
    env._action_endpoint(right_click_position, {"x": x, "y": y})
    time.sleep(DELAY)


@action
def double_click_position(x: int, y: int) -> None:
    """
    Double-click on the current desktop screen.

    Args:
        x: The X coordinate, as a floating-point number in the range [0.0, 1.0].
        y: The Y coordinate, as a floating-point number in the range [0.0, 1.0].
    """
    pyautogui.click(x, y, duration=DURATION, clicks=2, interval=0.2)


@action(local=True)
def double_click(element: int, env) -> None:
    """
    Double-click an UI element shown on the desktop screen using the mouse, which is
    usually used for opening a folder or a file. A simple use case can be
    double_click(5), which double-clicks the UI element labeled with the number 5 to
    open it.

    Args:
        element: A numeric tag assigned to an UI element shown on the screenshot.
    """
    x, y = get_element_position(element, env)
    env._action_endpoint(double_click_position, {"x": x, "y": y})
    time.sleep(DELAY)


@action
def mouse_scroll(click: int = 1) -> None:
    """
    Performs a scroll of the mouse scroll wheel.

    Args:
        click(int): The amount of scrolling. Default to 1.
    """
    pyautogui.scroll(click)
    time.sleep(DELAY)


class KeyEnum(str, Enum):
    KEY_TAB = "\t"
    KEY_LB = "\n"
    KEY_RR = "\r"
    KEY_SPACE = " "
    KEY_EXCLAMATION = "!"
    KEY_DQUOTE = '"'
    KEY_SHARP = "#"
    KEY_DOLLAR = "$"
    KEY_PER = "%"
    KEY_AND = "&"
    KEY_SQUOTE = "'"
    KEY_LPAR = "("
    KEY_RPAR = ")"
    KEY_MUL = "*"
    KEY_ADD = "+"
    KEY_COMMA = ","
    KEY_MIN = "-"
    KEY_DOT = "."
    KEY_SLASH = "/"
    KEY_0 = "0"
    KEY_1 = "1"
    KEY_2 = "2"
    KEY_3 = "3"
    KEY_4 = "4"
    KEY_5 = "5"
    KEY_6 = "6"
    KEY_7 = "7"
    KEY_8 = "8"
    KEY_9 = "9"
    KEY_COL = ":"
    KEY_SEMICOL = ";"
    KET_LT = "<"
    KEY_EQUAL = "="
    KEY_GT = ">"
    KEY_QM = "?"
    KEY_AT = "@"
    KEY_LBRA = "["
    KEY_RSLASH = "\\"
    KEY_RBRA = "]"
    KEY_CARET = "^"
    KEY_UNDERLINE = "_"
    KEY_BACKTICK = "`"
    KEY_LBRACE = "{"
    KEY_RBRACE = "}"
    KEY_PIPE = "|"
    KEY_TLIDE = "~"
    KEY_A = "a"
    KEY_B = "b"
    KEY_C = "c"
    KEY_D = "d"
    KEY_E = "e"
    KEY_F = "f"
    KEY_G = "g"
    KEY_H = "h"
    KEY_I = "i"
    KEY_J = "j"
    KEY_K = "k"
    KEY_L = "l"
    KEY_M = "m"
    KEY_N = "n"
    KEY_O = "o"
    KEY_P = "p"
    KEY_Q = "q"
    KEY_R = "r"
    KEY_S = "s"
    KEY_T = "t"
    KEY_U = "u"
    KEY_V = "v"
    KEY_W = "w"
    KEY_X = "x"
    KEY_Y = "y"
    KEY_Z = "z"
    KEY_ALT = "alt"
    KEY_SHIFT = "shift"
    KEY_CTRL = "ctrl"
    KEY_WIN = "win"
    KEY_BACKSPACE = "backspace"
    KEY_ENTER = "enter"
    KEY_ESC = "esc"
    KEY_F1 = "f1"
    KEY_F2 = "f2"
    KEY_F3 = "f3"
    KEY_F4 = "f4"
    KEY_F5 = "f5"
    KEY_F6 = "f6"
    KEY_F7 = "f7"
    KEY_F8 = "f8"
    KEY_F9 = "f9"
    KEY_F10 = "f10"
    KEY_F11 = "f11"
    KEY_F12 = "f12"
    KEY_LEFT = "left"
    KEY_UP = "up"
    KEY_RIGHT = "right"
    KEY_DOWN = "down"


@action
def key_press(key: KeyEnum) -> None:
    """
    Performs a keyboard key press down, followed by a release.

    Args:
        key (str): The key to be pressed.
    """
    if isinstance(key, KeyEnum):
        pyautogui.press(key.value)
    else:
        pyautogui.press(key)
    time.sleep(DELAY)


@action
def press_hotkey(keys: list[KeyEnum]) -> None:
    """
    Press multiple keyboard keys at the same time. For exmaple, if you want to use
    Ctrl-C hoykey to copy the selected text, you can call
    press_hotkey(keys=["ctrl", "c"]).

    Args:
        key (str): The key to be pressed.
    """
    if isinstance(keys[0], KeyEnum):
        keys = [key.value for key in keys]
    pyautogui.hotkey(*keys)
    time.sleep(DELAY)


@action
def write_text(text: str) -> None:
    """
    Typing the specified text. Note: This function does not move the mouse cursor.
    Ensure the cursor focuses in the correct text input field before calling this
    function.

    Args:
        text (str): The text to be typed.
    """
    pyautogui.write(text, interval=0.03)
    time.sleep(DELAY)


@action
def search_application(name: str) -> None:
    """
    Search an application name. For exmaple, if you want to open an application named
    "slack", you can call search_application(name="slack"). You MUST use this action to
    search for applications.

    Args:
        name: the application name.
    """
    pyautogui.press("esc")
    time.sleep(DELAY)
    pyautogui.hotkey("win", "a")
    time.sleep(DELAY)
    pyautogui.write(name)
    time.sleep(DELAY)


@action
def screenshot() -> str:
    "Get the current screenshot."
    with mss() as sct:
        # Get raw pixels from the screen
        sct_img = sct.grab(sct.monitors[1])
        # Create the Image
        png = tools.to_png(sct_img.rgb, sct_img.size)
        base64_img = base64.b64encode(png).decode("utf-8")
    return base64_img
