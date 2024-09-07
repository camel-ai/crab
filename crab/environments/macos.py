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
