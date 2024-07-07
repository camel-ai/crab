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

UBUNTU_ENV = EnvironmentConfig(
    name="ubuntu",
    action_space=[
        click,
        key_press,
        write_text,
        press_hotkey,
        search_application,
        right_click,
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
