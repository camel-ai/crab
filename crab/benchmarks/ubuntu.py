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
import os
import subprocess
from time import sleep

import networkx as nx
import psutil

from crab.actions.system_actions import delay
from crab.core import (
    BenchmarkConfig,
    Task,
    action,
    evaluator,
)
from crab.environments.linux import UBUNTU_2204


@evaluator
def is_process_open(process_name: str) -> bool:
    """
    Check if the given process is currently running.

    Args:
        process_name(str): The process name to check.

    Returns:
        bool: True if Firefox is running, False otherwise.
    """
    for process in psutil.process_iter(["name"]):
        try:
            if process_name.lower() in process.info["name"].lower():  # type: ignore
                return True
        except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
            pass
    return False


@action
def close_all_windows() -> None:
    """
    Close all windows. You can use this action to reset the task.
    """
    subprocess.run(["xdotool", "search", "", "windowkill", "%@"])
    sleep(3)


@action
def set_color_schema(scheme: str) -> None:
    subprocess.run(
        ["gsettings", "set", "org.gnome.desktop.interface", "color-scheme", scheme]
    )


@action
def run_command(command: str) -> str:
    out = subprocess.check_output([command], universal_newlines=True)
    return out


@evaluator
def check_color_scheme(assmue: str) -> bool:
    out = subprocess.check_output(
        ["gsettings", "get", "org.gnome.desktop.interface", "color-scheme"],
        universal_newlines=True,
    )
    return assmue in out


@evaluator
def check_file_exist(file_path: str) -> bool:
    return os.path.isfile(file_path)


@evaluator
def empty_evaluator() -> bool:
    return False


@action
def remove_file_if_exist(file_path: str) -> bool:
    if os.path.isfile(file_path):
        os.unlink(file_path)
        return True
    return False


CHANGE_DARK_MODE = Task(
    id="0",
    description='Change the system to the dark mode by the "Settings" application.',
    evaluator=nx.path_graph(
        [is_process_open("gnome-control-center"), check_color_scheme("prefer-dark")],
        create_using=nx.DiGraph,
    ),
    setup=set_color_schema("prefer-light") + close_all_windows,
)

OPEN_FIREFOX = Task(
    id="1",
    description="Open Firefox using the mouse.",
    evaluator=is_process_open("firefox"),
    setup=close_all_windows,
)

VIM = Task(
    id="2",
    description="First, open a terminal, then open ~/poem using vim, then write a "
    "random poem using your knowledge, then save and exit vim.",
    evaluator=check_file_exist("/home/crab/poem"),
    setup=remove_file_if_exist("/home/crab/poem")
    + close_all_windows
    + run_command("gnome-terminal")
    + delay(1),
)

SLACK = Task(
    id="3",
    description='Search and open "slack" applicaiton, find "test-agent" '
    "channel in the list, then send a poem in this channel.",
    evaluator=empty_evaluator,
)

ubuntu_benchmark = BenchmarkConfig(
    name="ubuntu_benchmark",
    tasks=[CHANGE_DARK_MODE, OPEN_FIREFOX, VIM, SLACK],
    environments=[UBUNTU_2204],
)
