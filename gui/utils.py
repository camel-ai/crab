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

from crab import (
    Benchmark,
    BenchmarkConfig,
    Task,
    create_benchmark,
    evaluator,
)
from crab.actions.crab_actions import complete
from crab.actions.visual_prompt_actions import (
    get_elements_prompt,
    groundingdino_easyocr,
)
from gui.envs import MAC_ENV, UBUNTU_ENV, WINDOWS_ENV
from gui.host_os import HostOS

_CACHED_HOST_OS = None

def check_host_os() -> HostOS:
    global _CACHED_HOST_OS

    if _CACHED_HOST_OS is None:
        import platform
        host_os = platform.system().lower()

        if host_os == "linux":
            _CACHED_HOST_OS = HostOS.LINUX
        elif host_os == "darwin":
            _CACHED_HOST_OS = HostOS.MAC
        elif host_os == "windows":
            _CACHED_HOST_OS = HostOS.WINDOWS
        else:
            raise ValueError(f"Host OS {host_os} is not supported")

    return _CACHED_HOST_OS

@evaluator(env_name="ubuntu")
def empty_evaluator_linux() -> bool:
    return False


@evaluator(env_name="mac")
def empty_evaluator_mac() -> bool:
    return False


@evaluator(env_name="windows")
def empty_evaluator_windows() -> bool:
    return False


def get_benchmark(task_id: str, task_description: str) -> Benchmark:
    host_os = check_host_os()

    ubuntu_tool = {
        "screenshot": groundingdino_easyocr(font_size=16) >> get_elements_prompt
    }
    mac_tool = {
        "screenshot": groundingdino_easyocr(font_size=24) >> get_elements_prompt
    }

    if host_os == HostOS.LINUX:
        prompting_tools = {"ubuntu": ubuntu_tool}
        benchmark_config = BenchmarkConfig(
            name="ubuntu_benchmark",
            tasks=[
                Task(
                    id=task_id,
                    description=task_description,
                    evaluator=empty_evaluator_linux,
                )
            ],
            environments=[UBUNTU_ENV],
            prompting_tools=prompting_tools,
            root_action_space=[complete],
        )
    elif host_os == HostOS.MAC:
        prompting_tools = {"macos": mac_tool}
        benchmark_config = BenchmarkConfig(
            name="mac_benchmark",
            tasks=[
                Task(
                    id=task_id,
                    description=task_description,
                    evaluator=empty_evaluator_mac,
                )
            ],
            environments=[MAC_ENV],
            prompting_tools=prompting_tools,
            root_action_space=[complete],
        )
    elif host_os == HostOS.WINDOWS:
        prompting_tools = {"windows": ubuntu_tool}
        benchmark_config = BenchmarkConfig(
            name="windows_benchmark",
            tasks=[
                Task(
                    id=task_id,
                    description=task_description,
                    evaluator=empty_evaluator_windows,
                )
            ],
            environments=[WINDOWS_ENV],
            prompting_tools=prompting_tools,
            root_action_space=[complete],
        )
    else:
        raise ValueError(f"Host OS {host_os} is not supported")

    benchmark_config.step_limit = 15
    return create_benchmark(benchmark_config)
