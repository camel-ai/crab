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
import warnings

from crab import (
    BenchmarkConfig,
    create_benchmark,
)
from crab.actions.crab_actions import complete
from crab.actions.visual_prompt_actions import (
    get_elements_prompt,
    groundingdino_easyocr,
)
from crab.environments.macos import mac_env
from gui.envs import ANDROID_ENV, UBUNTU_ENV, WINDOWS_ENV
from gui.host_os import HostOS

warnings.filterwarnings("ignore")


def check_host_os() -> HostOS:
    return HostOS.WINDOWS


def get_benchmark(env: str, ubuntu_url: str):
    ubuntu_tool = {
        "screenshot": groundingdino_easyocr(font_size=16) >> get_elements_prompt
    }
    android_tool = {
        "screenshot": groundingdino_easyocr(font_size=40) >> get_elements_prompt
    }
    mac_tool = {
        "screenshot": groundingdino_easyocr(font_size=24) >> get_elements_prompt
    }

    if env == "ubuntu":
        prompting_tools = {"ubuntu": ubuntu_tool}
        benchmark_config = BenchmarkConfig(
            name="ubuntu_benchmark",
            tasks=[],
            environments=[UBUNTU_ENV],
            prompting_tools=prompting_tools,
            root_action_space=[complete],
            multienv=True,
        )
    elif env == "android":
        prompting_tools = {"android": android_tool}
        benchmark_config = BenchmarkConfig(
            name="android_benchmark",
            tasks=[],
            environments=[ANDROID_ENV],
            prompting_tools=prompting_tools,
            root_action_space=[complete],
            multienv=True,
        )
    elif env == "cross":
        prompting_tools = {
            "android": android_tool,
            "ubuntu": ubuntu_tool,
        }
        benchmark_config = BenchmarkConfig(
            name="ubuntu_android_benchmark",
            tasks=[],
            environments=[UBUNTU_ENV, ANDROID_ENV],
            prompting_tools=prompting_tools,
            root_action_space=[complete],
            multienv=True,
        )
    elif env == "mac":
        prompting_tools = {"macos": mac_tool}
        benchmark_config = BenchmarkConfig(
            name="mac_benchmark",
            tasks=[],
            environments=[mac_env, ANDROID_ENV],
            prompting_tools=prompting_tools,
            root_action_space=[complete],
            multienv=True,
        )
    elif env == "windows":
        prompting_tools = {"windows": ubuntu_tool}
        benchmark_config = BenchmarkConfig(
            name="windows_benchmark",
            tasks=[],
            environments=[WINDOWS_ENV],
            prompting_tools=prompting_tools,
            root_action_space=[complete],
            multienv=True,
        )
    else:
        raise ValueError("Env not support")

    benchmark_config.step_limit = 15
    return create_benchmark(benchmark_config)


def main():
    host_os = check_host_os()
    print(f"Host OS: {host_os}")


if __name__ == "__main__":
    main()
