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
import argparse
import logging
import warnings
from pathlib import Path
from typing import Literal

from crab import (
    BenchmarkConfig,
    Experiment,
    MessageType,
    TaskGenerator,
    create_benchmark,
)
from crab.actions.crab_actions import complete, wait
from crab.actions.visual_prompt_actions import (
    get_elements_prompt,
    groundingdino_easyocr,
)
from crab.agents.backend_models import BackendModelConfig
from crab.agents.policies import (
    MultiAgentByEnvPolicy,
    MultiAgentByFuncPolicy,
    SingleAgentPolicy,
)
from crab.core.agent_policy import AgentPolicy
from crab.core.benchmark import Benchmark

from .android_env import ANDROID_ENV
from .dataset.android_subtasks import android_subtasks
from .dataset.handmade_tasks import handmade_tasks
from .dataset.ubuntu_subtasks import ubuntu_subtasks
from .ubuntu_env import UBUNTU_ENV

warnings.filterwarnings("ignore")


class CrabBenchmarkV0(Experiment):
    def __init__(
        self,
        benchmark: Benchmark,
        task_id: str,
        agent_policy: AgentPolicy | Literal["human"],
        log_dir: Path | None = None,
    ) -> None:
        super().__init__(benchmark, task_id, agent_policy, log_dir)

    def get_prompt(self):
        observation, ob_prompt = self.benchmark.observe_with_prompt()

        # construct prompt
        result_prompt = {}
        for env in ob_prompt:
            if env == "root":
                continue
            screenshot = observation[env]["screenshot"]
            marked_screenshot, _ = ob_prompt[env]["screenshot"]
            result_prompt[env] = [
                (f"Here is the current screenshot of {env}:", MessageType.TEXT),
                (screenshot, MessageType.IMAGE_JPG_BASE64),
                (
                    f"Here is the screenshot with element labels of {env}:",
                    MessageType.TEXT,
                ),
                (marked_screenshot, MessageType.IMAGE_JPG_BASE64),
            ]
        return result_prompt


def get_benchmark(env: str, ubuntu_url: str):
    ubuntu_env = UBUNTU_ENV.model_copy()
    ubuntu_env.remote_url = ubuntu_url
    ubuntu_tool = {
        "screenshot": groundingdino_easyocr(font_size=16) >> get_elements_prompt
    }
    android_tool = {
        "screenshot": groundingdino_easyocr(font_size=40) >> get_elements_prompt
    }

    if env == "ubuntu":
        prompting_tools = {"ubuntu": ubuntu_tool}
        benchmark_config = BenchmarkConfig(
            name="ubuntu_benchmark",
            tasks=[],
            environments=[ubuntu_env],
            prompting_tools=prompting_tools,
            root_action_space=[complete, wait],
            multienv=True,
        )
    elif env == "android":
        prompting_tools = {"android": android_tool}
        benchmark_config = BenchmarkConfig(
            name="android_benchmark",
            tasks=[],
            environments=[ANDROID_ENV],
            prompting_tools=prompting_tools,
            root_action_space=[complete, wait],
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
            environments=[ubuntu_env, ANDROID_ENV],
            prompting_tools=prompting_tools,
            root_action_space=[complete, wait],
            multienv=True,
        )
    else:
        raise ValueError("Env not support")

    # Load from json config files by combining sub-tasks
    generator = TaskGenerator(subtasks=android_subtasks + ubuntu_subtasks)
    dir_path = (Path(__file__).parent / "dataset").resolve()
    tasks = []
    for task_json_files in dir_path.rglob("*.json"):
        task = generator.get_task_from_file(task_json_files)
        tasks.append(task)
    benchmark_config.tasks.extend(tasks)

    # Load from handmade tasks
    benchmark_config.tasks.extend(handmade_tasks)

    benchmark_config.step_limit = 20
    return create_benchmark(benchmark_config)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Script for running benchmark with an agent."
    )
    parser.add_argument(
        "--model",
        type=str,
        help="gpt4o, gpt4turbo, gemini, claude or human",
        default="gpt4o",
    )
    parser.add_argument(
        "--policy",
        type=str,
        help="single, multi-by-func, or multi-by-env",
        default="single",
    )
    parser.add_argument(
        "--ubuntu-url",
        type=str,
        help="remote url of Ubunutu environment",
        default="http://127.0.0.1:8000",
    )
    parser.add_argument(
        "--env",
        type=str,
        help="ubuntu, android or cross",
        default="cross",
    )
    parser.add_argument("--task-id", type=str, help="task id")
    parser.add_argument(
        "--model-base-url",
        type=str,
        help="URL of the model API",
        default="http://127.0.0.1:8000/v1",
    )
    parser.add_argument(
        "--model-api-key",
        type=str,
        help="API key of the model API",
        default="EMPTY",
    )
    parser.add_argument(
        "--loglevel",
        type=str,
        help="logger level, debug, info, warning, or error",
        default="warning",
    )
    parser.add_argument(
        "--history-messages-len",
        type=int,
        help="The number of rounds of chat history to provide to the model",
        default=2,
    )
    args = parser.parse_args()
    loglevel = args.loglevel
    numeric_level = getattr(logging, loglevel.upper(), None)
    if not isinstance(numeric_level, int):
        raise ValueError("Invalid log level: %s" % loglevel)
    logging.basicConfig(level=numeric_level)

    benchmark = get_benchmark(args.env, args.ubuntu_url)

    if args.model == "human":
        expeirment = CrabBenchmarkV0(
            benchmark=benchmark,
            task_id=args.task_id,
            agent_policy="human",
        )
        expeirment.start_benchmark()
        exit()

    if args.model == "gpt4o":
        model = BackendModelConfig(
            model_class="openai",
            model_name="gpt-4o",
            history_messages_len=args.history_messages_len,
        )
    elif args.model == "gpt4turbo":
        model = BackendModelConfig(
            model_class="openai",
            model_name="gpt-4-turbo",
            history_messages_len=args.history_messages_len,
        )
    elif args.model == "gemini":
        model = BackendModelConfig(
            model_class="gemini",
            model_name="gemini-1.5-pro-latest",
            history_messages_len=args.history_messages_len,
        )
    elif args.model == "claude":
        model = BackendModelConfig(
            model_class="claude",
            model_name="claude-3-opus-20240229",
            history_messages_len=args.history_messages_len,
        )
    elif args.model == "pixtral":
        model = BackendModelConfig(
            model_class="openai",
            model_name="mistralai/Pixtral-12B-2409",
            json_structre_output=True,
            history_messages_len=args.history_messages_len,
            base_url=args.model_base_url,
            api_key=args.model_api_key,
        )
    elif args.model == "gpt4o-wofc":
        model = BackendModelConfig(
            model_class="openai",
            model_name="gpt-4o",
            json_structre_output=True,
            history_messages_len=args.history_messages_len,
        )
    elif args.model == "llava-ov72b":
        model = BackendModelConfig(
            model_class="sglang",
            model_name="lmms-lab/llava-onevision-qwen2-72b-ov-chat",
            json_structre_output=True,
            history_messages_len=args.history_messages_len,
            base_url=args.model_base_url,
            api_key=args.model_api_key,
        )
    else:
        print("Unsupported model: ", args.model)
        exit()

    if args.policy == "single":
        agent_policy = SingleAgentPolicy(model_backend=model)
    elif args.policy == "multi-by-func":
        agent_policy = MultiAgentByFuncPolicy(
            main_agent_model_backend=model, tool_agent_model_backend=model
        )
    elif args.policy == "multi-by-env":
        agent_policy = MultiAgentByEnvPolicy(
            main_agent_model_backend=model, env_agent_model_backend=model
        )
    else:
        print("Unsupported policy: ", args.policy)
        exit()

    log_dir = (Path(__file__).parent / "tianqi_logs").resolve()
    expeirment = CrabBenchmarkV0(
        benchmark=benchmark,
        task_id=args.task_id,
        agent_policy=agent_policy,
        log_dir=log_dir,
    )
    expeirment.start_benchmark()
