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
import warnings
from pathlib import Path
from typing import Literal
from uuid import uuid4

from crab import (
    BenchmarkConfig,
    Experiment,
    MessageType,
    Task,
    create_benchmark,
)
from crab.actions.crab_actions import complete
from crab.actions.visual_prompt_actions import (
    get_elements_prompt,
    groundingdino_easyocr,
)
from crab.agents.backend_models import OpenAIModel
from crab.agents.policies import (
    SingleAgentPolicy,
)
from crab.core.agent_policy import AgentPolicy
from crab.core.benchmark import Benchmark
from crab.core.decorators import evaluator
from crab.environments.macos import mac_env

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


@evaluator(env_name="macos")
def empty_evaluator() -> bool:
    return False


def get_mac_benchmark_local():
    mac_env.remote_url = "http://localhost:8000"
    mac_tool = {
        "screenshot": groundingdino_easyocr(font_size=24) >> get_elements_prompt
    }
    prompting_tools = {"macos": mac_tool}
    benchmark_config = BenchmarkConfig(
        name="mac_benchmark",
        tasks=[],
        environments=[mac_env],
        prompting_tools=prompting_tools,
        root_action_space=[complete],
        multienv=True,
    )

    benchmark_config.step_limit = 15
    return create_benchmark(benchmark_config)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Script for running benchmark with an agent."
    )
    parser.add_argument(
        "--task-description",
        type=str,
        help="task description. If provided, will overwrite the task id.",
        required=True,
    )
    args = parser.parse_args()
    benchmark = get_mac_benchmark_local()

    task_id = str(uuid4())
    benchmark.tasks = [
        Task(
            id=task_id,
            description=args.task_description,
            evaluator=empty_evaluator,
        )
    ]

    history_messages_len = 2
    model = OpenAIModel(model="gpt-4o", history_messages_len=history_messages_len)
    agent_policy = SingleAgentPolicy(model_backend=model)

    log_dir = (Path(__file__).parent / "logs").resolve()
    expeirment = CrabBenchmarkV0(
        benchmark=benchmark,
        task_id=task_id,
        agent_policy=agent_policy,
        log_dir=log_dir,
    )
    expeirment.start_benchmark()


"""
python -m crab.server.main --HOST 0.0.0.0
python -m crab-benchmark-v0.macos_local
"""
