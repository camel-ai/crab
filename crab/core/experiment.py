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
import json
import traceback
from datetime import datetime
from pathlib import Path
from time import sleep
from typing import Literal

from crab.utils.common import base64_to_image

from .agent_policy import AgentPolicy
from .benchmark import Benchmark
from .csv_log import CSVLog
from .models import ActionOutput, MessageType

CURRENT_EXPERIMENT_COLUMNS = [
    "step",
    "action",
    "total_nodes",
    "complete_nodes",
    "completeness",
    "completeness_per_action",
    "step_to_complete",
    "longest_unfinished_path_length",
    "token_usage",
]


MAIN_LOG_COLUMNS = [
    "time",
    "agent_policy",
    "model",
    "task_id",
    "total_steps",
    "terminate_reason",
    "total_nodes",
    "complete_nodes",
    "completeness",
    "completeness_per_action",
    "step_to_complete",
    "longest_unfinished_path_length",
    "token_usage",
]


class Experiment:
    def __init__(
        self,
        benchmark: Benchmark,
        task_id: str,
        agent_policy: AgentPolicy | Literal["human"],
        log_dir: Path | None = None,
    ) -> None:
        self.benchmark = benchmark
        self.task_id = task_id
        self.agent_policy = agent_policy
        self.log_dir = log_dir

    def write_message(self, message: str, step: int):
        with open(self.message_path, "a") as file:
            file.write("=" * 20 + f"Step: {step}" + "=" * 20 + "\n" + message + "\n")

    def write_task_info_json(self, task_info_path: Path):
        envs_info = {}
        for name, env in self.benchmark.environment_map.items():
            actions = {
                name: action.description for name, action in env._action_map.items()
            }
            observations = {
                action.name: action.description for action in env._observation_space
            }
            envs_info[name] = {
                "description": env.description,
                "actions": actions,
                "observations": observations,
            }
        task_info = {
            "benchmark_name": self.benchmark.name,
            "task_id": self.task_id,
            "task_description": self.task.description,
            "envs": envs_info,
        }
        with open(task_info_path, "w") as file:
            json.dump(task_info, file, indent=4)

    def init_log_dir(self):
        if self.log_dir is not None:
            self.log_dir.mkdir(exist_ok=True, parents=True)

            self.main_log = CSVLog(self.log_dir / "main_log.csv", MAIN_LOG_COLUMNS)

            self.task_info_dir = self.log_dir / self.task_id
            self.task_info_dir.mkdir(exist_ok=True, parents=True)
            self.write_task_info_json(self.task_info_dir / "task_info.json")

            self.time_now = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            self.current_experiment_dir = (
                self.task_info_dir / f"{self.agent_policy.__class__.__name__}"
                f"({self.agent_policy.get_backend_model_name()})" / self.time_now
            )
            self.current_experiment_dir.mkdir(parents=True)

            self.current_experiment_log = CSVLog(
                self.current_experiment_dir / "metrics.csv", CURRENT_EXPERIMENT_COLUMNS
            )

            self.prompt_path = self.current_experiment_dir / "prompt"
            self.image_path = self.current_experiment_dir / "images"
            self.prompt_path.mkdir()
            self.image_path.mkdir()

            self.message_path = self.current_experiment_dir / "messages.txt"

    def get_prompt(self) -> dict[str, list[tuple[str, MessageType]]]:
        return self.benchmark.observe()

    def execute_action(self, response: list[ActionOutput]) -> bool:
        for action in response:
            benchmark_result = self.benchmark.step(
                action=action.name,
                parameters=action.arguments,
                env_name=action.env,
            )
            self.metrics = benchmark_result.evaluation_results
            if benchmark_result.terminated:
                print("\033[92m" f"Task finished, result: {self.metrics}" "\033[0m")
                self.write_current_log_row(action)
                self.write_main_csv_row(benchmark_result.info["terminate_reason"])
                if "exception_detail" in benchmark_result.info:
                    self.write_exception_detail(
                        benchmark_result.info["exception_detail"]
                    )
                return True
            print(
                "\033[92m"
                f'Action "{action.name}" in env "{action.env}" success. '
                f"Current evaluation results: {self.metrics}\n"
                "\033[0m"
            )
            self.write_current_log_row(action)
            self.step_cnt += 1
        return False

    def log_prompt(self, prompt):
        for env in prompt:
            with open(self.prompt_path / f"{env}_prompt.md", "a") as prompt_file:
                prompt_file.write(f"### Step {self.step_cnt}\n\n")
                for message, message_type in prompt[env]:
                    if message_type == MessageType.IMAGE_JPG_BASE64:
                        file_name = f"{env}_{self.step_cnt}.png"
                        base64_to_image(message).save(self.image_path / file_name)
                        prompt_file.write(f"![](../images/{file_name})\n\n")
                    else:
                        prompt_file.write(message + "\n\n")

    def step(self, it) -> bool:
        print("=" * 40)
        print(f"Start agent step {self.step_cnt}:")
        prompt = self.get_prompt()
        self.log_prompt(prompt)
        try:
            response = self.agent_policy.chat(prompt)
        except Exception:
            print(traceback.format_exc())
            self.write_main_csv_row("agent_exception")
            self.write_exception_detail(traceback.format_exc())
            return True
        # content = response["content"]
        # self.write_message(str(content), it)
        # print("\033[94m" f"Agent Reponse: {content}" "\033[0m")
        print(f"So agent take action: {response}")
        return self.execute_action(response)

    def start_benchmark(self):
        if self.agent_policy == "human":
            self.benchmark.human_evaluation(self.task_id)
            return

        env_description = {}
        for env in self.benchmark.environment_map:
            env_description[env] = self.benchmark.environment_map[env].description

        self.task, action_space = self.benchmark.start_task(self.task_id)
        self.agent_policy.reset(
            task_description=self.task.description,
            action_spaces=action_space,
            env_descriptions=env_description,
        )
        print(
            f'Start benchmark "{self.benchmark.name}", task id "{self.task.id}": '
            f'"{self.task.description}"'
        )
        self.init_log_dir()
        self.step_cnt = 0
        self.metrics = self.benchmark.evaluate()
        if self.metrics["complete_nodes"] != 0:
            print("Graph Evaluator start with non-zero value. Check environment setup.")
            return
        for it in range(50):
            try:
                terminated = self.step(it)
            except KeyboardInterrupt:
                self.write_main_csv_row("keyboard_interrupt")
                return
            if terminated:
                return
            sleep(2)
            # input("Press enter to do next step:")

    def write_exception_detail(self, exception_info: str):
        if self.log_dir is None:
            return
        with open(self.current_experiment_dir / "exception_detail.txt", "w") as file:
            file.write(exception_info)

    def write_current_log_row(self, action):
        if self.log_dir is None:
            return
        self.current_experiment_log.write_row(
            [
                self.step_cnt,
                str(action),
                self.metrics["total_nodes"],
                self.metrics["complete_nodes"],
                self.metrics["completeness"],
                self.metrics["completeness_per_action"],
                self.metrics["step_to_complete"],
                self.metrics["longest_unfinished_path_length"],
                self.agent_policy.get_token_usage(),
            ]
        )

    def write_main_csv_row(self, terminate_reason):
        if self.log_dir is None:
            return
        self.main_log.write_row(
            [
                self.time_now,
                self.agent_policy.__class__.__name__,
                self.agent_policy.get_backend_model_name(),
                self.task_id,
                self.step_cnt,
                terminate_reason,
                self.metrics["total_nodes"],
                self.metrics["complete_nodes"],
                self.metrics["completeness"],
                self.metrics["completeness_per_action"],
                self.metrics["step_to_complete"],
                self.metrics["longest_unfinished_path_length"],
                self.agent_policy.get_token_usage(),
            ]
        )
