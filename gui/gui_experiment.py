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
from pathlib import Path
from typing import Literal

from crab import AgentPolicy, Benchmark, Experiment, MessageType


class GuiExperiment(Experiment):
    def __init__(
        self,
        benchmark: Benchmark,
        task_id: str,
        agent_policy: AgentPolicy | Literal["human"],
        log_dir: Path | None = None,
    ) -> None:
        super().__init__(benchmark, task_id, agent_policy, log_dir)
        self.display_callback = None

    def set_display_callback(self, callback):
        self.display_callback = callback

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

    def step(self, it) -> bool:
        if self.display_callback:
            self.display_callback(f"Step {self.step_cnt}:", "ai")

        prompt = self.get_prompt()
        self.log_prompt(prompt)

        try:
            response = self.agent_policy.chat(prompt)
            if self.display_callback:
                self.display_callback(f"Planning next action...", "ai")
        except Exception as e:
            if self.display_callback:
                self.display_callback(f"Error: {str(e)}", "ai")
            self.write_main_csv_row("agent_exception")
            return True
        
        if self.display_callback:
            self.display_callback(f"Executing: {response}", "ai")
        return self.execute_action(response)

    def execute_action(self, response: list[ActionOutput]) -> bool:
        for action in response:
            benchmark_result = self.benchmark.step(
                action=action.name,
                parameters=action.arguments,
                env_name=action.env,
            )
            self.metrics = benchmark_result.evaluation_results

            if benchmark_result.terminated:
                if self.display_callback:
                    self.display_callback(
                        f"✓ Task completed! Results: {self.metrics}", "ai"
                    )
                self.write_current_log_row(action)
                self.write_current_log_row(benchmark_result.info["terminate_reason"])
                return True

            if self.display_callback:
                self.display_callback(
                    f'Action "{action.name}" completed in {action.env}. '
                    f"Progress: {self.metrics}", "ai"
                )
            self.write_current_log_row(action)
            self.step_cnt += 1
        return False

    def start_benchmark(self):
        if self.display_callback:
            self.display_callback("Starting benchmark...", "ai")
        try:
            super().start_benchmark()
        except KeyboardInterrupt:
            if self.display_callback:
                self.display_callback("Experiment interrupted.", "ai")
            self.write_main_csv_row("experiment_interrupted")
        finally:
            if self.display_callback:
                self.display_callback("Experiment finished.", "ai")
