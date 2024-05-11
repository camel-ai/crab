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
from crab import create_benchmark
from crab.agents.single_agent.openai_agent import OpenAIAgent
from crab.benchmarks.template import template_benchmark_config


def start_benchmark(benchmark, agent):
    for step in range(20):
        print("=" * 40)
        print(f"Start agent step {step}:")
        observation = benchmark.observe()["template_env"]
        print(f"Current enviornment observation: {observation}")
        response = agent.chat(
            [
                (f"Current enviornment observation: {observation}", 0),
                ("Tell me the next action.", 0),
            ]
        )
        print("\033[94m" f"Agent Reponse: {response['content']}" "\033[0m")
        print(f"So agent take action: {response['action_list']}")

        for action in response["action_list"]:
            response = benchmark.step(*action)
            if response.terminated:
                print(
                    "\033[92m"
                    f"Task finished, result: {response.evaluation_results}"
                    "\033[0m"
                )
                return
            print(
                "\033[92m"
                f'Action "{action[0]}" success, stat: {response.evaluation_results}'
                "\033[0m"
            )


if __name__ == "__main__":
    benchmark = create_benchmark(template_benchmark_config)
    task, action_space = benchmark.start_task("0")
    agent = OpenAIAgent(
        task.description,
        action_space,
        model="gpt-4-turbo-preview",
    )
    start_benchmark(benchmark, agent)
    benchmark.reset()
