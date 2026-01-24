# =========== Copyright 2024 @ CAMEL-AI.org. All Rights Reserved. ===========
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# =========== Copyright 2024 @ CAMEL-AI.org. All Rights Reserved. ===========
from termcolor import colored
import os

from crab import Benchmark, create_benchmark
from crab.agents.backend_models.camel_model import CamelModel
from crab.agents.policies import SingleAgentPolicy
from crab.benchmarks.template import template_benchmark_config
from camel.types import ModelType, ModelPlatformType
from camel.models import ModelFactory


def start_benchmark(benchmark: Benchmark, agent: SingleAgentPolicy):
    for step in range(20):
        print("=" * 40)
        print(f"Start agent step {step}:")
        observation = benchmark.observe()["template_env"]
        print(f"Current environment observation: {observation}")
        response = agent.chat(
            {
                "template_env": [
                    (f"Current environment observation: {observation}", 0),
                ]
            }
        )
        print(colored(f"Agent take action: {response}", "blue"))

        for action in response:
            response = benchmark.step(
                action=action.name,
                parameters=action.arguments,
                env_name=action.env,
            )
            print(
                colored(
                    f'Action "{action.name}" success, stat: '
                    f"{response.evaluation_results}",
                    "green",
                )
            )
            if response.terminated:
                print("=" * 40)
                print(
                    colored(
                        f"Task finished, result: {response.evaluation_results}", "green"
                    )
                )
                return


if __name__ == "__main__":
    benchmark = create_benchmark(template_benchmark_config)
    #TODO: Use new task config
    task, action_space = benchmark.start_task("0")
    env_descriptions = benchmark.get_env_descriptions()

    # TODO: Use local model
    camel_model = CamelModel(
        model="gpt-4o",
        model_platform=ModelPlatformType.OPENAI,
        parameters={"temperature": 0.7},  
    )
    agent = SingleAgentPolicy(model_backend=camel_model)
    agent.reset(task.description, action_space, env_descriptions)
    print("Start performing task: " + colored(f'"{task.description}"', "green"))
    start_benchmark(benchmark, agent)
    benchmark.reset()
