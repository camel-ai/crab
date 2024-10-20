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

from camel.societies import RolePlaying
from camel.utils import print_text_animated

from crab import Benchmark, create_benchmark
from crab.agents.backend_models import OpenAIModel
from crab.agents.policies import SingleAgentPolicy
from crab.benchmarks.template import template_benchmark_config

def camel_task_generator():
    task_prompt = "Design a custom game using pygame"
    print(colored(f"Original task prompt:\n{task_prompt}\n", "yellow"))
    role_play_session = RolePlaying("Computer Programmer", "Gamer", task_prompt=task_prompt)
    print(colored(f"Specified task prompt:\n{role_play_session.task_prompt}\n", "cyan"))

    chat_turn_limit, n = 50, 0
    input_msg = role_play_session.init_chat()
    while n < chat_turn_limit:
        n += 1
        assistant_response, user_response = role_play_session.step(input_msg)
        print_text_animated(colored(f"AI User:\n\n{user_response.msg.content}\n", "blue"))
        print_text_animated(colored(f"AI Assistant:\n\n{assistant_response.msg.content}\n", "green"))

        if "CAMEL_TASK_DONE" in user_response.msg.content:
            break

        input_msg = assistant_response.msg

    return role_play_session.task_prompt

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
                        f"Task finished, result: {response.evaluation_results}",
                        "green"
                    )
                )
                return

if __name__ == "__main__":
    task_description = camel_task_generator()

    benchmark = create_benchmark(template_benchmark_config)
    task, action_space = benchmark.start_task("0", task_description)
    env_descriptions = benchmark.get_env_descriptions()

    model = OpenAIModel(model="gpt-4o", history_messages_len=5)
    agent = SingleAgentPolicy(model_backend=model)
    agent.reset(task_description, action_space, env_descriptions)

    print("Start performing task: " + colored(f'"{task_description}"', "green"))
    start_benchmark(benchmark, agent)
    benchmark.reset()