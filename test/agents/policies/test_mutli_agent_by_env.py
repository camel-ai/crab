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
import pytest

from crab import create_benchmark
from crab.agents.backend_models import BackendModelConfig
from crab.agents.policies.multi_agent_by_env import MultiAgentByEnvPolicy
from crab.benchmarks.template import multienv_template_benchmark_config


@pytest.fixture
def policy_fixture():
    model = BackendModelConfig(
        model_class="openai",
        model_name="gpt-4o",
        parameters={"max_tokens": 3000},
        history_messages_len=1,
    )
    benchmark_config = multienv_template_benchmark_config
    benchmark = create_benchmark(benchmark_config)
    task, action_spaces = benchmark.start_task("0")
    policy = MultiAgentByEnvPolicy(
        main_agent_model_backend=model,
        env_agent_model_backend=model,
    )
    policy.reset(
        task_description=task.description,
        action_spaces=action_spaces,
        env_descriptions=benchmark.get_env_descriptions(),
    )
    return policy, benchmark


@pytest.mark.skip(reason="Mock data to be added")
def test_policy(policy_fixture):
    policy, benchmark = policy_fixture
    observations = benchmark.observe()
    agent_observation = {}
    for env in observations:
        if env == "root":
            continue
        agent_observation[env] = [
            (
                f'The current state of "{env}" is '
                + str(observations[env]["current_state"])
                + ". ",
                0,
            )
        ]
    action_list = policy.chat(agent_observation)
    assert action_list
