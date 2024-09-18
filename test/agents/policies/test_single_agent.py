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
import os
from unittest.mock import MagicMock, patch

import pytest
from openai.types.chat.chat_completion import (
    ChatCompletionMessage,
    Choice,
    CompletionUsage,
)
from openai.types.chat.chat_completion_message_tool_call import (
    ChatCompletionMessageToolCall,
    Function,
)

from crab import create_benchmark
from crab.agents.backend_models import BackendModelConfig
from crab.agents.policies.single_agent import SingleAgentPolicy
from crab.benchmarks.template import multienv_template_benchmark_config

openai_mock_response = MagicMock(
    choices=[
        Choice(
            finish_reason="stop",
            index=0,
            logprobs=None,
            message=ChatCompletionMessage(
                content=None,
                role="assistant",
                function_call=None,
                tool_calls=[
                    ChatCompletionMessageToolCall(
                        id="call_3YIJZhrC5smSjAJKOeFcQxRf",
                        function=Function(
                            arguments='{"value": true}', name="set_state__in__testenv0"
                        ),
                        type="function",
                    ),
                    ChatCompletionMessageToolCall(
                        id="call_mA9Z9HQfmYn2TbzeGsEVcCr7",
                        function=Function(
                            arguments='{"value": true}', name="set_state__in__testenv1"
                        ),
                        type="function",
                    ),
                    ChatCompletionMessageToolCall(
                        id="call_GgxbBTd6afj2iDyOewaNattB",
                        function=Function(
                            arguments='{"value": true}', name="set_state__in__testenv2"
                        ),
                        type="function",
                    ),
                ],
            ),
        )
    ],
    model="gpt-4o-2024-05-13",
    object="chat.completion",
    usage=CompletionUsage(completion_tokens=74, prompt_tokens=648, total_tokens=722),
)


@pytest.fixture
def policy_fixture():
    os.environ["OPENAI_API_KEY"] = "MOCK"
    model = BackendModelConfig(
        model_class="openai",
        model_name="gpt-4o",
        parameters={"max_tokens": 3000},
        history_messages_len=1,
    )
    benchmark_config = multienv_template_benchmark_config
    benchmark = create_benchmark(benchmark_config)
    task, action_spaces = benchmark.start_task("0")
    policy = SingleAgentPolicy(model_backend=model)
    policy.reset(
        task_description=task.description,
        action_spaces=action_spaces,
        env_descriptions=benchmark.get_env_descriptions(),
    )
    return policy, benchmark


@patch(
    "openai.resources.chat.completions.Completions.create",
    return_value=openai_mock_response,
)
def test_policy(mock_create: MagicMock, policy_fixture):
    policy, benchmark = policy_fixture
    observation = benchmark.observe()
    for env in observation:
        if env == "root":
            continue
        observation[env] = [
            (
                'The current state of "{env}" is '
                + str(observation[env]["current_state"])
                + ". ",
                0,
            )
        ]
    action_list = policy.chat(observation)
    mock_create.assert_called_once()
    assert action_list
