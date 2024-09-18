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
from openai.types.chat.chat_completion_message_tool_call import Function

from crab import action
from crab.agents.backend_models import BackendModelConfig, create_backend_model
from crab.agents.backend_models.openai_model import MessageType

# Mock data for the OpenAI API response
openai_mock_response = MagicMock(
    choices=[
        MagicMock(
            finish_reason="stop",
            index=0,
            logprobs=None,
            message=MagicMock(
                content="Hi there! How can I assist you today?",
                role="assistant",
                function_call=None,
                tool_calls=None,
            ),
        )
    ],
    model="gpt-4o-2024-05-13",
    object="chat.completion",
    usage=MagicMock(completion_tokens=10, prompt_tokens=19, total_tokens=29),
)

openai_mock_response2 = MagicMock(
    choices=[
        MagicMock(
            finish_reason="stop",
            index=0,
            logprobs=None,
            message=MagicMock(
                content="Sure thing! ✋ How can I help you today?",
                role="assistant",
                function_call=None,
                tool_calls=None,
            ),
        )
    ],
    model="gpt-4o-2024-05-13",
    object="chat.completion",
    usage=MagicMock(completion_tokens=12, prompt_tokens=41, total_tokens=53),
)

openai_mock_response3 = MagicMock(
    choices=[
        MagicMock(
            finish_reason="stop",
            index=0,
            logprobs=None,
            message=MagicMock(
                content=None,
                role="assistant",
                function_call=None,
                tool_calls=[
                    MagicMock(
                        id="call_ceE9IX1uYeRqGShYYlHYrCCF",
                        function=Function(arguments='{"a":10,"b":15}', name="add"),
                        type="function",
                    )
                ],
            ),
        )
    ],
    model="gpt-4o-2024-05-13",
    object="chat.completion",
    usage=MagicMock(completion_tokens=15, prompt_tokens=93, total_tokens=108),
)


@pytest.fixture
def openai_model_text():
    os.environ["OPENAI_API_KEY"] = "MOCK"
    return create_backend_model(
        BackendModelConfig(
            model_class="openai",
            model_name="gpt-4o",
            parameters={"max_tokens": 3000},
            history_messages_len=1,
            tool_call_required=False,
        )
    )


@action
def add(a: int, b: int):
    """Add up two integers.

    Args:
        a: An addend
        b: Another addend
    """
    return a + b


@patch(
    "openai.resources.chat.completions.Completions.create",
    return_value=openai_mock_response,
)
def test_text_chat(mock_create, openai_model_text):
    message = ("Hello!", MessageType.TEXT)
    output = openai_model_text.chat(message)
    assert len(mock_create.call_args.kwargs["messages"]) == 2
    assert output.message == "Hi there! How can I assist you today?"
    assert output.action_list is None
    assert openai_model_text.token_usage == 29

    # Send another message to check accumulated tokens and history length
    message2 = ("Give me five!", MessageType.TEXT)
    mock_create.return_value = openai_mock_response2
    output = openai_model_text.chat(message2)
    assert len(mock_create.call_args.kwargs["messages"]) == 4
    assert openai_model_text.token_usage == 29 + 53
    assert output.message == "Sure thing! ✋ How can I help you today?"
    assert len(openai_model_text.chat_history) == 2

    # Send another message to check accumulated tokens and chat history
    output = openai_model_text.chat(message2)
    assert len(mock_create.call_args.kwargs["messages"]) == 4
    assert openai_model_text.token_usage == 29 + 53 + 53
    assert output.message == "Sure thing! ✋ How can I help you today?"
    assert len(openai_model_text.chat_history) == 3


@patch(
    "openai.resources.chat.completions.Completions.create",
    return_value=openai_mock_response3,
)
def test_action_chat(mock_create, openai_model_text):
    openai_model_text.reset("You are a helpful assistant.", [add])
    message = (
        (
            "I had 10 dollars. Miss Polaris gave me 15 dollars. "
            "How many money do I have now."
        ),
        0,
    )
    output = openai_model_text.chat(message)
    assert output.message is None
    assert len(output.action_list) == 1
    assert output.action_list[0].arguments == {"a": 10, "b": 15}
    assert output.action_list[0].name == "add"
    assert openai_model_text.token_usage == 108
