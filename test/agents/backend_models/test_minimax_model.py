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

from crab import action
from crab.agents.backend_models import BackendModelConfig, create_backend_model
from crab.agents.backend_models.minimax_model import (
    BASE_URLS,
    DEFAULT_BASE_URL,
    MiniMaxModel,
)
from crab.agents.backend_models.openai_model import MessageType

# Mock data for the OpenAI-compatible API response
mock_response = MagicMock(
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
    model="MiniMax-M3",
    object="chat.completion",
    usage=MagicMock(completion_tokens=10, prompt_tokens=19, total_tokens=29),
)


@pytest.fixture
def minimax_model_text():
    os.environ["OPENAI_API_KEY"] = "MOCK"
    return create_backend_model(
        BackendModelConfig(
            model_class="minimax",
            model_name="MiniMax-M3",
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


def test_defaults_to_global_endpoint(minimax_model_text):
    assert isinstance(minimax_model_text, MiniMaxModel)
    assert DEFAULT_BASE_URL == BASE_URLS["global_en"]
    assert str(minimax_model_text.client.base_url).startswith(DEFAULT_BASE_URL)


def test_respects_regional_base_url():
    os.environ["OPENAI_API_KEY"] = "MOCK"
    model = create_backend_model(
        BackendModelConfig(
            model_class="minimax",
            model_name="MiniMax-M2.7",
            base_url=BASE_URLS["cn_zh"],
        )
    )
    assert str(model.client.base_url).startswith(BASE_URLS["cn_zh"])


@patch(
    "openai.resources.chat.completions.Completions.create",
    return_value=mock_response,
)
def test_text_chat(mock_create, minimax_model_text):
    message = ("Hello!", MessageType.TEXT)
    output = minimax_model_text.chat(message)
    assert len(mock_create.call_args.kwargs["messages"]) == 2
    assert output.message == "Hi there! How can I assist you today?"
    assert output.action_list is None
    assert minimax_model_text.token_usage == 29
