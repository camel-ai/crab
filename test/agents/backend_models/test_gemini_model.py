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

from crab import MessageType, action
from crab.agents.backend_models import BackendModelConfig, create_backend_model

# TODO: Add mock data


@pytest.fixture
def gemini_model_text():
    return create_backend_model(
        BackendModelConfig(
            model_class="gemini",
            model_name="gemini-1.5-pro-latest",
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


@pytest.mark.skip(reason="Mock data to be added")
def test_text_chat(gemini_model_text):
    message = ("Hello!", MessageType.TEXT)
    output = gemini_model_text.chat(message)
    assert output.message
    assert output.action_list is None
    # assert gemini_model_text.token_usage > 0

    # Send another message to check accumulated tokens and history length
    message2 = ("Give me five!", MessageType.TEXT)
    output = gemini_model_text.chat(message2)
    # assert gemini_model_text.token_usage > 0
    assert output.message
    assert len(gemini_model_text.chat_history) == 2

    # Send another message to check accumulated tokens and chat history
    output = gemini_model_text.chat(message2)
    assert output.message
    assert len(gemini_model_text.chat_history) == 3


@pytest.mark.skip(reason="Mock data to be added")
def test_action_chat(gemini_model_text):
    gemini_model_text.reset("You are a helpful assistant.", [add])
    message = (
        (
            "I had 10 dollars. Miss Polaris gave me 15 dollars. "
            "How many money do I have now."
        ),
        0,
    )
    output = gemini_model_text.chat(message)
    assert output.message is None
    assert len(output.action_list) == 1
    assert output.action_list[0].arguments == {"a": 10, "b": 15}
    assert output.action_list[0].name == "add"
