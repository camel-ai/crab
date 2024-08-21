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
from crab.agents.backend_models import CamelModel

@pytest.fixture
def camel_model_text():
    return CamelModel(
        model_platform="openai",
        model="gpt-4o",
        parameters={"max_tokens": 3000},
        history_messages_len=1,
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
def test_text_chat(camel_model_text):
    message = ("Hello!", MessageType.TEXT)
    output = camel_model_text.chat(message)
    assert output.message
    assert output.action_list is None
    assert camel_model_text.token_usage > 0

    # Send another message to check accumulated tokens and history length
    message2 = ("Give me five!", MessageType.TEXT)
    output = camel_model_text.chat(message2)
    assert camel_model_text.token_usage > 0
    assert output.message
    assert len(camel_model_text.chat_history) == 2

    # Send another message to check accumulated tokens and chat history
    output = camel_model_text.chat(message2)
    assert output.message
    assert len(camel_model_text.chat_history) == 3

def test_action_chat(camel_model_text):
    camel_model_text.reset("You are a helpful assistant.", [add])
    message = (
        "I had 10 dollars. Miss Polaris gave me 15 dollars. How many money do I have now.",
        0,
    )
    output = camel_model_text.chat(message)
    assert output.message is None
    assert len(output.action_list) == 1
    assert output.action_list[0].arguments == {"a": 10, "b": 15}
    assert output.action_list[0].name == "add"
    assert camel_model_text.token_usage > 0
