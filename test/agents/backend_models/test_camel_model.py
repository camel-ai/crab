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

from crab import action
from crab.agents.backend_models import CamelModel


@pytest.fixture
def camel_model():
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
def test_action_chat(camel_model):
    camel_model.reset("You are a helpful assistant.", [add])
    message = (
        "I had 10 dollars. Miss Polaris gave me 15 dollars. "
        "How many money do I have now.",
        0,
    )
    output = camel_model.chat([message])
    assert not output.message
    assert len(output.action_list) == 1
    assert output.action_list[0].arguments == {"a": 10, "b": 15}
    assert output.action_list[0].name == "add"
    assert camel_model.token_usage > 0