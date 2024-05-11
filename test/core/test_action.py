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
from crab.core import Action, action
from crab.core.models.action import _check_no_param


@action
def dummy_function(a: int, b: str = "default") -> int:
    """
    This is a test function.

    Args:
        a (int): The first parameter.
        b (str, optional): The second parameter. Defaults to "default".

    Returns:
        int: The result.
    """
    return a + 1


@action
def dummy_env_action(a: int, env: int) -> int:
    """
    This is a kept parameter test function.

    Args:
        a (int): The first parameter.
        env (int): The current environemnt. Should not be appeared in the parameters.

    Returns:
        int: The result.
    """
    return a + env


def test_action_to_openai_json_schema():
    result = dummy_function.to_openai_json_schema()
    assert result["name"]
    assert result["description"]
    assert result["parameters"]

    parameters = result["parameters"]
    assert "properties" in parameters
    assert "a" in parameters["properties"]
    assert parameters["properties"]["a"]["type"] == "integer"
    assert "b" in parameters["properties"]
    assert parameters["properties"]["b"]["type"] == "string"
    assert parameters["properties"]["b"]["default"] == "default"
    assert "required" in parameters
    assert "a" in parameters["required"]


def test_from_function():
    action_instance: Action = dummy_function
    assert action_instance.description == "This is a test function."
    assert action_instance.name == "dummy_function"
    assert "a" in action_instance.parameters.model_fields
    assert "b" in action_instance.parameters.model_fields
    assert action_instance.name == "dummy_function"


def test_chaining():
    dummy_x2 = dummy_function >> dummy_function
    assert dummy_x2.entry(1) == 3


@action
def add_a_to_b(a: int, b: int = 1) -> int:
    return a + b


@action
def multiply_a_to_b(a: int, b: int = 1) -> int:
    return a * b


def test_closed_action():
    action = add_a_to_b(5)
    assert action.entry() == 6
    assert _check_no_param(action)


def test_kwargs_action():
    action = add_a_to_b(b=6)
    assert action.entry(1) == 7


def test_chain_various_actions():
    action = add_a_to_b(b=10) >> multiply_a_to_b(b=10) >> add_a_to_b()
    assert action.entry(0) == 101
    action = add_a_to_b(a=1, b=10) >> multiply_a_to_b(b=10) >> add_a_to_b()
    assert action.entry() == 111
    action = add_a_to_b(1, b=10) >> multiply_a_to_b(b=10) >> add_a_to_b()
    assert action.entry() == 111


def test_kept_param():
    action = dummy_env_action.set_kept_param(env=10)
    assert action.run(a=10) == 20
