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
import networkx as nx
import pytest

from crab.core import Environment, Evaluator, GraphEvaluator, evaluator

a = None


def set_a(value: int) -> None:
    global a
    a = value


@evaluator
def dummy_evaluator1() -> bool:
    """
    This is a test evaluator.

    Args:
        a (int): The first parameter.

    Returns:
        bool: The result.
    """
    return a > 0


@evaluator
def dummy_evaluator2() -> bool:
    """
    This is a test evaluator.

    Args:
        a (int): The first parameter.
        b (str, optional): The second parameter. Defaults to "default".

    Returns:
        bool: The result.
    """
    return a < 2


@evaluator
def dummy_evaluator3() -> bool:
    """
    This is a test evaluator.

    Args:
        a (int): The first parameter.
        b (str, optional): The second parameter. Defaults to "default".

    Returns:
        bool: The result.
    """
    return a > 100


@evaluator
def no_param_evaluator() -> bool:
    return True


@pytest.fixture
def root_env() -> Environment:
    return Environment(
        name="root",
        action_space=[],
        observation_space=[],
        description="The crab root server",
    )


def test_evaluator_run():
    assert isinstance(dummy_evaluator1, Evaluator)
    set_a(3)
    assert dummy_evaluator1.entry()
    set_a(-1)
    assert not dummy_evaluator1.entry()


def test_evaluator_and():
    set_a(1)
    assert (dummy_evaluator1 & dummy_evaluator2).entry()
    set_a(-1)
    assert not (dummy_evaluator1 & dummy_evaluator2).entry()
    set_a(3)
    assert not (dummy_evaluator1 & dummy_evaluator2).entry()


def test_evaluator_or():
    set_a(1)
    assert (dummy_evaluator1 | dummy_evaluator2).entry()
    set_a(-1)
    assert (dummy_evaluator1 | dummy_evaluator2).entry()
    set_a(3)
    assert (dummy_evaluator1 | dummy_evaluator2).entry()


def test_evaluator_not():
    set_a(3)
    assert not (~dummy_evaluator1).entry()
    set_a(-1)
    assert (~dummy_evaluator1).entry()


def test_chain_evaluator(root_env):
    graph_evaluator = GraphEvaluator(
        nx.path_graph(
            [dummy_evaluator1, dummy_evaluator2, no_param_evaluator],
            create_using=nx.DiGraph,
        )
    )
    graph_evaluator.reset()
    assert graph_evaluator.count == 0
    assert graph_evaluator.G.nodes[dummy_evaluator1]["remaining_predecessors"] == 0
    assert graph_evaluator.G.nodes[dummy_evaluator2]["remaining_predecessors"] == 1
    assert graph_evaluator.G.nodes[no_param_evaluator]["remaining_predecessors"] == 1

    set_a(3)
    graph_evaluator.step({"root": root_env})
    assert graph_evaluator.count == 1
    assert graph_evaluator.G.nodes[dummy_evaluator1]["passing_count"] == 0
    assert graph_evaluator.G.nodes[dummy_evaluator2]["remaining_predecessors"] == 0

    set_a(3)
    graph_evaluator.step({"root": root_env})
    assert graph_evaluator.count == 2
    assert graph_evaluator.G.nodes[dummy_evaluator2]["remaining_predecessors"] == 0
    assert graph_evaluator.G.nodes[dummy_evaluator2]["passing_count"] is None

    set_a(-1)
    graph_evaluator.step({"root": root_env})
    assert graph_evaluator.count == 3
    assert graph_evaluator.G.nodes[dummy_evaluator2]["passing_count"] == 2
    assert graph_evaluator.G.nodes[no_param_evaluator]["remaining_predecessors"] == 0
