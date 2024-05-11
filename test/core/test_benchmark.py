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
from fastapi.testclient import TestClient

from crab import Benchmark, action, create_benchmark
from crab.benchmarks.template import (
    multienv_template_benchmark_config,
    template_benchmark_config,
    template_environment_config,
)
from crab.server.main import init


@pytest.fixture
def benchmark(request):
    if request.param == "multienv":
        yield create_benchmark(multienv_template_benchmark_config)
    elif request.param == "multienv-remote":
        # TODO: fix multienv remote use the same env in different remote envs
        app0 = init(environment_config=template_environment_config)
        client0 = TestClient(app0)
        app1 = init(environment_config=template_environment_config)
        client1 = TestClient(app1)
        app2 = init(environment_config=template_environment_config)
        client2 = TestClient(app2)
        proxy_config = multienv_template_benchmark_config.model_copy()
        for env in proxy_config.environments:
            env.remote_url = "http://127.0.0.1:8000"
        benchmark = create_benchmark(proxy_config)
        benchmark.environment_map["testenv0"]._client = client0
        benchmark.environment_map["testenv1"]._client = client1
        benchmark.environment_map["testenv2"]._client = client2
        yield benchmark
    elif request.param == "singleenv":
        yield create_benchmark(template_benchmark_config)


@pytest.mark.parametrize("benchmark", ["multienv", "multienv-remote"], indirect=True)
def test_multi_env_benchmark_process(benchmark: Benchmark):
    assert benchmark.multienv
    task, actions = benchmark.start_task(task_id="0")
    assert benchmark.current_task == task
    assert len(actions) == 4
    assert len(actions["root"]) == 1
    assert actions["root"][0].name == "_submit"

    result = benchmark.step(
        action="set_state", parameters={"value": True}, env_name="testenv0"
    )
    assert result.evaluation_results["completeness"] == 0.25

    result = benchmark.step(
        action="set_state", parameters={"value": True}, env_name="testenv1"
    )
    assert result.evaluation_results["completeness"] == 0.5

    result = benchmark.step(
        action="set_state", parameters={"value": True}, env_name="testenv2"
    )
    assert result.evaluation_results["completeness"] == 0.75

    result = benchmark.step(
        action="_submit", parameters={"content": True}, env_name="root"
    )
    assert result.terminated
    assert result.evaluation_results["completeness"] == 1.0


@action
def to_str(input: bool) -> str:
    return f"The current state is {input}"


@pytest.mark.parametrize("benchmark", ["singleenv"], indirect=True)
def test_prompting_tool(benchmark: Benchmark):
    benchmark.prompting_tools = {"template_env": {"current_state": to_str}}
    benchmark.start_task("0")
    observe, prompt = benchmark.observe_with_prompt()
    assert observe["template_env"]["current_state"] is False
    assert prompt["template_env"]["current_state"] == "The current state is False"
    benchmark.close_task()
