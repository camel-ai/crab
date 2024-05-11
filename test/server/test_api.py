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
from fastapi.testclient import TestClient

from crab.environments.template import (
    current_state,
    set_state,
    template_environment_config,
)
from crab.server.main import init


def test_raw_action():
    app = init(template_environment_config)
    client = TestClient(app)
    response = client.post(
        "/raw_action",
        json={
            "action": set_state.to_raw_action(),
            "parameters": {"value": True},
        },
    )
    assert response.json()["action_returns"] is None

    response = client.post(
        "/raw_action",
        json={
            "action": current_state.to_raw_action(),
            "parameters": {},
        },
    )
    assert response.json()["action_returns"] is True

    action = set_state(True)
    response = client.post(
        "/raw_action",
        json={
            "action": action.to_raw_action(),
            "parameters": {},
        },
    )
    assert response.json()["action_returns"] is None

    action = current_state >> set_state
    response = client.post(
        "/raw_action",
        json={
            "action": action.to_raw_action(),
            "parameters": {},
        },
    )
    assert response.json()["action_returns"] is None

    action = set_state(True) + current_state
    response = client.post(
        "/raw_action",
        json={
            "action": action.to_raw_action(),
            "parameters": {},
        },
    )
    assert response.json()["action_returns"] is True
