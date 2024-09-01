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

from crab import create_environment
from crab.environments.template import (
    current_state,
    set_state,
    template_environment_config,
)
from crab.server.main import init


@pytest.fixture
def mock_env():
    mock_app = init(template_environment_config)
    mock_cli = TestClient(mock_app)
    mock_env = create_environment(template_environment_config)
    mock_env._client = mock_cli
    return mock_env


def test_raw_action_unencrypted(mock_env):
    assert mock_env._action_endpoint(set_state, {"value": True}) is None
    assert mock_env._action_endpoint(current_state, {}) is True
    assert mock_env._action_endpoint(set_state(True), {}) is None
    assert mock_env._action_endpoint(current_state >> set_state, {}) is None
    assert mock_env._action_endpoint(set_state(True) + current_state, {}) is True


def test_raw_action_encrypted(mock_env, monkeypatch):
    monkeypatch.setenv("ENCRYPTION_KEY", "the-cake-is-a-lie")
    assert mock_env._action_endpoint(set_state, {"value": True}) is None
    assert mock_env._action_endpoint(current_state, {}) is True
    assert mock_env._action_endpoint(set_state(True), {}) is None
    assert mock_env._action_endpoint(current_state >> set_state, {}) is None
    assert mock_env._action_endpoint(set_state(True) + current_state, {}) is True
