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
from fastapi import APIRouter, Request

from crab.utils.common import base64_to_callable

from .logger import crab_logger as logger

api_router = APIRouter()


@api_router.post("/raw_action")
async def raw_action(request: Request):
    """Perform the specified action with given parameters."""
    # Extract query parameters as a dictionary
    request_json = await request.json()
    action = request_json["action"]
    parameters = request_json["parameters"]
    entry = base64_to_callable(action["dumped_entry"])
    logger.info(f"remote action: {action['name']} received. parameters: {parameters}")
    if "env" in action["kept_params"]:
        parameters["env"] = request.app.environment
    return {"action_returns": entry(**parameters)}
