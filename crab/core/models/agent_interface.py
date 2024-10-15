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
from enum import IntEnum
from typing import Any

from pydantic import BaseModel

from .action import Action


class MessageType(IntEnum):
    TEXT = 0
    IMAGE_JPG_BASE64 = 1


Message = tuple[str, MessageType]


class ActionOutput(BaseModel):
    name: str
    arguments: dict[str, Any]
    env: str | None = None


class BackendOutput(BaseModel):
    message: str | None
    action_list: list[ActionOutput] | None


class EnvironmentInfo(BaseModel):
    description: str
    action_space: list[Action]
