from enum import IntEnum
from typing import Any

from pydantic import BaseModel

from .action import Action


class MessageType(IntEnum):
    TEXT = 0
    IMAGE_JPG_BASE64 = 1


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