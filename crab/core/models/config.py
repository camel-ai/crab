from typing import Any

from pydantic import BaseModel

from .action import Action, ClosedAction
from .task import Task


class EnvironmentConfig(BaseModel):
    name: str
    action_space: list[Action]
    observation_space: list[ClosedAction]
    description: str = ""
    reset: Action | None = None
    remote_url: str | None = None
    extra_attributes: dict[str, Any] = {}


class VMEnvironmentConfig(BaseModel):
    inside_environment: EnvironmentConfig
    remote_url: str = "http://192.168.0.0:8000"


class BenchmarkConfig(BaseModel):
    name: str
    tasks: list[Task]
    environments: list[EnvironmentConfig]
    default_env: str | None = None
    multienv: bool = False
    prompting_tools: dict[str, dict[str, Action]] = {}
    root_action_space: list[Action] = []
    step_limit: int = 30
    common_setup: list[ClosedAction] = []
