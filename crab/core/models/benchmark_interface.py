from typing import Any

from pydantic import BaseModel


class StepResult(BaseModel):
    truncated: bool
    terminated: bool
    action_returns: Any
    evaluation_results: dict[str, Any]
    info: dict[str, Any]
