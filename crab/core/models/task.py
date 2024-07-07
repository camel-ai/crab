from typing import Any, Callable, Literal
from uuid import uuid4

import networkx as nx
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    field_validator,
    model_serializer,
)

from .action import Action, ClosedAction
from .evaluator import Evaluator


class Task(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    id: str
    description: str
    evaluator: nx.DiGraph | Evaluator
    setup: list[ClosedAction] | ClosedAction = []
    teardown: list[ClosedAction] | ClosedAction = []
    extra_action: list[Action] = []

    @field_validator("evaluator")
    @classmethod
    def change_evaluator_to_graph(cls, evaluator: nx.DiGraph | Evaluator) -> str:
        if isinstance(evaluator, Evaluator):
            graph = nx.DiGraph()
            graph.add_node(evaluator)
            return graph
        return evaluator

    @field_validator("setup", "teardown")
    @classmethod
    def to_list(cls, action: Action | list[Action]) -> list[Action]:
        if isinstance(action, Action):
            return [action]
        return action


class SubTask(BaseModel):
    id: str
    description: str
    attribute_dict: dict[str, list[str] | str]
    output_type: str
    output_generator: Callable[[Any], str] | Literal["manual"] | None = None
    evaluator_generator: Callable[[Any], nx.DiGraph] | None = None
    setup: list[ClosedAction] | ClosedAction = []
    teardown: list[ClosedAction] | ClosedAction = []
    extra_action: list[Action] = []

    def __hash__(self) -> int:
        return hash(self.id)

    @field_validator("attribute_dict")
    @classmethod
    def expand_attribute_type(
        cls,
        attribute_dict: dict[str, list[str] | str],
    ) -> dict[str, list[str]]:
        attribute_dict = attribute_dict.copy()
        for key in attribute_dict:
            if isinstance(attribute_dict[key], str):
                attribute_dict[key] = [attribute_dict[key]]
        return attribute_dict


class SubTaskInstance(BaseModel):
    task: SubTask
    attribute: dict[str, Any]
    output: str | None = None
    id: str = Field(default_factory=uuid4)

    def __hash__(self) -> int:
        return hash(self.id)

    @model_serializer
    def dump_model(self) -> dict[str, Any]:
        return {
            "task": self.task.id,
            "attribute": self.attribute,
            "output": self.output,
        }


class GeneratedTask(BaseModel):
    description: str
    tasks: list[SubTaskInstance]
    adjlist: str
    id: str = Field(default_factory=uuid4)
