from abc import ABC, abstractmethod
from typing import Any

from .models import Action, BackendOutput, MessageType


class BackendModel(ABC):
    def __init__(
        self,
        model: str,
        history_messages_len: int = 0,
        parameters: dict[str, Any] = dict(),
    ) -> None:
        self.model = model
        self.parameters = parameters
        self.history_messages_len = history_messages_len

        assert self.history_messages_len >= 0

        self.reset("You are a helpfu assistant.", None)

    @abstractmethod
    def chat(self, contents: list[tuple[str, MessageType]]) -> BackendOutput:
        ...

    @abstractmethod
    def reset(
        self,
        system_message: str,
        action_space: list[Action] | None,
    ):
        ...

    @abstractmethod
    def get_token_usage(self):
        ...
