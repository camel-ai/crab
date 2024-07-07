from pydantic import BaseModel, field_validator

from .action import Action


class Evaluator(Action):
    require_submit: bool = False

    @field_validator("returns", mode="after")
    @classmethod
    def must_return_bool(cls, v: type[BaseModel]) -> type[BaseModel]:
        if v.model_fields["returns"].annotation != bool:
            raise ValueError("Evaluator must return bool.")
        return v

    def __and__(self, other: "Evaluator") -> "Evaluator":
        Action._check_combinable(self, other)
        result = self.model_copy()
        result.name = (f"{self.name}_and_{other.name}",)
        result.description = f"{self.description} In the same time, {other.description}"
        self_entry = self.entry
        other_entry = other.entry
        result.entry = lambda: self_entry() and other_entry()
        return result

    def __or__(self, other: "Evaluator") -> "Evaluator":
        Action._check_combinable(self, other)
        result = self.model_copy()
        result.name = (f"{self.name}_or_{other.name}",)
        result.description = (
            f"{self.description} If the previous one fails {other.description}"
        )
        self_entry = self.entry
        other_entry = other.entry
        result.entry = lambda: self_entry() or other_entry()
        return result

    def __invert__(self) -> "Evaluator":
        result = self.model_copy()
        result.name = f"not_{self.name}"
        result.description = (
            f"Check if the following description is False. {self.description}"
        )
        self_entry = self.entry
        result.entry = lambda: not self_entry()
        return result
