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
from pydantic import BaseModel, field_validator

from .action import Action


class Evaluator(Action):
    require_submit: bool = False

    @field_validator("returns", mode="after")
    @classmethod
    def must_return_bool(cls, v: type[BaseModel]) -> type[BaseModel]:
        if v.model_fields["returns"].annotation is not bool:
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
