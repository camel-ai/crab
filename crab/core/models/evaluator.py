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
        if v.model_fields["returns"].annotation != bool:
            raise ValueError("Evaluator must return bool.")
        return v

    def __and__(self, other: "Evaluator") -> "Evaluator":
        def entry(*args, **kwargs):
            return self.entry(*args, **kwargs) and other.entry(*args, **kwargs)

        return Evaluator(
            name=f"{self.name}_and_{other.name}",
            description=f"{self.description} In the same time, {other.description}",
            parameters=self.parameters,
            returns=self.returns,
            entry=entry,
        )

    def __or__(self, other: "Evaluator") -> "Evaluator":
        def entry(*args, **kwargs):
            return self.entry(*args, **kwargs) or other.entry(*args, **kwargs)

        return Evaluator(
            name=f"{self.name}_or_{other.name}",
            description=(
                f"{self.description} If the previous one fails {other.description}"
            ),
            parameters=self.parameters,
            returns=self.returns,
            entry=entry,
        )

    def __invert__(self) -> "Evaluator":
        def entry(*args, **kwargs):
            return not self.entry(*args, **kwargs)

        return Evaluator(
            name=f"not_{self.name}",
            description=(
                f"Check if the following description is False. {self.description}"
            ),
            parameters=self.parameters,
            returns=self.returns,
            entry=entry,
        )

    def generate_submit_action(self) -> Action:
        result = self.model_copy()
        result.name = "_submit"
        result.description = (
            "Use this function to submit you answer and finish the task."
        )
        return result
