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
from abc import ABC, abstractmethod

from .models import Action, ActionOutput, Message


class AgentPolicy(ABC):
    @abstractmethod
    def chat(
        self,
        observation: dict[str, list[Message]],
    ) -> list[ActionOutput]: ...

    @abstractmethod
    def reset(
        self,
        task_description: str,
        action_spaces: dict[str, list[Action]],
        env_descriptions: dict[str, str],
    ) -> None: ...

    @abstractmethod
    def get_token_usage(self) -> int: ...

    @abstractmethod
    def get_backend_model_name(self) -> str: ...
