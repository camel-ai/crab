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
from typing import Iterable

from .environment import Environment
from .evaluator import Evaluator


class Benchmark(ABC):
    @abstractmethod
    def get_corresponding_env(self) -> Environment:
        pass

    @abstractmethod
    def get_task_by_id(self, id: str) -> str:
        pass

    @abstractmethod
    def tasks(self) -> Iterable[tuple[str, Evaluator]]:
        pass
