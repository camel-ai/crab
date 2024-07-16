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
import csv
from pathlib import Path
from typing import Any


class CSVLog:
    def __init__(self, csv_path: Path, headers: list[str]) -> None:
        self.csv_path = csv_path
        self.header = headers
        if not csv_path.exists():
            with open(csv_path, "w", newline="") as file:
                writer = csv.writer(file)
                writer.writerow(headers)

    def write_row(self, data: list[Any]):
        assert len(data) == len(self.header)
        with open(self.csv_path, "a", newline="") as file:
            writer = csv.writer(file)
            writer.writerow(data)
