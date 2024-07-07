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
