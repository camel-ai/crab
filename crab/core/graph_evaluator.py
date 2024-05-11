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
from collections import deque
from typing import Any

import networkx as nx

from .environment import Environment
from .models import Evaluator


class GraphEvaluator:
    def __init__(
        self,
        incoming_graph_data,
        enable_shortcut: bool = False,
    ) -> None:
        self.G = nx.DiGraph(incoming_graph_data)
        assert nx.is_directed_acyclic_graph(self.G)
        self.count: int = 0
        self.total_nodes: int = self.G.number_of_nodes()
        assert self.total_nodes != 0
        self.complete_nodes: int = 0
        self.completeness: float = 0.0
        self.completeness_per_action: float = 0.0
        self.step_to_complete: int = self.G.number_of_edges()
        self.longest_unfinished_path_length: int = nx.dag_longest_path_length(self.G)
        self.enable_shortcut: bool = enable_shortcut

        # Set the sink node for the DAG:
        sink_nodes: list[Evaluator] = [
            node for node, out_degree in self.G.out_degree() if out_degree == 0
        ]
        if len(sink_nodes) != 1:
            raise ValueError("Graph should have exactly one sink node.")
        self.sink_node: Evaluator = sink_nodes[0]

        self.human_mode = False

        self.reset()

    def reset(self):
        self.count = 0
        for node in self.G.nodes():
            self.G.nodes[node]["remaining_predecessors"] = self.G.in_degree(node)
            self.G.nodes[node]["passing_count"] = None

    def step(
        self,
        envs: dict[str, Environment],
        default_env: str = "root",
    ):
        if self.is_complete():
            raise ValueError(
                "GraphEvaluator has already completed and "
                "cannot perform another step."
            )
        run_evaluators = set()
        evaluators = self.get_next_source_nodes()
        while evaluators:
            for evaluator in evaluators:
                if evaluator.local and self.human_mode:
                    result = True
                else:
                    environment = envs[evaluator.env_name or default_env]
                    result = environment.take_action(evaluator)
                if result:
                    self.G.nodes[evaluator]["passing_count"] = self.count
                    self.complete_nodes += 1
                    for _, out_node in self.G.out_edges(evaluator):
                        self.G.nodes[out_node]["remaining_predecessors"] -= 1
            if self.is_complete():
                self.complete_nodes = self.total_nodes
                break
            run_evaluators.update(evaluators)
            evaluators = self.get_next_source_nodes() - run_evaluators

        self.update()

    def get_next_source_nodes(self) -> set[Evaluator]:
        r"""Get next source nodes to evaluate."""
        if not self.enable_shortcut:
            source_nodes: list[Evaluator] = []
            for node in self.G.nodes(data=True):
                if (
                    node[1]["passing_count"] is None
                    and node[1]["remaining_predecessors"] == 0
                ):
                    source_nodes.append(node[0])
        else:
            source_nodes = list(self.G.nodes())

        return set(source_nodes)

    def entry(self) -> bool:
        return all(count is not None for _, count in self.G.nodes(data="passing_count"))

    def update(self):
        self.count += 1
        self.completeness = float(self.complete_nodes / self.total_nodes)
        self.completeness_per_action = self.completeness / self.count
        self.step_to_complete = self.calculate_step_to_complete()
        self.longest_unfinished_path_length = (
            self.calculate_longest_unfinished_path_length()
        )

    def calculate_longest_unfinished_path_length(self) -> int:
        longest_path_length: int = 0
        if self.G.nodes[self.sink_node]["passing_count"] is not None:
            return longest_path_length

        # Initialize set to keep track of visited nodes
        visited = set()
        # Initialize queue for BFS
        queue = deque([[self.sink_node]])
        # BFS traversal with path
        while queue:
            path = queue.popleft()
            node = path[0]
            # Mark the node as visited
            visited.add(node)
            longest_path_length = max(len(path), longest_path_length) - 1
            # Explore predecessor of the current node
            for predecessor in self.G.predecessors(node):
                # If predecessor is complete, skip it
                if self.G.nodes[predecessor]["passing_count"] is not None:
                    continue
                elif predecessor not in visited:
                    # Add path with predecessor to queue
                    queue.append([predecessor] + path)
        return longest_path_length

    def calculate_step_to_complete(self) -> int:
        # Initialize count for incomplete edges
        incomplete_edges: int = 0
        if self.G.nodes[self.sink_node]["passing_count"] is not None:
            return incomplete_edges

        # Initialize set to keep track of visited nodes
        visited = set()
        # Initialize queue for BFS
        queue = deque([self.sink_node])
        # BFS traversal
        while queue:
            # Pop node from queue
            node = queue.popleft()
            # Mark the node as visited
            visited.add(node)

            incomplete_edges += len(list(self.G.predecessors(node)))
            # Explore predecessor of the current node
            for predecessor in self.G.predecessors(node):
                # If predecessor is complete, skip it
                if self.G.nodes[predecessor]["passing_count"] is not None:
                    continue
                elif predecessor not in visited:
                    # Add predecessor to queue
                    queue.append(predecessor)

        return incomplete_edges

    def is_complete(self) -> bool:
        return self.G.nodes[self.sink_node]["passing_count"] is not None

    def get_completeness(self) -> float:
        return self.completeness

    def get_completeness_per_action(self) -> float:
        return self.completeness_per_action

    def get_step_to_complete(self) -> int:
        return self.step_to_complete

    def get_longest_unfinished_path_length(self) -> int:
        return self.longest_unfinished_path_length

    def stat(self) -> dict[str, Any]:
        return {
            "total_nodes": self.total_nodes,
            "complete_nodes": self.complete_nodes,
            "completeness": self.completeness,
            "completeness_per_action": self.completeness_per_action,
            "step_to_complete": self.step_to_complete,
            "longest_unfinished_path_length": self.longest_unfinished_path_length,
        }

    def _check_submit(self, environment: Environment) -> bool:
        """
        Check if the last action is _submit. If yes, return its result, either return
        False.
        """
        if not environment.trajectory:
            return False
        last_action = environment.trajectory[-1]
        if last_action[0] != "_submit":
            return False

        return last_action[2]

    def compute_radar_stats(self) -> dict[str, float]:
        longest_path_length = nx.dag_longest_path_length(self.G)
        return {
            "Completeness": float(self.completeness),
            "Efficiency": float(self.completeness_per_action),
            "Path Completeness Ratio": (
                longest_path_length - self.longest_unfinished_path_length
            )
            / longest_path_length,
        }

    @staticmethod
    def visualize(evaluators: list["GraphEvaluator"], path: str):
        import plotly.graph_objects as go

        fig = go.Figure()
        for i, evaluator in enumerate(evaluators):
            radar_stats = evaluator.compute_radar_stats()
            fig.add_trace(
                go.Scatterpolar(
                    r=list(radar_stats.values()),
                    theta=list(radar_stats.keys()),
                    fill="toself",
                    name=f"Graph Evaluator {i}",
                )
            )

        fig.update_layout(
            polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
            showlegend=True,
        )
        fig.update_layout(
            margin=dict(l=150, r=150, t=150, b=150),
        )
        fig.write_image(path, scale=12, width=600, height=600)
