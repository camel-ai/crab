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
# ruff: noqa: E501
import argparse
import importlib
import itertools
import json
import os
import random
from pathlib import Path

import networkx as nx
import yaml
from openai import OpenAI
from termcolor import colored

from .models import GeneratedTask, SubTask, SubTaskInstance, Task

SYSTEM_PROMPT_SINGLE = """
You are a wise operator who is familiar with both the Ubuntu and Android operating
systems. Our goal is to use the output of the source task as the input for the target
task. You should describe of the task they combined together using several imperative
sentences. You cannot provide any extra information such as detailed operation method,
yet only combined the taks description together in a reasonable way. You shouldn't fill
in the input attribute wrapped by curly brackets.

Source task:
Find out the city located at coordinate (8.65759263086632, 7.520403498426244) via Google Maps.

Target task:
Set the screen background as the first figure of {city_name} in Google.

Answer:
Using Google Maps, find the city located at coordinates (8.65759263086632,7.520403498426244), search Google for the first image of that city, and set this image as the desktop background on an Ubuntu system.
"""
USER_PROMPT_SINGLE = """
Source task:
{task1}

Target task:
{task2}

Answer:
"""

SELECT_USER_START = """
Source attribute:
{source_task}
Target tasks:
{target_tasks}
Select a task from target tasks
Answer:
"""

SELECT_SYSTEM_PROMPT = """
You are a wise operator who is familiar with both the Ubuntu and Android operating
systems. Our goal is to use the output of the source task as the input for the target
task. You should identify the most reasonable target task from the list, explain why you
choose it, and output the description of the task they combined together using several
imperative sentences. It is crucial to establish a connection between the source and
target tasks and select the best one as the output. Remember, you must select at least
one with the crucial output format. You must include the provided value and every
details in each task. You must use "======" to seperate each part (selected task number,
combined task description, and explanation) Here is an example:

Source task:
Find out the city located at coordinate (8.65759263086632, 7.520403498426244) via Google Maps.

Target tasks:
Task 0: Set the screen background as the first figure of {input attribute} in Google.
Task 1: Close the progress of {input attribute} app via task manager.
Task 2: Download {input attribute} from the app store.
Task 3: Create a PowerPoint with one page containing Mount Alps.jpg and named as {input attribute 2}.
Task 4: Send message {input attribute 1} to +81 09074540472.

Answer:
0
======
Using Google Maps, find the city located at coordinates (8.65759263086632,7.520403498426244), search Google for the first image of that city, and set this image as the desktop background on an Ubuntu system.
======
This task is the most relevant and directly utilizes the output of the source task.
Finding the city provides us with a specific location which can easily lead to a visual
representation. Searching for an image of the city to set as a background is a practical
application that visually celebrates the discovery of the city's identity.
"""

SELECT_USER_PROMPT = """
Source task:
{source_task}
Target tasks:
{target_tasks}

Answer:
"""


class TaskGenerator:
    """Class to generate tasks based on a directed graph of subtasks."""

    def __init__(
        self, attribute_pool: dict[str, list] = {}, subtasks: list[SubTask] = []
    ):
        """
        Initializes the TaskGenerator object.

        Parameters:
            attribute_pool (dict): A dictionary mapping attribute types to lists of possible values.
            subtasks (list): A list of SubTask objects to be included in the task generation graph.
        """
        self.G = nx.DiGraph()
        self.attribute_pool = attribute_pool
        self.graph_generation(subtasks)
        self.task_mapping = {task.id: task for task in subtasks}
        if not os.getenv("OPENAI_API_KEY"):
            os.environ["OPENAI_API_KEY"] = "EMPTY"
        self.client = OpenAI()

    @classmethod
    def from_config(cls, config_path: str) -> "TaskGenerator":
        """
        Class method to create a TaskGenerator instance from a configuration file.

        Parameters:
            config_path (str): Path to the YAML configuration file.

        Returns:
            TaskGenerator: An instance of TaskGenerator.
        """
        with open(config_path, "r") as f:
            data = yaml.safe_load(f)
        subtask_data = data["subtask"]
        attribute_pool = data["attribute_pool"]
        subtask_list = [
            SubTask(
                id=subtask["id"],
                description=subtask["description"],
                attribute_dict={
                    key: subtask["attribute_dict"][key].split("/")
                    for key in subtask["attribute_dict"]
                },
                output_type=subtask["output_type"],
            )
            for subtask in subtask_data
        ]
        return cls(attribute_pool, subtask_list)

    def graph_generation(self, subtask_list: list[SubTask]) -> None:
        """Generates a directed graph from a list of subtasks based on output and input types."""
        self.G.add_nodes_from(subtask_list)
        for input_node in self.G.nodes:
            for output_node in self.G.nodes:
                for name, type_list in output_node.attribute_dict.items():
                    for type in type_list:
                        if type == input_node.output_type:
                            self.G.add_edge(
                                input_node, output_node, attribute_name=name
                            )

    def combine(self, current_description: str, target_description: str) -> str:
        """
        Combines two task descriptions into a single task description using GPT model.

        Parameters:
            current_description (str): The current task description.
            target_description (str): The target task description to combine.

        Returns:
            str: The combined task description.
        """
        user_content = USER_PROMPT_SINGLE.format(
            task1=current_description, task2=target_description
        )
        response = self.client.chat.completions.create(
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT_SINGLE},
                {"role": "user", "content": user_content},
            ],
            model="gpt-4-turbo-preview",
        )
        return response.choices[0].message.content

    def gpt_choice(
        self,
        current_description: str,
        outgoing_edges: list[tuple[SubTask, SubTask, str]],
    ) -> tuple[SubTask, dict[str, str], str, str]:
        """
        Determines the best task choice from a list of possible target tasks using GPT model.

        Parameters:
            current_description (str): Description of the current task.
            outgoing_edges (list): List of possible outgoing edges representing target tasks.

        Returns:
            tuple: A tuple containing the chosen SubTask, attributes, new description, and combined description.
        """
        target_neighbours = ""
        selected_attributes = []
        new_descriptions = []
        for idx, edge in enumerate(outgoing_edges):
            _, node, attribute_name = edge
            attributes = self._fill_task_attributes(node, attribute_name)
            selected_attributes.append(attributes)
            kwargs = attributes.copy()
            kwargs[attribute_name] = "{" + attribute_name + "}"
            new_description = node.description.format(**kwargs)
            new_descriptions.append(new_description)
            target_neighbours += "Task {0}: {1}\n".format(idx, new_description)
        user_content = SELECT_USER_PROMPT.format(
            source_task=current_description,
            target_tasks=target_neighbours,
        )
        response = self.client.chat.completions.create(
            messages=[
                {"role": "system", "content": SELECT_SYSTEM_PROMPT},
                {"role": "user", "content": user_content},
            ],
            model="gpt-4-turbo-preview",
        )
        response_message = response.choices[0].message
        answers = response_message.content.split("======")
        index = int(answers[0].strip())
        combined_description = answers[1].strip()
        return (
            outgoing_edges[index][1],
            selected_attributes[index],
            new_descriptions[index],
            combined_description,
        )

    def random_walk(
        self, current_description: str, start_node: SubTask, random_number: int
    ) -> tuple[SubTask, dict[str, str]] | None:
        """
        Performs a random walk from the starting node to generate a task sequence.

        Parameters:
            current_description (str): The current task description.
            start_node (SubTask): The starting subtask node.
            random_number (int): Maximum number of edges to consider.

        Returns:
            tuple | None: A tuple containing the next SubTask, attributes if a next step is available, otherwise None.
        """
        out_edges = list(self.G.out_edges(start_node, data="attribute_name"))
        if len(out_edges) == 0:
            print(colored("\n*** No neighbour points, generation stopped ***\n", "red"))
            return None
        if start_node.output_type == "None":
            print(colored("\n*** Output None, generation will stop ***\n", "red"))
            return None

        if random_number <= len(out_edges):
            select_edge_list = random.sample(out_edges, random_number)
        else:
            select_edge_list = out_edges
        return self.gpt_choice(current_description, select_edge_list)

    def _fill_task_attributes(self, task: SubTask, kept_attribute: str):
        """
        Fills the task attributes by randomly selecting values from the attribute pool, except the kept attribute.

        Parameters:
            task (SubTask): The task whose attributes need to be filled.
            kept_attribute (str): The attribute to exclude from filling.

        Returns:
            dict: A dictionary of filled attributes.
        """
        attribute_types = task.attribute_dict.copy()
        attribute_types.pop(kept_attribute)
        return self._select_random_attributes(attribute_types)

    def _select_random_attributes(self, type_dict: dict[str, str]) -> dict[str, str]:
        """
        Randomly selects attributes for a task from the attribute pool based on the type dictionary.

        Parameters:
            type_dict (dict): A dictionary of attribute types to attribute names.

        Returns:
            dict: A dictionary of selected attributes.
        """
        result = {}
        for attr_name, attr_type_list in type_dict.items():
            pool = []
            for attr_type in attr_type_list:
                if attr_type not in self.attribute_pool:
                    raise ValueError(f"{attr_type} not in attribute pool.")
                pool.extend(self.attribute_pool[attr_type])
            result[attr_name] = random.choice(pool)
        return result

    @staticmethod
    def generate_single_node_task(subtask: SubTask):
        """
        Generates a single node task based on a SubTask instance.

        Parameters:
            subtask (SubTask): The subtask to generate a task for.

        Returns:
            tuple: A tuple containing the task description and a directed graph of the task.
        """
        print(colored(f"Generating task: {subtask.description}\n", "green"))
        attributes = {}
        for name, type_name in subtask.attribute_dict.items():
            value = input(
                colored(f'Input attribute "{name}" ({type_name}): ', "yellow")
            )
            attributes[name] = value
        description = subtask.description.format(**attributes)
        result_graph = nx.DiGraph()
        result_graph.add_node(SubTaskInstance(task=subtask, attribute=attributes))
        return description, result_graph

    def combine_subtask_list(self, subtask_list: list[SubTask]):
        """
        Combines a list of subtasks into a single task sequence.

        Parameters:
            subtask_list (list): A list of SubTask instances to combine.

        Returns:
            tuple: A tuple containing the final task description and a directed graph of the task sequence.
        """
        start_node = subtask_list[0]
        attributes = self._select_random_attributes(start_node.attribute_dict)
        result_graph = nx.DiGraph()
        output = input(
            colored(
                f"What is the output of {start_node.description.format(**attributes)}: ",
                "yellow",
            )
        )
        last_node = SubTaskInstance(
            task=start_node, attribute=attributes, output=output or None
        )
        result_graph.add_node(last_node)
        current_description = start_node.description.format(**attributes)
        for task in subtask_list[1:]:
            current_description = self.combine(current_description, task.description)
            key = next(iter(task.attribute_dict.keys()))
            attributes = {key: output}
            output = input(
                colored(
                    f"What is the output of {task.description.format(**attributes)}: ",
                    "yellow",
                )
            )
            current_node = SubTaskInstance(
                task=task, attribute=attributes, output=output or None
            )
            result_graph.add_edge(last_node, current_node)
            last_node = current_node
        return current_description, result_graph

    def combine_two_subtasks(
        self, sub_task_id_1: int, sub_task_id_2: int
    ) -> tuple[str, nx.DiGraph]:
        """
        Combines two subtasks into a single task sequence based on user input.

        Parameters:
            sub_task_id_1 (int): ID of the first subtask.
            sub_task_id_2 (int): ID of the second subtask.

        Returns:
            tuple: A tuple containing the combined task description and a directed graph of the task sequence.
        """
        sub_task_1 = self.task_mapping[sub_task_id_1]
        sub_task_2 = self.task_mapping[sub_task_id_2]
        print(colored(f"\nTask 1: {sub_task_1.description}", "cyan"))
        print(colored(f"Task 2: {sub_task_2.description}\n", "cyan"))
        attributes_1 = {}
        for name, types in sub_task_1.attribute_dict.items():
            value = input(
                colored(
                    f'Input attribute "{name}" ({types}) for the first task: ', "yellow"
                )
            )
            attributes_1[name] = value
        description_1 = sub_task_1.description.format(**attributes_1)
        output_1 = input(
            colored(
                f'What is the output of {description_1} ("{sub_task_1.output_type}"): ',
                "yellow",
            )
        )

        print(
            colored(
                f"\nThe output type of the first subtask is '{sub_task_1.output_type}'.\n",
                "cyan",
            )
        )
        attributes_2 = {}
        for name, types in sub_task_2.attribute_dict.items():
            if (
                sub_task_1.output_type in types
                or input(
                    colored(
                        f"Can the output '{sub_task_1.output_type}' be used as the '{name}' ({types}) of the second task? (yes/no): ",
                        "yellow",
                    )
                )
                .strip()
                .lower()
                == "yes"
            ):
                attributes_2[name] = output_1
            else:
                value = input(
                    colored(
                        f'Input attribute "{name}" ({types}) for the second task: ',
                        "yellow",
                    )
                )
                attributes_2[name] = value

        description_2 = sub_task_2.description.format(**attributes_2)

        while True:
            combined_description = self.combine(description_1, description_2)
            print(
                colored(f"\n*** Combined Task: {combined_description} ***\n", "green")
            )
            if (
                input(
                    colored(
                        "Do you want to re-generate the combined task? (yes/no): ",
                        "yellow",
                    )
                )
                .strip()
                .lower()
                != "yes"
            ):
                break
        result_graph = nx.DiGraph()
        node1 = SubTaskInstance(
            task=sub_task_1, attribute=attributes_1, output=output_1
        )
        node2 = SubTaskInstance(task=sub_task_2, attribute=attributes_2)
        result_graph.add_node(node1)
        result_graph.add_node(node2)
        result_graph.add_edge(node1, node2)

        return combined_description, result_graph

    def task_generation(
        self,
        start_id: int | None = None,
        max_iter: int = 3,
        random_number: int = 5,
    ) -> tuple[str, list[SubTask]]:
        """
        Generates a sequence of tasks starting from a given subtask ID or randomly.

        Parameters:
            start_id (int | None): The ID of the starting subtask or None to choose randomly.
            max_iter (int): The maximum number of iterations to perform in the generation process.
            random_number (int): The maximum number of neighbors to consider for random walk.

        Returns:
            tuple: A tuple containing the final task description and a list of SubTask objects.
        """
        description = ""
        task_list = []

        if start_id is None:
            start_node: SubTask = random.choice(list(self.G.nodes))
        else:
            for node in self.G.nodes:
                if node.id == start_id:
                    start_node: SubTask = node
                    break
        attributes = self._select_random_attributes(start_node.attribute_dict)
        description = start_node.description.format(**attributes)
        task_list.append((start_node, attributes, description))

        current_node = start_node
        for _ in range(max_iter - 1):
            next_node = self.random_walk(
                current_description=description,
                start_node=current_node,
                random_number=random_number,
            )
            if next_node is None:
                break
            task_list.append(next_node)
            description = next_node[3]
            current_node = next_node[0]
        return description, task_list

    @staticmethod
    def generate_evaluator(
        subtasks_graph: nx.DiGraph,
    ):
        """
        Generates an evaluator graph from a directed graph of subtask instances.

        Parameters:
            subtasks_graph (nx.DiGraph): A directed graph of subtask instances.

        Returns:
            nx.DiGraph: A directed graph representing the combined evaluator.
        """
        evaluator_map = {}
        for node in subtasks_graph.nodes:
            evaluator_map[node.id] = node.task.evaluator_generator(**node.attribute)
        combined_evaluator_graph = nx.union_all(list(evaluator_map.values()))
        for from_node, to_node in subtasks_graph.edges:
            from_node_evaluator = evaluator_map[from_node.id]
            sink_nodes = [
                node
                for node, out_degree in from_node_evaluator.out_degree()
                if out_degree == 0
            ]
            to_node_evaluator = evaluator_map[to_node.id]
            start_nodes = [
                node
                for node, in_degree in to_node_evaluator.in_degree()
                if in_degree == 0
            ]
            combined_evaluator_graph.add_edges_from(
                itertools.product(sink_nodes, start_nodes)
            )
        return combined_evaluator_graph

    @staticmethod
    def dump_generated_task(
        description,
        task_instance_graph,
        dir_path=".",
    ):
        """
        Saves a generated task to a file.

        Parameters:
            description (str): The description of the generated task.
            task_instance_graph (nx.DiGraph): The directed graph of the task instance.
            dir_path (str): The directory path where the task file will be saved.
        """
        mapping = {node: idx for idx, node in enumerate(task_instance_graph.nodes)}
        id_graph = nx.relabel_nodes(task_instance_graph, mapping)

        generated_task = GeneratedTask(
            description=description,
            tasks=list(task_instance_graph.nodes),
            adjlist="\n".join(nx.generate_adjlist(id_graph)),
        )
        file_path = Path(dir_path) / f"{generated_task.id}.json"
        with open(file_path, "w") as f:
            f.write(generated_task.model_dump_json(indent=4))

        print(
            colored(
                "\n====================================================================\n",
                "magenta",
            )
        )
        print(colored(f"Task saved to: {file_path}", "magenta"))

    def get_task_from_file(self, file_name) -> Task:
        """
        Loads a task from a file.

        Parameters:
            file_name (str): The file name containing the task data.

        Returns:
            Task: An instance of Task loaded from the file.
        """
        with open(file_name, "r") as f:
            config = json.load(f)
        description = config["description"]
        graph_map = {}
        for idx, task_config in enumerate(config["tasks"]):
            graph_map[idx] = SubTaskInstance(
                task=self.task_mapping[task_config["task"]],
                attribute=task_config["attribute"],
                output=task_config["output"],
            )
        lines = config["adjlist"].split("\n")
        graph = nx.parse_adjlist(lines, nodetype=int)
        subtask_graph = nx.relabel_nodes(graph, graph_map)
        evaluator = self.generate_evaluator(subtask_graph)

        setup_set = set()
        teardown_set = set()
        extra_action_set = set()
        for node in subtask_graph.nodes:
            setup_set.update(node.task.setup)
            teardown_set.update(node.task.teardown)
            extra_action_set.update(node.task.extra_action)
        return Task(
            id=config["id"],
            description=description,
            evaluator=evaluator,
            setup=list(setup_set),
            teardown=list(teardown_set),
            extra_action=list(extra_action_set),
        )


def load_subtasks(version):
    """
    Loads subtasks from specified benchmark version modules.

    Parameters:
        version (str): The version of the benchmark to load subtasks from.

    Returns:
        tuple: A tuple containing two collections of subtasks.
    """
    a_subtasks_module = importlib.import_module(
        f"benchmarks.crab-benchmark-{version}.subtasks.a_subtasks"
    )
    u_subtasks_module = importlib.import_module(
        f"benchmarks.crab-benchmark-{version}.subtasks.u_subtasks"
    )
    return a_subtasks_module.collection, u_subtasks_module.collection


def generate_length1_all(
    generator: TaskGenerator, dir_path: str, subtask_collection: list
):
    """
    Generates tasks for all subtasks in a collection and saves them.

    Parameters:
        generator (TaskGenerator): The task generator instance.
        dir_path (str): The directory path where the tasks will be saved.
        subtask_collection (list): The collection of subtasks to generate tasks for.
    """
    for task in subtask_collection:
        description, graph = generator.generate_single_node_task(task)
        generator.dump_generated_task(description, graph, dir_path)
        print(
            colored(
                "\n==================== Task Generation Completed ====================\n",
                "magenta",
            )
        )


def generate_length1_by_id(generator: TaskGenerator, dir_path: str):
    """
    Generates a single task for a specified subtask ID and saves it.

    Parameters:
        generator (TaskGenerator): The task generator instance.
        dir_path (str): The directory path where the task will be saved.
    """
    while True:
        subtask_id = input(colored("Please input the subtask ID: ", "yellow"))
        if subtask_id in generator.task_mapping:
            task = generator.task_mapping[subtask_id]
            print()
            description, graph = generator.generate_single_node_task(task)
            generator.dump_generated_task(description, graph, dir_path)
            print(
                colored(
                    "\n==================== Task Generation Completed ====================\n",
                    "magenta",
                )
            )
        else:
            print(colored("Invalid subtask ID. Please try again.", "red"))


def generate_length2_manual(generator: TaskGenerator, dir_path: str):
    """
    Manually generates a two-step task sequence from user-specified subtask IDs and saves it.

    Parameters:
        generator (TaskGenerator): The task generator instance.
        dir_path (str): The directory path where the task sequence will be saved.
    """
    while True:
        sub_task_id_1 = input(
            colored("Please input the id of the first subtask: ", "yellow")
        )
        sub_task_id_2 = input(
            colored("Please input the id of the second subtask: ", "yellow")
        )

        if (
            sub_task_id_1 in generator.task_mapping
            and sub_task_id_2 in generator.task_mapping
        ):
            description, graph = generator.combine_two_subtasks(
                sub_task_id_1=sub_task_id_1, sub_task_id_2=sub_task_id_2
            )
            generator.dump_generated_task(description, graph, dir_path)
            print(
                colored(
                    "\n==================== Task Composition Completed ====================\n",
                    "magenta",
                )
            )
        else:
            missing_ids = [
                id
                for id in [sub_task_id_1, sub_task_id_2]
                if id not in generator.task_mapping
            ]
            print(
                colored(
                    f"Invalid input: ID {', '.join(missing_ids)} not found. Please try again.",
                    "red",
                )
            )


def main():
    parser = argparse.ArgumentParser(description="Task Generator for CRAB Benchmarks")
    parser.add_argument(
        "--version", type=str, default="v0", help="Benchmark version (e.g., v0, v1)"
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=[
            "generate_length1_all",
            "generate_length2_manual",
            "generate_length1_by_id",
        ],
        help="Mode to run the task generator",
    )
    parser.add_argument(
        "--dir_path", type=str, help="Directory path to save the generated tasks"
    )
    parser.add_argument(
        "--config_path", type=str, help="Path to the task generation configuration file"
    )

    args = parser.parse_args()

    Path(args.dir_path).mkdir(parents=True, exist_ok=True)

    a_collection, u_collection = load_subtasks(args.version)
    all_collection = u_collection + a_collection

    print(
        colored(
            "\n==================== Task Generation Starting ====================\n",
            "magenta",
        )
    )
    if args.mode == "generate_length1_all":
        generator = TaskGenerator(subtasks=all_collection)
        generate_length1_all(generator, args.dir_path, all_collection)
    elif args.mode == "generate_length2_manual":
        with open(args.config_path, "r") as f:
            data = yaml.safe_load(f)
        attribute_pool = data["attribute_pool"]
        generator = TaskGenerator(attribute_pool, all_collection)
        generate_length2_manual(generator, args.dir_path)
    elif args.mode == "generate_length1_by_id":
        generator = TaskGenerator(subtasks=all_collection)
        generate_length1_by_id(generator, args.dir_path)
    else:
        print(
            colored(
                "Invalid mode selected. Please choose 'generate_length1_all', 'generate_length2_manual', or 'generate_length1_by_id'.",
                "red",
            )
        )


if __name__ == "__main__":
    main()
