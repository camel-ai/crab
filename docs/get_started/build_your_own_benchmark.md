# Build your own benchmark

## Overview

![](../assets/benchmark_config.png)

Crab benchmark system mainly consists of five types of component:

* `Action`: The fundamental building block of Crab framework, which represents a unit operation that can be taken by an agent or as a fixed process that called multi times in a benchmark.
* `Evaluator`: A specific type of `Action` that assess whether an agent has achieved its goal. Multiple evaluators can be combined together as a graph to enable complex evaluation.
* `Environment` A abstraction of an environment that the agent can take action and obverse in a given action and observation space. An environment can be launched on the local machine, a physical remote machine, or a virtual machine.
* `Task`: A task with a natural language description to instruct the agent to perform. It can include interaction with multiple environments. Notice that in the benchmark, a task should have an graph evaluator to judge if the task progress.
* `Benchmark`: The main body of the crab system that contains all required component to build a benchmark, including environments, tasks, prompting method. It controls several 

## Actions

Actions are the fundamental building blocks of the Crab system's operations.  Each action is encapsulated as an instance of the `Action` class. An action can convert into a JSON schema for language model agents to use.

An action is characterized by the following attributes:

- **Name**: A string identifier uniquely represents the action.
- **Entry**: A callable entry point to the actual Python function that executes the action.
- **Parameters**: A Pydantic model class that defines the input parameters the action accepts.
- **Returns**: A Pydantic model class that defines the structure of the return type the action produces.
- **Description**: An string providing a clear and concise description of what the action does and how it behaves.
- **Kept Parameters**: A list of parameters retained for internal use by the Crab system, which do not appear in the action's parameter list but are injected automatically at runtime. For exmaple we use `env` to represent the current environment object that action are taken in.
- **Environment Name**: An optional string that can specify the environment the action is associated with. Usually this attribute is only used by predifined actions like `setup` in an environment.

Here is an example of creating an action through python function:

```python
@action
def click(x: float, y: float) -> None:
    """
    click on the current desktop screen.

    Args:
        x (float): The X coordinate, as a floating-point number in the range [0.0, 1.0].
        y (float): The Y coordinate, as a floating-point number in the range [0.0, 1.0].

    """
    import pyautogui

    pyautogui.click(x,y)
```

The `@action` decorator transforms the `click` function into an `Action` with these mappings:

- The function name `click` becomes the action **name**.
- The parameters `x: float, y: float` with their type hints become the action **parameters**.
- The return type hint `-> None` is used for the action's **returns** field, indicating no value returned.
- The function's docstring provides a **description** for the action and its parameters, utilized in the JSON schema for the agent.
- The function body defines the action's behavior, executed when the action is called.


The `Action` class allows for different combination operations such as:

- **Pipe**: Using the `>>` operator, actions can be piped together, where the output of one action becomes the input to another, provided their parameters and return types are compatible.
- **Sequential Combination**: The `+` operator allows for two actions to be combined sequentially, executing one after the other.

## Evaluators

Evaluators in the Crab system are a specific type of `Action` that assess whether an agent has achieved its goal. They should return a boolean value, indicating whether the task's objective has been met. Multiple evaluators can be connected into a graph using the `networkx` package, enabling multi-stage evaluation, where different conditions can be checked in sequence or in parallel.

An example evaluator `check_file_exist` confirms the presence of a file at a given path, using the `os.path.isfile` method to return `True` if the file exists or `False` otherwise:

```python
@evaluator
def check_file_exist(file_path: str) -> bool:
    return os.path.isfile(file_path)
```

Extra attributes of evaluators:

- **Require Submit**: Indicates if the evaluator awaits a specific submission to carry out its assessment.

Logical operators allow for evaluator combinations:

- **AND (&)**: Requires all evaluators to succeed for a task to pass.
- **OR (|)**: Passes if any of the evaluators succeed.
- **NOT (~)**: Reverses the evaluation outcome.

The combined evaluator is still considered as **one evaluator** rather than a graph evaluator.

