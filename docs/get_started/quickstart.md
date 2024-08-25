# Quickstart

The `Benchmark` class is a comprehensive framework for evaluating language model agents across various tasks and environments. It provides a flexible structure to manage multiple environments and tasks, offering single and multi-environment execution modes.

The following image shows an overview of how `Benchmark` works.

![](../assets/crab_overview.png)

## Basic Usage

### Step 1: Importing the Benchmark

Begin by importing the predefined benchmark from the `crab.benchmarks` module. For exmple, here we import `template_benchmark_config`:

```python
from crab.benchmarks import template_benchmark_config
```

### Step 2: Creating the Benchmark

Use the `create_benchmark` function to create an instance of a `Benchmark` class based on the imported benchmark configuration:

```python
from crab import create_benchmark

benchmark = create_benchmark(template_benchmark_config)
```

### Step 3: Starting a Task

Select a task to start within the benchmark. The task ID should correspond to one of the predefined tasks in the benchmark configuration. Use the `start_task` method to initialize and begin the task:

```python
# Starting the task with ID "0"
task, action_space = benchmark.start_task("0")
```

### Step 4: Running the Benchmark Loop

Execute actions and observe the results using the `step` and `observe` methods:

```python
from crab.client.openai_interface import OpenAIAgent

# Initialize the agent by benchmark task and action_space
agent = OpenAIAgent(task, action_space)

# Define a function to run the benchmark
def run_benchmark(benchmark, agent):
    for step in range(20):  # Define the number of steps as per your requirements
        print("=" * 40)
        print(f"Starting step {step}:")

        # Get the current observations and prompts
        observation = benchmark.observe()

        # Process the observations and determine the next action
        action_result = agent.determine_next_action(observation)
        
        # Execute the action and get the result
        step_result = benchmark.step(action_result.action, action_result.parameters)

        # Check current evaluation result.
        print(step_result.evaluation_results)

        # Check if the task is terminated and break the loop if so
        if step_result.terminated:
            print("Task completed successfully.")
            print(step_result.evaluation_results)
            break

run_benchmark(benchmark, agent)
```

### Step 5: Completing the Benchmark

Clean up and reset the benchmark after completion using the`reset`:

```python
benchmark.reset()
```
