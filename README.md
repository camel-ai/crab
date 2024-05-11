# 🦀 Crab: Cross-platform Agent Benchmark for Multimodal Embodied Language Model Agents

## Overview

Crab is a framework for building LLM agent benchmark environments in a Python-centric way.

#### Key Features

* 🌐 Cross-platform
  * Build agent environments in memory, hosted through a docker environment, installed a virtual machine, or as a distributed physical machine, as long as they can be accessed by Python functions.
  * Let the agent access all the environments in the same time through a unified interface.
* ⚙ ️Easy-to-use Configuration
  * Add a new action by simply adding a `@action` decorator on a Python function.
  * Deine the environment by integrating several actions together.
* 📐 Novel Benchmarking Suite
  * Define tasks and the corresponding evlauators in an intuitive Python-native way.
  * Introduce a novel graph evaluator method proving fine-grained metrics.

## Installation

#### Prerequisites

- Python 3.10 or newer
- pip

```bash
pip install crab-framework[visual-prompt]
```

## Examples

#### Run template environment with openai agent

You can run the examples using the following command.

```bash
export OPENAI_API_KEY=<your api key>
python examples/single_env.py
python examples/multi_env.py
```

#### Run desktop environment with openai agent

You can run the examples using the following command.

```bash
export OPENAI_API_KEY=<your api key>
python examples/desktop_env.py "Open Firefox"
```