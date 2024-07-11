# 🦀 Crab: Cross-platform Agent Benchmark for Multimodal Embodied Language Model Agents

## Overview

Crab is a framework for building LLM agent benchmark environments in a Python-centric way.

#### Key Features

🌐 Cross-platform
* Create build agent environments that support various deployment options including in-memory, Docker-hosted, virtual machines, or distributed physical machines, provided they are accessible via Python functions.
* Let the agent access all the environments in the same time through a unified interface.

⚙ ️Easy-to-use Configuration
* Add a new action by simply adding a `@action` decorator on a Python function.
* Deine the environment by integrating several actions together.

📐 Novel Benchmarking Suite
* Define tasks and the corresponding evlauators in an intuitive Python-native way.
* Introduce a novel graph evaluator method providing fine-grained metrics.

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

## Cite
Please cite [our paper](https://arxiv.org/abs/2407.01511) if you use anything related in your work:
```
@article{xu2024crab,
  title={CRAB: Cross-environment Agent Benchmark for Multimodal Language Model Agents},
  author={Xu, Tianqi and Chen, Linyao and Wu, Dai-Jie and Chen, Yanjun and Zhang, Zecheng and Yao, Xiang and Xie, Zhiqiang and Chen, Yongchao and Liu, Shilong and Qian, Bochen and others},
  journal={arXiv e-prints},
  pages={arXiv--2407},
  year={2024}
}
```