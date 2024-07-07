import argparse
import warnings
from functools import partial
from pathlib import Path

from crab import (
    BenchmarkConfig,
    Experiment,
    MessageType,
    TaskGenerator,
    create_benchmark,
)
from crab.actions.crab_actions import complete
from crab.actions.visual_prompt_actions import (
    get_elements_prompt,
    groundingdino_easyocr,
)
from crab.agents.backend_models import ClaudeModel, GeminiModel, OpenAIModel
from crab.agents.policies import (
    MultiAgentByEnvPolicy,
    MultiAgentByFuncPolicy,
    SingleAgentPolicy,
)
from crab.utils.common import base64_to_image

from .android_env import ANDROID_ENV
from .dataset.android_subtasks import android_subtasks
from .dataset.handmade_subtasks import handmade_subtasks
from .dataset.ubuntu_subtasks import ubuntu_subtasks
from .ubuntu_env import UBUNTU_ENV

warnings.filterwarnings("ignore")


class CrabBenchmarkV0(Experiment):
    def get_prompt(self):
        observation, ob_prompt = self.benchmark.observe_with_prompt()

        for env in ob_prompt:
            with open(self.prompt_path / f"{env}_prompt.md", "a") as prompt_file:
                prompt_file.write(f"### Step {self.step_cnt}\n\n")
                for message, message_type in ob_prompt[env]:
                    if message_type == MessageType.IMAGE_JPG_BASE64:
                        file_name = f"{env}_{self.step_cnt}.png"
                        base64_to_image(message).save(self.image_path / file_name)
                        prompt_file.write(f"![](./images/{file_name})\n\n")
                    else:
                        prompt_file.write(message + "\n\n")
        return ob_prompt


def get_benchmark(env: str, ubuntu_url: str):
    ubuntu_env = UBUNTU_ENV.model_copy()
    ubuntu_env.remote_url = ubuntu_url
    ubuntu_tool = {
        "screenshot": groundingdino_easyocr(font_size=16) >> get_elements_prompt
    }
    android_tool = {
        "screenshot": groundingdino_easyocr(font_size=40) >> get_elements_prompt
    }

    if env == "ubuntu":
        prompting_tools = {"ubuntu": ubuntu_tool}
        benchmark_config = BenchmarkConfig(
            name="ubuntu_benchmark",
            tasks=[],
            environments=[ubuntu_env],
            prompting_tools=prompting_tools,
            root_action_space=[complete],
            multienv=True,
        )
    elif env == "android":
        prompting_tools = {"android": android_tool}
        benchmark_config = BenchmarkConfig(
            name="android_benchmark",
            tasks=[],
            environments=[ANDROID_ENV],
            prompting_tools=prompting_tools,
            root_action_space=[complete],
            multienv=True,
        )
    elif env == "cross":
        prompting_tools = {
            "android": android_tool,
            "ubuntu2204": ubuntu_tool,
        }
        benchmark_config = BenchmarkConfig(
            name="ubuntu_android_benchmark",
            tasks=[],
            environments=[ubuntu_env, ANDROID_ENV],
            prompting_tools=prompting_tools,
            root_action_space=[complete],
            multienv=True,
        )
    else:
        raise ValueError("Env not support")

    # Load from json config files by combining sub-tasks
    generator = TaskGenerator(subtasks=android_subtasks + ubuntu_subtasks)
    dir_path = (Path(__file__).parent / "dataset").resolve()
    tasks = []
    for task_json_files in dir_path.rglob("*.json"):
        task = generator.get_task_from_file(task_json_files)
        tasks.append(task)
    benchmark_config.tasks.extend(tasks)

    # Load from handmade tasks
    benchmark_config.tasks.extend(handmade_subtasks)

    benchmark_config.step_limit = 15
    return create_benchmark(benchmark_config)


MODEL_MAP = {
    "gpt4o": OpenAIModel(model="gpt4-o"),
    "gpt4turbo": OpenAIModel(model="gpt4-o"),
    "gemini": GeminiModel(model="gemini-1.5-pro-latest"),
    "claude": ClaudeModel(model="claude-3-opus-20240229"),
}

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Script for running benchmark with an agent."
    )
    parser.add_argument(
        "--model",
        type=str,
        help="gpt4o, gpt4turbo, gemini, claude or human",
        default="gpt4o",
    )
    parser.add_argument(
        "--policy",
        type=str,
        help="single, multi-by-func, or multi-by-env",
        default="single",
    )
    parser.add_argument(
        "--remote_url",
        type=str,
        help="remote url of Ubunutu environment",
        default="http://127.0.0.1:8000",
    )
    parser.add_argument("--env", type=str, help="ubuntu, android or cross")
    parser.add_argument("--id", type=str, help="task id")
    args = parser.parse_args()
    benchmark = get_benchmark(args.env)
    model = MODEL_MAP.get(args.model, None)
    if model is None:
        print("Unsupported model: ", args.model)
        exit()

    if args.policy == "single":
        agent_policy = SingleAgentPolicy(model_backend=model)
    elif args.policy == "multi-by-func":
        agent_policy = MultiAgentByFuncPolicy(
            main_agent_model_backend=model, tool_agent_model_backend=model
        )
    elif args.policy == "multi-by-env":
        agent_policy = MultiAgentByEnvPolicy(
            main_agent_model_backend=model, env_agent_model_backend=model
        )
    else:
        print("Unsupported policy: ", args.policy)
        exit()

    log_dir = (Path(__file__).parent / "logs").resolve()
    expeirment = Experiment(
        benchmark=benchmark,
        task_id=args.id,
        agent_policy=agent_policy,
        log_dir=log_dir,
    )
    expeirment.start_benchmark()
