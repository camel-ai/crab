from pathlib import Path
from time import sleep

from crab import (
    Benchmark,
    BenchmarkConfig,
    EnvironmentConfig,
    Task,
    create_benchmark,
    evaluator,
)
from crab.actions.desktop_actions import (
    click,
    hotkey_press,
    key_press,
    screenshot,
    set_screen_size,
    write_text,
)
from crab.actions.visual_prompt_actions import (
    get_elements_prompt,
    groundingdino_easyocr,
)
from crab.agents.single_agent.openai_agent import OpenAIAgent
from crab.utils.common import base64_to_image


def start_benchmark(benchmark: Benchmark, agent: OpenAIAgent):
    for step in range(20):
        print("=" * 40)
        print(f"Start agent step {step}:")
        observation, prompt = benchmark.observe_with_prompt()
        screenshot = observation["desktop"]["screenshot"]
        marked_screenshot, text_prompt = prompt["desktop"]["screenshot"]
        base64_to_image(marked_screenshot).save(f"logs/screenshot_{step}.png")
        response = agent.chat(
            [
                ("Here is the current screenshot:", 0),
                (screenshot, 1),
                ("Here is the screenshot with element labels", 0),
                (marked_screenshot, 1),
                (
                    f"Your target: {task.description}\n"
                    "Explain what do you see from the current observation and a plan of"
                    " next action. You should choose an action from the action space.",
                    0,
                ),
            ]
        )
        print("\033[94m" f"Agent Reponse: {response['content']}" "\033[0m")
        print(f"So agent take action: {response['action_list']}")

        for a in response["action_list"]:
            benchmark_result = benchmark.step(*a)
            if benchmark_result.terminated:
                print(
                    "\033[92m"
                    f"Task finished, result: {benchmark_result.evaluation_results}"
                    "\033[0m"
                )
                return
            print(
                "\033[92m"
                f'Action "{a[0]}" success, stat: {benchmark_result.evaluation_results}'
                "\033[0m"
            )

        sleep(1)


ENV_CONFIG = EnvironmentConfig(
    name="desktop",
    action_space=[click, key_press, write_text, hotkey_press],
    observation_space=[screenshot],
    description="A desktop environment with a single display.",
    reset=set_screen_size,
)

BENCHMARK_CONFIG = BenchmarkConfig(
    name="desktop",
    tasks=[],
    environments=[ENV_CONFIG],
    prompting_tools={
        "desktop": {
            "screenshot": groundingdino_easyocr(font_size=20) >> get_elements_prompt
        }
    },
)


@evaluator
def empty_evaluator() -> bool:
    return False


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Script for running benchmark with an agent."
    )
    parser.add_argument("instruction", type=str, help="Instruction of the task.")
    args = parser.parse_args()

    BENCHMARK_CONFIG.tasks.append(
        Task(id="0", description=args.instruction, evaluator=empty_evaluator)
    )
    benchmark = create_benchmark(BENCHMARK_CONFIG)

    task, action_space = benchmark.start_task("0")
    agent = OpenAIAgent(
        task.description,
        action_space,
    )
    Path("logs").mkdir(exist_ok=True)
    start_benchmark(benchmark, agent)
