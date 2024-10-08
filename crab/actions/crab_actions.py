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
from time import sleep

from crab import action, evaluator


@action(env_name="root")
def submit(content: str) -> None:
    """Submit your answer through this action. For exmaple, if you are required to
    submit a word "apple", you can use submit(content="apple").

    Args:
        content: the content to submit
    """
    pass


@evaluator(env_name="root")
def check_submit(text: str, env) -> bool:
    if env.trajectory:
        action_name, params, _ = env.trajectory[-1]
        if action_name == "submit" and text in params["content"]:
            return True
    return False


@action(env_name="root")
def complete() -> bool:
    """When you think the task is completed, use this action to notify the system. For
    exmaple, if you successfully complete the task, you can use complete().
    """
    pass


@action(env_name="root")
def wait() -> bool:
    """If the environment is still processing your action and you have nothing to do in
    this step, you can use wait().
    """
    sleep(5)


def get_element_position(element_id, env):
    """Get element position provided by function `zs_object_detection`"""
    box = env.element_position_map[element_id]
    x = (box[0] + box[2]) / 2
    y = (box[1] + box[3]) / 2
    return round(x), round(y)
