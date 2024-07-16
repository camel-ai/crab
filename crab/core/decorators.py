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
from typing import Callable

from .models import Action, Evaluator


def _decorator(func, cls: type[Action], options: dict | None = None) -> Action:
    action = cls.from_function(func)
    if options is not None:
        for key in options:
            setattr(action, key, options[key])

    return action


def action(*args: Callable, env_name: str | None = None, local=False):
    """Use @action to change a function to an Action"""
    if args and callable(args[0]):
        return _decorator(args[0], Action)

    return lambda func: _decorator(func, Action, {"env_name": env_name, "local": local})


def evaluator(
    *args: Callable,
    require_submit: bool = False,
    env_name: str | None = None,
    local=False,
):
    """Use @evaluator to change a function to an Evaluator"""
    if args and callable(args[0]):
        return _decorator(args[0], Evaluator)

    return lambda func: _decorator(
        func,
        Evaluator,
        {"require_submit": require_submit, "env_name": env_name, "local": local},
    )
