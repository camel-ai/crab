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
