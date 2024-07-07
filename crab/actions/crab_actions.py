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
