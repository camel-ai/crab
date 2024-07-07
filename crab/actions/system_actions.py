import subprocess
from time import sleep

from crab.core.decorators import action


@action
def delay(time: float) -> None:
    sleep(time)


@action
def run_bash_command(command: str) -> str:
    """
    Run a command using bash shell. You can use this command to open any application by
    their name.

    Args:
        command: The commmand to be run.

    Return:
        stdout and stderr
    """
    p = subprocess.run(["bash", command], capture_output=True)
    return f'stdout: "{p.stdout}"\nstderr: "{p.stderr}"'
