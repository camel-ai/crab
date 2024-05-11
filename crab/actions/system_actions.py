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
