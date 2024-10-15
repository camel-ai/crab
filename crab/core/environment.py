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
import json
import logging
from typing import Any

from httpx import Client

from crab.utils import decrypt_message, encrypt_message, generate_key_from_env
from crab.utils.measure import timed

from .exceptions import ActionNotFound
from .models import Action, ClosedAction, EnvironmentConfig

logger = logging.getLogger("crab-server")


class Environment:
    """
    A crab environment for language model agent interaction and evaluation.

    This class supports action execution and observation within a simulated or actual
    ecosystem. The environment is defined by customizable action and observation spaces,
    comprising various crab actions. Actions should include comprehensive docstrings to
    facilitate agent understanding and interaction.

    Typically, users instantiate this class directly to perform actions within the local
    execution context (i.e., the device running the crab framework). This class may also
    serve as a base for specialized environments requiring unique action execution
    processes, such as forwarding actions to remote systems for execution. This is
    achieved by overriding the `take_action` method.

    Actions defined in the `action_space`, `observation_space`, or `reset`, as well as
    those invoked through the `take_action` method that include an `env` parameter, will
    have this parameter automatically populated with the current environment instance.
    This allows actions to access and manipulate environment states and variables.

    Attributes:
        name (str): The name of the environment.
        description (str): A description of the environment.
        trajectory (List[tuple[str, dict[str, Any], Any]]): A record of actions taken,
            their parameters, and the results.

    Args:
        name (str): The name of the environment.
        action_space (List[Action]): A list of actions that can be executed, defining
            the possible interactions agents can undertake.
        observation_space (List[ClosedAction]): A list of observations defining the
            possible states agents can perceive.
        description (str, optional): A textual description of the environment. Defaults
            to an empty string.
        reset (Action | None, optional): An action to reset the environment to its
            initial state. Defaults to `None`.
        remote_url (Action | None, optional): If set, the action will be taken at
            remote machine, by default it will be taken at local. Example:
            `http://192.168.1.1:8000`. Defaults to `None`.
    """

    def __init__(
        self,
        name: str,
        action_space: list[Action],
        observation_space: list[ClosedAction],
        description: str = "",
        reset: Action | None = None,
        remote_url: str | None = None,
        extra_attributes: dict[str, Any] = {},
    ) -> None:
        self.name = name
        self.description = description
        self.trajectory: list[tuple[str, dict[str, Any], Any]] = []
        self.observation_history: list[dict[str, Any]] = []

        self._origin_action_space = action_space
        self._observation_space = observation_space
        self._reset = reset
        self._action_map = {action.name: action for action in action_space}

        self._client: Client | None = None
        if remote_url is not None:
            self._client = Client(base_url=remote_url, timeout=60)
        for key, value in extra_attributes.items():
            setattr(self, key, value)

        self._enc_key = generate_key_from_env()

    def step(
        self,
        action_name: str,
        parameters: dict[str, Any] = {},
    ):
        """
        Executes an action that is in the action space and recorded to the trajectory.

        Args:
            action_name: Name of the action to execute. Must be in action space.
            parameters (dict[str, Any], optional): Parameters for the action. Defaults
                to an empty `dict`.

        Returns:
            Any: The result of the action execution.

        Raises:
            ActionNotFound: If the action is not found within the environment's action
                space.
        """
        if action_name not in self._action_map:
            logger.error(f'Env "{self.name}": receives unkown action "{action_name}"')
            raise ActionNotFound(f"Action {action_name} not found in the environment")
        action_handler = self._action_map[action_name]
        result = self.take_action(action_handler, parameters)
        self.trajectory.append((action_handler.name, parameters, result))
        return result

    def take_action(
        self,
        action: Action,
        parameters: dict[str, Any] = {},
    ) -> Any:
        """
        Executes an action within the environment.

        Args:
            action (Action): The action to execute. Can be an action name or an
                `Action` object.
            parameters (dict[str, Any], optional): Parameters for the action. Defaults
                to an empty `dict`.

        Returns:
            Any: The result of the action execution.
        """
        try:
            result = self._action_endpoint(action, parameters)
            logger.info(
                f'Env "{self.name}": action: "{action.name}" successed. '
                "result: {result}."
            )
            return result
        except:
            logger.exception(
                f'Env "{self.name}": action: "{action}" failed:', stack_info=True
            )
            raise

    @timed
    def observe(self) -> dict[str, Any]:
        """
        Observes the current state.

        Returns:
            Dict[str, Any]: A dictionary containing the current observations. Keys
                represent the names of the observation actions.
        """
        result = {o.name: self.take_action(o) for o in self.observation_space}
        self.observation_history.append(result)
        return result

    @timed
    def observe_with_prompt(
        self, prompt_tools: dict[str, Action]
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        """
        Observes the current state with prompt.
        """
        observations = self.observe()
        prompts = {}
        for ob_name, value in observations.items():
            if ob_name in prompt_tools:
                action = prompt_tools[ob_name]
                key = next(iter(action.get_required_params()))
                prompts[ob_name] = self._action_endpoint(action, {key: value})
        return observations, prompts

    def set_action(self, action: Action) -> None:
        """
        Adds an action in the environment's action space, either replace if the action
        name exist.

        Args:
            action (Action): The action to replace or add.
        """
        self._action_map[action.name] = action

    def start(self) -> None:
        """Starts the environment."""
        pass

    def close(self) -> None:
        """Closes the environment, performing any necessary cleanup."""
        pass

    def reset(self) -> None:
        """Resets the environment based on the provided reset action"""
        self._action_space = self._origin_action_space
        self.action_map = {action.name: action for action in self._action_space}
        if self._reset is not None:
            self.take_action(self._reset)

    @property
    def action_space(self) -> list[Action]:
        return list(self._action_map.values())

    @property
    def observation_space(self) -> list[ClosedAction]:
        return self._observation_space

    def _action_endpoint(self, action: Action, parameters: dict[str, Any]):
        """Rewrite to support different environments."""
        if self._client is not None and not action.local:
            data = json.dumps(
                {
                    "action": action.to_raw_action(),
                    "parameters": action.parameters(**parameters).model_dump(),
                }
            )
            content_type = "application/json"
            if self._enc_key is not None:
                data = encrypt_message(data, self._enc_key)
                content_type = "text/plain"

            # send action to remote machine
            response = self._client.post(
                "/raw_action",
                content=data,
                headers={"Content-Type": content_type},
            )

            resp_content = response.content.decode("utf-8")
            if self._enc_key is not None:
                resp_content = decrypt_message(resp_content, self._enc_key)

            resp_json = json.loads(resp_content)
            return resp_json["action_returns"]
        else:
            # or directly execute it
            action = action.set_kept_param(env=self)
            return action.run(**parameters)


def create_environment(config):
    if isinstance(config, EnvironmentConfig):
        return Environment(**dict(config))
    else:
        raise ValueError("Unsupported environment config type.")
