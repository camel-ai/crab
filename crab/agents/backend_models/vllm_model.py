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
from typing import Any

from openai.types.chat import ChatCompletionMessage

from crab import Action, ActionOutput, BackendOutput
from crab.agents.backend_models.openai_model import OpenAIModel
from crab.agents.utils import extract_text_and_code_prompts


class VLLMModel(OpenAIModel):
    def __init__(
        self,
        model: str,
        parameters: dict[str, Any] = dict(),
        history_messages_len: int = 0,
        base_url: str | None = None,
        api_key: str | None = None,
    ) -> None:
        if base_url is None:
            raise ValueError("base_url is required for VLLMModel")
        super().__init__(
            model,
            parameters,
            history_messages_len,
            False,
            base_url,
            api_key,
        )
        self.support_tool_call = False

    def reset(self, system_message: str, action_space: list[Action] | None) -> None:
        super().reset(system_message, action_space)
        self.action_schema = None

    def record_message(
        self, new_message: dict, response_message: ChatCompletionMessage
    ) -> None:
        self.chat_history.append([new_message])
        self.chat_history[-1].append(
            {"role": "assistant", "content": response_message.content}
        )

    def generate_backend_output(
        self, response_message: ChatCompletionMessage
    ) -> BackendOutput:
        content = response_message.content
        text_list, code_list = extract_text_and_code_prompts(content)

        action_list = []
        try:
            for code_block in code_list:
                action_object = json.loads(code_block)
                action_list.append(
                    ActionOutput(
                        name=action_object["name"], arguments=action_object["arguments"]
                    )
                )
        except json.JSONDecodeError as e:
            raise RuntimeError(f"Failed to parse code block: {code_block}") from e
        except KeyError as e:
            raise RuntimeError(f"Received invalid action format: {code_block}") from e

        return BackendOutput(
            message="".join(text_list),
            action_list=action_list,
        )
