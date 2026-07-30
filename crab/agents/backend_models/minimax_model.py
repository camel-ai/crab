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
from typing import Any

from crab.agents.backend_models.openai_model import OpenAIModel

# OpenAI-compatible base URLs for the available regions. The global endpoint is
# used by default; set ``base_url`` explicitly to target another region.
BASE_URLS: dict[str, str] = {
    "global_en": "https://api.minimax.io/v1",
    "cn_zh": "https://api.minimaxi.com/v1",
}
DEFAULT_BASE_URL = BASE_URLS["global_en"]


class MiniMaxModel(OpenAIModel):
    """Backend model served through an OpenAI-compatible API.

    Reuses :class:`OpenAIModel` for the chat protocol and only supplies the
    default base URL when ``base_url`` is not provided, so callers can point to
    a regional endpoint via :attr:`BASE_URLS`.
    """

    def __init__(
        self,
        model: str,
        parameters: dict[str, Any] | None = None,
        history_messages_len: int = 0,
        tool_call_required: bool = True,
        base_url: str | None = None,
        api_key: str | None = None,
    ) -> None:
        super().__init__(
            model=model,
            parameters=parameters,
            history_messages_len=history_messages_len,
            tool_call_required=tool_call_required,
            base_url=base_url or DEFAULT_BASE_URL,
            api_key=api_key,
        )
