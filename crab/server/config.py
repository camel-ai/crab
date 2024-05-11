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
import argparse

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    HOST: str = "127.0.0.1"
    PORT: int = 8000
    ENVIRONMENT: str = "template_environment_config"


class EnvSettings(BaseSettings):
    DISPLAY: str = ":0"


def parse_args():
    parser = argparse.ArgumentParser(description="Application settings")
    parser.add_argument("--HOST", type=str, help="Host of the application")
    parser.add_argument("--PORT", type=int, help="Port of the application")
    parser.add_argument("--ENVIRONMENT", type=str, help="Environment to be loaded")
    return parser.parse_args()
