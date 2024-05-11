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
import os

import uvicorn
from fastapi import FastAPI
from fastapi.exceptions import RequestValidationError

from crab import EnvironmentConfig, create_environment

from .api import api_router
from .config import EnvSettings, Settings, parse_args
from .exception_handlers import (
    request_validation_exception_handler,
    unhandled_exception_handler,
)
from .logger import LOGGING_CONFIG
from .middleware import log_request_middleware
from .utils import get_benchmarks_environments


def init(environment_config: EnvironmentConfig) -> FastAPI:
    app = FastAPI(title="Desktop Agent Benchmark Environment Server")

    app.middleware("http")(log_request_middleware)
    app.add_exception_handler(
        RequestValidationError, request_validation_exception_handler
    )
    app.add_exception_handler(Exception, unhandled_exception_handler)
    app.include_router(api_router)

    app.environment = create_environment(environment_config)
    return app


if __name__ == "__main__":
    env_settings = EnvSettings()
    for field in env_settings.model_fields.keys():
        value = getattr(env_settings, field)
        os.environ[field] = value

    args = parse_args()
    kwargs = {k: v for k, v in vars(args).items() if v is not None}
    settings = Settings(**kwargs)

    benchmarks, environments = get_benchmarks_environments()
    app = init(environment_config=environments[settings.ENVIRONMENT])

    app.server_settings = settings
    uvicorn.run(
        app,
        host=settings.HOST,
        port=settings.PORT,
        access_log=False,
        log_config=LOGGING_CONFIG,
    )
