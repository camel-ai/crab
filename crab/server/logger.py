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
import logging

uvicorn_logger = logging.getLogger("uvicorn")
uvicorn_logger.setLevel(logging.INFO)

crab_logger = logging.getLogger("crab-server")
crab_logger.setLevel(logging.INFO)

LOGGING_CONFIG = {
    "version": 1,
    "disable_existing_loggers": False,
    "formatters": {
        "default": {
            "()": "uvicorn.logging.DefaultFormatter",
            "format": "[%(asctime)s %(process)d:%(threadName)s] %(name)s - "
            "%(levelname)s - %(message)s | %(filename)s:%(lineno)d",
        },
        "logformat": {
            "format": "[%(asctime)s %(process)d:%(threadName)s] %(name)s - "
            "%(levelname)s - %(message)s | %(filename)s:%(lineno)d"
        },
    },
    "handlers": {
        "file_handler": {
            "class": "logging.FileHandler",
            "level": "INFO",
            "formatter": "logformat",
            "filename": "info.log",
            "encoding": "utf8",
            "mode": "a",
        },
        "default": {
            "formatter": "default",
            "class": "logging.StreamHandler",
            "stream": "ext://sys.stderr",
        },
    },
    "loggers": {
        "uvicorn.error": {
            "level": "INFO",
            "handlers": ["default", "file_handler"],
            "propagate": False,
        }
    },
    "root": {
        "level": "INFO",
        "handlers": ["default", "file_handler"],
        "propagate": False,
    },
}
