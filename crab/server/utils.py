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
import importlib
import inspect
import pkgutil


def get_instances(package, class_type):
    instance_dict = {}
    # Iterate through all modules in the specified package
    for _, name, ispkg in pkgutil.iter_modules(
        package.__path__, package.__name__ + "."
    ):
        if ispkg:
            continue  # Skip subpackages
        module = importlib.import_module(name)
        for name, obj in inspect.getmembers(module):
            if isinstance(obj, class_type):
                instance_dict[name] = obj
    return instance_dict


def get_benchmarks_environments():
    from crab import BenchmarkConfig, EnvironmentConfig, benchmarks, environments

    benchmark_configs = get_instances(benchmarks, BenchmarkConfig)
    environment_configs = get_instances(environments, EnvironmentConfig)

    return benchmark_configs, environment_configs
