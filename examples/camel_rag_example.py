# =========== Copyright 2024 @ CAMEL-AI.org. All Rights Reserved. ===========
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# =========== Copyright 2024 @ CAMEL-AI.org. All Rights Reserved. ===========
from termcolor import colored
import os
import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin

from crab import Benchmark, create_benchmark
from crab.agents.backend_models.camel_rag_model import CamelRAGModel
from crab.agents.policies import SingleAgentPolicy
from crab.benchmarks.template import template_benchmark_config
from camel.types import ModelType, ModelPlatformType


def start_benchmark(benchmark: Benchmark, agent: SingleAgentPolicy):
    for step in range(20):
        print("=" * 40)
        print(f"Start agent step {step}:")
        observation = benchmark.observe()["template_env"]
        print(f"Current environment observation: {observation}")
        response = agent.chat(
            {
                "template_env": [
                    (f"Current environment observation: {observation}", 0),
                ]
            }
        )
        print(colored(f"Agent take action: {response}", "blue"))

        for action in response:
            response = benchmark.step(
                action=action.name,
                parameters=action.arguments,
                env_name=action.env,
            )
            print(
                colored(
                    f'Action "{action.name}" success, stat: '
                    f"{response.evaluation_results}",
                    "green",
                )
            )
            if response.terminated:
                print("=" * 40)
                print(
                    colored(
                        f"Task finished, result: {response.evaluation_results}",
                        "green"
                    )
                )
                return


def prepare_vim_docs():
    """Prepare Vim documentation for RAG"""
    print(colored("Starting Vim documentation preparation...", "yellow"))
    base_url = "https://vimdoc.sourceforge.net/htmldoc/usr_07.html"
    content_dir = "vim_docs"
    os.makedirs(content_dir, exist_ok=True)
    
    print(colored("Fetching main page...", "yellow"))
    response = requests.get(base_url)
    soup = BeautifulSoup(response.text, 'html.parser')
    
    # Process the main page first
    main_content = soup.get_text(separator='\n', strip=True)
    with open(os.path.join(content_dir, "main.txt"), 'w', encoding='utf-8') as f:
        f.write(f"Source: {base_url}\n\n{main_content}")
    
    links = [link for link in soup.find_all('a') 
             if link.get('href') and not link.get('href').startswith(('#', 'http'))]
    total_links = len(links)
    print(colored(f"Found {total_links} documentation pages to process", "yellow"))
    
    processed_files = []
    for idx, link in enumerate(links, 1):
        href = link.get('href')
        full_url = urljoin(base_url, href)
        try:
            print(colored(f"Processing page {idx}/{total_links}: {href}", "yellow"))
            
            # Fetch and process page
            page_response = requests.get(full_url)
            page_soup = BeautifulSoup(page_response.text, 'html.parser')
            for tag in page_soup(['script', 'style']):
                tag.decompose()
            content = page_soup.get_text(separator='\n', strip=True)
            
            # Save content
            filename = os.path.join(content_dir, f"{href.replace('/', '_')}.txt")
            with open(filename, 'w', encoding='utf-8') as f:
                f.write(f"Source: {full_url}\n\n{content}")
            processed_files.append(filename)
            print(colored(f"✓ Saved {href}", "green"))
            
        except Exception as e:
            print(colored(f"✗ Error processing {full_url}: {e}", "red"))
    
    print(colored("Documentation preparation completed!", "green"))
    return processed_files


if __name__ == "__main__":
    print(colored("=== Starting RAG-enhanced benchmark ===", "cyan"))
    
    # Initialize benchmark and environment
    print(colored("\nInitializing benchmark environment...", "yellow"))
    benchmark = create_benchmark(template_benchmark_config)
    task, action_space = benchmark.start_task("0")
    env_descriptions = benchmark.get_env_descriptions()

    doc_files = prepare_vim_docs()
    
    print(colored("\nInitializing RAG model...", "yellow"))
    rag_model = CamelRAGModel(
        model="gpt-4o",
        model_platform=ModelPlatformType.OPENAI,
        parameters={"temperature": 0.7}
    )
    
    print(colored("Processing documents for RAG...", "yellow"))
    for doc_file in doc_files:
        print(colored(f"Processing {doc_file}...", "yellow"))
        rag_model.process_documents(doc_file)
    print(colored("RAG model initialization complete!", "green"))
    
    print(colored("\nSetting up agent...", "yellow"))
    agent = SingleAgentPolicy(model_backend=rag_model)
    agent.reset(task.description, action_space, env_descriptions)
    
    print(colored("\nStarting benchmark execution:", "cyan"))
    print("Start performing task: " + colored(f'"{task.description}"', "green"))
    start_benchmark(benchmark, agent)
    benchmark.reset()
