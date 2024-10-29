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
import warnings
from pathlib import Path
from uuid import uuid4

import customtkinter as ctk

from crab import Experiment
from crab.agents.backend_models import OpenAIModel, ClaudeModel, GeminiModel
from crab.agents.policies import SingleAgentPolicy
from gui.utils import get_benchmark

warnings.filterwarnings("ignore")

AVAILABLE_MODELS = {
    "gpt-4o": ("OpenAIModel", "gpt-4o"),
    "gpt-4turbo": ("OpenAIModel", "gpt-4-turbo"),
    "gemini": ("GeminiModel", "gemini-1.5-pro-latest"),
    "claude": ("ClaudeModel", "claude-3-opus-20240229"),
}

def get_model_instance(model_key: str):
    if model_key not in AVAILABLE_MODELS:
        raise ValueError(f"Model {model_key} not supported")
    
    model_config = AVAILABLE_MODELS[model_key]
    model_class_name = model_config[0]
    model_name = model_config[1]

    if model_class_name == "OpenAIModel":
        return OpenAIModel(model=model_name, history_messages_len=2)
    elif model_class_name == "GeminiModel":
        return GeminiModel(model=model_name, history_messages_len=2)
    elif model_class_name == "ClaudeModel":
        return ClaudeModel(model=model_name, history_messages_len=2)

def assign_task():
    task_description = input_entry.get()
    input_entry.delete(0, "end")
    display_message(task_description)

    model = get_model_instance(selected_model.get())
    agent_policy = SingleAgentPolicy(model_backend=model)

    task_id = str(uuid4())
    benchmark = get_benchmark(task_id, task_description)
    experiment = Experiment(
        benchmark=benchmark,
        task_id=task_id,
        agent_policy=agent_policy,
        log_dir=log_dir,
    )
    # TODO: redirect the output to the GUI
    experiment.start_benchmark()


def display_message(message, sender="user"):
    chat_display.configure(state="normal")
    if sender == "user":
        chat_display.insert("end", f"User: {message}\n", "user")
    else:
        chat_display.insert("end", f"AI: {message}\n", "ai")
    chat_display.tag_config("user", justify="left", foreground="blue")
    chat_display.tag_config("ai", justify="right", foreground="green")
    chat_display.configure(state="disabled")
    chat_display.see("end")
    app.update_idletasks()


if __name__ == "__main__":
    # TODO: Handle JSON decode error from environment action endpoint and display model response in GUI
    log_dir = (Path(__file__).parent / "logs").resolve()

    ctk.set_appearance_mode("System")
    ctk.set_default_color_theme("blue")

    app = ctk.CTk()
    app.title("CRAB")
    app.geometry("400x500")

    model_frame = ctk.CTkFrame(app)
    model_frame.pack(pady=10, padx=10, fill="x")

    model_label = ctk.CTkLabel(model_frame, text="Select Model:")
    model_label.pack(side="left", padx=(0, 10))

    selected_model = ctk.StringVar(value="gpt-4o")
    model_dropdown = ctk.CTkOptionMenu(
        model_frame,
        values=list(AVAILABLE_MODELS.keys()),
        variable=selected_model,
    )
    model_dropdown.pack(side="left", fill="x", expand=True)

    chat_display_frame = ctk.CTkFrame(app, width=380, height=380)
    chat_display_frame.pack(pady=10)
    chat_display = ctk.CTkTextbox(
        chat_display_frame, width=380, height=380, state="disabled"
    )
    chat_display.pack()

    input_frame = ctk.CTkFrame(app)
    input_frame.pack(pady=10, padx=10, fill="x")

    input_entry = ctk.CTkEntry(input_frame)
    input_entry.pack(side="left", fill="x", expand=True, padx=(0, 10))

    send_button = ctk.CTkButton(input_frame, text="Send", command=assign_task)
    send_button.pack(side="right")
    app.mainloop()
