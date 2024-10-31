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

from crab.agents.backend_models import ClaudeModel, GeminiModel, OpenAIModel
from crab.agents.policies import SingleAgentPolicy
from gui.gui_experiment import GuiExperiment
from gui.utils import get_benchmark

warnings.filterwarnings("ignore")

AVAILABLE_MODELS = {
    "GPT-4o": ("OpenAIModel", "gpt-4o"),
    "GPT-4 Turbo": ("OpenAIModel", "gpt-4-turbo"),
    "Gemini": ("GeminiModel", "gemini-1.5-pro-latest"),
    "Claude": ("ClaudeModel", "claude-3-opus-20240229"),
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

    try:
        model = get_model_instance(model_dropdown.get())
        agent_policy = SingleAgentPolicy(model_backend=model)

        task_id = str(uuid4())
        benchmark = get_benchmark(task_id, task_description)
        experiment = GuiExperiment(
            benchmark=benchmark,
            task_id=task_id,
            agent_policy=agent_policy,
            log_dir=log_dir,
        )
    
        experiment.set_display_callback(display_message)

        def run_experiment():
            try:
                experiment.start_benchmark()
            except Exception as e:
                display_message(f"Error: {str(e)}", "ai")

        import threading
        thread = threading.Thread(target=run_experiment, daemon=True)
        thread.start()
    
    except Exception as e:
        display_message(f"Error: {str(e)}", "ai")


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
    log_dir = (Path(__file__).parent / "logs").resolve()

    ctk.set_appearance_mode("System")
    ctk.set_default_color_theme("blue")

    app = ctk.CTk()
    app.title("CRAB")
    app.geometry("500x1000")

    bold_font = ctk.CTkFont(family="Crimson Pro", size=18, weight="bold")
    normal_font = ctk.CTkFont(family="Crimson Pro", size=18, weight="normal")

    model_dropdown = ctk.CTkOptionMenu(
        app,
        values=list(AVAILABLE_MODELS.keys()),
        font=bold_font,
        width=200,
    )
    model_dropdown.set(list(AVAILABLE_MODELS.keys())[0])
    model_dropdown.pack(pady=10, padx=10, fill="x")

    chat_display_frame = ctk.CTkFrame(app, width=480, height=880)
    chat_display_frame.pack(pady=10, expand=True, fill="y")
    chat_display = ctk.CTkTextbox(
        chat_display_frame, width=480, height=880, state="disabled", font=normal_font
    )
    chat_display.pack(expand=True, fill="both")

    input_frame = ctk.CTkFrame(app)
    input_frame.pack(pady=10, padx=10, fill="x", expand=True)

    input_entry = ctk.CTkEntry(input_frame, font=normal_font)
    input_entry.pack(side="left", fill="x", expand=True, padx=(0, 10))

    send_button = ctk.CTkButton(
        input_frame, text="Send", font=bold_font, command=assign_task
    )
    send_button.pack(side="right")
    app.mainloop()
