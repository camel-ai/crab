# Get started

`crab-benchmark-v0` is a benchmark released with the crab framework to provide a standard usage. It includes two virtual machine environments: an Android smartphone and an Ubuntu desktop computer, with 100 tasks and 59 different evaluator functions in the dataset. It effectively evaluates the MLM-based agents' performance on operating real-world tasks across multiple platforms.

## Concept

Our benchmark contains two important parts: **Environments** and **Tasks**.

#### Environment

Since our Ubuntu environment is built upon KVM, setting it up locally requires you an experienced Linux user to deal with many small and miscellaneous issues. Therefore, we provide two environment setup methods:

* [Local setup](./environment_local_setup.md) provides you a step-by-step guideline to build environments on a Linux Machine with **at least one monitor and 32G memory**, but it doesn't cover details like how to install KVM on your machine because they are various on different Linux distros.
* For those who want a quicker setup, we also provide a setup through [Google Clould Platform](./environment_gcp_setup.md). Specifically, we publish a disk image contains all required software and configurations on google cloud, you can use your own google account to create a cloud computer through this disk image and use [google remote desktop](https://remotedesktop.google.com/access/) to connect to it. This method doesn't have any hardware limitations and when you set it up you can run the experiment immediately. As a tradeoff, the cloud computer that meets the minimum hardware requirement costs around $0.4 per hour (depend on the machine zone).

We connect to the Android environment via ADB, so any Android device, from an emulator to a physical smartphone, will work. You should ensure ADB is installed on your system and can be directly called through the command line. In our experiment, we used the built-in emulator of [Android Studio](https://developer.android.com/studio) to create a Google Pixel 8 Pro virtual device with the release name \textit{R} and installed necessary extra Apps.

#### Task

We manage our task dataset using a CRAB-recommended method. Sub-tasks are defined through Pydantic models written in Python code, and composed tasks are defined in JSON format, typically combining several sub-tasks. The sub-tasks are defined in [android_subtasks](https://github.com/camel-ai/crab/tree/main/crab-benchmark-v0/dataset/android_subtasks.py) and [ubuntu_subtasks](https://github.com/camel-ai/crab/tree/main/crab-benchmark-v0/dataset/ubuntu_subtasks.py). The JSON files storing composed tasks are categorized into [android](https://github.com/camel-ai/crab/tree/main/crab-benchmark-v0/dataset/android/), [ubuntu](https://github.com/camel-ai/crab/tree/main/crab-benchmark-v0/dataset/ubuntu/), and [cross-platform](https://github.com/camel-ai/crab/tree/main/crab-benchmark-v0/dataset/cross/). The tasks in android and ubuntu directories are single-environment task and those in cross directory are cross-environment tasks. Additionally, we create several tasks by hand instead of composing sub-tasks to provide semantically more meaningful tasks, which are found in [handmade tasks](https://github.com/camel-ai/crab/tree/main/crab-benchmark-v0/dataset/handmade_tasks.py).

## Experiment

After setting up the environment, you can start the experiment. A brief overview of the experiment is as follows:

1. Open the Ubuntu environment virtual machine and the Android environment emulator.
2. Start the CRAB server in the Ubuntu environment and get its IP address and port. Let's say they are `192.168.122.72` and `8000`.
3. Choose a task. As an example, we take the task with ID `a3476778-e512-40ca-b1c0-d7aab0c7f18b` from [handmade_tasks](https://github.com/camel-ai/crab/tree/main/crab-benchmark-v0/dataset/handmade_tasks.py). The task is: "Open the 'Tasks' app on Android, check the first incomplete task, then perform the task according to its description."
4. Run [main.py](./main.py) with the command `poetry run python -m crab-benchmark-v0.main --model gpt4o --policy single --remote-url http://192.168.122.72:8000 --task-id a3476778-e512-40ca-b1c0-d7aab0c7f18b`. In this command, `--model gpt4o` and `--policy single` determine the agent system, `--remote-url` specifies the Ubuntu environment interface, and `--task-id` indicates the task to be performed.

