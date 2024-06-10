# Crab Benchmark v0

## Overview

crab-benchmark-v0 is the first benchmark released with the crab framework to provide a standard usage. It includes two virtual machine environments: an Android smartphone and an Ubuntu desktop computer, with 100 tasks and 59 different evaluator functions in the dataset. It effectively evaluates the MLM-based agents' performance on operating real-world tasks across multiple platforms.

## Get Started

Our benchmark contains two important parts: **Environments** and **Tasks**.

Since our Ubuntu environment is built upon KVM, setting it up locally requires you an experienced Linux user to deal with many small and miscellaneous issues. Therefore, we provide two environment setup methods:

* [Local Setup](./docs/environment_local_setup.md) provides you a step-by-step guideline to build environments on a Linux Machine with **at least one monitor and 32G memory**, but it will not cover details like how to install KVM on your machine because it's various on different Linux distros.
* For those want a quicker setup, we also provide a setup through [Google Clould Platform](./docs/environment_gcp_setup.md). Specifically, a disk image contains all required softwares and configurations, you can use [google remote desktop](https://remotedesktop.google.com/access/) to connect to the cloud computer. This method doesn't have any hardware limitations and when you set it up you can run the experiment immediately. As a tradeoff, the cloud computer cost around $0.4 per hour (depend on the machine zone) to meet the minimum hardware requirement.