# Crab Benchmark v0

## Overview

`crab-benchmark-v0` is a benchmark released with the crab framework to provide a standard usage. It includes two virtual machine environments: an Android smartphone and an Ubuntu desktop computer, with 100 tasks and 59 different evaluator functions in the dataset. It effectively evaluates the MLM-based agents' performance on operating real-world tasks across multiple platforms.

## Get Started

Our benchmark contains two important parts: **Environments** and **Tasks**.

Since our Ubuntu environment is built upon KVM, setting it up locally requires you an experienced Linux user to deal with many small and miscellaneous issues. Therefore, we provide two environment setup methods:

* [Local setup](./docs/environment_local_setup.md) provides you a step-by-step guideline to build environments on a Linux Machine with **at least one monitor and 32G memory**, but it doesn't cover details like how to install KVM on your machine because they are various on different Linux distros.
* For those who want a quicker setup, we also provide a setup through [Google Clould Platform](./docs/environment_gcp_setup.md). Specifically, we publish a disk image contains all required softwares and configurations on google cloud, you can use your own google account to create a cloud computer through this disk image and use [google remote desktop](https://remotedesktop.google.com/access/) to connect to it. This method doesn't have any hardware limitations and when you set it up you can run the experiment immediately. As a tradeoff, the cloud computer that meets the minimum hardware requirement costs around $0.4 per hour (depend on the machine zone).