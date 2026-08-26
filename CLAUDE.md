# 1. Introduction

This project aims at building an LLM inference runner to be run on an HPC, Slurm-backed environment.

## 2. Characteristics

The project should be able to run open-source models, like Qwen or Kimi, in a Slurm-based environment. In order to do this,
we can exploit Singularity as container runner, in case we need to run containerized, pre-built environment for inference.
The inference must be run in distributed way, on multiple nodes endowed with Nvidia GPUs. Moreover, the software must provide
API accessible through Claude Code or Avante neovim plugin

## 3. Constraints

- Compute nodes (where jobs are run) are not accessible through the Internet, so the API endpoint must be run on login nodes (e.g. through reverse proxy)
- Jobs are killed due to wall time, so there should be a way to automatically restart the jobs when needed
