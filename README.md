# model-runner

Self-hosted LLM inference on an HPC Slurm cluster ([CINECA Leonardo](https://www.hpc.cineca.it/systems/hardware/leonardo/)),
exposed through an OpenAI/Anthropic-compatible gateway that [Claude Code](https://claude.com/claude-code)
and [Avante.nvim](https://github.com/yetone/avante.nvim) can talk to directly.

Compute nodes have no internet access and no local disk, jobs get killed on wall-time
limits, and GPU allocations are shared and expensive to leave idle — this project handles
all three: [vLLM](https://github.com/vllm-project/vllm) under Singularity does the actual
serving (single- or multi-node, via Ray), a small supervisor daemon reconciles Slurm state
against desired state (rolling handoff before wall-time expiry, idle reaping, lazy
cold-start on demand), and a login-node gateway renders live backend state into a
[LiteLLM](https://github.com/BerriAI/litellm) config so clients always see a stable
endpoint regardless of which compute nodes are actually running it.

See **[docs/architecture.md](docs/architecture.md)** for the full design: measured cluster
constraints, the topology/quantization tradeoffs behind each model choice, the phased
build-out, and a running log of concrete issues hit and how they were fixed.

## Status

| Model | Role | Nodes | Status |
|---|---|---|---|
| `qwen3-32b` | fast/small | 1 | serving |
| `qwen3-coder-480b-awq` | primary (flagship) | 2 | serving |
| `kimi-k2-thinking` | — | 4 | staged, blocked on an upstream vLLM/tool-parser bug |

Both serving models are lazy-wake: idle backends are reaped automatically and cold-started
again on the next request. See `docs/architecture.md` §9 for open risks and known
limitations, including the Kimi blocker.

## Layout

```
config/models/*.toml   per-model config (Slurm resources + engine flags)
slurm/                 sbatch template: Singularity + Ray + vLLM + registry heartbeat
scripts/               login-node staging: container build, weight download
src/mr/
  config.py            paths and model specs
  registry.py          file-based service discovery on Lustre
  slurm.py             typed wrapper over the Slurm CLI
  cli.py               mrctl
  supervisor.py        reconcile loop (walltime handoff, idle reaping)
  demand.py            per-model lazy-wake flag (should this model be up right now?)
  gateway.py           renders LiteLLM config from the registry; runs the waker stub
tests/
  test_supervisor.py   decide()/apply() unit tests -- no cluster needed
  test_demand.py       demand.py unit tests
  test_gateway.py      render() + a real-HTTP test of the waker stub
```

## Setup

```bash
export MR_ROOT=$HOME/model-runner
export MR_STATE=$SCRATCH/model-runner       # registry, logs, caches
export MR_WEIGHTS=$FAST/weights             # flash tier -- there is no node-local disk
export PYTHONPATH=$MR_ROOT/src:$PYTHONPATH
alias mrctl='python3 -m mr.cli'

./scripts/build-container.sh                # login node only; needs internet
```

## Usage

```bash
mrctl models                    # list configured models
mrctl stage <model>              # download weights to $MR_WEIGHTS (login node only)
mrctl up <model>                 # submit a Slurm job for this model
mrctl status                     # show live backends: state, node, port, wall time left
mrctl down <model>                # cancel all backends for a model, clear lazy-wake demand
mrctl gateway                    # run the LiteLLM gateway (owns the client-facing endpoint)
mrctl supervise                  # run the reconcile loop (rolling handoff, idle reaping)
```

`mrctl gateway` and `mrctl supervise` are meant to run continuously (e.g. under `systemd
--user`, `tmux`, or a login-node cron-managed respawn) — the gateway is the one process
clients actually connect to, and the supervisor is what keeps backends matching desired
state without a human watching Slurm queues.

Point a client at the gateway (default `127.0.0.1:4000`) over an SSH tunnel from off-cluster
(login nodes are not internet-reachable, by design — see Constraints below):

```bash
ssh -L 4000:localhost:4000 <user>@login.leonardo.cineca.it

export ANTHROPIC_BASE_URL=http://127.0.0.1:4000   # Claude Code
export OPENAI_BASE_URL=http://127.0.0.1:4000/v1   # Avante.nvim / any OpenAI-compatible client
```

Adding a new model is a new `config/models/<name>.toml` (see existing ones for the shape:
Slurm resources, tensor/pipeline parallelism, the tool-call and reasoning parsers vLLM
needs for that specific model family) plus `mrctl stage` + `mrctl up`.

## Constraints worth remembering

- **A100 is SM 8.0 — no FP8.** Never `--kv-cache-dtype fp8`. Native-FP8 checkpoints
  (DeepSeek-V3, Kimi K2) must be dequantized to BF16 or re-quantized to AWQ/GPTQ-INT4.
- **Booster's driver is R535 (535.274.02) — CUDA 12.x generation only, no forward
  compat to CUDA 13.** Always build the container from a `-cu12x` image tag (see
  `scripts/build-container.sh`'s default) — verified with a real `torch.cuda.init()` +
  matmul on `boost_qos_dbg` before trusting any new tag in a real job.
- **No node-local disk.** `/tmp` is a 10 GB tmpfs. Weights come off Lustre every start.
- **`$HOME` is a 50 GB quota** and writes fail *silently* when it is full. Every cache
  (`HF_HOME`, `TRITON_CACHE_DIR`, `VLLM_CACHE_ROOT`, Singularity's) is redirected to
  `$MR_STATE`. Keep it that way.
- **Compute nodes have no internet access; login nodes are not internet-reachable.**
  Staging and container builds run on login nodes; client access goes through an SSH
  tunnel to the gateway, not a public endpoint.
- **Login-node processes get `RLIMIT_CPU=600s` soft** (hard is unlimited). Anything
  long-running or CPU-heavy there must raise it or it dies at 10 CPU-minutes with a
  confusing error. Already handled in `scripts/build-container.sh`, `mr.supervisor.run`,
  and `mr.gateway.run` — add it to anything new placed on a login node.
- **The Slurm allocation is shared.** A 2-node backend burns real node-hours per multi-day
  job. Idle reaping is on by default — an idle model sitting on 8 A100s is antisocial even
  when the budget allows it.
