# model-runner

LLM inference serving on **Leonardo (CINECA)**: multi-node vLLM under Slurm, exposed
through a login-node gateway that Claude Code and Avante.nvim can talk to.

See **[docs/architecture.md](docs/architecture.md)** for the design, the measured
cluster facts it rests on, and the phased plan.

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
```

## Phase 0 — get one model serving

```bash
./scripts/build-container.sh              # login node; needs internet
mrctl stage qwen3-32b                     # login node; stripes then downloads
mrctl up qwen3-32b
mrctl status
curl http://<node>:<port>/v1/models       # login->compute is direct, no tunnel
```

The `READY after Ns` line in `$MR_STATE/logs/qwen3-32b.<jobid>.out` is the Lustre
load time. **Record it** — it sets `handoff_lead_s` for every model, and it is the
main unknown the rest of the design is parameterised on.

Measured (job 54085509): **244s** cold off Lustre for Qwen3-32B/TP4, striped `-c 16 -S
4M`. `config/models/qwen3-32b.toml` sets `handoff_lead_s = 3000` from this.

## Phase 2 — LiteLLM gateway

```bash
PIP_CACHE_DIR=$MR_STATE/cache/pip pip install --user "litellm[proxy]"   # one-time, login node
mrctl gateway                              # owns litellm; renders config from the registry,
                                            # restarts it whenever the backend set changes
```

Verified end-to-end against a live backend: OpenAI-format `/v1/chat/completions`
(Avante), Anthropic-format `/v1/messages` (Claude Code), tool calling (`get_weather`
round-trip returned a correct `tool_use` block), and SSE streaming on `/v1/messages`.

Then verified with the **actual `claude` CLI**, not just raw `curl`:

```bash
cd /some/test/dir   # anywhere claude can read a file from, for a real tool-use check
ANTHROPIC_BASE_URL=http://127.0.0.1:4000 ANTHROPIC_API_KEY=sk-local-litellm-test \
  claude --bare -p --model sonnet --allowedTools=Read "Read secret.txt and tell me what it says."
```

This surfaced three real bugs the raw-`curl` tests didn't catch (all fixed, all in
`docs/architecture.md` §9 risks 10–12): Claude Code resolves `--model sonnet` to
`claude-sonnet-5` client-side, so the original `model_alias_map` (keyed on the bare
word) never matched — fixed with wildcard `model_list` entries; Claude Code's default
`max_tokens=64000` exceeds any of our models' context — fixed with
`model_info.max_output_tokens` + `litellm_settings.modify_params: true`; and Qwen3's
`<think>` block leaked into the visible response — fixed with `--reasoning-parser qwen3`.

```bash
export ANTHROPIC_BASE_URL=http://127.0.0.1:4000   # Claude Code, after an SSH tunnel to the login node
export OPENAI_BASE_URL=http://127.0.0.1:4000/v1   # Avante
```

No `LITELLM_MASTER_KEY` is set yet — fine while this is single-user and the proxy only
listens on `127.0.0.1`, revisit for §9 risk 5 (multi-user).

## Phase 3 — supervisor (rolling handoff + idle reaping)

```bash
PYTHONPATH=src python3 -m unittest tests.test_supervisor -v   # decide() is pure; no cluster needed
mrctl supervise                                                # the real reconcile loop
```

`decide()` (pure, unit-tested) reads the registry and Slurm state and returns what to do;
`refresh_activity()` (the one I/O step, scrapes each backend's own vLLM `/metrics`) and
`apply()` carry it out. Three real bugs were caught by actually running this against a live
backend rather than trusting the unit tests alone (all in `docs/architecture.md` §9,
risks 13-15):

- Idle detection originally used the heartbeat, which never goes stale enough to fire since
  it refreshes every 30s regardless of real traffic.
- The very first cold start (empty registry, nothing running yet) launched three duplicate
  jobs a tick apart, because a job has no registry record until its own script starts
  running, which can be minutes after `sbatch` returns on a busy queue.
- A long-lived `mrctl gateway` process silently dropped a perfectly healthy backend for
  ~34 minutes after a mid-session schema change to the registry's `Backend` record — its
  in-memory class was older than what a compute job's (always-fresh-code) heartbeat had
  started writing, so every read raised a silently-swallowed `TypeError`. Fixed structurally:
  `registry.read_all()` now filters unknown JSON keys before constructing, so an old reader
  survives a newer writer instead of losing the record. This is a real, recurring hazard for
  this architecture (long-lived login-node processes vs. always-fresh compute jobs) any time
  `Backend`'s shape changes — not a one-off.

**Live-verified end to end, not just unit-tested**: a genuinely idle backend was correctly
reaped after exactly one `idle_timeout_s` window and the supervisor launched a fresh
replacement with no manual intervention; and — caught by accident when a manual `mrctl up`
raced the supervisor's own cold-start launch — two backends briefly went READY together,
`start_drain` correctly picked the older one, and it was cancelled within one tick once
`refresh_activity()` saw its in-flight count reach zero, ending in exactly one healthy
backend. **Not yet live-verified**: the walltime-*triggered* path specifically (a job
launching a successor because its own wall is running out, rather than any other reason two
backends end up READY) — that needs a job to actually approach its multi-day wall, which
isn't practical to wait out; that trigger condition rests on the unit tests instead.

Operational note (§9 risk 16): once the supervisor is running, manual `mrctl up` for
cold-start recovery is redundant and can race it — reach for `mrctl up --force` only to
deliberately add a replica or replace a specific job.

### Lazy-wake (§9 risk 17)

Idle-reaping only saves anything if the backend it cancels actually *stays down* until
someone wants it again. Caught live: 22 reap/relaunch cycles overnight, none of them
saving a thing, because `desired` was hardcoded to 1 forever. Fixed with `mr.demand` (a
tiny persisted per-model flag — `reap_idle` clears it, the supervisor reads it fresh every
tick) plus a "waker" stub `mr.gateway` runs (`_WakerHandler`, `127.0.0.1:4100`): a cold
model still gets a real `model_list` row in LiteLLM, pointed at the waker instead of a
vLLM URL, so a request for it bumps demand back to 1 and gets an immediate, clear answer
instead of an opaque routing error. The supervisor's next tick (≤ 60s) sees the demand and
cold-starts it.

Getting the waker's response right took two real bugs, both invisible except over real
HTTP: not draining the request body risked stalling an HTTP/1.1 client mid-connection, and
— the actual cause of a ~90+ second hang with zero log output — answering with **503**
makes LiteLLM auto-retry it as a transient `ServiceUnavailableError` before ever replying
to the client. **400** (correctly: this isn't infra flakiness, it's "come back later")
returns in under 100ms. `tests/test_gateway.py` locks this in with a real HTTP round-trip,
not just a direct handler call — that's the only way this class of bug shows up at all.

```bash
curl http://127.0.0.1:4000/v1/chat/completions -d '{"model":"qwen3-32b", ...}'  # cold
# -> immediate 400, "model 'qwen3-32b' is cold; a start has been triggered ..."
# retry in ~1-4 min once the supervisor's next tick launches it
```

## Phase 4 — multi-node

```bash
./scripts/build-container.sh v0.27.1-cu129   # rebuild: `ray` isn't in the upstream image at all
mrctl up qwen3-32b-4n                        # validation config: Qwen3-32B deliberately spread
                                              # TP4/PP4 across 4 real nodes, no new download needed
```

`ray` is not on `PATH` or importable in the upstream `vllm/vllm-openai` image — confirmed
empirically. Adding it needed a real container customization step, and `--fakeroot` is
unavailable on this system (no subuid/subgid mapping for this user), which rules out a `.def`
file's `%post`. `scripts/build-container.sh` now builds an unprivileged writable **sandbox**
(same as the plain `docker://` pull always did), `pip install`s `ray[default]` directly into
it, then repacks to an immutable `.sif`. Two things bit this specifically, both fixed in the
script: a writable sandbox can't auto-create missing bind targets (`mkdir -p` Leonardo's
top-level mountpoints into the sandbox first), and `-C`/`--containall` — only ever there to
dodge that mount error — swaps in a tiny tmpfs home/tmp and made `pip`'s download overflow it
with `No space left on device` despite `$SCRATCH` having petabytes free.

**Live-verified, not just started**: job 54343864, Qwen3-32B (TP4×PP4, 16 GPUs, 4 real nodes)
reached `READY after 167s` with correct per-node rank/PP/TP assignment in the log, a real
completion succeeded through the actual TP4/PP4 endpoint, and 10/10 concurrent requests
succeeded (200 OK, ~0.6s each) — Phase 4's "4+ nodes stable under load" exit criterion, met.
Getting there needed two more fixes: `--distributed-executor-backend ray` (vLLM assumes
single-node otherwise, even under a live multi-node Ray cluster — it names this exact flag in
its own error), and `--min-nodes` on `ray symmetric-run` (undocumented as required, but there's
no stated guarantee otherwise that the entrypoint waits for every node before running). Full
narrative, including the first attempt's failure, in `docs/architecture.md` §9 risk 18.

**Not yet done**: the full BF16 flagship (~960 GB) doesn't fit in `$FAST` at all and needs
5-6 nodes; the AWQ-INT4 version (~262 GB, 2 nodes) staged in Phase 5 below is the one
actually running. NCCL/IB tuning beyond the HCAs already pinned in `vllm-server.sbatch` is
untouched; nothing so far has needed more.

## Phase 5 — multi-model routing (done, live-verified)

```bash
mrctl stage qwen3-coder-480b-awq   # ~262 GB AWQ-INT4 community checkpoint; verify size via
                                    # the HF tree API before staging a repo this large
mrctl up qwen3-coder-480b-awq
```

Adding a real second model exposed a routing bug before it ever shipped: `mr.gateway`
picked the sonnet/opus target as `models[0]` (alphabetical) and the haiku target via a
`"32b"` name match — both only ever worked because `qwen3-32b` was the sole model
configured. `"qwen3-32b"` sorts *before* `"qwen3-coder-480b-awq"`, so this would have
silently kept routing real Claude Code traffic to the small model. Fixed with an explicit
per-model `role` (`"primary"`/`"small"` in `config.ModelSpec`), checked first, falling back
to the old heuristic only when unset. `tests/test_gateway.py` locks it in with
alphabetically-hostile model names. Full narrative: `docs/architecture.md` §9 risk 20.

Staging the ~262 GB checkpoint hit a real, reproducible bug: `hf download` failed twice in
a row with an identical `httpx`/`brotlicffi` decoding error, both times on the exact same
file (`model.safetensors.index.json`, 8.2 MB) — not random flakiness, since a retry
reproduced it exactly rather than succeeding or failing differently. Diffing the HF tree
API's file list against what was actually on disk pinned it to that one file; fetched it
directly with `curl` instead (an unaffected code path), then verified completeness two
ways (every listed file present, every shard the index references exists). §9 risk 19 has
the full story — worth knowing if a future large download fails identically on retry.

Avante.nvim is wired to the gateway as a real second client alongside Claude Code
(`~/.config/nvim/lua/plugins/avante.lua`, overriding LazyVim's `ai.avante` extra): two
custom OpenAI-compatible providers, `model-runner` (flagship) and `model-runner-fast`
(`qwen3-32b`), no API key needed (gateway has no master key), a 5-minute timeout to cover a
lazy-wake cold start. Verified three ways, not just configured: the spec loads cleanly in a
real headless Neovim + avante.nvim session, the exact request Avante's own code builds
matches the gateway's expectations, and firing that exact request against the live backend
returned a correct streamed completion (reasoning content separated from the final answer,
correct `finish_reason` and usage). One caveat: Avante doesn't know to retry a cold-model
response automatically (unlike this project's own lazy-wake design intent) — the first
request after idle-reaping shows an error; resend it once `mrctl status` shows `ready`.

The flagship's first real bring-up (job 54379904, `READY after 407s`, TP4×PP2 across 2
nodes) surfaced a bug the earlier tests couldn't have caught: `--tool-call-parser hermes`
(correct for `qwen3-32b`, copied over without checking) doesn't match this model's tool-call
syntax — a `get_weather` request came back as *raw unparsed XML text* in `content`, no error,
just silently unusable, exactly what architecture.md §7 calls "the most common way a
self-hosted Claude Code setup silently fails." Checked the container's own tool-parser
registry instead of guessing: `qwen3_coder` is the one actually registered for this syntax.
Resubmitted (job 54381732, `READY after 348s`) and confirmed live — proper `tool_calls`
array, correct arguments, `finish_reason: "tool_calls"`. **Lesson**: `--tool-call-parser`
is per-model-family, not per-vendor; "it's a Qwen model" isn't enough to assume a sibling
model's parser carries over.

Then hit a second bug getting `sonnet` to actually resolve to the newly-role-tagged
flagship: it kept resolving to `qwen3-32b` even after the role fix and the new model were
both live. Not a bug in the fix (confirmed correct by the unit test and a direct check) —
the running `mr.gateway` process had been up since before the fix was written, still
running the old alphabetical-order logic in memory with no way to know newer code existed
on disk. Same hazard as risk 15, just failing silently instead of crashing loudly this
time. Restarting the gateway fixed it immediately. **There's no automatic detection for
this yet** — restarting `mr.cli gateway`/`mr.cli supervise` after a code change that
affects their behavior is still a manual step. Full narrative: `docs/architecture.md` §9
risks 21-22.

End-to-end proof, not just each piece in isolation: the real `claude` CLI, using the
`sonnet` alias, used its own `Read` tool against the flagship model and got the correct
answer — the same test pattern as Phase 2, now against the actual production model.

**Not started**: queue-aware UX (a real Slurm-start-estimate ETA in the waker's response,
instead of the current rough "~1-4 min" guess), metrics, and the K2-class INT4 conversion.

## Constraints worth remembering

- **A100 is SM 8.0 — no FP8.** Never `--kv-cache-dtype fp8`. Native-FP8 checkpoints
  (DeepSeek-V3, Kimi K2) must be dequantized to BF16 or re-quantized to AWQ/GPTQ-INT4.
- **Booster's driver is R535 (535.274.02) — CUDA 12.x generation only, no forward
  compat to CUDA 13.** The default `vllm/vllm-openai` Docker tags moved to a CUDA 13.0
  base (torch `+cu130`); that fails at engine init with `driver on your system is too
  old (found version 12020)`. Always build from a `-cu12x` tag (currently
  `v0.27.1-cu129`, set as the default in `scripts/build-container.sh`) — verified with
  a real `torch.cuda.init()` + matmul on `boost_qos_dbg` before trusting it in a job.
- **No node-local disk.** `/tmp` is a 10 GB tmpfs. Weights come off Lustre every start.
- **`$HOME` is a 50 GB quota** and writes fail *silently* when it is full. Every cache
  (`HF_HOME`, `TRITON_CACHE_DIR`, `VLLM_CACHE_ROOT`, Singularity's) is redirected to
  `$MR_STATE`. Keep it that way.
- **Login → compute works on any port** (verified, 0.1 ms). No SSH tunnels internally.
- **Login-node processes get `RLIMIT_CPU=600s` soft** (hard is unlimited). Anything
  long-running or CPU-heavy there must raise it or it dies at 10 CPU-minutes with a
  confusing error — this already killed `mksquashfs` during a container build. Raised in
  `scripts/build-container.sh`, `mr.supervisor.run`, `mr.gateway.run`. Add it to anything
  new you put on a login node.
- **`cin_staff` is a shared allocation.** A 2-node backend burns ~192 node-hours per 4-day
  job. Idle reaping is on by default — an idle model sitting on 8 A100s is antisocial even
  when the budget allows it.
