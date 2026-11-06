# Porting to CINECA Pitagora: status

Handoff note for whoever (human or agent) picks this up on Pitagora. The site
profile is `config/sites/pitagora.toml`. Select it with `MR_SITE=pitagora`.
The profile mechanism is described in the README ("porting") and in
`docs/architecture.md` §10.

## Verified (2026-11-06)

These come from a 1-GPU `srun` on Booster node r309c06s03 and from the
login node.

| Item | Value |
|---|---|
| GPU | H100 SXM, 95830 MiB (the 94 GB part), compute capability 9.0 |
| Driver | 580.178.04, so CUDA 13 runs here (it does not on Leonardo) |
| HCAs | `mlx5_0` .. `mlx5_5` |
| IB netdevs | `ibp63s0`, `ibp82s0`, `ibp209s0`, `ibp227s0`; only `ibp63s0` has IPv4 |
| Container runtime | Apptainer 1.4.5 (`singularity` is an alias) |
| Node-local `/tmp` | 10 GB NVMe, unused |
| Login nodes | have internet access (HF reachable) |

- **IB VLAN:** `ibp63s0.8005` is firewalled. NCCL and vLLM are pinned to `ibp63s0`.
- **Login to compute:** login nodes reach a compute node's ports while a job runs there, as on Leonardo. The gateway therefore proxies directly to the backend, with no tunnel.
- **Weights:** they go on `$SCRATCH/weights`. There is no `$FAST`. `$SCRATCH` has no quota but is purged.

## Decisions taken

- **Image:** `v0.32.0-cu129`, the same as Leonardo, so that only one thing changes at a time. CUDA-13 vLLM (v0.33+) is a later, separate step.
- **FlashInfer sampler:** the workaround (`VLLM_USE_FLASHINFER_SAMPLER=0`) is **not** carried over. It was an SM80 kernel-image bug. Re-test it here before adding it.
- **`NCCL_IB_HCA`:** pinned to `=mlx5_0,mlx5_1,mlx5_4,mlx5_5`, the four IB rails seen on the first multi-node run (`mlx5_2` is RoCE).
- **Kimi-K3:** `[sites.pitagora]` in its TOML sets 6 nodes and PP6. That gives about 61 GiB of weights per GPU and about 15 GiB KV headroom at 0.85.

## Done (2026-11-06)

- **Image** `$SCRATCH/model-runner/containers/vllm.sif`: vllm 0.32.0, torch 2.13.0+cu129, ray 2.60.0; `torch.cuda` matmul OK on an H100.
  `scripts/build-container.sh` does **not** work end to end here yet:
  - Login nodes have `max_user_namespaces=0` and no `starter-suid`, so `singularity exec` cannot run there at all. The `docker://` pull works; the `pip install ray` step does not.
  - What worked: pull straight to a SIF on the login node (`singularity build x.sif docker://vllm/vllm-openai:<tag>`, temp dir on `/scratch_local`); `pip download` the `ray[default]` wheels for cp312 on the login node; then, in a compute-node job, unpack the SIF to a sandbox on node-local disk (`--gres=tmpfs:200g`), `pip install --no-index` the wheels, remove `torchcodec`, repack. That job took 3.5 minutes.
  - Keep the sandbox off Lustre: copying a 25 GB sandbox back off `$SCRATCH` ran at about 0.25 GB/min.
  - Installing ray 2.60 upgrades opentelemetry to 1.45, which conflicts on paper with `lmcache` and the OTLP exporters. vLLM imports and serves fine.
- **Weights:** `mrctl stage qwen3-32b`, 61 GB in a few minutes. `hf` is not installed on Pitagora; it was installed with `pip install --user huggingface_hub`.
- **Single node:** `mrctl up qwen3-32b` (job 2791349) reached `ready`. Weights load in 14.6 s, KV cache is 591k tokens. A real completion and a real `auto` tool call both work.
- **FlashInfer sampler** works on H100 without the A100 workaround: the first sampled token did not crash.
- **Multi-node:** `mrctl up qwen3-coder-480b-awq` (job 2793087, 2 nodes, TP4 x PP2, `NCCL_DEBUG=INFO`) reached `ready`. Weights load in about 175 s, KV cache is 485k tokens. A real completion and a real `auto` tool call work, directly and through the gateway.
  - NCCL lists `mlx5_0`, `mlx5_1`, `mlx5_4`, `mlx5_5` as IB and `mlx5_2` as **RoCE**; `mlx5_3` does not appear. Every channel logged goes `via NET/IB/<n>/GDRDMA` on the IB devices, none on the RoCE one. Auto-detection was correct. The site `[env]` now pins `NCCL_IB_HCA="=mlx5_0,mlx5_1,mlx5_4,mlx5_5"` anyway, to rule RoCE out for good (from the next job on).
  - vLLM's usage-stats thread raises `OSError: Could not find a suitable TLS CA certificate bundle`, because the host's `REQUESTS_CA_BUNDLE` (a RHEL path) leaks into the Ubuntu container. It is harmless (compute nodes are offline anyway); the site `[env]` now sets `VLLM_NO_USAGE_STATS=1`.

## Control plane on the login node

Gateway and supervisor run in a dedicated tmux server, `tmux -L mr` (session `mr`, windows `gateway` and `supervise`), started by `$SCRATCH/model-runner/tools/start-control-plane.sh <models>`. It raises `ulimit -t` before tmux starts, because the tmux server inherits the 600 s soft CPU limit. System tmux 3.2a segfaulted twice (13:56 and 14:03 on 2026-11-06), both times with opencode's TUI open inside it, and both times while a 244 GB download kept the user memory cgroup (32 GiB) at its limit (`memory.events max` 7464, `oom` 0). A separate tmux server keeps the control plane alive when the user's tmux dies. Always name the models for `supervise`: with none, every model without a demand file defaults to `desired=1` and is launched.

## Bring-up jobs: `system` partition, `maint_hpcmd` reservation

The site profile submits to the hidden staff partition `system` on the `maint_hpcmd` reservation (4 Booster nodes, `r309c01s[01-04]`). The public partitions have partition-QoS group caps (`boost_fua_dbg` 2 nodes, `boost_fua_prod` 158) that left jobs on `QOSGrpNodeLimit` even inside the reservation. kimi-k3 (6 nodes) does not fit and stays on `boost_fua_prod`. Go back to `boost_fua_prod` with no reservation for production.

`squeue` hides jobs in hidden partitions unless it gets `-a`. `mr.slurm` now passes it. Without it, `mrctl status` deleted the record of a running backend.

## Not done yet, in order

1. **Make `build-container.sh` work here**: split it into a login-node part (pull, wheel download) and a compute-node part (install, repack), as described above.
2. **Before production:**
   - The walltime cap is 24 h (`boost_fua_prod`, QoS `normal`), so the handoff to a successor job happens every day, not rarely.
   - Fix first the handoff-path risks in `docs/study-guide.md` §7. These are R1 (every handoff launches one extra job), R13 (an empty EndTime gives a deadline in the past) and R18 (a late successor leaves a gap).
   - Also fix R4 first: nothing restarts the control plane (gateway and supervisor).
   - R12 was hit today: after `scancel`, the heartbeat recreated the evicted record, and `mrctl up` refused until `status` swept it. The `ready` publish writes only `state` and `ready_at`, so a lost `loading` record becomes a "ready" backend with port 0 and no deadline.
