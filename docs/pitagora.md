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
- **`NCCL_IB_HCA`:** left to auto-detection. There are six mlx5 devices for four IB rails, and which two are not IB is not known yet.
- **Kimi-K3:** `[sites.pitagora]` in its TOML sets 6 nodes and PP6. That gives about 61 GiB of weights per GPU and about 15 GiB KV headroom at 0.85.

## Not done yet, in order

1. **Build the image** on a login node:
   ```
   MR_SITE=pitagora ./scripts/build-container.sh
   ```
   This produces `$SCRATCH/model-runner/containers/vllm.sif`.
2. **Stage weights:** `mrctl stage qwen3-32b`.
3. **Single-node bring-up:** `mrctl up qwen3-32b`, then a real completion and a real tool call.
4. **First multi-node bring-up** with `NCCL_DEBUG=INFO`:
   - Read the `NET/IB` line and pin `NCCL_IB_HCA` if NCCL picks a non-IB device.
   - `ibv_devinfo | grep -E 'hca_id|link_layer'` tells which mlx5 are IB.
5. **Before production:**
   - The walltime cap is 24 h (`boost_fua_prod`, QoS `normal`), so the handoff to a successor job happens every day, not rarely.
   - Fix first the handoff-path risks in `docs/study-guide.md` §7. These are R1 (every handoff launches one extra job), R13 (an empty EndTime gives a deadline in the past) and R18 (a late successor leaves a gap).
   - Also fix R4 first: nothing restarts the control plane (gateway and supervisor).
