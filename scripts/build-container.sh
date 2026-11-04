#!/bin/bash
# Build the vLLM Singularity image. LOGIN NODE ONLY -- compute nodes have no internet.
#
# Leonardo has no site vLLM/PyTorch module, so the container is the whole runtime.
set -euo pipefail

# Login nodes ship RLIMIT_CPU=600s soft (hard is unlimited). mksquashfs on a
# ~10 GB vLLM image burns well past that and dies with
#   "while creating squashfs: signal: CPU time limit exceeded (core dumped)"
# -- which looks like a Singularity bug and is not. Raise it first thing.
ulimit -t unlimited

# Leonardo's Booster driver is R535 (535.274.02) -- CUDA 12.x generation only.
# The plain "latest"/"vX.Y.Z" tags default to a CUDA 13.0 build (needs driver
# >=580) and fail at engine init with "driver on your system is too old
# (found version 12020)". The "-cu129" variant is still CUDA 12.9, which
# NVIDIA's own image label (NVIDIA_REQUIRE_CUDA) certifies for driver
# >=535,<536 via minor-version compatibility -- i.e. it runs natively on
# what we actually have. Always pick a -cu12x tag here, never the bare one.
#
# v0.28.0-cu129 (bumped from v0.27.1-cu129): tried specifically to see if it
# fixed Kimi-K2-Thinking's broken tool-call/reasoning-parser output
# (architecture.md §9 risk 25) -- it didn't (identical garbled tool-call
# tokens, identical content:null on plain completions), but real-GPU sanity
# (debug QoS, torch.cuda matmul) and full regression tests (real completion +
# real tool-call request) against both qwen3-32b and qwen3-coder-480b-awq
# passed cleanly, so it's kept as the new baseline anyway (risk 26).
#
# v0.32.0-cu129 (2026-11-03): the LAST CUDA-12 image vLLM will publish --
# v0.33.0 dropped cu129 images and wheels entirely (#62575, CUDA 13.0 only),
# so anything newer needs a driver >=580 on Booster or a from-source build.
# Wanted for the sparse-attention backports (v0.29 #52861: DeepSeek-V3.2 /
# GLM-5.x DSA on the CUDA path on all GPUs, i.e. A100 too). Built to its own
# file so models opt in via `sif = "vllm-0.32.sif"` in their TOML, and a bare
# re-run never overwrites vllm.sif, which the proven models still run on.
TAG="${1:-v0.32.0-cu129}"
STATE_DIR="${MR_STATE:-${SCRATCH}/model-runner}"
DEST="${2:-${STATE_DIR}/containers/vllm-0.32.sif}"

mkdir -p "$(dirname "$DEST")"

# Singularity's own build cache defaults under $HOME, which is a 50 GB quota.
export SINGULARITY_CACHEDIR="${STATE_DIR}/cache/singularity"
export SINGULARITY_TMPDIR="${TMPDIR:-/tmp}/singularity-$USER"
mkdir -p "$SINGULARITY_CACHEDIR" "$SINGULARITY_TMPDIR"

# The upstream vllm-openai image does not bundle `ray` at all (not on PATH,
# not importable) -- confirmed empirically, not documented anywhere obvious.
# Phase 4's multi-node path (vllm-server.sbatch) needs `ray symmetric-run`,
# so this can no longer be a bare `docker://` pull. `ray[default]` matches
# what vLLM's own multi-node docs use; vLLM itself declares no hard version
# pin on it (lazy/optional import).
#
# Can't do this via a .def file's %post: that needs --fakeroot, --remote, or
# proot, and --fakeroot is unavailable here ("no valid mapping entry found
# for amonteru" -- no subuid/subgid range configured for this user, a site
# setting, not something to work around). The unprivileged path is a
# writable sandbox: build one the same way the bare docker:// pull always
# worked, pip-install directly into it, then repack to an immutable .sif.
SANDBOX="${STATE_DIR}/containers/vllm-sandbox"
rm -rf "$SANDBOX"
echo "building sandbox from docker://vllm/vllm-openai:${TAG}"
singularity build --sandbox "$SANDBOX" "docker://vllm/vllm-openai:${TAG}"

# A writable sandbox exec can't auto-create missing bind targets the way a
# normal overlay-backed container run does -- confirmed empirically: without
# this, `singularity exec --writable` on this site fails with "can't create
# /leonardo destination automatically without overlay or underlay". Bind
# targets must exist for real inside the sandbox first.
mkdir -p "$SANDBOX"/{leonardo,leonardo_scratch,leonardo_work,leonardo_prod}

echo "installing ray[default] into the sandbox"
# No -C/--containall: it swaps in a small tmpfs-backed ephemeral home/tmp
# instead of the real (Lustre-backed) ones, and pip's in-progress download
# temp file overflowed it -- "No space left on device" despite $SCRATCH
# having petabytes free. It was only ever there to dodge the missing bind
# targets, which are now pre-created above, so a plain --writable exec
# (using the real host home/tmp) works and has real room to work with.
#
# Singularity forwards the host environment, and on this account PIP_CERT /
# REQUESTS_CA_BUNDLE / CURL_CA_BUNDLE point at the RHEL path
# /etc/pki/tls/certs/ca-bundle.crt, which does not exist in the Ubuntu image:
# pip then dies with "Could not find a suitable TLS CA certificate bundle"
# (hit on the v0.32.0-cu129 build, 2026-11-03). Drop them for this one exec so
# pip uses the image's own CA store.
env -u PIP_CERT -u REQUESTS_CA_BUNDLE -u CURL_CA_BUNDLE -u SSL_CERT_FILE \
  singularity exec --writable "$SANDBOX" pip install --no-cache-dir "ray[default]"

# v0.32.0-cu129 ships a torchcodec built against CUDA 13 (dlopens
# libnvrtc.so.13, absent from a cu129 image). vLLM imports it unconditionally
# via vllm.multimodal.media.audio and only catches ImportError/RuntimeError,
# but the failure is an OSError -- so *every* `vllm serve` dies at import,
# confirmed on a real A100 node (debug job 63458694). We never decode audio;
# removing it turns the import into a caught ImportError. No-op on older tags.
echo "removing torchcodec (CUDA-13 build in a CUDA-12 image)"
singularity exec --writable "$SANDBOX" pip uninstall -y torchcodec || true

echo "repacking sandbox -> ${DEST}"
singularity build --force "$DEST" "$SANDBOX"
rm -rf "$SANDBOX"

echo
echo "sanity check:"
singularity exec "$DEST" python3 -c "import vllm, torch, ray; print('vllm', vllm.__version__, '| torch', torch.__version__, '| ray', ray.__version__)"
singularity exec "$DEST" ray --version
echo "ok -> $DEST"
