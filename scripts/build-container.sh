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
TAG="${1:-v0.27.1-cu129}"
STATE_DIR="${MR_STATE:-${SCRATCH}/model-runner}"
DEST="${MR_SIF:-${STATE_DIR}/containers/vllm.sif}"

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
singularity exec -C --writable "$SANDBOX" pip install --no-cache-dir "ray[default]"

echo "repacking sandbox -> ${DEST}"
singularity build --force "$DEST" "$SANDBOX"
rm -rf "$SANDBOX"

echo
echo "sanity check:"
singularity exec "$DEST" python3 -c "import vllm, torch, ray; print('vllm', vllm.__version__, '| torch', torch.__version__, '| ray', ray.__version__)"
singularity exec "$DEST" ray --version
echo "ok -> $DEST"
