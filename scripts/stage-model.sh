#!/bin/bash
# Download model weights to the flash tier. LOGIN NODE ONLY (needs internet).
#
# Booster nodes have no local disk -- /tmp is a 10 GB tmpfs. Every job start
# re-reads the full checkpoint from Lustre, so striping is not an optimisation,
# it is the difference between a 5-minute and a 40-minute cold start.
set -euo pipefail

REPO="${1:?usage: stage-model.sh <hf_repo> <dest_dir>}"
DEST="${2:?usage: stage-model.sh <hf_repo> <dest_dir>}"

STATE_DIR="${MR_STATE:-${SCRATCH}/model-runner}"
export HF_HOME="${STATE_DIR}/hf"      # keep the HF cache off the 50 GB $HOME
mkdir -p "$HF_HOME" "$(dirname "$DEST")"

# Stripe BEFORE any file is created -- Lustre striping is set at creation time
# and cannot be changed afterwards without a rewrite.
if [[ ! -d "$DEST" ]]; then
  mkdir -p "$DEST"
  lfs setstripe -c 16 -S 4M "$DEST" 2>/dev/null \
    || echo "warning: lfs setstripe failed; cold starts will be slower"
fi
lfs getstripe -d "$DEST" 2>/dev/null || true

echo "downloading ${REPO} -> ${DEST}"
hf download "$REPO" --local-dir "$DEST" --max-workers 16

echo
du -sh "$DEST"
echo "ok. remember: no --kv-cache-dtype fp8 on A100, and native-FP8 checkpoints"
echo "    (DeepSeek-V3, Kimi K2) need conversion before they will run here."
