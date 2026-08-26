#!/bin/bash
# vLLM inference server under Slurm on Leonardo Booster.
#
# Not submitted by hand -- `mrctl up <model>` renders $MR_SPEC and passes the
# sbatch resource flags on the command line. Everything variable lives in the
# spec so this file stays readable.
#
# Contract with the rest of the system: this script is the *only* writer of its
# own registry record, and it must keep the heartbeat fresh or the supervisor
# will treat it as dead (registry.STALE_AFTER_S).

set -euo pipefail

: "${MR_SPEC:?MR_SPEC not exported}"
: "${MR_ROOT:?MR_ROOT not exported}"

export PYTHONPATH="${MR_ROOT}/src:${PYTHONPATH:-}"

spec() { python3 -c "import json,sys;print(json.load(open('${MR_SPEC}'))${1})"; }

MODEL=$(spec "['name']")
SERVED=$(spec "['served_name']")
WEIGHTS=$(spec "['weights_dir']")
SIF=$(spec "['sif']")
TP=$(spec "['engine']['tensor_parallel_size']")
PP=$(spec "['engine']['pipeline_parallel_size']")
MAXLEN=$(spec "['engine']['max_model_len']")
GPUUTIL=$(spec "['engine']['gpu_memory_utilization']")
EXTRA=$(spec "['engine']['extra_args_str']")
STATE_DIR=$(spec "['state_dir']")

JOB_ID="${SLURM_JOB_ID}"
HEAD_NODE=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -1)
NODES=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | tr '\n' ' ')
# Deterministic-but-collision-unlikely port; several jobs can share a node set
# during a handoff overlap.
PORT=$(( 8000 + SLURM_JOB_ID % 1000 ))

# --- caches OFF $HOME -------------------------------------------------------
# $HOME is a 50 GB quota and writes fail silently once it is full. Every cache
# any part of this stack might touch gets redirected.
export HF_HOME="${STATE_DIR}/hf"
export HF_HUB_OFFLINE=1            # compute nodes have no internet, fail loudly
export XDG_CACHE_HOME="${STATE_DIR}/cache"
export TRITON_CACHE_DIR="${STATE_DIR}/cache/triton"
export VLLM_CACHE_ROOT="${STATE_DIR}/cache/vllm"
export OUTLINES_CACHE_DIR="${STATE_DIR}/cache/outlines"
mkdir -p "$HF_HOME" "$XDG_CACHE_HOME" "$TRITON_CACHE_DIR" "$VLLM_CACHE_ROOT"

# --- NCCL over Leonardo's dual-rail HDR100 ----------------------------------
# mlx5_0 is Down on these nodes; mlx5_bond_0 is the 25G ethernet, not IB.
# Pinning explicitly avoids NCCL picking the slow path or hanging on the dead HCA.
export NCCL_IB_HCA=mlx5_1,mlx5_4
export NCCL_SOCKET_IFNAME=ib0
export NCCL_IB_DISABLE=0
export NCCL_DEBUG=${NCCL_DEBUG:-WARN}
# NOTE: VLLM_HOST_IP is deliberately NOT set here. It must be each node's own
# IP, resolved per-node -- srun propagates the parent's exported environment
# to every task, so setting it once on the head node would tell every worker
# to advertise the *head's* IP as its own. Set inline for the single-node
# case below; resolved fresh inside the per-node worker script for multi-node.

# --- registry helper --------------------------------------------------------
publish() {
  python3 - "$@" <<'PY'
import os, sys, time
from mr import registry
kw = dict(a.split("=", 1) for a in sys.argv[1:])
for k in ("port", ):
    if k in kw: kw[k] = int(kw[k])
for k in ("started_at", "ready_at", "expires_at"):
    if k in kw: kw[k] = float(kw[k])
if "nodes" in kw: kw["nodes"] = kw["nodes"].split()
registry.heartbeat_from_job(os.environ["MR_MODEL"], os.environ["SLURM_JOB_ID"], **kw)
PY
}
export MR_MODEL="$MODEL"

# EndTime is authoritative for the handoff deadline -- it accounts for whatever
# the scheduler actually granted, not what we asked for.
EXPIRES=$(date -d "$(scontrol show job "$JOB_ID" | grep -oP 'EndTime=\K\S+')" +%s)

cleanup() {
  publish "state=draining" || true
  sleep 2
  python3 -c "
import os
from mr import registry
registry.evict(os.environ['MR_MODEL'], os.environ['SLURM_JOB_ID'])" || true
}
trap cleanup EXIT TERM

publish "state=loading" "served_name=$SERVED" "host=$HEAD_NODE" "port=$PORT" \
        "nodes=$NODES" "started_at=$(date +%s)" "expires_at=$EXPIRES"

# --- heartbeat --------------------------------------------------------------
# NOTE: call publish with *no* args, not publish "". An empty-string arg makes
# it into sys.argv, "".split("=", 1) == [""], and dict() over a 1-element
# sequence raises -- silently swallowed by `|| true`, so the heartbeat looked
# like it was running but never once updated the registry after the initial
# ready publish. Caught via `mrctl status` showing `ready!` (stale) on job
# 54085509 within 2 minutes of READY.
( while true; do sleep 30; publish 2>/dev/null || true; done ) &
HEARTBEAT=$!

# --- launch -----------------------------------------------------------------
SING="singularity exec --nv \
  --bind ${WEIGHTS}:${WEIGHTS}:ro \
  --bind ${STATE_DIR}:${STATE_DIR} \
  --bind ${MR_ROOT}:${MR_ROOT}:ro \
  ${SIF}"

VLLM_ARGS="--model ${WEIGHTS} \
  --served-model-name ${SERVED} \
  --host 0.0.0.0 --port ${PORT} \
  --tensor-parallel-size ${TP} \
  --pipeline-parallel-size ${PP} \
  --max-model-len ${MAXLEN} \
  --gpu-memory-utilization ${GPUUTIL} \
  ${EXTRA}"

if [[ "${SLURM_JOB_NUM_NODES}" -eq 1 ]]; then
  export VLLM_HOST_IP="$(hostname -i | awk '{print $1}')"
  $SING vllm serve $VLLM_ARGS &
else
  # Phase 4. `ray symmetric-run` launches the identical entrypoint on every node
  # and handles head/worker election itself -- no manual `ray start --head` dance.
  #
  # --distributed-executor-backend ray: without it vLLM assumes the whole
  # world size must fit on one node's GPUs and refuses to start --
  # "World size (16) is larger than the number of available GPUs (4) in this
  # node" -- even though a real multi-node Ray cluster was already up. vLLM
  # names this exact fix in its own error message.
  #
  # A real per-node script file, not an inline srun command: VLLM_HOST_IP has
  # to be resolved fresh on whichever node actually runs it (each rank
  # advertises its own IP), which an inline `srun ... $SING ray ...` can't do
  # cleanly without nested-quoting hell across srun/bash/singularity/ray/vllm.
  # --min-nodes is not optional in practice: per `ray symmetric-run --help`,
  # without it there's no guarantee the entrypoint waits for every node to
  # join before running, and vLLM's distributed init needs the full world
  # size present from the start.
  WORKER_SCRIPT="${STATE_DIR}/specs/${MODEL}.${JOB_ID}.worker.sh"
  cat > "$WORKER_SCRIPT" <<WORKEREOF
#!/bin/bash
export VLLM_HOST_IP="\$(hostname -i | awk '{print \$1}')"
exec $SING ray symmetric-run --address "${HEAD_NODE}:6379" --min-nodes ${SLURM_JOB_NUM_NODES} -- vllm serve $VLLM_ARGS --distributed-executor-backend ray
WORKEREOF
  chmod +x "$WORKER_SCRIPT"
  srun --ntasks-per-node=1 "$WORKER_SCRIPT" &
fi
SERVER=$!

# --- readiness --------------------------------------------------------------
# /health alone is not enough: it can pass while the model is still loading
# weights. A real completion is the only honest readiness signal.
for _ in $(seq 1 240); do   # 240 * 15s = 60 min ceiling for Lustre weight load
  if curl -sf "http://127.0.0.1:${PORT}/v1/models" >/dev/null 2>&1; then
    publish "state=ready" "ready_at=$(date +%s)"
    echo "READY after $SECONDS s"    # <-- Phase 0: this is the load time to measure
    break
  fi
  kill -0 $SERVER 2>/dev/null || { publish "state=failed" "error=engine exited during load"; exit 1; }
  sleep 15
done

wait $SERVER
kill $HEARTBEAT 2>/dev/null || true
