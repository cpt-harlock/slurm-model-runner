"""mrctl -- control plane CLI.

    mrctl models                list configured models
    mrctl up <model>            submit an inference job
    mrctl status [model]        show backends and their Slurm state
    mrctl down <model>          cancel all backends for a model
    mrctl stage <model>         download weights to $FAST (login node; needs internet)
"""

from __future__ import annotations

import argparse
import json
import shlex
import subprocess
import sys
import time

from . import config, demand, registry, slurm

JOB_PREFIX = "mr-"


def _write_spec(m: config.ModelSpec) -> str:
    config.ensure_dirs()
    path = config.SPEC_DIR / f"{m.name}.json"
    path.write_text(
        json.dumps(
            {
                "name": m.name,
                "served_name": m.served_name,
                "weights_dir": str(m.weights_dir),
                "sif": str(config.MR_SIF),
                "state_dir": str(config.MR_STATE),
                "engine": {
                    "tensor_parallel_size": m.engine.tensor_parallel_size,
                    "pipeline_parallel_size": m.engine.pipeline_parallel_size,
                    "max_model_len": m.engine.max_model_len,
                    "gpu_memory_utilization": m.engine.gpu_memory_utilization,
                    "extra_args_str": " ".join(shlex.quote(a) for a in m.engine.extra_args),
                },
            },
            indent=2,
        )
    )
    return str(path)


def cmd_up(args) -> int:
    m = config.load(args.model)

    if not config.MR_SIF.exists():
        print(f"container missing: {config.MR_SIF}\n  run scripts/build-container.sh", file=sys.stderr)
        return 1
    if not m.weights_dir.exists():
        print(f"weights missing: {m.weights_dir}\n  run `mrctl stage {m.name}`", file=sys.stderr)
        return 1
    if m.world_size != m.slurm.nodes * m.slurm.gpus_per_node:
        print(
            f"parallelism mismatch: TP*PP={m.world_size} but "
            f"{m.slurm.nodes} nodes x {m.slurm.gpus_per_node} GPUs="
            f"{m.slurm.nodes * m.slurm.gpus_per_node}",
            file=sys.stderr,
        )
        return 1

    existing = [b for b in registry.for_model(m.name) if b.state != registry.DRAINING]
    if existing and not args.force:
        print(f"{m.name} already has {len(existing)} backend(s); --force to add another")
        return 0

    # A prior idle-reap may have left desired=0 for this model (see
    # mr.demand). Bringing it up manually is exactly the demand signal that
    # should override that -- otherwise the supervisor's next tick sees
    # desired=0 and immediately drains what was just launched.
    demand.wake(m.name)

    spec_path = _write_spec(m)
    s = m.slurm
    job_id = slurm.submit(
        script=config.MR_ROOT / "slurm" / "vllm-server.sbatch",
        export={"MR_SPEC": spec_path, "MR_ROOT": str(config.MR_ROOT)},
        sbatch_args=[
            "-A", s.account, "-p", s.partition, f"--qos={s.qos}",
            "-N", str(s.nodes), "--ntasks-per-node=1",
            "-c", str(s.cpus_per_task), f"--gres=gpu:{s.gpus_per_node}",
            "-t", s.time, "-J", f"{JOB_PREFIX}{m.name}",
            "-o", str(config.LOG_DIR / f"{m.name}.%j.out"),
            "-e", str(config.LOG_DIR / f"{m.name}.%j.err"),
        ],
    )
    print(f"submitted {job_id} for {m.name} ({s.nodes} node(s), {s.qos}, {s.time})")
    eta = slurm.start_estimate(job_id)
    if eta:
        print(f"  slurm estimates start at {time.strftime('%H:%M', time.localtime(eta))}")
    print(f"  logs: {config.LOG_DIR}/{m.name}.{job_id}.out")
    return 0


def cmd_status(args) -> int:
    live = {j.job_id: j for j in slurm.jobs(JOB_PREFIX)}
    removed = registry.sweep(set(live))
    for b in removed:
        print(f"[swept stale record {b.model}/{b.job_id}]", file=sys.stderr)

    backends = registry.read_all(include_stale=True)
    if args.model:
        backends = [b for b in backends if b.model == args.model]
    if not backends and not live:
        print("no backends")
        return 0

    hdr = f"{'MODEL':<22} {'JOB':<10} {'STATE':<9} {'SLURM':<11} {'ENDPOINT':<24} {'WALL LEFT':>10}"
    print(hdr)
    print("-" * len(hdr))
    for b in sorted(backends, key=lambda x: (x.model, x.job_id)):
        j = live.get(b.job_id)
        left = b.seconds_left()
        wall = f"{int(left // 3600)}h{int(left % 3600 // 60):02d}m" if left else "-"
        state = b.state + ("!" if b.stale else "")
        print(
            f"{b.model:<22} {b.job_id:<10} {state:<9} "
            f"{(j.state if j else 'GONE'):<11} "
            f"{(b.url if b.port else '-'):<24} {wall:>10}"
        )
    # Jobs Slurm knows about that have not yet written a record (still QUEUED).
    for jid, j in live.items():
        if not any(b.job_id == jid for b in backends):
            print(f"{j.name.removeprefix(JOB_PREFIX):<22} {jid:<10} {'-':<9} {j.state:<11} "
                  f"{'-':<24} {'-':>10}   ({j.reason})")
    return 0


def cmd_down(args) -> int:
    # Operator explicitly wants this off -- without this, a supervisor
    # running alongside would just relaunch it on its next tick.
    demand.set_desired(args.model, 0)
    n = 0
    for b in registry.for_model(args.model, include_stale=True):
        slurm.cancel(b.job_id)
        registry.evict(b.model, b.job_id)
        print(f"cancelled {b.job_id}")
        n += 1
    for j in slurm.jobs(JOB_PREFIX):
        if j.name == f"{JOB_PREFIX}{args.model}":
            slurm.cancel(j.job_id)
            n += 1
    if not n:
        print(f"nothing running for {args.model}")
    return 0


def cmd_stage(args) -> int:
    m = config.load(args.model)
    script = config.MR_ROOT / "scripts" / "stage-model.sh"
    return subprocess.call([str(script), m.hf_repo, str(m.weights_dir)])


def cmd_gateway(args) -> int:
    """Start LiteLLM plus the registry->config sync loop (login node)."""
    import logging

    from . import gateway

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
    if args.sync_only:
        gateway.watch_only()
    else:
        gateway.run()
    return 0


def cmd_supervise(args) -> int:
    import logging

    from . import supervisor

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
    supervisor.run(args.models or config.available())
    return 0


def cmd_models(_args) -> int:
    for name in config.available():
        m = config.load(name)
        staged = "staged" if m.weights_dir.exists() else "not staged"
        print(f"{name:<22} {m.hf_repo:<40} TP{m.engine.tensor_parallel_size}"
              f"xPP{m.engine.pipeline_parallel_size}  {m.slurm.nodes}n  [{staged}]")
    return 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="mrctl", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    up = sub.add_parser("up", help="submit an inference job")
    up.add_argument("model")
    up.add_argument("--force", action="store_true", help="submit even if one is already up")
    up.set_defaults(fn=cmd_up)

    st = sub.add_parser("status", help="show backends")
    st.add_argument("model", nargs="?")
    st.set_defaults(fn=cmd_status)

    dn = sub.add_parser("down", help="cancel all backends for a model")
    dn.add_argument("model")
    dn.set_defaults(fn=cmd_down)

    sg = sub.add_parser("stage", help="download weights (login node only)")
    sg.add_argument("model")
    sg.set_defaults(fn=cmd_stage)

    gw = sub.add_parser("gateway", help="run the LiteLLM gateway + registry sync")
    gw.add_argument("--sync-only", action="store_true",
                    help="only keep the config current; do not launch litellm")
    gw.set_defaults(fn=cmd_gateway)

    sv = sub.add_parser("supervise", help="run the reconcile loop (phase 3)")
    sv.add_argument("models", nargs="*", help="default: every configured model")
    sv.set_defaults(fn=cmd_supervise)

    sub.add_parser("models", help="list configured models").set_defaults(fn=cmd_models)

    args = p.parse_args(argv)
    return args.fn(args)


if __name__ == "__main__":
    sys.exit(main())
