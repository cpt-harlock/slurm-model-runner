"""Typed wrapper over the Slurm CLI.

Deliberately shells out rather than binding libslurm: the CLI is stable, always
present on login nodes, and keeps this dependency-free.
"""

from __future__ import annotations

import subprocess
import time
from dataclasses import dataclass
from pathlib import Path

LIVE_STATES = {"PENDING", "CONFIGURING", "RUNNING", "COMPLETING", "SUSPENDED"}


@dataclass
class Job:
    job_id: str
    name: str
    state: str
    nodelist: str
    start_time: float  # epoch; for PENDING jobs this is Slurm's estimate
    end_time: float
    reason: str = ""

    @property
    def live(self) -> bool:
        return self.state in LIVE_STATES

    @property
    def running(self) -> bool:
        return self.state == "RUNNING"


def _run(cmd: list[str]) -> str:
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(f"{' '.join(cmd)} failed ({r.returncode}): {r.stderr.strip()}")
    return r.stdout


def _epoch(v: str) -> float:
    # squeue renders unknown/unset times as N/A, Unknown, or an ISO timestamp.
    if not v or v in ("N/A", "Unknown", "None"):
        return 0.0
    try:
        return time.mktime(time.strptime(v, "%Y-%m-%dT%H:%M:%S"))
    except ValueError:
        return 0.0


def jobs(name_prefix: str = "mr-", user: str | None = None) -> list[Job]:
    fmt = "%i|%j|%T|%N|%S|%e|%r"
    cmd = ["squeue", "-h", "-o", fmt]
    cmd += ["-u", user] if user else ["--me"]
    out = []
    for line in _run(cmd).splitlines():
        parts = line.split("|")
        if len(parts) < 7 or not parts[1].startswith(name_prefix):
            continue
        out.append(
            Job(
                job_id=parts[0],
                name=parts[1],
                state=parts[2],
                nodelist=parts[3],
                start_time=_epoch(parts[4]),
                end_time=_epoch(parts[5]),
                reason=parts[6],
            )
        )
    return out


def get(job_id: str) -> Job | None:
    return next((j for j in jobs() if j.job_id == job_id), None)


def submit(script: Path, export: dict[str, str], sbatch_args: list[str]) -> str:
    exports = ",".join(["ALL"] + [f"{k}={v}" for k, v in export.items()])
    cmd = ["sbatch", "--parsable", f"--export={exports}", *sbatch_args, str(script)]
    # --parsable gives "jobid" or "jobid;cluster"
    return _run(cmd).strip().split(";")[0]


def cancel(job_id: str) -> None:
    subprocess.run(["scancel", job_id], capture_output=True, text=True)


def start_estimate(job_id: str) -> float:
    """Slurm's predicted start for a pending job, or 0 if it won't say."""
    try:
        out = _run(["squeue", "-h", "-j", job_id, "-o", "%S"]).strip()
    except RuntimeError:
        return 0.0
    return _epoch(out)


def node_hostnames(nodelist: str) -> list[str]:
    if not nodelist or nodelist.startswith("("):
        return []
    return _run(["scontrol", "show", "hostnames", nodelist]).split()
