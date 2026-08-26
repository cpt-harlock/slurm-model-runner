"""File-based service discovery on Lustre.

Each running job owns exactly one JSON file and rewrites it as it advances through
its lifecycle. The gateway and supervisor only ever read.

Why files and not an HTTP callback to the login node: login->compute is verified
working, but compute->login is not, and this design does not need to find out.
Lustre is visible from everywhere by construction.

Writes are atomic (tmp + os.replace within the same directory) so a reader never
sees a partial record.
"""

from __future__ import annotations

import json
import os
import socket
import time
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path

from . import config

# Lifecycle states. See docs/architecture.md §4.
QUEUED = "queued"
LOADING = "loading"
READY = "ready"
DRAINING = "draining"
FAILED = "failed"

# A record whose heartbeat is older than this is considered dead, regardless of
# what it claims -- covers jobs killed hard enough to skip their cleanup trap.
STALE_AFTER_S = 90


@dataclass
class Backend:
    model: str
    served_name: str
    job_id: str
    state: str
    host: str = ""
    port: int = 0
    nodes: list[str] = field(default_factory=list)
    started_at: float = 0.0
    ready_at: float = 0.0
    expires_at: float = 0.0
    heartbeat: float = 0.0
    error: str = ""
    # Set once by the supervisor when it transitions READY -> DRAINING (never
    # by the job itself). Backstops the grace timer if in-flight requests
    # never reach zero -- see mr.supervisor.DRAIN_GRACE_S.
    draining_since: float = 0.0
    # Maintained by mr.supervisor.refresh_activity() from vLLM's own /metrics,
    # not from `heartbeat` -- heartbeat refreshes every 30s purely as a
    # liveness signal and would never let idle reaping fire if used for this.
    last_active_at: float = 0.0
    last_activity_count: float = 0.0

    @property
    def url(self) -> str:
        return f"http://{self.host}:{self.port}"

    @property
    def stale(self) -> bool:
        return time.time() - self.heartbeat > STALE_AFTER_S

    @property
    def serving(self) -> bool:
        return self.state in (READY, DRAINING) and not self.stale

    def seconds_left(self, now: float | None = None) -> float:
        return max(0.0, self.expires_at - (now if now is not None else time.time()))


def _path(model: str, job_id: str) -> Path:
    return config.REGISTRY_DIR / f"{model}.{job_id}.json"


def publish(b: Backend) -> None:
    config.REGISTRY_DIR.mkdir(parents=True, exist_ok=True)
    b.heartbeat = time.time()
    dest = _path(b.model, b.job_id)
    tmp = dest.with_suffix(f".tmp.{os.getpid()}")
    tmp.write_text(json.dumps(asdict(b), indent=2))
    os.replace(tmp, dest)


_BACKEND_FIELDS = {f.name for f in fields(Backend)}


def read_all(include_stale: bool = False) -> list[Backend]:
    out = []
    for p in sorted(config.REGISTRY_DIR.glob("*.json")):
        try:
            raw = json.loads(p.read_text())
        except (json.JSONDecodeError, OSError):
            continue  # torn write; the owner will rewrite it
        try:
            # Forward-compat: a long-lived reader (gateway, supervisor) can be
            # running older code than whatever last wrote this record -- a
            # compute job's heartbeat calls always run whatever's currently on
            # disk, so a mid-flight schema change shows up in its writes
            # immediately while a long-running process is still on the old
            # Backend class. Unknown keys are dropped rather than failing the
            # whole record -- constructing with **raw here made an old reader
            # raise TypeError on every record a newer writer touched, which
            # this exact except swallowed silently: a live gateway process
            # showed "no backends" against a perfectly healthy job for ~34
            # minutes before this was diagnosed (architecture.md §9 risk 15).
            b = Backend(**{k: v for k, v in raw.items() if k in _BACKEND_FIELDS})
        except TypeError:
            continue  # missing a field with no default; genuinely malformed
        if include_stale or not b.stale:
            out.append(b)
    return out


def for_model(model: str, include_stale: bool = False) -> list[Backend]:
    return [b for b in read_all(include_stale) if b.model == model]


def active(model: str) -> Backend | None:
    """The backend the gateway should send new requests to.

    Prefers READY over DRAINING, and among equals the one with the most walltime
    left -- during a handoff that is the successor, which is exactly the cutover.
    """
    candidates = [b for b in for_model(model) if b.state == READY]
    if not candidates:
        candidates = [b for b in for_model(model) if b.state == DRAINING]
    return max(candidates, key=lambda b: b.expires_at, default=None)


def evict(model: str, job_id: str) -> None:
    _path(model, job_id).unlink(missing_ok=True)


def sweep(live_job_ids: set[str]) -> list[Backend]:
    """Drop records whose Slurm job no longer exists. Returns what was removed."""
    removed = []
    for b in read_all(include_stale=True):
        if b.job_id not in live_job_ids:
            evict(b.model, b.job_id)
            removed.append(b)
    return removed


def heartbeat_from_job(model: str, job_id: str, **updates) -> Backend:
    """Called from inside the sbatch script to advance/refresh this job's record."""
    existing = {b.job_id: b for b in for_model(model, include_stale=True)}
    b = existing.get(job_id) or Backend(
        model=model,
        served_name=updates.pop("served_name", model),
        job_id=job_id,
        state=QUEUED,
        host=socket.gethostname(),
    )
    for k, v in updates.items():
        setattr(b, k, v)
    publish(b)
    return b
