"""Consecutive bring-up failures per model: the supervisor's crash backoff.

Without this, a model that can never come up (bad config, a NCCL/Ray hang,
a broken node in every allocation) is relaunched forever: the job fails,
its cleanup trap evicts the registry record, the next supervisor tick sees
"no backend and none pending" and launches again -- each round holding up
to 8 nodes for up to ready_timeout_s. (docs/study-guide.md R5.)

A tiny persisted record per model, atomic tmp+rename like mr.demand:
  - the job script records a failure when the engine dies during load or
    misses its readiness deadline (R6), with the error and job id;
  - it resets the count the moment the model is READY;
  - `mrctl up` (a human) resets it too.
At MAX_CONSECUTIVE the supervisor stops launching that model.

Deliberately separate from mr.demand: the gateway's waker sets desired=1
on every incoming request, so a backoff stored as desired=0 would be
undone by the next client retry and the crash loop would resume.
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from pathlib import Path

from . import config

FAILURES_DIR = config.MR_STATE / "failures"
MAX_CONSECUTIVE = 3


@dataclass
class Failures:
    consecutive: int = 0
    last_error: str = ""
    last_job: str = ""
    at: float = 0.0

    @property
    def blocked(self) -> bool:
        return self.consecutive >= MAX_CONSECUTIVE


def _path(model: str) -> Path:
    return FAILURES_DIR / f"{model}.json"


def get(model: str) -> Failures:
    try:
        raw = json.loads(_path(model).read_text())
        return Failures(**{k: raw[k] for k in ("consecutive", "last_error", "last_job", "at") if k in raw})
    except (FileNotFoundError, json.JSONDecodeError, TypeError):
        return Failures()


def _write(model: str, f: Failures) -> None:
    FAILURES_DIR.mkdir(parents=True, exist_ok=True)
    dest = _path(model)
    tmp = dest.with_suffix(f".tmp.{os.getpid()}")
    tmp.write_text(json.dumps(f.__dict__))
    os.replace(tmp, dest)


def record(model: str, job_id: str, error: str) -> Failures:
    f = get(model)
    f = Failures(consecutive=f.consecutive + 1, last_error=error, last_job=job_id, at=time.time())
    _write(model, f)
    return f


def reset(model: str) -> None:
    try:
        _path(model).unlink()
    except FileNotFoundError:
        pass
