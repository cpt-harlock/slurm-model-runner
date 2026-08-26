"""Per-model demand: should the supervisor keep this model up right now?

Idle reaping (mr.supervisor) only saves anything if the backend it cancels
actually *stays down* until someone wants it again. Without this, the
supervisor's `desired` was a hardcoded 1 forever, so every idle-reap was
immediately followed by a relaunch a tick later -- pure waste: a full
Lustre weight-reload every idle_timeout_s, with the GPU allocation barely
ever actually free. Caught live: 22 reap/relaunch cycles overnight, zero of
them saving anything.

Demand is a tiny persisted flag per model, atomic tmp+rename like the
registry:
  - reap_idle clears it (desired -> 0): stay down.
  - wake() sets it (desired -> 1): the supervisor's next tick (<= TICK_S
    later) sees desired=1, ready=[], pending=[] and cold-starts it.
  - A model with no demand file yet defaults to desired=1 -- preserves the
    original "just keep it up" behavior until the first idle-reap ever
    happens to it.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

from . import config

DEMAND_DIR = config.MR_STATE / "demand"


def _path(model: str) -> Path:
    return DEMAND_DIR / f"{model}.json"


def get_desired(model: str) -> int:
    try:
        return int(json.loads(_path(model).read_text())["desired"])
    except (FileNotFoundError, json.JSONDecodeError, KeyError, ValueError):
        return 1  # no signal yet -- default to "keep it up"


def set_desired(model: str, desired: int) -> None:
    DEMAND_DIR.mkdir(parents=True, exist_ok=True)
    dest = _path(model)
    tmp = dest.with_suffix(f".tmp.{os.getpid()}")
    tmp.write_text(json.dumps({"desired": desired}))
    os.replace(tmp, dest)


def wake(model: str) -> None:
    set_desired(model, 1)
