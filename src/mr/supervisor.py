"""Reconcile loop: keep the desired number of healthy backends alive.

Design rule: **stateless**. Every tick rebuilds the world from `squeue` plus the
registry and decides from scratch. No local state file, no event log. Killing and
restarting the supervisor is therefore always safe -- it is the main reason this
is a loop and not a daemon with an in-memory FSM.

Phase 3 in docs/architecture.md. `decide()` is pure and unit-testable: it only
reads what `refresh_activity()` (the one I/O step, scraping each backend's own
vLLM /metrics) has already written into the registry. `apply()` carries out
whatever `decide()` returned.
"""

from __future__ import annotations

import logging
import resource
import time
import urllib.error
import urllib.request
from dataclasses import dataclass

from . import config, demand, registry, slurm

log = logging.getLogger("mr.supervisor")

TICK_S = 60

# How long a DRAINING backend gets to finish in-flight requests before it's
# cancelled regardless. refresh_activity() usually beats this by cancelling
# the moment in-flight actually reaches zero; this is just the backstop for
# one that never quiets down. Matches the "5-minute grace" in architecture.md §4.
DRAIN_GRACE_S = 300


@dataclass
class Decision:
    action: str  # "none" | "launch" | "launch_successor" | "start_drain" | "cancel" | "reap_idle"
    model: str
    reason: str
    job_id: str = ""


@dataclass
class Activity:
    running: float
    waiting: float
    completed: float  # cumulative finished-request count, all reasons summed


def _poll_activity(url: str, timeout: float = 5.0) -> Activity | None:
    """Scrape one backend's own vLLM /metrics for request activity.

    Returns None on any failure to reach or parse it -- callers must treat
    that as "can't tell right now", not "idle": a network hiccup or a vLLM
    that's briefly slow to answer must never look like zero traffic.
    """
    try:
        with urllib.request.urlopen(f"{url}/metrics", timeout=timeout) as r:
            text = r.read().decode()
    except (OSError, urllib.error.URLError):
        return None
    running = waiting = completed = None
    for line in text.splitlines():
        if line.startswith("vllm:num_requests_running{"):
            running = (running or 0.0) + float(line.rsplit(" ", 1)[-1])
        elif line.startswith("vllm:num_requests_waiting{"):
            waiting = (waiting or 0.0) + float(line.rsplit(" ", 1)[-1])
        elif line.startswith("vllm:request_success_total{"):
            completed = (completed or 0.0) + float(line.rsplit(" ", 1)[-1])
    if running is None or waiting is None or completed is None:
        return None
    return Activity(running, waiting, completed)


def raise_cpu_limit() -> None:
    """Login nodes ship RLIMIT_CPU=600s soft / unlimited hard.

    A long-lived process is killed at 10 minutes of accumulated CPU without this.
    Raising the soft limit to the hard limit is permitted and needs no privilege.
    """
    _, hard = resource.getrlimit(resource.RLIMIT_CPU)
    resource.setrlimit(resource.RLIMIT_CPU, (hard, hard))


def decide(model: str, desired: int, now: float | None = None) -> list[Decision]:
    """Pure function: registry + Slurm state -> what to do. Unit-testable.

    No I/O here -- this only ever reads what refresh_activity() has already
    persisted into the registry on a previous call. Keeping the two separate
    is what makes this function testable without a live vLLM to poll.
    """
    now = now or time.time()
    m = config.load(model)
    live = {j.job_id for j in slurm.jobs() if j.name == f"mr-{model}"}
    backends = [b for b in registry.for_model(model, include_stale=True) if b.job_id in live]

    ready = [b for b in backends if b.state == registry.READY]
    draining = [b for b in backends if b.state == registry.DRAINING]
    # Anything Slurm still has for this model that isn't READY/DRAINING yet --
    # deliberately Slurm-sourced, not registry-sourced. A job has no registry
    # record at all until its own script starts running and self-publishes,
    # which can be minutes after `sbatch` returns on a busy queue. Using
    # registry state alone for "pending" missed exactly that gap: caught live,
    # it launched three duplicate jobs a tick apart before this fix, because
    # every tick saw "no backend, nothing pending" for a job that was in fact
    # sitting PENDING in squeue the whole time.
    pending_ids = live - {b.job_id for b in ready + draining}
    out: list[Decision] = []

    # Backstop for anything draining, independent of `desired`: refresh_activity()
    # cancels the moment a draining backend's in-flight count actually reaches
    # zero; this catches one that never quiets down. `draining_since` is always
    # set by apply()'s start_drain -- `or now` only guards a legacy/corrupt
    # record so it doesn't get force-cancelled the instant this code runs.
    out += [
        Decision("cancel", model, "drain grace expired", b.job_id)
        for b in draining
        if now - (b.draining_since or now) > DRAIN_GRACE_S
    ]

    if desired == 0:
        out += [Decision("start_drain", model, "desired=0", b.job_id) for b in ready]
        out += [Decision("cancel", model, "desired=0", job_id) for job_id in pending_ids]
        return out or [Decision("none", model, "steady state")]

    # Nothing at all and nothing coming -> cold start. A draining backend
    # doesn't count -- it's on its way out, not available to serve.
    if not ready and not pending_ids:
        out.append(Decision("launch", model, "no backend and none pending"))
        return out

    # Walltime handoff: a successor must be in flight before the incumbent's wall.
    # handoff_lead_s must exceed measured load time; see architecture.md §4/§6.
    for b in ready:
        if b.seconds_left(now) < m.handoff_lead_s and not pending_ids:
            out.append(Decision("launch_successor", model,
                                f"{int(b.seconds_left(now) / 60)}m of walltime left", b.job_id))

    # Successor is up: start draining the incumbent with the least walltime
    # left. Once apply() flips its registry state to DRAINING, it drops out
    # of `ready` on the next tick -- so this doesn't keep re-firing while a
    # drain already in progress.
    if len(ready) > desired:
        oldest = min(ready, key=lambda b: b.expires_at)
        out.append(Decision("start_drain", model, "successor is ready", oldest.job_id))

    # Idle reaping. GPU budget is finite; an idle 2-node backend still bills.
    # last_active_at comes from refresh_activity() polling vLLM's own
    # /metrics -- NOT from `heartbeat`, which refreshes every 30s purely as a
    # liveness signal regardless of whether anything is actually being
    # served, and would never let this fire if used here.
    for b in ready:
        idle_for = now - (b.last_active_at or b.ready_at or now)
        if m.idle_timeout_s and idle_for > m.idle_timeout_s:
            out.append(Decision("reap_idle", model,
                                f"idle {int(idle_for / 60)}m", b.job_id))

    return out or [Decision("none", model, "steady state")]


def refresh_activity(model: str) -> None:
    """I/O step: poll each READY/DRAINING backend's own vLLM /metrics and
    persist what it says into the registry. Run once per model per tick,
    before decide() -- keeps decide() itself free of network calls.
    """
    now = time.time()
    for b in registry.for_model(model):
        if b.state not in (registry.READY, registry.DRAINING):
            continue
        a = _poll_activity(b.url)
        if a is None:
            continue  # can't tell right now; leave existing timers alone

        if b.state == registry.DRAINING and a.running == 0 and a.waiting == 0:
            log.info("cancel %s (drain: in-flight reached zero)", b.job_id)
            slurm.cancel(b.job_id)
            continue

        is_active = a.running > 0 or a.waiting > 0 or a.completed != b.last_activity_count
        if is_active:
            registry.heartbeat_from_job(
                model, b.job_id, last_active_at=now, last_activity_count=a.completed
            )


def apply(d: Decision) -> None:
    if d.action == "start_drain":
        # Just a state flip -- registry.active() already prefers READY over
        # DRAINING, so this alone stops new requests from landing here; the
        # gateway needs no separate signal. The job's own cleanup trap
        # (vllm-server.sbatch) evicts the registry record when it actually
        # exits, so this never conflicts with the job's own heartbeat writes
        # (a bare heartbeat call re-publishes whatever state is already on
        # disk -- see registry.heartbeat_from_job).
        log.info("draining %s (%s)", d.job_id, d.reason)
        registry.heartbeat_from_job(
            d.model, d.job_id, state=registry.DRAINING, draining_since=time.time()
        )
    elif d.action == "reap_idle":
        # Clear demand so the *next* tick's cold-start check doesn't just
        # relaunch this immediately -- an idle-reap that gets undone a tick
        # later saves nothing (see mr.demand). wake() (from an incoming
        # request via the waker stub) sets it back to 1 on real demand.
        log.info("cancel %s (%s: %s)", d.job_id, d.action, d.reason)
        slurm.cancel(d.job_id)
        demand.set_desired(d.model, 0)
    elif d.action == "cancel":
        log.info("cancel %s (%s: %s)", d.job_id, d.action, d.reason)
        slurm.cancel(d.job_id)
    elif d.action in ("launch", "launch_successor"):
        from .cli import cmd_up
        log.info("launch for %s (%s)", d.model, d.reason)
        cmd_up(type("A", (), {"model": d.model, "force": True})())


def run(models: list[str]) -> None:
    """models: names to supervise. Runs until interrupted.

    Desired replica count is NOT fixed here -- it's read from mr.demand fresh
    every tick, because idle-reaping needs to actually lower it (see
    apply()'s reap_idle branch) for reaping to save anything at all.
    """
    raise_cpu_limit()
    config.ensure_dirs()
    log.info("supervising %s", models)
    while True:
        for model in models:
            try:
                refresh_activity(model)
            except Exception:
                log.exception("activity refresh failed for %s", model)
        for model in models:
            try:
                for d in decide(model, demand.get_desired(model)):
                    if d.action != "none":
                        apply(d)
            except Exception:
                log.exception("tick failed for %s", model)
        registry.sweep({j.job_id for j in slurm.jobs()})
        time.sleep(TICK_S)
