"""Unit tests for mr.supervisor.decide() -- the pure state-machine function.

No live Slurm or vLLM needed: config.load, slurm.jobs, and registry.for_model
are patched per test. This is exactly the point of keeping decide() free of
I/O (see its docstring) -- run with:

    PYTHONPATH=src python3 -m unittest tests.test_supervisor -v
"""

from __future__ import annotations

import unittest
from unittest.mock import patch

from mr import config, failures, registry, slurm, supervisor

MODEL = "qwen3-32b"
JOB_NAME = f"mr-{MODEL}"


def make_spec(handoff_lead_s: int = 3000, idle_timeout_s: int = 1800) -> config.ModelSpec:
    return config.ModelSpec(
        name=MODEL,
        hf_repo="Qwen/Qwen3-32B",
        slurm=config.SlurmSpec(**config.SITE.slurm),
        engine=config.EngineSpec(),
        handoff_lead_s=handoff_lead_s,
        idle_timeout_s=idle_timeout_s,
    )


def make_job(job_id: str, state: str = "RUNNING") -> slurm.Job:
    return slurm.Job(job_id=job_id, name=JOB_NAME, state=state, nodelist="lrdn0001",
                      start_time=0.0, end_time=0.0)


def make_backend(job_id: str, state: str, *, now: float, ready_at: float = 0.0,
                  expires_at: float = 0.0, draining_since: float = 0.0,
                  last_active_at: float = 0.0) -> registry.Backend:
    return registry.Backend(
        model=MODEL, served_name=MODEL, job_id=job_id, state=state,
        host="lrdn0001", port=8000, ready_at=ready_at, expires_at=expires_at,
        heartbeat=now, draining_since=draining_since, last_active_at=last_active_at,
    )


class DecideTest(unittest.TestCase):
    def setUp(self) -> None:
        self.now = 1_000_000.0
        self._spec = make_spec()
        patcher = patch.object(supervisor.config, "load", return_value=self._spec)
        self.addCleanup(patcher.stop)
        patcher.start()
        # Never read the live failures dir from a unit test.
        self.failures = failures.Failures()
        fp = patch.object(supervisor.failures, "get", side_effect=lambda _m: self.failures)
        self.addCleanup(fp.stop)
        fp.start()

    def decide(self, backends: list[registry.Backend], jobs: list[slurm.Job], desired: int = 1):
        with patch.object(supervisor.slurm, "jobs", return_value=jobs), \
             patch.object(supervisor.registry, "for_model", return_value=backends):
            return supervisor.decide(MODEL, desired, now=self.now)

    def test_cold_start_launches(self):
        out = self.decide(backends=[], jobs=[])
        self.assertEqual([d.action for d in out], ["launch"])

    def test_steady_state_does_nothing(self):
        b = make_backend("1", registry.READY, now=self.now,
                          ready_at=self.now - 100, expires_at=self.now + 100_000,
                          last_active_at=self.now - 5)
        out = self.decide([b], [make_job("1")])
        self.assertEqual([d.action for d in out], ["none"])

    def test_low_walltime_triggers_successor(self):
        b = make_backend("1", registry.READY, now=self.now,
                          ready_at=self.now - 100,
                          expires_at=self.now + self._spec.handoff_lead_s - 10,
                          last_active_at=self.now)
        out = self.decide([b], [make_job("1")])
        self.assertEqual([d.action for d in out], ["launch_successor"])
        self.assertEqual(out[0].job_id, "1")

    def test_pending_successor_suppresses_duplicate_launch(self):
        incumbent = make_backend("1", registry.READY, now=self.now,
                                  expires_at=self.now + self._spec.handoff_lead_s - 10,
                                  last_active_at=self.now)
        successor = make_backend("2", registry.LOADING, now=self.now)
        out = self.decide([incumbent, successor], [make_job("1"), make_job("2")])
        self.assertEqual([d.action for d in out], ["none"])

    def test_successor_ready_drains_oldest_incumbent(self):
        older = make_backend("1", registry.READY, now=self.now,
                              expires_at=self.now + 1000, last_active_at=self.now)
        newer = make_backend("2", registry.READY, now=self.now,
                              expires_at=self.now + 90_000, last_active_at=self.now)
        out = self.decide([older, newer], [make_job("1"), make_job("2")])
        actions = {d.action for d in out}
        self.assertIn("start_drain", actions)
        drain = next(d for d in out if d.action == "start_drain")
        self.assertEqual(drain.job_id, "1")

    def test_draining_within_grace_is_left_alone(self):
        b = make_backend("1", registry.DRAINING, now=self.now,
                          draining_since=self.now - 10)
        out = self.decide([b], [make_job("1")])
        self.assertNotIn("cancel", {d.action for d in out})

    def test_draining_past_grace_is_cancelled(self):
        b = make_backend("1", registry.DRAINING, now=self.now,
                          draining_since=self.now - supervisor.DRAIN_GRACE_S - 1)
        out = self.decide([b], [make_job("1")])
        cancels = [d for d in out if d.action == "cancel"]
        self.assertEqual(len(cancels), 1)
        self.assertEqual(cancels[0].job_id, "1")

    def test_idle_backend_is_reaped(self):
        b = make_backend("1", registry.READY, now=self.now,
                          ready_at=self.now - 100_000,
                          expires_at=self.now + 100_000,
                          last_active_at=self.now - self._spec.idle_timeout_s - 1)
        out = self.decide([b], [make_job("1")])
        self.assertEqual([d.action for d in out], ["reap_idle"])

    def test_recently_active_backend_is_not_reaped(self):
        b = make_backend("1", registry.READY, now=self.now,
                          ready_at=self.now - 100_000,
                          expires_at=self.now + 100_000,
                          last_active_at=self.now - 5)
        out = self.decide([b], [make_job("1")])
        self.assertEqual([d.action for d in out], ["none"])

    def test_desired_zero_drains_ready_and_cancels_pending(self):
        ready = make_backend("1", registry.READY, now=self.now, expires_at=self.now + 100_000)
        pending = make_backend("2", registry.LOADING, now=self.now)
        out = self.decide([ready, pending], [make_job("1"), make_job("2")], desired=0)
        by_job = {d.job_id: d.action for d in out}
        self.assertEqual(by_job["1"], "start_drain")
        self.assertEqual(by_job["2"], "cancel")

    def test_backend_not_in_squeue_is_ignored(self):
        # Job finished/vanished from Slurm but its registry record hasn't been
        # swept yet -- decide() must not treat it as live.
        b = make_backend("1", registry.READY, now=self.now, expires_at=self.now + 100_000)
        out = self.decide([b], jobs=[])  # squeue no longer reports job "1"
        self.assertEqual([d.action for d in out], ["launch"])

    def test_freshly_submitted_job_suppresses_duplicate_launch(self):
        # Regression test: caught live, three duplicate jobs launched a tick
        # apart. A job sitting PENDING in squeue has no registry record at
        # all until its own script starts running -- `decide()` must count
        # it as pending from Slurm state, not wait for a registry record that
        # doesn't exist yet, or it re-launches a fresh one every tick.
        out = self.decide([], jobs=[make_job("1", state="PENDING")])
        self.assertEqual([d.action for d in out], ["none"])


class ApplyTest(unittest.TestCase):
    def test_reap_idle_clears_demand(self):
        # Regression test: idle-reaping only saves anything if the backend it
        # cancels actually stays down. Without clearing demand here, the next
        # tick's cold-start check just relaunches it immediately -- caught
        # live as 22 reap/relaunch cycles overnight, none of them saving
        # anything (see docs/architecture.md §9 risk 17).
        with patch.object(supervisor.slurm, "cancel") as cancel, \
             patch.object(supervisor.demand, "set_desired") as set_desired:
            supervisor.apply(supervisor.Decision("reap_idle", MODEL, "idle 30m", "1"))
        cancel.assert_called_once_with("1")
        set_desired.assert_called_once_with(MODEL, 0)

    def test_plain_cancel_does_not_touch_demand(self):
        # A drain-grace-expired cancel is not idleness -- something was using
        # this backend until very recently (or the drain path is misbehaving
        # some other way); it shouldn't suppress the next launch.
        with patch.object(supervisor.slurm, "cancel") as cancel, \
             patch.object(supervisor.demand, "set_desired") as set_desired:
            supervisor.apply(supervisor.Decision("cancel", MODEL, "drain grace expired", "1"))
        cancel.assert_called_once_with("1")
        set_desired.assert_not_called()


class CrashBackoffTest(DecideTest):
    """R5: stop relaunching a model whose bring-ups keep failing."""

    def block(self):
        self.failures = failures.Failures(consecutive=failures.MAX_CONSECUTIVE,
                                          last_error="readiness timeout", last_job="9")

    def test_below_threshold_still_launches(self):
        self.failures = failures.Failures(consecutive=failures.MAX_CONSECUTIVE - 1)
        self.assertEqual([d.action for d in self.decide([], [])], ["launch"])

    def test_blocked_model_is_not_cold_started(self):
        self.block()
        out = self.decide([], [])
        self.assertEqual([d.action for d in out], ["blocked"])
        self.assertIn("mrctl up", out[0].reason)

    def test_blocked_model_gets_no_successor(self):
        self.block()
        b = make_backend("1", registry.READY, now=self.now, ready_at=self.now - 100,
                          expires_at=self.now + self._spec.handoff_lead_s - 10,
                          last_active_at=self.now)
        out = self.decide([b], [make_job("1")])
        self.assertNotIn("launch_successor", {d.action for d in out})

    def test_blocked_model_can_still_be_reaped_and_drained(self):
        self.block()
        b = make_backend("1", registry.READY, now=self.now, ready_at=self.now - 100_000,
                          expires_at=self.now + 100_000,
                          last_active_at=self.now - self._spec.idle_timeout_s - 1)
        self.assertEqual([d.action for d in self.decide([b], [make_job("1")])], ["reap_idle"])
        self.assertEqual([d.action for d in self.decide([b], [make_job("1")], desired=0)],
                         ["start_drain"])


class FailuresStoreTest(unittest.TestCase):
    def setUp(self):
        import tempfile
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        p = patch.object(failures, "FAILURES_DIR", __import__("pathlib").Path(self.tmp.name))
        self.addCleanup(p.stop)
        p.start()

    def test_record_counts_up_and_reset_clears(self):
        self.assertEqual(failures.get("m").consecutive, 0)
        for i in range(1, failures.MAX_CONSECUTIVE + 1):
            f = failures.record("m", str(i), "boom")
        self.assertTrue(f.blocked)
        self.assertEqual((failures.get("m").last_job, failures.get("m").last_error), ("3", "boom"))
        failures.reset("m")
        self.assertFalse(failures.get("m").blocked)
        failures.reset("m")  # idempotent

    def test_manual_up_resets_but_supervisor_launch_does_not(self):
        import tempfile
        from pathlib import Path
        from mr import cli
        d = Path(tempfile.mkdtemp())
        (d / "img.sif").touch()
        spec = config.ModelSpec(name="m", hf_repo="o/M", slurm=config.SlurmSpec(**config.SITE.slurm),
                                engine=config.EngineSpec(), weights_dir=str(d), sif=str(d / "img.sif"))
        seen = []
        with patch.object(cli.config, "load", return_value=spec), \
             patch.object(cli.registry, "for_model", return_value=[]), \
             patch.object(cli.demand, "wake"), \
             patch.object(cli, "_write_spec", return_value=str(d / "spec.json")), \
             patch.object(cli.slurm, "submit", return_value="123"), \
             patch.object(cli.slurm, "start_estimate", return_value=None):
            for auto in (True, False):   # supervisor's launch, then a human's
                failures.record("m", "1", "boom")
                cli.cmd_up(type("A", (), {"model": "m", "force": True, "auto": auto})())
                seen.append(failures.get("m").consecutive)
        self.assertEqual(seen[0], 1)   # auto: count kept (then +1 below)
        self.assertEqual(seen[1], 0)   # manual: reset


if __name__ == "__main__":
    unittest.main()
