"""Tests for mr.gateway: config rendering and the waker stub.

The waker test spins up a real (ephemeral-port) HTTP server -- worth it
because the actual bug here was only visible over real HTTP: LiteLLM
auto-retries a 503 (ServiceUnavailableError) for ~90s before ever returning
to the client, so the waker must answer 400 instead. That's a wire-level
behavior no amount of calling the handler method directly would have caught.
"""

from __future__ import annotations

import http.client
import http.server
import json
import tempfile
import threading
import time
import unittest
import unittest.mock
from unittest.mock import patch

from pathlib import Path

from mr import gateway, litellm_hooks, registry, slurm

MODEL = "qwen3-32b"
JOB_NAME = f"mr-{MODEL}"


def make_job(job_id: str, state: str) -> slurm.Job:
    return slurm.Job(job_id=job_id, name=JOB_NAME, state=state, nodelist="",
                      start_time=0.0, end_time=0.0)


class RenderTest(unittest.TestCase):
    def setUp(self):
        # render() reads the live spec for active backends; never the real one.
        self._tmp = tempfile.TemporaryDirectory()
        p = patch("mr.config.SPEC_DIR", Path(self._tmp.name))
        p.start()
        self.addCleanup(p.stop)
        self.addCleanup(self._tmp.cleanup)

    def test_empty_status_is_empty_list(self):
        self.assertIn("model_list: []", gateway.render({}))

    def test_active_backend_routes_directly(self):
        b = registry.Backend(model="qwen3-32b", served_name="qwen3-32b", job_id="1",
                              state=registry.READY, host="lrdn1", port=8000)
        with patch("mr.config.load") as load:
            load.return_value.engine.max_model_len = 32768
            out = gateway.render({"qwen3-32b": b})
        self.assertIn("api_base: http://lrdn1:8000/v1", out)
        self.assertNotIn(f"127.0.0.1:{gateway.WAKER_PORT}", out)

    def test_cold_model_routes_to_waker(self):
        with patch("mr.config.load") as load:
            load.return_value.engine.max_model_len = 32768
            out = gateway.render({"qwen3-32b": None})
        self.assertIn(f"api_base: http://127.0.0.1:{gateway.WAKER_PORT}/qwen3-32b/v1", out)

    def test_max_input_tokens_leaves_room_for_output(self):
        # Regression test for a real bug (architecture.md §9 risk 29): rendering
        # max_input_tokens as the *full* max_model_len while max_output_tokens
        # was a quarter of it *on top* double-counted vLLM's combined
        # prompt+completion budget. A real Avante session hit this live.
        b = registry.Backend(model="qwen3-32b", served_name="qwen3-32b", job_id="1",
                              state=registry.READY, host="lrdn1", port=8000)
        with patch("mr.config.load") as load:
            load.return_value.engine.max_model_len = 32768
            load.return_value.chat_template_kwargs = {}
            out = gateway.render({"qwen3-32b": b})
        self.assertIn("max_input_tokens: 24576", out)
        self.assertIn("max_output_tokens: 8192", out)

    def test_chat_template_kwargs_rendered_when_set(self):
        # deepseek-v3.1-terminus needs {"thinking": true} forwarded into every
        # request or its reasoning parser dumps the whole answer into
        # `reasoning` with content: null (architecture.md §9 risk 28).
        b = registry.Backend(model="deepseek-v3.1-terminus", served_name="deepseek-v3.1-terminus",
                              job_id="1", state=registry.READY, host="lrdn1", port=8000)
        with patch("mr.config.load") as load:
            load.return_value.engine.max_model_len = 32768
            load.return_value.chat_template_kwargs = {"thinking": True}
            out = gateway.render({"deepseek-v3.1-terminus": b})
        self.assertIn("chat_template_kwargs:", out)
        self.assertIn("thinking: true", out)

    def test_no_chat_template_kwargs_block_when_unset(self):
        b = registry.Backend(model="qwen3-32b", served_name="qwen3-32b", job_id="1",
                              state=registry.READY, host="lrdn1", port=8000)
        with patch("mr.config.load") as load:
            load.return_value.engine.max_model_len = 32768
            load.return_value.chat_template_kwargs = {}
            out = gateway.render({"qwen3-32b": b})
        self.assertNotIn("chat_template_kwargs:", out)

    def test_primary_role_wins_over_alphabetical_order(self):
        # "aaa-small" sorts first alphabetically; the old logic picked
        # models[0] as primary and matched "32b" in the name for small,
        # which only ever worked because qwen3-32b was the only model
        # configured. A real second model that doesn't happen to sort last
        # would have silently routed sonnet/opus to the small model instead
        # -- caught before it ever shipped, not live.
        small = registry.Backend(model="aaa-small", served_name="aaa-small", job_id="1",
                                  state=registry.READY, host="h1", port=8001)
        primary = registry.Backend(model="zzz-primary", served_name="zzz-primary", job_id="2",
                                    state=registry.READY, host="h2", port=8002)

        def fake_load(name):
            spec = type("Spec", (), {})()
            spec.engine = type("Engine", (), {"max_model_len": 32768})()
            spec.role = "small" if name == "aaa-small" else "primary"
            spec.chat_template_kwargs = {}
            return spec

        with patch("mr.config.load", side_effect=fake_load):
            out = gateway.render({"aaa-small": small, "zzz-primary": primary})

        blocks = {
            block.splitlines()[0]: block
            for block in out.split("  - model_name: ")[1:]
        }
        self.assertIn("h2:8002", blocks["claude-*sonnet*"])
        self.assertNotIn("h1:8001", blocks["claude-*sonnet*"])
        self.assertIn("h1:8001", blocks["claude-*haiku*"])


class ServedMaxLenTest(unittest.TestCase):
    """2026-11-14: a TOML raised to 1010000 for the next run made the gateway
    advertise 757500 input tokens for a job serving 65536."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.spec_dir = Path(self._tmp.name)
        p = patch("mr.config.SPEC_DIR", self.spec_dir)
        p.start()
        self.addCleanup(p.stop)
        (self.spec_dir / "m.json").write_text(json.dumps({"engine": {"max_model_len": 65536}}))
        self.b = registry.Backend(model="m", served_name="m", job_id="1",
                                  state=registry.READY, host="h", port=1)

    def test_live_backend_uses_its_spec_not_the_toml(self):
        with patch("mr.config.load") as load:
            load.return_value.engine.max_model_len = 1010000
            self.assertEqual(gateway._served_max_len("m", self.b), 65536)

    def test_cold_model_uses_the_toml(self):
        with patch("mr.config.load") as load:
            load.return_value.engine.max_model_len = 1010000
            self.assertEqual(gateway._served_max_len("m", None), 1010000)

    def test_missing_spec_falls_back_to_the_toml(self):
        (self.spec_dir / "m.json").unlink()
        with patch("mr.config.load") as load:
            load.return_value.engine.max_model_len = 1010000
            self.assertEqual(gateway._served_max_len("m", self.b), 1010000)


QWEN_MAP = {"high": "xhigh", "max": "xhigh", "minimal": "low", "none": "low"}


class EffortHookTest(unittest.TestCase):
    """mr/litellm_hooks.rewrite. Live 2026-11-14: Anthropic thinking budgets
    of 6000 and 16000 400'd on qwen3.8-2.4t-mxfp4 (\"Unexpected reasoning
    effort high\") because LiteLLM buckets them into \"high\"."""

    maps = {"q": QWEN_MAP, "claude-*sonnet*": QWEN_MAP}

    def rw(self, **data):
        return litellm_hooks.rewrite(dict(data), self.maps)

    def test_high_becomes_xhigh(self):
        self.assertEqual(self.rw(model="q", reasoning_effort="high")["reasoning_effort"], "xhigh")

    def test_accepted_value_passes_through(self):
        self.assertEqual(self.rw(model="q", reasoning_effort="medium")["reasoning_effort"], "medium")

    def test_large_thinking_budget_maps_to_xhigh(self):
        out = self.rw(model="q", thinking={"type": "enabled", "budget_tokens": 16000})
        self.assertEqual(out["reasoning_effort"], "xhigh")

    def test_tiny_thinking_budget_maps_minimal_to_low(self):
        out = self.rw(model="q", thinking={"type": "enabled", "budget_tokens": 500})
        self.assertEqual(out["reasoning_effort"], "low")

    def test_medium_budget_is_set_explicitly(self):
        out = self.rw(model="q", thinking={"type": "enabled", "budget_tokens": 3000})
        self.assertEqual(out["reasoning_effort"], "medium")

    def test_explicit_effort_wins_over_budget(self):
        out = self.rw(model="q", reasoning_effort="low", thinking={"type": "enabled", "budget_tokens": 16000})
        self.assertEqual(out["reasoning_effort"], "low")

    def test_alias_pattern_gets_the_target_map(self):
        out = self.rw(model="claude-sonnet-5", reasoning_effort="high")
        self.assertEqual(out["reasoning_effort"], "xhigh")

    def test_unmapped_model_is_untouched(self):
        data = {"model": "other", "reasoning_effort": "high"}
        self.assertEqual(litellm_hooks.rewrite(dict(data), self.maps), data)

    def test_no_effort_no_thinking_adds_nothing(self):
        self.assertNotIn("reasoning_effort", self.rw(model="q"))

    def test_disabled_thinking_adds_nothing(self):
        self.assertNotIn("reasoning_effort", self.rw(model="q", thinking={"type": "disabled"}))


class EffortMapsTest(unittest.TestCase):
    def test_served_name_and_aliases_carry_the_map(self):
        specs = {"q": QWEN_MAP, "small": {}}

        def load(name):
            m = unittest.mock.MagicMock()
            m.reasoning_effort_map = specs[name]
            m.role = {"q": "primary", "small": "small"}[name]
            return m

        with patch("mr.config.load", side_effect=load):
            maps = gateway.effort_maps({"q": None, "small": None})
        self.assertEqual(maps["q"], QWEN_MAP)
        self.assertEqual(maps["claude-*opus*"], QWEN_MAP)
        self.assertNotIn("small", maps)
        self.assertNotIn("claude-*haiku*", maps)


class WakerTest(unittest.TestCase):
    def setUp(self) -> None:
        self.server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), gateway._WakerHandler)
        self.port = self.server.server_address[1]
        threading.Thread(target=self.server.serve_forever, daemon=True).start()
        self.addCleanup(self.server.shutdown)
        self.addCleanup(self.server.server_close)
        # The ETA queries the real squeue, which can take longer than the 5 s
        # client timeout on a busy login node (seen 2026-11-14); it has its
        # own tests in WakeEtaTest.
        p = patch.object(gateway, "_wake_eta_message", return_value="eta")
        p.start()
        self.addCleanup(p.stop)

    def test_returns_400_not_503(self):
        # The load-bearing assertion: LiteLLM treats 503 as retriable and
        # stalls for ~90s before answering the client at all. 400 (a client
        # error, correctly -- "the model isn't up, that's not negotiable
        # right now") returns immediately. See mr.gateway._WakerHandler.
        with patch.object(gateway.demand, "wake") as wake:
            conn = http.client.HTTPConnection("127.0.0.1", self.port, timeout=5)
            conn.request("POST", "/qwen3-32b/chat/completions", body=b'{"model":"qwen3-32b"}')
            resp = conn.getresponse()
            status, body = resp.status, resp.read()
        self.assertEqual(status, 400)
        self.assertIn(b"model_warming_up", body)
        wake.assert_called_once_with("qwen3-32b")

    def test_drains_request_body(self):
        # A client can stall waiting on this connection if the server
        # responds without ever reading what it sent -- see the handler.
        with patch.object(gateway.demand, "wake"):
            conn = http.client.HTTPConnection("127.0.0.1", self.port, timeout=5)
            conn.request("POST", "/qwen3-32b/chat/completions", body=b"x" * 10_000)
            resp = conn.getresponse()
            resp.read()
            self.assertEqual(resp.status, 400)


class WakeEtaTest(unittest.TestCase):
    """mr.gateway._wake_eta_message: queue-aware UX, not a fixed guess."""

    def setUp(self) -> None:
        self.spec = type("Spec", (), {"measured_load_s": 300})()
        patcher = patch.object(gateway.config, "load", return_value=self.spec)
        self.addCleanup(patcher.stop)
        patcher.start()

    def eta(self, jobs, backends=()):
        with patch.object(gateway.slurm, "jobs", return_value=jobs), \
             patch.object(gateway.registry, "for_model", return_value=backends):
            return gateway._wake_eta_message(MODEL)

    def test_nothing_yet_says_so_honestly(self):
        msg = self.eta(jobs=[])
        self.assertIn("will be requested within 60s", msg)
        self.assertIn("300s", msg)  # measured_load_s, not a hardcoded number

    def test_pending_with_slurm_estimate_uses_real_eta(self):
        with patch.object(gateway.slurm, "start_estimate", return_value=time.time() + 120):
            msg = self.eta(jobs=[make_job("1", "PENDING")])
        self.assertIn("Slurm estimates start", msg)

    def test_pending_without_slurm_estimate_says_so_honestly(self):
        with patch.object(gateway.slurm, "start_estimate", return_value=0):
            msg = self.eta(jobs=[make_job("1", "PENDING")])
        self.assertIn("no Slurm estimate yet", msg)

    def test_already_loading_uses_measured_remaining_time(self):
        b = registry.Backend(model=MODEL, served_name=MODEL, job_id="1",
                              state=registry.LOADING, started_at=time.time() - 100)
        msg = self.eta(jobs=[make_job("1", "RUNNING")], backends=[b])
        self.assertIn("already loading", msg)
        self.assertIn("s left", msg)

    def test_never_raises_even_if_everything_fails(self):
        with patch.object(gateway.config, "load", side_effect=RuntimeError("boom")):
            msg = gateway._wake_eta_message(MODEL)
        self.assertIsInstance(msg, str)
        self.assertTrue(msg)


class _FakeProc:
    def __init__(self, pid: int, returncode: int | None):
        self.pid, self.returncode = pid, returncode

    def poll(self):
        return self.returncode


class EnsureLitellmTest(unittest.TestCase):
    """R3: a dead or never-started litellm must be (re)started on the next tick."""

    def test_live_process_is_kept(self):
        live = _FakeProc(1, None)
        with patch.object(gateway, "start_litellm") as start:
            self.assertIs(gateway.ensure_litellm(live), live)
        start.assert_not_called()

    def test_exited_process_is_replaced(self):
        new = _FakeProc(2, None)
        with patch.object(gateway, "start_litellm", return_value=new):
            self.assertIs(gateway.ensure_litellm(_FakeProc(1, 1)), new)

    def test_failed_start_returns_none_and_next_tick_retries(self):
        new = _FakeProc(3, None)
        boom = FileNotFoundError(2, "No such file or directory", "litellm")
        with patch.object(gateway, "start_litellm", side_effect=[boom, new]):
            with self.assertLogs("mr.gateway", level="ERROR"):
                self.assertIsNone(gateway.ensure_litellm(None))
            self.assertIs(gateway.ensure_litellm(None), new)

    def test_restart_with_no_process_just_starts(self):
        new = _FakeProc(4, None)
        with patch.object(gateway, "start_litellm", return_value=new):
            self.assertIs(gateway.restart_litellm(None), new)

    def test_shutdown_is_not_swallowed(self):
        with patch.object(gateway, "start_litellm", side_effect=gateway._Shutdown()):
            with self.assertRaises(gateway._Shutdown):
                gateway.ensure_litellm(None)


class FindLitellmTest(unittest.TestCase):
    def test_env_var_wins(self):
        with patch.dict("os.environ", {"MR_LITELLM": "/x/litellm"}), \
             patch("os.access", return_value=True):
            self.assertEqual(gateway.find_litellm(), "/x/litellm")

    def test_path_then_venv_fallback(self):
        with patch.dict("os.environ", {}, clear=False) as env, \
             patch("shutil.which", return_value=None), \
             patch("os.access", side_effect=lambda p, m: p.endswith(".venvs/litellm/bin/litellm")):
            env.pop("MR_LITELLM", None)
            self.assertTrue(gateway.find_litellm().endswith(".venvs/litellm/bin/litellm"))

    def test_missing_is_a_clear_error(self):
        with patch.dict("os.environ", {}, clear=False) as env, \
             patch("shutil.which", return_value=None), patch("os.access", return_value=False):
            env.pop("MR_LITELLM", None)
            with self.assertRaisesRegex(FileNotFoundError, "MR_LITELLM"):
                gateway.find_litellm()


if __name__ == "__main__":
    unittest.main()
