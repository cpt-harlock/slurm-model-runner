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
import threading
import unittest
from unittest.mock import patch

from mr import gateway, registry


class RenderTest(unittest.TestCase):
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


class WakerTest(unittest.TestCase):
    def setUp(self) -> None:
        self.server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), gateway._WakerHandler)
        self.port = self.server.server_address[1]
        threading.Thread(target=self.server.serve_forever, daemon=True).start()
        self.addCleanup(self.server.shutdown)
        self.addCleanup(self.server.server_close)

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


if __name__ == "__main__":
    unittest.main()
