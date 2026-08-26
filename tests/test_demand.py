"""Unit tests for mr.demand -- the per-model lazy-wake flag."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from mr import demand


class DemandTest(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        patcher = patch.object(demand, "DEMAND_DIR", Path(self.tmp.name))
        self.addCleanup(patcher.stop)
        patcher.start()

    def test_default_is_desired(self):
        # No file yet -- default to "keep it up" (Phase 0-3 behavior),
        # not "stay down", so a newly-configured model still comes up.
        self.assertEqual(demand.get_desired("never-touched"), 1)

    def test_set_and_get_roundtrip(self):
        demand.set_desired("qwen3-32b", 0)
        self.assertEqual(demand.get_desired("qwen3-32b"), 0)

    def test_wake_sets_desired_to_one(self):
        demand.set_desired("qwen3-32b", 0)
        demand.wake("qwen3-32b")
        self.assertEqual(demand.get_desired("qwen3-32b"), 1)

    def test_models_are_independent(self):
        demand.set_desired("a", 0)
        self.assertEqual(demand.get_desired("b"), 1)


if __name__ == "__main__":
    unittest.main()
