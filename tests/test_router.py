#!/usr/bin/env python3
"""Unit tests for strategy router (ROUTER_V1)."""
import os
import sys
import unittest
from unittest.mock import MagicMock

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class TestRouterDisabled(unittest.TestCase):
    def test_router_returns_current_when_disabled(self):
        os.environ["ROUTER_V1"] = "0"
        from strategies import select_strategy, RegimeResult
        regime = RegimeResult(regime="RANGE", confidence=0.8, why=[], snapshot={}, scores={"range": 0.9}, vol_ratio=0.02)
        target, switched, reason = select_strategy(regime, "smart_dca", 0, 1000)
        self.assertEqual(target, "smart_dca")
        self.assertFalse(switched)
        self.assertIn("router", reason.lower())

    def test_forced_strategy_takes_precedence_when_router_disabled(self):
        os.environ["ROUTER_V1"] = "0"
        from strategies import select_strategy, RegimeResult
        regime = RegimeResult(regime="RANGE", confidence=0.8, why=[], snapshot={}, scores={"range": 0.9}, vol_ratio=0.02)
        target, switched, reason = select_strategy(regime, "smart_dca", 0, 1000, forced="grid")
        self.assertEqual(target, "grid")
        self.assertFalse(switched)
        self.assertEqual(reason, "forced")


class TestRouterEnabled(unittest.TestCase):
    def setUp(self):
        os.environ["ROUTER_V1"] = "1"

    def tearDown(self):
        os.environ["ROUTER_V1"] = "0"

    def test_router_selects_range_strategy(self):
        from strategies import select_strategy, RegimeResult
        regime = RegimeResult(regime="RANGE", confidence=0.8, why=[], snapshot={}, scores={"range": 0.9}, vol_ratio=0.02)
        target, switched, reason = select_strategy(regime, "smart_dca", 0, 1000)
        self.assertEqual(target, "range_mean_reversion")
        self.assertTrue(switched)

    def test_router_selects_trend_for_uptrend(self):
        from strategies import select_strategy, RegimeResult
        regime = RegimeResult(regime="BULL", confidence=0.8, why=[], snapshot={}, scores={"uptrend_score": 0.9}, vol_ratio=0.02)
        target, switched, reason = select_strategy(regime, "smart_dca", 0, 1000)
        self.assertEqual(target, "trend_follow")


if __name__ == "__main__":
    unittest.main()
