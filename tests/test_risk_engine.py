#!/usr/bin/env python3
"""Unit tests for risk_engine."""
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Force risk engine OFF for tests that expect it off
os.environ["RISK_ENGINE_ENABLED"] = "0"


class TestRiskEngineComputeExposure(unittest.TestCase):
    def test_compute_exposure_basic(self):
        from risk_engine import compute_exposure
        a, t = compute_exposure(1000, {"BTC/USD": 200}, "BTC/USD")
        self.assertAlmostEqual(a, 0.2)
        self.assertAlmostEqual(t, 0.2)

    def test_compute_exposure_multiple(self):
        from risk_engine import compute_exposure
        pos = {"BTC/USD": 100, "ETH/USD": 50}
        a, t = compute_exposure(1000, pos, "BTC/USD")
        self.assertAlmostEqual(a, 0.1)
        self.assertAlmostEqual(t, 0.15)

    def test_compute_exposure_zero_balance(self):
        from risk_engine import compute_exposure
        a, t = compute_exposure(0, {"BTC/USD": 100}, "BTC/USD")
        self.assertEqual(a, 0.0)
        self.assertEqual(t, 0.0)


class TestRiskEngineCanOpenTrade(unittest.TestCase):
    def setUp(self):
        os.environ["RISK_ENGINE_ENABLED"] = "1"

    def tearDown(self):
        os.environ["RISK_ENGINE_ENABLED"] = "0"

    def test_allows_normal_exposure(self):
        from risk_engine import RiskContext, can_open_trade
        ctx = RiskContext(
            bot_id=1, symbol="BTC/USD",
            balance_total_usd=10000, balance_free_usd=5000,
            positions_usd={"BTC/USD": 100},
            symbol_position_usd=100,
            trades_today=5,
        )
        ok, reason = can_open_trade(ctx)
        self.assertTrue(ok, reason)
        self.assertEqual(reason, "")

    def test_blocks_excessive_total_exposure(self):
        from risk_engine import RiskContext, can_open_trade
        ctx = RiskContext(
            bot_id=1, symbol="BTC/USD",
            balance_total_usd=1000, balance_free_usd=100,
            positions_usd={"BTC/USD": 250},
            symbol_position_usd=250,
        )
        ok, reason = can_open_trade(ctx)
        self.assertFalse(ok)
        self.assertIn("exposure", reason.lower())

    def test_blocks_trade_limit(self):
        from risk_engine import RiskContext, can_open_trade
        ctx = RiskContext(
            bot_id=1, symbol="BTC/USD",
            balance_total_usd=10000, balance_free_usd=5000,
            positions_usd={},
            symbol_position_usd=0,
            trades_today=25,
        )
        ok, reason = can_open_trade(ctx)
        self.assertFalse(ok)
        self.assertIn("trades", reason.lower())


if __name__ == "__main__":
    unittest.main()
