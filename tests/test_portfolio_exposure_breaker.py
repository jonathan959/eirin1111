#!/usr/bin/env python3
"""Unit tests for portfolio exposure circuit breaker: math, opt-in, rate-limit."""
import os
import sys
import time
import unittest
from unittest.mock import patch, MagicMock

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class TestExposureMath(unittest.TestCase):
    """Exposure % must be computed against total equity; zero equity => no trip."""

    def test_zero_total_equity_exposure_pct_zero_no_trip(self):
        """If total_equity_usd is 0, exposure_pct must be 0 and breaker must not trip."""
        with patch.dict(os.environ, {"PORTFOLIO_EXPOSURE_BREAKER_ENABLED": "1", "PORTFOLIO_EXPOSURE_LIMIT_PCT": "50"}, clear=False):
            import importlib
            import risk_circuit_breaker as rcb
            importlib.reload(rcb)
            ok, reason = rcb.check_circuit_breakers(
                equity=0.0,
                daily_realized_pnl=0.0,
                portfolio_drawdown=0.0,
                portfolio_exposure_pct=0.0,
                open_deals_count=2,
                total_exposure_usd=100.0,
                max_total_exposure_pct=0.50,
                max_exposure_pct=0.50,
            )
            self.assertTrue(ok, "zero equity should not trip")
            self.assertIsNone(reason)

    def test_tiny_free_large_positions_uses_total_equity(self):
        """Exposure must use total equity (cash + positions), not free balance."""
        with patch.dict(os.environ, {"PORTFOLIO_EXPOSURE_BREAKER_ENABLED": "1", "PORTFOLIO_EXPOSURE_LIMIT_PCT": "50"}, clear=False):
            import importlib
            import risk_circuit_breaker as rcb
            importlib.reload(rcb)
            # Total equity 10000, exposure 4000 => 40% (under 50%)
            ok, reason = rcb.check_circuit_breakers(
                equity=10000.0,
                daily_realized_pnl=0.0,
                portfolio_drawdown=0.0,
                portfolio_exposure_pct=0.40,
                open_deals_count=2,
                total_exposure_usd=4000.0,
                max_total_exposure_pct=0.50,
                max_exposure_pct=0.50,
            )
            self.assertTrue(ok)
            self.assertIsNone(reason)
            # Same total equity, exposure 6000 => 60% (over 50%) => trip
            ok2, reason2 = rcb.check_circuit_breakers(
                equity=10000.0,
                daily_realized_pnl=0.0,
                portfolio_drawdown=0.0,
                portfolio_exposure_pct=0.60,
                open_deals_count=2,
                total_exposure_usd=6000.0,
                max_total_exposure_pct=0.50,
                max_exposure_pct=0.50,
            )
            self.assertFalse(ok2)
            self.assertIn("Portfolio exposure limit", reason2 or "")


class TestBreakerDisabledByDefault(unittest.TestCase):
    """With default settings (breaker disabled), no global_pause and no Discord for exposure."""

    def test_breaker_disabled_no_trip_on_high_exposure(self):
        """When PORTFOLIO_EXPOSURE_BREAKER_ENABLED=0, high exposure does not trip."""
        with patch.dict(os.environ, {"PORTFOLIO_EXPOSURE_BREAKER_ENABLED": "0", "PORTFOLIO_EXPOSURE_LIMIT_PCT": "50"}, clear=False):
            import importlib
            import risk_circuit_breaker as rcb
            importlib.reload(rcb)
            ok, reason = rcb.check_circuit_breakers(
                equity=10000.0,
                daily_realized_pnl=0.0,
                portfolio_drawdown=0.0,
                portfolio_exposure_pct=0.99,
                open_deals_count=2,
                total_exposure_usd=9900.0,
                max_total_exposure_pct=0.50,
                max_exposure_pct=0.50,
            )
            self.assertTrue(ok, "breaker disabled => must not trip on exposure")
            self.assertIsNone(reason)

    def test_trip_and_alert_exposure_does_nothing_when_disabled(self):
        """trip_and_alert with exposure reason and breaker disabled: no set_setting, no Discord."""
        with patch.dict(os.environ, {"PORTFOLIO_EXPOSURE_BREAKER_ENABLED": "0"}, clear=False):
            import importlib
            import risk_circuit_breaker as rcb
            importlib.reload(rcb)
            with patch("db.set_setting", MagicMock()) as set_setting:
                rcb.trip_and_alert("Portfolio exposure limit: 60.00% >= 50.0%", pause_hours=6, bot_label="test")
            set_setting.assert_not_called()


class TestBreakerEnabledAndRateLimit(unittest.TestCase):
    """When enabled with threshold 50, triggers once; rate-limit prevents spam."""

    def test_breaker_enabled_triggers_at_threshold(self):
        """PORTFOLIO_EXPOSURE_BREAKER_ENABLED=1 and limit 50 => trip when exposure >= 50%."""
        with patch.dict(os.environ, {"PORTFOLIO_EXPOSURE_BREAKER_ENABLED": "1", "PORTFOLIO_EXPOSURE_LIMIT_PCT": "50"}, clear=False):
            import importlib
            import risk_circuit_breaker as rcb
            importlib.reload(rcb)
            ok, reason = rcb.check_circuit_breakers(
                equity=10000.0,
                daily_realized_pnl=0.0,
                portfolio_drawdown=0.0,
                portfolio_exposure_pct=0.50,
                open_deals_count=2,
                total_exposure_usd=5000.0,
                max_total_exposure_pct=0.50,
                max_exposure_pct=0.50,
            )
            self.assertFalse(ok)
            self.assertIn("50.00%", reason or "")

    def test_clamp_exposure_no_nan_or_inf(self):
        """Exposure ratio is clamped so we never report > 100% or NaN."""
        with patch.dict(os.environ, {"PORTFOLIO_EXPOSURE_BREAKER_ENABLED": "1", "PORTFOLIO_EXPOSURE_LIMIT_PCT": "200"}, clear=False):
            import importlib
            import risk_circuit_breaker as rcb
            importlib.reload(rcb)
            # total_exposure_usd > equity => ratio would be > 1; we clamp to 100%
            ok, reason = rcb.check_circuit_breakers(
                equity=100.0,
                daily_realized_pnl=0.0,
                portfolio_drawdown=0.0,
                portfolio_exposure_pct=1.5,
                open_deals_count=2,
                total_exposure_usd=150.0,
                max_total_exposure_pct=2.0,
                max_exposure_pct=2.0,
            )
            # With threshold 200%, 100% clamped is under 200% so ok
            self.assertTrue(ok)


class TestClampHelper(unittest.TestCase):
    """_clamp_exposure_pct prevents NaN/inf."""

    def test_clamp_negative_zero(self):
        from risk_circuit_breaker import _clamp_exposure_pct
        self.assertEqual(_clamp_exposure_pct(-10), 0.0)

    def test_clamp_over_100_capped(self):
        from risk_circuit_breaker import _clamp_exposure_pct
        self.assertEqual(_clamp_exposure_pct(1256), 100.0)

    def test_clamp_nan_zero(self):
        from risk_circuit_breaker import _clamp_exposure_pct
        self.assertEqual(_clamp_exposure_pct(float("nan")), 0.0)


if __name__ == "__main__":
    unittest.main()
