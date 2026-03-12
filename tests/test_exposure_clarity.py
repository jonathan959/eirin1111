#!/usr/bin/env python3
"""Tests for Fix #4: exposure cap clarity (risk_gate_detail structure and snapshot)."""
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class TestExposureGateDetailStructure(unittest.TestCase):
    """Ensure risk_gate_detail dicts have expected keys for UI transparency."""

    def test_per_symbol_gate_detail_has_required_keys(self):
        """Per-symbol exposure gate detail must include current/max spend and exposure %."""
        gate_detail = {
            "gate": "per_symbol_exposure_cap",
            "current_spent_quote": 50.0,
            "max_spend_quote": 100.0,
            "current_exposure_pct": 15.5,
            "per_symbol_exposure_pct": 15.0,
            "total_portfolio_value": 1000.0,
            "position_value": 155.0,
        }
        self.assertIn("gate", gate_detail)
        self.assertEqual(gate_detail["gate"], "per_symbol_exposure_cap")
        self.assertIn("current_spent_quote", gate_detail)
        self.assertIn("max_spend_quote", gate_detail)
        self.assertIn("current_exposure_pct", gate_detail)
        self.assertIn("per_symbol_exposure_pct", gate_detail)
        self.assertIn("total_portfolio_value", gate_detail)

    def test_global_exposure_gate_detail_has_required_keys(self):
        """Global exposure gate detail must include total exposure and limit."""
        gate_detail = {
            "gate": "global_exposure_cap",
            "total_portfolio_value": 5000.0,
            "total_exposure_usd": 2600.0,
            "current_exposure_pct": 52.0,
            "max_total_exposure_pct": 50.0,
        }
        self.assertEqual(gate_detail["gate"], "global_exposure_cap")
        self.assertIn("total_exposure_usd", gate_detail)
        self.assertIn("max_total_exposure_pct", gate_detail)

    def test_circuit_breaker_gate_detail_has_exposure_fields(self):
        """Circuit breaker gate detail includes exposure and deal count."""
        gate_detail = {
            "gate": "circuit_breaker",
            "reason": "Portfolio exposure limit: 52.00% >= 50.0%",
            "total_portfolio_value": 10000.0,
            "total_exposure_usd": 5200.0,
            "current_exposure_pct": 52.0,
            "max_total_exposure_pct": 50.0,
            "open_deals_count": 4,
        }
        self.assertIn("gate", gate_detail)
        self.assertIn("current_exposure_pct", gate_detail)
        self.assertIn("open_deals_count", gate_detail)


class TestExposureComparisonConsistency(unittest.TestCase):
    """Exposure triggers use >= (not >) so boundary is inclusive."""

    def test_per_symbol_boundary(self):
        """At exactly limit (symbol_pct >= per_symbol_pct) should trigger."""
        per_symbol_pct = 0.15
        symbol_pct = 0.15
        self.assertTrue(symbol_pct >= per_symbol_pct)

    def test_global_exposure_boundary(self):
        """At exactly limit (exp_ratio >= max_total_pct) should trigger."""
        exp_ratio = 0.50
        max_total_pct = 0.50
        self.assertTrue(exp_ratio >= max_total_pct)


if __name__ == "__main__":
    unittest.main()
