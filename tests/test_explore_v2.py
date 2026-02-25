#!/usr/bin/env python3
"""Unit tests for explore_v2."""
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

os.environ["EXPLORE_V2"] = "1"


class TestExploreV2Gates(unittest.TestCase):
    def test_gate_passes_normal(self):
        from explore_v2 import apply_universe_gates
        ok, reason = apply_universe_gates(
            "BTC/USD",
            volume_24h_quote=100000,
            spread_bps=25,
        )
        self.assertTrue(ok)
        self.assertIsNone(reason)

    def test_gate_blocks_low_volume(self):
        from explore_v2 import apply_universe_gates
        ok, reason = apply_universe_gates(
            "BTC/USD",
            volume_24h_quote=1000,
        )
        self.assertFalse(ok)
        self.assertIn("volume", reason.lower())

    def test_gate_blocks_wide_spread(self):
        from explore_v2 import apply_universe_gates
        ok, reason = apply_universe_gates(
            "BTC/USD",
            volume_24h_quote=10000,
            spread_bps=150,
        )
        self.assertFalse(ok)
        self.assertIn("spread", reason.lower())


class TestExploreV2EnhanceScore(unittest.TestCase):
    def test_enhance_score_penalizes_spread(self):
        from explore_v2 import enhance_score
        score, reasons = enhance_score(70, {}, "BULL", spread_bps=50)
        self.assertLess(score, 70)
        self.assertTrue(any("spread" in r.lower() for r in reasons))


class TestExploreV2Diversify(unittest.TestCase):
    def test_diversify_limits_output(self):
        from explore_v2 import diversify_picks
        items = [{"symbol": f"X{i}", "score": 80 - i} for i in range(30)]
        out = diversify_picks(items, top_k=10)
        self.assertEqual(len(out), 10)

    def test_diversify_respects_cluster(self):
        from explore_v2 import diversify_picks
        items = [
            {"symbol": "A", "score": 90, "cluster": "c1"},
            {"symbol": "B", "score": 89, "cluster": "c1"},
            {"symbol": "C", "score": 85, "cluster": "c2"},
        ]
        out = diversify_picks(items, top_k=3, cluster_key="cluster")
        self.assertLessEqual(len(out), 3)


if __name__ == "__main__":
    unittest.main()
