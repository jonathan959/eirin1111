"""
Integration tests for the Explore composite scoring overhaul.

Validates:
  - No BUY signal from strategy with <40% backtest win rate
  - No BUY signal with negative avg backtest return
  - Score never exceeds strategy quality cap
  - Conviction reflects actual factor alignment
  - Fear & Greed extreme values properly penalize
  - Crypto BUY blocked during extreme fear + BTC downtrend
  - Assets up >12% in 24h flagged as chasing
  - Grade A/B/C/D/F assigned correctly
  - Max 3 signals per strategy type
  - Max 10 total BUY signals
  - 200MA display fixed when percentage is negative
"""
import unittest

from explore_composite_scorer import (
    strategy_quality_gate,
    macro_environment_score,
    technical_score,
    calculate_composite_score,
    assign_grade,
    apply_safety_filters,
    filter_correlated_signals,
    fix_200ma_display,
)


class TestStrategyQualityGate(unittest.TestCase):

    def test_net_loser_blocked(self):
        bt = {"pullback_support": {"win_rate": 11, "avg_return": -1.12, "signals": 100}}
        gate = strategy_quality_gate("pullback_support", bt)
        self.assertFalse(gate["passed"])
        self.assertLessEqual(gate["max_score_cap"], 25)

    def test_low_winrate_blocked(self):
        bt = {"crypto_momentum": {"win_rate": 7.7, "avg_return": -1.38, "signals": 100}}
        gate = strategy_quality_gate("crypto_momentum", bt)
        self.assertFalse(gate["passed"])
        self.assertLessEqual(gate["max_score_cap"], 25)

    def test_marginal_strategy_capped(self):
        bt = {"momentum_breakout": {"win_rate": 29.9, "avg_return": 0.21, "signals": 100}}
        gate = strategy_quality_gate("momentum_breakout", bt)
        self.assertFalse(gate["passed"])
        self.assertTrue(gate["max_score_cap"] <= 35)

    def test_good_strategy_passes(self):
        bt = {"trend_continuation": {"win_rate": 64.1, "avg_return": 3.02, "signals": 100}}
        gate = strategy_quality_gate("trend_continuation", bt)
        self.assertTrue(gate["passed"])
        self.assertGreaterEqual(gate["max_score_cap"], 80)

    def test_insufficient_data_not_hard_fail(self):
        bt = {"test_strat": {"win_rate": 50, "avg_return": 1.0, "signals": 5}}
        gate = strategy_quality_gate("test_strat", bt)
        self.assertTrue(gate["passed"])

    def test_no_buy_from_sub40_winrate(self):
        """No BUY signal should come from a strategy with <40% win rate."""
        bt = {"oversold_reversal": {"win_rate": 16.5, "avg_return": -0.73, "signals": 100}}
        gate = strategy_quality_gate("oversold_reversal", bt)
        self.assertFalse(gate["passed"])


class TestMacroEnvironment(unittest.TestCase):

    def test_extreme_fear_crypto_penalized(self):
        result = macro_environment_score(8, "RANGE", 0.65, 0.5, "crypto")
        self.assertFalse(result["block_buy"])
        self.assertLess(result["score_adjustment"], -15)

    def test_extreme_fear_stocks_not_blocked(self):
        result = macro_environment_score(8, "RANGE", 0.65, 0.5, "stock")
        self.assertFalse(result["block_buy"])

    def test_btc_downtrend_crypto_penalized(self):
        result = macro_environment_score(50, "BEAR", 0.8, 0.5, "crypto")
        self.assertLess(result["score_adjustment"], -10)

    def test_btc_uptrend_crypto_favorable(self):
        result = macro_environment_score(60, "BULL", 0.2, 0.3, "crypto")
        self.assertGreater(result["score_adjustment"], 0)

    def test_neutral_conditions(self):
        result = macro_environment_score(50, "RANGE", 0.3, 0.4, "stock")
        self.assertEqual(result["macro_label"], "neutral")


class TestSafetyFilters(unittest.TestCase):

    def test_chasing_24h_flagged(self):
        result = apply_safety_filters("TEST", 15.0, None, None, None, [], "crypto")
        self.assertTrue(any("Chasing" in f or "chasing" in f for f in result["flags"]))

    def test_extreme_move_blocked(self):
        result = apply_safety_filters("TEST", 35.0, None, None, None, [], "crypto")
        self.assertTrue(result["block_buy"])

    def test_normal_move_not_flagged(self):
        result = apply_safety_filters("TEST", 3.0, None, None, None, [100]*30, "stock")
        self.assertFalse(result["block_buy"])
        self.assertEqual(len(result["flags"]), 0)


class TestGradeAssignment(unittest.TestCase):

    def test_grade_f_on_block(self):
        gate = {"passed": True, "max_score_cap": 100, "quality_score": 80}
        macro = {"score_adjustment": 0, "block_buy": True}
        tech = {"risk_reward_ratio": 2.5, "trend_direction": "up"}
        safety = {"block_buy": False}
        self.assertEqual(assign_grade(80, gate, macro, tech, safety), "F")

    def test_grade_a_best_case(self):
        gate = {"passed": True, "max_score_cap": 100, "quality_score": 80}
        macro = {"score_adjustment": 5, "block_buy": False}
        tech = {"risk_reward_ratio": 3.0, "trend_direction": "up"}
        safety = {"block_buy": False}
        self.assertEqual(assign_grade(85, gate, macro, tech, safety), "A")

    def test_grade_f_gate_failed(self):
        gate = {"passed": False, "max_score_cap": 0, "quality_score": 0}
        macro = {"score_adjustment": 0, "block_buy": False}
        tech = {"risk_reward_ratio": None}
        safety = {"block_buy": False}
        self.assertEqual(assign_grade(10, gate, macro, tech, safety), "F")


class TestCorrelationFilter(unittest.TestCase):

    def test_max_3_per_strategy(self):
        signals = [
            {"symbol": f"SYM{i}", "explore_strategy_id": "momentum_breakout", "score": 90 - i, "signal": "buy"}
            for i in range(10)
        ]
        filtered = filter_correlated_signals(signals)
        mb_count = sum(1 for s in filtered if s["explore_strategy_id"] == "momentum_breakout")
        self.assertLessEqual(mb_count, 3)

    def test_max_10_buy_signals(self):
        signals = []
        for i in range(20):
            strat = f"strat_{i % 5}"
            signals.append({"symbol": f"SYM{i}", "explore_strategy_id": strat, "score": 80, "signal": "buy"})
        filtered = filter_correlated_signals(signals)
        buy_count = sum(1 for s in filtered if s["signal"] == "buy")
        self.assertLessEqual(buy_count, 10)


class TestFix200MADisplay(unittest.TestCase):

    def test_negative_pct_fixed(self):
        result = fix_200ma_display(-53.0, "Above 200-day MA (~-53% above MA), factor momentum 60.")
        self.assertIn("BELOW 200-day MA by 53%", result)
        self.assertNotIn("Above 200-day MA", result)

    def test_positive_pct_unchanged(self):
        result = fix_200ma_display(12.0, "Above 200-day MA (~12% above MA), momentum 70.")
        self.assertIn("Above 200-day MA", result)


class TestCompositeScore(unittest.TestCase):

    def _make_closes(self, n=250, base=100.0, trend=0.001):
        return [base + i * trend * base for i in range(n)]

    def test_losing_strategy_never_buy(self):
        closes = self._make_closes()
        bt = {"pullback_support": {"win_rate": 11, "avg_return": -1.12, "signals": 100}}
        result = calculate_composite_score(
            "TEST", "pullback_support", 80.0, bt,
            50, {}, "stock", closes, closes, closes, [1000]*250,
        )
        self.assertNotEqual(result["signal"], "buy")
        self.assertTrue(result["blocked"])

    def test_extreme_fear_blocks_crypto(self):
        closes = self._make_closes()
        bt = {"trend_continuation": {"win_rate": 65, "avg_return": 3.0, "signals": 100}}
        # Macro hard-block only when extreme fear + strong BTC downtrend + very high vol.
        result = calculate_composite_score(
            "BTC/USD", "trend_continuation", 80.0, bt,
            8, {"regime": "BEAR", "downtrend_score": 0.71, "hv": 0.95}, "crypto",
            closes, closes, closes, [1000]*250,
        )
        self.assertNotEqual(result["signal"], "buy")
        self.assertTrue(result["blocked"])

    def test_good_strategy_can_buy(self):
        closes = self._make_closes(trend=0.003)
        bt = {"trend_continuation": {"win_rate": 65, "avg_return": 3.0, "signals": 100}}
        result = calculate_composite_score(
            "AAPL", "trend_continuation", 85.0, bt,
            55, {"regime": "BULL"}, "stock",
            closes, closes, closes, [1000]*250,
        )
        self.assertIn(result["signal"], ("buy", "watch"))
        self.assertGreater(result["score"], 50)

    def test_score_capped_by_strategy_quality(self):
        closes = self._make_closes(trend=0.005)
        bt = {"test": {"win_rate": 45, "avg_return": 0.5, "signals": 50, "warn": True}}
        result = calculate_composite_score(
            "TEST", "test", 90.0, bt,
            50, {}, "stock",
            closes, closes, closes, [1000]*250,
        )
        self.assertLessEqual(result["score"], 55)

    def test_conviction_reflects_factors(self):
        closes = self._make_closes()
        bt = {}
        result = calculate_composite_score(
            "TEST", "unknown", 50.0, bt,
            50, {}, "stock",
            closes, closes, closes, [1000]*250,
        )
        self.assertIsInstance(result["conviction"], int)
        self.assertGreaterEqual(result["conviction"], 0)
        self.assertLessEqual(result["conviction"], 100)


if __name__ == "__main__":
    unittest.main()
