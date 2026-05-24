"""Unit tests for Explore Market Screener filter pipeline (bugs 1–5)."""
import unittest

from services.explore_screener_filters import (
    HIGH_CONVICTION_MIN,
    apply_screener_filters,
    effective_composite_score,
    is_high_conviction,
    is_negative_edge,
    passes_min_score,
    passes_strategy_active_gate,
    screener_reject_reason,
)


def _row(**kwargs):
    base = {
        "symbol": "TEST",
        "composite_score": 72,
        "score": 80,
        "risk_reward_ratio": 2.0,
        "backtest_win_rate_90d": 55,
        "strategy_is_active": True,
        "strategy_is_pending": False,
        "explore_strategy": "Trend Continuation",
    }
    base.update(kwargs)
    return base


class TestHighConvictionFilter(unittest.TestCase):
    def test_high_conviction_uses_composite_not_legacy_score(self):
        row = _row(composite_score=55, score=82)
        self.assertFalse(is_high_conviction(row, True))
        self.assertTrue(is_high_conviction(_row(composite_score=70), True))

    def test_threshold_is_70(self):
        self.assertEqual(HIGH_CONVICTION_MIN, 70.0)
        self.assertFalse(is_high_conviction(_row(composite_score=69.9), True))
        self.assertTrue(is_high_conviction(_row(composite_score=70), True))

    def test_disabled_passes_all_scores(self):
        self.assertTrue(is_high_conviction(_row(composite_score=10), False))


class TestMinScoreFilter(unittest.TestCase):
    def test_min_score_uses_composite(self):
        row = _row(composite_score=54, score=90)
        self.assertFalse(passes_min_score(row, 55))
        self.assertTrue(passes_min_score(_row(composite_score=55), 55))

    def test_apply_filters_min_score_regression(self):
        rows = [
            _row(symbol="LMT", composite_score=82),
            _row(symbol="BAD", composite_score=54),
        ]
        main, _side = apply_screener_filters(rows, min_score=55, high_conviction_only=False)
        self.assertEqual([r["symbol"] for r in main], ["LMT"])
        self.assertGreaterEqual(min(effective_composite_score(r) for r in main), 55)


class TestNegativeEdgeFilter(unittest.TestCase):
    def test_low_risk_reward_rejected(self):
        row = _row(risk_reward_ratio=0.1, risk_reward_display="0.1:1")
        self.assertTrue(is_negative_edge(row))

    def test_low_win_rate_rejected(self):
        row = _row(backtest_win_rate_90d=19)
        self.assertTrue(is_negative_edge(row))

    def test_actionable_passes(self):
        self.assertFalse(is_negative_edge(_row()))

    def test_coin_like_row_sidelined(self):
        row = _row(symbol="COIN", composite_score=55, risk_reward_ratio=0.9, backtest_win_rate_90d=19)
        reason = screener_reject_reason(row, min_score=55, high_conviction_only=True, show_unproven=False)
        self.assertIsNotNone(reason)


class TestStrategyActiveGate(unittest.TestCase):
    def test_pending_hidden_without_toggle(self):
        row = _row(strategy_is_active=False, strategy_is_pending=True)
        self.assertFalse(passes_strategy_active_gate(row, False))

    def test_pending_visible_with_toggle(self):
        row = _row(strategy_is_active=False, strategy_is_pending=True)
        self.assertTrue(passes_strategy_active_gate(row, True))

    def test_active_always_passes(self):
        row = _row(strategy_is_active=True, strategy_is_pending=False)
        self.assertTrue(passes_strategy_active_gate(row, False))


class TestDefaultFilterStack(unittest.TestCase):
    """Acceptance: default filters leave only high-quality active rows."""

    def test_default_stack_keeps_lmt_xlu_tlt_style_rows(self):
        rows = [
            _row(symbol="LMT", composite_score=82, strategy_is_active=True),
            _row(symbol="XLU", composite_score=76, strategy_is_active=True),
            _row(symbol="TLT", composite_score=71, strategy_is_active=True),
            _row(symbol="COIN", composite_score=55, risk_reward_ratio=0.9, backtest_win_rate_90d=19,
                 strategy_is_active=False, strategy_is_pending=True),
            _row(symbol="ZS", composite_score=60, risk_reward_ratio=0.1, strategy_is_active=True),
            _row(symbol="ARKK", composite_score=55, backtest_win_rate_90d=19,
                 strategy_is_active=False, strategy_is_pending=True),
        ]
        main, sidelined = apply_screener_filters(
            rows, min_score=55, high_conviction_only=True, show_unproven=False,
        )
        symbols = {r["symbol"] for r in main}
        self.assertEqual(symbols, {"LMT", "XLU", "TLT"})
        self.assertEqual(len(sidelined), 3)

    def test_high_conviction_off_keeps_min_score_with_rr_floor(self):
        rows = [
            _row(symbol="MID", composite_score=62, strategy_is_active=True),
            _row(symbol="LOW", composite_score=62, risk_reward_ratio=0.5, strategy_is_active=True),
        ]
        main, _ = apply_screener_filters(
            rows, min_score=55, high_conviction_only=False, show_unproven=False,
        )
        self.assertEqual([r["symbol"] for r in main], ["MID"])


class TestEffectiveCompositeScore(unittest.TestCase):
    def test_prefers_composite_over_score(self):
        self.assertEqual(effective_composite_score(_row(composite_score=72, score=99)), 72)


if __name__ == "__main__":
    unittest.main()
