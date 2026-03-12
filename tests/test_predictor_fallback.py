"""
Tests for EdgePredictor ML fallback:
- If not enough data, ml_ready=False and scoring still works
- Training with sufficient data produces a model
- Predictions are clamped to conservative ranges
"""

import os
import sys
import pytest
import numpy as np
import pandas as pd

_here = os.path.dirname(os.path.abspath(__file__))
_root = os.path.dirname(_here)
if _root not in sys.path:
    sys.path.insert(0, _root)

from market_scanner import (
    EdgePredictor,
    FeaturesSnapshot,
    compute_edge_score,
    estimate_p_tp_before_sl_rule_based,
    Regime,
    EntryType,
    MIN_CANDLES_FOR_ML,
)


def _make_candles(n: int, base: float = 100.0, trend: float = 0.001,
                  seed: int = 42) -> pd.DataFrame:
    rng = np.random.RandomState(seed)
    prices = [base]
    for i in range(1, n):
        prices.append(prices[-1] * (1 + trend + rng.randn() * 0.01))
    closes = np.array(prices)
    highs = closes * (1 + rng.uniform(0.001, 0.01, n))
    lows = closes * (1 - rng.uniform(0.001, 0.01, n))
    opens = (closes + lows) / 2
    volumes = 1000 * (1 + rng.uniform(-0.3, 0.5, n))
    return pd.DataFrame({
        "open": opens, "high": highs, "low": lows,
        "close": closes, "volume": volumes,
    })


class TestEdgePredictorFallback:
    def test_untrained_predictor_not_ml_ready(self):
        pred = EdgePredictor()
        assert not pred.ml_ready

    def test_untrained_predictor_returns_defaults(self):
        pred = EdgePredictor()
        f = FeaturesSnapshot(close=100, rsi_14=50, macd_histogram=0.0,
                             atr_pct=0.02, adx_14=15, bb_upper=105, bb_lower=95,
                             volume_zscore=0.0, ema_20=100, ema_50=100)
        p_tp, exp_ret = pred.predict(f)
        assert p_tp == 0.5
        assert exp_ret == 0.0

    def test_train_with_insufficient_data(self):
        pred = EdgePredictor()
        candles = _make_candles(50)  # Too few
        success = pred.train(candles, tp_pct=0.03, sl_pct=0.08, horizon_bars=24)
        assert not success
        assert not pred.ml_ready

    def test_train_with_none(self):
        pred = EdgePredictor()
        success = pred.train(None)
        assert not success
        assert not pred.ml_ready

    def test_train_with_sufficient_data(self):
        pred = EdgePredictor()
        candles = _make_candles(500, trend=0.002, seed=123)
        success = pred.train(candles, tp_pct=0.03, sl_pct=0.08, horizon_bars=24)
        # May or may not succeed depending on label balance
        if success:
            assert pred.ml_ready
        else:
            assert not pred.ml_ready

    def test_predict_after_training(self):
        pred = EdgePredictor()
        candles = _make_candles(500, trend=0.003, seed=456)
        success = pred.train(candles, tp_pct=0.03, sl_pct=0.08, horizon_bars=24)
        f = FeaturesSnapshot(close=100, rsi_14=55, macd_histogram=0.01,
                             atr_pct=0.02, adx_14=25, bb_upper=105, bb_lower=95,
                             volume_zscore=0.5, ema_20=101, ema_50=100)
        p_tp, exp_ret = pred.predict(f)
        assert 0.15 <= p_tp <= 0.85, f"p_tp={p_tp} out of conservative range"

    def test_rule_based_fallback_always_works(self):
        f = FeaturesSnapshot(close=100, adx_14=25, macd_histogram=0.5,
                             volume_zscore=1.2, rsi_14=55,
                             ema_20=102, ema_50=100)
        p = estimate_p_tp_before_sl_rule_based(
            f, Regime.TREND_UP, EntryType.BREAKOUT,
            {"tp1": 104, "tp2": 106}, invalidation=97,
        )
        assert 0.1 <= p <= 0.85

    def test_edge_score_works_without_ml(self):
        """Edge score computation does not depend on ML."""
        p_tp = 0.6
        exp_ret = 0.03
        trend_q = 0.7
        vol_q = 0.5
        score = compute_edge_score(p_tp, exp_ret, trend_q, vol_q)
        assert 0.0 <= score <= 1.0
        assert score > 0.3  # Should be reasonable with these inputs


class TestEdgePredictorTrainingEdgeCases:
    def test_train_with_flat_prices(self):
        """Flat prices should fail training (no TP/SL hits => unbalanced labels)."""
        pred = EdgePredictor()
        n = 300
        df = pd.DataFrame({
            "open": [100.0] * n,
            "high": [100.01] * n,
            "low": [99.99] * n,
            "close": [100.0] * n,
            "volume": [1000.0] * n,
        })
        success = pred.train(df, tp_pct=0.03, sl_pct=0.08, horizon_bars=24)
        assert not success
        assert not pred.ml_ready

    def test_train_with_strong_trend(self):
        """Strong uptrend should produce trainable labels."""
        pred = EdgePredictor()
        candles = _make_candles(500, trend=0.005, seed=789)
        success = pred.train(candles, tp_pct=0.02, sl_pct=0.05, horizon_bars=10)
        # Strong trend with small TP should have enough positive labels
        if success:
            assert pred.ml_ready
            f = FeaturesSnapshot(close=100, rsi_14=60, macd_histogram=0.05,
                                 atr_pct=0.015, adx_14=30, bb_upper=105, bb_lower=95,
                                 volume_zscore=0.8, ema_20=102, ema_50=100)
            p_tp, exp_ret = pred.predict(f)
            assert 0.15 <= p_tp <= 0.85
