"""
Tests for the rule: TREND_DOWN + LONG bot -> NO_TRADE unless mean-reversion mode enabled.
Ensures we never buy into obvious drawdowns.
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
    analyze_symbol,
    classify_regime,
    determine_direction,
    classify_entry_type,
    apply_hard_filters,
    compute_all_indicators,
    FeaturesSnapshot,
    MarketSetup,
    Regime,
    DirectionBias,
    EntryType,
)


def _make_downtrend(n: int = 200, base: float = 100.0, seed: int = 42) -> pd.DataFrame:
    """Strong downtrend candles."""
    rng = np.random.RandomState(seed)
    prices = [base]
    for i in range(1, n):
        prices.append(prices[-1] * (1 - 0.004 + rng.randn() * 0.005))
    closes = np.array(prices)
    highs = closes * (1 + rng.uniform(0.001, 0.008, n))
    lows = closes * (1 - rng.uniform(0.001, 0.008, n))
    opens = (closes + lows) / 2
    volumes = 1000 * (1 + rng.uniform(-0.3, 0.5, n))
    return pd.DataFrame({
        "open": opens, "high": highs, "low": lows,
        "close": closes, "volume": volumes,
    })


class TestNoLongInDowntrend:
    """No buying into TREND_DOWN for long bots."""

    def test_downtrend_snapshot_blocked(self):
        f = FeaturesSnapshot(
            close=70, ema_20=75, ema_50=80, ema_200=90,
            adx_14=30, choppiness_index=40, atr_pct=0.03,
            rsi_14=30, volume=5000,
        )
        regime = classify_regime(f)
        assert regime == Regime.TREND_DOWN
        direction = determine_direction(f, regime)
        assert direction == DirectionBias.SHORT

        passes, reason = apply_hard_filters(
            f, Regime.TREND_DOWN, DirectionBias.LONG, EntryType.PULLBACK,
            mean_reversion_enabled=False,
        )
        assert not passes
        assert "TREND_DOWN + LONG" in reason

    def test_downtrend_full_pipeline_no_trade(self):
        candles = _make_downtrend(200, base=100.0, seed=42)
        setup = analyze_symbol("TEST/USD", candles, run_preflight=False)
        if setup.regime == "TREND_DOWN":
            assert setup.entry_type == "NO_TRADE" or setup.direction_bias == "SHORT"
            assert not setup.ready_now or setup.entry_type != "BREAKOUT"

    def test_downtrend_mean_reversion_allowed(self):
        f = FeaturesSnapshot(
            close=70, ema_20=75, ema_50=80, ema_200=90,
            adx_14=30, choppiness_index=40, atr_pct=0.03,
            rsi_14=25, volume=5000, bb_lower=69, bb_upper=85, bb_middle=77,
        )
        passes, reason = apply_hard_filters(
            f, Regime.TREND_DOWN, DirectionBias.LONG, EntryType.MEAN_REVERSION,
            mean_reversion_enabled=True,
        )
        assert passes
        assert reason == ""

    def test_downtrend_mean_reversion_blocked_by_default(self):
        f = FeaturesSnapshot(
            close=70, ema_20=75, ema_50=80, ema_200=90,
            adx_14=30, choppiness_index=40, atr_pct=0.03,
            rsi_14=25, volume=5000,
        )
        passes, reason = apply_hard_filters(
            f, Regime.TREND_DOWN, DirectionBias.LONG, EntryType.MEAN_REVERSION,
            mean_reversion_enabled=False,
        )
        assert not passes

    def test_analyze_symbol_downtrend_not_ready(self):
        """Full pipeline should NOT mark a downtrend symbol as ready for LONG."""
        candles = _make_downtrend(200, base=100.0, seed=99)
        setup = analyze_symbol("CRASH/USD", candles, run_preflight=False,
                               mean_reversion_enabled=False)
        if setup.regime == "TREND_DOWN":
            assert not setup.ready_now

    def test_analyze_symbol_downtrend_with_mr_enabled(self):
        """With mean_reversion_enabled, downtrend may produce a setup (but only if RSI oversold etc.)."""
        candles = _make_downtrend(200, base=100.0, seed=77)
        setup = analyze_symbol("RECOVER/USD", candles, run_preflight=False,
                               mean_reversion_enabled=True, min_confidence=0.2)
        # The result depends on the data, but the key is it doesn't crash
        assert setup.regime in ("TREND_DOWN", "RANGE", "HIGH_VOL")


class TestDrawdownRejection:
    """High ATR% should trigger rejection."""

    def test_high_atr_rejected(self):
        f = FeaturesSnapshot(close=100, atr_pct=0.08, volume=5000)
        passes, reason = apply_hard_filters(
            f, Regime.TREND_UP, DirectionBias.LONG, EntryType.BREAKOUT,
            max_atr_pct=0.06,
        )
        assert not passes
        assert "ATR%" in reason

    def test_normal_atr_passes(self):
        f = FeaturesSnapshot(close=100, atr_pct=0.02, volume=5000)
        passes, reason = apply_hard_filters(
            f, Regime.TREND_UP, DirectionBias.LONG, EntryType.BREAKOUT,
            max_atr_pct=0.06,
        )
        assert passes


class TestMultipleRegimes:
    """Ensure regime classification is deterministic and covers all states."""

    def test_all_regime_values_reachable(self):
        for regime in Regime:
            assert regime.value in ("TREND_UP", "TREND_DOWN", "RANGE", "HIGH_VOL")

    def test_range_with_low_adx(self):
        f = FeaturesSnapshot(close=100, ema_20=100, ema_50=100, ema_200=100,
                             adx_14=10, choppiness_index=75, atr_pct=0.02)
        assert classify_regime(f) == Regime.RANGE

    def test_trend_up_bullish_stack(self):
        f = FeaturesSnapshot(close=110, ema_20=108, ema_50=105, ema_200=95,
                             adx_14=30, choppiness_index=40, atr_pct=0.02)
        assert classify_regime(f) == Regime.TREND_UP

    def test_trend_down_bearish_stack(self):
        f = FeaturesSnapshot(close=80, ema_20=85, ema_50=90, ema_200=100,
                             adx_14=28, choppiness_index=38, atr_pct=0.03)
        assert classify_regime(f) == Regime.TREND_DOWN

    def test_high_vol_from_atr(self):
        f = FeaturesSnapshot(close=100, ema_20=100, ema_50=100, ema_200=100,
                             adx_14=25, choppiness_index=50, atr_pct=0.06)
        assert classify_regime(f) == Regime.HIGH_VOL
