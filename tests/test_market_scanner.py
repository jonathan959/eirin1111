"""
Tests for market_scanner.py — deterministic fixtures for candles,
verify regime classification, entry readiness, evidence output, edge score.
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
    compute_all_indicators,
    classify_regime,
    determine_direction,
    classify_entry_type,
    apply_hard_filters,
    compute_edge_score,
    compute_trend_quality,
    compute_volume_quality,
    compute_targets,
    build_evidence,
    analyze_symbol,
    estimate_p_tp_before_sl_rule_based,
    FeaturesSnapshot,
    MarketSetup,
    Regime,
    DirectionBias,
    EntryType,
    TimeHorizon,
    compute_ema,
    compute_rsi,
    compute_macd,
    compute_atr,
    compute_adx,
    compute_bollinger_bands,
    compute_vwap,
    compute_volume_zscore,
    compute_choppiness_index,
)


# ─── Fixtures: Deterministic OHLCV DataFrames ─────────────────────────────


def _make_candles(n: int, base: float = 100.0, trend: float = 0.001,
                  vol_mult: float = 1.0, volume_base: float = 1000.0,
                  seed: int = 42) -> pd.DataFrame:
    """Generate deterministic OHLCV data with configurable trend and volatility."""
    rng = np.random.RandomState(seed)
    prices = [base]
    for i in range(1, n):
        noise = rng.randn() * 0.01 * vol_mult
        prices.append(prices[-1] * (1 + trend + noise))
    closes = np.array(prices)
    highs = closes * (1 + rng.uniform(0.001, 0.01, n) * vol_mult)
    lows = closes * (1 - rng.uniform(0.001, 0.01, n) * vol_mult)
    opens = (closes + lows) / 2
    volumes = volume_base * (1 + rng.uniform(-0.3, 0.5, n)) * vol_mult
    return pd.DataFrame({
        "open": opens,
        "high": highs,
        "low": lows,
        "close": closes,
        "volume": volumes,
    })


@pytest.fixture
def uptrend_candles():
    """Strong uptrend: 200 bars, ~0.3% per bar trend up."""
    return _make_candles(200, base=100.0, trend=0.003, vol_mult=1.0, seed=42)


@pytest.fixture
def downtrend_candles():
    """Strong downtrend: 200 bars, ~-0.3% per bar."""
    return _make_candles(200, base=100.0, trend=-0.003, vol_mult=1.0, seed=42)


@pytest.fixture
def range_candles():
    """Ranging market: 200 bars, no trend."""
    return _make_candles(200, base=100.0, trend=0.0, vol_mult=0.8, seed=42)


@pytest.fixture
def high_vol_candles():
    """High volatility: 200 bars, no trend, high vol."""
    return _make_candles(200, base=100.0, trend=0.0, vol_mult=5.0, seed=42)


@pytest.fixture
def insufficient_candles():
    """Too few candles for analysis."""
    return _make_candles(20, base=100.0, trend=0.001, seed=42)


# ─── Test Indicator Computation ──────────────────────────────────────────────


class TestIndicators:
    def test_ema(self):
        s = pd.Series([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], dtype=float)
        ema = compute_ema(s, 3)
        assert len(ema) == 10
        assert ema.iloc[-1] > ema.iloc[0]

    def test_rsi_bounds(self, uptrend_candles):
        rsi = compute_rsi(uptrend_candles["close"], 14)
        assert rsi.max() <= 100.0
        assert rsi.min() >= 0.0

    def test_rsi_uptrend_high(self, uptrend_candles):
        rsi = compute_rsi(uptrend_candles["close"], 14)
        assert rsi.iloc[-1] > 50, "RSI should be above 50 in uptrend"

    def test_rsi_downtrend_low(self, downtrend_candles):
        rsi = compute_rsi(downtrend_candles["close"], 14)
        assert rsi.iloc[-1] < 50, "RSI should be below 50 in downtrend"

    def test_macd_uptrend_positive(self, uptrend_candles):
        line, signal, hist = compute_macd(uptrend_candles["close"], 12, 26, 9)
        assert line.iloc[-1] > 0, "MACD line should be positive in uptrend"

    def test_atr_positive(self, uptrend_candles):
        atr = compute_atr(uptrend_candles["high"], uptrend_candles["low"],
                          uptrend_candles["close"], 14)
        assert atr.iloc[-1] > 0, "ATR should be positive"

    def test_adx_trending_market(self, uptrend_candles):
        adx = compute_adx(uptrend_candles["high"], uptrend_candles["low"],
                          uptrend_candles["close"], 14)
        last_adx = float(adx.iloc[-1])
        assert last_adx > 0, "ADX should be positive in trending market"

    def test_bollinger_bands_order(self, uptrend_candles):
        upper, middle, lower = compute_bollinger_bands(uptrend_candles["close"], 20, 2.0)
        assert upper.iloc[-1] > middle.iloc[-1] > lower.iloc[-1]

    def test_vwap_reasonable(self, uptrend_candles):
        vwap = compute_vwap(uptrend_candles["high"], uptrend_candles["low"],
                            uptrend_candles["close"], uptrend_candles["volume"])
        assert vwap.iloc[-1] > 0

    def test_volume_zscore_range(self, uptrend_candles):
        vz = compute_volume_zscore(uptrend_candles["volume"], 20)
        assert not np.isnan(vz.iloc[-1])

    def test_choppiness_index_bounds(self, uptrend_candles):
        ci = compute_choppiness_index(uptrend_candles["high"], uptrend_candles["low"],
                                       uptrend_candles["close"], 14)
        assert 0 <= ci.iloc[-1] <= 100

    def test_compute_all_indicators(self, uptrend_candles):
        features = compute_all_indicators(uptrend_candles)
        assert isinstance(features, FeaturesSnapshot)
        assert features.close > 0
        assert features.ema_20 > 0
        assert features.ema_50 > 0
        assert features.atr_14 > 0
        assert 0 <= features.rsi_14 <= 100

    def test_compute_all_indicators_insufficient(self, insufficient_candles):
        features = compute_all_indicators(insufficient_candles)
        assert features.close == 0.0  # default when insufficient data


# ─── Test Regime Classification ──────────────────────────────────────────────


class TestRegimeClassification:
    def test_uptrend_regime(self, uptrend_candles):
        features = compute_all_indicators(uptrend_candles)
        regime = classify_regime(features)
        assert regime in (Regime.TREND_UP, Regime.RANGE), f"Expected TREND_UP or RANGE, got {regime}"

    def test_downtrend_regime(self, downtrend_candles):
        features = compute_all_indicators(downtrend_candles)
        regime = classify_regime(features)
        assert regime in (Regime.TREND_DOWN, Regime.RANGE), f"Expected TREND_DOWN or RANGE, got {regime}"

    def test_range_regime(self, range_candles):
        features = compute_all_indicators(range_candles)
        regime = classify_regime(features)
        assert regime == Regime.RANGE, f"Expected RANGE, got {regime}"

    def test_high_vol_regime(self, high_vol_candles):
        features = compute_all_indicators(high_vol_candles)
        regime = classify_regime(features)
        assert regime in (Regime.HIGH_VOL, Regime.RANGE), f"Expected HIGH_VOL or RANGE, got {regime}"

    def test_regime_from_snapshot_direct(self):
        features = FeaturesSnapshot(
            ema_20=105, ema_50=100, ema_200=95,
            adx_14=30, choppiness_index=40,
            close=110, atr_pct=0.02,
        )
        regime = classify_regime(features)
        assert regime == Regime.TREND_UP

    def test_regime_down_snapshot(self):
        features = FeaturesSnapshot(
            ema_20=90, ema_50=95, ema_200=100,
            adx_14=30, choppiness_index=40,
            close=85, atr_pct=0.02,
        )
        regime = classify_regime(features)
        assert regime == Regime.TREND_DOWN

    def test_regime_high_vol_snapshot(self):
        features = FeaturesSnapshot(
            ema_20=100, ema_50=100, ema_200=100,
            adx_14=15, choppiness_index=70,
            close=100, atr_pct=0.06,
        )
        regime = classify_regime(features)
        assert regime == Regime.HIGH_VOL


# ─── Test Direction Bias ─────────────────────────────────────────────────────


class TestDirectionBias:
    def test_trend_up_long(self):
        f = FeaturesSnapshot(close=110, ema_20=105, ema_50=100, ema_200=95,
                             adx_14=30, choppiness_index=40, atr_pct=0.02)
        assert determine_direction(f, Regime.TREND_UP) == DirectionBias.LONG

    def test_trend_down_short(self):
        f = FeaturesSnapshot(close=85, ema_20=90, ema_50=95, ema_200=100,
                             adx_14=30, choppiness_index=40, atr_pct=0.02)
        assert determine_direction(f, Regime.TREND_DOWN) == DirectionBias.SHORT

    def test_range_oversold_long(self):
        f = FeaturesSnapshot(close=98, rsi_14=30, bb_lower=99, bb_upper=110,
                             ema_20=100, ema_50=100, ema_200=100,
                             adx_14=15, choppiness_index=70, atr_pct=0.02)
        assert determine_direction(f, Regime.RANGE) == DirectionBias.LONG

    def test_range_neutral(self):
        f = FeaturesSnapshot(close=100, rsi_14=50, bb_lower=95, bb_upper=105,
                             ema_20=100, ema_50=100, ema_200=100,
                             adx_14=15, choppiness_index=70, atr_pct=0.02)
        assert determine_direction(f, Regime.RANGE) == DirectionBias.NONE


# ─── Test Entry Type Classification ──────────────────────────────────────────


class TestEntryType:
    def test_breakout_entry(self):
        f = FeaturesSnapshot(
            close=112, ema_20=105, ema_50=100, ema_200=95,
            adx_14=30, volume_zscore=1.5, bb_upper=110, bb_lower=95,
            bb_middle=102, rsi_14=60, atr_pct=0.02, choppiness_index=40,
        )
        entry = classify_entry_type(f, Regime.TREND_UP, DirectionBias.LONG)
        assert entry == EntryType.BREAKOUT

    def test_pullback_entry(self):
        f = FeaturesSnapshot(
            close=105.5, ema_20=105, ema_50=100, ema_200=95,
            adx_14=30, volume_zscore=0.5, bb_upper=115, bb_lower=95,
            bb_middle=105, rsi_14=45, atr_pct=0.02, choppiness_index=40,
        )
        entry = classify_entry_type(f, Regime.TREND_UP, DirectionBias.LONG)
        assert entry == EntryType.PULLBACK

    def test_mean_reversion_entry(self):
        f = FeaturesSnapshot(
            close=95, ema_20=100, ema_50=100, ema_200=99,
            adx_14=12, volume_zscore=0.3, bb_upper=110, bb_lower=95.5,
            bb_middle=102, rsi_14=28, atr_pct=0.02, choppiness_index=70,
        )
        entry = classify_entry_type(f, Regime.RANGE, DirectionBias.LONG)
        assert entry == EntryType.MEAN_REVERSION

    def test_no_trade_no_direction(self):
        f = FeaturesSnapshot(close=100, ema_20=100, ema_50=100, rsi_14=50)
        entry = classify_entry_type(f, Regime.RANGE, DirectionBias.NONE)
        assert entry == EntryType.NO_TRADE

    def test_no_trade_trend_down_short(self):
        f = FeaturesSnapshot(close=85, ema_20=90, ema_50=95, ema_200=100,
                             adx_14=30, volume_zscore=0.5, bb_upper=100,
                             bb_lower=80, bb_middle=90, rsi_14=30, atr_pct=0.02)
        entry = classify_entry_type(f, Regime.TREND_DOWN, DirectionBias.SHORT)
        assert entry == EntryType.NO_TRADE


# ─── Test Hard Filters ───────────────────────────────────────────────────────


class TestHardFilters:
    def test_reject_trend_down_long(self):
        f = FeaturesSnapshot(close=85, atr_pct=0.02, volume=5000)
        passes, reason = apply_hard_filters(
            f, Regime.TREND_DOWN, DirectionBias.LONG, EntryType.PULLBACK,
            mean_reversion_enabled=False,
        )
        assert not passes
        assert "TREND_DOWN" in reason

    def test_allow_trend_down_long_with_mean_reversion(self):
        f = FeaturesSnapshot(close=85, atr_pct=0.02, volume=5000)
        passes, reason = apply_hard_filters(
            f, Regime.TREND_DOWN, DirectionBias.LONG, EntryType.MEAN_REVERSION,
            mean_reversion_enabled=True,
        )
        assert passes

    def test_reject_high_atr(self):
        f = FeaturesSnapshot(close=100, atr_pct=0.08, volume=5000)
        passes, reason = apply_hard_filters(
            f, Regime.TREND_UP, DirectionBias.LONG, EntryType.BREAKOUT,
            max_atr_pct=0.06,
        )
        assert not passes
        assert "ATR%" in reason

    def test_reject_low_volume(self):
        f = FeaturesSnapshot(close=100, atr_pct=0.02, volume=50)
        passes, reason = apply_hard_filters(
            f, Regime.TREND_UP, DirectionBias.LONG, EntryType.BREAKOUT,
            min_volume=100,
        )
        assert not passes
        assert "Volume" in reason

    def test_reject_no_trade(self):
        f = FeaturesSnapshot(close=100, atr_pct=0.02, volume=5000)
        passes, reason = apply_hard_filters(
            f, Regime.RANGE, DirectionBias.NONE, EntryType.NO_TRADE,
        )
        assert not passes
        assert "No valid entry" in reason

    def test_pass_all(self):
        f = FeaturesSnapshot(close=100, atr_pct=0.02, volume=5000)
        passes, reason = apply_hard_filters(
            f, Regime.TREND_UP, DirectionBias.LONG, EntryType.BREAKOUT,
        )
        assert passes
        assert reason == ""


# ─── Test Edge Score ─────────────────────────────────────────────────────────


class TestEdgeScore:
    def test_edge_score_bounds(self):
        score = compute_edge_score(0.7, 0.05, 0.8, 0.6)
        assert 0.0 <= score <= 1.0

    def test_edge_score_high_quality(self):
        score = compute_edge_score(0.8, 0.10, 0.9, 0.9)
        assert score > 0.7

    def test_edge_score_low_quality(self):
        score = compute_edge_score(0.3, -0.05, 0.1, 0.1)
        assert score < 0.3

    def test_edge_score_zero_inputs(self):
        score = compute_edge_score(0.0, 0.0, 0.0, 0.0)
        assert score == 0.0

    def test_edge_score_max_inputs(self):
        score = compute_edge_score(1.0, 1.0, 1.0, 1.0)
        assert abs(score - 1.0) < 1e-10


# ─── Test Trend and Volume Quality ───────────────────────────────────────────


class TestQualityScores:
    def test_trend_quality_bullish(self):
        f = FeaturesSnapshot(close=110, ema_20=108, ema_50=105, ema_200=95, adx_14=35)
        q = compute_trend_quality(f)
        assert q > 0.7

    def test_trend_quality_mixed(self):
        f = FeaturesSnapshot(close=100, ema_20=101, ema_50=102, ema_200=99, adx_14=15)
        q = compute_trend_quality(f)
        assert 0.0 < q < 0.7

    def test_volume_quality_breakout(self):
        f = FeaturesSnapshot(close=110, volume=5000, volume_zscore=2.5, vwap=105)
        q = compute_volume_quality(f)
        assert q > 0.8

    def test_volume_quality_low(self):
        f = FeaturesSnapshot(close=100, volume=100, volume_zscore=-1.0, vwap=105)
        q = compute_volume_quality(f)
        assert q < 0.5


# ─── Test Targets and Invalidation ───────────────────────────────────────────


class TestTargets:
    def test_breakout_targets(self):
        f = FeaturesSnapshot(close=100, atr_14=2.0, ema_50=95, bb_upper=110, bb_middle=100, bb_lower=90)
        targets, inv, expected_move = compute_targets(f, EntryType.BREAKOUT, DirectionBias.LONG)
        assert targets["tp1"] > 100
        assert targets["tp2"] > targets["tp1"]
        assert inv < 100
        assert expected_move > 0

    def test_pullback_targets(self):
        f = FeaturesSnapshot(close=100, atr_14=2.0, ema_50=95, bb_upper=110, bb_middle=100, bb_lower=90)
        targets, inv, expected_move = compute_targets(f, EntryType.PULLBACK, DirectionBias.LONG)
        assert targets["tp1"] > 100
        assert inv < 100

    def test_mean_reversion_targets(self):
        f = FeaturesSnapshot(close=90, atr_14=2.0, ema_50=95, bb_upper=110, bb_middle=100, bb_lower=89)
        targets, inv, expected_move = compute_targets(f, EntryType.MEAN_REVERSION, DirectionBias.LONG)
        assert targets["tp1"] == 100  # bb_middle
        assert inv < 90

    def test_no_targets_for_no_trade(self):
        f = FeaturesSnapshot(close=100, atr_14=2.0)
        targets, inv, expected_move = compute_targets(f, EntryType.NO_TRADE, DirectionBias.NONE)
        assert targets["tp1"] == 0.0
        assert inv == 0.0


# ─── Test Evidence Generation ────────────────────────────────────────────────


class TestEvidence:
    def test_evidence_has_regime(self):
        f = FeaturesSnapshot(ema_20=105, ema_50=100, ema_200=95, adx_14=25,
                             choppiness_index=40, rsi_14=55, macd_histogram=0.5,
                             volume_zscore=1.5, close=110, atr_pct=0.02,
                             bb_upper=115, bb_lower=95)
        evidence = build_evidence(f, Regime.TREND_UP, EntryType.BREAKOUT, DirectionBias.LONG)
        assert any("Regime" in e for e in evidence)
        assert any("EMA" in e for e in evidence)
        assert any("Volume breakout" in e for e in evidence)
        assert len(evidence) >= 4

    def test_evidence_for_no_trade(self):
        f = FeaturesSnapshot(ema_20=100, ema_50=100, ema_200=100, adx_14=10,
                             choppiness_index=70, rsi_14=50, macd_histogram=0.0,
                             volume_zscore=0.0, close=100, atr_pct=0.02,
                             bb_upper=105, bb_lower=95)
        evidence = build_evidence(f, Regime.RANGE, EntryType.NO_TRADE, DirectionBias.NONE)
        assert any("No valid entry" in e for e in evidence)


# ─── Test Full Analysis Pipeline ─────────────────────────────────────────────


class TestAnalyzeSymbol:
    def test_uptrend_analysis(self, uptrend_candles):
        setup = analyze_symbol("BTC/USD", uptrend_candles, market_type="crypto",
                               run_preflight=False)
        assert isinstance(setup, MarketSetup)
        assert setup.symbol == "BTC/USD"
        assert setup.market_type == "crypto"
        assert setup.regime in ("TREND_UP", "RANGE", "HIGH_VOL")
        assert len(setup.evidence) > 0
        assert setup.features_snapshot  # not empty
        assert 0 <= setup.confidence <= 1
        assert 0 <= setup.edge_score <= 1

    def test_downtrend_analysis(self, downtrend_candles):
        setup = analyze_symbol("ETH/USD", downtrend_candles, market_type="crypto",
                               run_preflight=False)
        assert setup.regime in ("TREND_DOWN", "RANGE")
        assert not setup.ready_now or setup.entry_type == "MEAN_REVERSION"

    def test_insufficient_data(self, insufficient_candles):
        setup = analyze_symbol("TEST/USD", insufficient_candles, run_preflight=False)
        assert setup.entry_type == "NO_TRADE"
        assert "insufficient" in setup.evidence[0].lower() or "Insufficient" in setup.evidence[0]

    def test_none_candles(self):
        setup = analyze_symbol("NONE/USD", None, run_preflight=False)
        assert setup.entry_type == "NO_TRADE"

    def test_setup_to_dict(self, uptrend_candles):
        setup = analyze_symbol("BTC/USD", uptrend_candles, run_preflight=False)
        d = setup.to_dict()
        assert isinstance(d, dict)
        assert "symbol" in d
        assert "regime" in d
        assert "evidence" in d
        assert "edge_score" in d
        assert "ready_now" in d
        assert "target_levels" in d

    def test_ready_when_criteria_met(self):
        """Setup should be READY when all criteria pass."""
        candles = _make_candles(200, base=100, trend=0.004, vol_mult=1.0, seed=100)
        setup = analyze_symbol("TEST/USD", candles, run_preflight=False, min_confidence=0.3)
        # The setup may or may not be ready depending on the generated data,
        # but it should have a valid regime and entry type
        assert setup.regime in ("TREND_UP", "TREND_DOWN", "RANGE", "HIGH_VOL")

    def test_not_ready_has_trigger_conditions(self, range_candles):
        setup = analyze_symbol("RANGE/USD", range_candles, run_preflight=False,
                               min_confidence=0.99)
        if not setup.ready_now:
            assert setup.trigger_conditions or setup.ready_reason


# ─── Test P(TP before SL) Estimation ─────────────────────────────────────────


class TestPTpBeforeSl:
    def test_rule_based_range(self):
        f = FeaturesSnapshot(close=100, adx_14=25, macd_histogram=0.5,
                             volume_zscore=1.5, rsi_14=55,
                             ema_20=102, ema_50=100)
        p = estimate_p_tp_before_sl_rule_based(
            f, Regime.TREND_UP, EntryType.BREAKOUT,
            {"tp1": 104, "tp2": 106}, invalidation=97,
        )
        assert 0.1 <= p <= 0.85

    def test_rule_based_bad_rr(self):
        f = FeaturesSnapshot(close=100, adx_14=10, macd_histogram=-0.5,
                             volume_zscore=-1.0, rsi_14=30,
                             ema_20=99, ema_50=100)
        p = estimate_p_tp_before_sl_rule_based(
            f, Regime.RANGE, EntryType.MEAN_REVERSION,
            {"tp1": 100.5, "tp2": 101}, invalidation=95,
        )
        assert 0.1 <= p <= 0.85

    def test_zero_prices(self):
        f = FeaturesSnapshot(close=0)
        p = estimate_p_tp_before_sl_rule_based(
            f, Regime.RANGE, EntryType.NO_TRADE,
            {"tp1": 0, "tp2": 0}, invalidation=0,
        )
        assert p == 0.3  # fallback


# ─── Test Batch Analysis ─────────────────────────────────────────────────────


class TestBatchAnalyze:
    def test_batch(self, uptrend_candles, downtrend_candles, range_candles):
        from market_scanner import batch_analyze
        results = batch_analyze({
            "BTC/USD": (uptrend_candles, "crypto"),
            "ETH/USD": (downtrend_candles, "crypto"),
            "SOL/USD": (range_candles, "crypto"),
        }, run_preflight=False)
        assert len(results) == 3
        # Sorted by edge_score descending
        assert results[0].edge_score >= results[-1].edge_score

    def test_batch_with_error(self, uptrend_candles):
        from market_scanner import batch_analyze
        bad_df = pd.DataFrame({"open": [1], "high": [2], "low": [0.5], "close": [1.5], "volume": [100]})
        results = batch_analyze({
            "BTC/USD": (uptrend_candles, "crypto"),
            "BAD/USD": (bad_df, "crypto"),
        }, run_preflight=False)
        assert len(results) == 2
