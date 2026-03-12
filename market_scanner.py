"""
Market Scanner: Deterministic, evidence-based entry setup detection.

Scans universe (crypto + stocks), computes multi-timeframe indicators,
classifies market regime, identifies entry types, and produces structured
"setup" objects with confidence scores and invalidation levels.

All indicator math is pure numpy/pandas — no TA-Lib required.
"""

import logging
import math
import os
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ─── Configuration ───────────────────────────────────────────────────────────

MIN_ENTRY_CONFIDENCE = float(os.getenv("MIN_ENTRY_CONFIDENCE", "0.65"))
MIN_P_TP_BEFORE_SL = float(os.getenv("MIN_P_TP_BEFORE_SL", "0.55"))
MAX_ALLOWED_DRAWDOWN_ENTRY_PCT = float(os.getenv("MAX_ALLOWED_DRAWDOWN_ENTRY_PCT", "2.0"))
ML_TRAIN_LOOKBACK_DAYS = int(os.getenv("ML_TRAIN_LOOKBACK_DAYS", "180"))
PREDICTION_HORIZON = os.getenv("PREDICTION_HORIZON", "4h")
MIN_CANDLES_FOR_ANALYSIS = 50
MIN_CANDLES_FOR_ML = 200

# ADX threshold to distinguish trending vs choppy
ADX_TREND_THRESHOLD = float(os.getenv("ADX_TREND_THRESHOLD", "20"))


# ─── Enums ───────────────────────────────────────────────────────────────────

class Regime(str, Enum):
    TREND_UP = "TREND_UP"
    TREND_DOWN = "TREND_DOWN"
    RANGE = "RANGE"
    HIGH_VOL = "HIGH_VOL"


class DirectionBias(str, Enum):
    LONG = "LONG"
    SHORT = "SHORT"
    NONE = "NONE"


class EntryType(str, Enum):
    BREAKOUT = "BREAKOUT"
    PULLBACK = "PULLBACK"
    MEAN_REVERSION = "MEAN_REVERSION"
    NO_TRADE = "NO_TRADE"


class TimeHorizon(str, Enum):
    SHORT = "short"
    MID = "mid"
    LONG = "long"


# ─── Data Structures ─────────────────────────────────────────────────────────

@dataclass
class FeaturesSnapshot:
    ema_20: float = 0.0
    ema_50: float = 0.0
    ema_200: float = 0.0
    adx_14: float = 0.0
    rsi_14: float = 50.0
    macd_line: float = 0.0
    macd_signal: float = 0.0
    macd_histogram: float = 0.0
    atr_14: float = 0.0
    atr_pct: float = 0.0
    bb_upper: float = 0.0
    bb_middle: float = 0.0
    bb_lower: float = 0.0
    vwap: float = 0.0
    volume_zscore: float = 0.0
    choppiness_index: float = 50.0
    close: float = 0.0
    volume: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {k: round(v, 6) if isinstance(v, float) else v
                for k, v in self.__dict__.items()}


@dataclass
class MarketSetup:
    symbol: str = ""
    market_type: str = "crypto"
    regime: str = Regime.RANGE.value
    direction_bias: str = DirectionBias.NONE.value
    entry_type: str = EntryType.NO_TRADE.value
    confidence: float = 0.0
    expected_move_pct: float = 0.0
    invalidation_level: float = 0.0
    target_levels: Dict[str, float] = field(default_factory=lambda: {"tp1": 0.0, "tp2": 0.0})
    time_horizon: str = TimeHorizon.MID.value
    evidence: List[str] = field(default_factory=list)
    features_snapshot: Dict[str, Any] = field(default_factory=dict)
    edge_score: float = 0.0
    ml_ready: bool = False
    p_up: float = 0.5
    p_tp_before_sl: float = 0.5
    expected_return: float = 0.0
    trend_quality: float = 0.0
    volume_quality: float = 0.0
    gate_details: Optional[Dict[str, Any]] = None
    ready_now: bool = False
    ready_reason: str = ""
    trigger_conditions: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "symbol": self.symbol,
            "market_type": self.market_type,
            "regime": self.regime,
            "direction_bias": self.direction_bias,
            "entry_type": self.entry_type,
            "confidence": round(self.confidence, 4),
            "expected_move_pct": round(self.expected_move_pct, 4),
            "invalidation_level": round(self.invalidation_level, 6),
            "target_levels": {k: round(v, 6) for k, v in self.target_levels.items()},
            "time_horizon": self.time_horizon,
            "evidence": self.evidence,
            "features_snapshot": self.features_snapshot,
            "edge_score": round(self.edge_score, 4),
            "ml_ready": self.ml_ready,
            "p_up": round(self.p_up, 4),
            "p_tp_before_sl": round(self.p_tp_before_sl, 4),
            "expected_return": round(self.expected_return, 4),
            "trend_quality": round(self.trend_quality, 4),
            "volume_quality": round(self.volume_quality, 4),
            "gate_details": self.gate_details,
            "ready_now": self.ready_now,
            "ready_reason": self.ready_reason,
            "trigger_conditions": self.trigger_conditions,
        }


# ─── Indicator Computation ───────────────────────────────────────────────────

def compute_ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False).mean()


def compute_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    return rsi.fillna(50)


def compute_macd(series: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9
                 ) -> Tuple[pd.Series, pd.Series, pd.Series]:
    ema_fast = compute_ema(series, fast)
    ema_slow = compute_ema(series, slow)
    macd_line = ema_fast - ema_slow
    signal_line = compute_ema(macd_line, signal)
    histogram = macd_line - signal_line
    return macd_line, signal_line, histogram


def compute_atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    prev_close = close.shift(1)
    tr = pd.concat([
        (high - low),
        (high - prev_close).abs(),
        (low - prev_close).abs(),
    ], axis=1).max(axis=1)
    return tr.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()


def compute_adx(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    prev_high = high.shift(1)
    prev_low = low.shift(1)
    prev_close = close.shift(1)

    tr = pd.concat([
        (high - low),
        (high - prev_close).abs(),
        (low - prev_close).abs(),
    ], axis=1).max(axis=1)

    plus_dm = (high - prev_high).clip(lower=0)
    minus_dm = (prev_low - low).clip(lower=0)
    plus_dm[plus_dm < minus_dm] = 0
    minus_dm[minus_dm < plus_dm] = 0

    atr_smooth = tr.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    plus_di = 100 * (plus_dm.ewm(alpha=1 / period, min_periods=period, adjust=False).mean() /
                      atr_smooth.replace(0, np.nan))
    minus_di = 100 * (minus_dm.ewm(alpha=1 / period, min_periods=period, adjust=False).mean() /
                       atr_smooth.replace(0, np.nan))

    dx = 100 * ((plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan))
    adx = dx.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    return adx.fillna(0)


def compute_bollinger_bands(series: pd.Series, period: int = 20, num_std: float = 2.0
                            ) -> Tuple[pd.Series, pd.Series, pd.Series]:
    middle = series.rolling(window=period).mean()
    std = series.rolling(window=period).std()
    upper = middle + num_std * std
    lower = middle - num_std * std
    return upper, middle, lower


def compute_vwap(high: pd.Series, low: pd.Series, close: pd.Series, volume: pd.Series) -> pd.Series:
    typical = (high + low + close) / 3.0
    cumvol = volume.cumsum()
    cumtp = (typical * volume).cumsum()
    return (cumtp / cumvol.replace(0, np.nan)).fillna(close)


def compute_volume_zscore(volume: pd.Series, lookback: int = 20) -> pd.Series:
    roll_mean = volume.rolling(window=lookback).mean()
    roll_std = volume.rolling(window=lookback).std().replace(0, np.nan)
    return ((volume - roll_mean) / roll_std).fillna(0)


def compute_choppiness_index(high: pd.Series, low: pd.Series, close: pd.Series,
                              period: int = 14) -> pd.Series:
    atr1 = compute_atr(high, low, close, period=1)
    atr_sum = atr1.rolling(window=period).sum()
    highest = high.rolling(window=period).max()
    lowest = low.rolling(window=period).min()
    hl_range = (highest - lowest).replace(0, np.nan)
    ci = 100 * np.log10(atr_sum / hl_range) / np.log10(period)
    return ci.fillna(50).clip(0, 100)


def compute_all_indicators(df: pd.DataFrame) -> FeaturesSnapshot:
    """Compute all indicators from an OHLCV DataFrame and return the latest values."""
    if df is None or len(df) < MIN_CANDLES_FOR_ANALYSIS:
        return FeaturesSnapshot()

    c = df["close"]
    h = df["high"]
    l = df["low"]
    v = df["volume"]

    ema20 = compute_ema(c, 20)
    ema50 = compute_ema(c, 50)
    ema200 = compute_ema(c, 200) if len(c) >= 200 else pd.Series([c.iloc[-1]] * len(c))
    adx = compute_adx(h, l, c, 14)
    rsi = compute_rsi(c, 14)
    macd_line, macd_signal, macd_hist = compute_macd(c, 12, 26, 9)
    atr = compute_atr(h, l, c, 14)
    bb_upper, bb_middle, bb_lower = compute_bollinger_bands(c, 20, 2.0)
    vwap = compute_vwap(h, l, c, v)
    vol_z = compute_volume_zscore(v, 20)
    ci = compute_choppiness_index(h, l, c, 14)

    last = len(df) - 1
    close_val = float(c.iloc[last])
    atr_val = float(atr.iloc[last]) if not np.isnan(atr.iloc[last]) else 0.0
    atr_pct = (atr_val / close_val) if close_val > 0 else 0.0

    return FeaturesSnapshot(
        ema_20=float(ema20.iloc[last]),
        ema_50=float(ema50.iloc[last]),
        ema_200=float(ema200.iloc[last]),
        adx_14=float(adx.iloc[last]) if not np.isnan(adx.iloc[last]) else 0.0,
        rsi_14=float(rsi.iloc[last]) if not np.isnan(rsi.iloc[last]) else 50.0,
        macd_line=float(macd_line.iloc[last]) if not np.isnan(macd_line.iloc[last]) else 0.0,
        macd_signal=float(macd_signal.iloc[last]) if not np.isnan(macd_signal.iloc[last]) else 0.0,
        macd_histogram=float(macd_hist.iloc[last]) if not np.isnan(macd_hist.iloc[last]) else 0.0,
        atr_14=atr_val,
        atr_pct=atr_pct,
        bb_upper=float(bb_upper.iloc[last]) if not np.isnan(bb_upper.iloc[last]) else close_val,
        bb_middle=float(bb_middle.iloc[last]) if not np.isnan(bb_middle.iloc[last]) else close_val,
        bb_lower=float(bb_lower.iloc[last]) if not np.isnan(bb_lower.iloc[last]) else close_val,
        vwap=float(vwap.iloc[last]) if not np.isnan(vwap.iloc[last]) else close_val,
        volume_zscore=float(vol_z.iloc[last]) if not np.isnan(vol_z.iloc[last]) else 0.0,
        choppiness_index=float(ci.iloc[last]) if not np.isnan(ci.iloc[last]) else 50.0,
        close=close_val,
        volume=float(v.iloc[last]),
    )


# ─── Regime Classification ───────────────────────────────────────────────────

def classify_regime(features: FeaturesSnapshot) -> Regime:
    """Deterministic regime classification from indicator snapshot."""
    c = features.close
    if c <= 0:
        return Regime.RANGE

    adx = features.adx_14
    ci = features.choppiness_index
    atr_pct = features.atr_pct

    if atr_pct > 0.05:
        return Regime.HIGH_VOL

    is_trending = adx >= ADX_TREND_THRESHOLD and ci < 61.8

    if is_trending:
        bullish_ema = features.ema_20 > features.ema_50
        price_above_50 = c > features.ema_50
        if bullish_ema and price_above_50:
            return Regime.TREND_UP
        elif not bullish_ema and not price_above_50:
            return Regime.TREND_DOWN
        else:
            return Regime.RANGE
    else:
        return Regime.RANGE


# ─── Direction Bias ──────────────────────────────────────────────────────────

def determine_direction(features: FeaturesSnapshot, regime: Regime) -> DirectionBias:
    if regime == Regime.TREND_UP:
        return DirectionBias.LONG
    if regime == Regime.TREND_DOWN:
        return DirectionBias.SHORT
    if regime in (Regime.RANGE, Regime.HIGH_VOL):
        if features.rsi_14 < 35 and features.close < features.bb_lower:
            return DirectionBias.LONG
        if features.rsi_14 > 65 and features.close > features.bb_upper:
            return DirectionBias.SHORT
    return DirectionBias.NONE


# ─── Entry Type Classification ───────────────────────────────────────────────

def classify_entry_type(features: FeaturesSnapshot, regime: Regime,
                        direction: DirectionBias) -> EntryType:
    if direction == DirectionBias.NONE:
        return EntryType.NO_TRADE

    c = features.close

    if regime == Regime.TREND_UP and direction == DirectionBias.LONG:
        if features.volume_zscore > 1.0 and c > features.bb_upper:
            return EntryType.BREAKOUT
        ema_zone = min(features.ema_20, features.ema_50)
        if c <= features.ema_20 * 1.01 and features.rsi_14 > 35:
            return EntryType.PULLBACK
        return EntryType.NO_TRADE

    if regime == Regime.TREND_DOWN and direction == DirectionBias.SHORT:
        return EntryType.NO_TRADE

    if regime == Regime.RANGE:
        if direction == DirectionBias.LONG and features.rsi_14 < 35 and c <= features.bb_lower * 1.005:
            ema50_slope = (features.ema_50 - features.ema_200) / features.ema_200 if features.ema_200 > 0 else 0
            if ema50_slope > -0.02:
                return EntryType.MEAN_REVERSION
        if direction == DirectionBias.SHORT and features.rsi_14 > 65 and c >= features.bb_upper * 0.995:
            return EntryType.MEAN_REVERSION

    return EntryType.NO_TRADE


# ─── Trend Quality & Volume Quality ─────────────────────────────────────────

def compute_trend_quality(features: FeaturesSnapshot) -> float:
    """0..1 score measuring trend strength and alignment."""
    score = 0.0
    c = features.close
    if c <= 0:
        return 0.0

    if features.ema_20 > features.ema_50:
        score += 0.25
    if features.ema_50 > features.ema_200:
        score += 0.25
    if c > features.ema_20:
        score += 0.15
    if c > features.ema_200:
        score += 0.10

    adx_norm = min(features.adx_14 / 50.0, 1.0) * 0.25
    score += adx_norm

    return min(score, 1.0)


def compute_volume_quality(features: FeaturesSnapshot) -> float:
    """0..1 score measuring volume confirmation."""
    score = 0.0
    vz = features.volume_zscore
    if vz > 2.0:
        score += 0.6
    elif vz > 1.0:
        score += 0.4
    elif vz > 0.0:
        score += 0.2
    if features.close > features.vwap:
        score += 0.2
    if features.volume > 0:
        score += 0.2
    return min(score, 1.0)


# ─── Target & Invalidation Computation ───────────────────────────────────────

def compute_targets(features: FeaturesSnapshot, entry_type: EntryType,
                    direction: DirectionBias) -> Tuple[Dict[str, float], float, float]:
    """Returns (target_levels, invalidation_level, expected_move_pct)."""
    c = features.close
    atr = features.atr_14

    if c <= 0 or atr <= 0:
        return {"tp1": 0.0, "tp2": 0.0}, 0.0, 0.0

    if direction == DirectionBias.LONG:
        if entry_type == EntryType.BREAKOUT:
            tp1 = c + atr * 2.0
            tp2 = c + atr * 3.5
            invalidation = c - atr * 1.5
        elif entry_type == EntryType.PULLBACK:
            tp1 = c + atr * 1.5
            tp2 = c + atr * 3.0
            invalidation = min(features.ema_50, c - atr * 1.2)
        elif entry_type == EntryType.MEAN_REVERSION:
            tp1 = features.bb_middle
            tp2 = features.bb_upper * 0.98
            invalidation = c - atr * 1.0
        else:
            return {"tp1": 0.0, "tp2": 0.0}, 0.0, 0.0
    elif direction == DirectionBias.SHORT:
        if entry_type == EntryType.MEAN_REVERSION:
            tp1 = features.bb_middle
            tp2 = features.bb_lower * 1.02
            invalidation = c + atr * 1.0
        else:
            return {"tp1": 0.0, "tp2": 0.0}, 0.0, 0.0
    else:
        return {"tp1": 0.0, "tp2": 0.0}, 0.0, 0.0

    expected_move = abs(tp1 - c) / c if c > 0 else 0.0
    return {"tp1": tp1, "tp2": tp2}, invalidation, expected_move


# ─── Evidence Generation ─────────────────────────────────────────────────────

def build_evidence(features: FeaturesSnapshot, regime: Regime,
                   entry_type: EntryType, direction: DirectionBias) -> List[str]:
    """Human-readable evidence bullets for the setup."""
    evidence = []

    evidence.append(f"Regime: {regime.value} (ADX={features.adx_14:.1f}, CI={features.choppiness_index:.1f})")

    if features.ema_20 > features.ema_50 > features.ema_200:
        evidence.append("EMA stack: bullish (20>50>200)")
    elif features.ema_20 < features.ema_50 < features.ema_200:
        evidence.append("EMA stack: bearish (20<50<200)")
    else:
        evidence.append("EMA stack: mixed")

    evidence.append(f"RSI={features.rsi_14:.1f}, MACD hist={features.macd_histogram:.4f}")

    if features.volume_zscore > 1.0:
        evidence.append(f"Volume breakout: z-score={features.volume_zscore:.2f}")
    elif features.volume_zscore < -1.0:
        evidence.append(f"Volume drying up: z-score={features.volume_zscore:.2f}")

    if entry_type == EntryType.BREAKOUT:
        evidence.append("Entry: price breaking above BB upper with volume")
    elif entry_type == EntryType.PULLBACK:
        evidence.append("Entry: pullback to EMA20/50 zone with RSI support")
    elif entry_type == EntryType.MEAN_REVERSION:
        evidence.append("Entry: mean reversion at BB lower + RSI oversold")
    else:
        evidence.append("No valid entry trigger identified")

    evidence.append(f"ATR%={features.atr_pct*100:.2f}%, BB width=${features.bb_upper - features.bb_lower:.4f}")

    return evidence


# ─── Rule-Based Quality Filters (Hard Reject) ────────────────────────────────

def apply_hard_filters(features: FeaturesSnapshot, regime: Regime,
                       direction: DirectionBias, entry_type: EntryType,
                       mean_reversion_enabled: bool = False,
                       max_atr_pct: float = 0.06,
                       min_volume: float = 0.0) -> Tuple[bool, str]:
    """
    Hard reject filters. Returns (pass, reject_reason).
    If rejected, the symbol should not trade.
    """
    if regime == Regime.TREND_DOWN and direction == DirectionBias.LONG and not mean_reversion_enabled:
        return False, "TREND_DOWN + LONG rejected (no mean-reversion mode)"

    if features.atr_pct > max_atr_pct:
        return False, f"ATR% too high: {features.atr_pct*100:.2f}% > {max_atr_pct*100:.1f}%"

    if min_volume > 0 and features.volume < min_volume:
        return False, f"Volume below minimum: {features.volume:.0f} < {min_volume:.0f}"

    if entry_type == EntryType.NO_TRADE:
        return False, "No valid entry type identified"

    return True, ""


# ─── Edge Score Computation ──────────────────────────────────────────────────

def compute_edge_score(
    p_tp_before_sl: float,
    expected_return: float,
    trend_quality: float,
    volume_quality: float,
) -> float:
    """
    EdgeScore = 0.45 * p_tp_before_sl + 0.25 * expected_return_norm + 0.20 * trend_quality + 0.10 * volume_quality
    """
    er_norm = min(max(expected_return * 10.0, 0.0), 1.0)
    edge = (0.45 * p_tp_before_sl +
            0.25 * er_norm +
            0.20 * trend_quality +
            0.10 * volume_quality)
    return min(max(edge, 0.0), 1.0)


def estimate_p_tp_before_sl_rule_based(
    features: FeaturesSnapshot, regime: Regime, entry_type: EntryType,
    target_levels: Dict[str, float], invalidation: float,
) -> float:
    """
    Rule-based estimation of P(TP before SL) when ML is not available.
    Uses trend strength, momentum alignment, and risk/reward geometry.
    """
    c = features.close
    tp1 = target_levels.get("tp1", 0.0)
    if c <= 0 or tp1 <= 0 or invalidation <= 0:
        return 0.3

    reward = abs(tp1 - c)
    risk = abs(c - invalidation)
    rr_ratio = (reward / risk) if risk > 0 else 0.5

    base = 0.40

    if regime == Regime.TREND_UP and entry_type in (EntryType.BREAKOUT, EntryType.PULLBACK):
        base += 0.15
    if features.adx_14 > 25:
        base += 0.05
    if features.macd_histogram > 0:
        base += 0.05
    if features.volume_zscore > 1.0:
        base += 0.05
    if 40 < features.rsi_14 < 70:
        base += 0.03

    if rr_ratio > 2.0:
        base += 0.05
    elif rr_ratio < 1.0:
        base -= 0.10

    return min(max(base, 0.1), 0.85)


# ─── ML Predictor (Lightweight, Fallback-Safe) ──────────────────────────────

class EdgePredictor:
    """Lightweight ML predictor for P(TP before SL) and expected return.
    Falls back to rule-based scoring when insufficient data."""

    def __init__(self):
        self._model = None
        self._scaler = None
        self._ml_ready = False
        self._feature_names: List[str] = []

    @property
    def ml_ready(self) -> bool:
        return self._ml_ready

    def train(self, candles_df: pd.DataFrame, tp_pct: float = 0.03,
              sl_pct: float = 0.08, horizon_bars: int = 24) -> bool:
        """
        Train on historical candles. Label = did price hit TP before SL within horizon.
        Returns True if model trained successfully.
        """
        if candles_df is None or len(candles_df) < MIN_CANDLES_FOR_ML:
            self._ml_ready = False
            return False

        try:
            from sklearn.linear_model import LogisticRegression
            from sklearn.preprocessing import StandardScaler
        except ImportError:
            logger.warning("scikit-learn not available, ML predictor disabled")
            self._ml_ready = False
            return False

        try:
            df = candles_df.copy()
            features_df = self._build_feature_matrix(df)
            labels = self._build_labels(df, tp_pct, sl_pct, horizon_bars)

            if features_df is None or labels is None:
                self._ml_ready = False
                return False

            min_len = min(len(features_df), len(labels))
            features_df = features_df.iloc[:min_len]
            labels = labels.iloc[:min_len]

            valid_mask = features_df.notna().all(axis=1) & labels.notna()
            features_df = features_df[valid_mask]
            labels = labels[valid_mask]

            if len(features_df) < 50 or labels.sum() < 10 or (len(labels) - labels.sum()) < 10:
                logger.info("Insufficient labeled data for ML (%d rows, %d positive)", len(features_df), labels.sum())
                self._ml_ready = False
                return False

            self._feature_names = list(features_df.columns)
            self._scaler = StandardScaler()
            X = self._scaler.fit_transform(features_df.values)
            y = labels.values

            self._model = LogisticRegression(C=0.1, max_iter=500, class_weight="balanced")
            self._model.fit(X, y)
            self._ml_ready = True

            train_acc = self._model.score(X, y)
            logger.info("EdgePredictor trained: %d samples, accuracy=%.3f", len(X), train_acc)
            return True

        except Exception as e:
            logger.warning("EdgePredictor training failed: %s", e)
            self._ml_ready = False
            return False

    def predict(self, features: FeaturesSnapshot) -> Tuple[float, float]:
        """Returns (p_tp_before_sl, expected_return_estimate).
        Falls back to (0.5, 0.0) if not trained."""
        if not self._ml_ready or self._model is None or self._scaler is None:
            return 0.5, 0.0

        try:
            x = self._features_from_snapshot(features)
            x_scaled = self._scaler.transform([x])
            proba = self._model.predict_proba(x_scaled)[0]
            p_tp = float(proba[1]) if len(proba) > 1 else 0.5
            p_tp = min(max(p_tp, 0.15), 0.85)
            expected_ret = (p_tp - 0.5) * features.atr_pct * 2
            return p_tp, expected_ret
        except Exception as e:
            logger.debug("EdgePredictor predict failed: %s", e)
            return 0.5, 0.0

    def _build_feature_matrix(self, df: pd.DataFrame) -> Optional[pd.DataFrame]:
        """Build feature matrix from OHLCV DataFrame."""
        try:
            c = df["close"]
            h = df["high"]
            l = df["low"]
            v = df["volume"]

            features = pd.DataFrame(index=df.index)
            features["rsi"] = compute_rsi(c, 14)
            macd_l, macd_s, macd_h = compute_macd(c, 12, 26, 9)
            features["macd_hist"] = macd_h
            features["atr_pct"] = compute_atr(h, l, c, 14) / c.replace(0, np.nan)
            features["adx"] = compute_adx(h, l, c, 14)
            features["bb_pct"] = (c - compute_bollinger_bands(c, 20, 2.0)[2]) / \
                                  (compute_bollinger_bands(c, 20, 2.0)[0] - compute_bollinger_bands(c, 20, 2.0)[2]).replace(0, np.nan)
            features["vol_zscore"] = compute_volume_zscore(v, 20)
            features["ret_5"] = c.pct_change(5)
            features["ret_20"] = c.pct_change(20)
            features["ema_ratio"] = compute_ema(c, 20) / compute_ema(c, 50).replace(0, np.nan)
            return features.dropna()
        except Exception:
            return None

    def _build_labels(self, df: pd.DataFrame, tp_pct: float, sl_pct: float,
                      horizon: int) -> Optional[pd.Series]:
        """Label each bar: 1 if TP hit before SL within horizon, else 0."""
        try:
            c = df["close"]
            h = df["high"]
            l = df["low"]
            labels = pd.Series(index=df.index, dtype=float)

            for i in range(len(df) - horizon):
                entry = float(c.iloc[i])
                if entry <= 0:
                    labels.iloc[i] = np.nan
                    continue
                tp_level = entry * (1 + tp_pct)
                sl_level = entry * (1 - sl_pct)
                hit_tp = False
                hit_sl = False
                for j in range(1, horizon + 1):
                    if i + j >= len(df):
                        break
                    if float(h.iloc[i + j]) >= tp_level:
                        hit_tp = True
                        break
                    if float(l.iloc[i + j]) <= sl_level:
                        hit_sl = True
                        break
                labels.iloc[i] = 1.0 if hit_tp else 0.0

            return labels.iloc[:len(df) - horizon]
        except Exception:
            return None

    def _features_from_snapshot(self, features: FeaturesSnapshot) -> List[float]:
        """Extract feature vector matching training order from a snapshot."""
        c = features.close
        bb_range = features.bb_upper - features.bb_lower
        bb_pct = (c - features.bb_lower) / bb_range if bb_range > 0 else 0.5
        return [
            features.rsi_14,
            features.macd_histogram,
            features.atr_pct,
            features.adx_14,
            bb_pct,
            features.volume_zscore,
            0.0,  # ret_5 placeholder
            0.0,  # ret_20 placeholder
            features.ema_20 / features.ema_50 if features.ema_50 > 0 else 1.0,
        ]


# ─── Candidate Preflight (Execution Gate Check) ─────────────────────────────

def run_candidate_preflight(symbol: str, market_type: str = "crypto") -> Tuple[bool, Dict[str, Any]]:
    """
    Check if a candidate would pass the execution gate RIGHT NOW.
    Uses the unified execution gate with live data if available.
    Returns (passes, gate_details_dict).
    """
    try:
        from execution_gate import check_execution_gate, GateResult
    except ImportError:
        return True, {"note": "execution_gate not available"}

    try:
        if market_type == "crypto":
            try:
                from worker_api import _safe_kraken_client
                kc = _safe_kraken_client()
                if kc:
                    from execution_gate import fetch_gate_inputs
                    inputs = fetch_gate_inputs(kc, symbol)
                    result = check_execution_gate(
                        symbol=symbol,
                        bid=inputs.get("bid"),
                        ask=inputs.get("ask"),
                        last_price=inputs.get("last_price"),
                        ticker_ts=inputs.get("ticker_ts"),
                        volume_24h=inputs.get("volume_24h"),
                        dry_run=True,
                    )
                    return result.allowed, result.to_dict()
            except Exception as e:
                logger.debug("Crypto preflight failed for %s: %s", symbol, e)
        else:
            result = check_execution_gate(
                symbol=symbol,
                side="buy",
                dry_run=True,
                skip_spread=True,
                skip_stale=True,
            )
            return result.allowed, result.to_dict()
    except Exception as e:
        logger.debug("Preflight check failed for %s: %s", symbol, e)

    return True, {"note": "preflight skipped (no live data)"}


# ─── Main Scanner Entry Point ────────────────────────────────────────────────

_predictor = EdgePredictor()


def analyze_symbol(
    symbol: str,
    candles_df: pd.DataFrame,
    market_type: str = "crypto",
    mean_reversion_enabled: bool = False,
    run_preflight: bool = True,
    min_confidence: float = 0.0,
) -> MarketSetup:
    """
    Full analysis pipeline for a single symbol.

    Args:
        symbol: Trading symbol (e.g. "BTC/USD", "AAPL")
        candles_df: OHLCV DataFrame with columns: open, high, low, close, volume
        market_type: "crypto" or "stocks"
        mean_reversion_enabled: Allow long entries in TREND_DOWN
        run_preflight: Check execution gate
        min_confidence: Minimum confidence to mark as ready (0 = use global default)

    Returns:
        MarketSetup with all analysis fields populated.
    """
    setup = MarketSetup(symbol=symbol, market_type=market_type)

    if candles_df is None or len(candles_df) < MIN_CANDLES_FOR_ANALYSIS:
        setup.entry_type = EntryType.NO_TRADE.value
        setup.evidence = ["Insufficient candle data for analysis"]
        setup.ready_reason = "insufficient_data"
        return setup

    features = compute_all_indicators(candles_df)
    setup.features_snapshot = features.to_dict()

    regime = classify_regime(features)
    setup.regime = regime.value

    direction = determine_direction(features, regime)
    setup.direction_bias = direction.value

    entry_type = classify_entry_type(features, regime, direction)
    setup.entry_type = entry_type.value

    passes_hard, reject_reason = apply_hard_filters(
        features, regime, direction, entry_type,
        mean_reversion_enabled=mean_reversion_enabled,
    )

    if not passes_hard:
        setup.entry_type = EntryType.NO_TRADE.value
        setup.evidence = [reject_reason]
        setup.ready_reason = f"hard_filter: {reject_reason}"
        setup.trigger_conditions = _build_trigger_conditions(features, regime)
        return setup

    targets, invalidation, expected_move = compute_targets(features, entry_type, direction)
    setup.target_levels = targets
    setup.invalidation_level = invalidation
    setup.expected_move_pct = expected_move

    setup.evidence = build_evidence(features, regime, entry_type, direction)

    trend_q = compute_trend_quality(features)
    volume_q = compute_volume_quality(features)
    setup.trend_quality = trend_q
    setup.volume_quality = volume_q

    if _predictor.ml_ready:
        p_tp, exp_ret = _predictor.predict(features)
        setup.ml_ready = True
    else:
        p_tp = estimate_p_tp_before_sl_rule_based(features, regime, entry_type, targets, invalidation)
        exp_ret = expected_move * (p_tp - 0.3)
        setup.ml_ready = False

    setup.p_tp_before_sl = p_tp
    setup.p_up = p_tp
    setup.expected_return = exp_ret

    edge = compute_edge_score(p_tp, exp_ret, trend_q, volume_q)
    setup.edge_score = edge
    setup.confidence = edge

    if entry_type == EntryType.BREAKOUT or entry_type == EntryType.PULLBACK:
        setup.time_horizon = TimeHorizon.MID.value
    elif entry_type == EntryType.MEAN_REVERSION:
        setup.time_horizon = TimeHorizon.SHORT.value

    gate_passes = True
    gate_details = None
    if run_preflight:
        gate_passes, gate_details = run_candidate_preflight(symbol, market_type)
        setup.gate_details = gate_details

    min_conf = min_confidence if min_confidence > 0 else MIN_ENTRY_CONFIDENCE
    is_ready = (
        entry_type != EntryType.NO_TRADE
        and setup.confidence >= min_conf
        and setup.p_tp_before_sl >= MIN_P_TP_BEFORE_SL
        and gate_passes
    )

    setup.ready_now = is_ready
    if is_ready:
        setup.ready_reason = "all_criteria_met"
    else:
        reasons = []
        if entry_type == EntryType.NO_TRADE:
            reasons.append("no_entry_type")
        if setup.confidence < min_conf:
            reasons.append(f"confidence {setup.confidence:.2f} < {min_conf:.2f}")
        if setup.p_tp_before_sl < MIN_P_TP_BEFORE_SL:
            reasons.append(f"p_tp_before_sl {setup.p_tp_before_sl:.2f} < {MIN_P_TP_BEFORE_SL:.2f}")
        if not gate_passes:
            reasons.append("execution_gate_blocked")
        setup.ready_reason = "; ".join(reasons)
        setup.trigger_conditions = _build_trigger_conditions(features, regime)

    return setup


def _build_trigger_conditions(features: FeaturesSnapshot, regime: Regime) -> str:
    """Describe what needs to happen for this symbol to become tradeable."""
    conditions = []
    c = features.close

    if regime == Regime.TREND_DOWN:
        conditions.append(f"Wait for EMA20 ({features.ema_20:.4f}) to cross above EMA50 ({features.ema_50:.4f})")
    elif regime == Regime.RANGE:
        if features.adx_14 < ADX_TREND_THRESHOLD:
            conditions.append(f"ADX needs to rise above {ADX_TREND_THRESHOLD} (currently {features.adx_14:.1f})")
    if features.volume_zscore < 0.5:
        conditions.append(f"Volume needs pickup (z-score={features.volume_zscore:.2f}, need >1.0)")
    if features.rsi_14 > 70:
        conditions.append(f"RSI overbought ({features.rsi_14:.1f}), wait for pullback to 40-60 zone")
    if not conditions:
        conditions.append("Monitor for entry trigger alignment")

    return " | ".join(conditions)


def batch_analyze(
    symbols_candles: Dict[str, Tuple[pd.DataFrame, str]],
    mean_reversion_enabled: bool = False,
    run_preflight: bool = True,
    min_confidence: float = 0.0,
) -> List[MarketSetup]:
    """
    Analyze multiple symbols and return sorted by edge score.

    Args:
        symbols_candles: {symbol: (candles_df, market_type)}
    """
    results = []
    for symbol, (candles_df, market_type) in symbols_candles.items():
        try:
            setup = analyze_symbol(
                symbol=symbol,
                candles_df=candles_df,
                market_type=market_type,
                mean_reversion_enabled=mean_reversion_enabled,
                run_preflight=run_preflight,
                min_confidence=min_confidence,
            )
            results.append(setup)
        except Exception as e:
            logger.warning("Scanner failed for %s: %s", symbol, e)
            setup = MarketSetup(symbol=symbol, market_type=market_type)
            setup.evidence = [f"Analysis error: {e}"]
            results.append(setup)

    results.sort(key=lambda s: s.edge_score, reverse=True)
    return results


def train_predictor(candles_df: pd.DataFrame, tp_pct: float = 0.03,
                    sl_pct: float = 0.08, horizon_bars: int = 24) -> bool:
    """Train the global edge predictor on historical data."""
    return _predictor.train(candles_df, tp_pct, sl_pct, horizon_bars)
