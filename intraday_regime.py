"""
IntraDayRegimeDetector: Day-trading regime detection using VWAP, opening range, volume profile.

Uses 1m/5m/15m candles. No tick data required - approximates from OHLCV.
"""
import os
import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

ENABLE_DAY_TRADING_MODE = os.getenv("ENABLE_DAY_TRADING_MODE", "0").strip().lower() in ("1", "true", "yes", "y", "on")
INTRADAY_TIMEFRAMES = os.getenv("INTRADAY_TIMEFRAMES", "1m,5m,15m").strip().split(",")


@dataclass
class IntraDayRegimeResult:
    regime: str  # BULL_ORB, BEAR_ORB, RANGE, VWAP_BULL, VWAP_BEAR, LOW_VOLUME
    confidence: float
    vwap: Optional[float]
    or_high: Optional[float]
    or_low: Optional[float]
    price_vs_vwap: float  # >0 above VWAP, <0 below
    volume_spike_ratio: float
    why: List[str] = field(default_factory=list)
    snapshot: Dict[str, Any] = field(default_factory=dict)


def _typical_price(candle: List[float]) -> float:
    if len(candle) < 5:
        return 0.0
    o, h, l, c = float(candle[1]), float(candle[2]), float(candle[3]), float(candle[4])
    return (h + l + c) / 3.0


def _vwap_from_candles(candles: List[List[float]]) -> Optional[float]:
    if not candles:
        return None
    cum_tp_vol = 0.0
    cum_vol = 0.0
    for c in candles:
        if len(c) < 6:
            continue
        tp = _typical_price(c)
        vol = float(c[5]) if len(c) > 5 else 0.0
        if vol > 0:
            cum_tp_vol += tp * vol
            cum_vol += vol
    return cum_tp_vol / cum_vol if cum_vol > 0 else None


def _opening_range(candles_1m: List[List[float]], bars: int = 30) -> Tuple[Optional[float], Optional[float]]:
    """First N minutes (e.g. 30) high/low. Uses 1m candles."""
    if not candles_1m or len(candles_1m) < bars:
        return None, None
    subset = candles_1m[-bars:]
    highs = [float(c[2]) for c in subset if len(c) >= 3]
    lows = [float(c[3]) for c in subset if len(c) >= 4]
    if not highs or not lows:
        return None, None
    return max(highs), min(lows)


def _volume_spike_ratio(candles: List[List[float]], lookback: int = 20) -> float:
    if not candles or len(candles) < lookback + 1:
        return 1.0
    vols = [float(c[5]) for c in candles if len(c) >= 6]
    if len(vols) < lookback + 1:
        return 1.0
    recent = vols[-1]
    avg = sum(vols[-lookback-1:-1]) / lookback if lookback > 0 else 1.0
    return recent / avg if avg > 0 else 1.0


class IntraDayRegimeDetector:
    """
    Detects intraday regime using VWAP, opening range, volume profile.
    """

    def __init__(self, orb_bars: int = 30):
        self.orb_bars = orb_bars

    def detect(
        self,
        candles_1m: Optional[List[List[float]]] = None,
        candles_5m: Optional[List[List[float]]] = None,
        candles_15m: Optional[List[List[float]]] = None,
        current_price: float = 0.0,
    ) -> IntraDayRegimeResult:
        why: List[str] = []
        vwap = None
        or_high, or_low = None, None
        vol_spike = 1.0
        price_vs_vwap = 0.0

        candles = candles_5m or candles_15m or candles_1m or []
        candles_1m = candles_1m or []

        if not candles and not candles_1m:
            return IntraDayRegimeResult(
                regime="LOW_VOLUME",
                confidence=0.0,
                vwap=None,
                or_high=None,
                or_low=None,
                price_vs_vwap=0.0,
                volume_spike_ratio=1.0,
                why=["no_data"],
                snapshot={},
            )

        vwap = _vwap_from_candles(candles)
        or_high, or_low = _opening_range(candles_1m, self.orb_bars)
        vol_spike = _volume_spike_ratio(candles or candles_1m, 20)

        if vwap and current_price > 0:
            price_vs_vwap = (current_price - vwap) / vwap if vwap else 0.0

        regime = "RANGE"
        confidence = 0.5

        if or_high and or_low and current_price > 0:
            if current_price > or_high:
                regime = "BULL_ORB"
                confidence = 0.7
                why.append("price_above_opening_range_high")
            elif current_price < or_low:
                regime = "BEAR_ORB"
                confidence = 0.7
                why.append("price_below_opening_range_low")

        if vwap and current_price > 0:
            if price_vs_vwap > 0.005:
                if regime == "RANGE":
                    regime = "VWAP_BULL"
                    confidence = 0.6
                why.append("price_above_vwap")
            elif price_vs_vwap < -0.005:
                if regime == "RANGE":
                    regime = "VWAP_BEAR"
                    confidence = 0.6
                why.append("price_below_vwap")

        if vol_spike < 0.5:
            regime = "LOW_VOLUME"
            confidence = 0.8
            why.append("low_volume_spike")

        return IntraDayRegimeResult(
            regime=regime,
            confidence=confidence,
            vwap=vwap,
            or_high=or_high,
            or_low=or_low,
            price_vs_vwap=price_vs_vwap,
            volume_spike_ratio=vol_spike,
            why=why or ["default_range"],
            snapshot={
                "vwap": vwap,
                "or_high": or_high,
                "or_low": or_low,
                "vol_spike": vol_spike,
            },
        )
