"""
Chart Pattern Recognition for swing trading.

Detects: head & shoulders, double top/bottom, bull/bear flags, ascending/descending triangles.
Pure Python - no OpenCV/TA-Lib. Uses price geometry on OHLCV candles.
"""
import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


@dataclass
class PatternResult:
    pattern: str  # head_shoulders, double_top, double_bottom, bull_flag, bear_flag, ascending_triangle, descending_triangle
    direction: str  # bullish, bearish
    confidence: float
    entry_zone: Optional[Tuple[float, float]] = None
    target: Optional[float] = None
    stop: Optional[float] = None
    bars_used: int = 0


def _closes(candles: List[List[float]]) -> List[float]:
    return [float(c[4]) for c in candles if len(c) >= 5]


def _highs(candles: List[List[float]]) -> List[float]:
    return [float(c[2]) for c in candles if len(c) >= 3]


def _lows(candles: List[List[float]]) -> List[float]:
    return [float(c[3]) for c in candles if len(c) >= 4]


def _find_peaks(closes: List[float], lookback: int = 3) -> List[Tuple[int, float]]:
    """Return (index, value) of local peaks."""
    peaks = []
    n = len(closes)
    for i in range(lookback, n - lookback):
        v = closes[i]
        if all(closes[j] <= v for j in range(i - lookback, i + lookback + 1) if j != i):
            peaks.append((i, v))
    return peaks


def _find_troughs(closes: List[float], lookback: int = 3) -> List[Tuple[int, float]]:
    """Return (index, value) of local troughs."""
    troughs = []
    n = len(closes)
    for i in range(lookback, n - lookback):
        v = closes[i]
        if all(closes[j] >= v for j in range(i - lookback, i + lookback + 1) if j != i):
            troughs.append((i, v))
    return troughs


def _detect_head_shoulders(closes: List[float]) -> Optional[PatternResult]:
    """Head & shoulders: 3 peaks, middle (head) highest. Bearish reversal."""
    if len(closes) < 30:
        return None
    peaks = _find_peaks(closes, 2)
    if len(peaks) < 3:
        return None
    last3 = peaks[-3:]
    left, head, right = last3[0][1], last3[1][1], last3[2][1]
    if head <= left or head <= right:
        return None
    tolerance = 0.02
    if abs(left - right) / max(left, right, 1e-9) > tolerance * 2:
        return None
    neckline = (left + right) / 2
    target = neckline - (head - neckline)
    conf = min(0.95, 0.6 + (head - max(left, right)) / head * 2)
    return PatternResult("head_shoulders", "bearish", conf, (neckline * 0.99, neckline * 1.01), target, head * 1.01, len(closes))


def _detect_inverse_head_shoulders(closes: List[float]) -> Optional[PatternResult]:
    """Inverse H&S: 3 troughs, middle lowest. Bullish reversal."""
    if len(closes) < 30:
        return None
    troughs = _find_troughs(closes, 2)
    if len(troughs) < 3:
        return None
    last3 = troughs[-3:]
    left, head, right = last3[0][1], last3[1][1], last3[2][1]
    if head >= left or head >= right:
        return None
    tolerance = 0.02
    if abs(left - right) / max(left, right, 1e-9) > tolerance * 2:
        return None
    neckline = (left + right) / 2
    target = neckline + (neckline - head)
    conf = min(0.95, 0.6 + (min(left, right) - head) / head * 2)
    return PatternResult("inverse_head_shoulders", "bullish", conf, (neckline * 0.99, neckline * 1.01), target, head * 0.99, len(closes))


def _detect_double_top(closes: List[float]) -> Optional[PatternResult]:
    """Double top: 2 peaks at similar level. Bearish."""
    if len(closes) < 20:
        return None
    peaks = _find_peaks(closes, 2)
    if len(peaks) < 2:
        return None
    p1, p2 = peaks[-2][1], peaks[-1][1]
    if abs(p1 - p2) / max(p1, p2, 1e-9) > 0.015:
        return None
    trough_idx = max(peaks[-2][0], peaks[-1][0]) - 1
    while trough_idx > 0 and closes[trough_idx] >= min(p1, p2) * 0.98:
        trough_idx -= 1
    neckline = closes[trough_idx] if trough_idx >= 0 else (p1 + p2) / 2 * 0.95
    target = neckline - (max(p1, p2) - neckline)
    return PatternResult("double_top", "bearish", 0.7, (neckline * 0.99, neckline * 1.01), target, max(p1, p2) * 1.005, len(closes))


def _detect_double_bottom(closes: List[float]) -> Optional[PatternResult]:
    """Double bottom: 2 troughs at similar level. Bullish."""
    if len(closes) < 20:
        return None
    troughs = _find_troughs(closes, 2)
    if len(troughs) < 2:
        return None
    t1, t2 = troughs[-2][1], troughs[-1][1]
    if abs(t1 - t2) / max(t1, t2, 1e-9) > 0.015:
        return None
    peak_val = max(closes[troughs[-2][0]:troughs[-1][0] + 1]) if troughs[-1][0] > troughs[-2][0] else (t1 + t2) / 2 * 1.05
    neckline = peak_val
    target = neckline + (neckline - min(t1, t2))
    return PatternResult("double_bottom", "bullish", 0.7, (neckline * 0.99, neckline * 1.01), target, min(t1, t2) * 0.995, len(closes))


def _detect_flag(closes: List[float], highs: List[float], lows: List[float]) -> Optional[PatternResult]:
    """Bull/bear flag: strong move followed by narrow consolidation."""
    if len(closes) < 25:
        return None
    n = len(closes)
    pole_len = 10
    flag_len = 12
    if n < pole_len + flag_len:
        return None
    pole = closes[pole_len - 1] - closes[0]
    flag_highs = highs[-flag_len:]
    flag_lows = lows[-flag_len:]
    if not flag_highs or not flag_lows:
        return None
    flag_range = max(flag_highs) - min(flag_lows)
    pole_pct = abs(pole) / closes[0] if closes[0] else 0
    if pole_pct < 0.03 or flag_range / (closes[-1] or 1) > 0.03:
        return None
    if pole > 0:
        return PatternResult("bull_flag", "bullish", 0.65, (closes[-1] * 0.998, closes[-1] * 1.002), closes[-1] + pole, min(flag_lows) * 0.995, n)
    return PatternResult("bear_flag", "bearish", 0.65, (closes[-1] * 0.998, closes[-1] * 1.002), closes[-1] + pole, max(flag_highs) * 1.005, n)


def _detect_triangle(closes: List[float], highs: List[float], lows: List[float]) -> Optional[PatternResult]:
    """Ascending/descending triangle: converging range."""
    if len(closes) < 25:
        return None
    n = len(closes)
    first_half_h = max(highs[: n // 2]) if highs[: n // 2] else closes[-1]
    first_half_l = min(lows[: n // 2]) if lows[: n // 2] else closes[-1]
    second_half_h = max(highs[n // 2 :]) if highs[n // 2 :] else closes[-1]
    second_half_l = min(lows[n // 2 :]) if lows[n // 2 :] else closes[-1]
    range1 = first_half_h - first_half_l
    range2 = second_half_h - second_half_l
    if range1 < 1e-9:
        return None
    contraction = 1.0 - range2 / range1
    if contraction < 0.2:
        return None
    cur = closes[-1]
    if abs(first_half_h - second_half_h) / max(first_half_h, 1e-9) < 0.02:
        return PatternResult("ascending_triangle", "bullish", 0.6, (cur * 0.998, cur * 1.002), first_half_h * 1.02, second_half_l * 0.995, n)
    if abs(first_half_l - second_half_l) / max(first_half_l, 1e-9) < 0.02:
        return PatternResult("descending_triangle", "bearish", 0.6, (cur * 0.998, cur * 1.002), second_half_l * 0.98, second_half_h * 1.005, n)
    return None


def detect_patterns(candles: List[List[float]]) -> List[PatternResult]:
    """
    Run all pattern detectors. Returns list of detected patterns (best first).
    """
    if not candles or len(candles) < 20:
        return []
    closes = _closes(candles)
    highs = _highs(candles)
    lows = _lows(candles)
    if len(closes) < 20:
        return []
    results = []
    for detector in [
        _detect_head_shoulders,
        _detect_inverse_head_shoulders,
        _detect_double_top,
        _detect_double_bottom,
    ]:
        try:
            r = detector(closes)
            if r and r.confidence >= 0.5:
                results.append(r)
        except Exception as e:
            logger.debug("Pattern detector %s: %s", detector.__name__, e)
    try:
        r = _detect_flag(closes, highs, lows)
        if r:
            results.append(r)
    except Exception:
        pass
    try:
        r = _detect_triangle(closes, highs, lows)
        if r:
            results.append(r)
    except Exception:
        pass
    results.sort(key=lambda x: -x.confidence)
    return results


def pattern_score_boost(patterns: List[PatternResult], direction: str = "bullish") -> float:
    """
    Return 0-15 score boost for recommendation when patterns align with direction.
    direction: "bullish" or "bearish"
    """
    if not patterns:
        return 0.0
    best = patterns[0]
    if best.direction != direction:
        return -5.0
    return min(15.0, best.confidence * 20)
