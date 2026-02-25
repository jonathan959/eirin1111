"""
Volume Profile Analysis. (12.md Part 3)
Identifies high-volume price levels (POC, value area).
"""
import logging
import os
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)
ENABLED = os.getenv("ENABLE_VOLUME_PROFILE", "0").strip().lower() in ("1", "true", "yes")


def _fetch_candles(symbol: str, timeframe: str = "1h", periods: int = 720) -> Optional[List]:
    """Fetch OHLCV candles."""
    try:
        from phase2_data_fetcher import fetch_recent_candles
        c = fetch_recent_candles(symbol, timeframe=timeframe, periods=periods)
        return c if c else None
    except Exception as e:
        logger.debug("volume_profile fetch %s: %s", symbol, e)
        return None


def calculate_volume_profile(
    symbol: str,
    lookback_days: int = 30,
    num_bins: int = 50,
) -> Optional[Dict]:
    """Calculate volume profile: POC, value area, high/low volume nodes."""
    if not ENABLED:
        return None
    try:
        import numpy as np
        candles = _fetch_candles(symbol, "1h", min(720, lookback_days * 24))
        if not candles or len(candles) < 50:
            return None
        lows = [float(c[3]) for c in candles]
        highs = [float(c[2]) for c in candles]
        closes = [float(c[4]) for c in candles]
        volumes = [float(c[5]) if len(c) > 5 else 1.0 for c in candles]
        price_min, price_max = min(lows), max(highs)
        if price_max <= price_min:
            return None
        bins = np.linspace(price_min, price_max, num_bins + 1)
        vol_by_price = np.zeros(num_bins)
        for i, c in enumerate(candles):
            lo, hi, vol = lows[i], highs[i], volumes[i]
            for j in range(num_bins):
                bl, bh = bins[j], bins[j + 1]
                ol = max(lo, bl)
                oh = min(hi, bh)
                if oh > ol:
                    vol_by_price[j] += vol * (oh - ol) / (hi - lo) if hi > lo else vol / num_bins
        poc_idx = int(np.argmax(vol_by_price))
        poc = float((bins[poc_idx] + bins[poc_idx + 1]) / 2)
        total_vol = float(np.sum(vol_by_price)) or 1.0
        va_vol = total_vol * 0.7
        lo_idx = poc_idx
        hi_idx = poc_idx
        acc = float(vol_by_price[poc_idx])
        while acc < va_vol and (lo_idx > 0 or hi_idx < num_bins - 1):
            vb = vol_by_price[lo_idx - 1] if lo_idx > 0 else 0
            va = vol_by_price[hi_idx + 1] if hi_idx < num_bins - 1 else 0
            if va >= vb and hi_idx < num_bins - 1:
                hi_idx += 1
                acc += float(vol_by_price[hi_idx])
            elif lo_idx > 0:
                lo_idx -= 1
                acc += float(vol_by_price[lo_idx])
            else:
                break
        va_lo = float((bins[lo_idx] + bins[lo_idx + 1]) / 2)
        va_hi = float((bins[hi_idx] + bins[hi_idx + 1]) / 2)
        current = closes[-1]
        if current > va_hi:
            pos, sig = "above_value_area", "overbought"
        elif current < va_lo:
            pos, sig = "below_value_area", "oversold"
        else:
            pos, sig = "within_value_area", "neutral"
        return {
            "symbol": symbol,
            "poc": poc,
            "value_area_high": va_hi,
            "value_area_low": va_lo,
            "current_price": current,
            "current_position": pos,
            "signal": sig,
        }
    except Exception as e:
        logger.exception("volume_profile: %s", e)
        return None


def generate_trading_signals(profile: Dict) -> List[Dict]:
    """Generate trading signals from volume profile."""
    if not profile:
        return []
    signals = []
    cp = profile.get("current_price", 0)
    poc = profile.get("poc", 0)
    va_lo = profile.get("value_area_low", 0)
    va_hi = profile.get("value_area_high", 0)
    if poc <= 0 or cp <= 0:
        return []
    if abs(cp - poc) / poc < 0.02:
        signals.append({"type": "mean_reversion", "direction": "neutral", "reasoning": "Price at POC (fair value)"})
    elif profile.get("current_position") == "below_value_area" and va_lo > 0:
        dist = (va_lo - cp) / cp
        if dist > 0.05:
            signals.append({"type": "oversold", "direction": "long", "reasoning": f"Price {dist*100:.1f}% below value area"})
    elif profile.get("current_position") == "above_value_area" and va_hi > 0:
        dist = (cp - va_hi) / cp
        if dist > 0.05:
            signals.append({"type": "overbought", "direction": "short", "reasoning": f"Price {dist*100:.1f}% above value area"})
    return signals
