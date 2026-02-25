"""
Phase 4B: Earnings Momentum Trading
Post-earnings drift - stocks tend to drift in earnings direction for ~60 days.
"""
import os
import logging
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

ENABLE_EARNINGS_MOMENTUM = os.getenv("ENABLE_EARNINGS_MOMENTUM", "0").strip().lower() in (
    "1", "true", "yes", "y", "on",
)


def _fetch_candles(symbol: str, timeframe: str = "1d", periods: int = 90) -> List[List[float]]:
    """Fetch candles via phase2_data_fetcher."""
    try:
        from phase2_data_fetcher import fetch_recent_candles
        return fetch_recent_candles(symbol, timeframe, periods) or []
    except Exception as e:
        logger.debug("earnings_momentum fetch %s: %s", symbol, e)
        return []


def _closes(candles: List[List[float]]) -> List[float]:
    return [float(c[4]) for c in candles if len(c) >= 5]


class EarningsMomentumTrader:
    """
    Trades post-earnings drift.
    Stocks that beat earnings tend to outperform for ~60 days (drift).
    Uses earnings calendar + price action to detect drift.
    """

    def __init__(self, lookback_days: int = 60) -> None:
        self.lookback_days = lookback_days

    def get_earnings_signal(
        self, symbol: str, days_since_earnings: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Get earnings momentum signal.

        Returns:
        {
            'signal': 'bullish' | 'bearish' | 'neutral',
            'confidence': 0-1,
            'days_since_earnings': int or None,
            'drift_pct': float,
            'reasoning': str,
        }
        """
        if not ENABLE_EARNINGS_MOMENTUM:
            return {"signal": "neutral", "confidence": 0.0, "reasoning": "Earnings momentum disabled"}

        try:
            from earnings_calendar import days_until_earnings
            de = days_until_earnings(symbol)
            if de is None:
                return {"signal": "neutral", "confidence": 0.0, "reasoning": "No earnings data"}
            # Negative = past earnings
            if de > 0:
                return {"signal": "neutral", "confidence": 0.0, "reasoning": f"Earnings in {de} days (pre-event)"}
            days_since = abs(de)
        except Exception as e:
            logger.debug("Earnings calendar: %s", e)
            days_since = days_since_earnings or 0

        if days_since > self.lookback_days:
            return {"signal": "neutral", "confidence": 0.0, "reasoning": "Beyond drift window"}

        candles = _fetch_candles(symbol, "1d", self.lookback_days + 10)
        closes = _closes(candles)
        if len(closes) < 20:
            return {"signal": "neutral", "confidence": 0.0, "reasoning": "Insufficient data"}

        # Post-earnings return (earnings day = 0, measure 5-30 day drift)
        earn_idx = max(0, len(closes) - days_since - 1)
        if earn_idx + 30 > len(closes):
            return {"signal": "neutral", "confidence": 0.0, "reasoning": "Drift period incomplete"}

        price_at_earnings = closes[earn_idx]
        price_5d = closes[min(earn_idx + 5, len(closes) - 1)]
        price_20d = closes[min(earn_idx + 20, len(closes) - 1)]

        drift_5d = (price_5d - price_at_earnings) / price_at_earnings if price_at_earnings else 0
        drift_20d = (price_20d - price_at_earnings) / price_at_earnings if price_at_earnings else 0

        # Strong positive drift = bullish continuation
        if drift_20d > 0.05 and drift_5d > 0.02:
            return {
                "signal": "bullish",
                "confidence": min(0.9, 0.5 + abs(drift_20d) * 5),
                "days_since_earnings": days_since,
                "drift_pct": drift_20d * 100,
                "reasoning": f"Post-earnings drift +{drift_20d*100:.1f}% over 20 days",
            }
        if drift_20d < -0.05 and drift_5d < -0.02:
            return {
                "signal": "bearish",
                "confidence": min(0.9, 0.5 + abs(drift_20d) * 5),
                "days_since_earnings": days_since,
                "drift_pct": drift_20d * 100,
                "reasoning": f"Post-earnings drift {drift_20d*100:.1f}% over 20 days",
            }
        return {
            "signal": "neutral",
            "confidence": 0.3,
            "days_since_earnings": days_since,
            "drift_pct": drift_20d * 100,
            "reasoning": f"Mixed drift {drift_20d*100:.1f}%",
        }
