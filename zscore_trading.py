"""
Phase 4B: Mean Reversion Z-Score Trading
Trade extremes: when price deviates significantly from mean, expect reversion.
"""
import logging
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


def _closes(candles: List[List[float]]) -> List[float]:
    return [float(c[4]) for c in candles if len(c) >= 5]


class ZScoreTrader:
    """
    Mean reversion using Z-scores.
    Z = (price - mean) / std
    |Z| > 2 = extreme, expect reversion.
    """

    def __init__(self, window: int = 20, z_threshold: float = 2.0) -> None:
        self.window = window
        self.z_threshold = z_threshold

    def get_zscore_signal(
        self, candles: List[List[float]], symbol: str = ""
    ) -> Dict[str, Any]:
        """
        Get Z-score mean reversion signal.

        Returns:
        {
            'signal': 'buy' | 'sell' | 'neutral',
            'z_score': float,
            'confidence': 0-1,
            'mean': float,
            'std': float,
            'reasoning': str,
        }
        """
        closes = _closes(candles)
        if len(closes) < self.window + 5:
            return {
                "signal": "neutral",
                "z_score": 0.0,
                "confidence": 0.0,
                "mean": 0.0,
                "std": 0.0,
                "reasoning": "Insufficient data",
            }

        recent = closes[-self.window:]
        mean = sum(recent) / len(recent)
        variance = sum((x - mean) ** 2 for x in recent) / len(recent)
        std = variance ** 0.5 if variance > 0 else 1e-9
        current = closes[-1]
        z_score = (current - mean) / std if std > 0 else 0.0

        if z_score < -self.z_threshold:
            conf = min(0.9, 0.5 + abs(z_score) * 0.15)
            return {
                "signal": "buy",
                "z_score": z_score,
                "confidence": conf,
                "mean": mean,
                "std": std,
                "reasoning": f"Oversold Z={z_score:.2f} (mean reversion buy)",
            }
        if z_score > self.z_threshold:
            conf = min(0.9, 0.5 + abs(z_score) * 0.15)
            return {
                "signal": "sell",
                "z_score": z_score,
                "confidence": conf,
                "mean": mean,
                "std": std,
                "reasoning": f"Overbought Z={z_score:.2f} (mean reversion sell)",
            }
        return {
            "signal": "neutral",
            "z_score": z_score,
            "confidence": 0.3,
            "mean": mean,
            "std": std,
            "reasoning": f"Z={z_score:.2f} within range",
        }
