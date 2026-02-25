"""
Phase 4B: Momentum Ranking System
Rank symbols by momentum strength; trade top rankers.
"""
import logging
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


def _fetch_candles(symbol: str, timeframe: str = "1d", periods: int = 60) -> List[List[float]]:
    try:
        from phase2_data_fetcher import fetch_recent_candles
        return fetch_recent_candles(symbol, timeframe, periods) or []
    except Exception as e:
        logger.debug("momentum_ranking fetch %s: %s", symbol, e)
        return []


def _closes(candles: List[List[float]]) -> List[float]:
    return [float(c[4]) for c in candles if len(c) >= 5]


class MomentumRanker:
    """
    Ranks symbols by momentum (returns over multiple windows).
    Combines 5d, 20d, 60d momentum for robust ranking.
    """

    def __init__(self, weights: Optional[Dict[str, float]] = None) -> None:
        self.weights = weights or {"5d": 0.3, "20d": 0.4, "60d": 0.3}

    def calculate_momentum_score(self, symbol: str) -> Dict[str, Any]:
        """
        Calculate momentum score for a symbol.

        Returns:
        {
            'score': float 0-100,
            'rank': int (if in universe),
            'mom_5d': float,
            'mom_20d': float,
            'mom_60d': float,
            'composite': float,
            'reasoning': str,
        }
        """
        candles = _fetch_candles(symbol, "1d", 65)
        closes = _closes(candles)
        if len(closes) < 60:
            return {"score": 50.0, "mom_5d": 0, "mom_20d": 0, "mom_60d": 0, "composite": 0, "reasoning": "Insufficient data"}

        mom_5d = (closes[-1] - closes[-6]) / closes[-6] if len(closes) >= 6 and closes[-6] else 0
        mom_20d = (closes[-1] - closes[-21]) / closes[-21] if len(closes) >= 21 and closes[-21] else 0
        mom_60d = (closes[-1] - closes[-61]) / closes[-61] if len(closes) >= 61 and closes[-61] else 0

        composite = (
            mom_5d * self.weights["5d"]
            + mom_20d * self.weights["20d"]
            + mom_60d * self.weights["60d"]
        )
        # Map to 0-100 (assume -20% to +20% typical range)
        score = 50 + composite * 250
        score = max(0, min(100, score))

        return {
            "score": round(score, 1),
            "mom_5d": round(mom_5d * 100, 2),
            "mom_20d": round(mom_20d * 100, 2),
            "mom_60d": round(mom_60d * 100, 2),
            "composite": round(composite * 100, 2),
            "reasoning": f"5d:{mom_5d*100:.1f}% 20d:{mom_20d*100:.1f}% 60d:{mom_60d*100:.1f}%",
        }

    def rank_universe(self, symbols: List[str]) -> List[Dict[str, Any]]:
        """
        Rank a universe of symbols by momentum.
        Returns list of {symbol, score, ...} sorted by score descending.
        """
        results = []
        for sym in symbols:
            r = self.calculate_momentum_score(sym)
            r["symbol"] = sym
            results.append(r)
        results.sort(key=lambda x: -x["score"])
        for i, r in enumerate(results):
            r["rank"] = i + 1
        return results
