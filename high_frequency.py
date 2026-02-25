"""
Phase 4C: High Frequency / Order Book Microstructure
Order book analysis for short-term edge.
Note: Full HFT requires tick data; we use order book snapshot when available.
"""
import os
import logging
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

ENABLE_HIGH_FREQUENCY = os.getenv("ENABLE_HIGH_FREQUENCY", "0").strip().lower() in (
    "1", "true", "yes", "y", "on",
)


def _fetch_order_book(symbol: str, depth: int = 20) -> Optional[Dict[str, Any]]:
    try:
        from phase2_data_fetcher import fetch_order_book
        return fetch_order_book(symbol, depth)
    except Exception as e:
        logger.debug("high_frequency fetch_order_book %s: %s", symbol, e)
        return None


class OrderBookAnalyzer:
    """
    Order book microstructure analysis.
    - Bid/ask imbalance
    - Spread analysis
    - Depth at various levels
    """

    def __init__(self) -> None:
        pass

    def analyze_order_book(self, symbol: str) -> Dict[str, Any]:
        """
        Analyze order book for microstructure signals.

        Returns:
        {
            'imbalance': float -1 to 1 (negative=ask heavy, positive=bid heavy),
            'spread_bps': float,
            'signal': 'bullish'|'bearish'|'neutral',
            'confidence': 0-1,
            'reasoning': str,
        }
        """
        if not ENABLE_HIGH_FREQUENCY:
            return {"imbalance": 0.0, "spread_bps": 0.0, "signal": "neutral", "confidence": 0.0, "reasoning": "Disabled"}

        ob = _fetch_order_book(symbol)
        if not ob or not ob.get("bids") or not ob.get("asks"):
            return {"imbalance": 0.0, "spread_bps": 0.0, "signal": "neutral", "confidence": 0.0, "reasoning": "No order book"}

        bids = ob.get("bids", [])[:10]
        asks = ob.get("asks", [])[:10]
        bid_vol = sum(float(b[1]) for b in bids) if bids else 0
        ask_vol = sum(float(a[1]) for a in asks) if asks else 0
        total = bid_vol + ask_vol
        imbalance = (bid_vol - ask_vol) / total if total > 0 else 0.0

        best_bid = float(bids[0][0]) if bids else 0
        best_ask = float(asks[0][0]) if asks else 0
        mid = (best_bid + best_ask) / 2 if best_bid and best_ask else 0
        spread_bps = ((best_ask - best_bid) / mid * 10000) if mid else 0

        if imbalance > 0.2:
            signal = "bullish"
            conf = min(0.8, 0.5 + imbalance)
        elif imbalance < -0.2:
            signal = "bearish"
            conf = min(0.8, 0.5 + abs(imbalance))
        else:
            signal = "neutral"
            conf = 0.3

        return {
            "imbalance": round(imbalance, 3),
            "spread_bps": round(spread_bps, 2),
            "signal": signal,
            "confidence": conf,
            "reasoning": f"Order book imbalance {imbalance:.2f}",
        }
