"""
Phase 4: Correlation Trading
Trade based on historical correlations between assets.
Example: Gold ↑ → Miners ↑↑ (amplified move)
"""
import logging
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


def _fetch_candles(symbol: str, timeframe: str = "1d", periods: int = 10) -> List[List[float]]:
    """Fetch candles via phase2_data_fetcher or market_data."""
    try:
        from phase2_data_fetcher import fetch_recent_candles
        return fetch_recent_candles(symbol, timeframe, periods) or []
    except Exception as e:
        logger.debug("correlation_trading fetch %s: %s", symbol, e)
        return []


def _close_prices(candles: List[List[float]]) -> List[float]:
    """Extract close prices. Candles: [[ts, o, h, l, c, v], ...]."""
    return [float(c[4]) for c in candles if len(c) >= 5]


class CorrelationTrader:
    """
    Identifies and trades correlation opportunities.
    """

    def __init__(self) -> None:
        # Define known correlations
        self.correlations: Dict[str, Dict[str, float]] = {
            # Commodities → Related stocks
            "GLD": {"GDX": 1.5, "GOLD": 1.3},
            "USO": {"XLE": 0.8, "JETS": -0.6},
            # Crypto → Altcoins
            "BTC/USD": {"ETH/USD": 1.3, "SOL/USD": 1.8},
            "XBT/USD": {"ETH/USD": 1.3, "SOL/USD": 1.8},
            # Indices → Sectors
            "SPY": {"QQQ": 1.1, "IWM": 0.9},
            # Bonds → Stocks (inverse)
            "TLT": {"SPY": -0.5},
        }

    def find_opportunities(self, days: int = 5) -> List[Dict[str, Any]]:
        """
        Scan for correlation opportunities.

        Logic:
        1. Check if leading asset moved significantly
        2. Check if correlated asset lagged
        3. Trade the lag (catch-up trade)
        """
        opportunities: List[Dict[str, Any]] = []

        for leader, followers in self.correlations.items():
            leader_move = self._get_recent_move(leader, days=days)
            if abs(leader_move) < 0.05:
                continue

            for follower, beta in followers.items():
                follower_move = self._get_recent_move(follower, days=days)
                expected_move = leader_move * beta
                lag = expected_move - follower_move

                if abs(lag) > 0.03:
                    direction = "buy" if lag > 0 else "sell"
                    opportunities.append({
                        "symbol": follower,
                        "direction": direction,
                        "leader": leader,
                        "leader_move": leader_move,
                        "expected_move": expected_move,
                        "actual_move": follower_move,
                        "lag": lag,
                        "target_move": lag * 0.8,
                        "reasoning": f"{follower} lagging {leader} by {lag*100:.1f}%",
                    })
        return opportunities

    def _get_recent_move(self, symbol: str, days: int = 5) -> float:
        """Get % move over last N days."""
        periods = min(max(days * 2, 10), 60)
        candles = _fetch_candles(symbol, timeframe="1d", periods=periods)
        closes = _close_prices(candles)
        if len(closes) < 2:
            return 0.0
        start_price = closes[0]
        end_price = closes[-1]
        if start_price <= 0:
            return 0.0
        return (end_price - start_price) / start_price
