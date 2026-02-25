"""
Statistical arbitrage through pairs trading. (12.md Part 1)
Profits from mean reversion of correlated asset spreads.
"""
import logging
import os
from datetime import datetime
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)
ENABLED = os.getenv("ENABLE_PAIRS_TRADING", "0").strip().lower() in ("1", "true", "yes")


def _get_price_history(symbol: str, days: int = 60):
    """Get historical daily closes as list."""
    try:
        from phase2_data_fetcher import fetch_recent_candles
        bars = days
        candles = fetch_recent_candles(symbol, timeframe="1d", periods=bars)
        if candles and len(candles) >= 10:
            return [float(c[4]) for c in candles]  # close
    except Exception as e:
        logger.debug("pairs_trading _get_price_history %s: %s", symbol, e)
    return None


def _adf_test(series) -> Dict:
    """Augmented Dickey-Fuller test for stationarity."""
    try:
        from statsmodels.tsa.stattools import adfuller
        import pandas as pd
        s = pd.Series(series)
        result = adfuller(s.dropna())
        return {"pvalue": result[1], "is_stationary": result[1] < 0.05}
    except ImportError:
        return {"pvalue": 0.1, "is_stationary": False}
    except Exception as e:
        logger.debug("adf_test: %s", e)
        return {"pvalue": 0.1, "is_stationary": False}


def _build_pairs_universe() -> List[Tuple[str, str]]:
    crypto = [("XBT/USD", "ETH/USD"), ("ETH/USD", "SOL/USD")]
    stock = [("AAPL", "MSFT"), ("GOOGL", "META"), ("JPM", "BAC")]
    return crypto + stock


def find_trading_opportunities() -> List[Dict]:
    """Scan pairs for arbitrage opportunities."""
    if not ENABLED:
        return []
    try:
        import numpy as np
        opportunities = []
        for a1, a2 in _build_pairs_universe():
            p1 = _get_price_history(a1, 60)
            p2 = _get_price_history(a2, 60)
            if not p1 or not p2 or len(p1) < 20 or len(p2) < 20:
                continue
            arr1 = np.array(p1)
            arr2 = np.array(p2)
            min_len = min(len(arr1), len(arr2))
            arr1, arr2 = arr1[-min_len:], arr2[-min_len:]
            corr = np.corrcoef(arr1, arr2)[0, 1] if min_len > 1 else 0
            if np.isnan(corr) or corr < 0.7:
                continue
            beta, alpha = np.polyfit(arr2, arr1, 1)
            spread = arr1 - (beta * arr2 + alpha)
            adf = _adf_test(spread.tolist())
            if not adf.get("is_stationary", False):
                continue
            mean_s = float(np.mean(spread))
            std_s = float(np.std(spread)) or 1e-6
            current_s = float(spread[-1])
            z = (current_s - mean_s) / std_s
            if abs(z) < 2.0:
                continue
            opportunities.append({
                "pair": f"{a1}/{a2}",
                "asset1": a1, "asset2": a2,
                "z_score": float(z),
                "correlation": float(corr),
                "direction": "short_spread" if z > 2 else "long_spread",
                "expected_profit_pct": 6.0 if abs(z) >= 2.5 else 4.0,
            })
        opportunities.sort(key=lambda x: abs(x["z_score"]), reverse=True)
        return opportunities[:5]
    except Exception as e:
        logger.exception("pairs_trading find_opportunities: %s", e)
        return []
