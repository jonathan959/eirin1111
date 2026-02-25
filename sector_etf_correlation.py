"""
Sector ETF correlation - filter stock entries by sector ETF trend.

Don't buy AAPL if XLK (tech ETF) is in downtrend.
"""
import logging
import os
from typing import Any, Dict, List, Optional, Tuple

from stock_metadata import get_sector, STOCK_SECTOR_MAP

logger = logging.getLogger(__name__)

# Sector -> primary sector ETF
SECTOR_ETF_MAP: Dict[str, str] = {
    "Technology": "XLK",
    "Financial": "XLF",
    "Healthcare": "XLV",
    "Consumer Cyclical": "XLY",
    "Consumer Defensive": "XLP",
    "Energy": "XLE",
    "Industrial": "XLI",
    "Communication": "XLC",
    "Utilities": "XLU",
    "ETF": "",  # ETFs skip sector trend check
}


def get_sector_etf(symbol: str) -> Optional[str]:
    """Return sector ETF for symbol, or None (empty string means skip)."""
    sector = get_sector(symbol)
    if not sector:
        return None
    etf = SECTOR_ETF_MAP.get(sector)
    return etf if etf else None


def sector_etf_trend_ok(
    symbol: str,
    candles_1d: Optional[List] = None,
    get_ohlcv_fn=None,
) -> Tuple[bool, str]:
    """
    Check if sector ETF is in uptrend. Don't buy stock if sector ETF downtrend.
    Returns (ok, reason). ok=False means block entry.
    """
    etf = get_sector_etf(symbol)
    if not etf:
        return True, ""
    if not get_ohlcv_fn and not candles_1d:
        return True, ""
    try:
        if candles_1d is None and get_ohlcv_fn:
            candles_1d = get_ohlcv_fn(etf, "1d", 60)
        if not candles_1d or len(candles_1d) < 50:
            return True, ""
        closes = [float(c[4]) for c in candles_1d if len(c) >= 5]
        if len(closes) < 50:
            return True, ""
        sma20 = sum(closes[-20:]) / 20
        sma50 = sum(closes[-50:]) / 50
        last = closes[-1]
        if sma20 < sma50 * 0.98:
            return False, f"Sector ETF {etf} in downtrend - avoid {symbol}"
        return True, f"Sector {etf} trend OK"
    except Exception as e:
        logger.debug("Sector ETF trend %s: %s", symbol, e)
        return True, ""
