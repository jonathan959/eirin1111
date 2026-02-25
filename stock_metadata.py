"""
Stock metadata for intelligence scoring.
Sector mapping for common symbols; extend via env or API integration (e.g. Polygon).
"""

from typing import Dict, Optional

# Sector mapping for common stocks (extend as needed; Polygon/Finnhub can provide dynamically)
STOCK_SECTOR_MAP: Dict[str, str] = {
    "AAPL": "Technology", "MSFT": "Technology", "GOOGL": "Technology", "GOOG": "Technology",
    "AMZN": "Consumer Cyclical", "NVDA": "Technology", "META": "Technology", "TSLA": "Consumer Cyclical",
    "AMD": "Technology", "NFLX": "Communication", "INTC": "Technology", "QCOM": "Technology",
    "CRM": "Technology", "ADBE": "Technology", "AVGO": "Technology", "TXN": "Technology",
    "JPM": "Financial", "BAC": "Financial", "V": "Financial", "MA": "Financial",
    "WFC": "Financial", "GS": "Financial", "MS": "Financial", "BLK": "Financial",
    "C": "Financial", "AXP": "Financial",
    "SPY": "ETF", "QQQ": "ETF", "IWM": "ETF", "DIA": "ETF",
    "XLK": "ETF", "XLF": "ETF", "XLE": "ETF", "XLV": "ETF", "XLY": "ETF",
    "COIN": "Financial", "MSTR": "Technology", "MARA": "Technology", "RIOT": "Technology",
    "GME": "Consumer Cyclical", "AMC": "Communication", "PLTR": "Technology",
    "JNJ": "Healthcare", "PG": "Consumer Defensive", "KO": "Consumer Defensive",
    "PEP": "Consumer Defensive", "WMT": "Consumer Defensive", "COST": "Consumer Defensive",
    "UNH": "Healthcare", "PFE": "Healthcare", "MRK": "Healthcare",
}


def get_sector(symbol: str) -> Optional[str]:
    """Return sector for symbol, or None if unknown."""
    return STOCK_SECTOR_MAP.get(str(symbol).strip().upper())


def get_liquidity_tier(price: float, volume_24h: float) -> str:
    """
    Infer liquidity tier from price * volume (proxy for market cap / liquidity).
    Returns: "mega", "large", "mid", "small"
    """
    if not price or not volume_24h or price <= 0:
        return "unknown"
    notional = price * volume_24h
    if notional >= 1e11:  # $100B+
        return "mega"
    if notional >= 1e10:  # $10B+
        return "large"
    if notional >= 1e9:   # $1B+
        return "mid"
    return "small"
