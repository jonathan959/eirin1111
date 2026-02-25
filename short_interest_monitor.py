"""
Short interest monitor - short squeeze detector.

- High short interest + positive catalyst = squeeze potential
- Alert: "GME short interest 40% - squeeze risk/opportunity"
"""
import logging
import os
from typing import Any, Dict, Optional, Tuple

logger = logging.getLogger(__name__)

# Finnhub returns days-to-cover ratio. >2.0 = high squeeze risk/opportunity.
SHORT_SQUEEZE_THRESHOLD = float(os.getenv("SHORT_SQUEEZE_THRESHOLD", "2.0"))
FINNHUB_API_KEY = os.getenv("FINNHUB_API_KEY", os.getenv("FINNHUB_TOKEN", "")).strip()


def get_short_interest(symbol: str) -> Optional[Dict[str, Any]]:
    """
    Fetch short interest from Finnhub. Returns ratio, shares short, etc.
    """
    if not FINNHUB_API_KEY:
        return None
    try:
        import requests
        sym = symbol.upper().split("/")[0]
        r = requests.get(
            "https://finnhub.io/api/v1/stock/short-interest",
            params={"symbol": sym, "token": FINNHUB_API_KEY},
            timeout=5,
        )
        if r.status_code != 200:
            return None
        data = r.json()
        if not data or not isinstance(data, list):
            return None
        latest = data[0] if data else {}
        # Finnhub: ratio typically = days-to-cover (short interest / avg daily volume)
        ratio = float(latest.get("shortInterestRatio", latest.get("ratio", 0)) or 0)
        return {
            "short_interest_ratio": ratio,
            "days_to_cover": ratio,
            "squeeze_risk": ratio >= SHORT_SQUEEZE_THRESHOLD,
        }
    except Exception as e:
        logger.debug("Short interest %s: %s", symbol, e)
        return None


def short_squeeze_alert(symbol: str, score: float = 50) -> Optional[str]:
    """
    If high short interest (days-to-cover) + bullish score, return squeeze opportunity alert.
    """
    si = get_short_interest(symbol)
    if not si or not si.get("squeeze_risk"):
        return None
    dtc = si.get("days_to_cover", si.get("short_interest_ratio", 0))
    if score >= 55:
        return f"{symbol} short interest {dtc:.1f} days to cover - squeeze opportunity"
    return f"{symbol} short interest {dtc:.1f} days to cover - squeeze risk"
