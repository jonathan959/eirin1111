"""
Stock profile - market cap, IPO date, basic company info for filters.

Used for: MIN_MARKET_CAP, MAX_MARKET_CAP, AVOID_RECENT_IPOS
"""
import logging
import os
from datetime import datetime, timezone, timedelta
from typing import Any, Dict, Optional, Tuple

logger = logging.getLogger(__name__)

MIN_MARKET_CAP = float(os.getenv("MIN_MARKET_CAP", "0"))  # $0 = no filter
MAX_MARKET_CAP = float(os.getenv("MAX_MARKET_CAP", "0"))  # $0 = no filter
AVOID_RECENT_IPOS = os.getenv("AVOID_RECENT_IPOS", "1").strip().lower() in ("1", "true", "yes")
IPO_DAYS_THRESHOLD = int(os.getenv("IPO_DAYS_THRESHOLD", "90"))
FINNHUB_API_KEY = os.getenv("FINNHUB_API_KEY", os.getenv("FINNHUB_TOKEN", "")).strip()

# Market cap tiers (in billions)
TIER_LARGE = 10.0   # >$10B
TIER_MID = 2.0      # $2-10B
TIER_SMALL = 0.0    # <$2B


def get_stock_profile(symbol: str) -> Dict[str, Any]:
    """
    Fetch stock profile from Finnhub: market cap, IPO date, sector.
    Returns { market_cap, market_cap_b, ipo_date, ipo_days_ago, days_listed, tier }.
    """
    out: Dict[str, Any] = {"market_cap": None, "market_cap_b": 0.0, "ipo_date": None, "ipo_days_ago": None, "days_listed": None, "tier": "unknown"}
    if not FINNHUB_API_KEY:
        return out
    try:
        import requests
        sym = symbol.upper().split("/")[0]
        r = requests.get(
            "https://finnhub.io/api/v1/stock/profile2",
            params={"symbol": sym, "token": FINNHUB_API_KEY},
            timeout=5,
        )
        if r.status_code != 200:
            return out
        data = r.json()
        if not data or not isinstance(data, dict):
            return out
        mc = float(data.get("marketCapitalization") or 0)
        out["market_cap"] = mc
        out["market_cap_b"] = mc / 1e9 if mc else 0.0
        ipo_str = data.get("ipo") or ""
        if ipo_str:
            try:
                dt = datetime.strptime(ipo_str[:10], "%Y-%m-%d").replace(tzinfo=timezone.utc)
                out["ipo_date"] = ipo_str[:10]
                delta = datetime.now(timezone.utc) - dt
                out["ipo_days_ago"] = delta.days
                out["days_listed"] = delta.days
            except ValueError:
                pass
        if out["market_cap_b"] >= TIER_LARGE:
            out["tier"] = "large"
        elif out["market_cap_b"] >= TIER_MID:
            out["tier"] = "mid"
        elif out["market_cap_b"] > 0:
            out["tier"] = "small"
        return out
    except Exception as e:
        logger.debug("Stock profile %s: %s", symbol, e)
        return out


def passes_market_cap_filter(symbol: str) -> Tuple[bool, str]:
    """Check MIN/MAX_MARKET_CAP. Returns (ok, reason)."""
    if MIN_MARKET_CAP <= 0 and MAX_MARKET_CAP <= 0:
        return True, ""
    p = get_stock_profile(symbol)
    mc = p.get("market_cap") or 0
    if not mc:
        return True, ""  # Unknown = don't block
    if MIN_MARKET_CAP > 0 and mc < MIN_MARKET_CAP:
        return False, f"Market cap ${mc/1e9:.1f}B below MIN_MARKET_CAP ${MIN_MARKET_CAP/1e9:.0f}B"
    if MAX_MARKET_CAP > 0 and mc > MAX_MARKET_CAP:
        return False, f"Market cap ${mc/1e9:.1f}B above MAX_MARKET_CAP ${MAX_MARKET_CAP/1e9:.0f}B"
    return True, ""


def is_recent_ipo(symbol: str) -> Tuple[bool, Optional[int]]:
    """Returns (is_recent_ipo, days_listed)."""
    if not AVOID_RECENT_IPOS:
        return False, None
    p = get_stock_profile(symbol)
    days = p.get("days_listed") or p.get("ipo_days_ago")
    if days is None:
        return False, None
    return days < IPO_DAYS_THRESHOLD, days
