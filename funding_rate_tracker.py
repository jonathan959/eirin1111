"""
Perpetual futures funding rate tracking.

- High positive funding: market overleveraged long (bearish signal)
- High negative funding: market overleveraged short (bullish signal)
- Integrate into regime detection
"""
import logging
import os
import time
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

TRACK_FUNDING_RATES = os.getenv("TRACK_FUNDING_RATES", "1").strip().lower() in ("1", "true", "yes")
FUNDING_BEARISH_THRESHOLD = float(os.getenv("FUNDING_BEARISH_THRESHOLD", "0.01"))
FUNDING_BULLISH_THRESHOLD = float(os.getenv("FUNDING_BULLISH_THRESHOLD", "-0.01"))


def get_funding_rate(symbol: str) -> Optional[Dict[str, Any]]:
    """
    Get current perpetual funding rate. Symbol: BTC/USD, ETH/USD, etc.
    Uses Binance fapi (no key required).
    Returns: { rate, rate_8h_apr, signal, next_funding_ts }
    signal: "bearish" | "bullish" | "neutral"
    """
    if not TRACK_FUNDING_RATES:
        return None
    try:
        import requests
        sym = (symbol.split("/")[0] or symbol).upper()
        if sym == "XBT":
            sym = "BTC"
        binance_sym = f"{sym}USDT"
        url = "https://fapi.binance.com/fapi/v1/premiumIndex"
        r = requests.get(url, params={"symbol": binance_sym}, timeout=5)
        if r.status_code != 200:
            return None
        data = r.json()
        rate = float(data.get("lastFundingRate", 0) or 0)
        next_ts = int(data.get("nextFundingTime", 0) or 0)
        rate_8h_apr = rate * 3 * 365 * 100 if rate else 0
        if rate >= FUNDING_BEARISH_THRESHOLD:
            signal = "bearish"
        elif rate <= FUNDING_BULLISH_THRESHOLD:
            signal = "bullish"
        else:
            signal = "neutral"
        return {
            "rate": rate,
            "rate_pct": round(rate * 100, 4),
            "rate_8h_apr_pct": round(rate_8h_apr, 2),
            "signal": signal,
            "next_funding_ts": next_ts,
        }
    except Exception as e:
        logger.debug("Funding rate fetch failed %s: %s", symbol, e)
        return None


def funding_signal_for_regime(symbol: str) -> Optional[str]:
    """Returns 'bearish' | 'bullish' | None for regime integration."""
    fr = get_funding_rate(symbol)
    return fr.get("signal") if fr and fr.get("signal") != "neutral" else None
