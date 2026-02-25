"""
Analyst ratings tracker - upgrades/downgrades from major analysts.

- Buy signal: upgrade from neutral to buy
- Sell signal: downgrade from buy to hold/sell
- Weight analyst ratings in recommendation score
"""
import logging
import os
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

TRACK_ANALYST_RATINGS = os.getenv("TRACK_ANALYST_RATINGS", "1").strip().lower() in ("1", "true", "yes")
ANALYST_WEIGHT = float(os.getenv("ANALYST_WEIGHT", "0.15"))
FINNHUB_API_KEY = os.getenv("FINNHUB_API_KEY", os.getenv("FINNHUB_TOKEN", "")).strip()


def get_analyst_ratings(symbol: str) -> Dict[str, Any]:
    """
    Fetch Finnhub analyst recommendations. Returns consensus and trend.
    score_delta: -1 (downgrade) to +1 (upgrade)
    consensus: "strong_buy" | "buy" | "hold" | "sell" | "strong_sell"
    """
    if not TRACK_ANALYST_RATINGS or not FINNHUB_API_KEY:
        return {"score_delta": 0, "consensus": "hold", "enabled": False}
    try:
        import requests
        sym = symbol.upper().split("/")[0]
        r = requests.get(
            "https://finnhub.io/api/v1/stock/recommendation",
            params={"symbol": sym, "token": FINNHUB_API_KEY},
            timeout=5,
        )
        if r.status_code != 200:
            return {"score_delta": 0, "consensus": "hold"}
        data = r.json()
        if not data or not isinstance(data, list):
            return {"score_delta": 0, "consensus": "hold"}
        current = data[0]
        prev = data[1] if len(data) > 1 else {}
        sb = int(current.get("strongBuy") or 0)
        b = int(current.get("buy") or 0)
        h = int(current.get("hold") or 0)
        s = int(current.get("sell") or 0)
        ss = int(current.get("strongSell") or 0)
        total = sb + b + h + s + ss
        if total == 0:
            return {"score_delta": 0, "consensus": "hold", "raw": current}
        buy_pct = (sb * 2 + b) / max(1, total * 2)
        sell_pct = (ss * 2 + s) / max(1, total * 2)
        if buy_pct >= 0.6:
            consensus = "strong_buy" if sb > b else "buy"
        elif sell_pct >= 0.6:
            consensus = "strong_sell" if ss > s else "sell"
        else:
            consensus = "hold"
        prev_buy = (int(prev.get("strongBuy") or 0) * 2 + int(prev.get("buy") or 0)) / max(1, (sum(int(prev.get(k) or 0) for k in ["strongBuy", "buy", "hold", "sell", "strongSell"])) * 2) if prev else 0.5
        curr_buy = buy_pct
        score_delta = (curr_buy - prev_buy) * 2
        score_delta = max(-1, min(1, score_delta))
        return {
            "score_delta": round(score_delta, 2),
            "consensus": consensus,
            "strong_buy": sb, "buy": b, "hold": h, "sell": s, "strong_sell": ss,
            "raw": current,
        }
    except Exception as e:
        logger.debug("Analyst ratings %s: %s", symbol, e)
        return {"score_delta": 0, "consensus": "hold"}


def analyst_score_contribution(symbol: str) -> Tuple[float, str]:
    """
    Returns (score_adjustment, reason) for recommendation.
    score_adjustment: -20 to +20 (added to base score)
    """
    r = get_analyst_ratings(symbol)
    if r.get("enabled") is False:
        return 0.0, ""
    delta = r.get("score_delta", 0)
    consensus = r.get("consensus", "hold")
    contrib = delta * 15
    if consensus == "strong_buy":
        contrib = max(contrib, 10)
    elif consensus == "strong_sell":
        contrib = min(contrib, -15)
    return max(-20, min(20, contrib)), f"Analyst: {consensus} (delta {delta:+.2f})"
