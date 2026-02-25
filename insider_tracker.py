"""
Insider Trading Tracker - SEC Form 4 filings.
Bullish: CEO/CFO buying >$100k.
Bearish: multiple executives selling.
"""
import os
import logging
import time
from datetime import datetime, timezone, timedelta
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

FINNHUB_API_KEY = os.getenv("FINNHUB_API_KEY", "")  # Free tier has insider data


def _normalize_symbol(symbol: str) -> str:
    return symbol.split("/")[0].upper()


def fetch_and_store_insider_transactions(symbol: str, days_back: int = 90) -> int:
    """
    Fetch insider transactions from Finnhub and store in DB.
    Returns count of new records saved.
    """
    if not FINNHUB_API_KEY:
        return 0
    try:
        from db import save_insider_transaction, get_insider_transactions
        import requests
        sym = _normalize_symbol(symbol)
        if sym in ("BTC", "ETH"):
            return 0
        url = f"https://finnhub.io/api/v1/stock/insider-transactions"
        params = {"symbol": sym, "token": FINNHUB_API_KEY, "from": (datetime.now(timezone.utc) - timedelta(days=days_back)).strftime("%Y-%m-%d")}
        r = requests.get(url, params=params, timeout=10)
        if r.status_code != 200:
            return 0
        data = r.json()
        existing = {(t["transaction_date"], t["transaction_type"], t["shares"]) for t in get_insider_transactions(sym, days_back)}
        count = 0
        for t in data.get("data", []):
            td = t.get("transactionDate", "") or ""
            try:
                dt = datetime.strptime(td[:10], "%Y-%m-%d")
                ts = int(dt.replace(tzinfo=timezone.utc).timestamp())
            except Exception:
                ts = int(time.time())
            tt = str(t.get("transactionCode", "P") or "P")  # P=purchase, S=sale
            sh = float(t.get("share", t.get("shares", 0)) or 0)
            val = None
            if t.get("value") is not None:
                try:
                    val = float(t["value"])
                except (TypeError, ValueError):
                    pass
            title = str(t.get("name", t.get("insider_title", ""))) or ""
            key = (ts, tt, sh)
            if key not in existing:
                save_insider_transaction(
                    symbol=sym,
                    transaction_date=ts,
                    transaction_type="P" if "P" in tt or "A" in tt else "S",
                    shares=sh,
                    value_usd=val,
                    insider_title=title,
                )
                count += 1
                existing.add(key)
        return count
    except Exception as e:
        logger.debug("Insider fetch failed: %s", e)
        return 0


def get_insider_score(symbol: str) -> Tuple[float, List[str]]:
    """
    Compute insider signal for recommendation scoring.
    Returns: (score_delta -5 to +5, reasons).
    Bullish: CEO/CFO buying >$100k -> +5
    Bearish: multiple execs selling -> -3
    """
    from db import get_insider_transactions
    sym = _normalize_symbol(symbol)
    txns = get_insider_transactions(sym, 90)
    if not txns:
        return 0.0, []
    reasons = []
    score = 0.0
    buys = [t for t in txns if str(t.get("transaction_type", "")).upper() in ("P", "A")]
    sells = [t for t in txns if str(t.get("transaction_type", "")).upper() in ("S", "D")]
    for t in buys:
        val = float(t.get("value_usd") or 0)
        title = str(t.get("insider_title") or "").upper()
        if val >= 100000 and ("CEO" in title or "CFO" in title or "DIRECTOR" in title):
            score += 5.0
            reasons.append(f"Insider buy ${val/1e6:.1f}M ({title or 'exec'})")
            break
    if len(buys) >= 2 and not reasons:
        total_buy = sum(float(t.get("value_usd") or 0) for t in buys)
        if total_buy > 50000:
            score += 2.0
            reasons.append(f"Net insider buying ${total_buy/1e3:.0f}K")
    if len(sells) >= 3:
        score -= 3.0
        reasons.append(f"Multiple insider sales ({len(sells)})")
    return min(5, max(-5, score)), reasons
