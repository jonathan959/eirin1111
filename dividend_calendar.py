"""
Dividend calendar - track ex-dividend dates, hold through ex-div, factor yield.

- Fetch dividends from Finnhub
- Save to dividend_events
- days_to_exdiv, dividend_yield for entry logic
"""
import logging
import os
from datetime import datetime, timezone, timedelta
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

FINNHUB_API_KEY = os.getenv("FINNHUB_API_KEY", os.getenv("FINNHUB_TOKEN", "")).strip()


def _date_to_ts(dt: datetime) -> int:
    return int(dt.replace(hour=0, minute=0, second=0, microsecond=0).timestamp())


def fetch_dividends(symbol: str, from_date: str = None, to_date: str = None) -> List[Dict[str, Any]]:
    """
    Fetch dividend calendar from Finnhub.
    Returns list of { date, amount, currency, declaredDate, payDate }.
    """
    if not FINNHUB_API_KEY:
        return []
    try:
        import requests
        sym = symbol.upper().split("/")[0]
        now = datetime.now(timezone.utc)
        from_d = from_date or (now - timedelta(days=30)).strftime("%Y-%m-%d")
        to_d = to_date or (now + timedelta(days=365)).strftime("%Y-%m-%d")
        r = requests.get(
            "https://finnhub.io/api/v1/stock/dividend",
            params={"symbol": sym, "from": from_d, "to": to_d, "token": FINNHUB_API_KEY},
            timeout=5,
        )
        if r.status_code != 200:
            return []
        data = r.json()
        if not isinstance(data, list):
            return []
        out = []
        for d in data[:20]:
            date_str = d.get("date") or d.get("exDate") or ""
            if not date_str:
                continue
            amount = float(d.get("amount") or d.get("dividend") or 0)
            if amount <= 0:
                continue
            try:
                dt = datetime.strptime(date_str[:10], "%Y-%m-%d").replace(tzinfo=timezone.utc)
                out.append({
                    "date": date_str[:10],
                    "date_ts": _date_to_ts(dt),
                    "amount": amount,
                    "currency": d.get("currency", "USD"),
                    "declareDate": d.get("declareDate"),
                    "payDate": d.get("payDate"),
                })
            except ValueError:
                pass
        return sorted(out, key=lambda x: x["date_ts"])
    except Exception as e:
        logger.debug("Dividend fetch %s: %s", symbol, e)
        return []


def days_to_exdiv(symbol: str, current_price: float = 0.0) -> Tuple[Optional[int], Optional[float], Optional[float]]:
    """
    Days until next ex-dividend, amount, yield_pct.
    Returns (days, amount, yield_pct).
    """
    divs = fetch_dividends(symbol)
    if not divs:
        return None, None, None
    now = _date_to_ts(datetime.now(timezone.utc))
    for d in divs:
        if d["date_ts"] >= now:
            days = (d["date_ts"] - now) // 86400
            amount = d.get("amount", 0)
            yld = (amount / current_price * 100) if current_price and current_price > 0 else None
            return days, amount, round(yld, 2) if yld else None
    return None, None, None


def sync_dividends_to_db(symbol: str, current_price: float = 0.0) -> int:
    """Fetch dividends and save to dividend_events. Returns count added."""
    divs = fetch_dividends(symbol)
    if not divs:
        return 0
    try:
        from db import save_dividend_event, list_dividend_events
        added = 0
        existing = {e["ex_date"] for e in list_dividend_events(symbol, 50)}
        for d in divs:
            ts = d["date_ts"]
            if ts in existing:
                continue
            amount = d.get("amount", 0)
            yld = (amount / current_price * 100) if current_price and current_price > 0 else None
            pay = d.get("payDate")
            pay_ts = None
            if pay:
                try:
                    pay_dt = datetime.strptime(str(pay)[:10], "%Y-%m-%d").replace(tzinfo=timezone.utc)
                    pay_ts = _date_to_ts(pay_dt)
                except ValueError:
                    pass
            save_dividend_event(symbol, ts, amount, payment_date=pay_ts, dividend_yield_pct=yld)
            added += 1
        return added
    except Exception as e:
        logger.debug("Sync dividends: %s", e)
        return 0
