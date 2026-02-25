"""
Earnings calendar integration - track upcoming earnings, auto-exit/reduce before earnings.

- Fetch earnings from Finnhub
- Populate market_events for event_calendar
- EXIT_BEFORE_EARNINGS: reduce/exit position N days before
- EARNINGS_LOOKBACK_DAYS: fetch earnings within window
"""
import logging
import os
from datetime import datetime, timezone, timedelta
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

EXIT_BEFORE_EARNINGS = os.getenv("EXIT_BEFORE_EARNINGS", "0").strip().lower() in ("1", "true", "yes")
EARNINGS_LOOKBACK_DAYS = int(os.getenv("EARNINGS_LOOKBACK_DAYS", "7"))
EARNINGS_DAYS_AHEAD = int(os.getenv("EARNINGS_DAYS_AHEAD", "14"))
FINNHUB_API_KEY = os.getenv("FINNHUB_API_KEY", os.getenv("FINNHUB_TOKEN", "")).strip()


def _date_to_ts(dt: datetime) -> int:
    """Unix ts of midnight UTC for date."""
    return int(dt.replace(hour=0, minute=0, second=0, microsecond=0).timestamp())


def fetch_earnings_for_symbol(symbol: str, days_ahead: int = 14) -> List[Dict[str, Any]]:
    """
    Fetch upcoming earnings for symbol from Finnhub.
    Returns list of { date, eps_estimate, revenue_estimate, quarter, year }.
    """
    if not FINNHUB_API_KEY:
        return []
    try:
        import requests
        sym = symbol.upper().split("/")[0]
        now = datetime.now(timezone.utc)
        from_d = (now - timedelta(days=1)).strftime("%Y-%m-%d")
        to_d = (now + timedelta(days=days_ahead)).strftime("%Y-%m-%d")
        url = "https://finnhub.io/api/v1/calendar/earnings"
        r = requests.get(url, params={"from": from_d, "to": to_d, "symbol": sym, "token": FINNHUB_API_KEY}, timeout=10)
        if r.status_code != 200:
            return []
        data = r.json()
        out = []
        for e in (data.get("earningsCalendar") or [])[:20]:
            if str(e.get("symbol", "")).upper() != sym:
                continue
            date_str = e.get("date") or e.get("reportDate") or ""
            if not date_str:
                continue
            try:
                dt = datetime.strptime(date_str[:10], "%Y-%m-%d").replace(tzinfo=timezone.utc)
                out.append({
                    "date": date_str[:10],
                    "date_ts": _date_to_ts(dt),
                    "eps_estimate": e.get("epsEstimate"),
                    "revenue_estimate": e.get("revenueEstimate"),
                    "quarter": e.get("quarter"),
                    "year": e.get("year"),
                })
            except ValueError:
                pass
        return sorted(out, key=lambda x: x["date_ts"])
    except Exception as e:
        logger.debug("Earnings fetch %s: %s", symbol, e)
        return []


def sync_earnings_to_events(symbol: str) -> int:
    """Fetch earnings and save to market_events. Returns count added."""
    earnings = fetch_earnings_for_symbol(symbol, days_ahead=EARNINGS_DAYS_AHEAD)
    if not earnings:
        return 0
    try:
        from db import save_market_event, get_events_for_dates
        added = 0
        for e in earnings:
            ts = e["date_ts"]
            start, end = ts - 86400, ts + 86400
            existing = get_events_for_dates(start, end, symbol)
            if not any(ex.get("event_type") == "earnings" for ex in existing):
                save_market_event(ts, "earnings", symbol, impact_level=2, description=f"Earnings {e.get('quarter','')} {e.get('year','')}")
                added += 1
        return added
    except Exception as e:
        logger.debug("Sync earnings to events: %s", e)
        return 0


def days_until_earnings(symbol: str) -> Optional[int]:
    """Days until next earnings. None if none scheduled."""
    earnings = fetch_earnings_for_symbol(symbol, days_ahead=30)
    if not earnings:
        return None
    now = _date_to_ts(datetime.now(timezone.utc))
    for e in earnings:
        if e["date_ts"] >= now:
            return (e["date_ts"] - now) // 86400
    return None


def should_reduce_before_earnings(symbol: str) -> Tuple[bool, str]:
    """
    If EXIT_BEFORE_EARNINGS and earnings within 2 days, return (True, reason).
    """
    if not EXIT_BEFORE_EARNINGS:
        return False, ""
    days = days_until_earnings(symbol)
    if days is not None and days <= 2:
        return True, f"Earnings in {days} day(s) - reduce/exit per config"
    return False, ""
