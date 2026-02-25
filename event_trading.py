"""
News event trading - avoid trading around earnings, Fed, etc. (12.md Part 5)
"""
import logging
import os
from datetime import datetime, timedelta
from typing import Dict, List

logger = logging.getLogger(__name__)
ENABLED = os.getenv("ENABLE_EVENT_TRADING", "0").strip().lower() in ("1", "true", "yes")
FINNHUB_KEY = os.getenv("FINNHUB_API_KEY", "").strip()

FOMC_2026 = ["2026-01-27", "2026-03-17", "2026-04-28", "2026-06-16", "2026-07-28", "2026-09-22", "2026-11-03", "2026-12-15"]


def fetch_upcoming_events(days_ahead: int = 7) -> Dict[str, List]:
    """Fetch earnings, economic, Fed events."""
    if not ENABLED:
        return {"earnings": [], "economic": [], "fed": []}
    fed_events = []
    today = datetime.now().date()
    end = today + timedelta(days=days_ahead)
    for dstr in FOMC_2026:
        try:
            d = datetime.strptime(dstr, "%Y-%m-%d").date()
            if today <= d <= end:
                fed_events.append({"event": "FOMC Meeting", "date": dstr, "impact": "critical", "type": "fed"})
        except Exception:
            pass
    if FINNHUB_KEY:
        try:
            import requests
            r = requests.get(
                "https://finnhub.io/api/v1/calendar/earnings",
                params={"from": today.isoformat(), "to": end.isoformat(), "token": FINNHUB_KEY},
                timeout=5
            )
            earnings = [{"symbol": e.get("symbol"), "date": e.get("date"), "type": "earnings"} for e in (r.json().get("earningsCalendar") or [])]
        except Exception as e:
            logger.debug("event_trading earnings: %s", e)
            earnings = []
    else:
        earnings = []
    return {"earnings": earnings, "economic": [], "fed": fed_events}


def check_symbol_events(symbol: str, days_ahead: int = 7) -> List[Dict]:
    """Check if symbol has upcoming events."""
    events = fetch_upcoming_events(days_ahead)
    out = []
    sym_clean = symbol.replace("/USD", "").replace("-USD", "") if symbol else ""
    for e in events.get("earnings", []):
        if str(e.get("symbol", "")).upper() == sym_clean.upper():
            out.append(e)
    out.extend(events.get("fed", []))
    return sorted(out, key=lambda x: x.get("date", ""))


def should_avoid_trading(symbol: str, market_type: str) -> Dict:
    """Determine if trading should be avoided due to events."""
    if not ENABLED:
        return {"avoid_trading": False}
    upcoming = check_symbol_events(symbol, 3)
    critical = []
    for e in upcoming:
        dstr = e.get("date", "")
        if not dstr:
            continue
        try:
            ed = datetime.strptime(dstr[:10], "%Y-%m-%d").date()
            days = (ed - datetime.now().date()).days
            if e.get("type") == "earnings" and 0 <= days <= 2:
                critical.append(e)
            elif e.get("type") == "fed" and days == 0:
                critical.append(e)
        except Exception:
            pass
    if critical:
        return {"avoid_trading": True, "reason": f"{len(critical)} critical event(s)", "events": critical}
    return {"avoid_trading": False}
