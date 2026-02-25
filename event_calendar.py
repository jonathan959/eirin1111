"""
Event Calendar - avoid entering positions day before earnings/Fed meetings.
"""
import time
from datetime import datetime, timezone, timedelta
from typing import List, Optional, Tuple

from db import get_events_for_dates


def _date_ts(dt: datetime) -> int:
    """Unix ts of midnight UTC for date."""
    return int(dt.replace(hour=0, minute=0, second=0, microsecond=0).timestamp())


def should_avoid_entry(
    symbol: Optional[str] = None,
    days_ahead: int = 2,
) -> Tuple[bool, str]:
    """
    True if we should avoid entering (event within days_ahead).
    Returns (avoid, reason).
    """
    now = datetime.now(timezone.utc)
    start = _date_ts(now)
    end = _date_ts(now + timedelta(days=days_ahead))
    events = get_events_for_dates(start, end, symbol)
    high_impact = [e for e in events if int(e.get("impact_level") or 0) >= 2]
    if not high_impact:
        return False, ""
    types = set(e.get("event_type") or "" for e in high_impact)
    if "earnings" in types or "fed" in types or "FOMC" in str(types).upper():
        return True, f"Avoid: {', '.join(types)} within {days_ahead} days"
    return False, ""


def events_near_date(symbol: Optional[str], date_ts: int, window_days: int = 3) -> List[dict]:
    """Get events within window of date."""
    from datetime import datetime
    dt = datetime.fromtimestamp(date_ts, tz=timezone.utc)
    start = _date_ts(dt - timedelta(days=window_days))
    end = _date_ts(dt + timedelta(days=window_days))
    return get_events_for_dates(start, end, symbol)
