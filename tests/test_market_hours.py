"""
Unit tests for US equities market-hours / trade-window logic.
Pure functions, no network. Uses phase1_intelligence.is_us_equities_trade_window_ok.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from datetime import datetime, timezone, time as dt_time

from zoneinfo import ZoneInfo

ET = ZoneInfo("America/New_York")
UTC = timezone.utc


def _et(y: int, mo: int, d: int, h: int, mi: int) -> datetime:
    """Build datetime in ET, then convert to UTC for is_us_equities_trade_window_ok."""
    et = datetime(y, mo, d, h, mi, 0, tzinfo=ET)
    return et.astimezone(UTC)


def test_market_hours():
    from phase1_intelligence import is_us_equities_trade_window_ok

    # Friday 09:30 ET -> skip first 30 min (boundary open)
    ok, r = is_us_equities_trade_window_ok(_et(2025, 1, 31, 9, 30), True, True)
    assert ok is False
    assert "first 30min" in (r or "")

    # Friday 09:45 ET -> still skip first 30 min (+15m)
    ok, r = is_us_equities_trade_window_ok(_et(2025, 1, 31, 9, 45), True, True)
    assert ok is False
    assert "first 30min" in (r or "")

    # Friday 10:05 ET -> allowed (+35m)
    ok, r = is_us_equities_trade_window_ok(_et(2025, 1, 31, 10, 5), True, True)
    assert ok is True, r
    assert r is None

    # Friday 15:35 ET -> skip last 30 min (close-25m)
    ok, r = is_us_equities_trade_window_ok(_et(2025, 1, 31, 15, 35), True, True)
    assert ok is False
    assert "closing soon" in (r or "")

    # Friday 15:55 ET -> skip last 30 min (close-5m)
    ok, r = is_us_equities_trade_window_ok(_et(2025, 1, 31, 15, 55), True, True)
    assert ok is False
    assert "closing soon" in (r or "")

    # Friday 10:15 ET -> ok (inside window, past first 30 min)
    ok, r = is_us_equities_trade_window_ok(_et(2025, 1, 31, 10, 15), True, True)
    assert ok is True, r
    assert r is None

    # Friday 09:45 ET -> skip first 30 min
    ok, r = is_us_equities_trade_window_ok(_et(2025, 1, 31, 9, 45), True, True)
    assert ok is False
    assert "first 30min" in (r or "")

    # Friday 15:45 ET -> skip last 30 min
    ok, r = is_us_equities_trade_window_ok(_et(2025, 1, 31, 15, 45), True, True)
    assert ok is False
    assert "closing soon" in (r or "")

    # Saturday -> weekend
    ok, r = is_us_equities_trade_window_ok(_et(2025, 2, 1, 10, 0), True, True)
    assert ok is False
    assert "weekend" in (r or "")

    # Sunday -> weekend
    ok, r = is_us_equities_trade_window_ok(_et(2025, 2, 2, 10, 0), True, True)
    assert ok is False
    assert "weekend" in (r or "")

    # Before open
    ok, r = is_us_equities_trade_window_ok(_et(2025, 1, 31, 9, 0), True, True)
    assert ok is False
    assert "Market closed" in (r or "")

    # At 16:00 ET -> closed
    ok, r = is_us_equities_trade_window_ok(_et(2025, 1, 31, 16, 0), True, True)
    assert ok is False
    assert "Market closed" in (r or "")

    # No skip first/last: 09:45 and 15:45 ok
    ok, r = is_us_equities_trade_window_ok(_et(2025, 1, 31, 9, 45), False, False)
    assert ok is True, r
    ok, r = is_us_equities_trade_window_ok(_et(2025, 1, 31, 15, 45), False, False)
    assert ok is True, r


if __name__ == "__main__":
    test_market_hours()
    print("test_market_hours passed")
