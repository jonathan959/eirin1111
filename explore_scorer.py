"""
Lightweight explore row scoring helpers (signal age, price confirmation).
"""
from __future__ import annotations

from typing import Optional, Tuple


def price_confirmation_score(
    change_pct: Optional[float],
    tentative_signal: str,
) -> Tuple[float, str]:
    """Returns (score_delta, label) for 24h move vs signal direction."""
    if change_pct is None:
        return 0.0, ""
    chg = float(change_pct)
    ts = str(tentative_signal or "").lower()
    # Penalize buying into large same-day pumps; small reward when aligned
    if ts == "buy":
        if chg > 12.0:
            return -12.0, "Chasing 24h pump"
        if chg < -8.0:
            return 3.0, "Pullback from highs"
    elif ts == "watch":
        if abs(chg) < 1.0:
            return 0.0, "Flat"
    return 0.0, ""


def signal_age_penalty(
    signal_age_minutes: Optional[float],
) -> Tuple[float, str, bool]:
    """Returns (penalty_points, reason, force_watch cap)."""
    if signal_age_minutes is None:
        return 0.0, "", False
    age = float(signal_age_minutes)
    if age >= 480:
        return 0.0, "stale", True
    if age >= 120:
        return 15.0, "age penalty", False
    if age >= 60:
        return 7.0, "age penalty", False
    return 0.0, "", False
