"""
Lightweight explore row scoring helpers (signal age, price confirmation, conviction).
"""
from __future__ import annotations

from typing import Any, Dict, Optional, Tuple


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


def compute_conviction(
    sid: str,
    facts: Dict[str, Any],
    *,
    is_crypto: bool = False,
    sector_etf_ok: Optional[bool] = None,
    market_cap_tier: Optional[str] = None,
) -> int:
    """
    Compute a 0-100 conviction score for a strategy signal.
    Higher = stronger setup; used to rank competing matches and decide buy vs watch.
    """
    base = 52  # neutral starting point

    # Volume confirmation
    vol = float(facts.get("volume_mult") or facts.get("volume_ratio") or 1.0)
    if vol >= 3.0:
        base += 12
    elif vol >= 2.0:
        base += 8
    elif vol >= 1.5:
        base += 4
    elif vol < 0.8:
        base -= 6

    # Trend alignment bonus
    if facts.get("trend_aligned"):
        base += 8

    # Near key support/resistance level
    if facts.get("near_key_level"):
        base += 5

    # RSI — penalise overbought, slight reward oversold
    rsi_val = facts.get("rsi")
    if rsi_val is not None:
        r = float(rsi_val)
        if r >= 80:
            base -= 12
        elif r >= 70:
            base -= 6
        elif r <= 30:
            base += 6
        elif r <= 40:
            base += 3

    # Relative-strength / outperformance bonus
    outperf = facts.get("outperformance_pct")
    if outperf is not None:
        base += min(10, int(float(outperf) / 2))

    # Sector ETF confirmation for stocks
    if not is_crypto:
        if sector_etf_ok is True:
            base += 5
        elif sector_etf_ok is False:
            base -= 5

    # Small cap gets slight risk discount
    if market_cap_tier in ("micro", "small"):
        base -= 3

    return max(0, min(100, base))


def status_from_conviction(conv: int, sid: str = "") -> str:
    """Map conviction score to signal status string ("buy" or "watch")."""
    return "buy" if int(conv) >= 62 else "watch"
