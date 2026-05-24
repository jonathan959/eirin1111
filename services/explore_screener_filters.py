"""
Explore Market Screener — shared filter rules for /api/explore/feed and the UI.

Keeps server-side and client-side filtering aligned on one definition per rule.
"""
from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Tuple

HIGH_CONVICTION_MIN = 70.0
ACTIONABLE_MIN_WIN_RATE_90D = 40.0
ACTIONABLE_MIN_RISK_REWARD = 1.0


def effective_composite_score(item: Dict[str, Any]) -> float:
    """Primary score for screener filters (composite when present)."""
    if item.get("composite_score") is not None:
        try:
            return float(item["composite_score"])
        except (TypeError, ValueError):
            pass
    try:
        return float(item.get("score") or 0.0)
    except (TypeError, ValueError):
        return 0.0


def parse_risk_reward(item: Dict[str, Any]) -> Optional[float]:
    """Numeric risk:reward from ratio field or display string (e.g. '0.9:1')."""
    raw = item.get("risk_reward_ratio")
    if raw is not None:
        try:
            return float(raw)
        except (TypeError, ValueError):
            pass
    disp = str(item.get("risk_reward_display") or "").strip()
    if not disp or disp in ("—", "N/A", "n/a"):
        return None
    m = re.match(r"^([\d.]+)\s*:?\s*1", disp)
    if m:
        try:
            return float(m.group(1))
        except ValueError:
            pass
    try:
        return float(disp.split(":")[0])
    except (ValueError, IndexError):
        return None


def is_high_conviction(item: Dict[str, Any], enabled: bool) -> bool:
    if not enabled:
        return True
    return effective_composite_score(item) >= HIGH_CONVICTION_MIN


def passes_min_score(item: Dict[str, Any], min_score: float) -> bool:
    if min_score <= 0:
        return True
    return effective_composite_score(item) >= float(min_score)


def is_negative_edge(item: Dict[str, Any]) -> bool:
    """Unactionable: poor R:R or sub-threshold 90d strategy win rate."""
    rr = parse_risk_reward(item)
    if rr is not None and rr < ACTIONABLE_MIN_RISK_REWARD:
        return True
    wr = item.get("backtest_win_rate_90d")
    if wr is not None:
        try:
            if float(wr) < ACTIONABLE_MIN_WIN_RATE_90D:
                return True
        except (TypeError, ValueError):
            pass
    return False


def negative_edge_reason(item: Dict[str, Any]) -> str:
    rr = parse_risk_reward(item)
    if rr is not None and rr < ACTIONABLE_MIN_RISK_REWARD:
        return f"Risk:reward {rr:.1f}:1 below minimum {ACTIONABLE_MIN_RISK_REWARD:.1f}:1"
    wr = item.get("backtest_win_rate_90d")
    if wr is not None:
        try:
            if float(wr) < ACTIONABLE_MIN_WIN_RATE_90D:
                return f"90d strategy win rate {float(wr):.0f}% below {ACTIONABLE_MIN_WIN_RATE_90D:.0f}% active threshold"
        except (TypeError, ValueError):
            pass
    return "Below actionable quality thresholds"


def passes_strategy_active_gate(item: Dict[str, Any], show_unproven: bool) -> bool:
    """Active strategies always pass; pending only when show_unproven."""
    if item.get("strategy_is_active"):
        return True
    if show_unproven and item.get("strategy_is_pending"):
        return True
    return False


def screener_reject_reason(
    item: Dict[str, Any],
    *,
    min_score: float,
    high_conviction_only: bool,
    show_unproven: bool,
) -> Optional[str]:
    """Return rejection reason if row must not appear in the main actionable table."""
    if is_negative_edge(item):
        return negative_edge_reason(item)
    if not passes_strategy_active_gate(item, show_unproven):
        return "Strategy not active — enable Show unproven signals to include pending patterns"
    if not passes_min_score(item, min_score):
        return f"Composite score {effective_composite_score(item):.0f} below min {min_score:.0f}"
    if not is_high_conviction(item, high_conviction_only):
        return f"Composite score {effective_composite_score(item):.0f} below high-conviction floor ({HIGH_CONVICTION_MIN:.0f})"
    return None


def apply_screener_filters(
    items: List[Dict[str, Any]],
    *,
    min_score: float = 0.0,
    high_conviction_only: bool = False,
    show_unproven: bool = False,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Split feed rows into main-table candidates and rows routed to rejection/watchlist.

    Returns (main_items, sidelined_items). Sidelined items include screener_reject_reason.
    """
    main: List[Dict[str, Any]] = []
    sidelined: List[Dict[str, Any]] = []
    for it in items:
        reason = screener_reject_reason(
            it,
            min_score=min_score,
            high_conviction_only=high_conviction_only,
            show_unproven=show_unproven,
        )
        if reason:
            row = dict(it)
            row["screener_reject_reason"] = reason
            sidelined.append(row)
        else:
            main.append(it)
    return main, sidelined


def strategy_flags_for_id(
    strategy_id: str,
    strategy_health: List[Dict[str, Any]],
) -> Tuple[bool, bool]:
    """(is_active, is_pending) for a strategy id from Strategy Health panel data."""
    sid = (strategy_id or "").strip()
    if not sid:
        return False, False
    for sh in strategy_health:
        if str(sh.get("id") or "") == sid:
            status = str(sh.get("status") or "").lower()
            return bool(sh.get("active")), status == "pending"
    return False, False
