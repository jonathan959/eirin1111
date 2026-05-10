"""Journal exit-reason taxonomy + helpers (strategies import this for one-liners)."""

from __future__ import annotations

from typing import Optional

JOURNAL_EXIT_REASONS = frozenset({
    "tp_hit",
    "sl_hit",
    "trailing_stop",
    "hard_stop_loss",
    "max_hold_timeout",
    "regime_change_exit",
    "manual_close",
    "kill_switch",
    "autotune_close",
})


def normalize_exit_reason(
    exit_strategy: Optional[str],
    *,
    explicit: Optional[str] = None,
) -> str:
    if explicit and explicit in JOURNAL_EXIT_REASONS:
        return explicit
    s = (exit_strategy or "").lower()
    if not s:
        return "manual_close"
    if "kill" in s and "switch" in s:
        return "kill_switch"
    if "trail" in s:
        return "trailing_stop"
    if "hard_sl" in s or "hard stop" in s:
        return "hard_stop_loss"
    if "take_profit" in s or "tp" in s or s.endswith("_tp"):
        return "tp_hit"
    if "stop_loss" in s or "sl_hit" in s or "stop loss" in s:
        return "sl_hit"
    if "timeout" in s or "max_hold" in s or "hold_timeout" in s:
        return "max_hold_timeout"
    if "regime" in s:
        return "regime_change_exit"
    if "autotune" in s:
        return "autotune_close"
    return "manual_close"


def write_close_from_strategy(
    *,
    deal_id: int,
    bot_id: int,
    symbol: str,
    strategy: str,
    side: str,
    qty: float,
    entry_px: Optional[float],
    exit_px: Optional[float],
    entry_ts: int,
    exit_ts: int,
    pnl_quote: float,
    pnl_pct: Optional[float],
    entry_reason: str,
    exit_strategy: Optional[str],
    lessons: Optional[str],
    source: str,
    dry_run: bool,
    journal_exit_explicit: Optional[str] = None,
) -> None:
    """One-line helper for strategy code; persists via db.upsert_journal_trade."""
    from db import upsert_journal_trade

    src = "paper" if dry_run else "live"
    if source:
        src = str(source)
    ex = normalize_exit_reason(exit_strategy, explicit=journal_exit_explicit)
    upsert_journal_trade(
        int(bot_id),
        str(symbol),
        str(strategy),
        str(side or "long"),
        float(qty or 0.0),
        float(entry_px or 0.0),
        float(exit_px or 0.0),
        int(entry_ts),
        int(exit_ts),
        float(pnl_quote or 0.0),
        pnl_pct,
        str(entry_reason or "")[:2000],
        ex,
        (lessons or None),
        src,
        deal_id=int(deal_id),
    )
