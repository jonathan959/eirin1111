"""
Tax-Loss Harvesting - detect unrealized losses, suggest harvesting, track wash-sale.
"""
import os
import time
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import logging
from db import _conn, now_ts

logger = logging.getLogger(__name__)

ENABLE_TAX_HARVESTING = os.getenv("ENABLE_TAX_HARVESTING", "0").strip().lower() in ("1", "true", "yes", "y", "on")
WASH_SALE_DAYS = 30
MIN_LOSS_PCT = 5.0
YEAR_END_DAYS = 45


def _year_end_approaching() -> bool:
    """True if within YEAR_END_DAYS of year end."""
    now = datetime.now(timezone.utc)
    year_end = datetime(now.year, 12, 31, tzinfo=timezone.utc)
    delta = (year_end - now).days
    return 0 <= delta <= YEAR_END_DAYS


def save_tax_harvest_suggestion(
    symbol: str,
    unrealized_loss_pct: float,
    wash_sale_until_ts: int,
    alternate_symbol: Optional[str] = None,
) -> None:
    """Record a tax-loss harvesting suggestion."""
    con = _conn()
    try:
        con.execute(
            """
            INSERT INTO tax_harvest_suggestions(symbol, unrealized_loss_pct, suggest_sell_ts, wash_sale_until_ts, alternate_symbol, recorded_at)
            VALUES (?,?,?,?,?,?)
            """,
            (str(symbol), float(unrealized_loss_pct), now_ts(), int(wash_sale_until_ts), str(alternate_symbol or ""), now_ts()),
        )
        con.commit()
    finally:
        con.close()


def get_recent_sales(symbol: str, since_ts: int) -> List[Dict[str, Any]]:
    """Get recent sales for wash-sale check. Uses deals table."""
    con = _conn()
    rows = con.execute(
        """
        SELECT id, symbol, closed_at, realized_pnl_quote
        FROM deals
        WHERE symbol=? AND state='CLOSED' AND closed_at >= ?
        ORDER BY closed_at DESC
        """,
        (str(symbol).strip(), int(since_ts)),
    ).fetchall()
    con.close()
    return [dict(r) for r in rows]


def wash_sale_blocked_until(symbol: str) -> Optional[int]:
    """Return timestamp until which buying symbol would trigger wash sale."""
    since_ts = now_ts() - (WASH_SALE_DAYS * 86400)
    sales = get_recent_sales(symbol, since_ts)
    if not sales:
        return None
    latest = max(int(s.get("closed_at") or 0) for s in sales)
    return latest + (WASH_SALE_DAYS * 86400)


def tax_harvest_suggestions(
    positions: List[Dict[str, Any]],
    min_loss_pct: float = MIN_LOSS_PCT,
) -> List[Dict[str, Any]]:
    """
    Positions: [{symbol, entry_price, current_price, qty, ...}]
    Returns suggestions for positions with unrealized loss >= min_loss_pct.
    """
    if not ENABLE_TAX_HARVESTING or not _year_end_approaching():
        return []
    suggestions = []
    for pos in positions:
        sym = pos.get("symbol")
        entry = float(pos.get("entry_price") or pos.get("avg_entry") or 0)
        current = float(pos.get("current_price") or pos.get("last_price") or 0)
        if not entry or not current or entry <= 0:
            continue
        loss_pct = (entry - current) / entry * 100
        if loss_pct >= min_loss_pct:
            wash_until = wash_sale_blocked_until(sym or "")
            suggestions.append({
                "symbol": sym,
                "unrealized_loss_pct": round(loss_pct, 2),
                "entry_price": entry,
                "current_price": current,
                "wash_sale_until_ts": wash_until,
                "wash_sale_days_remaining": max(0, (wash_until - now_ts()) // 86400) if wash_until else 0,
                "alternate_symbol": _find_alternate(sym),
            })
    return suggestions


def _find_alternate(symbol: str) -> Optional[str]:
    """Find similar but not identical symbol (avoid wash sale). Stub - expand with sector/ETF mapping."""
    from stock_metadata import get_sector, STOCK_SECTOR_MAP
    sector = get_sector(symbol)
    if not sector:
        return None
    candidates = [s for s, sec in STOCK_SECTOR_MAP.items() if sec == sector and s != symbol]
    return candidates[0] if candidates else None
