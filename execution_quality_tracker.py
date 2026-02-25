"""
Execution quality tracking: slippage, post-trade analysis vs VWAP/TWAP.
"""
import logging
import os
import time
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

TRACK_EXECUTION_QUALITY = os.getenv("TRACK_EXECUTION_QUALITY", "1").strip().lower() in ("1", "true", "yes", "y", "on")


def record_execution(
    order_id: str,
    bot_id: int,
    symbol: str,
    side: str,
    intended_price: float,
    executed_price: Optional[float] = None,
    strategy: Optional[str] = None,
    vwap_at_execution: Optional[float] = None,
    twap_at_execution: Optional[float] = None,
) -> bool:
    """Record execution quality for post-trade analysis."""
    if not TRACK_EXECUTION_QUALITY or intended_price <= 0:
        return False
    try:
        from db import _conn, now_ts
        exec_price = executed_price or intended_price
        slippage_pct = ((exec_price - intended_price) / intended_price) * 100 if intended_price > 0 else 0
        # For sells: positive slippage = got more; for buys: negative = paid more
        if side.lower() == "buy":
            slippage_dollars = (intended_price - exec_price)  # negative = paid more
        else:
            slippage_dollars = exec_price - intended_price
        score = _compute_quality_score(
            intended_price, exec_price, vwap_at_execution, twap_at_execution, side
        )
        con = _conn()
        con.execute(
            """
            INSERT INTO execution_quality (order_id, bot_id, symbol, side, strategy, intended_price, executed_price, slippage_pct, slippage_dollars, vwap_at_execution, twap_at_execution, execution_quality_score, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (order_id, bot_id, symbol, side, strategy or "", intended_price, exec_price, slippage_pct, slippage_dollars, vwap_at_execution, twap_at_execution, score, now_ts()),
        )
        con.commit()
        con.close()
        return True
    except Exception as e:
        logger.debug("record_execution failed: %s", e)
        return False


def _compute_quality_score(
    intended: float,
    executed: float,
    vwap: Optional[float],
    twap: Optional[float],
    side: str,
) -> int:
    """Compute execution quality score 0-100. Higher = better execution."""
    if intended <= 0:
        return 50
    slippage_pct = abs((executed - intended) / intended) * 100
    # Penalize slippage: 0.1% = -5 pts, 0.5% = -25, 1% = -50
    slip_penalty = min(50, slippage_pct * 50)
    score = 100 - int(slip_penalty)
    if vwap and vwap > 0:
        beat_vwap = (executed < vwap and side == "buy") or (executed > vwap and side == "sell")
        if beat_vwap:
            score = min(100, score + 5)
    if twap and twap > 0:
        beat_twap = (executed < twap and side == "buy") or (executed > twap and side == "sell")
        if beat_twap:
            score = min(100, score + 5)
    return max(0, min(100, score))


def get_avg_slippage_by_symbol(days: int = 30) -> Dict[str, Dict[str, Any]]:
    """Average slippage per symbol."""
    try:
        from db import _conn, now_ts
        since = now_ts() - (days * 86400)
        con = _conn()
        rows = con.execute(
            """
            SELECT symbol, AVG(slippage_pct) as avg_slip_pct, AVG(slippage_dollars) as avg_slip_usd, COUNT(*) as cnt
            FROM execution_quality
            WHERE created_at >= ?
            GROUP BY symbol
            """,
            (since,),
        ).fetchall()
        con.close()
        return {r["symbol"]: {"avg_slippage_pct": r["avg_slip_pct"], "avg_slippage_usd": r["avg_slip_usd"], "count": r["cnt"]} for r in rows}
    except Exception as e:
        logger.debug("get_avg_slippage_by_symbol failed: %s", e)
        return {}


def get_avg_slippage_by_strategy(days: int = 30) -> Dict[str, Dict[str, Any]]:
    """Average slippage per strategy."""
    try:
        from db import _conn, now_ts
        since = now_ts() - (days * 86400)
        con = _conn()
        rows = con.execute(
            """
            SELECT strategy, AVG(slippage_pct) as avg_slip_pct, AVG(execution_quality_score) as avg_score, COUNT(*) as cnt
            FROM execution_quality
            WHERE created_at >= ? AND (strategy IS NULL OR strategy != '')
            GROUP BY strategy
            """,
            (since,),
        ).fetchall()
        con.close()
        return {r["strategy"] or "unknown": {"avg_slippage_pct": r["avg_slip_pct"], "avg_score": r["avg_score"], "count": r["cnt"]} for r in rows}
    except Exception as e:
        logger.debug("get_avg_slippage_by_strategy failed: %s", e)
        return {}


def get_execution_summary(order_id: str) -> Optional[Dict[str, Any]]:
    """Get execution quality summary for display. Returns e.g. 'Execution quality: 87/100 - beat VWAP by 0.15%'."""
    try:
        from db import _conn
        con = _conn()
        row = con.execute(
            "SELECT * FROM execution_quality WHERE order_id = ? ORDER BY created_at DESC LIMIT 1",
            (order_id,),
        ).fetchone()
        con.close()
        if not row:
            return None
        score = row["execution_quality_score"] or 50
        slip = row["slippage_pct"] or 0
        vwap = row["vwap_at_execution"]
        msg = f"Execution quality: {score}/100"
        if vwap and row["executed_price"]:
            diff = ((float(row["executed_price"]) - vwap) / vwap) * 100
            beat = (row["side"] == "buy" and diff < 0) or (row["side"] == "sell" and diff > 0)
            msg += f" - {'beat' if beat else 'missed'} VWAP by {abs(diff):.2f}%"
        elif slip != 0:
            msg += f" - slippage {slip:+.2f}%"
        return {"score": score, "message": msg, "slippage_pct": slip}
    except Exception as e:
        logger.debug("get_execution_summary failed: %s", e)
        return None
