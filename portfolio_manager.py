"""
Portfolio-level capital management: cash reserve, leverage, CAGR.
"""
import logging
import math
import os
import time
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

MIN_CASH_RESERVE_PCT = float(os.getenv("MIN_CASH_RESERVE_PCT", "20")) / 100.0
MAX_LEVERAGE = float(os.getenv("MAX_LEVERAGE", "2.0"))
REINVEST_PROFITS = os.getenv("REINVEST_PROFITS", "1").strip().lower() in ("1", "true", "yes", "y", "on")


def compute_cash_reserve(
    total_portfolio: float,
    volatility_pct: Optional[float] = None,
    high_vol_threshold: float = 0.05,
) -> float:
    """
    Minimum cash reserve. Base 20%, increase in high volatility.
    """
    base_reserve = total_portfolio * MIN_CASH_RESERVE_PCT
    if volatility_pct is not None and volatility_pct >= high_vol_threshold:
        extra = total_portfolio * 0.05
        base_reserve += extra
    return base_reserve


def get_available_for_allocation(
    total_portfolio: float,
    positions_usd: float,
    volatility_pct: Optional[float] = None,
) -> float:
    """Capital available after reserve and positions."""
    reserve = compute_cash_reserve(total_portfolio, volatility_pct)
    used = positions_usd
    return max(0.0, total_portfolio - reserve - used)


def check_leverage(
    equity: float,
    margin_used: float,
) -> Dict[str, Any]:
    """
    Track leverage. Returns {current_leverage, within_limit, alert}.
    """
    if equity <= 0:
        return {"current_leverage": 0.0, "within_limit": True, "alert": None}
    lev = (equity + margin_used) / equity if equity > 0 else 0
    within = lev <= MAX_LEVERAGE
    alert = None
    if lev >= MAX_LEVERAGE * 0.9:
        alert = f"Leverage {lev:.2f}x approaching max {MAX_LEVERAGE}x"
    elif lev >= MAX_LEVERAGE:
        alert = f"Leverage {lev:.2f}x exceeds max {MAX_LEVERAGE}x - de-leverage recommended"
    return {
        "current_leverage": round(lev, 2),
        "within_limit": within,
        "alert": alert,
        "max_leverage": MAX_LEVERAGE,
    }


def compute_cagr(
    initial_value: float,
    final_value: float,
    years: float,
) -> Optional[float]:
    """Compound annual growth rate. Years can be fractional."""
    if initial_value <= 0 or years <= 0:
        return None
    if final_value <= 0:
        return -1.0
    return (final_value / initial_value) ** (1.0 / years) - 1.0


def get_portfolio_cagr(days: int = 365) -> Optional[float]:
    """CAGR from portfolio value history or deals PnL."""
    try:
        from db import _conn, now_ts
        since = now_ts() - (days * 86400)
        con = _conn()
        rows = con.execute(
            """
            SELECT opened_at, closed_at, entry_avg, base_amount, realized_pnl_quote
            FROM deals
            WHERE state='CLOSED' AND closed_at IS NOT NULL AND closed_at >= ?
            ORDER BY closed_at ASC
            """,
            (since,),
        ).fetchall()
        con.close()
        if not rows:
            return None
        # Approximate: sum of cost bases at start, sum of values at end
        total_cost = 0.0
        total_realized = 0.0
        for r in rows:
            entry = float(r["entry_avg"] or 0)
            base = float(r["base_amount"] or 0)
            pnl = float(r["realized_pnl_quote"] or 0)
            total_cost += entry * base if entry > 0 and base > 0 else 0
            total_realized += pnl
        if total_cost <= 0:
            return None
        final = total_cost + total_realized
        years = days / 365.25
        return compute_cagr(total_cost, final, years)
    except Exception as e:
        logger.debug("CAGR calc failed: %s", e)
        return None


def get_portfolio_heat_map_data(
    bots: List[Dict[str, Any]],
    deals: List[Dict[str, Any]],
    portfolio_total: float,
) -> Dict[str, Any]:
    """
    Data for portfolio heat map: % per symbol, sector, strategy.
    Highlight overconcentration.
    """
    by_symbol: Dict[str, float] = {}
    by_strategy: Dict[str, float] = {}
    total_exposure = 0.0
    for d in deals or []:
        if d.get("state") != "OPEN":
            continue
        entry = float(d.get("entry_avg") or 0)
        base = float(d.get("base_amount") or 0)
        val = entry * base
        sym = str(d.get("symbol") or "")
        strat = str(d.get("exit_strategy") or d.get("entry_strategy") or "unknown")
        if sym:
            by_symbol[sym] = by_symbol.get(sym, 0) + val
        by_strategy[strat] = by_strategy.get(strat, 0) + val
        total_exposure += val
    pct_by_symbol = {s: (v / portfolio_total * 100) if portfolio_total > 0 else 0 for s, v in by_symbol.items()}
    pct_by_strategy = {s: (v / portfolio_total * 100) if portfolio_total > 0 else 0 for s, v in by_strategy.items()}
    overconcentrated = [s for s, p in pct_by_symbol.items() if p > 15]
    return {
        "by_symbol": pct_by_symbol,
        "by_strategy": pct_by_strategy,
        "total_exposure_pct": (total_exposure / portfolio_total * 100) if portfolio_total > 0 else 0,
        "overconcentration_risks": overconcentrated,
    }
