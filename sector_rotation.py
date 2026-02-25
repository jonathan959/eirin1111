"""
Sector Rotation Strategy - track sector momentum, rotate capital quarterly.

Uses sector_performance_history and stock_metadata sector mapping.
"""
import os
import time
import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from db import _conn, now_ts

logger = logging.getLogger(__name__)

REBALANCE_FREQUENCY = os.getenv("REBALANCE_FREQUENCY", "quarterly").strip().lower()
TARGET_ALLOCATIONS = os.getenv("TARGET_ALLOCATIONS", "").strip()


def _parse_target_allocations() -> Dict[str, float]:
    """Parse TARGET_ALLOCATIONS env: 'Technology:0.4,Healthcare:0.3,Financial:0.3'."""
    if not TARGET_ALLOCATIONS:
        return {
            "Technology": 0.25, "Healthcare": 0.20, "Financial": 0.20,
            "Consumer Cyclical": 0.15, "Consumer Defensive": 0.10,
            "Energy": 0.05, "Industrial": 0.05,
        }
    out = {}
    for part in TARGET_ALLOCATIONS.split(","):
        if ":" in part:
            k, v = part.split(":", 1)
            try:
                out[k.strip()] = float(v.strip())
            except ValueError:
                pass
    return out if out else {"Technology": 0.25, "Healthcare": 0.20, "Financial": 0.20}


def record_sector_performance(sector: str, quarter_ts: int, return_pct: float, momentum_score: float, rank: int) -> None:
    """Store sector performance for rotation analysis."""
    con = _conn()
    try:
        con.execute(
            """
            INSERT INTO sector_performance_history(sector, quarter_ts, return_pct, momentum_score, rank)
            VALUES (?,?,?,?,?)
            """,
            (str(sector), int(quarter_ts), float(return_pct), float(momentum_score), int(rank)),
        )
        con.commit()
    finally:
        con.close()


def get_sector_momentum(days: int = 90) -> List[Dict[str, Any]]:
    """Get sector performance ranked by momentum (latest quarter)."""
    since_ts = now_ts() - (days * 86400)
    con = _conn()
    rows = con.execute(
        """
        SELECT sector, quarter_ts, return_pct, momentum_score, rank
        FROM sector_performance_history
        WHERE quarter_ts >= ?
        ORDER BY quarter_ts DESC, rank ASC
        LIMIT 100
        """,
        (since_ts,),
    ).fetchall()
    con.close()
    return [dict(r) for r in rows]


def get_rotation_suggestions(current_allocations: Dict[str, float]) -> List[Dict[str, Any]]:
    """
    Compare current allocations to target. Return suggestions to rebalance.
    Drift > 10% triggers rebalance.
    """
    targets = _parse_target_allocations()
    suggestions = []
    total_current = sum(current_allocations.values()) or 1.0
    for sector, target_pct in targets.items():
        current_pct = current_allocations.get(sector, 0.0) / total_current
        drift = current_pct - target_pct
        if abs(drift) > 0.10:
            suggestions.append({
                "sector": sector,
                "current_pct": round(current_pct * 100, 1),
                "target_pct": round(target_pct * 100, 1),
                "drift_pct": round(drift * 100, 1),
                "action": "reduce" if drift > 0 else "increase",
            })
    return suggestions


@dataclass
class SectorRotationSignal:
    overweight_sectors: List[str]
    underweight_sectors: List[str]
    suggestions: List[Dict[str, Any]]


def sector_rotation_signal(
    sector_returns: Optional[Dict[str, float]] = None,
    current_allocations: Optional[Dict[str, float]] = None,
) -> SectorRotationSignal:
    """
    Detect sector momentum shifts. Overweight strong, underweight weak.
    """
    current_allocations = current_allocations or {}
    sector_returns = sector_returns or {}
    suggestions = get_rotation_suggestions(current_allocations)
    sorted_sectors = sorted(sector_returns.items(), key=lambda x: -float(x[1] or 0))
    top_n = max(2, len(sorted_sectors) // 3)
    overweight = [s[0] for s in sorted_sectors[:top_n] if (s[1] or 0) > 0]
    underweight = [s[0] for s in sorted_sectors[-top_n:] if (s[1] or 0) < 0]
    return SectorRotationSignal(
        overweight_sectors=overweight,
        underweight_sectors=underweight,
        suggestions=suggestions,
    )
