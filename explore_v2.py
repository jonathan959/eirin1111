# explore_v2.py
"""
Explore V2: Risk-adjusted, regime-aware, diversified recommendations.

When EXPLORE_V2=1, apply hard gates and enhanced scoring.
Feature flag: EXPLORE_V2 (default: 0)
"""

import os
import logging
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

_ENABLED = os.getenv("EXPLORE_V2", "1").strip().lower() in ("1", "true", "yes", "y", "on")

# Gates (env overridable)
MIN_24H_QUOTE_VOLUME = float(os.getenv("EXPLORE_V2_MIN_VOLUME", "5000"))
MAX_SPREAD_BPS = float(os.getenv("EXPLORE_V2_MAX_SPREAD_BPS", "100"))
VOL_SPIKE_MULT = float(os.getenv("EXPLORE_V2_VOL_SPIKE_MULT", "2.5"))


def apply_universe_gates(
    symbol: str,
    volume_24h_quote: Optional[float] = None,
    spread_bps: Optional[float] = None,
    volatility_pct: Optional[float] = None,
    volatility_avg_pct: Optional[float] = None,
) -> Tuple[bool, Optional[str]]:
    """
    Apply hard gates for universe inclusion.
    Returns (pass, fail_reason).
    """
    if not _ENABLED:
        return True, None
    if volume_24h_quote is not None and volume_24h_quote < MIN_24H_QUOTE_VOLUME:
        return False, f"Low volume: {volume_24h_quote:.0f} < {MIN_24H_QUOTE_VOLUME:.0f}"
    if spread_bps is not None and spread_bps > MAX_SPREAD_BPS:
        return False, f"Wide spread: {spread_bps:.0f} bps > {MAX_SPREAD_BPS:.0f}"
    if volatility_pct is not None and volatility_avg_pct is not None and volatility_avg_pct > 0:
        mult = volatility_pct / volatility_avg_pct
        if mult > VOL_SPIKE_MULT:
            return False, f"Volatility spike: {mult:.1f}x avg"
    return True, None


def enhance_score(
    base_score: float,
    snap: Dict[str, Any],
    regime: str,
    spread_bps: Optional[float] = None,
    volatility_pct: Optional[float] = None,
) -> Tuple[float, List[str]]:
    """
    Enhance recommendation score with risk penalties.
    Returns (adjusted_score, extra_reasons).
    """
    if not _ENABLED:
        return base_score, []
    score = base_score
    reasons = []
    if spread_bps is not None and spread_bps > 30:
        penalty = min(15, (spread_bps - 30) * 0.2)
        score -= penalty
        reasons.append(f"Spread penalty: -{penalty:.0f}")
    if volatility_pct is not None and volatility_pct > 0.08:
        penalty = min(10, (volatility_pct - 0.08) * 100)
        score -= penalty
        reasons.append(f"High vol penalty: -{penalty:.0f}")
    return max(0.0, min(100.0, score)), reasons


def diversify_picks(
    items: List[Dict[str, Any]],
    top_k: int = 20,
    cluster_key: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """
    Take top K across different clusters to avoid correlated picks.
    cluster_key: if provided, spread picks across clusters.
    """
    if not _ENABLED or not items:
        return items[:top_k]
    if not cluster_key or cluster_key not in (items[0] or {}):
        return items[:top_k]
    by_cluster: Dict[str, List[Dict]] = {}
    for it in items:
        c = str(it.get(cluster_key) or "default")
        by_cluster.setdefault(c, []).append(it)
    out = []
    per_cluster = max(1, top_k // max(1, len(by_cluster)))
    for c, lst in sorted(by_cluster.items()):
        out.extend(lst[:per_cluster])
    out.sort(key=lambda x: float(x.get("score") or 0), reverse=True)
    return out[:top_k]


def is_enabled() -> bool:
    return _ENABLED
