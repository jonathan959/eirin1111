# explore_v2.py
"""
Explore V2: Risk-adjusted, regime-aware, diversified recommendations.

When EXPLORE_V2=1, apply hard gates and enhanced scoring.
Feature flag: EXPLORE_V2 (default: 1)

Gate failures set EXPLORE_V2_GATE:* on risk_flags; worker_api maps those to
db.explore_signals status=rejected on each scan so the main Explore table
and "Recently Rejected" stay mutually exclusive.
"""

import os
import logging
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

_ENABLED = os.getenv("EXPLORE_V2", "1").strip().lower() in ("1", "true", "yes", "y", "on")

# Gates (env overridable) — relaxed defaults so the scanner actually produces results
MIN_24H_QUOTE_VOLUME = float(os.getenv("EXPLORE_V2_MIN_VOLUME", "100000"))
MAX_SPREAD_BPS = float(os.getenv("EXPLORE_V2_MAX_SPREAD_BPS", "200"))
VOL_SPIKE_MULT = float(os.getenv("EXPLORE_V2_VOL_SPIKE_MULT", "4.0"))

MAX_LOW_LIQUIDITY_IN_TOP = int(os.getenv("EXPLORE_V2_MAX_LOW_LIQ", "3"))
LOW_LIQUIDITY_VOLUME_THRESH = float(os.getenv("EXPLORE_V2_LOW_LIQ_VOL", "50000"))


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
    Enhance recommendation score with risk penalties and quality bonuses.
    Returns (adjusted_score, extra_reasons).
    """
    if not _ENABLED:
        return base_score, []
    # Apply adaptive scoring weights from recommendation_validator calibration
    try:
        from adaptive_scorer import apply_adaptive_score
        base_score = apply_adaptive_score(base_score, regime=regime)
    except Exception:
        pass
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

    ret_30d = snap.get("return_30d") or snap.get("ret_30d")
    ret_90d = snap.get("return_90d") or snap.get("ret_90d")
    if ret_30d is not None:
        ret_30d = float(ret_30d)
        # Penalize material 30d drawdowns (<= -20%); scale so typical -20%..-35% trims score.
        if ret_30d <= -0.20:
            penalty = min(8.0, max(1.0, abs(ret_30d) * 20.0))
            score -= penalty
            reasons.append(f"30d crash: -{penalty:.0f}")
    if ret_90d is not None:
        ret_90d = float(ret_90d)
        if ret_90d < -0.40:
            penalty = min(5, abs(ret_90d) * 10)
            score -= penalty
            reasons.append(f"90d drop: -{penalty:.0f}")

    regime_1d = snap.get("regime_1d") or regime
    regime_4h = snap.get("regime_4h") or ""
    if regime_1d and regime_4h:
        r1 = str(regime_1d).upper()
        r4 = str(regime_4h).upper()
        bullish = {"BULL", "BREAKOUT"}
        if r1 in bullish and r4 in bullish:
            bonus = 5
            score += bonus
            reasons.append(f"Multi-TF alignment: +{bonus}")
        elif (r1 in bullish) != (r4 in bullish):
            penalty = 3
            score -= penalty
            reasons.append(f"TF divergence: -{penalty}")

    return max(0.0, min(92.0, score)), reasons


def diversify_picks(
    items: List[Dict[str, Any]],
    top_k: int = 20,
    cluster_key: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """
    Take top K across different clusters to avoid correlated picks.
    Also caps low-liquidity names to prevent the list being dominated by micro-caps.
    """
    if not _ENABLED or not items:
        return items[:top_k]

    if cluster_key and cluster_key in (items[0] if items else {}):
        by_cluster: Dict[str, List[Dict]] = {}
        for it in items:
            c = str(it.get(cluster_key) or "default")
            by_cluster.setdefault(c, []).append(it)
        per_cluster = max(2, top_k // max(1, len(by_cluster)))
        out: List[Dict[str, Any]] = []
        for c, lst in sorted(by_cluster.items(), key=lambda x: -max((float(i.get("score") or 0) for i in x[1]), default=0.0) if x[1] else 0.0):
            out.extend(lst[:per_cluster])
        out.sort(key=lambda x: float(x.get("score") or 0), reverse=True)
    else:
        out = list(items)

    low_liq_count = 0
    final: List[Dict[str, Any]] = []
    for it in out:
        vol = it.get("volume") or it.get("volume_24h") or 0
        try:
            vol = float(vol)
        except (TypeError, ValueError):
            vol = 0
        is_low_liq = vol > 0 and vol < LOW_LIQUIDITY_VOLUME_THRESH
        if is_low_liq:
            if low_liq_count >= MAX_LOW_LIQUIDITY_IN_TOP:
                continue
            low_liq_count += 1
        final.append(it)
        if len(final) >= top_k:
            break

    return final


def compute_score_breakdown(
    score: float,
    metrics: Dict[str, Any],
    regime: str,
) -> List[str]:
    """Return the top 3 contributors to the score for explainability."""
    contributors = []

    trend = metrics.get("weekly_trend") or metrics.get("trend")
    if trend:
        t = str(trend).lower()
        if "up" in t or "bull" in t:
            contributors.append(f"Trend: {trend} (+)")
        elif "down" in t or "bear" in t:
            contributors.append(f"Trend: {trend} (-)")

    regime_label = regime or metrics.get("regime") or "unknown"
    if str(regime_label).upper() in ("BULL", "BREAKOUT"):
        contributors.append(f"Regime: {regime_label} (+)")
    elif str(regime_label).upper() in ("BEAR", "RISK_OFF"):
        contributors.append(f"Regime: {regime_label} (-)")

    vol = metrics.get("atr_pct") or metrics.get("volatility")
    if vol is not None:
        v = float(vol)
        if v < 0.02:
            contributors.append(f"Low volatility: {v:.1%} (+)")
        elif v > 0.06:
            contributors.append(f"High volatility: {v:.1%} (-)")

    ret_30 = metrics.get("return_30d") or metrics.get("ret_30d")
    if ret_30 is not None:
        r = float(ret_30)
        if r > 0.05:
            contributors.append(f"30d return: {r:.1%} (+)")
        elif r < -0.10:
            contributors.append(f"30d return: {r:.1%} (-)")

    winrate = metrics.get("winrate") or metrics.get("win_rate")
    if winrate is not None:
        w = float(winrate)
        if w > 0.6:
            contributors.append(f"Win rate: {w:.0%} (+)")

    pf = metrics.get("profit_factor") or metrics.get("expected_return")
    if pf is not None:
        p = float(pf)
        if p > 1.5:
            contributors.append(f"Profit factor: {p:.1f} (+)")

    return contributors[:3] if contributors else [f"Base score: {score:.0f}"]


def is_enabled() -> bool:
    return _ENABLED
