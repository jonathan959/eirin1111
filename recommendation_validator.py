"""
Recommendation performance validation and adaptive scoring calibration.

Tracks outcomes of recommendations that led to bot creation and closed deals.
Uses historical performance to calibrate scoring weights.
"""
import json
import logging
from typing import Any, Dict, List

from db import (
    get_recommendation_performance_stats,
    save_scoring_calibration_log,
    set_setting,
    get_setting,
)

logger = logging.getLogger(__name__)

SCORING_WEIGHTS_KEY = "scoring_weights"
DEFAULT_WEIGHTS = {
    "score_80_100": 1.0,
    "score_60_80": 1.0,
    "score_40_60": 1.0,
    "score_0_40": 1.0,
    "regime_multipliers": {},
}


def get_scoring_weights() -> Dict[str, Any]:
    """Get current scoring weights from settings. Returns defaults if not set."""
    raw = get_setting(SCORING_WEIGHTS_KEY)
    if not raw:
        return DEFAULT_WEIGHTS.copy()
    try:
        out = json.loads(raw)
        return {**DEFAULT_WEIGHTS, **out}
    except Exception:
        return DEFAULT_WEIGHTS.copy()


def run_calibration(window_days: int = 30) -> Dict[str, Any]:
    """
    Analyze closed recommendation outcomes and produce suggested scoring weights.
    Logs to scoring_calibration_log. Updates settings if enough data.
    """
    stats = get_recommendation_performance_stats(days=window_days)
    total = stats["total_closed"]
    if total < 5:
        logger.info("Calibration skipped: only %d closed recommendations (need 5+)", total)
        return {"ok": False, "reason": "insufficient_data", "total_closed": total}

    by_range = {r["range"]: r for r in stats["by_score_range"]}
    by_regime = {r["regime"]: r for r in stats["by_regime"]}

    # Compute multipliers: higher win rate -> slightly boost (max 1.2), lower -> reduce (min 0.7)
    changes: Dict[str, Any] = {"score_ranges": {}, "regimes": {}}
    new_weights = get_scoring_weights()

    range_map = {"80-100": "score_80_100", "60-80": "score_60_80", "40-60": "score_40_60", "0-40": "score_0_40"}
    for rng, key in range_map.items():
        br = by_range.get(rng)
        if br and br["total"] >= 2:
            wr = br["win_rate"] / 100.0
            # 50% baseline -> 1.0; 70% -> 1.1; 30% -> 0.85
            mult = 0.7 + (wr * 0.8)
            mult = max(0.7, min(1.2, mult))
            new_weights[key] = round(mult, 2)
            changes["score_ranges"][rng] = {"win_rate": br["win_rate"], "multiplier": mult}

    regime_mults = new_weights.get("regime_multipliers", {})
    for reg, br in by_regime.items():
        if reg and br["total"] >= 2:
            wr = br["win_rate"] / 100.0
            mult = max(0.7, min(1.2, 0.7 + (wr * 0.8)))
            regime_mults[reg] = round(mult, 2)
            changes["regimes"][reg] = {"win_rate": br["win_rate"], "multiplier": mult}
    new_weights["regime_multipliers"] = regime_mults

    import time
    version = f"v{int(time.time())}"
    set_setting(SCORING_WEIGHTS_KEY, json.dumps(new_weights))
    save_scoring_calibration_log(
        scoring_version=version,
        changes_json=json.dumps(changes),
        analysis_window_days=window_days,
        notes=f"Calibrated from {total} closed recommendations. Win rate: {stats['win_rate']:.1f}%",
    )
    logger.info("Scoring calibration complete: %s from %d closed recs", version, total)
    return {
        "ok": True,
        "scoring_version": version,
        "total_closed": total,
        "win_rate_pct": stats["win_rate"],
        "changes": changes,
        "new_weights": new_weights,
    }
