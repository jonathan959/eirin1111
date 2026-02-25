"""
Adaptive scoring: applies calibrated weights to raw recommendation scores.
Uses get_scoring_weights from recommendation_validator (trained by run_calibration).
"""
import logging
from typing import Optional

logger = logging.getLogger(__name__)


def _score_range_key(score: float) -> str:
    """Map raw score to weight key (0-40, 40-60, 60-80, 80-100)."""
    if score >= 80:
        return "score_80_100"
    if score >= 60:
        return "score_60_80"
    if score >= 40:
        return "score_40_60"
    return "score_0_40"


def apply_adaptive_score(raw_score: float, regime: Optional[str] = None) -> float:
    """
    Apply calibrated scoring weights to raw recommendation score.
    Higher win-rate ranges/regimes get slight boost, lower get slight reduction.
    """
    try:
        from recommendation_validator import get_scoring_weights
        weights = get_scoring_weights()
    except Exception as e:
        logger.debug("adaptive_scorer: no weights available: %s", e)
        return raw_score

    mult = 1.0
    range_key = _score_range_key(raw_score)
    range_mult = weights.get(range_key, 1.0)
    if isinstance(range_mult, (int, float)):
        mult *= float(range_mult)

    regime_mult = 1.0
    if regime:
        regime_mults = weights.get("regime_multipliers") or {}
        regime_mult = regime_mults.get(str(regime).strip(), regime_mults.get("unknown", 1.0))
        if isinstance(regime_mult, (int, float)):
            mult *= float(regime_mult)

    adjusted = raw_score * mult
    return round(max(0.0, min(100.0, adjusted)), 1)
