"""
Crypto market cycle detection - Bitcoin halving cycle (4-year pattern).

Phases: accumulation -> markup -> distribution -> markdown
Adjust strategy based on cycle position.
"""
import logging
import os
import time
from typing import Any, Dict, Optional, Tuple

logger = logging.getLogger(__name__)

# Bitcoin halving dates (approx unix ts)
HALVING_DATES = [
    210240,      # 2012-11-28
    1415232000,  # 2014-07-09
    1519689600,  # 2016-07-09
    1585699200,  # 2020-05-11
    1743465600,  # 2024-04-19
]


def get_cycle_phase(now_ts: Optional[int] = None) -> Dict[str, Any]:
    """
    Determine current cycle phase relative to last halving.
    Phases (months from halving):
    - 0-6: accumulation
    - 6-12: early markup
    - 12-18: markup
    - 18-24: late markup / distribution
    - 24-36: distribution / markdown
    - 36-48: markdown
    """
    now = now_ts or int(time.time())
    last_halving = HALVING_DATES[-1]
    months_since = (now - last_halving) / (30.44 * 86400)
    phase = "unknown"
    if months_since < 0:
        phase = "pre_halving"
    elif months_since < 6:
        phase = "accumulation"
    elif months_since < 12:
        phase = "early_markup"
    elif months_since < 18:
        phase = "markup"
    elif months_since < 24:
        phase = "late_markup"
    elif months_since < 36:
        phase = "distribution"
    else:
        phase = "markdown"
    return {
        "phase": phase,
        "months_since_halving": round(months_since, 1),
        "last_halving_ts": last_halving,
        "next_halving_approx_year": 2028,
    }


def cycle_strategy_hint(phase: str) -> Dict[str, str]:
    """Suggested strategy adjustments per phase."""
    hints = {
        "accumulation": "favor_dca",
        "early_markup": "favor_trend",
        "markup": "favor_trend",
        "late_markup": "reduce_exposure",
        "distribution": "defensive",
        "markdown": "favor_dca_defensive",
        "pre_halving": "accumulate",
    }
    return {"hint": hints.get(phase, "neutral"), "phase": phase}
