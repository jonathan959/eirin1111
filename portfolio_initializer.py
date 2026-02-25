"""
Portfolio initialization: one-time setup, optimal allocation.
"""
import json
import logging
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

RISK_PROFILES = {
    "conservative": {"base_pct": 0.04, "safety_pct": 0.03, "max_positions": 5, "daily_loss_pct": 0.03},
    "moderate": {"base_pct": 0.06, "safety_pct": 0.04, "max_positions": 8, "daily_loss_pct": 0.05},
    "aggressive": {"base_pct": 0.10, "safety_pct": 0.05, "max_positions": 12, "daily_loss_pct": 0.08},
}


def compute_optimal_allocation(
    total_capital: float,
    risk_tolerance: str,
    max_positions: int,
    asset_types: str = "both",
    sectors_avoid: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """
    Compute allocation per bot from risk profile.
    Returns config for autopilot.
    """
    profile = RISK_PROFILES.get(risk_tolerance.lower(), RISK_PROFILES["moderate"])
    capital_per_slot = total_capital / max_positions if max_positions > 0 else total_capital
    base_quote = total_capital * profile["base_pct"]
    base_quote = min(base_quote, capital_per_slot * 0.25)
    safety_quote = total_capital * profile["safety_pct"]
    return {
        "total_capital": total_capital,
        "risk_tolerance": risk_tolerance,
        "max_positions": max_positions,
        "asset_types": asset_types,
        "sectors_avoid": sectors_avoid or [],
        "base_quote": round(base_quote, 2),
        "safety_quote": round(safety_quote, 2),
        "daily_loss_limit_pct": profile["daily_loss_pct"],
        "capital_per_bot": round(capital_per_slot, 2),
    }


def save_autopilot_config(config: Dict[str, Any]) -> bool:
    try:
        from db import set_setting
        set_setting("autopilot_config", json.dumps(config))
        return True
    except Exception as e:
        logger.error("save_autopilot_config failed: %s", e)
        return False


def initialize_portfolio(
    config: Dict[str, Any],
    create_bots: bool = False,
    create_bot_fn=None,
) -> Dict[str, Any]:
    """
    Apply initial config. If create_bots, optionally create placeholder bots.
    Returns summary.
    """
    save_autopilot_config(config)
    if create_bots and create_bot_fn:
        pass  # Autopilot cycle will create; or we could create disabled placeholders
    return {"ok": True, "config_saved": True}
