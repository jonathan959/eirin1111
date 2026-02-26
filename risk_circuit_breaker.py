"""
Global Risk Circuit Breakers — Single place for all risk gates.
Enforced BEFORE any order placement. When tripped: pause and Discord alert.
"""
import os
import time
import logging
from typing import Any, Dict, Optional, Tuple

logger = logging.getLogger(__name__)

_DEFAULT_MAX_EXPOSURE = float(os.getenv("MAX_TOTAL_EXPOSURE_PCT", "0.50"))
# Override: if set, use this minimum for exposure (unblocks bots with strict 35% limit)
_OVERRIDE_MIN_EXPOSURE = os.getenv("OVERRIDE_MAX_EXPOSURE_PCT", "").strip()
_OVERRIDE_MIN_EXPOSURE_VAL = float(_OVERRIDE_MIN_EXPOSURE) if _OVERRIDE_MIN_EXPOSURE and _OVERRIDE_MIN_EXPOSURE.replace(".", "").isdigit() else None


def check_circuit_breakers(
    equity: float,
    daily_realized_pnl: float,
    portfolio_drawdown: float,
    portfolio_exposure_pct: float,
    open_deals_count: int,
    total_exposure_usd: float,
    max_total_exposure_pct: Optional[float] = None,
    max_concurrent_deals: int = 6,
    max_daily_loss_pct: float = 0.06,
    max_drawdown_pct: float = 0.20,
    max_exposure_pct: Optional[float] = None,
) -> Tuple[bool, Optional[str]]:
    """
    Check all circuit breakers. Returns (ok, reason).
    If not ok, reason explains what tripped.
    """
    if max_total_exposure_pct is None:
        max_total_exposure_pct = _DEFAULT_MAX_EXPOSURE
    if max_exposure_pct is None:
        max_exposure_pct = max_total_exposure_pct
    # Apply override: if OVERRIDE_MAX_EXPOSURE_PCT set (e.g. 0.60), use as minimum limit
    if _OVERRIDE_MIN_EXPOSURE_VAL is not None:
        max_total_exposure_pct = max(max_total_exposure_pct, _OVERRIDE_MIN_EXPOSURE_VAL)
        max_exposure_pct = max(max_exposure_pct, _OVERRIDE_MIN_EXPOSURE_VAL)
    if equity <= 0:
        return True, None

    # 1. Daily loss limit
    if max_daily_loss_pct > 0 and daily_realized_pnl < 0:
        loss_pct = abs(daily_realized_pnl) / equity
        if loss_pct >= max_daily_loss_pct:
            reason = f"Daily loss circuit breaker: {loss_pct*100:.2f}% >= {max_daily_loss_pct*100:.2f}%"
            return False, reason

    # 2. Max drawdown
    if max_drawdown_pct > 0 and portfolio_drawdown >= max_drawdown_pct:
        reason = f"Max drawdown kill-switch: {portfolio_drawdown*100:.2f}% >= {max_drawdown_pct*100:.2f}%"
        return False, reason

    # 3. Portfolio exposure (total)
    if max_total_exposure_pct > 0 and total_exposure_usd > 0:
        exp_pct = total_exposure_usd / equity
        if exp_pct >= max_total_exposure_pct:
            reason = f"Portfolio exposure limit: {exp_pct*100:.2f}% >= {max_total_exposure_pct*100:.2f}%"
            return False, reason

    # 4. Per-portfolio exposure pct (from context)
    if max_exposure_pct > 0 and portfolio_exposure_pct >= max_exposure_pct:
        logger.info("Exposure check: portfolio=%.2f%% limit=%.2f%% (override=%s)", portfolio_exposure_pct * 100, max_exposure_pct * 100, "yes" if _OVERRIDE_MIN_EXPOSURE_VAL else "no")
        reason = f"Portfolio exposure limit: {portfolio_exposure_pct*100:.2f}% >= {max_exposure_pct*100:.2f}%"
        return False, reason

    # 5. Max concurrent deals
    if max_concurrent_deals > 0 and open_deals_count >= max_concurrent_deals:
        reason = f"Max concurrent deals reached: {open_deals_count} >= {max_concurrent_deals}"
        return False, reason

    return True, None


def trip_and_alert(
    reason: str,
    pause_hours: int = 6,
    bot_label: str = "",
) -> None:
    """
    When circuit breaker trips: log warning and send Discord alert.
    Only sets global pause for critical reasons (drawdown kill-switch, explicit global triggers).
    Routine per-bot exposure/deal-count limits do NOT pause all bots.
    """
    is_critical = any(k in reason.lower() for k in ("drawdown", "kill", "emergency", "global"))
    if is_critical:
        try:
            from db import set_setting
            set_setting("global_pause", "1")
            set_setting("global_pause_until", str(int(time.time()) + (pause_hours * 3600)))
            logger.warning("CRITICAL circuit breaker: %s — global pause set for %sh", reason, pause_hours)
        except Exception as e:
            logger.warning("Failed to set global pause: %s", e)
    else:
        logger.warning("Circuit breaker tripped (bot-level only): %s", reason)

    if os.getenv("DISCORD_NOTIFY_RISK", "1").strip().lower() in ("1", "true", "yes", "y", "on"):
        try:
            from discord_notifications import DiscordNotifier
            notifier = DiscordNotifier()
            if notifier.config.webhook_url:
                notifier.notify_risk_alert(
                    alert_type="CIRCUIT_BREAKER",
                    severity="critical",
                    message=reason,
                    details={"bot_label": bot_label or "System", "pause_hours": pause_hours},
                )
        except Exception as e:
            logger.debug("Discord risk alert failed: %s", e)
