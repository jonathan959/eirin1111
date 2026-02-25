# risk_engine.py
"""
Centralized Risk Engine — Single source of truth for "can we trade?"

All order placement MUST pass through RiskEngine checks when RISK_ENGINE_ENABLED=1.
Feature flag: RISK_ENGINE_ENABLED (default: 0)
"""

import os
import time
import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

def _enabled() -> bool:
    return os.getenv("RISK_ENGINE_ENABLED", "1").strip().lower() in ("1", "true", "yes", "y", "on")


@dataclass
class RiskConfig:
    """Risk limits from env + per-bot overrides. Sane defaults."""
    MAX_ASSET_EXPOSURE_PCT: float = 0.03
    MAX_TOTAL_EXPOSURE_PCT: float = 0.20
    MAX_DAILY_LOSS_PCT: float = 0.05
    MAX_TRADES_PER_DAY: int = 20
    MIN_24H_QUOTE_VOLUME: float = 5000.0
    MAX_SPREAD_BPS: float = 50.0
    VOLATILITY_SPIKE_MULTIPLIER: float = 2.0
    ERROR_CIRCUIT_BREAKER_N: int = 0

    def __post_init__(self):
        self.MAX_ASSET_EXPOSURE_PCT = float(os.getenv("RISK_MAX_ASSET_EXPOSURE_PCT", str(self.MAX_ASSET_EXPOSURE_PCT)))
        self.MAX_TOTAL_EXPOSURE_PCT = float(os.getenv("RISK_MAX_TOTAL_EXPOSURE_PCT", str(self.MAX_TOTAL_EXPOSURE_PCT)))
        self.MAX_DAILY_LOSS_PCT = float(os.getenv("RISK_MAX_DAILY_LOSS_PCT", str(self.MAX_DAILY_LOSS_PCT)))
        self.MAX_TRADES_PER_DAY = int(os.getenv("RISK_MAX_TRADES_PER_DAY", str(self.MAX_TRADES_PER_DAY)))
        self.MIN_24H_QUOTE_VOLUME = float(os.getenv("RISK_MIN_24H_QUOTE_VOLUME", str(self.MIN_24H_QUOTE_VOLUME)))
        self.MAX_SPREAD_BPS = float(os.getenv("RISK_MAX_SPREAD_BPS", str(self.MAX_SPREAD_BPS)))
        self.VOLATILITY_SPIKE_MULTIPLIER = float(os.getenv("RISK_VOLATILITY_SPIKE_MULTIPLIER", str(self.VOLATILITY_SPIKE_MULTIPLIER)))
        self.ERROR_CIRCUIT_BREAKER_N = int(os.getenv("ERROR_CIRCUIT_BREAKER_N", str(self.ERROR_CIRCUIT_BREAKER_N)))


@dataclass
class RiskContext:
    """Context passed to risk checks."""
    bot_id: int
    symbol: str
    balance_total_usd: float
    balance_free_usd: float
    positions_usd: Dict[str, float]  # symbol -> value
    symbol_position_usd: float
    spread_bps: Optional[float] = None
    volume_24h_quote: Optional[float] = None
    volatility_pct: Optional[float] = None
    volatility_avg_pct: Optional[float] = None
    daily_loss_pct: Optional[float] = None
    trades_today: int = 0
    last_error_count: int = 0
    config: Optional[RiskConfig] = None
    # Per-bot overrides from DB
    max_total_exposure_pct: Optional[float] = None
    per_symbol_exposure_pct: Optional[float] = None
    daily_loss_limit_pct: Optional[float] = None


def _get_config() -> RiskConfig:
    return RiskConfig()


def compute_exposure(
    balance_total: float,
    positions: Dict[str, float],
    symbol: str,
) -> Tuple[float, float]:
    """
    Compute asset exposure and total exposure.
    Returns (asset_exposure_pct, total_exposure_pct).
    """
    if balance_total <= 0:
        return 0.0, 0.0
    total_pos = sum(positions.values())
    asset_pos = positions.get(symbol, 0.0)
    total_exposure_pct = total_pos / balance_total if balance_total > 0 else 0.0
    asset_exposure_pct = asset_pos / balance_total if balance_total > 0 else 0.0
    return asset_exposure_pct, total_exposure_pct


def can_open_trade(context: RiskContext) -> Tuple[bool, str]:
    """
    Check if a new trade can be opened.
    Returns (allowed, reason).
    """
    if not _enabled():
        return True, ""

    cfg = context.config or _get_config()
    max_total = context.max_total_exposure_pct if context.max_total_exposure_pct is not None else cfg.MAX_TOTAL_EXPOSURE_PCT
    max_asset = context.per_symbol_exposure_pct if context.per_symbol_exposure_pct is not None else cfg.MAX_ASSET_EXPOSURE_PCT

    # Exposure
    positions = context.positions_usd or {}
    asset_exp, total_exp = compute_exposure(
        context.balance_total_usd,
        positions,
        context.symbol,
    )
    if total_exp >= max_total:
        reason = f"Total exposure {total_exp*100:.2f}% >= max {max_total*100:.1f}%"
        logger.warning("RISK_BLOCKED can_open_trade: %s", reason)
        return False, reason
    if asset_exp >= max_asset:
        reason = f"Asset exposure {asset_exp*100:.2f}% >= max {max_asset*100:.1f}%"
        logger.warning("RISK_BLOCKED can_open_trade: %s", reason)
        return False, reason

    # Daily loss
    daily_limit = context.daily_loss_limit_pct if context.daily_loss_limit_pct is not None else cfg.MAX_DAILY_LOSS_PCT
    if context.daily_loss_pct is not None and context.daily_loss_pct <= -daily_limit:
        reason = f"Daily loss {context.daily_loss_pct*100:.2f}% exceeds limit {daily_limit*100:.1f}%"
        logger.warning("RISK_BLOCKED can_open_trade: %s", reason)
        return False, reason

    # Trade limit
    if context.trades_today >= cfg.MAX_TRADES_PER_DAY:
        reason = f"Max trades today ({context.trades_today}) reached"
        logger.warning("RISK_BLOCKED can_open_trade: %s", reason)
        return False, reason

    # Liquidity gate
    if context.volume_24h_quote is not None and context.volume_24h_quote < cfg.MIN_24H_QUOTE_VOLUME:
        reason = f"Low volume: {context.volume_24h_quote:.0f} < {cfg.MIN_24H_QUOTE_VOLUME:.0f}"
        logger.warning("RISK_BLOCKED can_open_trade: %s", reason)
        return False, reason

    # Spread gate
    if context.spread_bps is not None and context.spread_bps > cfg.MAX_SPREAD_BPS:
        reason = f"Spread {context.spread_bps:.0f} bps > max {cfg.MAX_SPREAD_BPS:.0f}"
        logger.warning("RISK_BLOCKED can_open_trade: %s", reason)
        return False, reason

    # Volatility spike
    if context.volatility_pct is not None and context.volatility_avg_pct is not None and context.volatility_avg_pct > 0:
        mult = context.volatility_pct / context.volatility_avg_pct
        if mult > cfg.VOLATILITY_SPIKE_MULTIPLIER:
            reason = f"Volatility spike: {mult:.1f}x avg"
            logger.warning("RISK_BLOCKED can_open_trade: %s", reason)
            return False, reason

    # Circuit breaker
    if cfg.ERROR_CIRCUIT_BREAKER_N > 0 and context.last_error_count >= cfg.ERROR_CIRCUIT_BREAKER_N:
        reason = f"Circuit breaker: {context.last_error_count} consecutive errors"
        logger.warning("RISK_BLOCKED can_open_trade: %s", reason)
        return False, reason

    return True, ""


def can_add_dca(context: RiskContext) -> Tuple[bool, str]:
    """
    Check if a DCA add is allowed.
    Same as can_open_trade but may have different thresholds in future.
    """
    return can_open_trade(context)


def should_halt_bot(context: RiskContext) -> Tuple[bool, str]:
    """
    Check if a bot should be halted (stop trading, manage-only).
    """
    if not _enabled():
        return False, ""

    cfg = context.config or _get_config()

    # Circuit breaker
    if cfg.ERROR_CIRCUIT_BREAKER_N > 0 and context.last_error_count >= cfg.ERROR_CIRCUIT_BREAKER_N:
        reason = f"Circuit breaker: {context.last_error_count} consecutive errors"
        logger.warning("RISK_HALT should_halt_bot: %s", reason)
        return True, reason

    # Daily loss
    daily_limit = context.daily_loss_limit_pct if context.daily_loss_limit_pct is not None else cfg.MAX_DAILY_LOSS_PCT
    if context.daily_loss_pct is not None and context.daily_loss_pct <= -daily_limit:
        reason = f"Daily loss {context.daily_loss_pct*100:.2f}% exceeds limit"
        logger.warning("RISK_HALT should_halt_bot: %s", reason)
        return True, reason

    return False, ""


def is_enabled() -> bool:
    """Return whether the risk engine is enabled."""
    return _enabled()
