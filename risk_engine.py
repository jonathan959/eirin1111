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
    MAX_SINGLE_POSITION_PCT: float = 0.05   # Max 5% per position
    MAX_CRYPTO_EXPOSURE_PCT: float = 0.40  # Max 40% in crypto
    MAX_STOCK_EXPOSURE_PCT: float = 0.60   # Max 60% in stocks
    MAX_ASSET_EXPOSURE_PCT: float = 0.03
    MAX_TOTAL_EXPOSURE_PCT: float = 0.20
    MAX_DAILY_LOSS_PCT: float = 0.05
    MAX_DRAWDOWN_PCT: float = 0.15         # Disable autopilot if portfolio drops 15% from peak
    MAX_TRADES_PER_DAY: int = 20
    MIN_24H_QUOTE_VOLUME: float = 5000.0
    MAX_SPREAD_BPS: float = 50.0
    VOLATILITY_SPIKE_MULTIPLIER: float = 2.0
    ERROR_CIRCUIT_BREAKER_N: int = 0
    MIN_ORDER_USD_FLOOR: float = 1.0       # Don't block orders under this amount regardless of % limits
    SMALL_ACCOUNT_THRESHOLD: float = 100.0 # Accounts under this get relaxed limits
    SMALL_ACCOUNT_MAX_ASSET_PCT: float = 0.50  # 50% per symbol for small accounts (allows DCA to work)

    def __post_init__(self):
        self.MAX_SINGLE_POSITION_PCT = float(os.getenv("RISK_MAX_SINGLE_POSITION_PCT", str(self.MAX_SINGLE_POSITION_PCT)))
        self.MAX_CRYPTO_EXPOSURE_PCT = float(os.getenv("RISK_MAX_CRYPTO_EXPOSURE_PCT", str(self.MAX_CRYPTO_EXPOSURE_PCT)))
        self.MAX_STOCK_EXPOSURE_PCT = float(os.getenv("RISK_MAX_STOCK_EXPOSURE_PCT", str(self.MAX_STOCK_EXPOSURE_PCT)))
        self.MAX_DRAWDOWN_PCT = float(os.getenv("RISK_MAX_DRAWDOWN_PCT", str(self.MAX_DRAWDOWN_PCT)))
        self.MAX_ASSET_EXPOSURE_PCT = float(os.getenv("RISK_MAX_ASSET_EXPOSURE_PCT", str(self.MAX_ASSET_EXPOSURE_PCT)))
        self.MAX_ASSET_EXPOSURE_PCT = min(self.MAX_ASSET_EXPOSURE_PCT, self.MAX_SINGLE_POSITION_PCT)
        self.MAX_TOTAL_EXPOSURE_PCT = float(os.getenv("RISK_MAX_TOTAL_EXPOSURE_PCT", str(self.MAX_TOTAL_EXPOSURE_PCT)))
        self.MAX_DAILY_LOSS_PCT = float(os.getenv("RISK_MAX_DAILY_LOSS_PCT", str(self.MAX_DAILY_LOSS_PCT)))
        self.MAX_TRADES_PER_DAY = int(os.getenv("RISK_MAX_TRADES_PER_DAY", str(self.MAX_TRADES_PER_DAY)))
        self.MIN_24H_QUOTE_VOLUME = float(os.getenv("RISK_MIN_24H_QUOTE_VOLUME", str(self.MIN_24H_QUOTE_VOLUME)))
        self.MAX_SPREAD_BPS = float(os.getenv("RISK_MAX_SPREAD_BPS", str(self.MAX_SPREAD_BPS)))
        self.VOLATILITY_SPIKE_MULTIPLIER = float(os.getenv("RISK_VOLATILITY_SPIKE_MULTIPLIER", str(self.VOLATILITY_SPIKE_MULTIPLIER)))
        self.ERROR_CIRCUIT_BREAKER_N = int(os.getenv("ERROR_CIRCUIT_BREAKER_N", str(self.ERROR_CIRCUIT_BREAKER_N)))
        self.MIN_ORDER_USD_FLOOR = float(os.getenv("RISK_MIN_ORDER_USD_FLOOR", str(self.MIN_ORDER_USD_FLOOR)))
        self.SMALL_ACCOUNT_THRESHOLD = float(os.getenv("RISK_SMALL_ACCOUNT_THRESHOLD", str(self.SMALL_ACCOUNT_THRESHOLD)))
        self.SMALL_ACCOUNT_MAX_ASSET_PCT = float(os.getenv("RISK_SMALL_ACCOUNT_MAX_ASSET_PCT", str(self.SMALL_ACCOUNT_MAX_ASSET_PCT)))


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
    # Asset type for crypto vs stock exposure limits
    is_crypto: bool = False
    # 24h return - block if dropped >8%
    ret_24h_pct: Optional[float] = None
    # Macro regime - block unless defensive
    macro_risk_off: bool = False
    is_defensive_asset: bool = False
    # Proposed order size in USD (for minimum floor check)
    proposed_order_usd: Optional[float] = None
    # Safety order flag - exempt from per-symbol cap (DCA needs to add to position)
    is_safety_order: bool = False


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


def _crypto_stock_exposure(positions_usd: Dict[str, float], balance_total: float) -> Tuple[float, float]:
    """Returns (crypto_pct, stock_pct) of total portfolio."""
    if balance_total <= 0:
        return 0.0, 0.0
    crypto_val = sum(v for k, v in positions_usd.items() if "/" in k or k in ("BTC", "ETH", "SOL", "XRP", "ADA", "DOGE"))
    stock_val = sum(v for k, v in positions_usd.items() if "/" not in k and len(k) <= 5)
    return crypto_val / balance_total, stock_val / balance_total


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
    max_asset = min(max_asset, cfg.MAX_SINGLE_POSITION_PCT)
    
    # Small account override: relax limits for accounts under threshold
    is_small_account = context.balance_total_usd < cfg.SMALL_ACCOUNT_THRESHOLD
    if is_small_account:
        max_asset = max(max_asset, cfg.SMALL_ACCOUNT_MAX_ASSET_PCT)
        max_total = max(max_total, 0.80)  # Allow up to 80% total exposure for small accounts

    # Block if asset dropped >8% in 24h
    if context.ret_24h_pct is not None and context.ret_24h_pct <= -8.0:
        reason = f"Asset down {context.ret_24h_pct:.1f}% in 24h (block threshold -8%)"
        logger.warning("RISK_BLOCKED can_open_trade: %s", reason)
        return False, reason

    # Risk-Off: warn but allow trading (position sizing already reduced by intelligence layer)
    if context.macro_risk_off and not context.is_defensive_asset:
        logger.info("RISK_NOTE: Macro Risk-Off active — position sizing reduced, but trading allowed")

    # Exposure
    positions = context.positions_usd or {}
    asset_exp, total_exp = compute_exposure(
        context.balance_total_usd,
        positions,
        context.symbol,
    )
    
    # Calculate actual USD values for logging
    total_positions_usd = sum(positions.values())
    asset_position_usd = positions.get(context.symbol, 0.0)
    max_asset_usd = context.balance_total_usd * max_asset
    max_total_usd = context.balance_total_usd * max_total
    proposed_order = context.proposed_order_usd or 0.0
    
    # DEBUG: Log incoming values for floor check troubleshooting
    logger.info(
        "RISK_DEBUG: symbol=%s | proposed=$%.2f | floor=$%.2f | balance=$%.2f | position=$%.2f | max_asset=%.1f%% ($%.2f) | safety_order=%s | small_acct=%s",
        context.symbol, proposed_order, cfg.MIN_ORDER_USD_FLOOR,
        context.balance_total_usd, asset_position_usd, max_asset * 100, max_asset_usd,
        context.is_safety_order, is_small_account
    )
    
    # Minimum floor check: allow orders under MIN_ORDER_USD_FLOOR regardless of percentage
    # This prevents blocking very small orders on small accounts
    order_under_floor = proposed_order > 0 and proposed_order < cfg.MIN_ORDER_USD_FLOOR
    position_under_floor = asset_position_usd < cfg.MIN_ORDER_USD_FLOOR
    
    if total_exp >= max_total:
        # Allow if order is under floor and total position would still be small
        if order_under_floor and total_positions_usd < cfg.MIN_ORDER_USD_FLOOR * 2:
            logger.info(
                "RISK_ALLOW (floor): Total exposure %.2f%% >= max %.1f%%, but order $%.2f under $%.2f floor (total positions: $%.2f)",
                total_exp * 100, max_total * 100, proposed_order, cfg.MIN_ORDER_USD_FLOOR, total_positions_usd
            )
        else:
            reason = f"Total exposure {total_exp*100:.2f}% >= max {max_total*100:.1f}%"
            logger.warning(
                "RISK_BLOCKED can_open_trade: %s | Current: $%.2f of $%.2f (cap: $%.2f) | Order: $%.2f",
                reason, total_positions_usd, context.balance_total_usd, max_total_usd, proposed_order
            )
            return False, reason
    
    if asset_exp >= max_asset:
        # SAFETY ORDER EXEMPTION: DCA safety orders must be allowed to add to existing positions
        # Blocking safety orders defeats the entire purpose of DCA strategy
        if context.is_safety_order:
            logger.info(
                "RISK_ALLOW (safety order): Asset exposure %.2f%% >= max %.1f%%, but safety orders are exempt from per-symbol cap",
                asset_exp * 100, max_asset * 100
            )
        # Allow if current position is under floor (small account protection)
        elif position_under_floor or order_under_floor:
            logger.info(
                "RISK_ALLOW (floor): Asset exposure %.2f%% >= max %.1f%%, but position $%.2f or order $%.2f under $%.2f floor",
                asset_exp * 100, max_asset * 100, asset_position_usd, proposed_order, cfg.MIN_ORDER_USD_FLOOR
            )
        else:
            reason = f"Asset exposure {asset_exp*100:.2f}% >= max {max_asset*100:.1f}%"
            logger.warning(
                "RISK_BLOCKED can_open_trade: %s | Symbol: %s | Current: $%.2f of $%.2f (cap: $%.2f @ %.1f%%) | Order: $%.2f",
                reason, context.symbol, asset_position_usd, context.balance_total_usd, max_asset_usd, max_asset * 100, proposed_order
            )
            return False, reason

    # Crypto vs stock exposure limits (skip for small accounts — they're inherently concentrated)
    if not is_small_account and not context.is_safety_order:
        crypto_pct, stock_pct = _crypto_stock_exposure(positions, context.balance_total_usd)
        new_pos_val = context.symbol_position_usd
        if context.is_crypto:
            projected_crypto = crypto_pct + (max_asset if context.symbol_position_usd <= 0 else 0)
            if projected_crypto > cfg.MAX_CRYPTO_EXPOSURE_PCT:
                reason = f"Crypto exposure would exceed {cfg.MAX_CRYPTO_EXPOSURE_PCT*100:.0f}%"
                logger.warning("RISK_BLOCKED can_open_trade: %s", reason)
                return False, reason
        else:
            projected_stock = stock_pct + (max_asset if context.symbol_position_usd <= 0 else 0)
            if projected_stock > cfg.MAX_STOCK_EXPOSURE_PCT:
                reason = f"Stock exposure would exceed {cfg.MAX_STOCK_EXPOSURE_PCT*100:.0f}%"
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
        reason = f"Max trades (24h rolling) ({context.trades_today}/{cfg.MAX_TRADES_PER_DAY}) reached"
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
    DCA safety orders are exempt from per-symbol cap by default since blocking them
    defeats the entire purpose of dollar-cost averaging.
    """
    # Force is_safety_order=True for DCA adds since that's the whole point
    if not context.is_safety_order:
        # Create a modified context with safety order flag set
        from dataclasses import replace
        context = replace(context, is_safety_order=True)
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
