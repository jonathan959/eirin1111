"""
Dynamic capital allocation across bots.
Allocates more to high performers, high conviction; less to losers, choppy markets.
"""
import logging
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

AUTO_SCALE_ENABLED = os.getenv("AUTO_SCALE_ENABLED", "0").strip().lower() in ("1", "true", "yes", "y", "on")
WINNING_STREAK_THRESHOLD = int(os.getenv("WINNING_STREAK_THRESHOLD", "3"))
LOSING_STREAK_THRESHOLD = int(os.getenv("LOSING_STREAK_THRESHOLD", "3"))
MAX_ALLOCATION_PCT_PER_BOT = float(os.getenv("MAX_ALLOCATION_PCT_PER_BOT", "10")) / 100.0
WIN_STREAK_BOOST_PCT = float(os.getenv("WIN_STREAK_BOOST_PCT", "25")) / 100.0
LOSS_STREAK_CUT_PCT = float(os.getenv("LOSS_STREAK_CUT_PCT", "50")) / 100.0


@dataclass
class BotAllocationInput:
    bot_id: int
    symbol: str
    base_quote: float
    max_spend_quote: float
    streak: int  # positive = wins, negative = losses
    win_rate: float
    realized_total: float
    strategy_mode: str


@dataclass
class BotAllocationResult:
    bot_id: int
    allocation_mult: float
    effective_base_quote: float
    effective_max_spend: float
    reason: str


def get_allocation_mult(
    streak: int,
    win_rate: float,
    realized_total: float,
) -> float:
    """
    Compute allocation multiplier (0.5 to 1.25).
    - 3+ wins: +25%
    - 3+ losses: -50%
    - Neutral otherwise, slight boost for high win rate.
    """
    mult = 1.0
    if AUTO_SCALE_ENABLED:
        if streak >= WINNING_STREAK_THRESHOLD:
            mult = 1.0 + WIN_STREAK_BOOST_PCT
        elif streak <= -LOSING_STREAK_THRESHOLD:
            mult = 1.0 - LOSS_STREAK_CUT_PCT
        elif win_rate >= 0.65 and realized_total > 0:
            mult = 1.1
        elif win_rate <= 0.35 and realized_total < 0:
            mult = 0.85
    return max(0.5, min(1.25, mult))


def allocate_capital(
    portfolio_total: float,
    bots: List[BotAllocationInput],
) -> Dict[int, BotAllocationResult]:
    """
    Allocate capital across bots. Each bot gets a share of portfolio_total.
    Max per bot: MAX_ALLOCATION_PCT_PER_BOT of portfolio.
    """
    results: Dict[int, BotAllocationResult] = {}
    if portfolio_total <= 0 or not bots:
        return results
    n = len(bots)
    base_share = 1.0 / n if n > 0 else 1.0
    for bot in bots:
        mult = get_allocation_mult(bot.streak, bot.win_rate, bot.realized_total)
        cap_pct = min(MAX_ALLOCATION_PCT_PER_BOT, base_share * mult)
        alloc_quote = portfolio_total * cap_pct
        effective_base = min(bot.base_quote * mult, alloc_quote * 0.2)
        effective_max = min(bot.max_spend_quote or float("inf"), alloc_quote)
        if effective_max <= 0:
            effective_max = alloc_quote
        reason = "baseline"
        if bot.streak >= WINNING_STREAK_THRESHOLD:
            reason = f"{bot.streak} winning streak: +25%"
        elif bot.streak <= -LOSING_STREAK_THRESHOLD:
            reason = f"{abs(bot.streak)} losing streak: -50%"
        results[bot.bot_id] = BotAllocationResult(
            bot_id=bot.bot_id,
            allocation_mult=mult,
            effective_base_quote=effective_base,
            effective_max_spend=effective_max,
            reason=reason,
        )
    return results


def get_allocation_for_bot(
    bot_id: int,
    portfolio_total: float,
    bot_input: BotAllocationInput,
) -> BotAllocationResult:
    """Single-bot allocation (e.g. when BotRunner requests)."""
    results = allocate_capital(portfolio_total, [bot_input])
    if bot_id in results:
        return results[bot_id]
    return BotAllocationResult(
        bot_id=bot_id,
        allocation_mult=1.0,
        effective_base_quote=bot_input.base_quote,
        effective_max_spend=bot_input.max_spend_quote or 0,
        reason="baseline",
    )
