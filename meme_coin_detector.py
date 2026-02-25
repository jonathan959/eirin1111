"""
Meme coin detector & blocker.

Detect: low market cap, high volatility, no fundamentals
Block from recommendations unless user enables "degen mode"
"""
import logging
import os
from typing import Any, Dict, Optional, Tuple

logger = logging.getLogger(__name__)

BLOCK_MEME_COINS = os.getenv("BLOCK_MEME_COINS", "1").strip().lower() in ("1", "true", "yes")
DEGEN_MODE = os.getenv("DEGEN_MODE", "0").strip().lower() in ("1", "true", "yes")

# Known meme / high-risk symbols
MEME_SYMBOLS = frozenset({
    "DOGE", "SHIB", "PEPE", "FLOKI", "BONK", "WIF", "MEME", "BRETT",
    "POPCAT", "NEIRO", "TURBO", "WOJAK", "CHAD", "TOSHI", "BOME",
})


def is_meme_coin(
    symbol: str,
    market_cap_usd: Optional[float] = None,
    volatility_pct: Optional[float] = None,
    volume_usd: Optional[float] = None,
) -> Tuple[bool, str]:
    """
    Returns (is_meme, reason).
    If BLOCK_MEME_COINS=0 or DEGEN_MODE=1, returns (False, "") to allow.
    """
    if not BLOCK_MEME_COINS or DEGEN_MODE:
        return (False, "")
    base = (symbol.split("/")[0] or symbol).strip().upper()
    if base in MEME_SYMBOLS:
        return (True, f"{base} in meme list")
    if market_cap_usd is not None and market_cap_usd < 50_000_000:
        return (True, "low_mcap")
    if volatility_pct is not None and volatility_pct > 0.20:
        return (True, "extreme_volatility")
    return (False, "")


def should_block_crypto(symbol: str, metrics: Optional[Dict[str, Any]] = None) -> bool:
    """True if symbol should be blocked from recommendations."""
    meme, _ = is_meme_coin(
        symbol,
        market_cap_usd=metrics.get("market_cap_usd") if metrics else None,
        volatility_pct=metrics.get("volatility_pct") or metrics.get("atr_pct") if metrics else None,
    )
    return meme
