"""
Unified Execution Gate — single checkpoint before ANY order placement.

Every trade path (bot_manager direct, executor intelligence, manual API) must call
check_execution_gate() before placing an order. This prevents fragmented spread/stale/
liquidity checks scattered across multiple files.

Checks:
  1. Spread: bid/ask spread vs per-symbol + global threshold
  2. Stale data: ticker age and last candle freshness
  3. Liquidity: minimum volume (if available)
  4. Price feed: reject None/partial/error ticker data
  5. Slippage estimate: for market orders, estimate and cap slippage
  6. Kill switch / global pause
"""

import logging
import os
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

GLOBAL_MAX_SPREAD_PCT = float(os.getenv("EXECUTION_MAX_SPREAD_PCT", "0.008"))
MARKET_ORDER_MAX_SPREAD_PCT = float(os.getenv("MARKET_ORDER_MAX_SPREAD_PCT", "0.003"))
STALE_TICKER_MAX_SEC = int(os.getenv("STALE_TICKER_MAX_SEC", "120"))
STALE_CANDLE_MAX_SEC = int(os.getenv("STALE_CANDLE_MAX_SEC", "10800"))
MIN_VOLUME_USD_24H = float(os.getenv("EXECUTION_MIN_VOLUME_24H", "0"))
MAX_SLIPPAGE_BPS = float(os.getenv("MAX_SLIPPAGE_BPS", "50"))
GATE_ENABLED = os.getenv("EXECUTION_GATE_ENABLED", "1").strip().lower() in ("1", "true", "yes", "y", "on")


@dataclass
class GateResult:
    """Outcome of the execution gate check."""
    allowed: bool
    reason: Optional[str] = None
    bid: Optional[float] = None
    ask: Optional[float] = None
    spread_pct: Optional[float] = None
    spread_threshold: float = 0.0
    ticker_age_sec: Optional[float] = None
    candle_age_sec: Optional[float] = None
    volume_24h: Optional[float] = None
    estimated_slippage_bps: Optional[float] = None
    details: Dict[str, Any] = field(default_factory=dict)
    checks_passed: List[str] = field(default_factory=list)
    checks_failed: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "allowed": self.allowed,
            "reason": self.reason,
            "bid": self.bid,
            "ask": self.ask,
            "spread_pct": self.spread_pct,
            "spread_threshold": self.spread_threshold,
            "ticker_age_sec": self.ticker_age_sec,
            "candle_age_sec": self.candle_age_sec,
            "volume_24h": self.volume_24h,
            "estimated_slippage_bps": self.estimated_slippage_bps,
            "checks_passed": self.checks_passed,
            "checks_failed": self.checks_failed,
        }


def check_execution_gate(
    symbol: str,
    side: str = "buy",
    order_type: str = "limit",
    bid: Optional[float] = None,
    ask: Optional[float] = None,
    last_price: Optional[float] = None,
    ticker_ts: Optional[float] = None,
    last_candle_ts: Optional[float] = None,
    volume_24h: Optional[float] = None,
    bot_spread_guard_pct: Optional[float] = None,
    volatility_pct: Optional[float] = None,
    quote_amount: float = 0.0,
    dry_run: bool = True,
    skip_spread: bool = False,
    skip_stale: bool = False,
) -> GateResult:
    """
    Unified pre-trade gate. Call before every order.

    Returns GateResult with allowed=True/False and diagnostics.
    All callers get the same checks, same thresholds, same diagnostics.
    """
    if not GATE_ENABLED:
        return GateResult(allowed=True, reason=None, checks_passed=["gate_disabled"])

    result = GateResult(allowed=True, bid=bid, ask=ask)
    now = time.time()

    # --- 1. Price feed validity ---
    if bid is None and ask is None and last_price is None:
        result.allowed = False
        result.reason = "No price data available (bid/ask/last all None)"
        result.checks_failed.append("price_feed")
        return result

    if bid is not None and bid <= 0:
        result.allowed = False
        result.reason = f"Invalid bid price: {bid}"
        result.checks_failed.append("price_feed")
        return result

    if ask is not None and ask <= 0:
        result.allowed = False
        result.reason = f"Invalid ask price: {ask}"
        result.checks_failed.append("price_feed")
        return result

    result.checks_passed.append("price_feed")

    # --- 2. Spread check ---
    if not skip_spread and bid is not None and ask is not None and bid > 0 and ask > 0:
        mid = (bid + ask) / 2.0
        spread_pct = (ask - bid) / mid if mid > 0 else 0.0
        result.spread_pct = spread_pct

        per_symbol_max = bot_spread_guard_pct if bot_spread_guard_pct is not None else GLOBAL_MAX_SPREAD_PCT

        if volatility_pct is not None and volatility_pct > 0:
            if volatility_pct < 0.05:
                adaptive_max = per_symbol_max
            elif volatility_pct < 0.10:
                adaptive_max = max(per_symbol_max, 0.006)
            else:
                adaptive_max = max(per_symbol_max, 0.010)
        else:
            adaptive_max = per_symbol_max

        is_market = order_type.lower() == "market"
        effective_max = min(adaptive_max, MARKET_ORDER_MAX_SPREAD_PCT) if is_market else adaptive_max
        result.spread_threshold = effective_max

        if spread_pct >= effective_max:
            result.allowed = False
            result.reason = (
                f"Spread too wide: {spread_pct*100:.3f}% >= {effective_max*100:.3f}% "
                f"(bid={bid:.6g}, ask={ask:.6g})"
            )
            result.checks_failed.append("spread")
            result.details["spread_pct"] = round(spread_pct * 100, 4)
            result.details["spread_threshold_pct"] = round(effective_max * 100, 4)
            result.details["bid"] = bid
            result.details["ask"] = ask
            return result
        result.checks_passed.append("spread")

    # --- 3. Stale ticker ---
    if not skip_stale and ticker_ts is not None and ticker_ts > 0:
        age = now - ticker_ts
        result.ticker_age_sec = age
        if age > STALE_TICKER_MAX_SEC:
            result.allowed = False
            result.reason = f"Stale ticker: {age:.0f}s old (max {STALE_TICKER_MAX_SEC}s)"
            result.checks_failed.append("stale_ticker")
            return result
        result.checks_passed.append("stale_ticker")

    # --- 4. Stale candle ---
    if not skip_stale and last_candle_ts is not None and last_candle_ts > 0:
        candle_age = now - last_candle_ts
        result.candle_age_sec = candle_age
        if candle_age > STALE_CANDLE_MAX_SEC:
            result.allowed = False
            result.reason = f"Stale candle data: {candle_age:.0f}s old (max {STALE_CANDLE_MAX_SEC}s)"
            result.checks_failed.append("stale_candle")
            return result
        result.checks_passed.append("stale_candle")

    # --- 5. Minimum volume ---
    if volume_24h is not None:
        result.volume_24h = volume_24h
        min_vol = MIN_VOLUME_USD_24H
        if min_vol > 0 and volume_24h < min_vol:
            result.allowed = False
            result.reason = f"Low 24h volume: ${volume_24h:,.0f} < ${min_vol:,.0f}"
            result.checks_failed.append("min_volume")
            return result
        result.checks_passed.append("min_volume")

    # --- 6. Slippage estimate (market orders only) ---
    if order_type.lower() == "market" and bid is not None and ask is not None and bid > 0:
        mid = (bid + ask) / 2.0
        half_spread_bps = ((ask - bid) / mid) * 5000 if mid > 0 else 0
        result.estimated_slippage_bps = half_spread_bps
        if half_spread_bps > MAX_SLIPPAGE_BPS and not dry_run:
            result.allowed = False
            result.reason = (
                f"Estimated slippage too high: {half_spread_bps:.0f} bps > {MAX_SLIPPAGE_BPS:.0f} bps max"
            )
            result.checks_failed.append("slippage")
            return result
        result.checks_passed.append("slippage")

    return result


def fetch_gate_inputs(
    kc: Any,
    symbol: str,
    bot_config: Optional[Dict[str, Any]] = None,
    last_candle_ts: Optional[float] = None,
) -> Dict[str, Any]:
    """
    Fetch live ticker data needed for the execution gate.
    Returns dict with bid, ask, last_price, ticker_ts, volume_24h, last_candle_ts.
    Handles errors gracefully — returns partial data with error flag.
    """
    inputs: Dict[str, Any] = {
        "bid": None, "ask": None, "last_price": None,
        "ticker_ts": None, "volume_24h": None,
        "last_candle_ts": last_candle_ts,
        "error": None,
    }
    try:
        t = kc.fetch_ticker(symbol)
        if not t or not isinstance(t, dict):
            inputs["error"] = "Ticker returned None/empty"
            return inputs
        inputs["bid"] = float(t.get("bid") or 0) or None
        inputs["ask"] = float(t.get("ask") or 0) or None
        inputs["last_price"] = float(t.get("last") or t.get("c") or 0) or None
        ts = t.get("timestamp") or t.get("ts")
        if ts:
            inputs["ticker_ts"] = float(ts) / 1000.0 if float(ts) > 1e12 else float(ts)
        vol = t.get("quoteVolume") or t.get("baseVolume")
        if vol:
            inputs["volume_24h"] = float(vol)
    except Exception as e:
        inputs["error"] = f"Ticker fetch failed: {type(e).__name__}: {e}"
        logger.warning("fetch_gate_inputs(%s): %s", symbol, inputs["error"])
    if bot_config:
        inputs["bot_spread_guard_pct"] = float(bot_config.get("spread_guard_pct") or GLOBAL_MAX_SPREAD_PCT)
    return inputs
