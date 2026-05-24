"""Quick Trade service.

Backs the POST /api/trade/quick endpoint used by the Explore tab's
Quick Trade Drawer. The drawer sends a simple payload:

    {
        "symbol":       "ODFL",
        "side":         "buy",          # "buy" | "sell"
        "entry_price":  205.50,         # optional; market if None
        "tp_pct":       0.015,          # fraction, 1.5%
        "sl_pct":       0.010,          # fraction, 1.0%
        "size_quote":   250.0,          # $ to risk
        "mode":         "paper",        # "paper" | "live"
        "strategy":     "trend_follow_auto",
        "market_type":  "stock"         # optional; auto-detected
    }

In paper mode the service records a simulated deal row in the
existing `deals` table (dry_run=1) so it shows up in the UI.

In live mode it defers to the existing alpaca/kraken adapters. If
the live runtime is not available (e.g. development machine without
keys) we transparently downgrade to paper and flag the response.

The service exposes a small pure-python validator that is test-friendly.
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

logger = logging.getLogger(__name__)


VALID_SIDES = ("buy", "sell")
VALID_MODES = ("paper", "live")

# Max a single quick-trade can size in $ regardless of user config.
# Hard upper guard so a typo of 10000 doesn't wipe an account.
HARD_MAX_SIZE_QUOTE = 5_000.0
MIN_SIZE_QUOTE = 1.0

# Defaults if the caller omits these.
DEFAULT_TP_PCT = 0.015
DEFAULT_SL_PCT = 0.010
# Max slippage guard for live market fallback (30 bps = 0.3%).
DEFAULT_MAX_SLIPPAGE_PCT = 0.003


@dataclass
class QuickTradeRequest:
    symbol: str
    side: str = "buy"
    entry_price: Optional[float] = None
    tp_pct: float = DEFAULT_TP_PCT
    sl_pct: float = DEFAULT_SL_PCT
    size_quote: float = 100.0
    mode: str = "paper"
    strategy: str = "trend_follow_auto"
    market_type: Optional[str] = None  # "crypto" | "stock" | None=auto
    note: str = ""

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "QuickTradeRequest":
        return cls(
            symbol=str(data.get("symbol") or "").strip().upper(),
            side=str(data.get("side") or "buy").strip().lower(),
            entry_price=_safe_float(data.get("entry_price")),
            tp_pct=float(data.get("tp_pct") or DEFAULT_TP_PCT),
            sl_pct=float(data.get("sl_pct") or DEFAULT_SL_PCT),
            size_quote=float(data.get("size_quote") or 100.0),
            mode=str(data.get("mode") or "paper").strip().lower(),
            strategy=str(data.get("strategy") or "trend_follow_auto").strip(),
            market_type=(str(data.get("market_type")).strip().lower()
                         if data.get("market_type") else None),
            note=str(data.get("note") or ""),
        )


def _safe_float(v: Any) -> Optional[float]:
    try:
        if v in (None, ""):
            return None
        return float(v)
    except (TypeError, ValueError):
        return None


def validate(req: QuickTradeRequest) -> Tuple[bool, str]:
    """Validate a QuickTradeRequest. Return (ok, error_message)."""
    if not req.symbol or len(req.symbol) > 32:
        return False, "symbol is required"
    if req.side not in VALID_SIDES:
        return False, f"side must be one of {VALID_SIDES}"
    if req.mode not in VALID_MODES:
        return False, f"mode must be one of {VALID_MODES}"
    if req.size_quote <= 0 or req.size_quote < MIN_SIZE_QUOTE:
        return False, f"size_quote must be >= {MIN_SIZE_QUOTE}"
    if req.size_quote > HARD_MAX_SIZE_QUOTE:
        return False, f"size_quote exceeds hard cap {HARD_MAX_SIZE_QUOTE}"
    if not (0.001 <= req.tp_pct <= 0.50):
        return False, "tp_pct must be between 0.1% and 50%"
    if not (0.001 <= req.sl_pct <= 0.30):
        return False, "sl_pct must be between 0.1% and 30%"
    if req.entry_price is not None and req.entry_price <= 0:
        return False, "entry_price must be positive"
    return True, ""


def infer_market_type(symbol: str, explicit: Optional[str] = None) -> str:
    """Guess crypto vs stock from symbol shape.

    Symbols with "/" or ending in common fiats (USDT/USDC/USD/XBT) are crypto.
    Everything else is treated as a stock.
    """
    if explicit in ("crypto", "stock"):
        return explicit
    s = str(symbol or "").upper().strip()
    if "/" in s:
        return "crypto"
    crypto_quotes = ("USDT", "USDC", "USD", "XBT", "EUR", "GBP")
    for q in crypto_quotes:
        if s.endswith(q) and len(s) > len(q):
            base = s[:-len(q)]
            if base.isalpha() and 2 <= len(base) <= 6:
                # Heuristic: BTCUSD, ETHUSDT, etc.
                return "crypto"
    # Single-ticker symbols (AAPL, MSFT, ODFL) are stocks.
    return "stock"


def compute_targets(req: QuickTradeRequest, last_price: float) -> Dict[str, float]:
    """Compute TP / SL absolute prices from the request + last price."""
    if req.side == "buy":
        tp = last_price * (1.0 + req.tp_pct)
        sl = last_price * (1.0 - req.sl_pct)
    else:
        tp = last_price * (1.0 - req.tp_pct)
        sl = last_price * (1.0 + req.sl_pct)
    return {
        "entry_price": last_price,
        "tp_price": round(tp, 6),
        "sl_price": round(sl, 6),
        "risk_reward": round(req.tp_pct / max(req.sl_pct, 1e-6), 2),
    }


def simulate_paper_trade(req: QuickTradeRequest, last_price: float, now_ts: Optional[int] = None) -> Dict[str, Any]:
    """Return a deal-shaped dict for a simulated paper trade.

    The caller is responsible for persisting it. We keep this pure so it
    is trivially testable.
    """
    now_ts = int(now_ts or time.time())
    targets = compute_targets(req, last_price)
    qty = req.size_quote / max(last_price, 1e-9)
    return {
        "symbol": req.symbol,
        "side": req.side,
        "mode": "paper",
        "entry_avg": targets["entry_price"],
        "tp_price": targets["tp_price"],
        "sl_price": targets["sl_price"],
        "size_quote": req.size_quote,
        "amount": qty,
        "strategy": req.strategy,
        "opened_at": now_ts,
        "state": "OPEN",
        "dry_run": 1,
        "risk_reward": targets["risk_reward"],
    }
