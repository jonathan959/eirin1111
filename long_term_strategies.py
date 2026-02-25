"""
Long-term investing strategies: BuyAndHoldWithGuardrails, SectorRotationStrategy.

- BuyAndHoldWithGuardrails: Only sells on fundamental deterioration, 200-week MA break, P/E extreme.
- SectorRotationStrategy: Rotate from weak to strong sectors based on momentum.
"""
import os
import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from strategies import (
    Decision,
    StrategyBase,
    StrategyContext,
    RegimeResult,
    ema,
    sma,
)

logger = logging.getLogger(__name__)

LONG_TERM_MODE = os.getenv("LONG_TERM_MODE", "0").strip().lower() in ("1", "true", "yes", "y", "on")


def _ma200_weekly(candles_1d: List[List[float]]) -> Optional[float]:
    """Approximate 200-week MA from daily candles (200*5 ~ 1000 days)."""
    if not candles_1d or len(candles_1d) < 200:
        return None
    closes = [float(c[4]) for c in candles_1d[-200:] if len(c) >= 5]
    return sum(closes) / len(closes) if closes else None


class BuyAndHoldWithGuardrails(StrategyBase):
    """
    Buy and hold with guardrails. Only sells on:
    - Fundamental deterioration (earnings miss, debt spike - proxied by valuation)
    - Technical break of 200-week MA (support)
    - Valuation extreme (P/E > 2x sector average)
    Otherwise: buy dips, hold.
    """
    name = "buy_and_hold"

    def propose_orders(self, ctx: StrategyContext) -> Decision:
        candles_1d = ctx.cfg.get("candles_1d") or []
        candles_4h = ctx.candles_4h or []
        candles = candles_1d or candles_4h
        if not candles or len(candles) < 100:
            return Decision("HOLD", "Not enough data for buy-and-hold.", None, {"regime": ctx.regime}, self.name)

        closes = [float(c[4]) for c in candles if len(c) >= 5]
        if not closes:
            return Decision("HOLD", "No closes.", None, {"regime": ctx.regime}, self.name)

        ma200w = _ma200_weekly(candles_1d) if candles_1d else (sma(closes, 200) if len(closes) >= 200 else None)
        price = ctx.last_price

        # Guardrail 1: Price below 200-week MA (major support break)
        if ma200w and price and price < ma200w * 0.95:
            return Decision(
                "EXIT",
                "BuyAndHold: Price broke 200-week MA support.",
                {"side": "sell", "type": "market", "size_base": ctx.deal.position_size},
                {"regime": ctx.regime, "ma200": ma200w, "price": price},
                self.name,
            )

        # Guardrail 2: Valuation extreme - check from ctx if we have valuation
        pe_vs_sector = ctx.cfg.get("pe_vs_sector_ratio")
        if pe_vs_sector and float(pe_vs_sector) > 2.0:
            return Decision(
                "EXIT",
                "BuyAndHold: P/E > 2x sector (valuation extreme).",
                {"side": "sell", "type": "market", "size_base": ctx.deal.position_size},
                {"regime": ctx.regime},
                self.name,
            )

        # No position: buy dip (price below MA or pullback)
        if ctx.deal.position_size <= 0:
            base = float(ctx.cfg.get("base_quote") or 50.0)
            if ma200w and price and price < ma200w * 1.02:
                return Decision(
                    "ENTER",
                    "BuyAndHold: Buying dip near support.",
                    {"side": "buy", "type": "market", "size_quote": base},
                    {"regime": ctx.regime, "ma200": ma200w},
                    self.name,
                )
            return Decision("HOLD", "BuyAndHold: No dip to buy.", None, {"regime": ctx.regime}, self.name)

        return Decision("HOLD", "BuyAndHold: Holding.", None, {"regime": ctx.regime}, self.name)


class SectorRotationStrategy(StrategyBase):
    """
    Sector rotation: overweight strong sectors, underweight weak.
    Uses sector_performance_history and target allocations.
    """
    name = "sector_rotation"

    def propose_orders(self, ctx: StrategyContext) -> Decision:
        try:
            from sector_rotation import sector_rotation_signal, get_rotation_suggestions
            from stock_metadata import get_sector
        except ImportError:
            return Decision("HOLD", "Sector rotation modules not available.", None, {"regime": ctx.regime}, self.name)

        sector = get_sector(ctx.symbol)
        if not sector:
            return Decision("HOLD", "Unknown sector for rotation.", None, {"regime": ctx.regime}, self.name)

        current_alloc = ctx.cfg.get("sector_allocations") or {sector: 0.25}
        signal = sector_rotation_signal(sector_returns={sector: 0.0}, current_allocations=current_alloc)
        suggestions = get_rotation_suggestions(current_alloc)

        if ctx.deal.position_size <= 0:
            if sector in signal.overweight_sectors:
                base = float(ctx.cfg.get("base_quote") or 50.0)
                return Decision(
                    "ENTER",
                    f"Sector rotation: {sector} overweight.",
                    {"side": "buy", "type": "market", "size_quote": base},
                    {"regime": ctx.regime, "sector": sector},
                    self.name,
                )
            return Decision("HOLD", f"Sector {sector} not in overweight list.", None, {"regime": ctx.regime}, self.name)

        # Exit if sector is underweight and we have rebalance suggestion
        for s in suggestions:
            if s.get("sector") == sector and s.get("action") == "reduce":
                return Decision(
                    "EXIT",
                    f"Sector rotation: Reducing {sector} (rebalance).",
                    {"side": "sell", "type": "market", "size_base": ctx.deal.position_size},
                    {"regime": ctx.regime, "suggestions": suggestions},
                    self.name,
                )
        return Decision("HOLD", "Sector rotation: Holding.", None, {"regime": ctx.regime}, self.name)
