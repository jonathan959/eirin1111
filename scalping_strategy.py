"""
ScalpingStrategy: Quick TP (0.3-0.8%), tight SL (0.2-0.5%), max hold 30min-4h.

Entry: 5m EMA crossovers + volume confirmation + tight spreads.
Uses Level 2 / order book for entry timing when available.
"""
import os
import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from strategies import (
    Decision,
    StrategyBase,
    StrategyContext,
    ema,
    ema_series,
    volume_ratio,
    clamp,
)

logger = logging.getLogger(__name__)

SCALP_TP_PCT = float(os.getenv("SCALP_TP_PCT", "0.5")) / 100.0
SCALP_SL_PCT = float(os.getenv("SCALP_SL_PCT", "0.35")) / 100.0
SCALP_MAX_HOLD_MIN = int(os.getenv("SCALP_MAX_HOLD_MIN", "120"))
SCALP_MIN_HOLD_MIN = int(os.getenv("SCALP_MIN_HOLD_MIN", "5"))


@dataclass
class ScalpConfig:
    tp_pct: float
    sl_pct: float
    max_hold_sec: int
    min_hold_sec: int
    min_volume_ratio: float
    max_spread_pct: float


def _build_scalp_config(ctx: StrategyContext) -> ScalpConfig:
    cfg = ctx.cfg or {}
    tp = float(cfg.get("scalp_tp_pct") or cfg.get("tp") or SCALP_TP_PCT)
    sl = float(cfg.get("scalp_sl_pct") or SCALP_SL_PCT)
    if tp >= 1.0:
        tp = tp / 100.0
    if sl >= 1.0:
        sl = sl / 100.0
    max_hold = int(cfg.get("scalp_max_hold_min") or SCALP_MAX_HOLD_MIN) * 60
    min_hold = int(cfg.get("scalp_min_hold_min") or SCALP_MIN_HOLD_MIN) * 60
    return ScalpConfig(
        tp_pct=clamp(tp, 0.003, 0.008),
        sl_pct=clamp(sl, 0.002, 0.005),
        max_hold_sec=max_hold,
        min_hold_sec=min_hold,
        min_volume_ratio=float(cfg.get("min_volume_ratio") or 1.2),
        max_spread_pct=float(cfg.get("spread_guard_pct") or 0.005),
    )


class ScalpingStrategy(StrategyBase):
    name = "scalping"

    def propose_orders(self, ctx: StrategyContext) -> Decision:
        scalp_cfg = _build_scalp_config(ctx)
        candles = ctx.candles_5m or ctx.candles_15m or []
        if not candles or len(candles) < 30:
            return Decision("HOLD", "Not enough 5m/15m data for scalping.", None, {"regime": ctx.regime}, self.name)

        closes = [float(c[4]) for c in candles if len(c) >= 5]
        if len(closes) < 26:
            return Decision("HOLD", "Not enough closes for EMA.", None, {"regime": ctx.regime}, self.name)

        ema_fast = ema(closes, 9) or 0.0
        ema_slow = ema(closes, 21) or 0.0
        vol_ratio = volume_ratio(candles, 20) or 1.0
        spread_pct = float(ctx.cfg.get("spread_pct") or ctx.cfg.get("spread_guard_pct") or 0.005)
        if spread_pct >= 1.0:
            spread_pct = spread_pct / 100.0

        if ctx.deal.position_size <= 0:
            if ema_fast <= ema_slow:
                return Decision("HOLD", "No EMA crossover (fast below slow).", None, {"regime": ctx.regime}, self.name)
            if vol_ratio < scalp_cfg.min_volume_ratio:
                return Decision("HOLD", f"Volume too low ({vol_ratio:.2f}x).", None, {"regime": ctx.regime}, self.name)
            if spread_pct > scalp_cfg.max_spread_pct:
                return Decision("HOLD", f"Spread too wide ({spread_pct*100:.2f}%).", None, {"regime": ctx.regime}, self.name)

            size = float(ctx.cfg.get("base_quote") or 25.0)
            return Decision(
                "ENTER",
                "Scalp entry: EMA cross + volume.",
                {"side": "buy", "type": "market", "size_quote": size},
                {"regime": ctx.regime, "ema_fast": ema_fast, "ema_slow": ema_slow, "vol_ratio": vol_ratio},
                self.name,
            )

        entry = float(ctx.deal.avg_entry or ctx.last_price)
        opened = ctx.deal_opened_at or ctx.now_ts
        hold_sec = max(0, ctx.now_ts - opened)

        tp_price = entry * (1.0 + scalp_cfg.tp_pct)
        sl_price = entry * (1.0 - scalp_cfg.sl_pct)

        if ctx.last_price >= tp_price:
            return Decision(
                "TAKE_PROFIT",
                "Scalp TP reached.",
                {"side": "sell", "type": "limit", "price": tp_price, "size_base": ctx.deal.position_size},
                {"regime": ctx.regime, "tp_price": tp_price},
                self.name,
            )
        if ctx.last_price <= sl_price:
            return Decision(
                "STOP_LOSS",
                "Scalp SL hit.",
                {"side": "sell", "type": "market", "size_base": ctx.deal.position_size},
                {"regime": ctx.regime, "sl_price": sl_price},
                self.name,
            )
        if hold_sec >= scalp_cfg.max_hold_sec:
            return Decision(
                "EXIT",
                "Scalp max hold time reached.",
                {"side": "sell", "type": "market", "size_base": ctx.deal.position_size},
                {"regime": ctx.regime},
                self.name,
            )

        return Decision("HOLD", "Scalp holding.", None, {"regime": ctx.regime}, self.name)
