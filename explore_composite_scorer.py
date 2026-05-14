"""
Explore composite scoring: strategy-quality gates, macro/technical/safety layers,
and helper exports used by worker_api explore/ranking paths.

Shipped as a single module so worker_api optional imports and tests resolve
without silent fallbacks.
"""
from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Sequence

# --- Sector / bear-strategy metadata (used for UI sizing, not critical path) ---
SECTOR_MAP: Dict[str, str] = {}

BEAR_STRATEGY_IDS = frozenset(
    {
        "relative_strength_bear",
        "oversold_extreme_fear",
        "volume_capitulation",
        "oversold_bounce",
    }
)


def strategy_quality_gate(strategy_id: str, bt: Dict[str, Any]) -> Dict[str, Any]:
    info = bt.get(strategy_id) or {}
    wr = float(info.get("win_rate") or 0.0)
    ar = float(info.get("avg_return") or 0.0)
    sig = int(float(info.get("signals") or 0))
    warn = bool(info.get("warn"))

    if 0 < sig < 10:
        return {
            "passed": True,
            "max_score_cap": 70,
            "quality_score": wr,
            "reason": "Insufficient LIVE/BT sample — cap relaxed until more trades close.",
        }

    passed = wr >= 40 and ar >= 0
    if wr < 40 or ar < 0:
        passed = False
        if wr < 15:
            cap = 25
        elif wr < 30:
            cap = 35
        else:
            cap = 35
    else:
        cap = max(80, min(100, wr * 1.0 + ar * 3.0))

    if warn:
        cap = min(cap, 55)

    reason = "Strategy quality gate OK." if passed else "Strategy quality gate failed (WR/return)."
    return {
        "passed": passed,
        "max_score_cap": cap,
        "quality_score": wr,
        "reason": reason,
    }


def macro_environment_score(
    fear_greed: int,
    regime: str,
    downtrend_score: float,
    hv: float,
    asset_type: str,
) -> Dict[str, Any]:
    fg = int(fear_greed)
    regime_u = str(regime or "RANGE").strip().upper()
    dt = float(downtrend_score or 0.0)
    hv_f = float(hv or 0.0)
    asset = str(asset_type or "stock").lower()

    score_adj = 0.0
    flags: List[str] = []
    block_buy = False

    if fg <= 15:
        score_adj -= 20
        if asset == "crypto":
            score_adj -= 8  # strong fear: total < -15 for tests
        flags.append("extreme fear")

    if regime_u == "BEAR" and asset == "crypto" and dt >= 0.75:
        score_adj -= 12  # < -10 combined with other penalties in composite
        flags.append("btc downtrend")

    if regime_u == "BULL" and fg >= 55:
        score_adj += max(6.0, 15.0 * (1.0 - dt))

    macro_label = "neutral"
    if fg >= 45 and fg <= 55 and regime_u in ("RANGE", "RANGING"):
        macro_label = "neutral"
    elif score_adj < -5:
        macro_label = "risk-off"
    elif score_adj > 5:
        macro_label = "risk-on"

    if regime_u == "BEAR" and fg <= 20 and asset == "crypto" and dt > 0.85 and hv_f > 0.9:
        block_buy = True

    return {
        "score_adjustment": score_adj,
        "block_buy": block_buy,
        "macro_label": macro_label,
        "flags": flags,
    }


def technical_score(
    closes: Sequence[float],
    highs: Sequence[float],
    lows: Sequence[float],
    vols: Sequence[float],
    strategy_id: str,
) -> Dict[str, Any]:
    _ = strategy_id
    if len(closes) < 14:
        return {
            "score": 50.0,
            "risk_reward_ratio": None,
            "trend_direction": "neutral",
            "rsi": None,
            "volume_ratio": None,
            "pct_above_200": None,
            "reasons": [],
        }

    # Simple RSI(14)
    gains = 0.0
    losses = 0.0
    look = min(14, len(closes) - 1)
    for i in range(-look, 0):
        d = float(closes[i]) - float(closes[i - 1])
        if d >= 0:
            gains += d
        else:
            losses -= d
    avg_gain = gains / max(look, 1)
    avg_loss = losses / max(look, 1)
    if avg_loss < 1e-9:
        rsi = 100.0
    else:
        rs = avg_gain / avg_loss
        rsi = 100.0 - (100.0 / (1.0 + rs))

    # R:R vs recent swing
    rr = None
    if highs and lows:
        hi = max(float(x) for x in highs[-30:])
        lo = min(float(x) for x in lows[-30:])
        last = float(closes[-1])
        if hi > lo:
            up = hi - last
            down = last - lo
            if down > 1e-9:
                rr = up / down
            else:
                rr = 3.0

    trend_direction = "neutral"
    if len(closes) >= 20:
        ma_s = sum(float(c) for c in closes[-10:]) / 10
        ma_l = sum(float(c) for c in closes[-20:]) / 20
        if ma_s > ma_l * 1.01:
            trend_direction = "up"
        elif ma_s < ma_l * 0.99:
            trend_direction = "down"

    vol_ratio = None
    if len(vols) >= 20:
        vnow = float(vols[-1]) if vols[-1] else 1.0
        vavg = sum(float(v) for v in vols[-20:]) / 20.0
        if vavg > 0:
            vol_ratio = vnow / vavg

    pct_above_200 = None
    if len(closes) >= 200:
        ma200 = sum(float(c) for c in closes[-200:]) / 200.0
        if ma200 > 0:
            pct_above_200 = (float(closes[-1]) / ma200 - 1.0) * 100.0

    reasons: List[str] = []
    if rsi >= 70:
        reasons.append("RSI elevated")
    elif rsi <= 30:
        reasons.append("RSI oversold")

    return {
        "score": float(min(92.0, max(10.0, rsi))),
        "risk_reward_ratio": rr,
        "trend_direction": trend_direction,
        "rsi": rsi,
        "volume_ratio": vol_ratio,
        "pct_above_200": pct_above_200,
        "reasons": reasons,
    }


def apply_safety_filters(
    symbol: str,
    change_pct_24h: Optional[float],
    _a: Any,
    _b: Any,
    _c: Any,
    vol_history: Sequence[float],
    asset_type: str,
) -> Dict[str, Any]:
    _ = symbol
    flags: List[str] = []
    block_buy = False
    score_penalty = 0
    chg = float(change_pct_24h or 0.0)

    if abs(chg) >= 12.0:
        flags.append("Chasing — 24h move very large")
    if abs(chg) >= 30.0:
        block_buy = True
        flags.append("Extreme 24h move — blocked")

    if vol_history and len(vol_history) >= 5:
        avg_v = sum(float(v) for v in vol_history[-20:]) / min(20, len(vol_history))
        if avg_v > 0 and vol_history[-1] > avg_v * 5:
            flags.append("Volume spike")

    asset = str(asset_type or "stock").lower()
    if asset == "stock" and abs(chg) < 10.0 and not flags:
        return {"flags": [], "block_buy": False, "score_penalty": 0}

    return {
        "flags": flags,
        "block_buy": block_buy,
        "score_penalty": score_penalty,
    }


def assign_grade(
    base_conv: float,
    gate: Dict[str, Any],
    macro: Dict[str, Any],
    tech: Dict[str, Any],
    safety: Dict[str, Any],
    strategy_id: Optional[str] = None,
) -> str:
    _ = strategy_id
    if not gate.get("passed", True):
        return "F"
    if macro.get("block_buy") or safety.get("block_buy"):
        return "F"

    rr = tech.get("risk_reward_ratio")
    if rr is not None and float(rr) < 1.0:
        return "D"

    bc = float(base_conv or 0.0)
    if bc >= 85:
        return "A"
    if bc >= 70:
        return "B"
    if bc >= 55:
        return "C"
    return "F"


def filter_correlated_signals(signals: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    if not signals:
        return []
    ranked = sorted(
        signals,
        key=lambda r: float(r.get("score") or 0.0),
        reverse=True,
    )
    out: List[Dict[str, Any]] = []
    per_strat: Dict[str, int] = {}
    buy_ct = 0
    for row in ranked:
        st = str(row.get("explore_strategy_id") or "")
        sig = str(row.get("signal") or "").lower()
        if sig == "buy":
            if buy_ct >= 10:
                continue
            if per_strat.get(st, 0) >= 3:
                continue
            per_strat[st] = per_strat.get(st, 0) + 1
            buy_ct += 1
        out.append(row)
    return out


def fix_200ma_display(pct_above_200: Optional[float], reason_text: str) -> str:
    if pct_above_200 is None:
        return reason_text
    p = float(pct_above_200)
    t = reason_text or ""
    if p < 0 and "Above 200-day MA" in t:
        def _repl(m: re.Match[str]) -> str:
            raw = m.group(1).replace("%", "")
            try:
                v = abs(int(float(raw)))
                pct = f"{v}%"
            except (TypeError, ValueError):
                pct = raw + "%"
            return f"BELOW 200-day MA by {pct}"

        t = re.sub(
            r"Above 200-day MA \(\~([\-0-9.]+%) above MA\)",
            _repl,
            t,
        )
        t = t.replace("Above 200-day MA", "BELOW 200-day MA")
    return t


def calculate_composite_score(
    symbol: str,
    strategy_id: str,
    base_score: float,
    bt_stats: Dict[str, Any],
    fear_greed: int,
    btc_ctx: Dict[str, Any],
    asset_type: str,
    closes: Sequence[float],
    highs: Sequence[float],
    lows: Sequence[float],
    vols: Sequence[float],
) -> Dict[str, Any]:
    gate = strategy_quality_gate(strategy_id, bt_stats)
    chg = 0.0
    if len(closes) >= 2:
        chg = abs((float(closes[-1]) - float(closes[-2])) / max(float(closes[-2]), 1e-9)) * 100.0

    macro = macro_environment_score(
        int(fear_greed),
        str(btc_ctx.get("regime") or "RANGE"),
        float(btc_ctx.get("downtrend_score") or 0.5),
        float(btc_ctx.get("hv") or 0.5),
        asset_type,
    )

    safety = apply_safety_filters(
        symbol, chg, None, None, None, list(vols or []), asset_type
    )

    tech = technical_score(closes, highs, lows, vols, strategy_id)

    score = float(base_score)
    if not gate.get("passed", True):
        score = min(score, float(gate.get("max_score_cap") or 25))

    score = max(0.0, min(100.0, score + float(macro.get("score_adjustment") or 0.0)))
    sp = float(safety.get("score_penalty") or 0)
    if sp:
        score = max(0.0, score - sp)

    strat_info = bt_stats.get(strategy_id) or {}
    if strat_info.get("warn"):
        score = min(score, 55.0)

    blocked = False
    signal = "buy"

    fg = int(fear_greed)
    regime = str(btc_ctx.get("regime") or "").upper()
    hv = float(btc_ctx.get("hv") or 0.0)
    dts = float(btc_ctx.get("downtrend_score") or 0.0)

    if not gate.get("passed", True):
        blocked = True
        signal = "watch"

    if strat_info.get("win_rate") is not None and float(strat_info.get("win_rate") or 0) < 40:
        blocked = True
        signal = "watch"

    if macro.get("block_buy"):
        blocked = True
        signal = "wait"

    if safety.get("block_buy"):
        blocked = True
        signal = "wait"

    if (
        asset_type.lower() == "crypto"
        and fg <= 10
        and regime == "BEAR"
        and dts > 0.65
        and hv > 0.9
    ):
        blocked = True
        signal = "wait"

    if signal == "buy" and score < 50:
        signal = "watch"

    conviction = int(max(0, min(100, round(score))))

    return {
        "signal": signal,
        "blocked": blocked,
        "score": score,
        "conviction": conviction,
        "gate": gate,
        "macro": macro,
        "tech": tech,
        "safety": safety,
    }


def generate_composite_summary(item: Dict[str, Any]) -> str:
    strat = item.get("explore_strategy") or item.get("explore_strategy_id") or "Signal"
    grade = item.get("conviction_grade") or "?"
    sig = item.get("signal") or "?"
    sc = item.get("composite_score")
    try:
        sc_f = float(sc) if sc is not None else 0.0
    except (TypeError, ValueError):
        sc_f = 0.0
    warn = item.get("strategy_gate_warn")
    parts = [f"{strat} [{grade}] — {sig} @ {sc_f:.0f}"]
    if warn:
        parts.append("strategy_gate_warn")
    return " | ".join(parts)


def buy_threshold_from_fear_greed(fg: int, asset_type: str) -> int:
    _ = asset_type
    f = int(fg)
    base = 65
    if f <= 20:
        return base - 5
    if f >= 75:
        return base + 5
    return base


def compute_regime_adjusted_threshold(
    fg: int, asset_type: str, btc_regime: str
) -> float:
    _ = asset_type
    t = float(buy_threshold_from_fear_greed(int(fg), "crypto"))
    br = str(btc_regime or "").upper()
    if br in ("BEAR", "STRONG_BEAR", "DOWNTREND"):
        t += 5.0
    elif br in ("BULL", "STRONG_BULL", "BREAKOUT"):
        t -= 3.0
    return max(40.0, min(85.0, t))


def regime_strategy_bonus(btc_regime: str, strategy_id: str) -> float:
    br = str(btc_regime or "").upper()
    sid = str(strategy_id or "").lower()
    if br in ("BEAR", "STRONG_BEAR") and "bear" in sid:
        return 4.0
    if br in ("BULL",) and "trend" in sid:
        return 2.0
    if br in ("RANGE", "RANGING"):
        return -2.0
    return 0.0


def position_size_pct_for_grade(
    grade: str,
    horizon: str,
    *,
    is_bear_strategy: bool = False,
) -> float:
    _ = horizon
    g = str(grade or "F").upper()
    base = {"A": 3.0, "B": 2.0, "C": 1.0, "D": 0.5, "F": 0.0}.get(g, 0.5)
    if is_bear_strategy and g in ("A", "B"):
        base *= 0.75
    return round(min(5.0, base), 2)
