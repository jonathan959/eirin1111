"""Conviction Score service.

Computes a 0-100 "conviction" score for a trade setup using a weighted
blend of sub-scores. The goal is to replace the old confusing
"RISKY BUY" / "WAIT" labels on the Explore tab with a single, easy to
understand number.

Formula:
    conviction = 0.30 * trend_score        # higher-TF SMA stack alignment
               + 0.20 * momentum_score     # RSI in 50-70, MACD rising
               + 0.20 * volume_confirmation # vol > 1.5x 20-period avg
               + 0.15 * multi_strategy_agree # how many of 9 strategies agree
               + 0.10 * regime_score       # market regime favorable
               + 0.05 * liquidity_score    # spread, depth

Each sub-score is a float in [0, 100]. The final score is clamped to [0, 100].

This module is deliberately pure-python + no network calls so it can be
unit-tested quickly without mocking exchanges.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Optional, Sequence, Tuple

WEIGHTS: Dict[str, float] = {
    "trend": 0.30,
    "momentum": 0.20,
    "volume": 0.20,
    "strategy_agree": 0.15,
    "regime": 0.10,
    "liquidity": 0.05,
}

# Tier thresholds. Higher number = more conviction required.
TIER_HIGH = 85.0       # "High Conviction" — green, big CTA
TIER_SOLID = 70.0      # "Solid" — blue, normal CTA
TIER_SPECULATIVE = 55.0  # "Speculative" — amber, confirm CTA
# Below TIER_SPECULATIVE is hidden by default (user can toggle show_low=true).


@dataclass
class ConvictionInputs:
    """Inputs to the conviction calculation. All fields are optional; the
    scorer will fall back to neutral (50) if a field is missing."""
    # Trend signal inputs — EMA/SMA stack and slope
    sma_stack_aligned: Optional[bool] = None   # 20>50>200 on 1h
    higher_tf_trend_up: Optional[bool] = None  # 4h and 1d trend agree
    trend_strength: Optional[float] = None     # ADX-like, 0-100

    # Momentum inputs
    rsi14: Optional[float] = None              # last RSI(14) on primary TF
    macd_hist: Optional[float] = None          # MACD histogram
    macd_rising: Optional[bool] = None         # histogram > prior

    # Volume inputs
    vol_ratio: Optional[float] = None          # last_vol / 20-period avg
    vol_rising: Optional[bool] = None          # increasing over last 3 bars

    # Strategy agreement
    strategies_total: int = 9
    strategies_agreeing: int = 0

    # Market regime (uppercase names are stable across project)
    regime: Optional[str] = None               # "TRENDING", "RANGING", etc.
    regime_confidence: Optional[float] = None  # 0-1

    # Liquidity
    spread_bps: Optional[float] = None         # best ask - best bid in bps
    depth_usd: Optional[float] = None          # best-bid USD depth


def _clamp(x: float, lo: float = 0.0, hi: float = 100.0) -> float:
    try:
        return max(lo, min(hi, float(x)))
    except (TypeError, ValueError):
        return 50.0


def score_trend(inp: ConvictionInputs) -> float:
    """Return 0-100 trend sub-score."""
    if inp.sma_stack_aligned is None and inp.higher_tf_trend_up is None and inp.trend_strength is None:
        return 50.0
    score = 50.0
    if inp.sma_stack_aligned is True:
        score += 20.0
    elif inp.sma_stack_aligned is False:
        score -= 20.0
    if inp.higher_tf_trend_up is True:
        score += 15.0
    elif inp.higher_tf_trend_up is False:
        score -= 15.0
    if inp.trend_strength is not None:
        # Map 0-100 ADX-ish to -15..+15
        score += (_clamp(inp.trend_strength) - 50.0) * 0.30
    return _clamp(score)


def score_momentum(inp: ConvictionInputs) -> float:
    """Return 0-100 momentum sub-score.

    Favours RSI 50-70 (healthy uptrend) and a rising MACD histogram.
    RSI > 75 is actually bad (overbought) so the score rolls off above 70.
    """
    if inp.rsi14 is None and inp.macd_hist is None and inp.macd_rising is None:
        return 50.0
    score = 50.0
    if inp.rsi14 is not None:
        rsi = _clamp(inp.rsi14)
        if 50.0 <= rsi <= 70.0:
            # Healthy trending; peak at 60.
            score += 25.0 - abs(rsi - 60.0) * 1.5  # up to +25
        elif 40.0 <= rsi < 50.0:
            score += 5.0  # neutral drift
        elif 70.0 < rsi <= 80.0:
            score -= 10.0  # overbought
        elif rsi > 80.0:
            score -= 20.0
        else:  # rsi < 40
            score -= 15.0  # weak momentum
    if inp.macd_hist is not None:
        score += 5.0 if inp.macd_hist > 0 else -5.0
    if inp.macd_rising is True:
        score += 10.0
    elif inp.macd_rising is False:
        score -= 10.0
    return _clamp(score)


def score_volume(inp: ConvictionInputs) -> float:
    """Return 0-100 volume confirmation sub-score.

    vol_ratio > 1.5 is a strong confirmation; 1.0-1.5 is neutral-good;
    below 0.8 is weak volume (suspicious move).
    """
    if inp.vol_ratio is None and inp.vol_rising is None:
        return 50.0
    score = 50.0
    if inp.vol_ratio is not None:
        r = max(0.0, float(inp.vol_ratio))
        if r >= 2.0:
            score += 30.0
        elif r >= 1.5:
            score += 20.0
        elif r >= 1.2:
            score += 10.0
        elif r >= 0.8:
            score -= 0.0
        else:
            score -= 15.0
    if inp.vol_rising is True:
        score += 10.0
    elif inp.vol_rising is False:
        score -= 5.0
    return _clamp(score)


def score_strategy_agreement(inp: ConvictionInputs) -> float:
    """Return 0-100 based on fraction of strategies agreeing."""
    total = max(1, int(inp.strategies_total or 9))
    agree = max(0, min(total, int(inp.strategies_agreeing or 0)))
    frac = agree / total
    # 0 => 20, 0.5 => 60, 1.0 => 100
    return _clamp(20.0 + frac * 80.0)


_REGIME_SCORES: Dict[str, float] = {
    "TRENDING_UP": 90.0,
    "TRENDING": 80.0,
    "BULL": 80.0,
    "RANGING": 55.0,
    "CHOP": 40.0,
    "CHOPPY": 40.0,
    "HIGH_VOL": 35.0,
    "VOLATILE": 40.0,
    "TRENDING_DOWN": 20.0,
    "BEAR": 20.0,
    "UNKNOWN": 50.0,
}


def score_regime(inp: ConvictionInputs) -> float:
    if not inp.regime:
        return 50.0
    key = str(inp.regime).upper().strip()
    base = _REGIME_SCORES.get(key, 50.0)
    if inp.regime_confidence is not None:
        conf = max(0.0, min(1.0, float(inp.regime_confidence)))
        # Blend toward neutral when we're not confident.
        base = 50.0 + (base - 50.0) * conf
    return _clamp(base)


def score_liquidity(inp: ConvictionInputs) -> float:
    """Liquidity score based on spread and depth.

    Low spread (< 5 bps) and deep top-of-book is ideal. High spread
    kills scores quickly because slippage will eat the edge.
    """
    if inp.spread_bps is None and inp.depth_usd is None:
        return 50.0
    score = 50.0
    if inp.spread_bps is not None:
        s = max(0.0, float(inp.spread_bps))
        if s <= 2.0:
            score += 25.0
        elif s <= 5.0:
            score += 15.0
        elif s <= 15.0:
            score += 0.0
        elif s <= 40.0:
            score -= 15.0
        else:
            score -= 30.0
    if inp.depth_usd is not None:
        d = max(0.0, float(inp.depth_usd))
        if d >= 50_000:
            score += 15.0
        elif d >= 10_000:
            score += 8.0
        elif d >= 2_000:
            score += 0.0
        else:
            score -= 10.0
    return _clamp(score)


@dataclass
class ConvictionResult:
    score: float
    tier: str           # "high" | "solid" | "speculative" | "low"
    tier_label: str
    color: str          # tailwind-ish palette name for pill
    subscores: Dict[str, float]
    reasons: List[str]  # ordered strongest-to-weakest positive factors

    def as_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["score"] = round(float(self.score), 1)
        d["subscores"] = {k: round(float(v), 1) for k, v in self.subscores.items()}
        return d


def _tier_for(score: float) -> Tuple[str, str, str]:
    if score >= TIER_HIGH:
        return "high", "High Conviction", "green"
    if score >= TIER_SOLID:
        return "solid", "Solid", "blue"
    if score >= TIER_SPECULATIVE:
        return "speculative", "Speculative", "amber"
    return "low", "Low Conviction", "gray"


def _reason_from_subscore(name: str, value: float) -> Optional[str]:
    if value >= 80:
        return {
            "trend": "Strong multi-timeframe uptrend",
            "momentum": "Healthy momentum (RSI 50-70, MACD rising)",
            "volume": "Volume confirmation (> 1.5x avg)",
            "strategy_agree": "Most strategies agree",
            "regime": "Market regime favorable",
            "liquidity": "Tight spread / deep book",
        }.get(name)
    if value <= 30:
        return {
            "trend": "Trend weak or reversing",
            "momentum": "Weak or exhausted momentum",
            "volume": "Low volume — not confirmed",
            "strategy_agree": "Few strategies agree",
            "regime": "Regime unfavorable",
            "liquidity": "Wide spread or thin book",
        }.get(name)
    return None


def compute_conviction(inp: ConvictionInputs) -> ConvictionResult:
    sub: Dict[str, float] = {
        "trend": score_trend(inp),
        "momentum": score_momentum(inp),
        "volume": score_volume(inp),
        "strategy_agree": score_strategy_agreement(inp),
        "regime": score_regime(inp),
        "liquidity": score_liquidity(inp),
    }
    total = 0.0
    for key, weight in WEIGHTS.items():
        total += sub[key] * weight
    score = _clamp(total)
    tier, tier_label, color = _tier_for(score)
    # Positive-first, then any negatives for balance.
    pos_reasons: List[Tuple[float, str]] = []
    neg_reasons: List[Tuple[float, str]] = []
    for k, v in sub.items():
        r = _reason_from_subscore(k, v)
        if not r:
            continue
        if v >= 50:
            pos_reasons.append((v, r))
        else:
            neg_reasons.append((v, r))
    pos_reasons.sort(key=lambda p: -p[0])
    neg_reasons.sort(key=lambda p: p[0])
    reasons = [r for _, r in pos_reasons] + [r for _, r in neg_reasons]
    return ConvictionResult(
        score=score,
        tier=tier,
        tier_label=tier_label,
        color=color,
        subscores=sub,
        reasons=reasons,
    )


def checks_from_subscores(sub: Dict[str, float]) -> List[Dict[str, Any]]:
    """Return a list of 5 checks in the order the UI expects them.

    Trend, Volume, Momentum, Regime, Liquidity -- each with {label, passed}.
    """
    order: Sequence[Tuple[str, str]] = (
        ("trend", "Trend"),
        ("volume", "Volume"),
        ("momentum", "Momentum"),
        ("regime", "Regime"),
        ("liquidity", "Liquidity"),
    )
    out: List[Dict[str, Any]] = []
    for key, label in order:
        val = float(sub.get(key, 50.0) or 50.0)
        out.append({
            "label": label,
            "passed": val >= 55.0,
            "score": round(val, 1),
        })
    return out


def plain_english_summary(inp: ConvictionInputs, result: ConvictionResult) -> str:
    """A single-sentence explanation suitable for the Explore "Why?" panel."""
    sub = result.subscores
    strongest = max(sub.items(), key=lambda kv: kv[1])
    weakest = min(sub.items(), key=lambda kv: kv[1])
    head = {
        "high": "High-conviction setup.",
        "solid": "Solid setup.",
        "speculative": "Speculative setup — tread carefully.",
        "low": "Low-conviction — hidden by default.",
    }.get(result.tier, "")
    lead = {
        "trend": "trend is aligned across timeframes",
        "momentum": "momentum is healthy",
        "volume": "volume confirms the move",
        "strategy_agree": "most strategies agree",
        "regime": "market regime is favorable",
        "liquidity": "liquidity is clean",
    }.get(strongest[0], "setup looks good")
    drag = {
        "trend": "trend is unclear",
        "momentum": "momentum is weak",
        "volume": "volume isn't confirming",
        "strategy_agree": "strategies disagree",
        "regime": "regime is unfavorable",
        "liquidity": "spread is wide",
    }.get(weakest[0], "there is a drag")
    if weakest[1] >= 55:
        return f"{head} {lead.capitalize()}."
    return f"{head} {lead.capitalize()} but {drag}."
