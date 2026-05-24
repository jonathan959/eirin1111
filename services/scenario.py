"""Scenario / What-if projection service.

Given starting capital, per-trade risk, win rate, payoff ratio and a
horizon in days, project an expected equity curve by Monte Carlo
simulation.

This is intentionally light-weight: no numpy dependency, small N by
default (256 trials). It is used by the /api/scenario/simulate endpoint
and the /scenario page sliders.
"""
from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

MAX_N_TRIALS = 2000
MAX_DAYS = 1460  # 4 years
MIN_DAYS = 7


@dataclass
class ScenarioInputs:
    capital: float = 1_000.0
    risk_per_trade_pct: float = 0.01  # 1% of equity
    win_rate: float = 0.45
    payoff_ratio: float = 2.0         # avg_win / avg_loss
    trades_per_day: float = 2.0
    days: int = 90
    n_trials: int = 256
    seed: Optional[int] = None


def _clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


def _validate(inp: ScenarioInputs) -> ScenarioInputs:
    return ScenarioInputs(
        capital=max(1.0, float(inp.capital)),
        risk_per_trade_pct=_clamp(float(inp.risk_per_trade_pct), 0.0001, 0.10),
        win_rate=_clamp(float(inp.win_rate), 0.0, 1.0),
        payoff_ratio=_clamp(float(inp.payoff_ratio), 0.1, 10.0),
        trades_per_day=_clamp(float(inp.trades_per_day), 0.1, 20.0),
        days=int(_clamp(float(inp.days), MIN_DAYS, MAX_DAYS)),
        n_trials=int(_clamp(float(inp.n_trials), 10, MAX_N_TRIALS)),
        seed=inp.seed,
    )


def _percentile(sorted_values: List[float], q: float) -> float:
    if not sorted_values:
        return 0.0
    if q <= 0:
        return sorted_values[0]
    if q >= 1:
        return sorted_values[-1]
    idx = q * (len(sorted_values) - 1)
    lo = int(math.floor(idx))
    hi = int(math.ceil(idx))
    if lo == hi:
        return sorted_values[lo]
    frac = idx - lo
    return sorted_values[lo] * (1 - frac) + sorted_values[hi] * frac


def _expectancy(inp: ScenarioInputs) -> float:
    """Per-trade expectancy as a fraction of equity."""
    win_R = inp.payoff_ratio
    loss_R = -1.0
    return inp.risk_per_trade_pct * (inp.win_rate * win_R + (1.0 - inp.win_rate) * loss_R)


def project(inp: ScenarioInputs) -> Dict[str, Any]:
    """Run a Monte Carlo simulation and return summary + quantile curves.

    Output shape:
        {
            "inputs": {...},
            "expectancy_per_trade_pct": 0.003,
            "days": [...],
            "median_curve": [capital, ...],
            "p10_curve":    [...],
            "p90_curve":    [...],
            "final": {
                "median": ..., "p10": ..., "p90": ..., "mean": ...,
                "roi_median_pct": ..., "max_drawdown_median_pct": ...,
            },
        }
    """
    inp = _validate(inp)
    rng = random.Random(inp.seed)

    days = inp.days
    n_trials = inp.n_trials
    daily_trades = max(1, int(round(inp.trades_per_day)))

    # Record equity at end of each day across all trials.
    curves: List[List[float]] = [[] for _ in range(days + 1)]
    finals: List[float] = []
    drawdowns: List[float] = []

    for _ in range(n_trials):
        equity = inp.capital
        peak = equity
        max_dd = 0.0
        curves[0].append(equity)
        for d in range(1, days + 1):
            for _t in range(daily_trades):
                risk_amt = equity * inp.risk_per_trade_pct
                if rng.random() < inp.win_rate:
                    equity += risk_amt * inp.payoff_ratio
                else:
                    equity -= risk_amt
                if equity <= 0:
                    equity = 0.0
                    break
            if equity > peak:
                peak = equity
            if peak > 0:
                dd = (peak - equity) / peak
                if dd > max_dd:
                    max_dd = dd
            curves[d].append(equity)
            if equity <= 0:
                # Fill the remaining days with zero to keep curves aligned.
                for dd_idx in range(d + 1, days + 1):
                    curves[dd_idx].append(0.0)
                break
        finals.append(equity)
        drawdowns.append(max_dd)

    median_curve: List[float] = []
    p10_curve: List[float] = []
    p90_curve: List[float] = []
    for day_vals in curves:
        day_vals.sort()
        median_curve.append(round(_percentile(day_vals, 0.5), 2))
        p10_curve.append(round(_percentile(day_vals, 0.10), 2))
        p90_curve.append(round(_percentile(day_vals, 0.90), 2))

    finals_sorted = sorted(finals)
    dd_sorted = sorted(drawdowns)
    mean_final = sum(finals) / len(finals) if finals else 0.0
    median_final = _percentile(finals_sorted, 0.5)
    p10_final = _percentile(finals_sorted, 0.10)
    p90_final = _percentile(finals_sorted, 0.90)
    roi_median = (median_final - inp.capital) / inp.capital if inp.capital else 0.0
    median_dd = _percentile(dd_sorted, 0.5)

    return {
        "inputs": {
            "capital": inp.capital,
            "risk_per_trade_pct": inp.risk_per_trade_pct,
            "win_rate": inp.win_rate,
            "payoff_ratio": inp.payoff_ratio,
            "trades_per_day": inp.trades_per_day,
            "days": inp.days,
            "n_trials": inp.n_trials,
        },
        "expectancy_per_trade_pct": round(_expectancy(inp), 6),
        "days": list(range(days + 1)),
        "median_curve": median_curve,
        "p10_curve": p10_curve,
        "p90_curve": p90_curve,
        "final": {
            "median": round(median_final, 2),
            "mean": round(mean_final, 2),
            "p10": round(p10_final, 2),
            "p90": round(p90_final, 2),
            "roi_median_pct": round(roi_median * 100.0, 2),
            "max_drawdown_median_pct": round(median_dd * 100.0, 2),
        },
    }
