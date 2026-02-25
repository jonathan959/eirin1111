"""
Monte Carlo simulation for backtests.

- Run N simulations with random fill/slippage/order-timing variations
- Calculate: probability of success, expected value, max drawdown distribution
- Confidence intervals for returns
- Output: "95% confidence of +5% to +18% return"
"""
import logging
import math
import os
import random
import statistics
from typing import Any, Dict, List, Optional, Tuple

from backtest import run_backtest_multi

logger = logging.getLogger(__name__)

MONTE_CARLO_RUNS = int(os.getenv("MONTE_CARLO_RUNS", "1000"))
INCLUDE_SLIPPAGE = os.getenv("INCLUDE_SLIPPAGE", "1").strip().lower() in ("1", "true", "yes")
SLIPPAGE_STDEV_PCT = float(os.getenv("SLIPPAGE_STDEV_PCT", "0.05"))


def _perturb_candles(candles: List[List[float]], slippage_stdev: float) -> List[List[float]]:
    """Add random noise to close prices to simulate fill variation. [ts, o, h, l, c, v]"""
    out = []
    for c in candles:
        ts, o, h, l, close, v = c[0], c[1], c[2], c[3], c[4], c[5]
        noise = 1.0 + random.gauss(0, slippage_stdev / 100.0) if close > 0 else 1.0
        new_c = close * max(0.5, min(2.0, noise))
        out.append([ts, o, h, l, new_c, v])
    return out


def run_monte_carlo(
    candles: List[List[float]],
    strategy: str = "smart",
    params: Optional[Dict[str, Any]] = None,
    n_runs: int = 0,
    starting_cash: float = 1000.0,
    slippage_stdev_pct: float = 0.0,
) -> Dict[str, Any]:
    """
    Run N simulations with randomized slippage/fill variations.
    Returns: probability of profit, expected return, 95% CI, max drawdown distribution.
    """
    n = n_runs or MONTE_CARLO_RUNS
    slip = slippage_stdev_pct or (SLIPPAGE_STDEV_PCT if INCLUDE_SLIPPAGE else 0.02)
    base = params or {
        "fee_rate": 0.0026,
        "slippage": 0.001,
        "warmup": 60,
        "base_risk_pct": 0.02,
        "base_quote": 25.0,
        "safety_quote": 25.0,
        "max_safety": 3,
        "tp": 0.012,
    }

    returns_pct: List[float] = []
    final_equities: List[float] = []
    max_drawdowns: List[float] = []
    sharpes: List[float] = []

    for _ in range(n):
        try:
            perturbed = _perturb_candles(candles, slip)
            res = run_backtest_multi(perturbed, strategy=strategy, params=base, starting_cash=starting_cash)
            m = res.get("metrics", {})
            fe = float(m.get("final_equity", starting_cash))
            ret = (fe - starting_cash) / starting_cash * 100.0 if starting_cash > 0 else 0.0
            returns_pct.append(ret)
            final_equities.append(fe)
            max_drawdowns.append(float(m.get("max_drawdown", 0) or 0) * 100.0)
            sharpes.append(float(m.get("sharpe", 0) or 0))
        except Exception as e:
            logger.debug("monte carlo run failed: %s", e)

    if not returns_pct:
        return {"ok": False, "error": "No successful runs", "runs": 0}

    n_valid = len(returns_pct)
    mean_ret = statistics.mean(returns_pct)
    std_ret = statistics.stdev(returns_pct) if n_valid > 1 else 0.0
    prob_profit = sum(1 for r in returns_pct if r > 0) / n_valid * 100.0
    sorted_ret = sorted(returns_pct)
    p5 = sorted_ret[int(0.05 * n_valid)] if n_valid >= 20 else sorted_ret[0]
    p50 = sorted_ret[int(0.50 * n_valid)] if n_valid >= 2 else mean_ret
    p95 = sorted_ret[int(0.95 * n_valid)] if n_valid >= 20 else sorted_ret[-1]
    mean_dd = statistics.mean(max_drawdowns)
    median_dd = statistics.median(max_drawdowns)
    worst_dd = max(max_drawdowns) if max_drawdowns else 0

    return {
        "ok": True,
        "runs": n_valid,
        "probability_of_profit_pct": round(prob_profit, 1),
        "expected_return_pct": round(mean_ret, 2),
        "return_std_pct": round(std_ret, 2),
        "confidence_95_lower_pct": round(p5, 2),
        "confidence_95_upper_pct": round(p95, 2),
        "median_return_pct": round(p50, 2),
        "summary": f"95% confidence of {p5:.1f}% to {p95:.1f}% return (median {p50:.1f}%)",
        "max_drawdown": {
            "mean_pct": round(mean_dd, 2),
            "median_pct": round(median_dd, 2),
            "worst_pct": round(worst_dd, 2),
        },
        "sharpe_mean": round(statistics.mean(sharpes), 3),
    }
