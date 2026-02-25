"""
Automated strategy discovery via parameter evolution.

- Evolves parameter combinations using genetic algorithm
- Validates with out-of-sample testing (30% holdback)
- Only promotes strategies with robust in-sample vs out-of-sample performance
- Detects overfitting when OOS degrades significantly
"""
import logging
import os
import random
from typing import Any, Dict, List, Optional, Tuple

from backtest import run_backtest_multi
from optimizer import run_genetic_optimization, _default_params

logger = logging.getLogger(__name__)

OOS_HOLDBACK_PCT = float(os.getenv("OOS_HOLDBACK_PCT", "30"))  # 30% for validation
MIN_OOS_DEGRADATION = float(os.getenv("MIN_OOS_DEGRADATION", "0.5"))  # OOS/IS >= 0.5 = robust


def _split_holdout(candles: list, holdback_pct: float = 0.3) -> Tuple[list, list]:
    """Split into in-sample (train) and out-of-sample (test)."""
    n = len(candles)
    split = max(100, int(n * (1 - holdback_pct / 100.0)))
    return candles[:split], candles[split:]


def discover_strategies(
    candles: List[List[float]],
    strategies: Optional[List[str]] = None,
    generations: int = 15,
    population_size: int = 20,
    holdback_pct: float = 0.0,
    min_oos_ratio: float = 0.0,
) -> Dict[str, Any]:
    """
    For each strategy: evolve params on in-sample, validate on out-of-sample.
    Promote only if OOS performance >= min_oos_ratio * IS performance (avoid overfitting).
    """
    holdback = holdback_pct or OOS_HOLDBACK_PCT
    min_ratio = min_oos_ratio or MIN_OOS_DEGRADATION

    is_candles, oos_candles = _split_holdout(candles, holdback)
    if len(oos_candles) < 50:
        return {"ok": False, "error": "Insufficient data for OOS split"}

    strategies = strategies or ["smart", "classic_dca", "grid", "trend"]
    discoveries = []

    for strat in strategies:
        try:
            ga = run_genetic_optimization(
                is_candles,
                strategy=strat,
                generations=generations,
                population_size=population_size,
                objective="sharpe",
            )
            if not ga.get("ok"):
                discoveries.append({"strategy": strat, "error": ga.get("error")})
                continue

            best_p = ga.get("best_params", _default_params())
            is_sharpe = float(ga.get("best_metrics", {}).get("sharpe") or 0)
            res_oos = run_backtest_multi(oos_candles, strategy=strat, params=best_p)
            oos_sharpe = float(res_oos.get("metrics", {}).get("sharpe") or 0)
            oos_dd = float(res_oos.get("metrics", {}).get("max_drawdown") or 0)

            ratio = (oos_sharpe / is_sharpe) if is_sharpe and is_sharpe != 0 else 0
            robust = ratio >= min_ratio or (is_sharpe <= 0 and oos_sharpe >= 0)

            discoveries.append({
                "strategy": strat,
                "best_params": best_p,
                "is_sharpe": round(is_sharpe, 3),
                "oos_sharpe": round(oos_sharpe, 3),
                "oos_max_dd": round(oos_dd, 4),
                "oos_is_ratio": round(ratio, 3),
                "robust": robust,
                "overfitting": not robust and is_sharpe > 0.5,
            })
        except Exception as e:
            discoveries.append({"strategy": strat, "error": str(e)})

    robust_list = [d for d in discoveries if d.get("robust") and "error" not in d]
    best_robust = max(robust_list, key=lambda x: float(x.get("oos_sharpe", -999))) if robust_list else None

    return {
        "ok": True,
        "discoveries": discoveries,
        "best_robust_strategy": best_robust.get("strategy") if best_robust else None,
        "best_robust_params": best_robust.get("best_params") if best_robust else None,
        "holdback_pct": holdback,
    }
