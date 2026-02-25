"""
Optimization engine: walk-forward, grid search, genetic algorithm, strategy comparison.

- Walk-forward: train 90d, test 30d, roll forward
- Grid search: TP%, SL%, trailing stop, MA periods, RSI thresholds
- Genetic algorithm: evolve best parameter combinations
- Strategy comparison: run all strategies, compare Sharpe, max DD, win rate
- Regime-specific backtests: bull/bear/range performance
"""
import itertools
import logging
import math
import os
import random
import statistics
import time
from typing import Any, Dict, List, Optional, Tuple

from backtest import run_backtest_multi
from strategies import detect_regime, get_strategy, dominant_regime, STRATEGY_REGISTRY

logger = logging.getLogger(__name__)

ENABLE_WALK_FORWARD = os.getenv("ENABLE_WALK_FORWARD", "1").strip().lower() in ("1", "true", "yes")
OPTIMIZE_PARAMETERS = os.getenv("OPTIMIZE_PARAMETERS", "0").strip().lower() in ("1", "true", "yes")
MONTE_CARLO_RUNS = int(os.getenv("MONTE_CARLO_RUNS", "100"))
INCLUDE_SLIPPAGE = os.getenv("INCLUDE_SLIPPAGE", "1").strip().lower() in ("1", "true", "yes")
COMMISSION_PCT = float(os.getenv("COMMISSION_PCT", "0.26"))  # default crypto-like
SLIPPAGE_PCT = float(os.getenv("SLIPPAGE_PCT", "0.1"))


DEFAULT_PARAM_GRID = {
    "tp": [0.008, 0.01, 0.012, 0.015, 0.02],
    "first_dev": [0.01, 0.015, 0.02, 0.025],
    "step_mult": [1.1, 1.2, 1.3, 1.4],
    "max_safety": [2, 3, 4, 5],
    "base_quote": [20.0, 25.0, 30.0],
    "safety_quote": [20.0, 25.0, 30.0],
    "trend_sma": [50, 100, 200],
    "fee_rate": [0.001, 0.0026],
    "slippage": [0.0005, 0.001, 0.002],
}


def _default_params() -> Dict[str, Any]:
    slip = SLIPPAGE_PCT / 100.0 if INCLUDE_SLIPPAGE else 0.0005
    fee = COMMISSION_PCT / 100.0
    return {
        "fee_rate": fee,
        "slippage": slip,
        "warmup": 60,
        "base_risk_pct": 0.02,
        "base_quote": 25.0,
        "safety_quote": 25.0,
        "max_safety": 3,
        "tp": 0.012,
        "first_dev": 0.015,
        "step_mult": 1.2,
    }


def run_grid_search(
    candles: List[List[float]],
    strategy: str = "smart",
    param_grid: Optional[Dict[str, List[Any]]] = None,
    starting_cash: float = 1000.0,
    objective: str = "sharpe",
    max_combos: int = 200,
) -> Dict[str, Any]:
    """
    Grid search over parameter combinations. Returns best params and all results.
    objective: sharpe | sortino | profit_factor | final_equity | -max_drawdown
    """
    grid = param_grid or {k: [DEFAULT_PARAM_GRID[k][0]] for k in ["tp", "first_dev", "step_mult", "max_safety"]}
    keys = list(grid.keys())
    vals = [grid[k] for k in keys]
    combos = list(itertools.product(*vals))
    if len(combos) > max_combos:
        random.shuffle(combos)
        combos = combos[:max_combos]

    base = _default_params()
    results = []
    for combo in combos:
        params = {**base, **dict(zip(keys, combo))}
        try:
            res = run_backtest_multi(candles, strategy=strategy, params=params, starting_cash=starting_cash)
            m = res.get("metrics", {})
            score = _score_from_objective(m, objective)
            results.append({"params": params, "metrics": m, "score": score})
        except Exception as e:
            logger.debug("grid combo failed: %s", e)

    if not results:
        return {"ok": False, "error": "No valid results", "results": []}

    best = max(results, key=lambda x: x["score"])
    return {
        "ok": True,
        "best_params": best["params"],
        "best_metrics": best["metrics"],
        "best_score": best["score"],
        "objective": objective,
        "results": sorted(results, key=lambda x: -x["score"])[:50],
        "total_combos": len(combos),
    }


def _score_from_objective(metrics: Dict[str, Any], objective: str) -> float:
    if objective == "sharpe":
        return float(metrics.get("sharpe") or 0)
    if objective == "sortino":
        return float(metrics.get("sortino") or 0)
    if objective == "profit_factor":
        return float(metrics.get("profit_factor") or 0)
    if objective == "final_equity":
        return float(metrics.get("final_equity") or 0)
    if objective == "-max_drawdown":
        return -float(metrics.get("max_drawdown") or 1)
    return float(metrics.get("sharpe") or 0)


def run_genetic_optimization(
    candles: List[List[float]],
    strategy: str = "smart",
    generations: int = 20,
    population_size: int = 30,
    mutation_rate: float = 0.15,
    objective: str = "sharpe",
    starting_cash: float = 1000.0,
) -> Dict[str, Any]:
    """
    Genetic algorithm: evolve parameter vectors. Each individual = param dict.
    """
    param_bounds = {
        "tp": (0.005, 0.04),
        "first_dev": (0.008, 0.04),
        "step_mult": (1.05, 1.6),
        "max_safety": (2, 6),
        "base_quote": (15.0, 40.0),
        "safety_quote": (15.0, 40.0),
    }

    def random_ind() -> Dict[str, Any]:
        out = {}
        for k, b in param_bounds.items():
            if k in ("tp", "first_dev", "step_mult"):
                out[k] = random.uniform(b[0], b[1])
            elif k in ("base_quote", "safety_quote"):
                out[k] = random.uniform(b[0], b[1])
            else:
                out[k] = random.randint(int(b[0]), int(b[1]))
        return out

    def fitness(p: Dict[str, Any]) -> float:
        base = _default_params()
        params = {**base}
        for k, v in p.items():
            if k in param_bounds:
                params[k] = int(v) if k == "max_safety" else float(v)
        params["max_safety"] = max(2, min(6, int(params.get("max_safety", 3))))
        try:
            res = run_backtest_multi(candles, strategy=strategy, params=params, starting_cash=starting_cash)
            return _score_from_objective(res.get("metrics", {}), objective)
        except Exception:
            return -1e9

    population = [random_ind() for _ in range(population_size)]
    for gen in range(generations):
        scored = [(p, fitness(p)) for p in population]
        scored.sort(key=lambda x: -x[1])
        elite = [p for p, _ in scored[: population_size // 4]]
        next_pop = list(elite)
        while len(next_pop) < population_size:
            p1, p2 = random.choice(elite), random.choice(elite)
            child = {}
            for k in param_bounds:
                if random.random() < 0.5:
                    child[k] = p1.get(k, random_ind()[k])
                else:
                    child[k] = p2.get(k, random_ind()[k])
                if random.random() < mutation_rate:
                    child[k] = random_ind()[k]
            next_pop.append(child)
        population = next_pop

    scored = [(p, fitness(p)) for p in population]
    scored.sort(key=lambda x: -x[1])
    best_ind = scored[0][0]
    base = _default_params()
    best_params = {**base}
    for k, v in best_ind.items():
        if k in param_bounds:
            best_params[k] = int(v) if k == "max_safety" else float(v)
    try:
        res = run_backtest_multi(candles, strategy=strategy, params=best_params, starting_cash=starting_cash)
        return {
            "ok": True,
            "best_params": best_params,
            "best_metrics": res.get("metrics", {}),
            "generations": generations,
            "objective": objective,
        }
    except Exception as e:
        return {"ok": False, "error": str(e)}


def run_strategy_comparison(
    candles: List[List[float]],
    strategies: Optional[List[str]] = None,
    params: Optional[Dict[str, Any]] = None,
    starting_cash: float = 1000.0,
) -> Dict[str, Any]:
    """
    Run all strategies on same data. Compare Sharpe, max DD, win rate, profit factor.
    """
    strategies = strategies or list(STRATEGY_REGISTRY.keys())
    base = params or _default_params()
    results = []
    for name in strategies:
        try:
            res = run_backtest_multi(candles, strategy=name, params=base, starting_cash=starting_cash)
            m = res.get("metrics", {})
            results.append({
                "strategy": name,
                "sharpe": m.get("sharpe"),
                "max_drawdown": m.get("max_drawdown"),
                "win_rate": m.get("win_rate"),
                "profit_factor": m.get("profit_factor"),
                "final_equity": m.get("final_equity"),
                "trades": m.get("trades"),
                "regime_stats": m.get("regime_stats"),
            })
        except Exception as e:
            logger.debug("strategy %s failed: %s", name, e)
            results.append({"strategy": name, "error": str(e)})

    valid = [r for r in results if "error" not in r and r.get("sharpe") is not None]
    best_sharpe = max(valid, key=lambda x: float(x.get("sharpe") or 0)) if valid else None
    best_dd = min(valid, key=lambda x: float(x.get("max_drawdown") or 1)) if valid else None

    return {
        "ok": True,
        "strategies": results,
        "best_by_sharpe": best_sharpe.get("strategy") if best_sharpe else None,
        "best_by_drawdown": best_dd.get("strategy") if best_dd else None,
    }


def run_regime_specific_backtest(
    candles: List[List[float]],
    strategy: str = "smart",
    params: Optional[Dict[str, Any]] = None,
    starting_cash: float = 1000.0,
) -> Dict[str, Any]:
    """
    Run full backtest; regime_stats breaks down performance by bull/bear/range/high_vol.
    Identifies which strategies fail in which market conditions.
    """
    base = params or _default_params()
    try:
        res = run_backtest_multi(candles, strategy=strategy, params=base, starting_cash=starting_cash)
        m = res.get("metrics", {})
        regime_stats = m.get("regime_stats", {})
        return {
            "ok": True,
            "overall": {"sharpe": m.get("sharpe"), "max_drawdown": m.get("max_drawdown"), "trades": m.get("trades")},
            "regimes": regime_stats,
        }
    except Exception as e:
        return {"ok": False, "error": str(e)}


def parameter_sensitivity(
    candles: List[List[float]],
    strategy: str = "smart",
    param_name: str = "tp",
    param_values: Optional[List[Any]] = None,
    starting_cash: float = 1000.0,
) -> Dict[str, Any]:
    """
    Vary one parameter, hold others fixed. Return sensitivity curve.
    """
    base = _default_params()
    if param_values is None:
        param_values = DEFAULT_PARAM_GRID.get(param_name, [0.01, 0.012, 0.015])
    curve = []
    for v in param_values:
        p = {**base, param_name: v}
        try:
            res = run_backtest_multi(candles, strategy=strategy, params=p, starting_cash=starting_cash)
            m = res.get("metrics", {})
            curve.append({"value": v, "sharpe": m.get("sharpe"), "max_dd": m.get("max_drawdown"), "final_equity": m.get("final_equity")})
        except Exception as e:
            curve.append({"value": v, "error": str(e)})
    return {"ok": True, "param": param_name, "curve": curve}


def walk_forward_optimize(
    candles: List[List[float]],
    train_days: int = 90,
    test_days: int = 30,
    step_days: int = 30,
    strategy: str = "smart",
    objective: str = "sharpe",
    starting_cash: float = 1000.0,
) -> Dict[str, Any]:
    """
    Rolling walk-forward: train on train_days, test on test_days, roll step_days.
    For each window: grid-optimize on train, then run optimized params on test.
    """
    if not ENABLE_WALK_FORWARD:
        return {"ok": False, "error": "Walk-forward disabled (ENABLE_WALK_FORWARD=0)"}

    ms_per_day = 86400 * 1000
    train_ms = train_days * ms_per_day
    test_ms = test_days * ms_per_day
    step_ms = step_days * ms_per_day

    start_ts = candles[0][0]
    end_ts = candles[-1][0]
    windows = []
    t = start_ts + train_ms
    while t + test_ms <= end_ts:
        train = [c for c in candles if t - train_ms <= c[0] < t]
        test = [c for c in candles if t <= c[0] < t + test_ms]
        if len(train) < 200 or len(test) < 50:
            t += step_ms
            continue

        grid_res = run_grid_search(train, strategy=strategy, objective=objective, max_combos=30)
        best_p = grid_res.get("best_params", _default_params()) if grid_res.get("ok") else _default_params()
        try:
            test_res = run_backtest_multi(test, strategy=strategy, params=best_p, starting_cash=starting_cash)
            windows.append({
                "train_start": int(t - train_ms),
                "train_end": int(t),
                "test_start": int(t),
                "test_end": int(t + test_ms),
                "best_params": best_p,
                "test_metrics": test_res.get("metrics", {}),
            })
        except Exception as e:
            windows.append({"train_start": int(t - train_ms), "error": str(e)})
        t += step_ms

    if not windows:
        return {"ok": False, "error": "No valid walk-forward windows"}

    valid = [w for w in windows if "error" not in w and w.get("test_metrics")]
    avg_sharpe = statistics.mean([float(m.get("sharpe") or 0) for w in valid for m in [w.get("test_metrics", {})]]) if valid else 0
    return {
        "ok": True,
        "windows": windows,
        "windows_count": len(windows),
        "avg_test_sharpe": avg_sharpe,
    }
