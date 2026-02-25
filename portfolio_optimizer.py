"""
Multi-strategy portfolio optimization. (12.md Part 2)
Finds optimal capital allocation across strategies to maximize Sharpe ratio.
"""
import logging
import os
from typing import Dict, List

logger = logging.getLogger(__name__)
ENABLED = os.getenv("ENABLE_PORTFOLIO_OPTIMIZATION", "0").strip().lower() in ("1", "true", "yes")


def _strategy_metrics() -> Dict:
    """Get historical performance metrics per strategy from deals."""
    strategies = ["smart_dca", "trend_follow", "grid", "range_mean_reversion", "classic"]
    out = {}
    try:
        from db import _conn
        con = _conn()
        for s in strategies:
            rows = con.execute(
                "SELECT pnl_pct FROM deals WHERE (exit_strategy = ? OR entry_strategy = ?) AND state = 'CLOSED' AND closed_at IS NOT NULL AND pnl_pct IS NOT NULL LIMIT 100",
                (s, s)
            ).fetchall()
            con.close()
            if rows and len(rows) >= 5:
                pnls = [r["pnl_pct"] / 100 for r in rows]
                mean_r = sum(pnls) / len(pnls)
                vol = (sum((x - mean_r) ** 2 for x in pnls) / len(pnls)) ** 0.5 or 0.01
                out[s] = {"mean_return": mean_r, "volatility": vol, "max_drawdown": 0.1}
            else:
                out[s] = {"mean_return": 0.05, "volatility": 0.15, "max_drawdown": 0.10}
    except Exception as e:
        logger.debug("portfolio_optimizer metrics: %s", e)
        for s in strategies:
            out[s] = {"mean_return": 0.05, "volatility": 0.15, "max_drawdown": 0.10}
    return out


def optimize_allocation(
    total_capital: float,
    risk_tolerance: str = "moderate",
) -> Dict:
    """Find optimal capital allocation across strategies."""
    if not ENABLED:
        return {"allocations": {}, "expected_return": 0.1, "sharpe_ratio": 0}
    try:
        import numpy as np
        from scipy.optimize import minimize
        metrics = _strategy_metrics()
        strategies = list(metrics.keys())
        returns = np.array([metrics[s]["mean_return"] for s in strategies])
        vols = np.array([metrics[s]["volatility"] for s in strategies])
        n = len(strategies)
        cov = np.outer(vols, vols) * 0.5
        np.fill_diagonal(cov, vols ** 2)

        def neg_sharpe(w):
            r = np.dot(w, returns)
            v = np.sqrt(np.dot(w, np.dot(cov, w))) or 1e-6
            return -r / v

        bounds = [(0.05, 0.40) for _ in strategies]
        result = minimize(neg_sharpe, np.ones(n) / n, method="SLSQP", bounds=bounds,
                         constraints={"type": "eq", "fun": lambda w: np.sum(w) - 1.0})
        weights = result.x if result.success else np.ones(n) / n
        alloc = {strategies[i]: float(weights[i]) for i in range(n) if weights[i] > 0.01}
        port_ret = float(np.dot(weights, returns))
        port_vol = float(np.sqrt(np.dot(weights, np.dot(cov, weights))) or 1e-6)
        sharpe = port_ret / port_vol
        return {
            "allocations": alloc,
            "capital_amounts": {k: v * total_capital for k, v in alloc.items()},
            "expected_return": port_ret,
            "expected_volatility": port_vol,
            "sharpe_ratio": sharpe,
        }
    except Exception as e:
        logger.exception("portfolio_optimizer: %s", e)
        return {"allocations": {}, "expected_return": 0.1, "sharpe_ratio": 0}
