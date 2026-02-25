"""
Benchmark & Competitive Analysis - SPY, BTC, peer rank, beta, alpha.
"""
import logging
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple

from stock_metadata import STOCK_SECTOR_MAP, get_sector

logger = logging.getLogger(__name__)

BENCHMARK_STOCKS = "SPY"
BENCHMARK_CRYPTO = "XBT/USD"


def _return_from_candles(candles: List[List[float]], days: int) -> Optional[float]:
    """Compute return over last N days from daily candles. candles: [ts, o, h, l, c, v]."""
    if not candles or len(candles) < 2:
        return None
    n = min(days + 1, len(candles))
    old_close = float(candles[-n][4]) if len(candles[-n]) >= 5 else None
    new_close = float(candles[-1][4]) if len(candles[-1]) >= 5 else None
    if not old_close or not new_close or old_close <= 0:
        return None
    return (new_close - old_close) / old_close


def get_benchmark_return(
    benchmark: str,
    days: int,
    candles: Optional[List[List[float]]] = None,
    fetcher: Optional[Callable[[str, str, int], List[List[float]]]] = None,
) -> Optional[float]:
    """
    Get benchmark return over N days. benchmark='SPY' or 'BTC'.
    Pass candles directly or fetcher(symbol, timeframe, limit).
    """
    if candles:
        return _return_from_candles(candles, days)
    if fetcher:
        try:
            sym = "SPY" if benchmark.upper() in ("SPY", "SP500") else "XBT/USD"
            tf = "1d"
            limit = days + 5
            c = fetcher(sym, tf, limit)
            return _return_from_candles(c or [], days)
        except Exception as e:
            logger.debug("Benchmark fetch failed: %s", e)
    return None


def get_recommendation_vs_benchmark(
    symbol: str,
    asset_return_pct: float,
    days: int,
    benchmark_return_pct: Optional[float] = None,
) -> str:
    """
    Compare recommendation return vs benchmark.
    Returns e.g. "Outperformed SPY by +12.3% over 30 days" or "Underperformed SPY by -5.2%".
    """
    is_stock = len(symbol) <= 5 and "/" not in symbol and symbol.upper() not in ("BTC", "ETH")
    bench_name = "SPY" if is_stock else "BTC"
    if benchmark_return_pct is None:
        return ""
    diff = asset_return_pct - benchmark_return_pct
    if abs(diff) < 0.1:
        return f"In line with {bench_name} over {days} days"
    sign = "+" if diff > 0 else ""
    return f"Outperformed {bench_name} by {sign}{diff:.1f}% over {days} days" if diff > 0 else f"Underperformed {bench_name} by {diff:.1f}% over {days} days"


@dataclass
class PeerRankResult:
    rank: int
    total: int
    quartile: str  # "top", "upper_mid", "lower_mid", "bottom"
    return_30d: float
    return_90d: float
    sector: str


def get_peer_rank(
    symbol: str,
    sector: Optional[str] = None,
    return_30d: Optional[float] = None,
    return_90d: Optional[float] = None,
    peer_returns: Optional[Dict[str, Tuple[float, float]]] = None,
) -> Optional[PeerRankResult]:
    """
    Rank symbol within sector by 30d and 90d returns.
    peer_returns: {symbol: (return_30d, return_90d)}. If None, uses STOCK_SECTOR_MAP peers with placeholder 0.
    Returns PeerRankResult or None.
    """
    sec = sector or get_sector(symbol)
    if not sec:
        return None
    peers = [s for s, ssec in STOCK_SECTOR_MAP.items() if ssec == sec and s != symbol and ssec != "ETF"]
    if not peers:
        return None
    if peer_returns:
        peer_list = [(s, peer_returns.get(s, (0.0, 0.0))) for s in peers if s in peer_returns]
    else:
        # No peer return data: assume peers have 0 return, rank symbol by its own return
        peer_list = [(s, (0.0, 0.0)) for s in peers[:20]]
    if not peer_list:
        return None
    r30 = return_30d if return_30d is not None else 0.0
    r90 = return_90d if return_90d is not None else 0.0
    combined = [(s, (pr[0] * 0.4 + pr[1] * 0.6)) for s, pr in peer_list]
    combined.append((symbol, r30 * 0.4 + r90 * 0.6))
    combined.sort(key=lambda x: -x[1])
    rank = next((i + 1 for i, (s, _) in enumerate(combined) if s == symbol), 0)
    total = len(combined)
    if total == 0:
        return None
    q = rank / total if total > 0 else 0.5
    if q <= 0.25:
        quartile = "top"
    elif q <= 0.5:
        quartile = "upper_mid"
    elif q <= 0.75:
        quartile = "lower_mid"
    else:
        quartile = "bottom"
    return PeerRankResult(rank=rank, total=total, quartile=quartile, return_30d=r30, return_90d=r90, sector=sec or "")


def get_beta(
    asset_returns: List[float],
    market_returns: List[float],
) -> Optional[float]:
    """
    Calculate beta: Cov(asset, market) / Var(market).
    High beta = more aggressive, low beta = defensive.
    """
    if not asset_returns or not market_returns or len(asset_returns) != len(market_returns) or len(asset_returns) < 10:
        return None
    try:
        n = len(asset_returns)
        mean_a = sum(asset_returns) / n
        mean_m = sum(market_returns) / n
        cov = sum((asset_returns[i] - mean_a) * (market_returns[i] - mean_m) for i in range(n)) / n
        var_m = sum((market_returns[i] - mean_m) ** 2 for i in range(n)) / n
        if var_m <= 0:
            return None
        return cov / var_m
    except Exception as e:
        logger.debug("Beta calc failed: %s", e)
        return None


def get_beta_from_candles(
    asset_candles: List[List[float]],
    market_candles: List[List[float]],
) -> Optional[float]:
    """Compute beta from OHLCV candles. Uses close returns."""
    if not asset_candles or not market_candles or len(asset_candles) < 20:
        return None
    def returns(candles):
        out = []
        for i in range(1, len(candles)):
            if len(candles[i]) >= 5 and len(candles[i-1]) >= 5:
                prev = float(candles[i-1][4])
                curr = float(candles[i][4])
                if prev > 0:
                    out.append((curr - prev) / prev)
        return out
    ar = returns(asset_candles)
    mr = returns(market_candles)
    n = min(len(ar), len(mr))
    if n < 10:
        return None
    return get_beta(ar[-n:], mr[-n:])


def get_alpha_by_strategy(days: int = 30, benchmark_fetcher: Optional[Callable[[str, int, int], Optional[float]]] = None) -> Dict[str, Dict[str, Any]]:
    """
    For closed deals: alpha = return - benchmark_return.
    Group by exit_strategy. Returns e.g. {"smart_dca": {"alpha_pct": 8.0, "deals": 5, "avg_return_pct": 12.0}}.
    If benchmark_fetcher provided, computes alpha; otherwise returns avg_return_pct only.
    """
    try:
        import time
        from db import _conn
        since = int(time.time()) - (days * 86400)
        con = _conn()
        rows = con.execute(
            """
            SELECT d.symbol, d.entry_avg, d.exit_avg, d.opened_at, d.closed_at, d.exit_strategy, d.entry_strategy
            FROM deals d
            WHERE d.state='CLOSED' AND d.closed_at >= ? AND d.entry_avg > 0 AND d.exit_avg IS NOT NULL AND d.opened_at IS NOT NULL AND d.closed_at IS NOT NULL
            """,
            (since,),
        ).fetchall()
        con.close()
        by_strategy: Dict[str, List[Tuple[float, Optional[float]]]] = {}  # (return_pct, alpha_pct)
        for r in rows:
            strat = str((r["exit_strategy"] or r["entry_strategy"]) or "unknown").strip() or "unknown"
            entry = float(r["entry_avg"] or 0)
            exit_p = float(r["exit_avg"] or 0)
            ret_pct = ((exit_p - entry) / entry) * 100 if entry > 0 else 0.0
            alpha_pct = None
            if benchmark_fetcher:
                try:
                    bench_ret = benchmark_fetcher(
                        "SPY" if (len(str(r["symbol"] or "")) <= 5 and "/" not in str(r["symbol"] or "")) else "XBT/USD",
                        int(r["opened_at"]), int(r["closed_at"]),
                    )
                    if bench_ret is not None:
                        alpha_pct = ret_pct - (bench_ret * 100)
                except Exception:
                    pass
            if strat not in by_strategy:
                by_strategy[strat] = []
            by_strategy[strat].append((ret_pct, alpha_pct))
        result = {}
        for strat, items in by_strategy.items():
            returns = [x[0] for x in items]
            alphas = [x[1] for x in items if x[1] is not None]
            avg_ret = sum(returns) / len(returns) if returns else 0
            avg_alpha = sum(alphas) / len(alphas) if alphas else None
            out = {"avg_return_pct": round(avg_ret, 2), "deals": len(items)}
            if avg_alpha is not None:
                out["alpha_pct"] = round(avg_alpha, 2)
            result[strat] = out
        return result
    except Exception as e:
        logger.debug("Alpha by strategy failed: %s", e)
        return {}


def get_alpha_vs_benchmark(
    symbol: str,
    entry_ts: int,
    exit_ts: int,
    deal_return_pct: float,
    fetcher: Optional[Callable[[str, int, int], Optional[float]]] = None,
) -> Optional[float]:
    """
    Alpha = deal_return - benchmark_return over same period.
    fetcher(symbol, start_ts, end_ts) -> benchmark_return as decimal (e.g. 0.05 for 5%).
    """
    if not fetcher:
        return None
    try:
        is_stock = len(symbol) <= 5 and "/" not in symbol
        bench = "SPY" if is_stock else "XBT/USD"
        bench_ret = fetcher(bench, entry_ts, exit_ts)
        if bench_ret is None:
            return None
        alpha = (deal_return_pct / 100.0) - bench_ret
        return alpha * 100
    except Exception as e:
        logger.debug("Alpha vs benchmark failed: %s", e)
        return None


def enrich_recommendation_with_benchmark(
    symbol: str,
    current_price: float,
    candles_1d: Optional[List[List[float]]] = None,
    benchmark_candles: Optional[List[List[float]]] = None,
    sector: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Add benchmark comparison, peer rank, beta to recommendation metadata.
    """
    result = {"benchmark_vs": "", "peer_rank": "", "beta": None, "peer_quartile": ""}
    if not candles_1d or len(candles_1d) < 30:
        return result
    ret_30 = _return_from_candles(candles_1d, 30)
    ret_90 = _return_from_candles(candles_1d, 90)
    is_stock = len(symbol) <= 5 and "/" not in symbol
    if benchmark_candles:
        bench_ret_30 = _return_from_candles(benchmark_candles, 30)
        if bench_ret_30 is not None and ret_30 is not None:
            diff = (ret_30 - bench_ret_30) * 100
            bench_name = "SPY" if is_stock else "BTC"
            if abs(diff) < 0.1:
                result["benchmark_vs"] = f"In line with {bench_name} over 30 days"
            else:
                sign = "+" if diff > 0 else ""
                result["benchmark_vs"] = f"Outperformed {bench_name} by {sign}{diff:.1f}% over 30 days" if diff > 0 else f"Underperformed {bench_name} by {diff:.1f}% over 30 days"
            result["benchmark_return_30d"] = bench_ret_30 * 100
    if ret_30 is not None:
        result["asset_return_30d"] = ret_30 * 100
    if ret_90 is not None:
        result["asset_return_90d"] = ret_90 * 100
    peer = get_peer_rank(symbol, sector, ret_30 * 100 if ret_30 else None, ret_90 * 100 if ret_90 else None)
    if peer:
        result["peer_rank"] = f"{symbol} rank {peer.rank}/{peer.total} in {peer.sector}"
        result["peer_quartile"] = peer.quartile
    if benchmark_candles and len(candles_1d) >= 20:
        beta = get_beta_from_candles(candles_1d, benchmark_candles)
        if beta is not None:
            result["beta"] = round(beta, 2)
    return result
