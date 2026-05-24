"""
Explore strategy backtester — same detection as live, on historical bars.
Persists aggregates via db.save_explore_backtest_results.
"""
from __future__ import annotations

import json
import logging
import time
from typing import Any, Dict, List, Tuple

from explore_signals import STRATEGY_LABELS, detect_strategies_at
from explore_scorer import compute_conviction, status_from_conviction

logger = logging.getLogger(__name__)

# Asset-class-appropriate SL/TP.
# Crypto daily ranges are 4–8%; a 2% stop gets hit by normal intraday noise.
# Stocks are less volatile; 4% stop gives room to breathe without being too wide.
STOP_LOSS_STOCK = 0.04
TAKE_PROFIT_STOCK = 0.10
STOP_LOSS_CRYPTO = 0.07
TAKE_PROFIT_CRYPTO = 0.18


def _forward_outcome(
    opens: List[float],
    closes: List[float],
    highs: List[float],
    lows: List[float],
    entry_i: int,
    is_crypto: bool = False,
    max_hold: int = 20,
) -> float:
    """
    Long-only: daily bars. Entry at close[entry_i].
    Uses asset-class SL/TP (stock: 4%/10%, crypto: 7%/18%).
    Gap handling: if next bar opens below stop → exit at open (gap down).
                  if next bar opens above TP  → exit at TP price.
    Same-bar: TP wins (intraday assumption: bullish gap-through more likely
              to fill TP before stopping out on the same candle).
    Time exit: close at close[entry_i + max_hold] if neither hit.
    Returns PnL fraction (e.g. 0.10 = +10%).
    """
    sl = STOP_LOSS_CRYPTO if is_crypto else STOP_LOSS_STOCK
    tp = TAKE_PROFIT_CRYPTO if is_crypto else TAKE_PROFIT_STOCK

    if entry_i >= len(closes) - 1:
        return 0.0
    entry = closes[entry_i]
    if entry <= 0:
        return 0.0

    stop_px = entry * (1.0 - sl)
    tp_px = entry * (1.0 + tp)
    end = min(len(closes) - 1, entry_i + max_hold)

    for j in range(entry_i + 1, end + 1):
        # Use open of this bar; fall back to prior close if opens array is shorter
        op = opens[j] if j < len(opens) else closes[j - 1]
        lo = lows[j]
        hi = highs[j]

        # Gap handling: open already below stop → exit at open (slippage)
        if op <= stop_px:
            return (op - entry) / entry

        # Gap handling: open already above TP → exit at TP price (capped)
        if op >= tp_px:
            return tp

        hit_tp = hi >= tp_px
        hit_stop = lo <= stop_px

        # Same-bar: TP wins (conservative re: stop, generous re: TP)
        if hit_tp:
            return tp
        if hit_stop:
            return -sl

    # Time exit at last close in window
    return (closes[end] - entry) / entry


def _ohlcv_arrays(
    candles: List[List[float]],
) -> Tuple[List[float], List[float], List[float], List[float]]:
    """Returns (opens, highs, lows, closes) from [[ts,o,h,l,c,v],...] rows."""
    o, h, l, c = [], [], [], []
    for row in candles:
        if len(row) < 5:
            continue
        try:
            o.append(float(row[1]))
            h.append(float(row[2]))
            l.append(float(row[3]))
            c.append(float(row[4]))
        except (TypeError, ValueError):
            continue
    return o, h, l, c


def run_strategy_backtest_on_candles(
    candles: List[List[float]],
    *,
    is_crypto: bool,
    lookback_start: int,
    lookback_end: int,
) -> Dict[str, Any]:
    """
    Scan bars in [lookback_start, lookback_end] (indices), aggregate per strategy.
    lookback_end inclusive, relative to candle length.
    """
    opens, highs, lows, closes = _ohlcv_arrays(candles)
    n = len(closes)
    if n < 220:
        return {}

    per_strat: Dict[str, List[float]] = {k: [] for k in STRATEGY_LABELS}
    counts_window = {k: 0 for k in STRATEGY_LABELS}

    for i in range(max(60, lookback_start), min(lookback_end, n - 21)):
        matches = detect_strategies_at(candles, i, is_crypto=is_crypto, btc_ctx=None)
        if not matches:
            continue
        # pick best at this bar (same as live)
        best = None
        best_c = -1
        for sid, _reason, facts in matches:
            conv = compute_conviction(sid, facts, is_crypto=is_crypto)
            if conv > best_c:
                best_c = conv
                best = sid
        if not best or status_from_conviction(best_c, best) == "rejected":
            continue
        pnl = _forward_outcome(opens, closes, highs, lows, i, is_crypto=is_crypto, max_hold=20)
        per_strat.setdefault(best, []).append(pnl)
        counts_window[best] = counts_window.get(best, 0) + 1

    out: Dict[str, Any] = {}
    for sid in STRATEGY_LABELS:
        pnls = per_strat.get(sid) or []
        if not pnls:
            out[sid] = {
                "label": STRATEGY_LABELS[sid],
                "signals": 0,
                "win_rate": None,
                "avg_return": None,
                "best": None,
                "worst": None,
            }
            continue
        wins = sum(1 for p in pnls if p > 0)
        out[sid] = {
            "label": STRATEGY_LABELS[sid],
            "signals": len(pnls),
            "signals_window": counts_window.get(sid, 0),
            "win_rate": wins / len(pnls),
            "avg_return": sum(pnls) / len(pnls),
            "best": max(pnls),
            "worst": min(pnls),
            "low_accuracy": (wins / len(pnls)) < 0.55,
        }
    return out


def run_explore_backtest(
    *,
    fetch_candles,
    stock_symbols: List[str],
    crypto_symbols: List[str],
    horizon: str = "short",
) -> Dict[str, Any]:
    """
    fetch_candles(symbol) -> list of daily candles [[ts,o,h,l,c,v],...]
    """
    t0 = time.time()
    n = len(stock_symbols) + len(crypto_symbols)
    agg_30: Dict[str, Dict[str, Any]] = {k: {"pnls": [], "n": 0} for k in STRATEGY_LABELS}
    agg_60 = {k: {"pnls": [], "n": 0} for k in STRATEGY_LABELS}
    agg_90 = {k: {"pnls": [], "n": 0} for k in STRATEGY_LABELS}

    def _process(sym: str, is_crypto: bool) -> None:
        try:
            candles = fetch_candles(sym) or []
        except Exception as e:
            logger.debug("backtest fetch %s: %s", sym, e)
            return
        opens, highs, lows, closes = _ohlcv_arrays(candles)
        nc = len(closes)
        if nc < 220:
            return

        def _run_window(bar_start: int, bar_end: int, bucket: Dict[str, Dict[str, Any]]) -> None:
            for i in range(max(60, bar_start), min(bar_end, nc - 21)):
                matches = detect_strategies_at(candles, i, is_crypto=is_crypto, btc_ctx=None)
                if not matches:
                    continue
                best_sid = None
                best_c = -1
                for sid, _r, facts in matches:
                    conv = compute_conviction(sid, facts, is_crypto=is_crypto)
                    if conv > best_c:
                        best_c = conv
                        best_sid = sid
                if not best_sid or status_from_conviction(best_c, best_sid) == "rejected":
                    continue
                pnl = _forward_outcome(opens, closes, highs, lows, i, is_crypto=is_crypto, max_hold=20)
                bucket[best_sid]["pnls"].append(pnl)
                bucket[best_sid]["n"] += 1

        _run_window(nc - 30, nc - 1, agg_30)
        _run_window(nc - 60, nc - 1, agg_60)
        _run_window(nc - 90, nc - 1, agg_90)

    for s in stock_symbols:
        _process(s, False)
    for s in crypto_symbols:
        _process(s, True)

    def _finalize(bucket: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        rows = {}
        for sid in STRATEGY_LABELS:
            pnls = bucket[sid]["pnls"]
            sigs = bucket[sid]["n"]
            if not pnls:
                rows[sid] = {
                    "strategy": STRATEGY_LABELS[sid],
                    "signals": sigs,
                    "win_rate": None,
                    "avg_return": None,
                    "best_trade": None,
                    "worst_trade": None,
                    "warn": False,
                }
                continue
            wins = sum(1 for p in pnls if p > 0)
            wr = wins / len(pnls)
            rows[sid] = {
                "strategy": STRATEGY_LABELS[sid],
                "signals": sigs,
                "win_rate": round(wr * 100, 1),
                "avg_return": round(sum(pnls) / len(pnls) * 100, 2),
                "best_trade": round(max(pnls) * 100, 2),
                "worst_trade": round(min(pnls) * 100, 2),
                "warn": wr < 0.55,
            }
        return rows

    result = {
        "horizon": horizon,
        "computed_ts": int(time.time()),
        "elapsed_sec": round(time.time() - t0, 2),
        "symbols_processed": n,
        "windows": {
            "30d": _finalize(agg_30),
            "60d": _finalize(agg_60),
            "90d": _finalize(agg_90),
        },
    }
    return result


def default_universe_symbols() -> Tuple[List[str], List[str]]:
    """Liquid subset for backtest runtime."""
    stocks = [
        "AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META", "TSLA", "JPM", "V", "UNH",
        "XOM", "JNJ", "WMT", "PG", "MA", "HD", "CVX", "MRK", "ABBV", "PEP",
        "COST", "AVGO", "KO", "MCD", "CSCO", "ACN", "TMO", "DHR", "ABT", "WFC",
        "CRM", "ADBE", "BAC", "DIS", "NFLX", "TXN", "PM", "NEE", "ORCL", "INTC",
    ]
    crypto = [
        "BTC/USD",
        "ETH/USD",
        "SOL/USD",
        "XRP/USD",
        "ADA/USD",
        "DOGE/USD",
        "DOT/USD",
        "LINK/USD",
        "AVAX/USD",
        "LTC/USD",
        "ATOM/USD",
        "UNI/USD",
        "BCH/USD",
        "XLM/USD",
        "ETC/USD",
    ]
    return stocks, crypto
