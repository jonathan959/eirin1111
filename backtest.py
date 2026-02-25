import argparse
import csv
import math
import os
import statistics
import time
from typing import List, Dict, Any, Optional

from strategies import (
    DealState,
    AccountSnapshot,
    PerformanceStats,
    detect_regime,
    get_strategy,
    StrategyContext,
)
from kraken_client import KrakenClient
from executor import modeled_cost_pct

# Config from env (slippage & commission modeling)
INCLUDE_SLIPPAGE = os.getenv("INCLUDE_SLIPPAGE", "1").strip().lower() in ("1", "true", "yes")
COMMISSION_PCT = float(os.getenv("COMMISSION_PCT", "0.26"))  # 0 for stocks, 0.26 for crypto
SLIPPAGE_PCT = float(os.getenv("SLIPPAGE_PCT", "0.1"))
ENABLE_WALK_FORWARD = os.getenv("ENABLE_WALK_FORWARD", "1").strip().lower() in ("1", "true", "yes")


def _cost_params() -> Dict[str, float]:
    """Derive fee and slippage from env."""
    fee = COMMISSION_PCT / 100.0
    slip = (SLIPPAGE_PCT / 100.0) if INCLUDE_SLIPPAGE else 0.0005
    return {"fee_rate": fee, "slippage": slip}


def load_candles(path: str) -> List[List[float]]:
    candles: List[List[float]] = []
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            ts = float(r.get("timestamp") or r.get("ts") or 0)
            o = float(r.get("open") or 0)
            h = float(r.get("high") or 0)
            l = float(r.get("low") or 0)
            c = float(r.get("close") or 0)
            v = float(r.get("volume") or 0)
            candles.append([ts, o, h, l, c, v])
    return candles


def run_backtest_multi(
    candles: List[List[float]],
    strategy: str,
    params: dict,
    starting_cash: float = 1000.0,
) -> dict:
    cash = starting_cash
    position = 0.0
    avg_entry = None
    safety_used = 0
    realized = 0.0
    peak = starting_cash
    max_dd = 0.0
    win = 0
    loss = 0
    trades = 0
    fees_paid = 0.0
    equity_curve = []
    orders = []
    regime_stats = {}

    cost = _cost_params()
    fee_rate = float(params.get("fee_rate", cost["fee_rate"]))
    slip = float(params.get("slippage", cost["slippage"]))
    warmup = int(params.get("warmup", 60))
    base_risk_pct = float(params.get("base_risk_pct", 0.02))
    base_quote = float(params.get("base_quote", 0.0))

    # Ensure candle ordering to avoid lookahead
    for i in range(1, len(candles)):
        if candles[i][0] < candles[i - 1][0]:
            raise ValueError("Candles must be time-ordered (no lookahead)")

    strat = get_strategy(strategy)

    for i in range(warmup, len(candles)):
        window = candles[: i + 1]
        current_ts = window[-1][0]
        if any(c[0] > current_ts for c in window):
            raise ValueError("Future candle detected in indicator window")
        last_price = float(window[-1][4])
        equity = cash + position * last_price
        if equity > peak:
            peak = equity
        if peak > 0:
            max_dd = max(max_dd, (peak - equity) / peak)
        equity_curve.append({"time": int(window[-1][0]), "value": float(equity)})

        spent_quote = float(position * float(avg_entry or last_price))
        deal = DealState(
            avg_entry=avg_entry,
            position_size=position,
            safety_used=safety_used,
            tp_price=None,
            spent_quote=spent_quote,
        )
        account = AccountSnapshot(
            total_usd=cash + position * last_price,
            free_usd=cash,
            used_usd=0.0,
            positions_usd=position * last_price,
        )
        perf = PerformanceStats(realized_today=realized, drawdown=max_dd, open_deals=1 if position > 0 else 0)
        regime = detect_regime(window)
        regime_label = regime.regime

        ctx = StrategyContext(
            symbol="SIM",
            last_price=last_price,
            candles_5m=window,
            candles_15m=window,
            candles_1h=window,
            candles_4h=window,
            deal=deal,
            account=account,
            perf=perf,
            now_ts=int(window[-1][0]),
            cooldown_until=0,
            cfg=params,
            regime=regime,
        )
        decision = strat.decide(ctx)

        # process existing limit orders
        filled = []
        for o in orders:
            if o["side"] == "buy" and window[-1][3] <= o["price"]:
                filled.append(o)
            if o["side"] == "sell" and window[-1][2] >= o["price"]:
                filled.append(o)
        for o in filled:
            orders.remove(o)
            px = o["price"]
            if o["side"] == "buy" and cash >= (o["amount"] * px):
                cost = o["amount"] * px
                fee = cost * fee_rate
                fees_paid += fee
                cash -= cost + fee
                position += o["amount"]
                avg_entry = ((avg_entry or 0) * (position - o["amount"]) + cost) / position
            if o["side"] == "sell" and position >= o["amount"]:
                proceeds = o["amount"] * px
                fee = proceeds * fee_rate
                fees_paid += fee
                cash += proceeds - fee
                pnl = proceeds - (o["amount"] * float(avg_entry or px))
                realized += pnl
                trades += 1
                rs = regime_stats.setdefault(regime_label, {"trades": 0, "wins": 0, "pnl": 0.0})
                rs["trades"] += 1
                rs["pnl"] += pnl
                if pnl >= 0:
                    win += 1
                    rs["wins"] += 1
                else:
                    loss += 1
                position -= o["amount"]
                if position <= 0:
                    avg_entry = None
                    safety_used = 0

        if decision.action in ("ENTER", "SAFETY_ORDER"):
            risk_size_quote = max(0.0, equity * base_risk_pct)
            default_size_quote = base_quote if base_quote > 0 else risk_size_quote
            size_quote = float((decision.order or {}).get("size_quote") or default_size_quote)
            if cash >= size_quote and last_price > 0:
                # Cost model (no lookahead): use ATR-derived volatility and assumed spread
                atr_val = detect_regime(window).atr14 or 0.0
                vol_pct = (atr_val / last_price) if atr_val and last_price > 0 else 0.0
                cost_pct = modeled_cost_pct(0.001, vol_pct)
                fill_px = last_price * (1.0 + max(slip, cost_pct))
                size_base = size_quote / fill_px
                cost = size_base * fill_px
                fee = cost * fee_rate
                fees_paid += fee
                cash -= cost + fee
                position += size_base
                avg_entry = ((avg_entry or 0) * (position - size_base) + cost) / position
                safety_used += 1

        elif decision.action in ("TAKE_PROFIT", "EXIT", "STOP_LOSS"):
            size_base = float((decision.order or {}).get("size_base") or position)
            if size_base > 0 and position > 0:
                ord_type = (decision.order or {}).get("type", "market")
                px = float((decision.order or {}).get("price") or last_price)
                if ord_type == "limit":
                    orders.append({"side": "sell", "price": px, "amount": size_base})
                else:
                    atr_val = detect_regime(window).atr14 or 0.0
                    vol_pct = (atr_val / px) if atr_val and px > 0 else 0.0
                    cost_pct = modeled_cost_pct(0.001, vol_pct)
                    fill_px = px * (1.0 - max(slip, cost_pct))
                    proceeds = size_base * fill_px
                    fee = proceeds * fee_rate
                    fees_paid += fee
                    cash += proceeds - fee
                    pnl = proceeds - (size_base * float(avg_entry or fill_px))
                    realized += pnl
                    trades += 1
                    rs = regime_stats.setdefault(regime_label, {"trades": 0, "wins": 0, "pnl": 0.0})
                    rs["trades"] += 1
                    rs["pnl"] += pnl
                    if pnl >= 0:
                        win += 1
                        rs["wins"] += 1
                    else:
                        loss += 1
                    position -= size_base
                    if position <= 0:
                        avg_entry = None
                        safety_used = 0

        elif decision.action == "GRID_MAINTAIN":
            grid_low = float(decision.order.get("grid_low") or last_price * 0.95)
            grid_high = float(decision.order.get("grid_high") or last_price * 1.05)
            levels = int(decision.order.get("levels") or 6)
            step = (grid_high - grid_low) / max(1, levels)
            grid_quote = base_quote if base_quote > 0 else max(0.0, equity * base_risk_pct)
            for i in range(1, levels):
                px = grid_low + (step * i)
                if px < last_price:
                    orders.append({"side": "buy", "price": px, "amount": (grid_quote / px)})
                elif px > last_price and position > 0:
                    orders.append({"side": "sell", "price": px, "amount": position / max(1, levels)})

    final_equity = cash + position * float(candles[-1][4])
    win_rate = (win / trades) * 100 if trades else 0.0
    profit_factor = (sum([x for x in [realized] if x > 0]) / abs(sum([x for x in [realized] if x < 0])) if realized < 0 else 1.0)
    returns = []
    if equity_curve:
        for i in range(1, len(equity_curve)):
            prev = equity_curve[i - 1]["value"]
            cur = equity_curve[i]["value"]
            if prev > 0:
                returns.append((cur - prev) / prev)
    mean_ret = sum(returns) / len(returns) if returns else 0.0
    std_ret = statistics.pstdev(returns) if len(returns) > 1 else 0.0
    sharpe = (mean_ret / std_ret) * math.sqrt(252) if std_ret > 0 else 0.0
    neg_returns = [r for r in returns if r < 0]
    neg_std = statistics.pstdev(neg_returns) if len(neg_returns) > 1 else 0.0
    sortino = (mean_ret / neg_std) * math.sqrt(252) if neg_std > 0 else 0.0

    metrics = {
        "final_equity": final_equity,
        "realized_pnl": realized,
        "max_drawdown": max_dd,
        "trades": trades,
        "win_rate": win_rate,
        "fees": fees_paid,
        "sharpe": sharpe,
        "sortino": sortino,
        "profit_factor": profit_factor,
        "regime_stats": {k: {"trades": v["trades"], "win_rate": (v["wins"]/v["trades"]) if v["trades"] else 0.0, "pnl": v["pnl"]} for k, v in regime_stats.items()},
        "assumptions": "close-only fills, no intrabar, modeled spread+slippage",
    }
    return {"metrics": metrics, "equity": equity_curve}


def run_backtest_validated(
    candles: List[List[float]],
    strategy: str = "smart",
    params: Optional[Dict[str, Any]] = None,
    oos_pct: float = 30.0,
    starting_cash: float = 1000.0,
) -> Dict[str, Any]:
    """
    Out-of-sample validation: train on (100-oos_pct)%, test on last oos_pct%.
    Compare in-sample vs out-of-sample to detect overfitting.
    """
    if len(candles) < 200:
        return {"ok": False, "error": "Need at least 200 candles for validation"}
    split = max(100, int(len(candles) * (1 - oos_pct / 100.0)))
    in_sample = candles[:split]
    out_sample = candles[split:]
    base = params or {}
    cost = _cost_params()
    p = {**cost, "warmup": 60, "base_risk_pct": 0.02, "base_quote": 25.0, **base}
    try:
        res_is = run_backtest_multi(in_sample, strategy=strategy, params=p, starting_cash=starting_cash)
        res_oos = run_backtest_multi(out_sample, strategy=strategy, params=p, starting_cash=starting_cash)
        m_is = res_is.get("metrics", {})
        m_oos = res_oos.get("metrics", {})
        sharpe_is = float(m_is.get("sharpe") or 0)
        sharpe_oos = float(m_oos.get("sharpe") or 0)
        ratio = (sharpe_oos / sharpe_is) if sharpe_is and sharpe_is != 0 else 0
        return {
            "ok": True,
            "in_sample": m_is,
            "out_sample": m_oos,
            "oos_sharpe_ratio": round(ratio, 3),
            "overfitting_warning": ratio < 0.5 and sharpe_is > 0.5,
        }
    except Exception as e:
        return {"ok": False, "error": str(e)}


def run_walk_forward_kraken(
    symbol: str,
    timeframe: str,
    train_days: int,
    test_days: int,
    step_days: int,
    strategy: str,
    params: dict,
    starting_cash: float = 1000.0,
) -> Dict[str, Any]:
    if not ENABLE_WALK_FORWARD:
        return {"ok": False, "error": "Walk-forward disabled (ENABLE_WALK_FORWARD=0)"}
    kc = KrakenClient()
    now_ms = int(time.time() * 1000)
    total_days = train_days + test_days + 5
    since_ms = now_ms - int(total_days * 86400 * 1000)
    candles = kc.fetch_ohlcv_range(symbol, timeframe, since_ms, now_ms)
    if not candles:
        return {"ok": False, "error": "No candles fetched"}

    step_ms = int(step_days * 86400 * 1000)
    train_ms = int(train_days * 86400 * 1000)
    test_ms = int(test_days * 86400 * 1000)

    windows = []
    start = candles[0][0]
    end = candles[-1][0]
    t = start + train_ms
    while t + test_ms <= end:
        train_start = t - train_ms
        train_end = t
        test_start = t
        test_end = t + test_ms

        train = [c for c in candles if train_start <= c[0] < train_end]
        test = [c for c in candles if test_start <= c[0] < test_end]
        if len(train) < 100 or len(test) < 30:
            t += step_ms
            continue

        result = run_backtest_multi(test, strategy=strategy, params=params, starting_cash=starting_cash)
        windows.append(
            {
                "train_start": int(train_start),
                "train_end": int(train_end),
                "test_start": int(test_start),
                "test_end": int(test_end),
                "metrics": result.get("metrics") or {},
            }
        )
        t += step_ms

    if not windows:
        return {"ok": False, "error": "No walk-forward windows"}

    final_equities = [float(w["metrics"].get("final_equity", 0.0)) for w in windows]
    avg_final = sum(final_equities) / len(final_equities)
    return {
        "ok": True,
        "symbol": symbol,
        "timeframe": timeframe,
        "strategy": strategy,
        "windows": windows,
        "avg_final_equity": avg_final,
        "windows_count": len(windows),
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--csv", required=True, help="CSV with timestamp,open,high,low,close,volume")
    p.add_argument("--base-quote", type=float, default=25.0)
    p.add_argument("--safety-quote", type=float, default=25.0)
    p.add_argument("--max-safety", type=int, default=3)
    p.add_argument("--tp", type=float, default=0.012)
    args = p.parse_args()

    candles = load_candles(args.csv)
    params = {
        "base_quote": args.base_quote,
        "safety_quote": args.safety_quote,
        "max_safety": args.max_safety,
        "tp": args.tp,
    }
    res = run_backtest_multi(candles, strategy="smart", params=params)
    print("=== Backtest Results ===")
    print(res["metrics"])


if __name__ == "__main__":
    main()
