#!/usr/bin/env python
"""
Backtest screener signals over historical data.
Simulates: at each date, what would the screener recommend? Buy and hold 7 days, measure outcome.

Usage: python backtest_screener_signals.py [--days 90] [--min-score 70]
"""
import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
os.chdir(ROOT)
sys.path.insert(0, str(ROOT))

from env_utils import load_env
load_env()


def fetch_candles_yf(symbol: str, days: int = 120) -> list:
    """Fetch daily candles via yfinance. Returns [[ts, o, h, l, c, v], ...]"""
    try:
        import yfinance as yf
        from datetime import datetime, timezone, timedelta
        end = datetime.now(timezone.utc)
        start = end - timedelta(days=days)
        sym_clean = symbol.replace("/", "-").replace("XBT", "BTC")
        ticker = yf.Ticker(sym_clean)
        df = ticker.history(start=start, end=end, interval="1d")
        if df is None or df.empty or len(df) < 30:
            return []
        out = []
        for idx, row in df.iterrows():
            ts = int(row.name.timestamp()) if hasattr(row.name, "timestamp") else 0
            out.append([
                ts,
                float(row.get("Open", 0)),
                float(row.get("High", 0)),
                float(row.get("Low", 0)),
                float(row.get("Close", 0)),
                float(row.get("Volume", 0)),
            ])
        return sorted(out, key=lambda x: x[0])
    except Exception as e:
        return []


def simple_signal_score(candles: list, min_candles: int = 50) -> tuple:
    """
    Simplified regime-based score (no ML, no external deps).
    Returns (score 0-100, regime_str).
    """
    if not candles or len(candles) < min_candles:
        return 0.0, "NO_DATA"
    closes = [c[4] for c in candles]
    # EMA trend
    def ema(arr, span):
        if not arr:
            return 0.0
        k = 2.0 / (span + 1)
        e = arr[0]
        for x in arr[1:]:
            e = k * x + (1 - k) * e
        return e
    ema20 = ema(closes, 20)
    ema50 = ema(closes, 50)
    price = closes[-1]
    if ema20 <= 0:
        return 0.0, "RANGE"
    # Bullish: price > ema20 > ema50
    if price > ema20 and ema20 > ema50:
        score = 70.0 + min(20, (price - ema50) / ema50 * 100)
        return min(95, score), "BULL"
    if price < ema20 and ema20 < ema50:
        return 25.0, "BEAR"
    return 50.0, "RANGE"


def run_backtest(days: int = 90, min_score: float = 70, hold_days: int = 7):
    """Run backtest on top symbols."""
    symbols_stocks = [
        "AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META", "TSLA", "JPM", "V", "JNJ",
        "WMT", "PG", "MA", "UNH", "HD", "DIS", "BAC", "XOM", "PYPL", "NFLX",
        "CRM", "ADBE", "INTC", "AMD", "CSCO", "ORCL", "IBM", "QCOM", "TXN", "AVGO",
        "NKE", "CMCSA", "PEP", "KO", "COST", "ABT", "TMO", "ABBV", "MRK", "LLY",
        "BMY", "AMGN", "GILD", "MDT", "DHR", "HON", "UPS", "LOW", "CAT", "DE",
    ]
    symbols_crypto = ["BTC-USD", "ETH-USD", "SOL-USD", "XRP-USD", "ADA-USD", "DOGE-USD", "AVAX-USD", "LINK-USD", "DOT-USD", "MATIC-USD"]
    symbols = symbols_stocks[:50] + symbols_crypto[:30]

    signals = []
    for sym in symbols:
        candles = fetch_candles_yf(sym, days=days + 30)
        if len(candles) < 60:
            continue
        # Simulate day-by-day: at each day, compute signal, "buy" if score >= min_score
        for i in range(50, len(candles) - hold_days):
            window = candles[: i + 1]
            score, regime = simple_signal_score(window, min_candles=30)
            if score < min_score:
                continue
            entry_price = window[-1][4]
            exit_idx = min(i + hold_days, len(candles) - 1)
            exit_price = candles[exit_idx][4]
            ret = (exit_price - entry_price) / entry_price if entry_price > 0 else 0
            signals.append({
                "symbol": sym,
                "entry_ts": window[-1][0],
                "entry_price": entry_price,
                "exit_price": exit_price,
                "return_pct": ret * 100,
                "score": score,
                "regime": regime,
            })
    return signals


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--days", type=int, default=90)
    p.add_argument("--min-score", type=float, default=70)
    p.add_argument("--hold-days", type=int, default=7)
    args = p.parse_args()
    print(f"Backtesting screener signals: {args.days}d, min_score={args.min_score}, hold={args.hold_days}d")
    signals = run_backtest(days=args.days, min_score=args.min_score, hold_days=args.hold_days)
    if not signals:
        print("No signals generated. Check yfinance / symbol availability.")
        return
    wins = [s for s in signals if s["return_pct"] > 0]
    losses = [s for s in signals if s["return_pct"] <= 0]
    total = len(signals)
    win_rate = len(wins) / total * 100 if total else 0
    avg_profit = sum(s["return_pct"] for s in wins) / len(wins) if wins else 0
    avg_loss = sum(s["return_pct"] for s in losses) / len(losses) if losses else 0
    eq = 1000.0
    peak = eq
    max_dd = 0.0
    for s in signals:
        eq *= 1 + s["return_pct"] / 100
        if eq > peak:
            peak = eq
        if peak > 0:
            max_dd = max(max_dd, (peak - eq) / peak * 100)
    print("\n=== Backtest Results (Simplified Signal) ===")
    print(f"Total signals: {total}")
    print(f"Win rate: {win_rate:.1f}%")
    print(f"Avg profit (winning): {avg_profit:.2f}%")
    print(f"Avg loss (losing): {avg_loss:.2f}%")
    print(f"Max drawdown: {max_dd:.1f}%")
    print(f"Final equity ($1000 start): ${eq:.2f}")


if __name__ == "__main__":
    main()
