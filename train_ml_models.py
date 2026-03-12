#!/usr/bin/env python
"""
Train ML models (ml_ensemble, ml_predictor) using 2 years of OHLCV from yfinance.
Features: RSI, MACD, Bollinger Bands, EMA 20/50/200, volume ratio, ATR, momentum, sector, 52w high/low.
Predicts: higher in 5, 10, 30 days. 80/20 train/test. Saves to ml_models/.
"""
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
os.chdir(ROOT)
sys.path.insert(0, str(ROOT))

from env_utils import load_env
load_env()

import json
import logging
import time
from datetime import datetime, timezone, timedelta

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

TOP_30_CRYPTO = ["BTC-USD", "ETH-USD", "SOL-USD", "XRP-USD", "BNB-USD", "ADA-USD", "DOGE-USD", "AVAX-USD",
                 "LINK-USD", "DOT-USD", "MATIC-USD", "UNI-USD", "ATOM-USD", "LTC-USD", "NEAR-USD",
                 "APT-USD", "ARB-USD", "OP-USD", "INJ-USD", "FIL-USD", "IMX-USD", "HBAR-USD",
                 "VET-USD", "STX-USD", "SAND-USD", "RUNE-USD", "AAVE-USD", "MKR-USD", "GRT-USD"]
TOP_100_STOCKS = [
    "AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META", "TSLA", "BRK-B", "UNH", "JNJ",
    "JPM", "V", "PG", "MA", "HD", "CVX", "ABBV", "MRK", "PEP", "KO", "XOM", "COST",
    "WMT", "LLY", "AVGO", "MCD", "CSCO", "ACN", "ABT", "ADBE", "DHR", "NEE", "TMO",
    "NKE", "VZ", "PM", "CRM", "TXN", "CMCSA", "BMY", "UPS", "RTX", "QCOM", "HON",
    "AMGN", "LOW", "INTC", "BA", "UNP", "LMT", "AMD", "IBM", "GE", "CAT", "DE",
    "AMAT", "SBUX", "GILD", "ADI", "BKNG", "REGN", "MDT", "GS", "PLD", "LRCX",
    "BLK", "MMC", "SYK", "CI", "VRTX", "SO", "DUK", "MO", "ZTS", "BDX", "APD",
    "BSX", "EQIX", "EOG", "SLB", "ISRG", "KLAC", "WM", "ITW", "WM", "APTV",
]


def fetch_ohlcv_yf(symbol: str, days: int = 730) -> list:
    """Fetch daily OHLCV via yfinance. Returns [[ts, o, h, l, c, v], ...]"""
    try:
        import yfinance as yf
        sym = symbol.replace("/", "-").replace("XBT", "BTC")
        ticker = yf.Ticker(sym)
        end = datetime.now(timezone.utc)
        start = end - timedelta(days=days)
        df = ticker.history(start=start, end=end, interval="1d", auto_adjust=True)
        if df is None or df.empty or len(df) < 100:
            return []
        out = []
        for idx, row in df.iterrows():
            ts = int(idx.timestamp()) if hasattr(idx, "timestamp") else 0
            out.append([
                ts, float(row.get("Open", 0)), float(row.get("High", 0)),
                float(row.get("Low", 0)), float(row.get("Close", 0)), float(row.get("Volume", 0))
            ])
        return sorted(out, key=lambda x: x[0])
    except Exception as e:
        logger.debug("fetch %s: %s", symbol, e)
        return []


def engineer_features(candles: list, lookback: int = 50) -> dict:
    """Engineer RSI, MACD, BB, EMA, volume ratio, ATR, momentum, 52w high/low."""
    import numpy as np
    if not candles or len(candles) < lookback:
        return {}
    closes = np.array([c[4] for c in candles[-lookback:]])
    highs = np.array([c[2] for c in candles[-lookback:]])
    lows = np.array([c[3] for c in candles[-lookback:]])
    volumes = np.array([c[5] for c in candles[-lookback:]])

    def ema(series, span):
        return np.convolve(series, np.exp(np.linspace(-1, 0, span)) / np.sum(np.exp(np.linspace(-1, 0, span))), mode='valid')[-1] if len(series) >= span else series[-1]
    def rsi(series, period=14):
        d = np.diff(series)
        g = np.where(d > 0, d, 0)
        l = np.where(d < 0, -d, 0)
        rs = np.mean(g) / np.mean(l) if np.mean(l) > 0 else 100
        return 100 - 100 / (1 + rs)
    def atr(h, l, c, period=14):
        tr = np.maximum(h[1:] - l[1:], np.maximum(np.abs(h[1:] - c[:-1]), np.abs(l[1:] - c[:-1])))
        return np.mean(tr[-period:]) if len(tr) >= period else 0

    curr = closes[-1]
    ema20 = ema(closes, 20) if len(closes) >= 20 else curr
    ema50 = ema(closes, 50) if len(closes) >= 50 else curr
    ema200 = ema(closes, 200) if len(closes) >= 200 else curr
    rsi_val = rsi(closes) if len(closes) >= 15 else 50
    macd_12 = ema(closes, 12) if len(closes) >= 12 else curr
    macd_26 = ema(closes, 26) if len(closes) >= 26 else curr
    macd_val = macd_12 - macd_26
    bb_mid = np.mean(closes[-20:]) if len(closes) >= 20 else curr
    bb_std = np.std(closes[-20:]) if len(closes) >= 20 else 0
    bb_upper = bb_mid + 2 * bb_std
    bb_lower = bb_mid - 2 * bb_std
    bb_pos = (curr - bb_lower) / (bb_upper - bb_lower) if bb_upper > bb_lower else 0.5
    atr_val = atr(highs, lows, closes)
    atr_pct = atr_val / curr if curr > 0 else 0
    vol_ratio = volumes[-1] / np.mean(volumes[-20:]) if len(volumes) >= 20 and np.mean(volumes[-20:]) > 0 else 1
    mom_5 = (closes[-1] - closes[-6]) / closes[-6] if len(closes) >= 6 else 0
    high_52 = np.max(highs[-252:]) if len(highs) >= 252 else np.max(highs)
    low_52 = np.min(lows[-252:]) if len(lows) >= 252 else np.min(lows)
    pos_52w = (curr - low_52) / (high_52 - low_52) if high_52 > low_52 else 0.5

    return {
        "rsi": rsi_val, "macd": macd_val, "bb_position": bb_pos, "bb_width": (bb_upper - bb_lower) / curr if curr > 0 else 0,
        "ema20_dist": (curr - ema20) / ema20 if ema20 > 0 else 0, "ema50_dist": (curr - ema50) / ema50 if ema50 > 0 else 0,
        "ema200_dist": (curr - ema200) / ema200 if ema200 > 0 else 0,
        "volume_ratio": vol_ratio, "atr_pct": atr_pct, "momentum_5": mom_5,
        "position_52w": pos_52w, "ret_1": (closes[-1] - closes[-2]) / closes[-2] if len(closes) >= 2 else 0,
        "ret_5": (closes[-1] - closes[-6]) / closes[-6] if len(closes) >= 6 else 0,
        "ret_10": (closes[-1] - closes[-11]) / closes[-11] if len(closes) >= 11 else 0,
        "ret_20": (closes[-1] - closes[-21]) / closes[-21] if len(closes) >= 21 else 0,
    }


def build_training_dataset(symbols: list, days: int = 730, horizons: tuple = (5, 10, 30)):
    """Build labeled dataset for training."""
    X, y_5, y_10, y_30 = [], [], [], []
    for sym in symbols:
        candles = fetch_ohlcv_yf(sym, days)
        if len(candles) < 60:
            continue
        for i in range(50, len(candles) - max(horizons)):
            feat = engineer_features(candles[:i+1], 50)
            if not feat:
                continue
            vec = [
                feat.get("rsi", 50) / 100, feat.get("macd", 0) / (feat.get("atr_pct", 0.02) * feat.get("rsi", 50) / 100 + 1e-6),
                feat.get("bb_position", 0.5), feat.get("bb_width", 0.02), feat.get("ema20_dist", 0),
                feat.get("ema50_dist", 0), feat.get("ema200_dist", 0), feat.get("volume_ratio", 1),
                feat.get("atr_pct", 0.02), feat.get("momentum_5", 0), feat.get("position_52w", 0.5),
                feat.get("ret_1", 0), feat.get("ret_5", 0), feat.get("ret_10", 0), feat.get("ret_20", 0),
            ]
            X.append(vec)
            curr = candles[i][4]
            ret_5 = (candles[i + 5][4] - curr) / curr if i + 5 < len(candles) else 0
            ret_10 = (candles[i + 10][4] - curr) / curr if i + 10 < len(candles) else 0
            ret_30 = (candles[i + 30][4] - curr) / curr if i + 30 < len(candles) else 0
            y_5.append(1 if ret_5 > 0 else 0)
            y_10.append(1 if ret_10 > 0 else 0)
            y_30.append(1 if ret_30 > 0 else 0)
        logger.info("%s: %d samples", sym, len(X) - len([a for a in X if a == X[-1]]))
    return X, y_5, y_10, y_30


def train_and_save():
    """Train models and save to disk."""
    import numpy as np
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

    symbols = TOP_30_CRYPTO[:15] + TOP_100_STOCKS[:30]  # Subset for speed
    logger.info("Building dataset from %d symbols...", len(symbols))
    X, y_5, y_10, y_30 = build_training_dataset(symbols, days=730)
    if len(X) < 500:
        logger.error("Insufficient data: %d samples (need 500+)", len(X))
        return False

    X = np.array(X, dtype=np.float32)
    y_5, y_10, y_30 = np.array(y_5), np.array(y_10), np.array(y_30)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(np.nan_to_num(X, 0))

    models_dir = Path("./ml_models")
    models_dir.mkdir(exist_ok=True)

    for horizon, y in [("5d", y_5), ("10d", y_10), ("30d", y_30)]:
        X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
        clf = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, zero_division=0)
        rec = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        logger.info("Horizon %s: Accuracy=%.2f%% Precision=%.2f Recall=%.2f F1=%.2f", horizon, acc*100, prec*100, rec*100, f1*100)
        with open(models_dir / f"rf_{horizon}.pkl", "wb") as f:
            import pickle
            pickle.dump(clf, f)

    with open(models_dir / "scaler.pkl", "wb") as f:
        import pickle
        pickle.dump(scaler, f)
    with open(models_dir / "performance.json", "w") as f:
        json.dump({"trained_at": int(time.time()), "samples": len(X)}, f)
    logger.info("Models saved to %s", models_dir)
    return True


if __name__ == "__main__":
    success = train_and_save()
    sys.exit(0 if success else 1)
