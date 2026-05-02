"""
ML Signal Scorer - LightGBM model for trading signal prediction.
Predicts 12-candle price direction (up/down) with 32 engineered features.
Uses time-ordered 70/15/15 train/validation/test split — no data leakage.
"""

import os
import json
import pickle
import time
import logging
import datetime
import numpy as np
from typing import Optional, List, Tuple, Dict, Any
from pathlib import Path

import db

logger = logging.getLogger(__name__)

try:
    import lightgbm as lgb
    LGBM_AVAILABLE = True
except ImportError:
    LGBM_AVAILABLE = False
    logger.warning("LightGBM not available — falling back to RandomForest")

try:
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    logger.warning("scikit-learn not available - ML scorer disabled")


class MLSignalScorer:
    """
    LightGBM classifier: 32-feature price direction prediction.
    Falls back to RandomForest if LightGBM is unavailable.
    """

    FEATURE_NAMES = [
        # Price action (14)
        "rsi_14",           # RSI 14-period
        "rsi_delta",        # RSI now minus RSI 5 bars ago (momentum direction)
        "macd_hist",        # MACD histogram value
        "macd_delta",       # MACD histogram now minus 3 bars ago (acceleration)
        "bb_pct",           # Bollinger %B (0=lower, 0.5=mid, 1=upper)
        "bb_width",         # Bollinger Band width (squeeze detection)
        "ema_dist_20",      # Price vs 20 EMA % distance
        "ema_dist_50",      # Price vs 50 EMA % distance
        "ema_dist_200",     # Price vs 200 SMA % distance
        "slope5",           # Linear regression slope over 5 bars
        "slope20",          # Linear regression slope over 20 bars
        "roc10",            # Rate of change 10-period (%)
        "roc20",            # Rate of change 20-period (%)
        "atr_pct",          # ATR as % of price (normalized volatility)
        # Trend strength (4)
        "adx",              # ADX value (trend strength)
        "di_diff",          # +DI minus -DI (positive=bullish)
        "ema_cross",        # EMA 12/26 cross state (1=bullish, -1=bearish)
        "golden_cross",     # EMA 50 vs 200 (1=golden cross, -1=death cross)
        # Volume (4)
        "volume_ratio",     # Current volume / 20-bar volume MA
        "vol_trend",        # Volume slope over last 5 bars (increasing/decreasing)
        "vol_price_agree",  # Volume-price agreement (1=both up, -1=diverge, 0=neutral)
        "vol_rank",         # Volume rank in top/bottom 20% of last 50 bars (1/0/-1)
        # Volatility (3)
        "atr_ratio",        # Current ATR / 50-period ATR mean (expansion/contraction)
        "bb_width_pct",     # BB width percentile vs last 100 bars
        "hl_range_pct",     # High-low range as % of price
        # Pattern (4)
        "candle_bull",      # Bullish candlestick pattern (hammer, engulfing) 0/1
        "candle_bear",      # Bearish candlestick pattern (shooting star, engulfing) 0/1
        "higher_highs",     # 3-bar higher highs pattern (1/0)
        "lower_lows",       # 3-bar lower lows pattern (1/0)
        # Context (4)
        "hour_norm",        # Hour of day normalized 0-1
        "day_of_week_norm", # Day of week normalized 0-1
        "regime_code",      # Regime numeric (STRONG_BULL=2, WEAK_BULL=1, RANGE=0, WEAK_BEAR=-1, STRONG_BEAR=-2)
        "data_available",   # 1 if all features computed, 0 if any were imputed
    ]

    LGB_PARAMS = {
        "objective": "binary",
        "metric": "auc",
        "n_estimators": 500,
        "max_depth": 6,
        "learning_rate": 0.05,
        "num_leaves": 31,
        "min_child_samples": 20,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "reg_alpha": 0.1,
        "reg_lambda": 0.1,
        "is_unbalance": True,
        "verbose": -1,
        "random_state": 42,
    }

    def __init__(self, model_dir: str = "ml_models"):
        self.model_dir = model_dir
        self.model = None
        self.scaler: Optional[StandardScaler] = None
        self.last_train_ts: float = 0.0
        self.last_accuracy: float = 0.0
        self.last_auc: float = 0.0
        self.feature_names = self.FEATURE_NAMES  # alias for compatibility
        self.n_features = len(self.FEATURE_NAMES)
        self.model_path = os.path.join(model_dir, "lgbm_model.pkl")
        self.scaler_path = os.path.join(model_dir, "scaler.pkl")
        self.meta_path = os.path.join(model_dir, "meta.json")

        if SKLEARN_AVAILABLE or LGBM_AVAILABLE:
            Path(model_dir).mkdir(parents=True, exist_ok=True)
            self._load_model()

    # ─── DISK I/O ──────────────────────────────────────────────────────────

    def _load_model(self) -> None:
        try:
            if os.path.exists(self.model_path):
                with open(self.model_path, "rb") as f:
                    self.model = pickle.load(f)
                logger.info("ML model loaded from disk")
            if os.path.exists(self.scaler_path):
                with open(self.scaler_path, "rb") as f:
                    self.scaler = pickle.load(f)
            if os.path.exists(self.meta_path):
                with open(self.meta_path) as f:
                    m = json.load(f)
                    self.last_train_ts = m.get("last_train_ts", 0)
                    self.last_accuracy = m.get("accuracy", 0)
                    self.last_auc = m.get("auc", 0)
        except Exception as e:
            logger.warning(f"Failed to load ML model: {e}")

    def _save_model(self, meta: Dict[str, Any] = None) -> None:
        try:
            with open(self.model_path, "wb") as f:
                pickle.dump(self.model, f)
            if self.scaler:
                with open(self.scaler_path, "wb") as f:
                    pickle.dump(self.scaler, f)
            m = {"last_train_ts": self.last_train_ts,
                 "accuracy": self.last_accuracy, "auc": self.last_auc}
            if meta:
                m.update(meta)
            with open(self.meta_path, "w") as f:
                json.dump(m, f)
        except Exception as e:
            logger.warning(f"Failed to save ML model: {e}")

    # ─── INDICATOR HELPERS ──────────────────────────────────────────────────

    def _ema(self, arr: np.ndarray, period: int) -> np.ndarray:
        out = np.zeros_like(arr)
        out[0] = arr[0]
        k = 2.0 / (period + 1)
        for i in range(1, len(arr)):
            out[i] = arr[i] * k + out[i-1] * (1 - k)
        return out

    def _rsi(self, closes: np.ndarray, period: int = 14) -> np.ndarray:
        if len(closes) < period + 1:
            return np.full(len(closes), 50.0)
        delta = np.diff(closes, prepend=closes[0])
        gain = np.maximum(delta, 0)
        loss = np.maximum(-delta, 0)
        avg_gain = np.zeros_like(closes)
        avg_loss = np.zeros_like(closes)
        avg_gain[period] = np.mean(gain[1:period+1])
        avg_loss[period] = np.mean(loss[1:period+1])
        for i in range(period+1, len(closes)):
            avg_gain[i] = (avg_gain[i-1] * (period-1) + gain[i]) / period
            avg_loss[i] = (avg_loss[i-1] * (period-1) + loss[i]) / period
        rs = np.where(avg_loss > 0, avg_gain / avg_loss, 100.0)
        rsi = np.where(avg_loss > 0, 100 - 100/(1+rs), 100.0)
        rsi[:period] = 50.0
        return rsi

    def _macd_hist(self, closes: np.ndarray) -> np.ndarray:
        ema12 = self._ema(closes, 12)
        ema26 = self._ema(closes, 26)
        macd = ema12 - ema26
        signal = self._ema(macd, 9)
        return macd - signal

    def _atr(self, H: np.ndarray, L: np.ndarray, C: np.ndarray, period: int = 14) -> np.ndarray:
        tr1 = H - L
        tr2 = np.abs(H - np.roll(C, 1)); tr2[0] = tr1[0]
        tr3 = np.abs(L - np.roll(C, 1)); tr3[0] = tr1[0]
        tr = np.maximum(tr1, np.maximum(tr2, tr3))
        atr = np.zeros_like(C)
        atr[period-1] = np.mean(tr[:period])
        for i in range(period, len(C)):
            atr[i] = (atr[i-1] * (period-1) + tr[i]) / period
        return atr

    def _adx(self, H: np.ndarray, L: np.ndarray, C: np.ndarray, period: int = 14) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Returns (adx, plus_di, minus_di) arrays."""
        n = len(C)
        atr_arr = self._atr(H, L, C, period)
        plus_dm = np.zeros(n)
        minus_dm = np.zeros(n)
        for i in range(1, n):
            up = H[i] - H[i-1]
            dn = L[i-1] - L[i]
            plus_dm[i] = up if up > dn and up > 0 else 0
            minus_dm[i] = dn if dn > up and dn > 0 else 0
        # Wilder smooth
        sm_plus = np.zeros(n); sm_minus = np.zeros(n)
        sm_plus[period] = np.sum(plus_dm[1:period+1])
        sm_minus[period] = np.sum(minus_dm[1:period+1])
        sm_atr = np.zeros(n)
        sm_atr[period] = np.sum(atr_arr[1:period+1])
        for i in range(period+1, n):
            sm_plus[i] = sm_plus[i-1] - sm_plus[i-1]/period + plus_dm[i]
            sm_minus[i] = sm_minus[i-1] - sm_minus[i-1]/period + minus_dm[i]
            sm_atr[i] = sm_atr[i-1] - sm_atr[i-1]/period + atr_arr[i]
        safe_atr = np.where(sm_atr > 0, sm_atr, 1)
        plus_di = 100 * sm_plus / safe_atr
        minus_di = 100 * sm_minus / safe_atr
        dx = np.where((plus_di + minus_di) > 0, 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di), 0)
        adx = np.zeros(n)
        if n > 2*period:
            adx[2*period] = np.mean(dx[period:2*period+1])
            for i in range(2*period+1, n):
                adx[i] = (adx[i-1] * (period-1) + dx[i]) / period
        return adx, plus_di, minus_di

    def _bb(self, closes: np.ndarray, period: int = 20, mult: float = 2.0):
        """Returns (upper, middle, lower, width, pct_b) arrays."""
        n = len(closes)
        mid = np.zeros(n); std = np.zeros(n)
        for i in range(period-1, n):
            window = closes[i-period+1:i+1]
            mid[i] = window.mean()
            std[i] = window.std(ddof=1)
        mid[:period-1] = closes[:period-1]
        upper = mid + mult * std
        lower = mid - mult * std
        width = np.where(mid > 0, (upper - lower) / mid, 0)
        pct_b = np.where((upper - lower) > 0, (closes - lower) / (upper - lower), 0.5)
        return upper, mid, lower, width, pct_b

    def _slope(self, closes: np.ndarray, period: int) -> np.ndarray:
        """Normalized linear regression slope."""
        out = np.zeros(len(closes))
        x = np.arange(period, dtype=np.float64)
        xm = x.mean(); xv = ((x-xm)**2).sum()
        if xv == 0:
            return out
        for i in range(period, len(closes)):
            y = closes[i-period:i]
            ym = y.mean()
            out[i] = ((x-xm)*(y-ym)).sum() / xv / (ym + 1e-9)
        return out

    def _vol_rank(self, volumes: np.ndarray, period: int = 50) -> np.ndarray:
        """1 = top 20%, -1 = bottom 20%, 0 = middle for each bar."""
        out = np.zeros(len(volumes))
        for i in range(period, len(volumes)):
            window = volumes[i-period:i]
            v = volumes[i]
            pct = (window < v).sum() / period
            out[i] = 1 if pct >= 0.8 else (-1 if pct <= 0.2 else 0)
        return out

    def _candle_patterns(self, O: np.ndarray, H: np.ndarray, L: np.ndarray, C: np.ndarray):
        """Returns (bull_pattern, bear_pattern) 0/1 arrays."""
        n = len(C)
        bull = np.zeros(n); bear = np.zeros(n)
        for i in range(1, n):
            body = abs(C[i] - O[i])
            rng = H[i] - L[i]
            if rng < 1e-9:
                continue
            lower_wick = min(O[i], C[i]) - L[i]
            upper_wick = H[i] - max(O[i], C[i])
            # Hammer: small body, long lower wick, bullish context
            if lower_wick > 2 * body and body / rng < 0.3:
                bull[i] = 1
            # Bullish engulfing
            if C[i] > O[i] and C[i-1] < O[i-1] and O[i] < C[i-1] and C[i] > O[i-1]:
                bull[i] = 1
            # Shooting star: small body, long upper wick
            if upper_wick > 2 * body and body / rng < 0.3 and C[i] < O[i]:
                bear[i] = 1
            # Bearish engulfing
            if C[i] < O[i] and C[i-1] > O[i-1] and O[i] > C[i-1] and C[i] < O[i-1]:
                bear[i] = 1
        return bull, bear

    def _regime_code(self, regime: str) -> float:
        mapping = {
            "STRONG_BULL": 2.0, "BREAKOUT_UP": 2.0,
            "WEAK_BULL": 1.0, "TREND_UP": 1.5,
            "RANGE": 0.0, "RANGING": 0.0,
            "WEAK_BEAR": -1.0, "TREND_DOWN": -1.5,
            "STRONG_BEAR": -2.0, "BREAKOUT_DOWN": -2.0,
            "HIGH_VOL_RISK": -0.5,
        }
        return mapping.get((regime or "").upper(), 0.0)

    # ─── FEATURE EXTRACTION ─────────────────────────────────────────────────

    def extract_features(self, candles: List[dict], regime: str = "") -> Optional[np.ndarray]:
        """
        Extract 32-feature matrix from OHLCV candles.
        Each candle: {"open","high","low","close","volume","time"}.
        Returns shape (n_candles, 32).
        """
        if len(candles) < 60:
            return None
        try:
            C = np.array([c.get("close", 0) for c in candles], dtype=np.float64)
            O = np.array([c.get("open", C[i]) for i, c in enumerate(candles)], dtype=np.float64)
            H = np.array([c.get("high", C[i]) for i, c in enumerate(candles)], dtype=np.float64)
            L = np.array([c.get("low", C[i]) for i, c in enumerate(candles)], dtype=np.float64)
            V = np.array([c.get("volume", 1) for c in candles], dtype=np.float64)

            if np.any(C <= 0):
                return None

            n = len(C)
            data_ok = 1.0  # will be set to 0 if any feature fails

            # RSI
            rsi = self._rsi(C, 14)
            rsi_delta = np.zeros(n); rsi_delta[5:] = rsi[5:] - rsi[:-5]

            # MACD
            mh = self._macd_hist(C)
            macd_delta = np.zeros(n); macd_delta[3:] = mh[3:] - mh[:-3]

            # Bollinger
            _, mid, _, bb_width, bb_pct = self._bb(C, 20)

            # EMAs
            ema20 = self._ema(C, 20)
            ema50 = self._ema(C, 50)
            sma200 = np.zeros(n)
            for i in range(199, n):
                sma200[i] = C[i-199:i+1].mean()
            sma200[:199] = C[:199]

            ema_dist_20 = (C - ema20) / (C + 1e-9) * 100
            ema_dist_50 = (C - ema50) / (C + 1e-9) * 100
            ema_dist_200 = (C - sma200) / (C + 1e-9) * 100

            # Slopes
            slope5  = self._slope(C, 5)
            slope20 = self._slope(C, 20)

            # ROC
            roc10 = np.zeros(n); roc10[10:] = (C[10:]-C[:-10])/(C[:-10]+1e-9)*100
            roc20 = np.zeros(n); roc20[20:] = (C[20:]-C[:-20])/(C[:-20]+1e-9)*100

            # ATR
            atr = self._atr(H, L, C, 14)
            atr_pct = atr / (C + 1e-9) * 100

            # ADX
            try:
                adx_arr, plus_di, minus_di = self._adx(H, L, C, 14)
                di_diff = plus_di - minus_di
            except Exception:
                adx_arr = np.zeros(n); di_diff = np.zeros(n); data_ok = 0.0

            # EMA crosses
            ema12 = self._ema(C, 12); ema26 = self._ema(C, 26)
            ema_cross = np.sign(ema12 - ema26)
            ema50_arr = self._ema(C, 50); ema200_arr = self._ema(C, 200)
            golden_cross = np.sign(ema50_arr - ema200_arr)

            # Volume
            vol_ma = np.convolve(V, np.ones(20)/20, mode='same')
            vol_ma = np.maximum(vol_ma, 1e-9)
            volume_ratio = V / vol_ma

            vol_trend = np.zeros(n)
            for i in range(5, n):
                vol_trend[i] = np.polyfit(np.arange(5), V[i-5:i], 1)[0] / (V[i-5:i].mean()+1e-9)

            price_up = np.zeros(n); price_up[1:] = (C[1:] > C[:-1]).astype(float)
            vol_price_agree = np.where(
                (volume_ratio > 1.0) & (price_up > 0), 1.0,
                np.where((volume_ratio > 1.0) & (price_up == 0), -1.0, 0.0)
            )
            vol_rank = self._vol_rank(V, 50)

            # ATR ratio
            atr_mean50 = np.convolve(atr_pct, np.ones(50)/50, mode='same')
            atr_ratio = atr_pct / (atr_mean50 + 1e-9)

            # BB width percentile
            bb_width_pct = np.zeros(n)
            for i in range(100, n):
                w_window = bb_width[i-100:i]
                bb_width_pct[i] = (w_window < bb_width[i]).sum() / 100.0

            # High-low range %
            hl_range_pct = (H - L) / (C + 1e-9) * 100

            # Candle patterns
            candle_bull, candle_bear = self._candle_patterns(O, H, L, C)

            # Higher highs / lower lows (3-bar)
            higher_highs = np.zeros(n)
            lower_lows = np.zeros(n)
            for i in range(2, n):
                higher_highs[i] = 1.0 if H[i] > H[i-1] > H[i-2] else 0.0
                lower_lows[i]   = 1.0 if L[i] < L[i-1] < L[i-2] else 0.0

            # Context: hour and day from timestamps
            hour_norm = np.zeros(n); dow_norm = np.zeros(n)
            for j, c in enumerate(candles):
                ts = c.get("time") or c.get("ts") or 0
                if ts:
                    try:
                        ft = float(ts)
                        dt = datetime.datetime.utcfromtimestamp(ft/1000 if ft > 1e10 else ft)
                        hour_norm[j] = dt.hour / 23.0
                        dow_norm[j]  = dt.weekday() / 6.0
                    except Exception:
                        pass

            regime_code_val = self._regime_code(regime)
            regime_arr = np.full(n, regime_code_val)
            data_available = np.full(n, data_ok)

            features = np.column_stack([
                rsi, rsi_delta, mh, macd_delta, bb_pct, bb_width,
                ema_dist_20, ema_dist_50, ema_dist_200,
                slope5, slope20, roc10, roc20, atr_pct,
                adx_arr, di_diff, ema_cross, golden_cross,
                volume_ratio, vol_trend, vol_price_agree, vol_rank,
                atr_ratio, bb_width_pct, hl_range_pct,
                candle_bull, candle_bear, higher_highs, lower_lows,
                hour_norm, dow_norm, regime_arr, data_available,
            ])

            # Replace NaN/Inf with 0
            features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)
            return features

        except Exception as e:
            logger.warning(f"Feature extraction failed: {e}", exc_info=True)
            return None

    # ─── TRAINING ───────────────────────────────────────────────────────────

    def train(self, candles: List[dict], lookahead: int = 12, threshold_pct: float = 0.005,
              regime: str = "") -> bool:
        """
        Train LightGBM on historical candle data.

        Labels: price UP >= threshold in next `lookahead` candles = 1,
                price DOWN >= threshold = 0, sideways = excluded.
        Split:  70% train / 15% validation / 15% test (time-ordered, no shuffle).
        Threshold: 0.5% for crypto (default), 0.3% for stocks.
        """
        if not (SKLEARN_AVAILABLE or LGBM_AVAILABLE):
            return False
        try:
            features = self.extract_features(candles, regime)
            if features is None or len(features) < lookahead + 60:
                logger.warning("Not enough data to train ML model")
                return False

            closes = np.array([c.get("close", 0) for c in candles], dtype=np.float64)

            # Label generation — future-only, no lookahead contamination
            labels, valid_idx = [], []
            for i in range(len(closes) - lookahead):
                future = closes[i+1:i+lookahead+1]
                cur = closes[i]
                if cur <= 0:
                    continue
                up = (future.max() - cur) / cur
                dn = (cur - future.min()) / cur
                if up >= threshold_pct and up > dn:
                    labels.append(1); valid_idx.append(i)
                elif dn >= threshold_pct and dn > up:
                    labels.append(0); valid_idx.append(i)
                # sideways: skip

            if len(valid_idx) < 60:
                logger.warning(f"Only {len(valid_idx)} directional samples — need 60+")
                return False

            X = features[valid_idx]
            y = np.array(labels)

            if len(np.unique(y)) < 2:
                return False

            # 70 / 15 / 15 time-ordered split
            n = len(X)
            t1 = int(n * 0.70); t2 = int(n * 0.85)
            X_train, X_val, X_test = X[:t1], X[t1:t2], X[t2:]
            y_train, y_val, y_test = y[:t1], y[t1:t2], y[t2:]

            if len(X_train) < 30 or len(X_test) < 10:
                logger.warning(f"Insufficient split sizes: train={len(X_train)} test={len(X_test)}")
                return False

            # Normalize on training data only
            self.scaler = StandardScaler()
            X_train_s = self.scaler.fit_transform(X_train)
            X_val_s   = self.scaler.transform(X_val)
            X_test_s  = self.scaler.transform(X_test)

            if LGBM_AVAILABLE:
                params = dict(self.LGB_PARAMS)
                params["n_estimators"] = 1000  # will early-stop
                self.model = lgb.LGBMClassifier(**params)
                self.model.fit(
                    X_train_s, y_train,
                    eval_set=[(X_val_s, y_val)],
                    callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(-1)],
                )
            else:
                # Fallback to RandomForest
                from sklearn.ensemble import RandomForestClassifier
                self.model = RandomForestClassifier(
                    n_estimators=200, max_depth=8, min_samples_leaf=5,
                    class_weight="balanced", random_state=42, n_jobs=-1
                )
                self.model.fit(X_train_s, y_train)

            # Evaluate on test set
            y_pred = self.model.predict(X_test_s)
            y_prob = self.model.predict_proba(X_test_s)[:, 1]
            accuracy  = float(np.mean(y_pred == y_test))
            auc       = float(roc_auc_score(y_test, y_prob)) if len(np.unique(y_test)) > 1 else 0.5
            precision = float(precision_score(y_test, y_pred, zero_division=0))
            recall    = float(recall_score(y_test, y_pred, zero_division=0))
            f1        = float(f1_score(y_test, y_pred, zero_division=0))
            up_rate   = float(np.mean(y_test == 1))

            self.last_train_ts = time.time()
            self.last_accuracy = accuracy
            self.last_auc = auc

            # Feature importances
            feat_imp = []
            try:
                if hasattr(self.model, "feature_importances_"):
                    imp = sorted(zip(self.FEATURE_NAMES, self.model.feature_importances_), key=lambda x: -x[1])
                    feat_imp = [(n, round(float(v), 4)) for n, v in imp[:10]]
            except Exception:
                pass

            meta = {
                "accuracy": accuracy, "auc": auc, "precision": precision,
                "recall": recall, "f1": f1, "up_rate": up_rate,
                "n_train": len(X_train), "n_test": len(X_test),
                "n_features": X.shape[1], "top_features": feat_imp,
                "model_type": "LightGBM" if LGBM_AVAILABLE else "RandomForest",
                "trained_at": datetime.datetime.utcnow().isoformat(),
            }
            self._save_model(meta)
            self._log_version_to_db(meta)

            logger.info(
                f"ML trained ({meta['model_type']}): "
                f"train={len(X_train)} val={len(X_val)} test={len(X_test)} | "
                f"acc={accuracy*100:.1f}% auc={auc:.3f} prec={precision*100:.1f}% "
                f"rec={recall*100:.1f}% f1={f1:.3f} | up_rate={up_rate*100:.1f}%"
            )
            if feat_imp:
                logger.info(f"ML top features: {feat_imp[:5]}")
            if accuracy < 0.53:
                logger.warning(f"ML accuracy {accuracy*100:.1f}% < 53% — using as soft signal only")

            return True

        except Exception as e:
            logger.warning(f"ML training failed: {e}", exc_info=True)
            return False

    def _log_version_to_db(self, meta: dict) -> None:
        """Record training event in ml_model_versions table.

        Phase 1.2d: previous implementation imported a non-existent ``db.get_db``
        symbol and was silently swallowed by ``except Exception``. It now calls
        ``db.save_ml_model_version`` which routes through ``write_txn(None, ...)``
        and is logged on failure (no silent no-op — brief rule #4).
        """
        try:
            db.save_ml_model_version(
                model_type=meta.get("model_type", "LightGBM"),
                version=datetime.datetime.utcnow().strftime("%Y%m%d_%H%M%S"),
                validation_accuracy=round(float(meta.get("auc", meta.get("accuracy", 0)) or 0), 4),
                deployed=True,
            )
        except Exception:
            logger.exception("ml_signal_scorer: failed to log model version to db")

    # ─── PREDICTION ─────────────────────────────────────────────────────────

    def predict(self, candles: List[dict], regime: str = "") -> Optional[float]:
        """
        Return probability [0-1] that price will be higher in 12 candles.
        > 0.6: bullish signal (proceed)
        0.4-0.6: uncertain (reduce position 30%)
        < 0.4: bearish (block long entry)
        """
        if not self.model or not self.scaler:
            return None
        try:
            features = self.extract_features(candles, regime)
            if features is None:
                return None
            X = features[-1:].reshape(1, -1)
            X = self.scaler.transform(X)
            proba = self.model.predict_proba(X)[0][1]
            return float(proba)
        except Exception as e:
            logger.warning(f"ML prediction failed: {e}")
            return None

    def should_retrain(self, retrain_interval_hours: int = 168, new_samples: int = 0,
                       sample_threshold: int = 500) -> bool:
        """Retrain if 7 days elapsed OR 500 new samples accumulated."""
        if not self.model or self.last_train_ts == 0:
            return True
        elapsed_h = (time.time() - self.last_train_ts) / 3600
        return elapsed_h >= retrain_interval_hours or new_samples >= sample_threshold

    def get_performance_report(self) -> Dict[str, Any]:
        """Return last training metrics for API exposure."""
        if not os.path.exists(self.meta_path):
            return {"trained": False}
        try:
            with open(self.meta_path) as f:
                m = json.load(f)
            m["trained"] = True
            m["model_available"] = self.model is not None
            return m
        except Exception:
            return {"trained": False}
