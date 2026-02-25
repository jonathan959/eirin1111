"""
Machine Learning-based market regime detection (Phase 2 Advanced Intelligence).
Uses Random Forest + XGBoost ensemble to classify regimes with probability distribution.
"""
import os
import logging
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Regime mapping: spec uses STRONG_BULL/WEAK_BULL/RANGE/etc.
# Map strategies.legacy_regime -> ML class
REGIME_CLASSES = [
    "STRONG_BULL", "WEAK_BULL", "RANGE",
    "WEAK_BEAR", "STRONG_BEAR", "HIGH_VOL"
]
LEGACY_TO_ML = {
    "UPTREND": "STRONG_BULL",
    "TREND_UP": "STRONG_BULL",
    "BREAKOUT_UP": "STRONG_BULL",
    "RANGE": "RANGE",
    "RANGING": "RANGE",
    "DOWNTREND": "STRONG_BEAR",
    "TREND_DOWN": "STRONG_BEAR",
    "BREAKOUT_DOWN": "STRONG_BEAR",
    "HIGH_VOLATILITY": "HIGH_VOL",
    "HIGH_VOL_RISK": "HIGH_VOL",
    "RISK_OFF": "HIGH_VOL",
}


def _legacy_to_ml_regime(legacy: str) -> str:
    return LEGACY_TO_ML.get(str(legacy or "").upper(), "RANGE")


def _candles_to_df(candles: List[List[float]]) -> pd.DataFrame:
    """Convert [[ts, o, h, l, c, v], ...] to DataFrame with columns open, high, low, close, volume."""
    if not candles:
        return pd.DataFrame()
    arr = np.array(candles)
    if arr.ndim == 1 or len(arr) == 0:
        return pd.DataFrame()
    cols = min(6, arr.shape[1])
    df = pd.DataFrame(arr[:, :cols], columns=["ts", "open", "high", "low", "close", "volume"][:cols])
    for c in ["open", "high", "low", "close", "volume"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    if "volume" not in df.columns:
        df["volume"] = 0.0
    return df


def _ensure_df(candles: Any) -> pd.DataFrame:
    if isinstance(candles, pd.DataFrame):
        return candles
    return _candles_to_df(candles if isinstance(candles, list) else [])


class MLRegimeDetector:
    """
    ML-based regime detection with probability distributions.
    Trained on historical data to recognize regime patterns.
    """

    def __init__(self):
        self.model = None
        self.scaler = None
        self.regime_classes = REGIME_CLASSES
        self.feature_names: List[str] = []
        self._load_or_init()

    def _load_or_init(self) -> None:
        try:
            from sklearn.preprocessing import StandardScaler
            import joblib
            self.scaler = StandardScaler()
            model_path = os.path.join(os.path.dirname(__file__), "models", "ml_regime_detector.pkl")
            scaler_path = os.path.join(os.path.dirname(__file__), "models", "regime_scaler.pkl")
            if os.path.exists(model_path):
                self.model = joblib.load(model_path)
                if os.path.exists(scaler_path):
                    self.scaler = joblib.load(scaler_path)
                logger.info("ML Regime Detector loaded from disk")
                return
        except Exception as e:
            logger.debug("ML Regime Detector load failed: %s", e)
        self.model = None
        logger.debug("ML Regime Detector using fallback (no trained model)")

    def extract_features(self, candles: pd.DataFrame) -> np.ndarray:
        """
        Extract ML features from price data.
        Returns last valid feature vector as (n_features,) or empty array.
        """
        df = _ensure_df(candles)
        if df.empty or len(df) < 50:
            return np.array([]).reshape(0, 0)

        try:
            features: Dict[str, pd.Series] = {}

            # 1. Returns
            for period in [1, 3, 5, 10, 20]:
                features[f"return_{period}d"] = df["close"].pct_change(period)

            # 2. Volatility
            for period in [5, 10, 20]:
                features[f"volatility_{period}d"] = df["close"].pct_change().rolling(period).std()

            # 3. Trend (EMA)
            for period in [20, 50, 100, 200]:
                ema_ser = df["close"].ewm(span=period, adjust=False).mean()
                features[f"ema_{period}"] = ema_ser
                features[f"price_vs_ema_{period}"] = (df["close"] - ema_ser) / ema_ser.replace(0, np.nan)

            # 4. EMA ratios
            if "ema_20" in features and "ema_50" in features:
                features["ema_20_50_ratio"] = features["ema_20"] / features["ema_50"]
            if "ema_50" in features and "ema_200" in features:
                features["ema_50_200_ratio"] = features["ema_50"] / features["ema_200"]

            # 5. RSI
            delta = df["close"].diff()
            gain = delta.where(delta > 0, 0.0).rolling(14).mean()
            loss = (-delta.where(delta < 0, 0.0)).rolling(14).mean()
            rs = gain / loss.replace(0, np.nan)
            features["rsi_14"] = 100 - (100 / (1 + rs))

            # 6. ADX
            high = df["high"] if "high" in df.columns else df["close"]
            low = df["low"] if "low" in df.columns else df["close"]
            close = df["close"]
            plus_dm = high.diff().clip(lower=0)
            minus_dm = (-low.diff()).clip(lower=0)
            tr = pd.concat([
                high - low,
                (high - close.shift()).abs(),
                (low - close.shift()).abs(),
            ], axis=1).max(axis=1)
            atr = tr.rolling(14).mean()
            plus_di = 100 * (plus_dm.rolling(14).mean() / atr.replace(0, np.nan))
            minus_di = 100 * (minus_dm.rolling(14).mean() / atr.replace(0, np.nan))
            dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
            features["adx_14"] = dx.rolling(14).mean()

            # 7. Volume
            vol_sma = df["volume"].rolling(20).mean()
            features["volume_ratio"] = df["volume"] / vol_sma.replace(0, np.nan)

            # 8. Bollinger
            bb_mid = df["close"].rolling(20).mean()
            bb_std = df["close"].rolling(20).std()
            bb_upper = bb_mid + bb_std * 2
            bb_lower = bb_mid - bb_std * 2
            features["bb_width"] = (bb_upper - bb_lower) / bb_mid.replace(0, np.nan)
            features["bb_position"] = (df["close"] - bb_lower) / (bb_upper - bb_lower).replace(0, np.nan)

            # 9. ROC
            for period in [5, 10, 20]:
                shifted = df["close"].shift(period)
                features[f"roc_{period}"] = (df["close"] - shifted) / shifted.replace(0, np.nan) * 100

            feature_df = pd.DataFrame(features).dropna()
            if feature_df.empty or len(feature_df) < 5:
                return np.array([]).reshape(0, 0)

            self.feature_names = feature_df.columns.tolist()
            return feature_df.values[-1:]

        except Exception as e:
            logger.debug("extract_features failed: %s", e)
            return np.array([]).reshape(0, 0)

    def predict_regime(self, candles: Any) -> Dict[str, Any]:
        """
        Predict regime with probability distribution.
        Returns regime, confidence, probabilities, transition_probability.
        """
        if not self.model:
            return self._fallback_regime()

        features = self.extract_features(candles)
        if features.size == 0:
            return self._fallback_regime()

        try:
            from sklearn.ensemble import RandomForestClassifier

            X = features.reshape(1, -1)
            rf = self.model.get("rf") if isinstance(self.model, dict) else self.model
            if rf is None or not hasattr(rf, "predict_proba"):
                return self._fallback_regime()

            if self.scaler is not None:
                X = self.scaler.transform(X)

            proba = rf.predict_proba(X)[0]
            classes = getattr(rf, "classes_", np.arange(len(self.regime_classes)))
            # Map class indices to regime names
            class_names = [str(c) for c in classes]
            proba_map = dict(zip(class_names, proba.tolist()))
            # Fill missing with 0
            probabilities = {r: float(proba_map.get(r, 0.0)) for r in self.regime_classes}

            regime_idx = int(np.argmax(proba))
            regime = class_names[regime_idx] if regime_idx < len(class_names) else "RANGE"
            confidence = float(proba[regime_idx])
            sorted_probs = sorted(proba, reverse=True)
            transition_prob = sorted_probs[1] / sorted_probs[0] if sorted_probs[0] > 1e-9 else 0.5

            return {
                "regime": regime,
                "confidence": confidence,
                "probabilities": probabilities,
                "transition_probability": float(transition_prob),
                "feature_importance": self._get_top_features(rf),
            }
        except Exception as e:
            logger.debug("predict_regime failed: %s", e)
            return self._fallback_regime()

    def _get_top_features(self, rf: Any) -> List[Dict[str, Any]]:
        if not hasattr(rf, "feature_importances_") or not self.feature_names:
            return []
        imp = rf.feature_importances_
        top_idx = np.argsort(imp)[-5:][::-1]
        return [
            {"feature": self.feature_names[i], "importance": float(imp[i])}
            for i in top_idx if i < len(self.feature_names)
        ]

    def _fallback_regime(self) -> Dict[str, Any]:
        return {
            "regime": "RANGE",
            "confidence": 0.5,
            "probabilities": {r: 1.0 / len(self.regime_classes) for r in self.regime_classes},
            "transition_probability": 0.5,
            "feature_importance": [],
        }

    def train_model(self, historical_data: Dict[str, List[tuple]]) -> None:
        """
        Train on historical labeled data.
        historical_data: { 'symbol': [(candles_df_or_list, regime_label), ...] }
        regime_label can be legacy (UPTREND) or ML (STRONG_BULL).
        """
        try:
            from sklearn.ensemble import RandomForestClassifier
            try:
                from xgboost import XGBClassifier
            except ImportError:
                XGBClassifier = None
            from sklearn.preprocessing import StandardScaler
            import joblib

            X_all, y_all = [], []
            for symbol, data_points in historical_data.items():
                for candles, regime_label in data_points:
                    feats = self.extract_features(candles)
                    if feats.size > 0:
                        ml_regime = _legacy_to_ml_regime(regime_label)
                        if ml_regime not in self.regime_classes:
                            ml_regime = "RANGE"
                        X_all.append(feats[0])
                        y_all.append(ml_regime)

            if len(X_all) < 20:
                logger.warning("Insufficient training data (%d samples)", len(X_all))
                return

            X = np.array(X_all)
            y = np.array(y_all)
            self.scaler = StandardScaler()
            X_scaled = self.scaler.fit_transform(X)

            rf = RandomForestClassifier(n_estimators=200, max_depth=10, min_samples_split=20, random_state=42)
            rf.fit(X_scaled, y)
            acc_rf = rf.score(X_scaled, y)

            models = {"rf": rf}
            if XGBClassifier:
                xgb = XGBClassifier(n_estimators=200, max_depth=6, learning_rate=0.1, random_state=42)
                xgb.fit(X_scaled, y)
                models["xgb"] = xgb
                acc_xgb = xgb.score(X_scaled, y)
                logger.info("ML Regime trained - RF: %.2f%%, XGB: %.2f%%", acc_rf * 100, acc_xgb * 100)
            else:
                logger.info("ML Regime trained - RF: %.2f%%", acc_rf * 100)

            self.model = models
            os.makedirs(os.path.join(os.path.dirname(__file__), "models"), exist_ok=True)
            joblib.dump(models, os.path.join(os.path.dirname(__file__), "models", "ml_regime_detector.pkl"))
            if self.scaler:
                joblib.dump(self.scaler, os.path.join(os.path.dirname(__file__), "models", "regime_scaler.pkl"))
        except Exception as e:
            logger.exception("train_model failed: %s", e)


def train_regime_detector_on_historical_data() -> None:
    """
    Train regime detector using rule-based labels on historical data.
    Uses worker_api / bot_manager style clients if available.
    """
    from strategies import detect_regime

    historical_data: Dict[str, List[tuple]] = {}
    symbols = ["BTC/USD", "ETH/USD"]
    try:
        from worker_api import kc
        if kc:
            for sym in symbols:
                try:
                    ohlcv = kc.fetch_ohlcv(sym, timeframe="1d", limit=400)
                    if not ohlcv or len(ohlcv) < 80:
                        continue
                    labeled = []
                    for i in range(0, len(ohlcv) - 60, 30):
                        window = ohlcv[i : i + 60]
                        res = detect_regime(window)
                        regime = getattr(res, "legacy_regime", res.regime) or res.regime
                        labeled.append((window, regime))
                    if labeled:
                        historical_data[sym] = labeled
                except Exception as e:
                    logger.warning("Fetch %s failed: %s", sym, e)
    except ImportError:
        logger.warning("worker_api.kc not available for training")
        return

    if not historical_data:
        logger.warning("No historical data for training")
        return

    detector = MLRegimeDetector()
    detector.train_model(historical_data)
    logger.info("ML Regime Detector trained successfully")
