"""
ML Signal Scorer - Lightweight ML model for trading signal prediction
Uses scikit-learn RandomForest to predict 4-hour price direction
"""

import os
import json
import pickle
import time
import logging
import numpy as np
from typing import Optional, List, Tuple
from pathlib import Path

logger = logging.getLogger(__name__)

try:
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.preprocessing import StandardScaler
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    logger.warning("scikit-learn not available - ML scorer will be disabled")


class MLSignalScorer:
    """Random Forest classifier to predict 4-hour price direction."""

    def __init__(self, model_dir: str = "ml_models"):
        self.model_dir = model_dir
        self.model: Optional[RandomForestClassifier] = None
        self.scaler: Optional[StandardScaler] = None
        self.last_train_ts: float = 0.0
        self.feature_names = [
            "rsi_14", "ema_dist_20", "ema_dist_50", "volume_ratio",
            "atr_pct", "macd_hist", "bb_width", "day_of_week", "hour"
        ]
        self.model_path = os.path.join(model_dir, "model.pkl")
        self.scaler_path = os.path.join(model_dir, "scaler.pkl")
        self.meta_path = os.path.join(model_dir, "meta.json")

        if SKLEARN_AVAILABLE:
            Path(model_dir).mkdir(parents=True, exist_ok=True)
            self._load_model()

    def _load_model(self) -> None:
        """Load model from disk if it exists."""
        if not SKLEARN_AVAILABLE:
            return
        try:
            if os.path.exists(self.model_path):
                with open(self.model_path, "rb") as f:
                    self.model = pickle.load(f)
                logger.info("ML model loaded from disk")
            if os.path.exists(self.scaler_path):
                with open(self.scaler_path, "rb") as f:
                    self.scaler = pickle.load(f)
        except Exception as e:
            logger.warning(f"Failed to load ML model: {e}")

    def _save_model(self) -> None:
        """Save model and scaler to disk."""
        if not SKLEARN_AVAILABLE or not self.model or not self.scaler:
            return
        try:
            with open(self.model_path, "wb") as f:
                pickle.dump(self.model, f)
            with open(self.scaler_path, "wb") as f:
                pickle.dump(self.scaler, f)
            with open(self.meta_path, "w") as f:
                json.dump({"last_train_ts": self.last_train_ts}, f)
            logger.info("ML model saved to disk")
        except Exception as e:
            logger.warning(f"Failed to save ML model: {e}")

    def _calculate_rsi(self, closes: np.ndarray, period: int = 14) -> np.ndarray:
        """Calculate RSI indicator."""
        if len(closes) < period + 1:
            return np.full(len(closes), 50.0)
        delta = np.diff(closes)
        gain = np.where(delta > 0, delta, 0)
        loss = np.where(delta < 0, -delta, 0)
        avg_gain = np.convolve(gain, np.ones(period) / period, mode='valid')
        avg_loss = np.convolve(loss, np.ones(period) / period, mode='valid')
        rs = np.divide(avg_gain, avg_loss, where=avg_loss!=0, out=np.full_like(avg_gain, 50.0))
        rsi = 100 - (100 / (1 + rs))
        padding = np.full(len(closes) - len(rsi), 50.0)
        return np.concatenate([padding, rsi])

    def _calculate_ema(self, closes: np.ndarray, period: int) -> np.ndarray:
        """Calculate EMA."""
        ema = np.zeros_like(closes)
        ema[0] = closes[0]
        multiplier = 2.0 / (period + 1)
        for i in range(1, len(closes)):
            ema[i] = closes[i] * multiplier + ema[i-1] * (1 - multiplier)
        return ema

    def _calculate_macd(self, closes: np.ndarray) -> np.ndarray:
        """Calculate MACD histogram."""
        ema12 = self._calculate_ema(closes, 12)
        ema26 = self._calculate_ema(closes, 26)
        macd_line = ema12 - ema26
        signal = self._calculate_ema(macd_line, 9)
        return macd_line - signal

    def _calculate_atr(self, highs: np.ndarray, lows: np.ndarray, closes: np.ndarray, period: int = 14) -> np.ndarray:
        """Calculate ATR."""
        tr1 = highs - lows
        tr2 = np.abs(highs - np.roll(closes, 1))
        tr3 = np.abs(lows - np.roll(closes, 1))
        tr = np.maximum(tr1, np.maximum(tr2, tr3))
        atr = np.zeros_like(closes)
        atr[period-1] = np.mean(tr[:period])
        for i in range(period, len(closes)):
            atr[i] = (atr[i-1] * (period - 1) + tr[i]) / period
        return atr

    def _calculate_bb(self, closes: np.ndarray, period: int = 20, std_dev: float = 2.0) -> np.ndarray:
        """Calculate Bollinger Band width (upper - lower) / middle."""
        ma = np.convolve(closes, np.ones(period) / period, mode='valid')
        padding = np.full(len(closes) - len(ma), 0.0)
        ma = np.concatenate([padding, ma])
        std = np.zeros_like(closes)
        for i in range(period, len(closes)):
            std[i] = np.std(closes[i-period:i])
        upper = ma + std_dev * std
        lower = ma - std_dev * std
        bb_width = np.divide(upper - lower, ma, where=ma!=0, out=np.full_like(ma, 0.0))
        return bb_width

    def extract_features(self, candles: List[dict]) -> Optional[np.ndarray]:
        """Extract feature matrix from OHLCV candles.
        Each candle: {"open", "high", "low", "close", "volume", "time"}
        """
        if not SKLEARN_AVAILABLE or len(candles) < 50:
            return None

        try:
            closes = np.array([c.get("close", 0) for c in candles], dtype=np.float64)
            highs = np.array([c.get("high", 0) for c in candles], dtype=np.float64)
            lows = np.array([c.get("low", 0) for c in candles], dtype=np.float64)
            volumes = np.array([c.get("volume", 0) for c in candles], dtype=np.float64)

            if np.any(closes <= 0) or np.any(volumes <= 0):
                return None

            # Calculate indicators
            rsi = self._calculate_rsi(closes, 14)
            ema20 = self._calculate_ema(closes, 20)
            ema50 = self._calculate_ema(closes, 50)
            atr = self._calculate_atr(highs, lows, closes, 14)
            macd_hist = self._calculate_macd(closes)
            bb_width = self._calculate_bb(closes, 20)

            # Distance from EMAs (%)
            ema_dist_20 = (closes - ema20) / closes * 100
            ema_dist_50 = (closes - ema50) / closes * 100

            # Volume ratio (current / 20-period avg)
            vol_ma = np.convolve(volumes, np.ones(20) / 20, mode='same')
            vol_ma = np.maximum(vol_ma, 1e-9)
            volume_ratio = volumes / vol_ma

            # ATR as % of close
            atr_pct = (atr / closes) * 100

            # Day of week and hour (dummy values for historical data)
            day_of_week = np.zeros(len(candles))
            hour = np.zeros(len(candles))

            # Stack features into matrix (n_candles, n_features)
            features = np.column_stack([
                rsi, ema_dist_20, ema_dist_50, volume_ratio,
                atr_pct, macd_hist, bb_width, day_of_week, hour
            ])

            return features

        except Exception as e:
            logger.warning(f"Feature extraction failed: {e}")
            return None

    def train(self, candles: List[dict], lookahead: int = 4) -> bool:
        """Train model on historical data.
        Label = 1 if price higher in lookahead candles, 0 otherwise.
        """
        if not SKLEARN_AVAILABLE:
            return False

        try:
            features = self.extract_features(candles)
            if features is None or len(features) < lookahead + 10:
                logger.warning("Not enough data to train ML model")
                return False

            closes = np.array([c.get("close", 0) for c in candles], dtype=np.float64)

            # Create labels: 1 if price up in 4h, 0 if down
            labels = np.zeros(len(closes) - lookahead, dtype=int)
            for i in range(len(closes) - lookahead):
                future_close = closes[i + lookahead]
                current_close = closes[i]
                labels[i] = 1 if future_close > current_close else 0

            # Trim features to match labels
            X = features[:-lookahead]
            y = labels

            if len(np.unique(y)) < 2:
                logger.warning("Not enough class balance for training")
                return False

            # Normalize features
            self.scaler = StandardScaler()
            X = self.scaler.fit_transform(X)

            # Train RandomForest
            self.model = RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                random_state=42,
                n_jobs=-1
            )
            self.model.fit(X, y)
            self.last_train_ts = time.time()
            self._save_model()

            logger.info(f"ML model trained: {len(X)} samples, {self.model.n_features_in_} features")
            return True

        except Exception as e:
            logger.warning(f"ML training failed: {e}")
            return False

    def predict(self, candles: List[dict]) -> Optional[float]:
        """Return probability [0-1] that price will be higher in 4 hours."""
        if not SKLEARN_AVAILABLE or not self.model or not self.scaler:
            return None

        try:
            features = self.extract_features(candles)
            if features is None or len(features) == 0:
                return None

            # Use last candle
            X = features[-1:].reshape(1, -1)
            X = self.scaler.transform(X)

            # Predict probability of class 1 (price up)
            proba = self.model.predict_proba(X)[0][1]
            return float(proba)

        except Exception as e:
            logger.warning(f"ML prediction failed: {e}")
            return None

    def should_retrain(self, retrain_interval_hours: int = 168) -> bool:
        """Check if model needs retraining (default: weekly)."""
        if not SKLEARN_AVAILABLE or not self.model or self.last_train_ts == 0:
            return True
        elapsed_hours = (time.time() - self.last_train_ts) / 3600
        return elapsed_hours >= retrain_interval_hours
