"""
ML Ensemble - Enhanced Machine Learning for Trading Decisions.

Features:
- Ensemble of LSTM, XGBoost, and Random Forest
- Weighted voting based on recent performance
- Volatility-adaptive confidence thresholds
- Regular retraining support
- Integration with strategy system

Based on: "Upgrading to an Autonomous Multi-Asset Trading Bot" specification.
"""

import os
import time
import json
import math
import logging
import threading
import pickle
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any
from collections import deque
from pathlib import Path

logger = logging.getLogger(__name__)

# Optional ML library imports
try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False
    logger.warning("numpy not available")

try:
    from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import train_test_split
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    logger.warning("scikit-learn not available")

try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    logger.warning("xgboost not available")

try:
    import tensorflow as tf
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense, Dropout
    TENSORFLOW_AVAILABLE = True
except ImportError:
    TENSORFLOW_AVAILABLE = False
    logger.warning("tensorflow not available - LSTM disabled")


@dataclass
class EnsemblePrediction:
    """Result from ensemble prediction."""
    direction: str  # "UP", "DOWN", "NEUTRAL"
    confidence: float  # 0-1
    probability_up: float
    probability_down: float
    
    # Individual model predictions
    model_predictions: Dict[str, Dict[str, float]]
    
    # Metadata
    volatility_regime: str  # "low", "normal", "high"
    threshold_used: float
    features_count: int
    timestamp: int = field(default_factory=lambda: int(time.time()))
    
    def should_trade(self, min_confidence: float = 0.55) -> bool:
        """Check if prediction is strong enough to trade."""
        return self.direction != "NEUTRAL" and self.confidence >= min_confidence


@dataclass
class ModelPerformance:
    """Track individual model performance."""
    correct_predictions: int = 0
    total_predictions: int = 0
    recent_accuracy: float = 0.5
    weight: float = 1.0
    last_update: int = 0


class MLEnsemble:
    """
    Ensemble ML system for trading signal validation.
    
    Combines multiple models and requires agreement before signaling.
    Adapts confidence thresholds based on market volatility.
    """
    
    def __init__(self, 
                 models_dir: str = "./ml_models",
                 min_ensemble_confidence: float = 0.55,
                 retrain_interval_hours: int = 24):
        
        self.models_dir = Path(models_dir)
        self.models_dir.mkdir(exist_ok=True)
        
        self.min_confidence = min_ensemble_confidence
        self.retrain_interval = retrain_interval_hours * 3600
        
        # Models
        self._lstm_model = None
        self._xgb_model = None
        self._rf_model = None
        self._scaler = StandardScaler() if SKLEARN_AVAILABLE else None
        
        # Model performance tracking
        self._model_performance = {
            "lstm": ModelPerformance(),
            "xgb": ModelPerformance(),
            "rf": ModelPerformance()
        }
        
        # Training data storage
        self._training_data: deque = deque(maxlen=10000)
        self._prediction_history: deque = deque(maxlen=1000)
        
        # State
        self._last_retrain = 0
        self._is_trained = False
        self._lock = threading.Lock()
        
        # Volatility-based thresholds
        self._volatility_thresholds = {
            "low": 0.52,     # Lower threshold in calm markets
            "normal": 0.55,  # Standard threshold
            "high": 0.62     # Higher threshold in volatile markets
        }
        
        # Try to load existing models
        self._load_models()
        
        logger.info(f"MLEnsemble initialized (LSTM: {TENSORFLOW_AVAILABLE}, XGB: {XGBOOST_AVAILABLE}, RF: {SKLEARN_AVAILABLE})")

    FEATURE_NAMES = [
        "ret_1", "ret_5", "ret_10", "ret_20", "position_in_range",
        "sma5_dist", "sma10_dist", "sma20_dist", "sma50_dist", "ema_spread",
        "rsi", "rsi_oversold", "rsi_overbought", "macd", "stochastic",
        "atr_pct", "bb_width", "bb_position", "hist_vol",
        "vol_ratio", "vol_trend", "hh_count", "ll_count", "adx_like",
    ]
    
    def extract_features(self, 
                         candles: List[List[float]], 
                         lookback: int = 50) -> Optional[List[float]]:
        """
        Extract comprehensive features for ML prediction.
        
        Features include:
        - Price action (returns, ranges)
        - Technical indicators (RSI, MACD, BBands)
        - Volume patterns
        - Trend indicators
        - Volatility measures
        """
        if not NUMPY_AVAILABLE or not candles or len(candles) < lookback:
            return None
        
        try:
            # Extract OHLCV
            opens = np.array([c[1] for c in candles[-lookback:]])
            highs = np.array([c[2] for c in candles[-lookback:]])
            lows = np.array([c[3] for c in candles[-lookback:]])
            closes = np.array([c[4] for c in candles[-lookback:]])
            volumes = np.array([c[5] for c in candles[-lookback:]])
            
            features = []
            
            # === Price Features ===
            
            # Returns (various periods)
            for period in [1, 5, 10, 20]:
                if len(closes) > period:
                    ret = (closes[-1] - closes[-period-1]) / closes[-period-1]
                    features.append(ret)
                else:
                    features.append(0.0)
            
            # Price position in range
            high_20 = np.max(highs[-20:]) if len(highs) >= 20 else highs[-1]
            low_20 = np.min(lows[-20:]) if len(lows) >= 20 else lows[-1]
            range_20 = high_20 - low_20
            if range_20 > 0:
                position_in_range = (closes[-1] - low_20) / range_20
            else:
                position_in_range = 0.5
            features.append(position_in_range)
            
            # === Moving Averages ===
            
            for period in [5, 10, 20, 50]:
                if len(closes) >= period:
                    sma = np.mean(closes[-period:])
                    features.append((closes[-1] - sma) / sma)  # Distance from SMA
                else:
                    features.append(0.0)
            
            # EMA crossovers
            if len(closes) >= 21:
                ema_9 = self._ema(closes, 9)
                ema_21 = self._ema(closes, 21)
                features.append((ema_9 - ema_21) / closes[-1])  # EMA spread
            else:
                features.append(0.0)
            
            # === Momentum Indicators ===
            
            # RSI
            rsi = self._rsi(closes, 14)
            features.append(rsi / 100)  # Normalize to 0-1
            
            # RSI zones
            features.append(1.0 if rsi < 30 else 0.0)  # Oversold
            features.append(1.0 if rsi > 70 else 0.0)  # Overbought
            
            # MACD
            if len(closes) >= 26:
                ema_12 = self._ema(closes, 12)
                ema_26 = self._ema(closes, 26)
                macd = ema_12 - ema_26
                features.append(macd / closes[-1])  # Normalized MACD
            else:
                features.append(0.0)
            
            # Stochastic
            if len(closes) >= 14:
                lowest_low = np.min(lows[-14:])
                highest_high = np.max(highs[-14:])
                if highest_high > lowest_low:
                    stoch_k = (closes[-1] - lowest_low) / (highest_high - lowest_low)
                else:
                    stoch_k = 0.5
                features.append(stoch_k)
            else:
                features.append(0.5)
            
            # === Volatility ===
            
            # ATR
            atr = self._atr(highs, lows, closes, 14)
            atr_pct = atr / closes[-1] if closes[-1] > 0 else 0
            features.append(atr_pct)
            
            # Bollinger Band width
            if len(closes) >= 20:
                sma_20 = np.mean(closes[-20:])
                std_20 = np.std(closes[-20:])
                bb_width = (2 * std_20 * 2) / sma_20 if sma_20 > 0 else 0
                features.append(bb_width)
                
                # Position relative to bands
                bb_upper = sma_20 + 2 * std_20
                bb_lower = sma_20 - 2 * std_20
                bb_position = (closes[-1] - bb_lower) / (bb_upper - bb_lower) if bb_upper > bb_lower else 0.5
                features.append(bb_position)
            else:
                features.extend([0.0, 0.5])
            
            # Historical volatility
            if len(closes) >= 20:
                returns = np.diff(closes[-21:]) / closes[-21:-1]
                hist_vol = np.std(returns) * np.sqrt(252)  # Annualized
                features.append(hist_vol)
            else:
                features.append(0.0)
            
            # === Volume Features ===
            
            if len(volumes) >= 20:
                vol_sma = np.mean(volumes[-20:])
                vol_ratio = volumes[-1] / vol_sma if vol_sma > 0 else 1.0
                features.append(vol_ratio)
                
                # Volume trend
                vol_change = (volumes[-1] - volumes[-5]) / volumes[-5] if len(volumes) >= 5 and volumes[-5] > 0 else 0
                features.append(vol_change)
            else:
                features.extend([1.0, 0.0])
            
            # === Trend Features ===
            
            # Higher highs / lower lows
            if len(highs) >= 5:
                hh_count = sum(1 for i in range(1, 5) if highs[-i] > highs[-i-1])
                ll_count = sum(1 for i in range(1, 5) if lows[-i] < lows[-i-1])
                features.append(hh_count / 4)  # Normalize
                features.append(ll_count / 4)
            else:
                features.extend([0.0, 0.0])
            
            # ADX-like trend strength (simplified)
            if len(closes) >= 14:
                ups = []
                downs = []
                for i in range(1, min(14, len(closes))):
                    up = highs[-i] - highs[-i-1] if highs[-i] > highs[-i-1] else 0
                    down = lows[-i-1] - lows[-i] if lows[-i] < lows[-i-1] else 0
                    ups.append(up)
                    downs.append(down)
                
                avg_up = np.mean(ups) if ups else 0
                avg_down = np.mean(downs) if downs else 0
                
                if avg_up + avg_down > 0:
                    di_diff = abs(avg_up - avg_down) / (avg_up + avg_down)
                else:
                    di_diff = 0
                features.append(di_diff)
            else:
                features.append(0.0)
            
            return features
            
        except Exception as e:
            logger.error(f"Feature extraction failed: {e}")
            return None
    
    def _ema(self, prices: np.ndarray, period: int) -> float:
        """Calculate EMA."""
        if len(prices) < period:
            return prices[-1]
        multiplier = 2 / (period + 1)
        ema = np.mean(prices[:period])
        for price in prices[period:]:
            ema = (price - ema) * multiplier + ema
        return ema
    
    def _rsi(self, prices: np.ndarray, period: int = 14) -> float:
        """Calculate RSI."""
        if len(prices) < period + 1:
            return 50.0
        
        deltas = np.diff(prices)
        gains = np.where(deltas > 0, deltas, 0)
        losses = np.where(deltas < 0, -deltas, 0)
        
        avg_gain = np.mean(gains[-period:])
        avg_loss = np.mean(losses[-period:])
        
        if avg_loss == 0:
            return 100.0
        
        rs = avg_gain / avg_loss
        return 100 - (100 / (1 + rs))
    
    def _atr(self, highs: np.ndarray, lows: np.ndarray, closes: np.ndarray, period: int = 14) -> float:
        """Calculate ATR."""
        if len(highs) < period + 1:
            return 0.0
        
        tr_list = []
        for i in range(1, len(highs)):
            tr = max(
                highs[i] - lows[i],
                abs(highs[i] - closes[i-1]),
                abs(lows[i] - closes[i-1])
            )
            tr_list.append(tr)
        
        return np.mean(tr_list[-period:]) if tr_list else 0.0
    
    def get_volatility_regime(self, atr_pct: float) -> str:
        """Determine volatility regime for threshold adjustment."""
        if atr_pct < 0.015:
            return "low"
        elif atr_pct > 0.04:
            return "high"
        else:
            return "normal"
    
    def predict(self, 
                candles: List[List[float]],
                current_volatility: float = 0.02) -> EnsemblePrediction:
        """
        Generate ensemble prediction.
        
        Returns prediction with confidence and voting details.
        """
        # Default neutral prediction
        default = EnsemblePrediction(
            direction="NEUTRAL",
            confidence=0.5,
            probability_up=0.5,
            probability_down=0.5,
            model_predictions={},
            volatility_regime=self.get_volatility_regime(current_volatility),
            threshold_used=self._volatility_thresholds.get(
                self.get_volatility_regime(current_volatility), 0.55
            ),
            features_count=0
        )
        
        if not self._is_trained:
            return default
        
        # Extract features
        features = self.extract_features(candles)
        if features is None:
            return default
        
        features_array = np.array(features).reshape(1, -1)
        
        # Scale features
        if self._scaler:
            try:
                features_scaled = self._scaler.transform(features_array)
            except Exception as e:
                logger.debug("Scaler transform failed, using raw: %s", e)
                features_scaled = features_array
        else:
            features_scaled = features_array
        
        # Get predictions from each model
        model_preds = {}
        probabilities_up = []
        probabilities_down = []
        weights = []
        
        # XGBoost prediction
        if self._xgb_model and XGBOOST_AVAILABLE:
            try:
                xgb_proba = self._xgb_model.predict_proba(features_scaled)[0]
                prob_up = float(xgb_proba[1]) if len(xgb_proba) > 1 else 0.5
                model_preds["xgb"] = {
                    "prob_up": prob_up,
                    "prob_down": 1 - prob_up,
                    "direction": "UP" if prob_up > 0.5 else "DOWN"
                }
                weight = self._model_performance["xgb"].weight
                probabilities_up.append(prob_up * weight)
                probabilities_down.append((1 - prob_up) * weight)
                weights.append(weight)
            except Exception as e:
                logger.debug(f"XGBoost prediction failed: {e}")
        
        # Random Forest prediction
        if self._rf_model and SKLEARN_AVAILABLE:
            try:
                rf_proba = self._rf_model.predict_proba(features_scaled)[0]
                prob_up = float(rf_proba[1]) if len(rf_proba) > 1 else 0.5
                model_preds["rf"] = {
                    "prob_up": prob_up,
                    "prob_down": 1 - prob_up,
                    "direction": "UP" if prob_up > 0.5 else "DOWN"
                }
                weight = self._model_performance["rf"].weight
                probabilities_up.append(prob_up * weight)
                probabilities_down.append((1 - prob_up) * weight)
                weights.append(weight)
            except Exception as e:
                logger.debug(f"RF prediction failed: {e}")
        
        # LSTM prediction (if available)
        if self._lstm_model and TENSORFLOW_AVAILABLE:
            try:
                # LSTM needs sequence input
                lstm_input = features_scaled.reshape(1, 1, -1)
                lstm_pred = self._lstm_model.predict(lstm_input, verbose=0)[0]
                prob_up = float(lstm_pred[0]) if len(lstm_pred) > 0 else 0.5
                model_preds["lstm"] = {
                    "prob_up": prob_up,
                    "prob_down": 1 - prob_up,
                    "direction": "UP" if prob_up > 0.5 else "DOWN"
                }
                weight = self._model_performance["lstm"].weight
                probabilities_up.append(prob_up * weight)
                probabilities_down.append((1 - prob_up) * weight)
                weights.append(weight)
            except Exception as e:
                logger.debug(f"LSTM prediction failed: {e}")
        
        if not weights:
            return default
        
        # Weighted average
        total_weight = sum(weights)
        ensemble_prob_up = sum(probabilities_up) / total_weight
        ensemble_prob_down = sum(probabilities_down) / total_weight
        
        # Determine volatility regime and threshold
        vol_regime = self.get_volatility_regime(current_volatility)
        threshold = self._volatility_thresholds[vol_regime]
        
        # Determine direction
        if ensemble_prob_up > threshold:
            direction = "UP"
            confidence = ensemble_prob_up
        elif ensemble_prob_down > threshold:
            direction = "DOWN"
            confidence = ensemble_prob_down
        else:
            direction = "NEUTRAL"
            confidence = max(ensemble_prob_up, ensemble_prob_down)
        
        prediction = EnsemblePrediction(
            direction=direction,
            confidence=confidence,
            probability_up=ensemble_prob_up,
            probability_down=ensemble_prob_down,
            model_predictions=model_preds,
            volatility_regime=vol_regime,
            threshold_used=threshold,
            features_count=len(features)
        )
        
        # Store for later evaluation
        self._prediction_history.append({
            "prediction": prediction,
            "features": features,
            "timestamp": time.time()
        })
        
        return prediction
    
    def add_training_sample(self, 
                            candles: List[List[float]], 
                            future_return: float,
                            label: int) -> None:
        """
        Add a sample to training data.
        
        Args:
            candles: Historical candles at prediction time
            future_return: Actual return that occurred
            label: 1 for UP, 0 for DOWN
        """
        features = self.extract_features(candles)
        if features:
            self._training_data.append({
                "features": features,
                "label": label,
                "return": future_return,
                "timestamp": time.time()
            })
    
    def record_outcome(self, prediction_timestamp: int, actual_direction: str) -> None:
        """
        Record the actual outcome of a prediction for performance tracking.
        """
        actual_up = actual_direction == "UP"
        
        # Find the prediction
        for item in self._prediction_history:
            if abs(item["timestamp"] - prediction_timestamp) < 60:  # Within 1 minute
                pred = item["prediction"]
                
                # Update model performance
                for model_name, model_pred in pred.model_predictions.items():
                    perf = self._model_performance.get(model_name)
                    if perf:
                        perf.total_predictions += 1
                        if (model_pred["direction"] == "UP") == actual_up:
                            perf.correct_predictions += 1
                        
                        # Update recent accuracy (exponential moving average)
                        correct = 1.0 if (model_pred["direction"] == "UP") == actual_up else 0.0
                        perf.recent_accuracy = 0.9 * perf.recent_accuracy + 0.1 * correct
                        
                        # Update weight based on performance
                        perf.weight = 0.5 + perf.recent_accuracy  # Weight range: 0.5 to 1.5
                        perf.last_update = int(time.time())
                
                break
    
    def train(self, force: bool = False) -> bool:
        """
        Train or retrain the ensemble models.
        
        Returns True if training was successful.
        """
        if not force and self._last_retrain and (time.time() - self._last_retrain) < self.retrain_interval:
            return False
        
        if len(self._training_data) < 100:
            logger.info(f"Not enough training data: {len(self._training_data)} samples")
            return False
        
        with self._lock:
            try:
                # Prepare data
                X = np.array([d["features"] for d in self._training_data])
                y = np.array([d["label"] for d in self._training_data])
                
                # Train/test split
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=0.2, random_state=42
                )
                
                # Fit scaler
                if self._scaler:
                    X_train_scaled = self._scaler.fit_transform(X_train)
                    X_test_scaled = self._scaler.transform(X_test)
                else:
                    X_train_scaled = X_train
                    X_test_scaled = X_test
                
                # Train XGBoost
                if XGBOOST_AVAILABLE:
                    try:
                        self._xgb_model = xgb.XGBClassifier(
                            n_estimators=100,
                            max_depth=6,
                            learning_rate=0.1,
                            random_state=42,
                            eval_metric='logloss'
                        )
                        self._xgb_model.fit(X_train_scaled, y_train)
                        xgb_acc = self._xgb_model.score(X_test_scaled, y_test)
                        logger.info(f"XGBoost trained, accuracy: {xgb_acc:.2%}")
                        self._model_performance["xgb"].recent_accuracy = xgb_acc
                    except Exception as e:
                        logger.error(f"XGBoost training failed: {e}")
                
                # Train Random Forest
                if SKLEARN_AVAILABLE:
                    try:
                        self._rf_model = RandomForestClassifier(
                            n_estimators=100,
                            max_depth=10,
                            random_state=42,
                            n_jobs=-1
                        )
                        self._rf_model.fit(X_train_scaled, y_train)
                        rf_acc = self._rf_model.score(X_test_scaled, y_test)
                        logger.info(f"Random Forest trained, accuracy: {rf_acc:.2%}")
                        self._model_performance["rf"].recent_accuracy = rf_acc
                    except Exception as e:
                        logger.error(f"RF training failed: {e}")
                
                # Train LSTM
                if TENSORFLOW_AVAILABLE and len(X_train_scaled) > 200:
                    try:
                        X_lstm = X_train_scaled.reshape(-1, 1, X_train_scaled.shape[1])
                        
                        model = Sequential([
                            LSTM(50, input_shape=(1, X_train_scaled.shape[1]), return_sequences=False),
                            Dropout(0.2),
                            Dense(25, activation='relu'),
                            Dense(1, activation='sigmoid')
                        ])
                        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
                        model.fit(X_lstm, y_train, epochs=10, batch_size=32, verbose=0)
                        
                        X_test_lstm = X_test_scaled.reshape(-1, 1, X_test_scaled.shape[1])
                        _, lstm_acc = model.evaluate(X_test_lstm, y_test, verbose=0)
                        
                        self._lstm_model = model
                        logger.info(f"LSTM trained, accuracy: {lstm_acc:.2%}")
                        self._model_performance["lstm"].recent_accuracy = lstm_acc
                    except Exception as e:
                        logger.error(f"LSTM training failed: {e}")
                
                self._is_trained = True
                self._last_retrain = time.time()
                
                # Save models
                self._save_models()
                
                return True
                
            except Exception as e:
                logger.error(f"Ensemble training failed: {e}")
                return False
    
    def _save_models(self) -> None:
        """Save trained models to disk."""
        try:
            if self._xgb_model:
                self._xgb_model.save_model(str(self.models_dir / "xgb_model.json"))
            
            if self._rf_model:
                with open(self.models_dir / "rf_model.pkl", "wb") as f:
                    pickle.dump(self._rf_model, f)
            
            if self._scaler:
                with open(self.models_dir / "scaler.pkl", "wb") as f:
                    pickle.dump(self._scaler, f)
            
            if self._lstm_model:
                self._lstm_model.save(str(self.models_dir / "lstm_model.keras"))
            
            # Save performance metrics
            perf_data = {k: {"accuracy": v.recent_accuracy, "weight": v.weight} 
                         for k, v in self._model_performance.items()}
            with open(self.models_dir / "performance.json", "w") as f:
                json.dump(perf_data, f)
            
            logger.info("Models saved successfully")
        except Exception as e:
            logger.error(f"Failed to save models: {e}")
    
    def _load_models(self) -> None:
        """Load models from disk."""
        try:
            # Load XGBoost
            xgb_path = self.models_dir / "xgb_model.json"
            if xgb_path.exists() and XGBOOST_AVAILABLE:
                self._xgb_model = xgb.XGBClassifier()
                self._xgb_model.load_model(str(xgb_path))
                self._is_trained = True
                logger.info("XGBoost model loaded")
            
            # Load Random Forest
            rf_path = self.models_dir / "rf_model.pkl"
            if rf_path.exists():
                with open(rf_path, "rb") as f:
                    self._rf_model = pickle.load(f)
                self._is_trained = True
                logger.info("Random Forest model loaded")
            
            # Load scaler
            scaler_path = self.models_dir / "scaler.pkl"
            if scaler_path.exists():
                with open(scaler_path, "rb") as f:
                    self._scaler = pickle.load(f)
            
            # Load LSTM
            lstm_path = self.models_dir / "lstm_model.keras"
            if lstm_path.exists() and TENSORFLOW_AVAILABLE:
                self._lstm_model = tf.keras.models.load_model(str(lstm_path))
                self._is_trained = True
                logger.info("LSTM model loaded")
            
            # Load performance
            perf_path = self.models_dir / "performance.json"
            if perf_path.exists():
                with open(perf_path, "r") as f:
                    perf_data = json.load(f)
                for k, v in perf_data.items():
                    if k in self._model_performance:
                        self._model_performance[k].recent_accuracy = v["accuracy"]
                        self._model_performance[k].weight = v["weight"]
                        
        except Exception as e:
            logger.warning(f"Failed to load some models: {e}")
    
    def get_feature_importance(self) -> Dict[str, float]:
        """Get XGBoost feature importance for indicator refinement."""
        if not self._xgb_model or not XGBOOST_AVAILABLE:
            return {}
        try:
            imp = self._xgb_model.feature_importances_
            names = self.FEATURE_NAMES
            return {names[i]: float(imp[i]) for i in range(min(len(imp), len(names)))}
        except Exception as e:
            logger.debug("Feature importance failed: %s", e)
            return {}

    def get_status(self) -> Dict[str, Any]:
        """Get ensemble status for monitoring."""
        return {
            "is_trained": self._is_trained,
            "training_samples": len(self._training_data),
            "predictions_made": len(self._prediction_history),
            "last_retrain": self._last_retrain,
            "model_performance": {
                k: {
                    "accuracy": v.recent_accuracy,
                    "weight": v.weight,
                    "total_predictions": v.total_predictions
                }
                for k, v in self._model_performance.items()
            },
            "models_available": {
                "xgb": self._xgb_model is not None,
                "rf": self._rf_model is not None,
                "lstm": self._lstm_model is not None
            }
        }


# Singleton instance
_ml_ensemble: Optional[MLEnsemble] = None


def get_ml_ensemble() -> MLEnsemble:
    """Get or create the ML ensemble singleton."""
    global _ml_ensemble
    if _ml_ensemble is None:
        _ml_ensemble = MLEnsemble()
    return _ml_ensemble
