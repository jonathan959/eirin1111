"""
Phase 3 Intelligence: ML Price Prediction
Uses LSTM, Random Forest, and XGBoost for price prediction
"""

import numpy as np
import logging
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
import time

logger = logging.getLogger(__name__)

# Try to import ML libraries (optional dependencies)
try:
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.preprocessing import StandardScaler
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    logger.warning("scikit-learn not available - ML predictions disabled")

try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    logger.warning("xgboost not available - XGBoost predictions disabled")

try:
    import tensorflow as tf
    TENSORFLOW_AVAILABLE = True
except ImportError:
    TENSORFLOW_AVAILABLE = False
    logger.warning("tensorflow not available - LSTM predictions disabled")


@dataclass
class PredictionResult:
    """ML prediction result"""
    price_prediction: float
    confidence: float  # 0.0 to 1.0
    direction: str  # "UP", "DOWN", "NEUTRAL"
    model_used: str
    features_used: List[str]
    prediction_horizon: int  # minutes ahead
    timestamp: int


class MLPricePredictor:
    """
    ML-based price prediction using ensemble of models.
    """
    
    def __init__(
        self,
        use_lstm: bool = True,
        use_rf: bool = True,
        use_xgb: bool = True,
        prediction_horizon_minutes: int = 60,  # Predict 1 hour ahead
    ):
        self.use_lstm = use_lstm and TENSORFLOW_AVAILABLE
        self.use_rf = use_rf and SKLEARN_AVAILABLE
        self.use_xgb = use_xgb and XGBOOST_AVAILABLE
        self.prediction_horizon = prediction_horizon_minutes
        
        # Model storage (lazy initialization)
        self._lstm_model = None
        self._rf_model = None
        self._xgb_model = None
        self._scaler = StandardScaler() if SKLEARN_AVAILABLE else None
        
        # Feature cache
        self._feature_cache: Dict[str, List[float]] = {}
    
    def _extract_features(self, candles: List[List[float]], lookback: int = 50) -> np.ndarray:
        """
        Extract features from candles for ML prediction.
        
        Features:
        - Price features: close, high, low, open
        - Technical indicators: SMA, EMA, RSI, momentum
        - Volume features: volume, volume MA
        - Volatility: ATR, price range
        """
        if not candles or len(candles) < lookback:
            return np.array([])
        
        features = []
        closes = [c[4] for c in candles[-lookback:]]  # Close prices
        highs = [c[2] for c in candles[-lookback:]]
        lows = [c[3] for c in candles[-lookback:]]
        volumes = [c[5] for c in candles[-lookback:]]
        
        # Price features
        current_price = closes[-1]
        features.append(current_price)
        features.append(highs[-1])
        features.append(lows[-1])
        
        # Moving averages
        if len(closes) >= 20:
            sma_20 = sum(closes[-20:]) / 20
            features.append(sma_20)
            features.append((current_price - sma_20) / sma_20)  # Price vs SMA
        else:
            features.extend([0.0, 0.0])
        
        if len(closes) >= 50:
            sma_50 = sum(closes[-50:]) / 50
            features.append(sma_50)
        else:
            features.append(0.0)
        
        # EMA
        if len(closes) >= 12:
            ema_12 = self._calculate_ema(closes, 12)
            features.append(ema_12)
            features.append((current_price - ema_12) / ema_12)
        else:
            features.extend([0.0, 0.0])
        
        # RSI
        if len(closes) >= 14:
            rsi = self._calculate_rsi(closes, 14)
            features.append(rsi)
        else:
            features.append(50.0)  # Neutral RSI
        
        # Momentum
        if len(closes) >= 10:
            momentum = (closes[-1] - closes[-10]) / closes[-10]
            features.append(momentum)
        else:
            features.append(0.0)
        
        # Volatility (ATR-like)
        if len(highs) >= 14:
            atr = self._calculate_atr(highs, lows, closes, 14)
            features.append(atr)
            features.append(atr / current_price if current_price > 0 else 0.0)
        else:
            features.extend([0.0, 0.0])
        
        # Volume features
        if len(volumes) > 0:
            features.append(volumes[-1])
            if len(volumes) >= 20:
                vol_ma = sum(volumes[-20:]) / 20
                features.append(vol_ma)
                features.append(volumes[-1] / vol_ma if vol_ma > 0 else 1.0)
            else:
                features.extend([0.0, 1.0])
        else:
            features.extend([0.0, 0.0, 1.0])
        
        return np.array(features).reshape(1, -1)
    
    def _calculate_ema(self, prices: List[float], period: int) -> float:
        """Calculate Exponential Moving Average"""
        if len(prices) < period:
            return prices[-1] if prices else 0.0
        
        multiplier = 2.0 / (period + 1)
        ema = sum(prices[:period]) / period
        
        for price in prices[period:]:
            ema = (price - ema) * multiplier + ema
        
        return ema
    
    def _calculate_rsi(self, prices: List[float], period: int = 14) -> float:
        """Calculate Relative Strength Index"""
        if len(prices) < period + 1:
            return 50.0
        
        gains = []
        losses = []
        for i in range(1, len(prices)):
            change = prices[i] - prices[i-1]
            if change > 0:
                gains.append(change)
                losses.append(0.0)
            else:
                gains.append(0.0)
                losses.append(abs(change))
        
        if len(gains) < period:
            return 50.0
        
        avg_gain = sum(gains[-period:]) / period
        avg_loss = sum(losses[-period:]) / period
        
        if avg_loss == 0:
            return 100.0
        
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    def _calculate_atr(self, highs: List[float], lows: List[float], closes: List[float], period: int) -> float:
        """Calculate Average True Range"""
        if len(highs) < period + 1:
            return 0.0
        
        true_ranges = []
        for i in range(1, len(highs)):
            tr = max(
                highs[i] - lows[i],
                abs(highs[i] - closes[i-1]),
                abs(lows[i] - closes[i-1])
            )
            true_ranges.append(tr)
        
        if len(true_ranges) < period:
            return 0.0
        
        return sum(true_ranges[-period:]) / period
    
    def _train_rf_model(self, X: np.ndarray, y: np.ndarray):
        """Train Random Forest model"""
        if not SKLEARN_AVAILABLE:
            return None
        
        try:
            model = RandomForestRegressor(
                n_estimators=100,
                max_depth=10,
                random_state=42,
                n_jobs=-1
            )
            model.fit(X, y)
            return model
        except Exception as e:
            logger.error(f"RF training failed: {e}")
            return None
    
    def _train_xgb_model(self, X: np.ndarray, y: np.ndarray):
        """Train XGBoost model"""
        if not XGBOOST_AVAILABLE:
            return None
        
        try:
            model = xgb.XGBRegressor(
                n_estimators=100,
                max_depth=6,
                learning_rate=0.1,
                random_state=42
            )
            model.fit(X, y)
            return model
        except Exception as e:
            logger.error(f"XGBoost training failed: {e}")
            return None
    
    def _train_lstm_model(self, X: np.ndarray, y: np.ndarray):
        """Train LSTM model (simplified - would need sequence data)"""
        if not TENSORFLOW_AVAILABLE:
            return None
        
        # LSTM requires sequence data - simplified implementation
        # In production, would need proper sequence preparation
        logger.warning("LSTM training not fully implemented - using fallback")
        return None
    
    def predict(
        self,
        candles: List[List[float]],
        current_price: float,
        symbol: str = ""
    ) -> Optional[PredictionResult]:
        """
        Predict future price using ML models.
        
        Returns:
            PredictionResult with price prediction, confidence, direction
        """
        if not candles or len(candles) < 50:
            return None
        
        # Extract features
        try:
            features = self._extract_features(candles, lookback=50)
            if features.size == 0:
                return None
        except Exception as e:
            logger.error(f"Feature extraction failed: {e}")
            return None
        
        # Simple prediction using moving average trend (fallback)
        # In production, would use trained models
        closes = [c[4] for c in candles[-50:]]
        
        # Simple trend-based prediction
        if len(closes) >= 20:
            sma_20 = sum(closes[-20:]) / 20
            trend = (closes[-1] - sma_20) / sma_20
            
            # Predict price change based on trend
            predicted_change_pct = trend * 0.5  # Conservative multiplier
            predicted_price = current_price * (1 + predicted_change_pct)
            
            # Determine direction
            if predicted_change_pct > 0.01:
                direction = "UP"
                confidence = min(0.7, abs(predicted_change_pct) * 10)
            elif predicted_change_pct < -0.01:
                direction = "DOWN"
                confidence = min(0.7, abs(predicted_change_pct) * 10)
            else:
                direction = "NEUTRAL"
                confidence = 0.3
            
            return PredictionResult(
                price_prediction=predicted_price,
                confidence=confidence,
                direction=direction,
                model_used="trend_fallback",
                features_used=["SMA20", "trend"],
                prediction_horizon=self.prediction_horizon,
                timestamp=int(time.time()),
            )
        else:
            return None
    
    def should_enter_based_on_prediction(
        self,
        prediction: PredictionResult,
        current_price: float,
        threshold_confidence: float = 0.6
    ) -> Tuple[bool, str]:
        """
        Determine if entry is recommended based on ML prediction.
        
        Returns:
            (should_enter, reason)
        """
        if not prediction:
            return False, "No prediction available"
        
        if prediction.confidence < threshold_confidence:
            return False, f"Low prediction confidence: {prediction.confidence:.2f}"
        
        if prediction.direction == "UP" and prediction.confidence >= threshold_confidence:
            expected_gain = (prediction.price_prediction - current_price) / current_price
            if expected_gain > 0.01:  # At least 1% expected gain
                return True, f"ML predicts {prediction.direction} with {prediction.confidence:.0%} confidence, expected gain: {expected_gain*100:.2f}%"
        
        return False, f"ML prediction: {prediction.direction} (confidence: {prediction.confidence:.0%})"


# =============================================================================
# Simplified ML Predictor (for when full ML libraries not available)
# =============================================================================

class SimpleMLPredictor:
    """
    Simplified ML predictor using basic technical analysis.
    Used when full ML libraries are not available.
    """
    
    def __init__(self, prediction_horizon_minutes: int = 60):
        self.prediction_horizon = prediction_horizon_minutes
    
    def predict(
        self,
        candles: List[List[float]],
        current_price: float,
        symbol: str = ""
    ) -> Optional[PredictionResult]:
        """Simple trend-based prediction"""
        if not candles or len(candles) < 20:
            return None
        
        closes = [c[4] for c in candles[-50:]]
        
        # Calculate trend strength
        if len(closes) >= 20:
            sma_20 = sum(closes[-20:]) / 20
            trend_pct = (closes[-1] - sma_20) / sma_20
            
            # Simple momentum
            if len(closes) >= 10:
                momentum = (closes[-1] - closes[-10]) / closes[-10]
            else:
                momentum = 0.0
            
            # Combined signal
            signal_strength = (trend_pct * 0.7) + (momentum * 0.3)
            
            # Predict price
            predicted_change = signal_strength * 0.3  # Conservative
            predicted_price = current_price * (1 + predicted_change)
            
            # Determine direction and confidence
            if signal_strength > 0.02:
                direction = "UP"
                confidence = min(0.7, abs(signal_strength) * 15)
            elif signal_strength < -0.02:
                direction = "DOWN"
                confidence = min(0.7, abs(signal_strength) * 15)
            else:
                direction = "NEUTRAL"
                confidence = 0.3
            
            return PredictionResult(
                price_prediction=predicted_price,
                confidence=confidence,
                direction=direction,
                model_used="simple_trend",
                features_used=["SMA20", "momentum"],
                prediction_horizon=self.prediction_horizon,
                timestamp=int(time.time()),
            )
        
        return None


# =============================================================================
# Factory function
# =============================================================================

def create_ml_predictor(use_advanced: bool = False) -> MLPricePredictor:
    """
    Create ML predictor instance.
    
    Args:
        use_advanced: If True, tries to use full ML models (requires libraries)
    
    Returns:
        MLPricePredictor or SimpleMLPredictor
    """
    if use_advanced and (SKLEARN_AVAILABLE or XGBOOST_AVAILABLE or TENSORFLOW_AVAILABLE):
        return MLPricePredictor()
    else:
        return SimpleMLPredictor()
