
import os
import numpy as np
import pandas as pd
import joblib
import json
import logging
from typing import Dict, List, Any, Optional, Tuple

# Optional imports to prevent crash if not installed
try:
    from tensorflow.keras.models import Sequential, load_model
    from tensorflow.keras.layers import LSTM, Dense, Dropout
    from tensorflow.keras.optimizers import Adam
    import tensorflow as tf
except ImportError:
    pass

try:
    import xgboost as xgb
except ImportError:
    pass

from sklearn.preprocessing import MinMaxScaler
from datetime import datetime

logger = logging.getLogger(__name__)

class PricePredictor:
    """
    Handles LSTM (Time-Series) and XGBoost (Classification) predictions.
    Design: 
    - Online learning: Trains on recent history per bot startup/daily.
    - Lightweight: Uses simplified architectures suitable for VPS.
    """
    
    def __init__(self, data_dir: str = "ml_models"):
        self.data_dir = data_dir
        if not os.path.exists(data_dir):
            os.makedirs(data_dir)
            
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        self.lstm_model = None
        self.xgb_model = None
        
        # Hyperparams
        self.sequence_length = 60  # Candles lookback
        self.forecast_horizon = 12 # Candles ahead (e.g. 1h if 5m candles)
        
    def _prepare_lstm_data(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, Any]:
        """Prepares sequences for LSTM training."""
        data = df.filter(['close']).values
        scaled_data = self.scaler.fit_transform(data)
        
        x_train, y_train = [], []
        for i in range(self.sequence_length, len(scaled_data) - self.forecast_horizon):
            x_train.append(scaled_data[i-self.sequence_length:i, 0])
            # Predict "max high" or simple "close" in horizon? Let's predict curve
            y_train.append(scaled_data[i:i+self.forecast_horizon, 0])
            
        x_train = np.array(x_train)
        y_train = np.array(y_train)
        
        # Reshape for LSTM [samples, time steps, features]
        x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))
        
        return x_train, y_train, self.scaler

    def build_lstm(self):
        """Creates a lightweight LSTM model."""
        try:
            model = Sequential()
            model.add(LSTM(50, return_sequences=False, input_shape=(self.sequence_length, 1)))
            model.add(Dropout(0.2))
            model.add(Dense(25))
            model.add(Dense(self.forecast_horizon)) # Predict next N steps
            
            model.compile(optimizer=Adam(learning_rate=0.001), loss='mean_squared_error')
            self.lstm_model = model
            return model
        except Exception as e:
            logger.error(f"Failed to build LSTM: {e}")
            return None

    def train_lstm(self, candles: List[Dict[str, Any]], symbol: str):
        """Trains LSTM on provided candle history."""
        if not candles or len(candles) < 200:
            return False
            
        try:
            df = pd.DataFrame(candles)
            if 'close' not in df.columns:
                df['close'] = df['c'] # fallback for some data formats
                
            x_train, y_train, _ = self._prepare_lstm_data(df)
            
            if self.lstm_model is None:
                self.build_lstm()
                
            if self.lstm_model:
                # Fast training: 5 epochs, batch 32
                self.lstm_model.fit(x_train, y_train, batch_size=32, epochs=5, verbose=0)
                
                # Save model
                path = os.path.join(self.data_dir, f"{symbol}_lstm.keras")
                self.lstm_model.save(path)
                logger.info(f"Trained LSTM for {symbol}")
                return True
        except Exception as e:
            logger.error(f"LSTM Train Error: {e}")
            return False

    def predict_lstm(self, candles: List[Dict[str, Any]], symbol: str) -> Dict[str, Any]:
        """
        Returns predicted price curve for next horizon.
        """
        try:
            # Load if needed
            if self.lstm_model is None:
                path = os.path.join(self.data_dir, f"{symbol}_lstm.keras")
                if os.path.exists(path):
                    self.lstm_model = load_model(path)
                else:
                    return {} # Not trained yet
            
            df = pd.DataFrame(candles)
            if 'close' not in df.columns:
                df['close'] = df['c'] # simple normalization
                
            # Prepare last sequence
            data = df.filter(['close']).values
            # Re-fit scaler on current window to match range
            scaled_data = self.scaler.fit_transform(data) 
            
            last_60 = scaled_data[-self.sequence_length:]
            x_input = np.array([last_60])
            x_input = np.reshape(x_input, (x_input.shape[0], x_input.shape[1], 1))
            
            # Predict
            pred_scaled = self.lstm_model.predict(x_input, verbose=0)
            pred = self.scaler.inverse_transform(pred_scaled)
            
            return {
                "forecast": pred[0].tolist(),
                "current_price": data[-1][0]
            }
            
        except Exception as e:
            logger.error(f"LSTM Predict Error: {e}")
            return {}

    # ==========================
    # XGBoost (Feature Classifier)
    # ==========================
    def train_xgboost(self, candles: List[Dict[str, Any]], symbol: str):
        """
        Trains XGBoost to key trend features.
        Target: 1 if price rises > 1% in next 12 periods, else 0.
        """
        try:
            df = pd.DataFrame(candles)
            # Engineer features: RSI, SMA dist, Volatility
            df['close'] = df['close'].astype(float)
            df['returns'] = df['close'].pct_change()
            df['volatility'] = df['returns'].rolling(window=10).std()
            df['sma_20'] = df['close'].rolling(window=20).mean()
            df['dist_sma'] = (df['close'] - df['sma_20']) / df['sma_20']
            
            # Labeling
            future_returns = df['close'].shift(-12) / df['close'] - 1
            df['target'] = (future_returns > 0.01).astype(int)
            
            df.dropna(inplace=True)
            
            features = ['volatility', 'dist_sma']
            X = df[features]
            y = df['target']
            
            model = xgb.XGBClassifier(n_estimators=100, learning_rate=0.05, max_depth=3, eval_metric='logloss')
            model.fit(X, y)
            
            self.xgb_model = model
            path = os.path.join(self.data_dir, f"{symbol}_xgb.json")
            model.save_model(path)
            
            return True
        except Exception as e:
            logger.error(f"XGB Train Error: {e}")
            return False

    def predict_xgboost(self, candles: List[Dict[str, Any]], symbol: str):
        """Returns Bullish Probability (0.0 - 1.0) or None if model missing/unavailable."""
        try:
            # Need to re-calc features on latest candle
            if not self.xgb_model:
                path = os.path.join(self.data_dir, f"{symbol}_xgb.json")
                if os.path.exists(path):
                    self.xgb_model = xgb.XGBClassifier()
                    self.xgb_model.load_model(path)
                else:
                    logger.debug("ML predictor: no model for %s, returning None (unavailable)", symbol)
                    return None
            
            df = pd.DataFrame(candles[-50:]) # just need recent tail
            df['close'] = df['close'].astype(float)
            df['returns'] = df['close'].pct_change()
            df['volatility'] = df['returns'].rolling(window=10).std()
            df['sma_20'] = df['close'].rolling(window=20).mean()
            df['dist_sma'] = (df['close'] - df['sma_20']) / df['sma_20']
            
            last_row = df.iloc[[-1]][['volatility', 'dist_sma']]
            
            probs = self.xgb_model.predict_proba(last_row)
            # probs is [[prob_0, prob_1]]
            bullish_prob = probs[0][1]
            return float(bullish_prob)
            
        except Exception as e:
            logger.exception("predict_xgboost failed: %s", e)
            return None
