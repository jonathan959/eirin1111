
import numpy as np
from typing import List, Dict, Any, Tuple

class AnomalyDetector:
    """
    Monitors market data for statistical anomalies (Flash Crashes, Pumps, Latency).
    """
    
    def detect_price_anomaly(self, candles: List[Dict[str, Any]], z_threshold: float = 4.0) -> Tuple[bool, str]:
        """
        Checks if the latest price move is statistically improbable (Z-Score > threshold).
        """
        if not candles or len(candles) < 20:
            return False, ""
            
        closes = np.array([float(c.get('close') or c.get('c')) for c in candles])
        returns = np.diff(closes) / closes[:-1]
        
        current_return = returns[-1]
        
        # Calculate stats on previous window (excluding current)
        history = returns[:-1]
        mean = np.mean(history)
        std = np.std(history)
        
        if std == 0:
            return False, ""
            
        z_score = abs(current_return - mean) / std
        
        if z_score > z_threshold:
            direction = "CRASH" if current_return < 0 else "PUMP"
            return True, f"Flash {direction} detected! Z-Score: {z_score:.2f} (Threshold: {z_threshold})"
            
        return False, ""

    def detect_liquidity_void(self, orderbook: Dict[str, Any]) -> bool:
        """
        Checks for empty order book or massive spread (Liquidity Void).
        """
        bids = orderbook.get("bids", [])
        asks = orderbook.get("asks", [])
        
        if not bids or not asks:
            return True
        
        best_bid = bids[0][0]
        best_ask = asks[0][0]
        
        spread_pct = (best_ask - best_bid) / best_bid
        
        # If spread > 5%, market is broken/illiquid
        if spread_pct > 0.05:
            return True
            
        return False
