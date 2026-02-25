
import numpy as np
import logging
from typing import Dict, Any, List

logger = logging.getLogger(__name__)

class StrategyOptimizer:
    """
    Adaptive parameter tuning based on market conditions.
    """
    
    @staticmethod
    def calculate_volatility(candles: List[Dict[str, Any]], window: int = 20) -> float:
        """Calculates realized volatility (std dev of returns)."""
        if not candles or len(candles) < window:
            return 0.0
        
        closes = np.array([float(c.get('close') or c.get('c')) for c in candles])
        returns = np.diff(closes) / closes[:-1]
        
        # Annualized? No, let's keep it per-period for relative comparison
        vol = np.std(returns[-window:])
        return float(vol)

    def optimize_bot_config(self, bot_config: Dict[str, Any], candles: List[Dict[str, Any]], win_rate: float) -> Dict[str, Any]:
        """
        Returns a set of overrides for bot parameters.
        
        Logic:
        - High Volatility -> Increase Safety Step & Max Safety Orders (Scaling in wider).
        - Low Win Rate (<40%) -> Increase Step Multiplier (wait for deeper dips).
        - High Win Rate (>90%) -> Tighten TP (take profits faster) or Increase Base Order.
        """
        overrides = {}
        
        vol = self.calculate_volatility(candles)
        base_step = float(bot_config.get("first_dev", 1.0))
        base_tp = float(bot_config.get("tp", 1.0))
        
        # 1. Volatility Adaptation
        # If volatile (> 0.5% per period), widen the grid
        if vol > 0.005: 
            overrides["first_dev"] = base_step * 1.5
            overrides["step_mult"] = max(float(bot_config.get("step_mult", 1.0)), 1.3)
            logger.info(f"Bot {bot_config.get('id')}: High Volatility ({vol:.4f}). Widening grid.")
        
        # 2. Performance Adaptation
        if win_rate < 0.40:
            # Struggling bot: Protect capital
            # Widen safety steps significantly to lower entry price avg
            overrides["step_mult"] = max(float(bot_config.get("step_mult", 1.0)), 1.5)
            # Maybe reduce base order? (Requires restart usually, but we can suggest it)
        elif win_rate > 0.90:
            # Winning streak: optimize for turnover
            # Tighten TP slightly to cycle faster? Or let it ride?
            # Let's keep TP stable but maybe Decrease step slightly to catch shallow dips
            pass
            
        return overrides
