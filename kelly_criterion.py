"""
Phase 2 Intelligence: Kelly Criterion Position Sizing
Mathematically optimal bet sizing for maximum long-term growth
"""
import time
from typing import Dict, List, Optional, Tuple, Any
import logging

logger = logging.getLogger(__name__)


class KellyPositionSizer:
    """
    Implements Kelly Criterion for optimal position sizing.
    
    Kelly Formula:
    f* = (p × b - q) / b
    
    Where:
    - f* = optimal fraction of capital to risk
    - p = probability of win (win rate)
    - q = probability of loss (1 - p)
    - b = ratio of win to loss (avg_win / avg_loss)
    
    Example:
    - Win rate: 60% (p=0.6, q=0.4)
    - Avg win: $50, Avg loss: $30 (b=1.67)
    - Kelly% = (0.6 × 1.67 - 0.4) / 1.67 = 0.36 → 36% of capital
    
    Safety: We use fractional Kelly (typically 25%) to reduce volatility
    """
    
    def __init__(
        self,
        kelly_fraction: float = 0.25,  # Use 25% of full Kelly (safer)
        max_position_pct: float = 0.10,  # Cap at 10% of capital
        min_trades_required: int = 20   # Need 20+ trades for stats
    ):
        self.kelly_fraction = kelly_fraction
        self.max_position_pct = max_position_pct
        self.min_trades_required = min_trades_required
    
    def calculate_position_size(
        self,
        available_capital: float,
        bot_stats: Dict[str, Any]
    ) -> Tuple[float, str, Dict[str, Any]]:
        """
        Calculate optimal position size using Kelly Criterion.
        
        Args:
            available_capital: Current available capital (USD)
            bot_stats: Dict with keys:
                - total_trades: int
                - winning_trades: int
                - avg_profit_pct: float (e.g., 0.05 = 5%)
                - avg_loss_pct: float (e.g., -0.03 = -3%)
        
        Returns:
            (position_size_usd, reason, details)
        """
        total_trades = bot_stats.get('total_trades', 0)
        winning_trades = bot_stats.get('winning_trades', 0)
        avg_profit_pct = bot_stats.get('avg_profit_pct', 0.0)
        avg_loss_pct = bot_stats.get('avg_loss_pct', 0.0)
        
        # Not enough data yet - use conservative default
        if total_trades < self.min_trades_required:
            default_size = available_capital * 0.02  # 2% default
            reason = f"Insufficient trades ({total_trades}/{self.min_trades_required}), using 2% default"
            details = {
                'method': 'default',
                'kelly_pct': None,
                'position_pct': 0.02,
                'position_size': default_size
            }
            return default_size, reason, details
        
        # Calculate win rate
        win_rate = winning_trades / total_trades if total_trades > 0 else 0.0
        loss_rate = 1 - win_rate
        
        # Safety checks
        if win_rate <= 0.0 or win_rate >= 1.0:
            default_size = available_capital * 0.01  # 1% ultra-conservative
            reason = f"Invalid win rate ({win_rate:.1%}), using 1% fallback"
            details = {
                'method': 'fallback',
                'win_rate': win_rate,
                'position_pct': 0.01,
                'position_size': default_size
            }
            return default_size, reason, details
        
        if avg_profit_pct <= 0 or avg_loss_pct >= 0:
            default_size = available_capital * 0.01
            reason = f"Invalid profit/loss stats (profit: {avg_profit_pct:.2%}, loss: {avg_loss_pct:.2%})"
            details = {
                'method': 'fallback',
                'avg_profit_pct': avg_profit_pct,
                'avg_loss_pct': avg_loss_pct,
                'position_pct': 0.01,
                'position_size': default_size
            }
            return default_size, reason, details
        
        # Calculate win/loss ratio (b)
        # Note: avg_loss_pct is negative, so we take abs
        win_loss_ratio = avg_profit_pct / abs(avg_loss_pct)
        
        # Kelly Formula: f* = (p × b - q) / b
        kelly_pct = (win_rate * win_loss_ratio - loss_rate) / win_loss_ratio
        
        # Apply fractional Kelly (reduce volatility)
        fractional_kelly_pct = kelly_pct * self.kelly_fraction
        
        # Cap at maximum
        safe_kelly_pct = min(fractional_kelly_pct, self.max_position_pct)
        
        # Ensure positive
        if safe_kelly_pct <= 0:
            default_size = available_capital * 0.01
            reason = f"Negative Kelly ({kelly_pct:.2%}), system suggests NO trade - using 1% minimum"
            details = {
                'method': 'minimum',
                'kelly_pct': kelly_pct,
                'win_rate': win_rate,
                'win_loss_ratio': win_loss_ratio,
                'position_pct': 0.01,
                'position_size': default_size
            }
            return default_size, reason, details
        
        # Calculate final position size
        position_size = available_capital * safe_kelly_pct
        
        reason = f"Kelly sizing: {safe_kelly_pct:.1%} of capital (full Kelly: {kelly_pct:.1%}, fractional: {self.kelly_fraction:.0%})"
        
        details = {
            'method': 'kelly',
            'total_trades': total_trades,
            'win_rate': win_rate,
            'avg_profit_pct': avg_profit_pct,
            'avg_loss_pct': avg_loss_pct,
            'win_loss_ratio': win_loss_ratio,
            'full_kelly_pct': kelly_pct,
            'fractional_kelly_pct': fractional_kelly_pct,
            'safe_kelly_pct': safe_kelly_pct,
            'position_pct': safe_kelly_pct,
            'position_size': position_size,
            'available_capital': available_capital
        }
        
        return position_size, reason, details
    
    def update_stats(
        self,
        current_stats: Dict[str, Any],
        new_trade_profit_pct: float
    ) -> Dict[str, Any]:
        """
        Update bot performance statistics after a completed trade.
        
        Args:
            current_stats: Current statistics dict
            new_trade_profit_pct: Profit/loss from completed trade (e.g., 0.05 = +5%)
        
        Returns:
            Updated statistics dict
        """
        total_trades = current_stats.get('total_trades', 0)
        winning_trades = current_stats.get('winning_trades', 0)
        total_profit_pct = current_stats.get('total_profit_pct', 0.0)
        total_loss_pct = current_stats.get('total_loss_pct', 0.0)
        
        # Update counts
        total_trades += 1
        if new_trade_profit_pct > 0:
            winning_trades += 1
            total_profit_pct += new_trade_profit_pct
        else:
            total_loss_pct += new_trade_profit_pct  # Negative value
        
        # Calculate averages
        losing_trades = total_trades - winning_trades
        avg_profit_pct = total_profit_pct / winning_trades if winning_trades > 0 else 0.0
        avg_loss_pct = total_loss_pct / losing_trades if losing_trades > 0 else 0.0
        
        return {
            'total_trades': total_trades,
            'winning_trades': winning_trades,
            'losing_trades': losing_trades,
            'win_rate': winning_trades / total_trades if total_trades > 0 else 0.0,
            'avg_profit_pct': avg_profit_pct,
            'avg_loss_pct': avg_loss_pct,
            'total_profit_pct': total_profit_pct,
            'total_loss_pct': total_loss_pct
        }


class DynamicPositionSizer:
    """
    Dynamic position sizing that adapts to market conditions.
    Combines multiple factors:
    1. Kelly Criterion (base sizing)
    2. Volatility adjustment (reduce size in high volatility)
    3. Drawdown protection (reduce size during drawdown)
    4. Confidence scaling (increase size with high-confidence signals)
    """
    
    def __init__(
        self,
        kelly_sizer: Optional[KellyPositionSizer] = None,
        volatility_mult_high: float = 0.5,  # Reduce to 50% in high volatility
        drawdown_threshold: float = 0.10,   # 10% drawdown triggers reduction
        drawdown_mult: float = 0.5          # Reduce to 50% during drawdown
    ):
        self.kelly_sizer = kelly_sizer or KellyPositionSizer()
        self.volatility_mult_high = volatility_mult_high
        self.drawdown_threshold = drawdown_threshold
        self.drawdown_mult = drawdown_mult
    
    def calculate_position_size(
        self,
        available_capital: float,
        bot_stats: Dict[str, Any],
        market_data: Dict[str, Any]
    ) -> Tuple[float, str, Dict[str, Any]]:
        """
        Calculate dynamic position size with multiple adjustments.
        
        Args:
            available_capital: Available capital (USD)
            bot_stats: Bot performance statistics
            market_data: Dict with keys:
                - volatility_pct: Current volatility (ATR %)
                - signal_confidence: 0-1, higher = more confident
                - current_drawdown_pct: Current drawdown from peak
        
        Returns:
            (position_size_usd, reason, details)
        """
        # Start with Kelly sizing
        base_size, kelly_reason, kelly_details = self.kelly_sizer.calculate_position_size(
            available_capital,
            bot_stats
        )
        
        adjustments = []
        final_size = base_size
        
        # 1. Volatility adjustment
        volatility_pct = market_data.get('volatility_pct', 0.0)
        if volatility_pct > 0.10:  # >10% volatility
            volatility_mult = self.volatility_mult_high
            final_size *= volatility_mult
            adjustments.append(f"High volatility ({volatility_pct:.1%}): {volatility_mult:.0%} sizing")
        elif volatility_pct > 0.05:  # 5-10% volatility
            volatility_mult = 0.75
            final_size *= volatility_mult
            adjustments.append(f"Medium volatility ({volatility_pct:.1%}): {volatility_mult:.0%} sizing")
        
        # 2. Drawdown protection
        current_drawdown = market_data.get('current_drawdown_pct', 0.0)
        if current_drawdown > self.drawdown_threshold:
            drawdown_mult = self.drawdown_mult
            final_size *= drawdown_mult
            adjustments.append(f"Drawdown protection ({current_drawdown:.1%}): {drawdown_mult:.0%} sizing")
        
        # 3. Signal confidence scaling
        signal_confidence = market_data.get('signal_confidence', 0.5)
        if signal_confidence >= 0.8:  # Very high confidence
            confidence_mult = 1.2  # Increase by 20%
            final_size *= confidence_mult
            adjustments.append(f"High confidence ({signal_confidence:.0%}): {confidence_mult:.0%} sizing")
        elif signal_confidence < 0.6:  # Low confidence
            confidence_mult = 0.8  # Reduce by 20%
            final_size *= confidence_mult
            adjustments.append(f"Low confidence ({signal_confidence:.0%}): {confidence_mult:.0%} sizing")
        
        # Build reason string
        reason_parts = [kelly_reason]
        if adjustments:
            reason_parts.append(" | Adjustments: " + ", ".join(adjustments))
        reason = "".join(reason_parts)
        
        details = {
            **kelly_details,
            'base_size': base_size,
            'final_size': final_size,
            'adjustments': adjustments,
            'volatility_pct': volatility_pct,
            'current_drawdown_pct': current_drawdown,
            'signal_confidence': signal_confidence
        }
        
        return final_size, reason, details


# =============================================================================
# Testing / Example Usage
# =============================================================================

if __name__ == "__main__":
    # Example: Bot with good performance
    bot_stats = {
        'total_trades': 50,
        'winning_trades': 32,  # 64% win rate
        'avg_profit_pct': 0.05,  # 5% average win
        'avg_loss_pct': -0.03   # -3% average loss
    }
    
    available_capital = 1000.0  # $1000
    
    # Standard Kelly sizing
    kelly_sizer = KellyPositionSizer(kelly_fraction=0.25, max_position_pct=0.10)
    size, reason, details = kelly_sizer.calculate_position_size(available_capital, bot_stats)
    
    print("=== Standard Kelly Sizing ===")
    print(f"Position Size: ${size:.2f}")
    print(f"Reason: {reason}")
    print(f"Details: {details}")
    print()
    
    # Dynamic sizing with market conditions
    market_data = {
        'volatility_pct': 0.08,  # 8% volatility (medium-high)
        'signal_confidence': 0.85,  # 85% confidence (high)
        'current_drawdown_pct': 0.05  # 5% drawdown (below threshold)
    }
    
    dynamic_sizer = DynamicPositionSizer()
    dyn_size, dyn_reason, dyn_details = dynamic_sizer.calculate_position_size(
        available_capital,
        bot_stats,
        market_data
    )
    
    print("=== Dynamic Sizing (with adjustments) ===")
    print(f"Position Size: ${dyn_size:.2f}")
    print(f"Reason: {dyn_reason}")
    print(f"Base Size: ${dyn_details['base_size']:.2f}")
    print(f"Final Size: ${dyn_details['final_size']:.2f}")
    print(f"Adjustments: {dyn_details['adjustments']}")
