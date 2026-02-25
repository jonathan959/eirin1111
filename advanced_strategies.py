"""
Advanced Trading Strategies - Extended strategy arsenal for autonomous trading.

New strategies based on: "Upgrading to an Autonomous Multi-Asset Trading Bot" specification.

Includes:
- Volatility Breakout Strategy
- Enhanced Grid with ATR-based spacing
- Earnings/News Momentum Strategy
- Volume-Confirmed Breakout
- Enhanced Mean Reversion

These strategies integrate with the existing strategy framework and regime detection.
"""

import math
import time
import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any
from enum import Enum

logger = logging.getLogger(__name__)


@dataclass
class StrategySignal:
    """Signal from a strategy."""
    action: str  # "BUY", "SELL", "HOLD", "CLOSE"
    confidence: float  # 0-1
    size_multiplier: float  # 1.0 = normal, <1 = reduced, >1 = increased
    entry_price: Optional[float] = None
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    trailing_stop_pct: Optional[float] = None
    reasons: List[str] = field(default_factory=list)
    strategy_name: str = ""
    regime: str = ""


@dataclass
class MarketContext:
    """Context for strategy decision making."""
    symbol: str
    current_price: float
    candles_1h: List[List[float]]  # [ts, o, h, l, c, v]
    candles_4h: List[List[float]]
    candles_1d: List[List[float]]
    
    # Indicators (pre-calculated)
    atr_14: float
    atr_pct: float  # ATR as % of price
    rsi_14: float
    macd: Tuple[float, float, float]  # line, signal, histogram
    bb_upper: float
    bb_middle: float
    bb_lower: float
    bb_width: float  # Bandwidth as %
    
    # Trend indicators
    sma_20: float
    sma_50: float
    sma_200: float
    ema_9: float
    ema_21: float
    
    # Volume
    volume_sma_20: float
    current_volume: float
    volume_ratio: float  # Current / SMA
    
    # Regime
    regime: str  # "BULL", "BEAR", "RANGE", "HIGH_VOL", "BREAKOUT"
    regime_confidence: float
    
    # External data (optional)
    sentiment_score: Optional[float] = None  # -1 to 1
    news_impact: Optional[str] = None  # "positive", "negative", "neutral"
    earnings_soon: bool = False
    
    # Market type
    market_type: str = "crypto"  # "crypto" or "stock"
    market_open: bool = True


def calculate_atr(candles: List[List[float]], period: int = 14) -> float:
    """Calculate Average True Range."""
    if len(candles) < period + 1:
        return 0.0
    
    true_ranges = []
    for i in range(1, len(candles)):
        high = candles[i][2]
        low = candles[i][3]
        prev_close = candles[i-1][4]
        
        tr = max(
            high - low,
            abs(high - prev_close),
            abs(low - prev_close)
        )
        true_ranges.append(tr)
    
    if len(true_ranges) < period:
        return sum(true_ranges) / len(true_ranges) if true_ranges else 0.0
    
    return sum(true_ranges[-period:]) / period


def calculate_rsi(candles: List[List[float]], period: int = 14) -> float:
    """Calculate RSI."""
    if len(candles) < period + 1:
        return 50.0
    
    gains = []
    losses = []
    
    for i in range(1, len(candles)):
        change = candles[i][4] - candles[i-1][4]
        if change > 0:
            gains.append(change)
            losses.append(0)
        else:
            gains.append(0)
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


def calculate_bollinger_bands(candles: List[List[float]], period: int = 20, std_dev: float = 2.0) -> Tuple[float, float, float, float]:
    """Calculate Bollinger Bands. Returns (upper, middle, lower, width_pct)."""
    if len(candles) < period:
        close = candles[-1][4] if candles else 0
        return close * 1.02, close, close * 0.98, 0.04
    
    closes = [c[4] for c in candles[-period:]]
    middle = sum(closes) / period
    
    variance = sum((c - middle)**2 for c in closes) / period
    std = math.sqrt(variance)
    
    upper = middle + std_dev * std
    lower = middle - std_dev * std
    width_pct = (upper - lower) / middle if middle > 0 else 0
    
    return upper, middle, lower, width_pct


def calculate_sma(candles: List[List[float]], period: int) -> float:
    """Calculate Simple Moving Average."""
    if len(candles) < period:
        return candles[-1][4] if candles else 0
    
    closes = [c[4] for c in candles[-period:]]
    return sum(closes) / period


def calculate_ema(candles: List[List[float]], period: int) -> float:
    """Calculate Exponential Moving Average."""
    if len(candles) < period:
        return candles[-1][4] if candles else 0
    
    closes = [c[4] for c in candles]
    multiplier = 2 / (period + 1)
    ema = closes[0]
    
    for close in closes[1:]:
        ema = (close - ema) * multiplier + ema
    
    return ema


def calculate_macd(candles: List[List[float]]) -> Tuple[float, float, float]:
    """Calculate MACD. Returns (macd_line, signal_line, histogram)."""
    if len(candles) < 26:
        return 0, 0, 0
    
    ema_12 = calculate_ema(candles, 12)
    ema_26 = calculate_ema(candles, 26)
    macd_line = ema_12 - ema_26
    
    # For signal line, we need historical MACD values
    # Simplified: use current values
    signal_line = macd_line * 0.9  # Approximation
    histogram = macd_line - signal_line
    
    return macd_line, signal_line, histogram


class VolatilityBreakoutStrategy:
    """
    Volatility Breakout Strategy.
    
    Triggers when:
    - Price explodes out of tight ranges
    - Bollinger Band width contracts then expands
    - ATR spikes above normal threshold
    
    Best for: High volatility regimes, post-consolidation breakouts.
    """
    
    def __init__(self, 
                 bb_squeeze_threshold: float = 0.02,  # BB width below this = squeeze
                 bb_expansion_mult: float = 1.5,  # BB width must expand by this factor
                 atr_spike_mult: float = 1.3,  # ATR must be above this * avg
                 volume_confirm_mult: float = 1.5):  # Volume must be above this * avg
        self.bb_squeeze_threshold = bb_squeeze_threshold
        self.bb_expansion_mult = bb_expansion_mult
        self.atr_spike_mult = atr_spike_mult
        self.volume_confirm_mult = volume_confirm_mult
        self._last_bb_width = None
    
    def evaluate(self, ctx: MarketContext) -> StrategySignal:
        """Evaluate volatility breakout conditions."""
        signal = StrategySignal(
            action="HOLD",
            confidence=0.0,
            size_multiplier=1.0,
            strategy_name="volatility_breakout",
            regime=ctx.regime
        )
        
        # Check for Bollinger Band squeeze followed by expansion
        current_bb_width = ctx.bb_width
        was_squeezed = self._last_bb_width and self._last_bb_width < self.bb_squeeze_threshold
        is_expanding = self._last_bb_width and current_bb_width > self._last_bb_width * self.bb_expansion_mult
        
        self._last_bb_width = current_bb_width
        
        if not (was_squeezed and is_expanding):
            signal.reasons.append("No BB squeeze/expansion detected")
            return signal
        
        # Check ATR spike
        if ctx.atr_pct < 0.02 * self.atr_spike_mult:  # Below spike threshold
            signal.reasons.append("ATR not spiking")
            return signal
        
        # Check volume confirmation
        if ctx.volume_ratio < self.volume_confirm_mult:
            signal.reasons.append("Volume not confirming")
            return signal
        
        # Determine direction
        price = ctx.current_price
        
        if price > ctx.bb_upper:
            # Breakout to upside
            signal.action = "BUY"
            signal.confidence = min(0.9, 0.5 + ctx.volume_ratio * 0.1 + (ctx.atr_pct * 10))
            signal.stop_loss = ctx.bb_middle  # Stop at middle band
            signal.take_profit = price + 2 * ctx.atr_14  # 2 ATR target
            signal.trailing_stop_pct = ctx.atr_pct * 1.5  # Trail by 1.5 ATR
            signal.reasons = [
                f"Volatility breakout UP",
                f"BB expansion from squeeze ({self._last_bb_width:.2%} -> {current_bb_width:.2%})",
                f"Volume ratio: {ctx.volume_ratio:.1f}x",
                f"ATR: {ctx.atr_pct:.2%}"
            ]
        
        elif price < ctx.bb_lower:
            # Breakout to downside (short signal or avoid longs)
            signal.action = "SELL"  # Or "AVOID" for long-only
            signal.confidence = min(0.85, 0.4 + ctx.volume_ratio * 0.1)
            signal.stop_loss = ctx.bb_middle
            signal.reasons = [
                f"Volatility breakout DOWN",
                f"BB expansion from squeeze",
                f"Consider short or avoid longs"
            ]
        
        # Reduce size in extreme volatility
        if ctx.atr_pct > 0.08:  # Very high volatility
            signal.size_multiplier = 0.5
            signal.reasons.append("Reduced size due to extreme volatility")
        
        return signal


class EnhancedGridStrategy:
    """
    Enhanced Grid Trading with ATR-based spacing.
    
    Features:
    - Dynamic grid spacing based on ATR
    - Volatility-adjusted order sizing
    - Range detection for optimal deployment
    
    Best for: Range-bound markets, sideways consolidation.
    """
    
    def __init__(self,
                 grid_levels: int = 5,
                 atr_spacing_mult: float = 1.0,  # Grid step = ATR * this
                 min_range_atr: float = 3.0,  # Min range in ATR units
                 max_range_atr: float = 10.0):  # Max range in ATR units
        self.grid_levels = grid_levels
        self.atr_spacing_mult = atr_spacing_mult
        self.min_range_atr = min_range_atr
        self.max_range_atr = max_range_atr
    
    def evaluate(self, ctx: MarketContext) -> StrategySignal:
        """Evaluate grid trading conditions."""
        signal = StrategySignal(
            action="HOLD",
            confidence=0.0,
            size_multiplier=1.0,
            strategy_name="enhanced_grid",
            regime=ctx.regime
        )
        
        # Grid works best in RANGE regime
        if ctx.regime not in ("RANGE", "HIGH_VOL_DEFENSIVE"):
            signal.reasons.append(f"Regime {ctx.regime} not optimal for grid")
            signal.confidence = 0.3
            return signal
        
        # Calculate grid parameters
        atr = ctx.atr_14
        grid_step = atr * self.atr_spacing_mult
        price = ctx.current_price
        
        # Determine grid range
        range_size = (ctx.bb_upper - ctx.bb_lower)
        range_atr = range_size / atr if atr > 0 else 5
        
        if range_atr < self.min_range_atr:
            signal.reasons.append(f"Range too tight ({range_atr:.1f} ATR)")
            return signal
        
        if range_atr > self.max_range_atr:
            signal.reasons.append(f"Range too wide ({range_atr:.1f} ATR)")
            signal.size_multiplier = 0.7  # Can still trade but smaller
        
        # Determine entry signal based on position in range
        range_position = (price - ctx.bb_lower) / range_size if range_size > 0 else 0.5
        
        if range_position < 0.3:
            # Near bottom of range - buy zone
            signal.action = "BUY"
            signal.confidence = 0.6 + (0.3 - range_position)  # Higher confidence near bottom
            signal.entry_price = price
            signal.stop_loss = ctx.bb_lower - grid_step
            signal.take_profit = ctx.bb_middle  # Target middle
            signal.reasons = [
                f"Grid BUY near range bottom",
                f"Range position: {range_position:.0%}",
                f"Grid step: ${grid_step:.2f} ({ctx.atr_pct:.2%} ATR)",
                f"Target: ${ctx.bb_middle:.2f}"
            ]
        
        elif range_position > 0.7:
            # Near top of range - sell zone
            signal.action = "SELL"
            signal.confidence = 0.6 + (range_position - 0.7)
            signal.entry_price = price
            signal.stop_loss = ctx.bb_upper + grid_step
            signal.take_profit = ctx.bb_middle
            signal.reasons = [
                f"Grid SELL near range top",
                f"Range position: {range_position:.0%}",
                f"Grid step: ${grid_step:.2f}"
            ]
        
        else:
            # Middle of range - hold
            signal.action = "HOLD"
            signal.confidence = 0.4
            signal.reasons = [
                f"Middle of range ({range_position:.0%})",
                "Wait for better entry"
            ]
        
        # Adjust size based on volatility
        if ctx.atr_pct > 0.05:
            signal.size_multiplier *= 0.7
            signal.reasons.append("Reduced size due to high volatility")
        
        return signal
    
    def get_grid_levels(self, ctx: MarketContext, base_amount: float) -> List[Dict[str, Any]]:
        """
        Generate grid order levels.
        
        Returns list of orders to place.
        """
        atr = ctx.atr_14
        grid_step = atr * self.atr_spacing_mult
        price = ctx.current_price
        
        orders = []
        amount_per_level = base_amount / self.grid_levels
        
        for i in range(1, self.grid_levels + 1):
            # Buy orders below current price
            buy_price = price - (i * grid_step)
            orders.append({
                "side": "buy",
                "price": round(buy_price, 2),
                "amount": amount_per_level,
                "level": i
            })
            
            # Sell orders above current price
            sell_price = price + (i * grid_step)
            orders.append({
                "side": "sell",
                "price": round(sell_price, 2),
                "amount": amount_per_level,
                "level": i
            })
        
        return orders


class VolumeConfirmedBreakoutStrategy:
    """
    Volume-Confirmed Breakout Strategy.
    
    Only enters breakouts when volume confirms the move.
    Reduces false breakout risk.
    
    Best for: Trending markets, key level breaks.
    """
    
    def __init__(self,
                 lookback_periods: int = 20,
                 volume_threshold: float = 2.0,  # Volume must be 2x average
                 breakout_threshold_atr: float = 1.0):  # Price must break by 1 ATR
        self.lookback_periods = lookback_periods
        self.volume_threshold = volume_threshold
        self.breakout_threshold_atr = breakout_threshold_atr
        self._recent_high = None
        self._recent_low = None
    
    def evaluate(self, ctx: MarketContext) -> StrategySignal:
        """Evaluate volume-confirmed breakout conditions."""
        signal = StrategySignal(
            action="HOLD",
            confidence=0.0,
            size_multiplier=1.0,
            strategy_name="volume_confirmed_breakout",
            regime=ctx.regime
        )
        
        candles = ctx.candles_1d if ctx.candles_1d else ctx.candles_4h
        if len(candles) < self.lookback_periods:
            signal.reasons.append("Insufficient data")
            return signal
        
        # Calculate recent high/low (excluding most recent candle)
        lookback = candles[-self.lookback_periods-1:-1]
        recent_high = max(c[2] for c in lookback)  # Highest high
        recent_low = min(c[3] for c in lookback)  # Lowest low
        
        self._recent_high = recent_high
        self._recent_low = recent_low
        
        price = ctx.current_price
        atr = ctx.atr_14
        
        # Check for breakout
        upside_breakout = price > recent_high + (atr * self.breakout_threshold_atr)
        downside_breakout = price < recent_low - (atr * self.breakout_threshold_atr)
        
        if not (upside_breakout or downside_breakout):
            signal.reasons.append(f"No breakout (range: ${recent_low:.2f} - ${recent_high:.2f})")
            return signal
        
        # Check volume confirmation
        if ctx.volume_ratio < self.volume_threshold:
            signal.reasons.append(f"Volume not confirming ({ctx.volume_ratio:.1f}x vs {self.volume_threshold}x required)")
            signal.confidence = 0.3  # Low confidence without volume
            return signal
        
        if upside_breakout:
            signal.action = "BUY"
            signal.confidence = min(0.9, 0.5 + (ctx.volume_ratio - self.volume_threshold) * 0.1)
            signal.entry_price = price
            signal.stop_loss = recent_high - atr  # Stop just below breakout level
            signal.take_profit = price + 3 * atr  # 3 ATR target
            signal.trailing_stop_pct = ctx.atr_pct * 2  # Wide trail for trend
            signal.reasons = [
                f"Volume-confirmed breakout ABOVE ${recent_high:.2f}",
                f"Volume: {ctx.volume_ratio:.1f}x average",
                f"Breakout strength: {(price - recent_high) / atr:.1f} ATR",
                f"Target: ${signal.take_profit:.2f}, Stop: ${signal.stop_loss:.2f}"
            ]
        
        elif downside_breakout:
            signal.action = "SELL"  # Short signal
            signal.confidence = min(0.85, 0.4 + (ctx.volume_ratio - self.volume_threshold) * 0.1)
            signal.entry_price = price
            signal.stop_loss = recent_low + atr
            signal.reasons = [
                f"Volume-confirmed breakdown BELOW ${recent_low:.2f}",
                f"Volume: {ctx.volume_ratio:.1f}x average",
                "Consider short or exit longs"
            ]
        
        return signal


class TrendMomentumStrategy:
    """
    Enhanced Trend-Following Momentum Strategy.
    
    Features:
    - Multi-timeframe trend confirmation
    - Momentum indicators (MACD, RSI)
    - Pyramiding into winners
    - Wide trailing stops
    
    Best for: Strong trending markets (BULL or BEAR regime).
    """
    
    def __init__(self,
                 trend_ema_fast: int = 9,
                 trend_ema_slow: int = 21,
                 trend_sma_filter: int = 50,
                 rsi_overbought: float = 70,
                 rsi_oversold: float = 30):
        self.trend_ema_fast = trend_ema_fast
        self.trend_ema_slow = trend_ema_slow
        self.trend_sma_filter = trend_sma_filter
        self.rsi_overbought = rsi_overbought
        self.rsi_oversold = rsi_oversold
    
    def evaluate(self, ctx: MarketContext) -> StrategySignal:
        """Evaluate trend momentum conditions."""
        signal = StrategySignal(
            action="HOLD",
            confidence=0.0,
            size_multiplier=1.0,
            strategy_name="trend_momentum",
            regime=ctx.regime
        )
        
        # Check regime
        if ctx.regime not in ("BULL", "BEAR", "BREAKOUT"):
            signal.reasons.append(f"Regime {ctx.regime} not trending")
            return signal
        
        price = ctx.current_price
        
        # Trend filters
        above_sma50 = price > ctx.sma_50
        above_sma200 = price > ctx.sma_200
        ema_bullish = ctx.ema_9 > ctx.ema_21
        macd_bullish = ctx.macd[2] > 0  # Histogram positive
        
        # Strong uptrend
        if above_sma50 and above_sma200 and ema_bullish and macd_bullish:
            if ctx.rsi_14 > self.rsi_overbought:
                # Overbought in uptrend - wait for pullback
                signal.action = "HOLD"
                signal.confidence = 0.5
                signal.reasons = [
                    "Strong uptrend but RSI overbought",
                    f"RSI: {ctx.rsi_14:.0f}",
                    "Wait for pullback to add"
                ]
            else:
                # Buy the trend
                signal.action = "BUY"
                signal.confidence = 0.7 + (ctx.regime_confidence * 0.2)
                signal.entry_price = price
                signal.stop_loss = ctx.sma_50  # Stop at 50 SMA
                signal.take_profit = price * 1.15  # 15% target
                signal.trailing_stop_pct = ctx.atr_pct * 2.5  # Wide trail
                signal.reasons = [
                    "Strong uptrend confirmed",
                    f"Price > SMA50 > SMA200",
                    f"EMA9 > EMA21, MACD positive",
                    f"RSI: {ctx.rsi_14:.0f}",
                    f"Regime: {ctx.regime} ({ctx.regime_confidence:.0%})"
                ]
        
        # Strong downtrend
        elif not above_sma50 and not above_sma200 and not ema_bullish and not macd_bullish:
            signal.action = "SELL"
            signal.confidence = 0.6
            signal.reasons = [
                "Strong downtrend",
                "Price < SMA50 < SMA200",
                "Consider short or exit longs"
            ]
        
        # Pullback in uptrend - buying opportunity
        elif above_sma200 and not ema_bullish and ctx.rsi_14 < 40:
            signal.action = "BUY"
            signal.confidence = 0.65
            signal.entry_price = price
            signal.stop_loss = ctx.sma_200 * 0.98
            signal.reasons = [
                "Pullback in larger uptrend",
                f"Price above SMA200, RSI oversold ({ctx.rsi_14:.0f})",
                "Potential bounce entry"
            ]
        
        return signal


class NewsMomentumStrategy:
    """
    Earnings & News Momentum Strategy.
    
    Reacts to market-moving news/events.
    Short-term momentum plays.
    
    Best for: Event-driven trading, earnings releases.
    """
    
    def __init__(self,
                 min_gap_pct: float = 0.02,  # 2% minimum gap
                 volume_surge_mult: float = 3.0,  # Volume must be 3x
                 hold_periods: int = 6):  # Hold for ~6 hours
        self.min_gap_pct = min_gap_pct
        self.volume_surge_mult = volume_surge_mult
        self.hold_periods = hold_periods
    
    def evaluate(self, ctx: MarketContext) -> StrategySignal:
        """Evaluate news momentum conditions."""
        signal = StrategySignal(
            action="HOLD",
            confidence=0.0,
            size_multiplier=0.7,  # Smaller size for event trades
            strategy_name="news_momentum",
            regime=ctx.regime
        )
        
        # Skip if earnings coming (wait for result)
        if ctx.earnings_soon:
            signal.reasons.append("Earnings imminent - wait for result")
            return signal
        
        # Check for gap (comparing to previous close)
        candles = ctx.candles_1h
        if len(candles) < 2:
            signal.reasons.append("Insufficient data")
            return signal
        
        prev_close = candles[-2][4]
        current_open = candles[-1][1]
        gap_pct = (current_open - prev_close) / prev_close if prev_close > 0 else 0
        
        # Check for significant gap
        if abs(gap_pct) < self.min_gap_pct:
            signal.reasons.append(f"No significant gap ({gap_pct:.2%})")
            return signal
        
        # Check for volume surge
        if ctx.volume_ratio < self.volume_surge_mult:
            signal.reasons.append(f"Volume not surging ({ctx.volume_ratio:.1f}x)")
            return signal
        
        # Check sentiment if available
        sentiment_aligned = True
        if ctx.sentiment_score is not None:
            sentiment_aligned = (
                (gap_pct > 0 and ctx.sentiment_score > 0.2) or
                (gap_pct < 0 and ctx.sentiment_score < -0.2)
            )
        
        if not sentiment_aligned:
            signal.reasons.append("Sentiment not aligned with gap direction")
            signal.confidence = 0.3
            return signal
        
        price = ctx.current_price
        atr = ctx.atr_14
        
        if gap_pct > 0:
            # Gap up - momentum long
            signal.action = "BUY"
            signal.confidence = min(0.85, 0.5 + abs(gap_pct) * 5 + ctx.volume_ratio * 0.05)
            signal.entry_price = price
            signal.stop_loss = current_open - atr  # Stop below gap
            signal.take_profit = price + 2 * atr  # Quick target
            signal.reasons = [
                f"Bullish gap: {gap_pct:.1%}",
                f"Volume surge: {ctx.volume_ratio:.1f}x",
                f"News impact: {ctx.news_impact or 'unknown'}",
                "Momentum play - quick exit"
            ]
        else:
            # Gap down - avoid or short
            signal.action = "SELL"
            signal.confidence = min(0.75, 0.4 + abs(gap_pct) * 5)
            signal.reasons = [
                f"Bearish gap: {gap_pct:.1%}",
                f"Volume: {ctx.volume_ratio:.1f}x",
                "Consider short or avoid longs"
            ]
        
        return signal


class EnhancedMeanReversionStrategy:
    """
    Enhanced Mean Reversion Strategy.
    
    Fades extreme moves in range-bound markets.
    Uses Bollinger Bands + RSI + volume.
    
    Best for: Range-bound markets, oversold/overbought conditions.
    """
    
    def __init__(self,
                 rsi_oversold: float = 25,
                 rsi_overbought: float = 75,
                 bb_touch_threshold: float = 0.02,  # Within 2% of band
                 min_reversion_target_atr: float = 1.0):
        self.rsi_oversold = rsi_oversold
        self.rsi_overbought = rsi_overbought
        self.bb_touch_threshold = bb_touch_threshold
        self.min_reversion_target_atr = min_reversion_target_atr
    
    def evaluate(self, ctx: MarketContext) -> StrategySignal:
        """Evaluate mean reversion conditions."""
        signal = StrategySignal(
            action="HOLD",
            confidence=0.0,
            size_multiplier=1.0,
            strategy_name="enhanced_mean_reversion",
            regime=ctx.regime
        )
        
        # Best in range regime
        if ctx.regime not in ("RANGE", "HIGH_VOL_DEFENSIVE"):
            signal.confidence = 0.3
            signal.reasons.append(f"Regime {ctx.regime} less suitable for mean reversion")
        
        price = ctx.current_price
        atr = ctx.atr_14
        
        # Check proximity to bands
        upper_dist = (ctx.bb_upper - price) / price if price > 0 else 1
        lower_dist = (price - ctx.bb_lower) / price if price > 0 else 1
        
        near_upper = upper_dist < self.bb_touch_threshold
        near_lower = lower_dist < self.bb_touch_threshold
        
        # Oversold at lower band
        if near_lower and ctx.rsi_14 < self.rsi_oversold:
            # Confirm with volume (low volume = exhaustion)
            volume_exhaustion = ctx.volume_ratio < 0.8
            
            signal.action = "BUY"
            signal.confidence = 0.7 if volume_exhaustion else 0.55
            signal.entry_price = price
            signal.stop_loss = ctx.bb_lower - atr  # Stop below band
            signal.take_profit = ctx.bb_middle  # Target middle band
            signal.reasons = [
                f"Oversold at lower BB (RSI: {ctx.rsi_14:.0f})",
                f"Distance from lower band: {lower_dist:.1%}",
                f"Target: middle band (${ctx.bb_middle:.2f})",
                f"Volume exhaustion: {volume_exhaustion}"
            ]
        
        # Overbought at upper band
        elif near_upper and ctx.rsi_14 > self.rsi_overbought:
            volume_exhaustion = ctx.volume_ratio < 0.8
            
            signal.action = "SELL"
            signal.confidence = 0.65 if volume_exhaustion else 0.5
            signal.entry_price = price
            signal.stop_loss = ctx.bb_upper + atr
            signal.take_profit = ctx.bb_middle
            signal.reasons = [
                f"Overbought at upper BB (RSI: {ctx.rsi_14:.0f})",
                f"Distance from upper band: {upper_dist:.1%}",
                "Consider taking profits or shorting"
            ]
        
        else:
            signal.reasons.append(f"No extreme conditions (RSI: {ctx.rsi_14:.0f})")
        
        return signal


class StrategyEnsemble:
    """
    Combines multiple strategies and ML signals.
    
    Only takes trades when multiple strategies agree.
    """
    
    def __init__(self):
        self.strategies = {
            "volatility_breakout": VolatilityBreakoutStrategy(),
            "enhanced_grid": EnhancedGridStrategy(),
            "volume_breakout": VolumeConfirmedBreakoutStrategy(),
            "trend_momentum": TrendMomentumStrategy(),
            "news_momentum": NewsMomentumStrategy(),
            "mean_reversion": EnhancedMeanReversionStrategy()
        }
        
        # Strategy weights by regime
        self.regime_weights = {
            "BULL": {
                "trend_momentum": 1.5,
                "volume_breakout": 1.2,
                "news_momentum": 1.0,
                "mean_reversion": 0.5,
                "volatility_breakout": 0.8,
                "enhanced_grid": 0.3
            },
            "BEAR": {
                "trend_momentum": 1.2,
                "mean_reversion": 0.7,
                "news_momentum": 0.8,
                "volatility_breakout": 0.6,
                "volume_breakout": 0.8,
                "enhanced_grid": 0.3
            },
            "RANGE": {
                "enhanced_grid": 1.5,
                "mean_reversion": 1.3,
                "volatility_breakout": 0.5,
                "trend_momentum": 0.4,
                "volume_breakout": 0.6,
                "news_momentum": 0.7
            },
            "HIGH_VOL": {
                "volatility_breakout": 1.5,
                "enhanced_grid": 0.8,
                "mean_reversion": 0.6,
                "trend_momentum": 0.7,
                "volume_breakout": 1.0,
                "news_momentum": 1.2
            },
            "BREAKOUT": {
                "volatility_breakout": 1.5,
                "volume_breakout": 1.5,
                "trend_momentum": 1.2,
                "news_momentum": 1.0,
                "mean_reversion": 0.2,
                "enhanced_grid": 0.3
            }
        }
    
    def evaluate_all(self, ctx: MarketContext, ml_signal: Optional[Dict] = None) -> Dict[str, Any]:
        """
        Evaluate all strategies and combine signals.
        
        Args:
            ctx: Market context
            ml_signal: Optional ML prediction {"direction": "UP"/"DOWN", "confidence": 0-1}
        
        Returns:
            Combined signal with ensemble confidence.
        """
        signals = {}
        weights = self.regime_weights.get(ctx.regime, self.regime_weights["RANGE"])
        
        # Evaluate each strategy
        for name, strategy in self.strategies.items():
            try:
                signal = strategy.evaluate(ctx)
                signal.strategy_name = name
                signals[name] = signal
            except Exception as e:
                logger.error(f"Strategy {name} failed: {e}")
                signals[name] = StrategySignal(
                    action="HOLD",
                    confidence=0,
                    size_multiplier=1.0,
                    strategy_name=name,
                    reasons=[f"Error: {e}"]
                )
        
        # Combine signals
        buy_score = 0.0
        sell_score = 0.0
        hold_score = 0.0
        total_weight = 0.0
        
        reasons = []
        
        for name, signal in signals.items():
            weight = weights.get(name, 1.0)
            weighted_conf = signal.confidence * weight
            total_weight += weight
            
            if signal.action == "BUY":
                buy_score += weighted_conf
                reasons.append(f"{name}: BUY ({signal.confidence:.0%})")
            elif signal.action == "SELL":
                sell_score += weighted_conf
                reasons.append(f"{name}: SELL ({signal.confidence:.0%})")
            else:
                hold_score += weighted_conf * 0.5  # Hold gets half weight
        
        # Normalize
        if total_weight > 0:
            buy_score /= total_weight
            sell_score /= total_weight
            hold_score /= total_weight
        
        # Include ML signal
        ml_adjustment = 0.0
        if ml_signal:
            ml_conf = ml_signal.get("confidence", 0.5)
            ml_dir = ml_signal.get("direction", "NEUTRAL")
            
            if ml_dir == "UP":
                buy_score += ml_conf * 0.3  # ML contributes 30%
                ml_adjustment = ml_conf * 0.3
                reasons.append(f"ML: UP ({ml_conf:.0%})")
            elif ml_dir == "DOWN":
                sell_score += ml_conf * 0.3
                ml_adjustment = -ml_conf * 0.3
                reasons.append(f"ML: DOWN ({ml_conf:.0%})")
        
        # Determine final action
        if buy_score > sell_score and buy_score > 0.5:
            action = "BUY"
            confidence = buy_score
        elif sell_score > buy_score and sell_score > 0.5:
            action = "SELL"
            confidence = sell_score
        else:
            action = "HOLD"
            confidence = max(hold_score, 1 - max(buy_score, sell_score))
        
        # Get best signal for parameters
        best_signal = None
        best_conf = 0
        for name, signal in signals.items():
            if signal.action == action and signal.confidence > best_conf:
                best_signal = signal
                best_conf = signal.confidence
        
        return {
            "action": action,
            "confidence": min(confidence, 0.95),
            "buy_score": buy_score,
            "sell_score": sell_score,
            "ml_adjustment": ml_adjustment,
            "entry_price": best_signal.entry_price if best_signal else None,
            "stop_loss": best_signal.stop_loss if best_signal else None,
            "take_profit": best_signal.take_profit if best_signal else None,
            "trailing_stop_pct": best_signal.trailing_stop_pct if best_signal else None,
            "size_multiplier": best_signal.size_multiplier if best_signal else 1.0,
            "reasons": reasons,
            "strategy_signals": {k: {
                "action": v.action,
                "confidence": v.confidence,
                "reasons": v.reasons
            } for k, v in signals.items()},
            "regime": ctx.regime,
            "regime_confidence": ctx.regime_confidence
        }


# Create singleton ensemble
_strategy_ensemble: Optional[StrategyEnsemble] = None


def get_strategy_ensemble() -> StrategyEnsemble:
    """Get or create the strategy ensemble."""
    global _strategy_ensemble
    if _strategy_ensemble is None:
        _strategy_ensemble = StrategyEnsemble()
    return _strategy_ensemble
