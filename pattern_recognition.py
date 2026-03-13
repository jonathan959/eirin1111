"""
Candlestick Pattern Recognition Module for Trading Bot

This module detects technical analysis patterns in OHLCV candle data,
including flags, heads & shoulders, triangles, and single-candle reversals.
"""

import logging
from typing import List, Dict, Optional, Tuple
import numpy as np
from dataclasses import dataclass
from enum import Enum

# Configure logging
logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


class PatternType(Enum):
    """Enumeration for pattern types."""
    BULLISH = "bullish"
    BEARISH = "bearish"
    NEUTRAL = "neutral"


@dataclass
class Pattern:
    """Data class representing a detected pattern."""
    pattern: str
    type: str  # "bullish", "bearish", "neutral"
    confidence: float  # 0-1
    start_idx: int
    end_idx: int
    description: str


class PatternRecognizer:
    """
    Detects candlestick patterns in OHLCV data.

    Implements technical analysis patterns including flags, heads & shoulders,
    triangles, and reversal patterns.
    """

    def __init__(self, sensitivity: float = 1.0):
        """
        Initialize the pattern recognizer.

        Args:
            sensitivity: Float multiplier for pattern thresholds (default 1.0).
                        Lower values = stricter detection, higher = more lenient.
        """
        if sensitivity <= 0:
            raise ValueError("Sensitivity must be positive")

        self.sensitivity = sensitivity
        logger.debug(f"PatternRecognizer initialized with sensitivity={sensitivity}")

    # ==================== Helper Methods ====================

    def _body_size(self, candle: Dict) -> float:
        """Calculate absolute body size of a candle."""
        return abs(candle.get("close", 0) - candle.get("open", 0))

    def _is_bullish(self, candle: Dict) -> bool:
        """Check if candle is bullish (close > open)."""
        return candle.get("close", 0) > candle.get("open", 0)

    def _upper_wick(self, candle: Dict) -> float:
        """Calculate upper wick length."""
        body_top = max(candle.get("open", 0), candle.get("close", 0))
        return max(0, candle.get("high", 0) - body_top)

    def _lower_wick(self, candle: Dict) -> float:
        """Calculate lower wick length."""
        body_bottom = min(candle.get("open", 0), candle.get("close", 0))
        return max(0, body_bottom - candle.get("low", 0))

    def _trend_direction(self, candles: List[Dict], lookback: int = 10) -> str:
        """
        Determine trend direction using linear regression of closes.

        Args:
            candles: List of candle dicts
            lookback: Number of candles to analyze

        Returns:
            "up", "down", or "sideways"
        """
        if len(candles) < lookback:
            return "sideways"

        recent = candles[-lookback:]
        closes = np.array([c.get("close", 0) for c in recent])
        x = np.arange(len(closes))

        try:
            coefficients = np.polyfit(x, closes, 1)
            slope = coefficients[0]

            # Normalized slope
            avg_close = np.mean(closes)
            normalized_slope = slope / avg_close if avg_close != 0 else 0

            if normalized_slope > 0.001:
                return "up"
            elif normalized_slope < -0.001:
                return "down"
            else:
                return "sideways"
        except Exception as e:
            logger.warning(f"Error calculating trend direction: {e}")
            return "sideways"

    def _avg_volume(self, candles: List[Dict], period: int = 20) -> float:
        """Calculate average volume over period."""
        if len(candles) < period:
            recent_candles = candles
        else:
            recent_candles = candles[-period:]

        if not recent_candles:
            return 0

        volumes = [c.get("volume", 0) for c in recent_candles]
        avg = np.mean(volumes) if volumes else 0
        return float(avg)

    def _pct_change(self, start_price: float, end_price: float) -> float:
        """Calculate percentage change between two prices."""
        if start_price == 0:
            return 0
        return ((end_price - start_price) / start_price) * 100

    def _find_support_resistance(
        self, candles: List[Dict], window: int = 5
    ) -> Tuple[float, float]:
        """Find support and resistance levels using simple min/max."""
        if len(candles) < window:
            lows = [c.get("low", 0) for c in candles]
            highs = [c.get("high", 0) for c in candles]
        else:
            lows = [c.get("low", 0) for c in candles[-window:]]
            highs = [c.get("high", 0) for c in candles[-window:]]

        support = min(lows) if lows else 0
        resistance = max(highs) if highs else 0
        return support, resistance

    # ==================== Pattern Detection Methods ====================

    def _detect_bull_flag(self, candles: List[Dict]) -> List[Pattern]:
        """Detect bull flag patterns."""
        patterns = []
        min_pattern_length = 10  # 5 for pole + 5+ for flag

        if len(candles) < min_pattern_length:
            return patterns

        threshold_pole = 3.0 * self.sensitivity
        threshold_flag = 2.0 * self.sensitivity

        for i in range(len(candles) - min_pattern_length + 1):
            # Check for pole (strong upward move in ~5 candles)
            if i + 5 >= len(candles):
                break

            pole_candles = candles[i:i+5]
            pole_start = pole_candles[0].get("close", 0)
            pole_end = pole_candles[-1].get("close", 0)
            pole_pct = self._pct_change(pole_start, pole_end)

            if pole_pct < threshold_pole:
                continue

            # Look for flag consolidation after pole
            flag_start = i + 5
            if flag_start >= len(candles):
                break

            for flag_end in range(flag_start + 5, min(flag_start + 15, len(candles))):
                flag_candles = candles[flag_start:flag_end+1]
                flag_high = max(c.get("high", 0) for c in flag_candles)
                flag_low = min(c.get("low", 0) for c in flag_candles)
                flag_pct = self._pct_change(flag_low, flag_high)

                if flag_pct < threshold_flag:
                    # Check volume trend in flag (should decrease)
                    volumes = [c.get("volume", 0) for c in flag_candles]
                    if len(volumes) > 1:
                        vol_start = np.mean(volumes[:len(volumes)//2])
                        vol_end = np.mean(volumes[len(volumes)//2:])
                        vol_decrease = vol_start > vol_end
                    else:
                        vol_decrease = True

                    if vol_decrease:
                        # Validate trend is upward before flag
                        trend = self._trend_direction(candles[:flag_start])
                        if trend == "up":
                            avg_vol = self._avg_volume(candles, 20)
                            current_vol = candles[flag_end].get("volume", 0)
                            vol_confidence = min(1.0, current_vol / avg_vol / 2) if avg_vol > 0 else 0.6
                            confidence = 0.7 + (vol_confidence * 0.3)

                            patterns.append(Pattern(
                                pattern="Bull Flag",
                                type=PatternType.BULLISH.value,
                                confidence=min(1.0, confidence),
                                start_idx=i,
                                end_idx=flag_end,
                                description=f"Strong uptrend ({pole_pct:.1f}% pole) with {flag_pct:.1f}% consolidation"
                            ))

        return patterns

    def _detect_bear_flag(self, candles: List[Dict]) -> List[Pattern]:
        """Detect bear flag patterns (inverse of bull flag)."""
        patterns = []
        min_pattern_length = 10

        if len(candles) < min_pattern_length:
            return patterns

        threshold_pole = -3.0 * self.sensitivity
        threshold_flag = 2.0 * self.sensitivity

        for i in range(len(candles) - min_pattern_length + 1):
            if i + 5 >= len(candles):
                break

            pole_candles = candles[i:i+5]
            pole_start = pole_candles[0].get("close", 0)
            pole_end = pole_candles[-1].get("close", 0)
            pole_pct = self._pct_change(pole_start, pole_end)

            if pole_pct > threshold_pole:
                continue

            flag_start = i + 5
            if flag_start >= len(candles):
                break

            for flag_end in range(flag_start + 5, min(flag_start + 15, len(candles))):
                flag_candles = candles[flag_start:flag_end+1]
                flag_high = max(c.get("high", 0) for c in flag_candles)
                flag_low = min(c.get("low", 0) for c in flag_candles)
                flag_pct = self._pct_change(flag_low, flag_high)

                if flag_pct < threshold_flag:
                    volumes = [c.get("volume", 0) for c in flag_candles]
                    if len(volumes) > 1:
                        vol_start = np.mean(volumes[:len(volumes)//2])
                        vol_end = np.mean(volumes[len(volumes)//2:])
                        vol_decrease = vol_start > vol_end
                    else:
                        vol_decrease = True

                    if vol_decrease:
                        trend = self._trend_direction(candles[:flag_start])
                        if trend == "down":
                            avg_vol = self._avg_volume(candles, 20)
                            current_vol = candles[flag_end].get("volume", 0)
                            vol_confidence = min(1.0, current_vol / avg_vol / 2) if avg_vol > 0 else 0.6
                            confidence = 0.7 + (vol_confidence * 0.3)

                            patterns.append(Pattern(
                                pattern="Bear Flag",
                                type=PatternType.BEARISH.value,
                                confidence=min(1.0, confidence),
                                start_idx=i,
                                end_idx=flag_end,
                                description=f"Strong downtrend ({pole_pct:.1f}% pole) with {flag_pct:.1f}% consolidation"
                            ))

        return patterns

    def _detect_double_bottom(self, candles: List[Dict]) -> List[Pattern]:
        """Detect double bottom patterns."""
        patterns = []
        min_separation = 10
        max_separation = 30
        threshold_proximity = 1.0 * self.sensitivity
        threshold_valley = 2.0 * self.sensitivity

        if len(candles) < min_separation + 2:
            return patterns

        lows = [(i, c.get("low", 0)) for i, c in enumerate(candles)]

        for i in range(len(lows) - min_separation):
            for j in range(i + min_separation, min(i + max_separation, len(lows))):
                low1_idx, low1 = lows[i]
                low2_idx, low2 = lows[j]

                # Check if lows are within threshold
                low_pct = abs(self._pct_change(low1, low2))
                if low_pct > threshold_proximity:
                    continue

                # Check for valley (highs between lows should be at least 2% above lows)
                between_highs = [c.get("high", 0) for c in candles[i+1:j]]
                if between_highs:
                    max_between = max(between_highs)
                    valley_depth = self._pct_change(min(low1, low2), max_between)

                    if valley_depth >= threshold_valley:
                        # Confirm with neckline break (recent candle closes above valley)
                        recent_close = candles[-1].get("close", 0)
                        neckline = max_between

                        if recent_close > neckline * 0.99:  # Near or above neckline
                            confidence = 0.65 + (min(valley_depth / 10, 0.35))

                            patterns.append(Pattern(
                                pattern="Double Bottom",
                                type=PatternType.BULLISH.value,
                                confidence=min(1.0, confidence),
                                start_idx=low1_idx,
                                end_idx=j,
                                description=f"Two lows {low_pct:.2f}% apart with {valley_depth:.1f}% valley"
                            ))

        return patterns

    def _detect_double_top(self, candles: List[Dict]) -> List[Pattern]:
        """Detect double top patterns (inverse of double bottom)."""
        patterns = []
        min_separation = 10
        max_separation = 30
        threshold_proximity = 1.0 * self.sensitivity
        threshold_valley = 2.0 * self.sensitivity

        if len(candles) < min_separation + 2:
            return patterns

        highs = [(i, c.get("high", 0)) for i, c in enumerate(candles)]

        for i in range(len(highs) - min_separation):
            for j in range(i + min_separation, min(i + max_separation, len(highs))):
                high1_idx, high1 = highs[i]
                high2_idx, high2 = highs[j]

                high_pct = abs(self._pct_change(high1, high2))
                if high_pct > threshold_proximity:
                    continue

                between_lows = [c.get("low", 0) for c in candles[i+1:j]]
                if between_lows:
                    min_between = min(between_lows)
                    valley_depth = self._pct_change(min_between, max(high1, high2))

                    if valley_depth >= threshold_valley:
                        recent_close = candles[-1].get("close", 0)
                        neckline = min_between

                        if recent_close < neckline * 1.01:  # Near or below neckline
                            confidence = 0.65 + (min(valley_depth / 10, 0.35))

                            patterns.append(Pattern(
                                pattern="Double Top",
                                type=PatternType.BEARISH.value,
                                confidence=min(1.0, confidence),
                                start_idx=high1_idx,
                                end_idx=j,
                                description=f"Two highs {high_pct:.2f}% apart with {valley_depth:.1f}% valley"
                            ))

        return patterns

    def _detect_head_and_shoulders(self, candles: List[Dict]) -> List[Pattern]:
        """Detect head and shoulders patterns."""
        patterns = []
        min_length = 15
        shoulder_proximity = 1.5 * self.sensitivity

        if len(candles) < min_length:
            return patterns

        # Find local lows (potential neckline points)
        for i in range(5, len(candles) - 10):
            if i + 10 >= len(candles):
                break

            # Left shoulder, head, right shoulder
            left_low = min(c.get("low", 0) for c in candles[max(0, i-5):i])
            head_high = max(c.get("high", 0) for c in candles[i:i+5])
            right_low = min(c.get("low", 0) for c in candles[i+5:i+10])

            # Check shoulder proximity
            shoulder_pct = abs(self._pct_change(left_low, right_low))
            if shoulder_pct > shoulder_proximity:
                continue

            # Head should be significantly higher
            head_above_left = self._pct_change(left_low, head_high)
            head_above_right = self._pct_change(right_low, head_high)

            if head_above_left > 3.0 and head_above_right > 3.0:
                neckline = (left_low + right_low) / 2
                recent_close = candles[-1].get("close", 0)

                if recent_close < neckline * 0.99:  # Below neckline
                    confidence = 0.70

                    patterns.append(Pattern(
                        pattern="Head and Shoulders",
                        type=PatternType.BEARISH.value,
                        confidence=confidence,
                        start_idx=max(0, i-5),
                        end_idx=i+10,
                        description=f"Left shoulder {left_low:.2f}, head {head_high:.2f}, right shoulder {right_low:.2f}"
                    ))

        return patterns

    def _detect_inverse_head_and_shoulders(self, candles: List[Dict]) -> List[Pattern]:
        """Detect inverse head and shoulders patterns."""
        patterns = []
        min_length = 15
        shoulder_proximity = 1.5 * self.sensitivity

        if len(candles) < min_length:
            return patterns

        for i in range(5, len(candles) - 10):
            if i + 10 >= len(candles):
                break

            left_high = max(c.get("high", 0) for c in candles[max(0, i-5):i])
            head_low = min(c.get("low", 0) for c in candles[i:i+5])
            right_high = max(c.get("high", 0) for c in candles[i+5:i+10])

            shoulder_pct = abs(self._pct_change(left_high, right_high))
            if shoulder_pct > shoulder_proximity:
                continue

            head_below_left = self._pct_change(head_low, left_high)
            head_below_right = self._pct_change(head_low, right_high)

            if head_below_left > 3.0 and head_below_right > 3.0:
                neckline = (left_high + right_high) / 2
                recent_close = candles[-1].get("close", 0)

                if recent_close > neckline * 1.01:  # Above neckline
                    confidence = 0.70

                    patterns.append(Pattern(
                        pattern="Inverse Head and Shoulders",
                        type=PatternType.BULLISH.value,
                        confidence=confidence,
                        start_idx=max(0, i-5),
                        end_idx=i+10,
                        description=f"Left shoulder {left_high:.2f}, head {head_low:.2f}, right shoulder {right_high:.2f}"
                    ))

        return patterns

    def _detect_ascending_triangle(self, candles: List[Dict]) -> List[Pattern]:
        """Detect ascending triangle patterns."""
        patterns = []
        min_length = 15
        resistance_threshold = 0.5 * self.sensitivity

        if len(candles) < min_length:
            return patterns

        for start_idx in range(len(candles) - min_length):
            segment = candles[start_idx:start_idx + min_length]
            highs = [c.get("high", 0) for c in segment]
            lows = [c.get("low", 0) for c in segment]

            # Find flat resistance (multiple touches within 0.5%)
            max_high = max(highs)
            resistance_touches = sum(1 for h in highs if h > max_high * (1 - resistance_threshold/100))

            if resistance_touches < 3:
                continue

            # Check for rising lows
            lows_rising = True
            for i in range(len(lows) - 1):
                if lows[i] >= lows[i+1]:
                    lows_rising = False
                    break

            if lows_rising:
                confidence = 0.65 + (min(resistance_touches / 6, 0.35))

                patterns.append(Pattern(
                    pattern="Ascending Triangle",
                    type=PatternType.BULLISH.value,
                    confidence=min(1.0, confidence),
                    start_idx=start_idx,
                    end_idx=start_idx + min_length - 1,
                    description=f"Flat resistance at {max_high:.2f} with {resistance_touches} touches, rising lows"
                ))

        return patterns

    def _detect_descending_triangle(self, candles: List[Dict]) -> List[Pattern]:
        """Detect descending triangle patterns."""
        patterns = []
        min_length = 15
        support_threshold = 0.5 * self.sensitivity

        if len(candles) < min_length:
            return patterns

        for start_idx in range(len(candles) - min_length):
            segment = candles[start_idx:start_idx + min_length]
            highs = [c.get("high", 0) for c in segment]
            lows = [c.get("low", 0) for c in segment]

            min_low = min(lows)
            support_touches = sum(1 for l in lows if l < min_low * (1 + support_threshold/100))

            if support_touches < 3:
                continue

            highs_falling = True
            for i in range(len(highs) - 1):
                if highs[i] <= highs[i+1]:
                    highs_falling = False
                    break

            if highs_falling:
                confidence = 0.65 + (min(support_touches / 6, 0.35))

                patterns.append(Pattern(
                    pattern="Descending Triangle",
                    type=PatternType.BEARISH.value,
                    confidence=min(1.0, confidence),
                    start_idx=start_idx,
                    end_idx=start_idx + min_length - 1,
                    description=f"Flat support at {min_low:.2f} with {support_touches} touches, falling highs"
                ))

        return patterns

    def _detect_bullish_engulfing(self, candles: List[Dict]) -> List[Pattern]:
        """Detect bullish engulfing patterns."""
        patterns = []

        if len(candles) < 2:
            return patterns

        for i in range(len(candles) - 1):
            prev_candle = candles[i]
            curr_candle = candles[i+1]

            # Previous must be bearish
            if self._is_bullish(prev_candle):
                continue

            # Current must be bullish
            if not self._is_bullish(curr_candle):
                continue

            prev_body = self._body_size(prev_candle)
            curr_body = self._body_size(curr_candle)

            # Current body must fully engulf previous body
            prev_open = prev_candle.get("open", 0)
            prev_close = prev_candle.get("close", 0)
            curr_open = curr_candle.get("open", 0)
            curr_close = curr_candle.get("close", 0)

            engulfs_body = (curr_open < min(prev_open, prev_close) and
                           curr_close > max(prev_open, prev_close))

            if engulfs_body:
                # Volume confidence
                avg_vol = self._avg_volume(candles, 20)
                curr_vol = curr_candle.get("volume", 0)
                vol_ratio = curr_vol / avg_vol if avg_vol > 0 else 1.0
                vol_confidence = min(vol_ratio / 1.5, 1.0)

                confidence = 0.65 + (vol_confidence * 0.35)

                patterns.append(Pattern(
                    pattern="Bullish Engulfing",
                    type=PatternType.BULLISH.value,
                    confidence=min(1.0, confidence),
                    start_idx=i,
                    end_idx=i+1,
                    description=f"Red candle engulfed by green candle"
                ))

        return patterns

    def _detect_bearish_engulfing(self, candles: List[Dict]) -> List[Pattern]:
        """Detect bearish engulfing patterns."""
        patterns = []

        if len(candles) < 2:
            return patterns

        for i in range(len(candles) - 1):
            prev_candle = candles[i]
            curr_candle = candles[i+1]

            # Previous must be bullish
            if not self._is_bullish(prev_candle):
                continue

            # Current must be bearish
            if self._is_bullish(curr_candle):
                continue

            prev_body = self._body_size(prev_candle)
            curr_body = self._body_size(curr_candle)

            prev_open = prev_candle.get("open", 0)
            prev_close = prev_candle.get("close", 0)
            curr_open = curr_candle.get("open", 0)
            curr_close = curr_candle.get("close", 0)

            engulfs_body = (curr_open > max(prev_open, prev_close) and
                           curr_close < min(prev_open, prev_close))

            if engulfs_body:
                avg_vol = self._avg_volume(candles, 20)
                curr_vol = curr_candle.get("volume", 0)
                vol_ratio = curr_vol / avg_vol if avg_vol > 0 else 1.0
                vol_confidence = min(vol_ratio / 1.5, 1.0)

                confidence = 0.65 + (vol_confidence * 0.35)

                patterns.append(Pattern(
                    pattern="Bearish Engulfing",
                    type=PatternType.BEARISH.value,
                    confidence=min(1.0, confidence),
                    start_idx=i,
                    end_idx=i+1,
                    description=f"Green candle engulfed by red candle"
                ))

        return patterns

    def _detect_morning_star(self, candles: List[Dict]) -> List[Pattern]:
        """Detect morning star patterns (reversal)."""
        patterns = []

        if len(candles) < 3:
            return patterns

        for i in range(len(candles) - 2):
            candle1 = candles[i]      # Large red
            candle2 = candles[i+1]    # Small body (doji-like)
            candle3 = candles[i+2]    # Large green

            # Candle 1 must be bearish with decent size
            if self._is_bullish(candle1) or self._body_size(candle1) < 1.0:
                continue

            # Candle 2 must have small body (doji-like)
            if self._body_size(candle2) > self._body_size(candle1) * 0.3:
                continue

            # Candle 3 must be bullish with size
            if not self._is_bullish(candle3) or self._body_size(candle3) < 1.0:
                continue

            # Candle 3 should close above candle 1's midpoint
            c1_mid = (candle1.get("open", 0) + candle1.get("close", 0)) / 2
            c3_close = candle3.get("close", 0)

            if c3_close > c1_mid:
                confidence = 0.70

                patterns.append(Pattern(
                    pattern="Morning Star",
                    type=PatternType.BULLISH.value,
                    confidence=confidence,
                    start_idx=i,
                    end_idx=i+2,
                    description="Large red, small body, large green - reversal pattern"
                ))

        return patterns

    def _detect_evening_star(self, candles: List[Dict]) -> List[Pattern]:
        """Detect evening star patterns (reversal)."""
        patterns = []

        if len(candles) < 3:
            return patterns

        for i in range(len(candles) - 2):
            candle1 = candles[i]      # Large green
            candle2 = candles[i+1]    # Small body (doji-like)
            candle3 = candles[i+2]    # Large red

            # Candle 1 must be bullish with decent size
            if not self._is_bullish(candle1) or self._body_size(candle1) < 1.0:
                continue

            # Candle 2 must have small body (doji-like)
            if self._body_size(candle2) > self._body_size(candle1) * 0.3:
                continue

            # Candle 3 must be bearish with size
            if self._is_bullish(candle3) or self._body_size(candle3) < 1.0:
                continue

            # Candle 3 should close below candle 1's midpoint
            c1_mid = (candle1.get("open", 0) + candle1.get("close", 0)) / 2
            c3_close = candle3.get("close", 0)

            if c3_close < c1_mid:
                confidence = 0.70

                patterns.append(Pattern(
                    pattern="Evening Star",
                    type=PatternType.BEARISH.value,
                    confidence=confidence,
                    start_idx=i,
                    end_idx=i+2,
                    description="Large green, small body, large red - reversal pattern"
                ))

        return patterns

    def _detect_hammer(self, candles: List[Dict]) -> List[Pattern]:
        """Detect hammer patterns (bullish reversal at bottom)."""
        patterns = []

        if len(candles) < 2:
            return patterns

        for i in range(len(candles)):
            candle = candles[i]

            body_size = self._body_size(candle)
            lower_wick = self._lower_wick(candle)
            upper_wick = self._upper_wick(candle)

            # Lower wick must be at least 2x body size
            if lower_wick < body_size * 2.0:
                continue

            # Upper wick should be minimal
            if upper_wick > body_size * 0.5:
                continue

            # Check if at bottom of downtrend
            if i > 5:
                trend = self._trend_direction(candles[max(0, i-10):i])
                if trend != "down":
                    continue

            confidence = 0.70

            patterns.append(Pattern(
                pattern="Hammer",
                type=PatternType.BULLISH.value,
                confidence=confidence,
                start_idx=i,
                end_idx=i,
                description=f"Small body with long lower wick ({lower_wick:.2f} / body)"
            ))

        return patterns

    def _detect_shooting_star(self, candles: List[Dict]) -> List[Pattern]:
        """Detect shooting star patterns (bearish reversal at top)."""
        patterns = []

        if len(candles) < 2:
            return patterns

        for i in range(len(candles)):
            candle = candles[i]

            body_size = self._body_size(candle)
            lower_wick = self._lower_wick(candle)
            upper_wick = self._upper_wick(candle)

            # Upper wick must be at least 2x body size
            if upper_wick < body_size * 2.0:
                continue

            # Lower wick should be minimal
            if lower_wick > body_size * 0.5:
                continue

            # Check if at top of uptrend
            if i > 5:
                trend = self._trend_direction(candles[max(0, i-10):i])
                if trend != "up":
                    continue

            confidence = 0.70

            patterns.append(Pattern(
                pattern="Shooting Star",
                type=PatternType.BEARISH.value,
                confidence=confidence,
                start_idx=i,
                end_idx=i,
                description=f"Small body with long upper wick ({upper_wick:.2f} / body)"
            ))

        return patterns

    # ==================== Main Detection Method ====================

    def detect_patterns(self, candles: List[Dict]) -> List[Dict]:
        """
        Detect all patterns in the given candle data.

        Args:
            candles: List of OHLCV candle dictionaries with keys:
                    open, high, low, close, volume, time

        Returns:
            List of detected patterns as dicts with keys:
            pattern, type, confidence, start_idx, end_idx, description
        """
        if not candles:
            logger.warning("No candles provided for pattern detection")
            return []

        if len(candles) < 2:
            logger.warning("Insufficient candles for pattern detection (need at least 2)")
            return []

        # Validate candle data
        try:
            for i, candle in enumerate(candles):
                required_keys = ["open", "high", "low", "close", "volume"]
                for key in required_keys:
                    if key not in candle:
                        logger.warning(f"Candle {i} missing key '{key}'")
                        return []
        except Exception as e:
            logger.error(f"Error validating candles: {e}")
            return []

        all_patterns = []

        try:
            # Run all pattern detection methods
            all_patterns.extend(self._detect_bull_flag(candles))
            all_patterns.extend(self._detect_bear_flag(candles))
            all_patterns.extend(self._detect_double_bottom(candles))
            all_patterns.extend(self._detect_double_top(candles))
            all_patterns.extend(self._detect_head_and_shoulders(candles))
            all_patterns.extend(self._detect_inverse_head_and_shoulders(candles))
            all_patterns.extend(self._detect_ascending_triangle(candles))
            all_patterns.extend(self._detect_descending_triangle(candles))
            all_patterns.extend(self._detect_bullish_engulfing(candles))
            all_patterns.extend(self._detect_bearish_engulfing(candles))
            all_patterns.extend(self._detect_morning_star(candles))
            all_patterns.extend(self._detect_evening_star(candles))
            all_patterns.extend(self._detect_hammer(candles))
            all_patterns.extend(self._detect_shooting_star(candles))

        except Exception as e:
            logger.error(f"Error during pattern detection: {e}")
            return []

        # Convert Pattern objects to dicts
        result = [
            {
                "pattern": p.pattern,
                "type": p.type,
                "confidence": p.confidence,
                "start_idx": p.start_idx,
                "end_idx": p.end_idx,
                "description": p.description
            }
            for p in all_patterns
        ]

        logger.debug(f"Detected {len(result)} patterns in {len(candles)} candles")
        return result

    # ==================== Signal Analysis Method ====================

    def get_pattern_signals(self, candles: List[Dict]) -> Dict:
        """
        Get aggregated pattern signals and bias from detected patterns.

        Args:
            candles: List of OHLCV candles

        Returns:
            Dictionary with keys:
            - bullish_count: Number of bullish patterns
            - bearish_count: Number of bearish patterns
            - patterns: List of detected patterns
            - overall_bias: "bullish", "bearish", or "neutral"
            - strongest_pattern: Pattern with highest confidence, or None
        """
        patterns = self.detect_patterns(candles)

        bullish_patterns = [p for p in patterns if p["type"] == "bullish"]
        bearish_patterns = [p for p in patterns if p["type"] == "bearish"]

        bullish_count = len(bullish_patterns)
        bearish_count = len(bearish_patterns)

        # Determine overall bias
        if bullish_count > bearish_count:
            overall_bias = "bullish"
        elif bearish_count > bullish_count:
            overall_bias = "bearish"
        else:
            overall_bias = "neutral"

        # Find strongest pattern
        strongest = None
        if patterns:
            strongest = max(patterns, key=lambda p: p["confidence"])

        result = {
            "bullish_count": bullish_count,
            "bearish_count": bearish_count,
            "patterns": patterns,
            "overall_bias": overall_bias,
            "strongest_pattern": strongest
        }

        logger.debug(
            f"Signal summary: {bullish_count} bullish, {bearish_count} bearish, "
            f"bias={overall_bias}"
        )

        return result


# ==================== API Integration ====================

def analyze_patterns(candles: List[Dict], sensitivity: float = 1.0) -> Dict:
    """
    Convenience function for API endpoint use.

    Analyzes candlestick patterns and returns aggregated signals.

    Args:
        candles: List of OHLCV candles
        sensitivity: Pattern detection sensitivity multiplier (default 1.0)

    Returns:
        Dictionary with pattern analysis results
    """
    try:
        recognizer = PatternRecognizer(sensitivity=sensitivity)
        return recognizer.get_pattern_signals(candles)
    except Exception as e:
        logger.error(f"Error in analyze_patterns: {e}")
        return {
            "bullish_count": 0,
            "bearish_count": 0,
            "patterns": [],
            "overall_bias": "neutral",
            "strongest_pattern": None,
            "error": str(e)
        }
