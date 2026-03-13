"""
Anomaly detection module for trading bot.

Detects various market anomalies including volume spikes, price gaps,
volatility explosions, whale activity, and divergences.
"""

import logging
import numpy as np
from typing import List, Dict, Any, Optional
from collections import deque

logger = logging.getLogger(__name__)


class AnomalyDetector:
    """Detects market anomalies in OHLCV candle data."""

    def __init__(
        self,
        volume_zscore_threshold: float = 3.0,
        price_zscore_threshold: float = 3.0,
        lookback: int = 50,
    ):
        """
        Initialize anomaly detector.

        Args:
            volume_zscore_threshold: Z-score threshold for volume anomalies
            price_zscore_threshold: Z-score threshold for price anomalies
            lookback: Number of candles to use for rolling calculations
        """
        self.volume_zscore_threshold = volume_zscore_threshold
        self.price_zscore_threshold = price_zscore_threshold
        self.lookback = lookback
        logger.info(
            f"AnomalyDetector initialized with volume_threshold={volume_zscore_threshold}, "
            f"price_threshold={price_zscore_threshold}, lookback={lookback}"
        )

    def _calculate_zscore(
        self, values: np.ndarray, window: int = None
    ) -> np.ndarray:
        """Calculate Z-scores for array using rolling window."""
        if window is None:
            window = self.lookback

        if len(values) < window + 1:
            # Not enough data, return zeros
            return np.zeros(len(values))

        zscores = np.zeros(len(values))
        for i in range(window, len(values)):
            window_data = values[i - window : i]
            mean = np.mean(window_data)
            std = np.std(window_data)

            if std == 0:
                zscores[i] = 0
            else:
                zscores[i] = (values[i] - mean) / std

        return zscores

    def _calculate_atr(self, candles: List[dict], period: int = 14) -> List[float]:
        """Calculate Average True Range."""
        if len(candles) < period:
            return [0.0] * len(candles)

        atr_values = []
        tr_values = []

        for i, candle in enumerate(candles):
            high = candle.get("high", candle.get("h", 0))
            low = candle.get("low", candle.get("l", 0))
            close = candle.get("close", candle.get("c", 0))

            if i == 0:
                tr = high - low
            else:
                prev_close = candles[i - 1].get(
                    "close", candles[i - 1].get("c", 0)
                )
                tr = max(
                    high - low, abs(high - prev_close), abs(low - prev_close)
                )

            tr_values.append(tr)

            if i < period - 1:
                atr_values.append(0.0)
            else:
                atr = np.mean(tr_values[-period:])
                atr_values.append(atr)

        return atr_values

    def _detect_volume_spike(
        self, candles: List[dict]
    ) -> List[Dict[str, Any]]:
        """Detect volume spike anomalies."""
        anomalies = []

        if len(candles) < self.lookback + 1:
            return anomalies

        volumes = np.array(
            [candle.get("volume", candle.get("v", 0)) for candle in candles]
        )
        zscores = self._calculate_zscore(volumes)

        for i in range(len(candles)):
            if zscores[i] == 0:
                continue

            zscore = zscores[i]
            if zscore > 5.0:
                severity = "critical"
                recommended_action = "consider_exit"
            elif zscore > self.volume_zscore_threshold:
                severity = "high"
                recommended_action = "caution_entry"
            else:
                continue

            anomalies.append(
                {
                    "type": "volume_spike",
                    "severity": severity,
                    "zscore": float(zscore),
                    "index": i,
                    "time": candles[i].get("time", None),
                    "description": f"Volume spike detected: {zscore:.2f}σ above average",
                    "recommended_action": recommended_action,
                }
            )

        return anomalies

    def _detect_price_spike(self, candles: List[dict]) -> List[Dict[str, Any]]:
        """Detect price spike anomalies."""
        anomalies = []

        if len(candles) < self.lookback + 1:
            return anomalies

        price_spikes = []
        for candle in candles:
            open_price = candle.get("open", candle.get("o", 0))
            close_price = candle.get("close", candle.get("c", 0))

            if close_price == 0:
                price_spikes.append(0)
            else:
                spike = abs(close_price - open_price) / close_price
                price_spikes.append(spike)

        spikes_array = np.array(price_spikes)
        zscores = self._calculate_zscore(spikes_array)

        for i in range(len(candles)):
            if zscores[i] == 0:
                continue

            zscore = zscores[i]
            if zscore > self.price_zscore_threshold:
                anomalies.append(
                    {
                        "type": "price_spike",
                        "severity": "high",
                        "zscore": float(zscore),
                        "index": i,
                        "time": candles[i].get("time", None),
                        "description": f"Price spike detected: {zscore:.2f}σ above average",
                        "recommended_action": "wait_for_confirmation",
                    }
                )

        return anomalies

    def _detect_gaps(self, candles: List[dict]) -> List[Dict[str, Any]]:
        """Detect price gaps between candles."""
        anomalies = []

        for i in range(1, len(candles)):
            prev_close = candles[i - 1].get("close", candles[i - 1].get("c", 0))
            curr_open = candles[i].get("open", candles[i].get("o", 0))

            if prev_close == 0:
                continue

            gap_percent = abs(curr_open - prev_close) / prev_close

            if gap_percent > 0.05:
                severity = "high"
                # Determine direction for recommended action
                if curr_open > prev_close:
                    recommended_action = "trend_continuation"
                else:
                    recommended_action = "gap_fill_likely"
            elif gap_percent > 0.02:
                severity = "medium"
                if curr_open > prev_close:
                    recommended_action = "trend_continuation"
                else:
                    recommended_action = "gap_fill_likely"
            else:
                continue

            anomalies.append(
                {
                    "type": "gap",
                    "severity": severity,
                    "zscore": 0.0,
                    "index": i,
                    "time": candles[i].get("time", None),
                    "description": f"Price gap detected: {gap_percent*100:.2f}% between candles",
                    "recommended_action": recommended_action,
                }
            )

        return anomalies

    def _detect_volatility_explosion(
        self, candles: List[dict]
    ) -> List[Dict[str, Any]]:
        """Detect volatility explosions."""
        anomalies = []

        if len(candles) < 20 + 1:
            return anomalies

        atr_values = self._calculate_atr(candles, period=20)

        for i in range(20, len(candles)):
            current_atr = atr_values[i]
            previous_atrs = atr_values[i - 20 : i]
            avg_atr = np.mean(previous_atrs)

            if avg_atr == 0:
                continue

            atr_ratio = current_atr / avg_atr

            if atr_ratio > 2.0:
                anomalies.append(
                    {
                        "type": "volatility_explosion",
                        "severity": "high",
                        "zscore": float(atr_ratio),
                        "index": i,
                        "time": candles[i].get("time", None),
                        "description": f"Volatility explosion: ATR {atr_ratio:.2f}x average",
                        "recommended_action": "reduce_position_size",
                    }
                )

        return anomalies

    def _detect_volume_drought(self, candles: List[dict]) -> List[Dict[str, Any]]:
        """Detect volume droughts."""
        anomalies = []

        if len(candles) < 20 + 1:
            return anomalies

        volumes = np.array(
            [candle.get("volume", candle.get("v", 0)) for candle in candles]
        )

        for i in range(20, len(candles)):
            current_volume = volumes[i]
            avg_volume = np.mean(volumes[i - 20 : i])

            if avg_volume == 0:
                continue

            volume_ratio = current_volume / avg_volume

            if volume_ratio < 0.2:
                anomalies.append(
                    {
                        "type": "volume_drought",
                        "severity": "medium",
                        "zscore": float(volume_ratio),
                        "index": i,
                        "time": candles[i].get("time", None),
                        "description": f"Volume drought: {volume_ratio:.2f}x of average",
                        "recommended_action": "low_liquidity_caution",
                    }
                )

        return anomalies

    def _detect_whale_activity(self, candles: List[dict]) -> List[Dict[str, Any]]:
        """Detect potential whale activity."""
        anomalies = []

        if len(candles) < 2:
            return anomalies

        volumes = np.array(
            [candle.get("volume", candle.get("v", 0)) for candle in candles]
        )
        avg_volume = np.mean(volumes)

        if avg_volume == 0:
            return anomalies

        for i in range(len(candles)):
            current_volume = volumes[i]
            volume_ratio = current_volume / avg_volume

            if volume_ratio > 10.0:
                # Check for large body size
                open_price = candles[i].get("open", candles[i].get("o", 0))
                close_price = candles[i].get("close", candles[i].get("c", 0))
                high_price = candles[i].get("high", candles[i].get("h", 0))
                low_price = candles[i].get("low", candles[i].get("l", 0))

                if high_price == 0:
                    continue

                body_size = abs(close_price - open_price)
                candle_range = high_price - low_price

                if candle_range > 0:
                    body_ratio = body_size / candle_range

                    if body_ratio > 0.5:  # At least 50% of candle is body
                        # Determine direction
                        if close_price > open_price:
                            recommended_action = "follow_whale_direction"
                        else:
                            recommended_action = "follow_whale_direction"

                        anomalies.append(
                            {
                                "type": "whale_activity",
                                "severity": "high",
                                "zscore": float(volume_ratio),
                                "index": i,
                                "time": candles[i].get("time", None),
                                "description": f"Whale activity detected: {volume_ratio:.1f}x volume with large body",
                                "recommended_action": recommended_action,
                            }
                        )

        return anomalies

    def _detect_divergence(self, candles: List[dict]) -> List[Dict[str, Any]]:
        """Detect volume divergences with price trends."""
        anomalies = []

        if len(candles) < 20:
            return anomalies

        # Look at last 20 candles for divergence
        window = min(20, len(candles))
        start_idx = len(candles) - window

        volumes = np.array(
            [candle.get("volume", candle.get("v", 0)) for candle in candles[start_idx:]]
        )
        closes = np.array(
            [
                candle.get("close", candle.get("c", 0))
                for candle in candles[start_idx:]
            ]
        )

        if len(closes) < 2 or len(volumes) < 2:
            return anomalies

        # Check for bearish divergence (new high, declining volume)
        max_price_idx = np.argmax(closes)
        max_volume_idx = np.argmax(volumes)

        if max_price_idx > max_volume_idx and max_price_idx > 0:
            if closes[max_price_idx] > closes[0]:
                # Price made new high, but volume declined
                anomalies.append(
                    {
                        "type": "bearish_divergence",
                        "severity": "medium",
                        "zscore": 0.0,
                        "index": start_idx + max_price_idx,
                        "time": candles[start_idx + max_price_idx].get("time", None),
                        "description": "Bearish divergence: price high but declining volume",
                        "recommended_action": "prepare_for_reversal",
                    }
                )

        # Check for bullish divergence (new low, declining volume)
        min_price_idx = np.argmin(closes)
        if min_price_idx > max_volume_idx and min_price_idx > 0:
            if closes[min_price_idx] < closes[0]:
                # Price made new low, but volume declined
                anomalies.append(
                    {
                        "type": "bullish_divergence",
                        "severity": "medium",
                        "zscore": 0.0,
                        "index": start_idx + min_price_idx,
                        "time": candles[start_idx + min_price_idx].get("time", None),
                        "description": "Bullish divergence: price low but declining volume",
                        "recommended_action": "prepare_for_reversal",
                    }
                )

        return anomalies

    def detect_anomalies(self, candles: List[dict]) -> List[Dict[str, Any]]:
        """
        Detect all anomalies in candle data.

        Args:
            candles: List of OHLCV candle dictionaries

        Returns:
            List of anomaly dictionaries with type, severity, zscore, index, time, description, and recommended_action
        """
        if not candles or len(candles) < 2:
            logger.warning("Insufficient candles for anomaly detection")
            return []

        anomalies = []

        try:
            anomalies.extend(self._detect_volume_spike(candles))
            anomalies.extend(self._detect_price_spike(candles))
            anomalies.extend(self._detect_gaps(candles))
            anomalies.extend(self._detect_volatility_explosion(candles))
            anomalies.extend(self._detect_volume_drought(candles))
            anomalies.extend(self._detect_whale_activity(candles))
            anomalies.extend(self._detect_divergence(candles))

            # Sort by index for consistent ordering
            anomalies.sort(key=lambda x: x["index"])

            logger.info(f"Detected {len(anomalies)} anomalies in {len(candles)} candles")
            return anomalies

        except Exception as e:
            logger.error(f"Error detecting anomalies: {e}", exc_info=True)
            return []

    def get_risk_assessment(self, candles: List[dict]) -> Dict[str, Any]:
        """
        Get comprehensive risk assessment based on detected anomalies.

        Args:
            candles: List of OHLCV candle dictionaries

        Returns:
            Dictionary with anomaly_count, risk_level, anomalies, position_size_multiplier, and should_pause_trading
        """
        anomalies = self.detect_anomalies(candles)

        # Count anomalies by severity
        severity_counts = {
            "critical": 0,
            "high": 0,
            "medium": 0,
            "low": 0,
        }

        for anomaly in anomalies:
            severity = anomaly.get("severity", "low")
            severity_counts[severity] = severity_counts.get(severity, 0) + 1

        # Determine risk level
        if severity_counts["critical"] > 0:
            risk_level = "critical"
            position_size_multiplier = 0.0
        elif severity_counts["high"] > 2:
            risk_level = "critical"
            position_size_multiplier = 0.0
        elif severity_counts["high"] > 0:
            risk_level = "high"
            position_size_multiplier = 0.5
        elif severity_counts["medium"] > 0:
            risk_level = "medium"
            position_size_multiplier = 0.75
        else:
            risk_level = "low"
            position_size_multiplier = 1.0

        should_pause_trading = severity_counts["critical"] > 0

        assessment = {
            "anomaly_count": len(anomalies),
            "risk_level": risk_level,
            "anomalies": anomalies,
            "position_size_multiplier": position_size_multiplier,
            "should_pause_trading": should_pause_trading,
            "severity_breakdown": severity_counts,
        }

        logger.info(
            f"Risk assessment: {risk_level} (anomalies={len(anomalies)}, "
            f"multiplier={position_size_multiplier:.2f}, pause={should_pause_trading})"
        )

        return assessment


def assess_market_risk(candles: List[dict], **kwargs) -> Dict[str, Any]:
    """
    Convenience function for API use.

    Args:
        candles: List of OHLCV candle dictionaries
        **kwargs: Additional parameters for AnomalyDetector (volume_zscore_threshold, price_zscore_threshold, lookback)

    Returns:
        Risk assessment dictionary
    """
    detector = AnomalyDetector(**kwargs)
    return detector.get_risk_assessment(candles)
