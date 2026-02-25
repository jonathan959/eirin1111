"""
Self-tuning strategy parameters (Phase 2 Advanced Intelligence).
Adjusts TP/SL/trailing stops based on volatility, regime, recent performance.
"""
import logging
from datetime import datetime
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


def _is_stock_symbol(symbol: str) -> bool:
    try:
        from symbol_classifier import is_stock_symbol as check
        return check(symbol)
    except Exception:
        return len(str(symbol)) < 6 and "/" not in str(symbol)


def _get_current_spread(symbol: str) -> float:
    try:
        from phase2_data_fetcher import get_current_spread
        return get_current_spread(symbol)
    except Exception:
        return 0.0


class AdaptiveParameterEngine:
    """
    Dynamically adjusts trading parameters based on current conditions.
    """

    def calculate_adaptive_parameters(
        self,
        symbol: str,
        base_config: Dict[str, Any],
        market_conditions: Dict[str, Any],
        recent_performance: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Calculate optimal parameters for current conditions.
        base_config should have stop_loss_pct, take_profit_pct, trailing_activation_pct, trailing_distance_pct.
        """
        adjusted = dict(base_config)
        adjustments: List[str] = []
        reasoning_parts: List[str] = []

        vol = market_conditions.get("volatility", 0.20)
        regime = str(market_conditions.get("regime", "RANGE")).upper()
        trend_strength = float(market_conditions.get("trend_strength", 0.5))

        sl = float(adjusted.get("stop_loss_pct", -5.0))
        tp = float(adjusted.get("take_profit_pct", 10.0))
        trail_act = float(adjusted.get("trailing_activation_pct", 50))
        trail_dist = float(adjusted.get("trailing_distance_pct", 2.5))

        # 1. Volatility
        if vol > 0.40:
            sl *= 1.5
            tp *= 1.5
            trail_dist *= 1.3
            adjustments.append("high_volatility_adjustment")
            reasoning_parts.append(f"High volatility ({vol:.0%}): widened stops by 50%")
        elif vol < 0.15:
            sl *= 0.8
            tp *= 0.8
            adjustments.append("low_volatility_adjustment")
            reasoning_parts.append(f"Low volatility ({vol:.0%}): tightened stops by 20%")

        # 2. Regime
        if regime in ("STRONG_BULL", "TREND_UP", "UPTREND") and trend_strength > 0.7:
            tp *= 1.3
            trail_act *= 1.2
            adjustments.append("strong_trend_adjustment")
            reasoning_parts.append("Strong uptrend: increased TP targets, trail longer")
        elif regime in ("STRONG_BEAR", "WEAK_BEAR", "TREND_DOWN", "DOWNTREND"):
            tp *= 0.8
            sl *= 0.9
            adjustments.append("bear_market_adjustment")
            reasoning_parts.append("Bear market: tighter TP and SL")
        elif regime in ("RANGE", "RANGING"):
            tp *= 0.7
            trail_act *= 0.8
            adjustments.append("range_adjustment")
            reasoning_parts.append("Range-bound: smaller profit targets")

        # 3. Performance
        win_rate = float(recent_performance.get("last_10_trades_win_rate", 0.5))
        streak = str(recent_performance.get("current_streak", "neutral")).lower()
        streak_len = int(recent_performance.get("streak_length", 0))
        size_mult = 1.0

        if streak == "losing" and streak_len >= 3:
            size_mult = 0.7
            sl *= 0.85
            tp *= 0.9
            adjustments.append("losing_streak_protection")
            reasoning_parts.append(f"Losing streak ({streak_len}): reduced size 30%, tighter exits")
        elif streak == "winning" and streak_len >= 3:
            size_mult = 1.15
            tp *= 1.15
            adjustments.append("winning_streak_boost")
            reasoning_parts.append(f"Winning streak ({streak_len}): increased targets")
        elif win_rate < 0.40:
            size_mult = 0.8
            adjustments.append("poor_performance_defense")
            reasoning_parts.append(f"Low win rate ({win_rate:.0%}): reduced size")

        # 4. Time-of-day (stocks)
        if _is_stock_symbol(symbol):
            hour = datetime.utcnow().hour  # or local; spec uses ET
            if 14 <= hour <= 15:
                sl *= 1.2
                adjustments.append("market_open_adjustment")
                reasoning_parts.append("Market open: wider stops for volatility")

        # 5. Spread
        spread = _get_current_spread(symbol)
        if spread > 0.003:
            tp += spread * 200
            adjustments.append("wide_spread_adjustment")
            reasoning_parts.append(f"Wide spread ({spread:.2%}): adjusted TP")

        adjusted["stop_loss_pct"] = round(sl, 2)
        adjusted["take_profit_pct"] = round(tp, 2)
        adjusted["trailing_activation_pct"] = round(trail_act, 1)
        adjusted["trailing_distance_pct"] = round(trail_dist, 2)
        adjusted["position_size_multiplier"] = size_mult
        adjusted["adjustments_applied"] = adjustments
        adjusted["reasoning"] = ". ".join(reasoning_parts)

        return adjusted

    def get_market_conditions(self, symbol: str) -> Dict[str, Any]:
        """Gather current market conditions."""
        vol = 0.20
        regime = "RANGE"
        trend_strength = 0.5

        try:
            from phase2_data_fetcher import fetch_recent_candles
            import numpy as np

            candles = fetch_recent_candles(symbol, timeframe="1d", periods=100)
            if candles and len(candles) >= 20:
                arr = np.array(candles)
                close = arr[:, 4].astype(float)
                ret = np.diff(close) / close[:-1]
                vol = float(np.std(ret) * np.sqrt(252)) if len(ret) > 0 else 0.20

            try:
                from ml_regime_detector import MLRegimeDetector
                import pandas as pd
                det = MLRegimeDetector()
                df = pd.DataFrame(candles, columns=["ts", "open", "high", "low", "close", "volume"][:len(candles[0])]) if candles else pd.DataFrame()
                pred = det.predict_regime(df if not df.empty else candles)
                regime = pred.get("regime", regime)
                trend_strength = pred.get("confidence", 0.5)
            except Exception:
                from strategies import detect_regime
                res = detect_regime(candles)
                regime = getattr(res, "legacy_regime", res.regime) or res.regime
        except Exception as e:
            logger.debug("get_market_conditions: %s", e)

        return {"volatility": vol, "regime": regime, "trend_strength": trend_strength}

    def get_recent_performance(self, bot_id: int) -> Dict[str, Any]:
        """Get recent trading performance for this bot."""
        try:
            from db import list_deals

            deals = list_deals(int(bot_id), limit=10)
            closed = [d for d in deals if str(d.get("state", "")).upper() == "CLOSED"]
            if not closed:
                return {"last_10_trades_win_rate": 0.5, "current_streak": "neutral", "streak_length": 0}

            pnls = []
            for d in closed:
                pnl = float(d.get("realized_pnl_quote", 0) or 0)
                pnls.append((pnl > 0, pnl))

            wins = sum(1 for w, _ in pnls if w)
            win_rate = wins / len(pnls) if pnls else 0.5

            streak_type = None
            streak_len = 0
            for is_win, _ in pnls:
                if streak_type is None:
                    streak_type = "winning" if is_win else "losing"
                    streak_len = 1
                elif (streak_type == "winning" and is_win) or (streak_type == "losing" and not is_win):
                    streak_len += 1
                else:
                    break

            return {
                "last_10_trades_win_rate": win_rate,
                "current_streak": streak_type or "neutral",
                "streak_length": streak_len,
            }
        except Exception as e:
            logger.debug("get_recent_performance: %s", e)
            return {"last_10_trades_win_rate": 0.5, "current_streak": "neutral", "streak_length": 0}
