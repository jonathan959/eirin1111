"""
Phase 4: Seasonality Trading
Trade based on calendar patterns and seasonal trends.
"""
from datetime import datetime
from typing import Any, Dict, List


class SeasonalityAnalyzer:
    """
    Identifies seasonal trading opportunities.
    """

    def get_seasonal_bias(self, symbol: str, market_type: str) -> Dict[str, Any]:
        """
        Get current seasonal bias for symbol.

        Returns:
        {
            'bias': 'bullish' or 'bearish' or 'neutral',
            'strength': 0-1 scale,
            'patterns': ['sell_in_may', 'january_effect', ...],
            'recommendation': 'increase/decrease/neutral',
            'reasoning': 'Historically strong month for stocks'
        }
        """
        now = datetime.now()
        month = now.month
        day_of_month = now.day

        patterns: List[str] = []
        bias_score = 0.0  # -1 to +1

        if market_type == "stock" or market_type == "stocks":
            # November - April (strong 6 months)
            if month in (11, 12) or 1 <= month <= 4:
                patterns.append("strong_six_months")
                bias_score += 0.3

            # May - October (weak 6 months)
            elif 5 <= month <= 10:
                patterns.append("sell_in_may")
                bias_score -= 0.2

            # January Effect (small caps)
            if month == 1:
                patterns.append("january_effect")
                bias_score += 0.2

            # End of month (last 3 days + first 2 days)
            if day_of_month >= 28 or day_of_month <= 2:
                patterns.append("month_end_strength")
                bias_score += 0.15

            # Santa Claus Rally (last week of Dec)
            if month == 12 and day_of_month >= 24:
                patterns.append("santa_claus_rally")
                bias_score += 0.25

        elif market_type == "crypto":
            # Q4 rally (Nov-Dec)
            if month in (11, 12):
                patterns.append("q4_crypto_rally")
                bias_score += 0.4

            # January selloff
            elif month == 1:
                patterns.append("january_crypto_selloff")
                bias_score -= 0.3

        if bias_score > 0.2:
            bias = "bullish"
            recommendation = "increase_position"
        elif bias_score < -0.2:
            bias = "bearish"
            recommendation = "decrease_position"
        else:
            bias = "neutral"
            recommendation = "neutral"

        strength = abs(bias_score)
        reasoning = f"Seasonality: {', '.join(patterns) if patterns else 'No strong patterns'}"

        return {
            "bias": bias,
            "strength": strength,
            "patterns": patterns,
            "recommendation": recommendation,
            "reasoning": reasoning,
        }
