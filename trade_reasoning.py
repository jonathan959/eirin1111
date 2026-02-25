"""
Trade Reasoning Layer - Explainable AI for trading decisions.

Generates human-readable explanations for every trade, including:
- Why the asset was selected
- Why the strategy was chosen
- What signals triggered the entry
- Risk management parameters
- Confidence score breakdown

Based on: "Upgrading to an Autonomous Multi-Asset Trading Bot" specification.
"""

import time
import json
import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Tuple
from enum import Enum

logger = logging.getLogger(__name__)


class ReasonCategory(Enum):
    """Categories of reasoning."""
    TECHNICAL = "technical"
    SENTIMENT = "sentiment"
    FUNDAMENTAL = "fundamental"
    RISK = "risk"
    REGIME = "regime"
    ML_PREDICTION = "ml"
    PORTFOLIO = "portfolio"


@dataclass
class ReasoningFactor:
    """A single factor in the decision."""
    category: ReasonCategory
    factor: str
    value: Any
    impact: str  # "bullish", "bearish", "neutral"
    weight: float  # 0-1, how much this contributed
    explanation: str


@dataclass 
class TradeReasoning:
    """Complete reasoning for a trade decision."""
    symbol: str
    action: str  # "BUY", "SELL", "HOLD"
    confidence: float
    strategy: str
    regime: str
    
    # Breakdown
    factors: List[ReasoningFactor]
    
    # Summary
    primary_reason: str
    supporting_reasons: List[str]
    risk_considerations: List[str]
    
    # Parameters
    entry_price: Optional[float] = None
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    position_size: Optional[float] = None
    
    # Timing
    timestamp: int = field(default_factory=lambda: int(time.time()))
    
    def to_human_readable(self) -> str:
        """Generate a human-readable explanation."""
        lines = []
        
        # Header
        action_emoji = "🟢" if self.action == "BUY" else ("🔴" if self.action == "SELL" else "⚪")
        lines.append(f"{action_emoji} **{self.action} {self.symbol}** (Confidence: {self.confidence:.0%})")
        lines.append("")
        
        # Primary reason
        lines.append(f"**Primary Reason:** {self.primary_reason}")
        lines.append("")
        
        # Supporting reasons
        if self.supporting_reasons:
            lines.append("**Supporting Factors:**")
            for reason in self.supporting_reasons[:5]:
                lines.append(f"  • {reason}")
            lines.append("")
        
        # Strategy and regime
        lines.append(f"**Strategy:** {self.strategy}")
        lines.append(f"**Market Regime:** {self.regime}")
        lines.append("")
        
        # Risk management
        if any([self.entry_price, self.stop_loss, self.take_profit]):
            lines.append("**Risk Management:**")
            if self.entry_price:
                lines.append(f"  • Entry: ${self.entry_price:.2f}")
            if self.stop_loss:
                if self.entry_price:
                    sl_pct = (self.stop_loss - self.entry_price) / self.entry_price * 100
                    lines.append(f"  • Stop Loss: ${self.stop_loss:.2f} ({sl_pct:+.1f}%)")
                else:
                    lines.append(f"  • Stop Loss: ${self.stop_loss:.2f}")
            if self.take_profit:
                if self.entry_price:
                    tp_pct = (self.take_profit - self.entry_price) / self.entry_price * 100
                    lines.append(f"  • Take Profit: ${self.take_profit:.2f} ({tp_pct:+.1f}%)")
                else:
                    lines.append(f"  • Take Profit: ${self.take_profit:.2f}")
            if self.position_size:
                lines.append(f"  • Position Size: ${self.position_size:.2f}")
            lines.append("")
        
        # Risk considerations
        if self.risk_considerations:
            lines.append("**Risk Considerations:**")
            for risk in self.risk_considerations[:3]:
                lines.append(f"  ⚠️ {risk}")
        
        return "\n".join(lines)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for storage/API."""
        return {
            "symbol": self.symbol,
            "action": self.action,
            "confidence": self.confidence,
            "strategy": self.strategy,
            "regime": self.regime,
            "primary_reason": self.primary_reason,
            "supporting_reasons": self.supporting_reasons,
            "risk_considerations": self.risk_considerations,
            "entry_price": self.entry_price,
            "stop_loss": self.stop_loss,
            "take_profit": self.take_profit,
            "position_size": self.position_size,
            "timestamp": self.timestamp,
            "factors": [
                {
                    "category": f.category.value,
                    "factor": f.factor,
                    "value": f.value,
                    "impact": f.impact,
                    "weight": f.weight,
                    "explanation": f.explanation
                }
                for f in self.factors
            ]
        }
    
    def to_json(self) -> str:
        """Serialize to JSON."""
        return json.dumps(self.to_dict())


class TradeReasoningEngine:
    """
    Generates explainable reasoning for trading decisions.
    
    Analyzes various factors and produces human-readable explanations
    with confidence breakdowns.
    """
    
    def __init__(self):
        self._reasoning_history: List[TradeReasoning] = []
        self._max_history = 1000
    
    def generate_reasoning(self,
                           symbol: str,
                           action: str,
                           confidence: float,
                           strategy: str,
                           regime: str,
                           context: Dict[str, Any]) -> TradeReasoning:
        """
        Generate comprehensive reasoning for a trade decision.
        
        Args:
            symbol: Trading symbol
            action: "BUY", "SELL", or "HOLD"
            confidence: Overall confidence (0-1)
            strategy: Strategy name
            regime: Market regime
            context: Dict containing:
                - technical: Technical indicator values
                - sentiment: Sentiment scores
                - ml_prediction: ML model output
                - portfolio: Portfolio context
                - risk: Risk metrics
                - prices: Price data
        
        Returns:
            TradeReasoning with full explanation
        """
        factors = []
        supporting_reasons = []
        risk_considerations = []
        
        # Extract context
        technical = context.get("technical", {})
        sentiment = context.get("sentiment", {})
        ml_pred = context.get("ml_prediction", {})
        portfolio = context.get("portfolio", {})
        risk = context.get("risk", {})
        prices = context.get("prices", {})
        
        # ==================== Technical Factors ====================
        
        # Trend
        if technical.get("trend"):
            trend = technical["trend"]
            impact = "bullish" if trend == "UP" else ("bearish" if trend == "DOWN" else "neutral")
            factors.append(ReasoningFactor(
                category=ReasonCategory.TECHNICAL,
                factor="Trend Direction",
                value=trend,
                impact=impact,
                weight=0.15,
                explanation=f"Overall trend is {trend.lower()}"
            ))
            if action == "BUY" and trend == "UP":
                supporting_reasons.append(f"Price is in an uptrend")
            elif action == "BUY" and trend == "DOWN":
                risk_considerations.append("Trading against the trend")
        
        # RSI
        rsi = technical.get("rsi")
        if rsi is not None:
            if rsi < 30:
                impact = "bullish"
                explanation = f"RSI at {rsi:.0f} indicates oversold conditions"
                if action == "BUY":
                    supporting_reasons.append(f"RSI oversold at {rsi:.0f}")
            elif rsi > 70:
                impact = "bearish"
                explanation = f"RSI at {rsi:.0f} indicates overbought conditions"
                if action == "BUY":
                    risk_considerations.append(f"RSI overbought at {rsi:.0f}")
            else:
                impact = "neutral"
                explanation = f"RSI at {rsi:.0f} is neutral"
            
            factors.append(ReasoningFactor(
                category=ReasonCategory.TECHNICAL,
                factor="RSI",
                value=rsi,
                impact=impact,
                weight=0.10,
                explanation=explanation
            ))
        
        # MACD
        macd = technical.get("macd")
        if macd:
            macd_line, signal, histogram = macd
            if histogram > 0:
                impact = "bullish"
                explanation = "MACD histogram positive, bullish momentum"
                if action == "BUY":
                    supporting_reasons.append("MACD showing bullish momentum")
            else:
                impact = "bearish"
                explanation = "MACD histogram negative, bearish momentum"
                if action == "BUY":
                    risk_considerations.append("MACD showing bearish momentum")
            
            factors.append(ReasoningFactor(
                category=ReasonCategory.TECHNICAL,
                factor="MACD",
                value={"line": macd_line, "histogram": histogram},
                impact=impact,
                weight=0.10,
                explanation=explanation
            ))
        
        # Moving Averages
        ma_context = technical.get("ma_context", {})
        if ma_context:
            above_sma50 = ma_context.get("above_sma50")
            above_sma200 = ma_context.get("above_sma200")
            
            if above_sma50 and above_sma200:
                impact = "bullish"
                explanation = "Price above both 50 and 200 SMA - strong uptrend"
                supporting_reasons.append("Price above key moving averages")
            elif not above_sma50 and not above_sma200:
                impact = "bearish"
                explanation = "Price below both 50 and 200 SMA - downtrend"
                if action == "BUY":
                    risk_considerations.append("Price below key moving averages")
            else:
                impact = "neutral"
                explanation = "Mixed moving average signals"
            
            factors.append(ReasoningFactor(
                category=ReasonCategory.TECHNICAL,
                factor="Moving Averages",
                value=ma_context,
                impact=impact,
                weight=0.12,
                explanation=explanation
            ))
        
        # Volatility
        atr_pct = technical.get("atr_pct")
        if atr_pct:
            if atr_pct > 0.05:
                risk_considerations.append(f"High volatility ({atr_pct:.1%} ATR)")
            
            factors.append(ReasoningFactor(
                category=ReasonCategory.TECHNICAL,
                factor="Volatility (ATR)",
                value=atr_pct,
                impact="neutral",
                weight=0.05,
                explanation=f"Average True Range is {atr_pct:.2%} of price"
            ))
        
        # Volume
        volume_ratio = technical.get("volume_ratio")
        if volume_ratio:
            if volume_ratio > 2.0:
                explanation = f"Volume {volume_ratio:.1f}x above average - strong interest"
                supporting_reasons.append(f"High volume confirms move ({volume_ratio:.1f}x)")
            elif volume_ratio < 0.5:
                explanation = f"Volume {volume_ratio:.1f}x below average - weak conviction"
                risk_considerations.append("Low volume - weak conviction")
            else:
                explanation = f"Volume at {volume_ratio:.1f}x average"
            
            factors.append(ReasoningFactor(
                category=ReasonCategory.TECHNICAL,
                factor="Volume",
                value=volume_ratio,
                impact="bullish" if volume_ratio > 1.5 else "neutral",
                weight=0.08,
                explanation=explanation
            ))
        
        # ==================== Sentiment Factors ====================
        
        sentiment_score = sentiment.get("score")
        if sentiment_score is not None:
            if sentiment_score > 0.3:
                impact = "bullish"
                explanation = f"Positive sentiment ({sentiment_score:.2f})"
                supporting_reasons.append("Bullish market sentiment")
            elif sentiment_score < -0.3:
                impact = "bearish"
                explanation = f"Negative sentiment ({sentiment_score:.2f})"
                if action == "BUY":
                    risk_considerations.append("Bearish market sentiment")
            else:
                impact = "neutral"
                explanation = f"Neutral sentiment ({sentiment_score:.2f})"
            
            factors.append(ReasoningFactor(
                category=ReasonCategory.SENTIMENT,
                factor="Market Sentiment",
                value=sentiment_score,
                impact=impact,
                weight=0.10,
                explanation=explanation
            ))
        
        # News
        news_impact = sentiment.get("news_impact")
        if news_impact:
            factors.append(ReasoningFactor(
                category=ReasonCategory.SENTIMENT,
                factor="News Impact",
                value=news_impact,
                impact=news_impact,
                weight=0.05,
                explanation=f"Recent news has {news_impact} impact"
            ))
            if news_impact == "positive":
                supporting_reasons.append("Positive recent news")
            elif news_impact == "negative" and action == "BUY":
                risk_considerations.append("Recent negative news")
        
        # ==================== ML Prediction ====================
        
        if ml_pred:
            ml_direction = ml_pred.get("direction")
            ml_confidence = ml_pred.get("confidence", 0.5)
            
            if ml_direction == "UP":
                impact = "bullish"
                explanation = f"ML model predicts upward move ({ml_confidence:.0%} confidence)"
                supporting_reasons.append(f"ML model bullish ({ml_confidence:.0%})")
            elif ml_direction == "DOWN":
                impact = "bearish"
                explanation = f"ML model predicts downward move ({ml_confidence:.0%} confidence)"
                if action == "BUY":
                    risk_considerations.append(f"ML model bearish ({ml_confidence:.0%})")
            else:
                impact = "neutral"
                explanation = "ML model neutral"
            
            factors.append(ReasoningFactor(
                category=ReasonCategory.ML_PREDICTION,
                factor="ML Forecast",
                value={"direction": ml_direction, "confidence": ml_confidence},
                impact=impact,
                weight=0.15,
                explanation=explanation
            ))
        
        # ==================== Regime ====================
        
        regime_confidence = context.get("regime_confidence", 0.5)
        regime_impact = "bullish" if regime in ("BULL", "BREAKOUT") else ("bearish" if regime == "BEAR" else "neutral")
        
        factors.append(ReasoningFactor(
            category=ReasonCategory.REGIME,
            factor="Market Regime",
            value=regime,
            impact=regime_impact,
            weight=0.10,
            explanation=f"Market classified as {regime} ({regime_confidence:.0%} confidence)"
        ))
        
        if regime == "BULL":
            supporting_reasons.append(f"Bullish market regime ({regime_confidence:.0%})")
        elif regime == "BEAR" and action == "BUY":
            risk_considerations.append("Bear market regime")
        
        # ==================== Portfolio Context ====================
        
        if portfolio:
            exposure = portfolio.get("exposure_pct", 0)
            correlation = portfolio.get("correlation_with_existing", 0)
            
            if exposure > 0.6:
                risk_considerations.append(f"High portfolio exposure ({exposure:.0%})")
            
            if correlation > 0.7:
                risk_considerations.append(f"Highly correlated with existing positions ({correlation:.2f})")
                factors.append(ReasoningFactor(
                    category=ReasonCategory.PORTFOLIO,
                    factor="Correlation",
                    value=correlation,
                    impact="bearish",
                    weight=0.05,
                    explanation=f"Asset has {correlation:.2f} correlation with existing positions"
                ))
        
        # ==================== Risk ====================
        
        if risk:
            drawdown = risk.get("current_drawdown", 0)
            var_util = risk.get("var_utilization", 0)
            
            if drawdown > 0.05:
                risk_considerations.append(f"Portfolio in drawdown ({drawdown:.1%})")
            
            if var_util > 0.8:
                risk_considerations.append(f"High VaR utilization ({var_util:.0%})")
        
        # ==================== Generate Primary Reason ====================
        
        # Find highest-weighted bullish/bearish factor matching the action
        relevant_factors = [
            f for f in factors 
            if (action == "BUY" and f.impact == "bullish") or
               (action == "SELL" and f.impact == "bearish")
        ]
        relevant_factors.sort(key=lambda x: x.weight, reverse=True)
        
        if relevant_factors:
            primary_reason = relevant_factors[0].explanation
        else:
            primary_reason = f"Strategy {strategy} signal in {regime} regime"
        
        # ==================== Create Reasoning Object ====================
        
        reasoning = TradeReasoning(
            symbol=symbol,
            action=action,
            confidence=confidence,
            strategy=strategy,
            regime=regime,
            factors=factors,
            primary_reason=primary_reason,
            supporting_reasons=supporting_reasons,
            risk_considerations=risk_considerations,
            entry_price=prices.get("entry"),
            stop_loss=prices.get("stop_loss"),
            take_profit=prices.get("take_profit"),
            position_size=prices.get("position_size")
        )
        
        # Store in history
        self._reasoning_history.append(reasoning)
        if len(self._reasoning_history) > self._max_history:
            self._reasoning_history = self._reasoning_history[-self._max_history:]
        
        return reasoning
    
    def generate_hold_reasoning(self,
                                symbol: str,
                                current_position: Dict[str, Any],
                                context: Dict[str, Any]) -> TradeReasoning:
        """Generate reasoning for holding an existing position."""
        factors = []
        supporting_reasons = []
        risk_considerations = []
        
        entry_price = current_position.get("entry_price", 0)
        current_price = current_position.get("current_price", 0)
        pnl_pct = (current_price - entry_price) / entry_price * 100 if entry_price > 0 else 0
        
        # Check if in profit
        if pnl_pct > 0:
            supporting_reasons.append(f"Position in profit ({pnl_pct:.1f}%)")
            factors.append(ReasoningFactor(
                category=ReasonCategory.TECHNICAL,
                factor="Unrealized P/L",
                value=pnl_pct,
                impact="bullish",
                weight=0.15,
                explanation=f"Position is {pnl_pct:.1f}% in profit"
            ))
        else:
            risk_considerations.append(f"Position at {pnl_pct:.1f}% loss")
        
        # Check trend continuation
        trend = context.get("technical", {}).get("trend")
        if trend == "UP":
            supporting_reasons.append("Uptrend continuing")
        elif trend == "DOWN":
            risk_considerations.append("Trend turning bearish")
        
        # Check stop/target proximity
        stop_loss = current_position.get("stop_loss")
        take_profit = current_position.get("take_profit")
        
        if stop_loss and current_price:
            stop_dist = (current_price - stop_loss) / current_price * 100
            if stop_dist < 1:
                risk_considerations.append(f"Close to stop loss ({stop_dist:.1f}%)")
        
        if take_profit and current_price:
            tp_dist = (take_profit - current_price) / current_price * 100
            if tp_dist < 2:
                supporting_reasons.append(f"Approaching take profit ({tp_dist:.1f}%)")
        
        primary_reason = "Position still valid - maintaining hold"
        if supporting_reasons:
            primary_reason = supporting_reasons[0]
        
        return TradeReasoning(
            symbol=symbol,
            action="HOLD",
            confidence=0.5 + min(pnl_pct / 100, 0.3),  # Higher confidence if in profit
            strategy=current_position.get("strategy", "unknown"),
            regime=context.get("regime", "UNKNOWN"),
            factors=factors,
            primary_reason=primary_reason,
            supporting_reasons=supporting_reasons,
            risk_considerations=risk_considerations,
            entry_price=entry_price,
            stop_loss=stop_loss,
            take_profit=take_profit
        )
    
    def generate_exit_reasoning(self,
                                symbol: str,
                                exit_type: str,  # "TP", "SL", "SIGNAL", "MANUAL"
                                current_position: Dict[str, Any],
                                context: Dict[str, Any]) -> TradeReasoning:
        """Generate reasoning for exiting a position."""
        factors = []
        supporting_reasons = []
        
        entry_price = current_position.get("entry_price", 0)
        exit_price = current_position.get("exit_price") or current_position.get("current_price", 0)
        pnl_pct = (exit_price - entry_price) / entry_price * 100 if entry_price > 0 else 0
        
        exit_reasons = {
            "TP": f"Take profit target reached at ${exit_price:.2f}",
            "SL": f"Stop loss triggered at ${exit_price:.2f}",
            "SIGNAL": "Exit signal from strategy",
            "MANUAL": "Manual exit requested",
            "REGIME_CHANGE": "Market regime changed",
            "RISK": "Risk limit exceeded"
        }
        
        primary_reason = exit_reasons.get(exit_type, "Position closed")
        
        if pnl_pct > 0:
            supporting_reasons.append(f"Profit of {pnl_pct:.1f}% realized")
        else:
            supporting_reasons.append(f"Loss of {pnl_pct:.1f}% realized")
        
        if exit_type == "TP":
            supporting_reasons.append("Target reached as planned")
        elif exit_type == "SL":
            supporting_reasons.append("Risk management executed correctly")
        
        return TradeReasoning(
            symbol=symbol,
            action="SELL",
            confidence=0.9,  # Exit decisions are usually high confidence
            strategy=current_position.get("strategy", "unknown"),
            regime=context.get("regime", "UNKNOWN"),
            factors=factors,
            primary_reason=primary_reason,
            supporting_reasons=supporting_reasons,
            risk_considerations=[],
            entry_price=entry_price,
            stop_loss=None,
            take_profit=None
        )
    
    def get_recent_reasoning(self, limit: int = 10) -> List[TradeReasoning]:
        """Get recent reasoning history."""
        return self._reasoning_history[-limit:]
    
    def get_reasoning_for_symbol(self, symbol: str, limit: int = 5) -> List[TradeReasoning]:
        """Get recent reasoning for a specific symbol."""
        matching = [r for r in self._reasoning_history if r.symbol == symbol]
        return matching[-limit:]


# Singleton instance
_reasoning_engine: Optional[TradeReasoningEngine] = None


def get_reasoning_engine() -> TradeReasoningEngine:
    """Get or create the reasoning engine singleton."""
    global _reasoning_engine
    if _reasoning_engine is None:
        _reasoning_engine = TradeReasoningEngine()
    return _reasoning_engine
