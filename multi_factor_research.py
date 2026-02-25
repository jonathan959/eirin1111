"""
Multi-Factor Research - Comprehensive asset analysis for autonomous trading.

Features:
- Sentiment Analysis (news, social media, Fear & Greed)
- On-Chain Metrics (exchange flows, whale activity) - Crypto only
- Macroeconomic & Event Context (earnings, Fed decisions)
- Diversification & Correlation Analysis

Based on: "Upgrading to an Autonomous Multi-Asset Trading Bot" specification.
"""

import os
import time
import json
import logging
import requests
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timezone, timedelta
from collections import deque

logger = logging.getLogger(__name__)


@dataclass
class SentimentData:
    """Sentiment analysis results."""
    score: float  # -1 (bearish) to +1 (bullish)
    magnitude: float  # 0 to 1 (strength)
    sources: Dict[str, float]  # Source-specific scores
    fear_greed_index: Optional[int] = None  # 0-100
    social_volume_change: Optional[float] = None  # % change
    news_count_24h: int = 0
    bullish_mentions: int = 0
    bearish_mentions: int = 0
    timestamp: int = field(default_factory=lambda: int(time.time()))
    
    @property
    def signal(self) -> str:
        if self.score > 0.3:
            return "bullish"
        elif self.score < -0.3:
            return "bearish"
        return "neutral"


@dataclass
class OnChainMetrics:
    """On-chain analysis for crypto assets."""
    exchange_netflow: float  # Positive = inflow (bearish), Negative = outflow (bullish)
    whale_transactions: int  # Large transactions in 24h
    active_addresses_change: float  # % change in active addresses
    nvt_ratio: Optional[float] = None  # Network Value to Transactions
    mvrv_ratio: Optional[float] = None  # Market Value to Realized Value
    supply_on_exchanges_pct: Optional[float] = None
    staking_ratio: Optional[float] = None
    timestamp: int = field(default_factory=lambda: int(time.time()))
    
    @property
    def signal(self) -> str:
        # Exchange outflow (negative) is bullish
        if self.exchange_netflow < -1000000:  # Large outflow
            return "bullish"
        elif self.exchange_netflow > 1000000:  # Large inflow
            return "bearish"
        return "neutral"


@dataclass
class MacroContext:
    """Macroeconomic and event context."""
    spy_trend: str  # "up", "down", "sideways"
    vix_level: float  # Volatility index
    risk_sentiment: str  # "risk_on", "risk_off", "neutral"
    
    # Upcoming events
    earnings_soon: bool = False
    earnings_date: Optional[str] = None
    fed_meeting_soon: bool = False
    major_event_pending: bool = False
    event_description: Optional[str] = None
    
    # Market conditions
    market_open: bool = True
    pre_market: bool = False
    after_hours: bool = False
    
    timestamp: int = field(default_factory=lambda: int(time.time()))


@dataclass
class ResearchScore:
    """Combined multi-factor research score."""
    symbol: str
    overall_score: float  # -100 to +100
    
    # Component scores
    technical_score: float
    sentiment_score: float
    onchain_score: Optional[float]  # Crypto only
    macro_score: float
    diversification_score: float
    
    # Weights used
    weights: Dict[str, float]
    
    # Signals
    signals: Dict[str, str]  # Factor -> signal
    
    # Risk flags
    risk_flags: List[str]
    
    # Recommendation
    action: str  # "STRONG_BUY", "BUY", "HOLD", "SELL", "STRONG_SELL"
    confidence: float
    
    # Reasoning
    primary_reason: str
    supporting_factors: List[str]
    
    timestamp: int = field(default_factory=lambda: int(time.time()))


class SentimentAnalyzer:
    """
    Analyzes market sentiment from various sources.
    """
    
    def __init__(self):
        self._cache: Dict[str, SentimentData] = {}
        self._cache_ttl = 300  # 5 minutes
        
        # API keys (from environment)
        self._news_api_key = os.getenv("NEWS_API_KEY", "")
        
    def analyze(self, symbol: str, market_type: str = "crypto") -> SentimentData:
        """
        Analyze sentiment for a symbol.
        
        Uses:
        - Fear & Greed Index (crypto)
        - News sentiment
        - Social media signals
        """
        cache_key = f"{symbol}_{market_type}"
        
        # Check cache
        if cache_key in self._cache:
            cached = self._cache[cache_key]
            if time.time() - cached.timestamp < self._cache_ttl:
                return cached
        
        sentiment = SentimentData(
            score=0.0,
            magnitude=0.5,
            sources={}
        )
        
        try:
            # Get Fear & Greed for crypto
            if market_type == "crypto":
                fg = self._get_fear_greed()
                if fg is not None:
                    sentiment.fear_greed_index = fg
                    # Convert to -1 to 1 scale
                    fg_score = (fg - 50) / 50  # 0-100 -> -1 to 1
                    sentiment.sources["fear_greed"] = fg_score
            
            # Simple keyword-based news sentiment (placeholder)
            news_score = self._analyze_news_simple(symbol)
            if news_score is not None:
                sentiment.sources["news"] = news_score
            
            # Calculate overall score (weighted average)
            if sentiment.sources:
                weights = {
                    "fear_greed": 0.4,
                    "news": 0.4,
                    "social": 0.2
                }
                
                total_weight = 0
                weighted_sum = 0
                for source, score in sentiment.sources.items():
                    w = weights.get(source, 0.2)
                    weighted_sum += score * w
                    total_weight += w
                
                if total_weight > 0:
                    sentiment.score = weighted_sum / total_weight
                    sentiment.magnitude = min(1.0, total_weight)
            
        except Exception as e:
            logger.error(f"Sentiment analysis failed for {symbol}: {e}")
        
        # Cache result
        self._cache[cache_key] = sentiment
        
        return sentiment
    
    def _get_fear_greed(self) -> Optional[int]:
        """Get crypto Fear & Greed Index."""
        try:
            response = requests.get(
                "https://api.alternative.me/fng/",
                timeout=5
            )
            if response.status_code == 200:
                data = response.json()
                if data.get("data"):
                    return int(data["data"][0].get("value", 50))
        except Exception as e:
            logger.debug(f"Fear & Greed fetch failed: {e}")
        return None
    
    def _analyze_news_simple(self, symbol: str) -> Optional[float]:
        """Simple news sentiment (placeholder for more advanced NLP)."""
        # This would integrate with a news API and NLP model
        # For now, return neutral
        return 0.0


class OnChainAnalyzer:
    """
    Analyzes on-chain metrics for crypto assets.
    """
    
    def __init__(self):
        self._cache: Dict[str, OnChainMetrics] = {}
        self._cache_ttl = 600  # 10 minutes
    
    def analyze(self, symbol: str) -> Optional[OnChainMetrics]:
        """
        Analyze on-chain metrics for a crypto symbol.
        
        Uses:
        - Exchange flows
        - Whale activity
        - Network activity
        """
        # Only for crypto
        if "/" not in symbol:
            return None
        
        base = symbol.split("/")[0]
        
        # Check cache
        if base in self._cache:
            cached = self._cache[base]
            if time.time() - cached.timestamp < self._cache_ttl:
                return cached
        
        try:
            metrics = OnChainMetrics(
                exchange_netflow=0.0,
                whale_transactions=0,
                active_addresses_change=0.0
            )
            
            # This would integrate with on-chain data providers like:
            # - Glassnode
            # - CryptoQuant
            # - Santiment
            # For now, use placeholder data
            
            # Cache result
            self._cache[base] = metrics
            return metrics
            
        except Exception as e:
            logger.error(f"On-chain analysis failed for {symbol}: {e}")
            return None


class MacroAnalyzer:
    """
    Analyzes macroeconomic context and events.
    """
    
    def __init__(self):
        self._spy_cache = None
        self._spy_cache_ts = 0
        self._vix_cache = None
        self._vix_cache_ts = 0
        self._earnings_cache: Dict[str, Dict] = {}
    
    def analyze(self, symbol: str, market_type: str = "crypto") -> MacroContext:
        """
        Analyze macro context for trading.
        """
        context = MacroContext(
            spy_trend="sideways",
            vix_level=15.0,
            risk_sentiment="neutral"
        )
        
        try:
            # Get SPY trend for overall market direction
            spy_trend = self._get_spy_trend()
            if spy_trend:
                context.spy_trend = spy_trend
            
            # Get VIX for volatility/fear
            vix = self._get_vix_level()
            if vix:
                context.vix_level = vix
                # VIX > 25 = risk off, VIX < 15 = risk on
                if vix > 25:
                    context.risk_sentiment = "risk_off"
                elif vix < 15:
                    context.risk_sentiment = "risk_on"
            
            # Check for earnings (stocks only)
            if market_type == "stocks":
                earnings = self._check_earnings(symbol)
                if earnings:
                    context.earnings_soon = earnings.get("soon", False)
                    context.earnings_date = earnings.get("date")
            
            # Check for Fed meetings
            context.fed_meeting_soon = self._check_fed_meeting()
            
        except Exception as e:
            logger.error(f"Macro analysis failed: {e}")
        
        return context
    
    def _get_spy_trend(self) -> Optional[str]:
        """Get SPY trend direction."""
        # Would integrate with market data API
        # Returns "up", "down", or "sideways"
        return "sideways"
    
    def _get_vix_level(self) -> Optional[float]:
        """Get current VIX level."""
        # Would integrate with market data API
        return 18.0
    
    def _check_earnings(self, symbol: str) -> Optional[Dict]:
        """Check if earnings are upcoming."""
        # Would integrate with earnings calendar API
        return None
    
    def _check_fed_meeting(self) -> bool:
        """Check if Fed meeting is in next 7 days."""
        # Would integrate with economic calendar
        return False


class MultiFactorResearch:
    """
    Comprehensive multi-factor research engine.
    
    Combines:
    - Technical analysis (from strategies)
    - Sentiment analysis
    - On-chain metrics (crypto)
    - Macro context
    - Portfolio diversification
    """
    
    def __init__(self):
        self.sentiment = SentimentAnalyzer()
        self.onchain = OnChainAnalyzer()
        self.macro = MacroAnalyzer()
        
        # Scoring weights by market type
        self._weights = {
            "crypto": {
                "technical": 0.35,
                "sentiment": 0.25,
                "onchain": 0.20,
                "macro": 0.10,
                "diversification": 0.10
            },
            "stocks": {
                "technical": 0.40,
                "sentiment": 0.20,
                "onchain": 0.0,  # Not applicable
                "macro": 0.25,
                "diversification": 0.15
            }
        }
    
    def research(self,
                 symbol: str,
                 market_type: str,
                 technical_score: float,
                 diversification_score: float,
                 existing_positions: List[str] = None) -> ResearchScore:
        """
        Perform comprehensive multi-factor research.
        
        Args:
            symbol: Trading symbol
            market_type: "crypto" or "stocks"
            technical_score: Score from technical analysis (-100 to 100)
            diversification_score: How well this fits portfolio (0 to 1)
            existing_positions: List of current positions for correlation check
        
        Returns:
            ResearchScore with overall recommendation
        """
        weights = self._weights.get(market_type, self._weights["crypto"])
        signals = {}
        risk_flags = []
        supporting = []
        
        # Technical (already calculated)
        tech_normalized = technical_score / 100  # -1 to 1
        signals["technical"] = "bullish" if tech_normalized > 0.3 else ("bearish" if tech_normalized < -0.3 else "neutral")
        
        # Sentiment
        sentiment = self.sentiment.analyze(symbol, market_type)
        sent_score = sentiment.score * 100  # -100 to 100
        signals["sentiment"] = sentiment.signal
        
        if sentiment.fear_greed_index is not None:
            if sentiment.fear_greed_index < 25:
                supporting.append(f"Extreme Fear ({sentiment.fear_greed_index}) - contrarian bullish")
            elif sentiment.fear_greed_index > 75:
                risk_flags.append(f"Extreme Greed ({sentiment.fear_greed_index}) - potential top")
        
        # On-chain (crypto only)
        onchain_score = 0.0
        if market_type == "crypto":
            onchain = self.onchain.analyze(symbol)
            if onchain:
                signals["onchain"] = onchain.signal
                if onchain.signal == "bullish":
                    onchain_score = 50
                    supporting.append("Bullish on-chain flows")
                elif onchain.signal == "bearish":
                    onchain_score = -50
                    risk_flags.append("Bearish on-chain flows")
        
        # Macro
        macro = self.macro.analyze(symbol, market_type)
        macro_score = 0.0
        
        if macro.risk_sentiment == "risk_off":
            macro_score = -30
            risk_flags.append("Risk-off macro environment")
        elif macro.risk_sentiment == "risk_on":
            macro_score = 30
            supporting.append("Risk-on macro environment")
        
        if macro.earnings_soon:
            risk_flags.append(f"Earnings approaching ({macro.earnings_date})")
            macro_score -= 20  # Penalty for event risk
        
        if macro.fed_meeting_soon:
            risk_flags.append("Fed meeting soon - expect volatility")
            macro_score -= 10
        
        signals["macro"] = "bullish" if macro_score > 20 else ("bearish" if macro_score < -20 else "neutral")
        
        # Diversification
        div_score = diversification_score * 100  # 0-100
        signals["diversification"] = "bullish" if diversification_score > 0.5 else "bearish"
        
        if diversification_score < 0.3:
            risk_flags.append("Low diversification benefit")
        
        # Calculate overall score
        component_scores = {
            "technical": technical_score,
            "sentiment": sent_score,
            "onchain": onchain_score,
            "macro": macro_score,
            "diversification": div_score
        }
        
        weighted_sum = 0
        total_weight = 0
        for factor, score in component_scores.items():
            w = weights.get(factor, 0)
            weighted_sum += score * w
            total_weight += w
        
        overall_score = weighted_sum / total_weight if total_weight > 0 else 0
        
        # Determine action and confidence
        if overall_score >= 60:
            action = "STRONG_BUY"
            confidence = min(0.95, 0.7 + (overall_score - 60) / 100)
        elif overall_score >= 30:
            action = "BUY"
            confidence = 0.6 + (overall_score - 30) / 100
        elif overall_score <= -60:
            action = "STRONG_SELL"
            confidence = min(0.9, 0.6 + abs(overall_score + 60) / 100)
        elif overall_score <= -30:
            action = "SELL"
            confidence = 0.5 + abs(overall_score + 30) / 100
        else:
            action = "HOLD"
            confidence = 0.5
        
        # Reduce confidence if many risk flags
        if len(risk_flags) >= 3:
            confidence *= 0.8
            action = "HOLD" if action in ("BUY", "SELL") else action
        
        # Generate primary reason
        best_factor = max(
            [(k, v) for k, v in component_scores.items() if weights.get(k, 0) > 0],
            key=lambda x: abs(x[1]) * weights.get(x[0], 0),
            default=("technical", 0)
        )
        
        reason_map = {
            "technical": f"Strong technical setup (score: {technical_score:.0f})",
            "sentiment": f"{'Bullish' if sentiment.score > 0 else 'Bearish'} market sentiment",
            "onchain": "Favorable on-chain metrics",
            "macro": f"{'Supportive' if macro_score > 0 else 'Challenging'} macro environment",
            "diversification": "Good portfolio fit"
        }
        
        primary_reason = reason_map.get(best_factor[0], "Multi-factor analysis")
        
        return ResearchScore(
            symbol=symbol,
            overall_score=overall_score,
            technical_score=technical_score,
            sentiment_score=sent_score,
            onchain_score=onchain_score if market_type == "crypto" else None,
            macro_score=macro_score,
            diversification_score=div_score,
            weights=weights,
            signals=signals,
            risk_flags=risk_flags,
            action=action,
            confidence=confidence,
            primary_reason=primary_reason,
            supporting_factors=supporting
        )
    
    def quick_score(self, symbol: str, market_type: str, technical_score: float) -> Tuple[float, str]:
        """
        Quick scoring without full research (for screening).
        
        Returns (score, action)
        """
        # Get sentiment quickly
        sentiment = self.sentiment.analyze(symbol, market_type)
        
        # Simple weighted average
        if market_type == "crypto":
            score = technical_score * 0.6 + sentiment.score * 100 * 0.4
        else:
            score = technical_score * 0.7 + sentiment.score * 100 * 0.3
        
        if score >= 50:
            action = "BUY"
        elif score <= -50:
            action = "SELL"
        else:
            action = "HOLD"
        
        return score, action


# Singleton instance
_research_engine: Optional[MultiFactorResearch] = None


def get_research_engine() -> MultiFactorResearch:
    """Get or create the research engine singleton."""
    global _research_engine
    if _research_engine is None:
        _research_engine = MultiFactorResearch()
    return _research_engine
