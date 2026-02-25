"""
Phase 2 Intelligence: Sentiment Analysis
Analyzes social media (Twitter/Reddit) and news to gauge market sentiment.
News: NewsAPI, Finnhub, Benzinga. Analyst upgrades via Finnhub.
"""
import time
import os
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Optional, Tuple, Any
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class SentimentScore(Enum):
    """Sentiment levels"""
    VERY_POSITIVE = "VERY_POSITIVE"  # >0.5
    POSITIVE = "POSITIVE"            # 0.2 to 0.5
    NEUTRAL = "NEUTRAL"              # -0.2 to 0.2
    NEGATIVE = "NEGATIVE"            # -0.5 to -0.2
    VERY_NEGATIVE = "VERY_NEGATIVE"  # <-0.5


class SentimentAnalyzer:
    """
    Aggregates sentiment from multiple sources:
    1. Twitter API (mentions, hashtags, sentiment)
    2. News API (financial news headlines)
    3. Reddit API (discussion volume, upvotes)
    4. Fear & Greed Index (market-wide sentiment)
    
    Note: Requires API keys (set as environment variables):
    - TWITTER_API_KEY (optional, uses free tier)
    - NEWS_API_KEY (optional, free tier available)
    - Reddit API is free, no key needed
    """
    
    def __init__(
        self,
        twitter_weight: float = 0.30,
        news_weight: float = 0.40,
        reddit_weight: float = 0.20,
        fear_greed_weight: float = 0.10
    ):
        self.twitter_weight = twitter_weight
        self.news_weight = news_weight
        self.reddit_weight = reddit_weight
        self.fear_greed_weight = fear_greed_weight
        
        # API keys (optional)
        self.twitter_api_key = os.getenv('TWITTER_API_KEY')
        self.news_api_key = os.getenv('NEWS_API_KEY')
        self.finnhub_key = os.getenv('FINNHUB_API_KEY', '')
        self.benzinga_key = os.getenv('BENZINGA_API_KEY', '')
        self.gate_enabled = os.getenv("SENTIMENT_GATE_ENABLED", "0").strip().lower() in (
            "1",
            "true",
            "yes",
            "y",
            "on",
        )
    
    def get_combined_sentiment(
        self,
        symbol: str,
        lookback_hours: int = 24
    ) -> Tuple[SentimentScore, float, Dict[str, Any]]:
        """
        Get combined sentiment score from all sources.
        
        Args:
            symbol: Trading symbol (e.g., "BTC", "AAPL")
            lookback_hours: Hours to look back for sentiment data
        
        Returns:
            (sentiment_level, score, details)
            score is -1 to +1 scale
        """
        scores: Dict[str, float] = {}
        available: Dict[str, bool] = {}

        placeholder_sources = []

        # 1. Twitter sentiment (only if real data available)
        twitter_score, twitter_available, twitter_placeholder = self._get_twitter_sentiment(symbol, lookback_hours)
        scores['twitter'] = twitter_score
        available['twitter'] = twitter_available
        if twitter_placeholder:
            placeholder_sources.append("twitter")

        # 2. News sentiment (only if real data available)
        news_score, news_available, news_placeholder = self._get_news_sentiment(symbol, lookback_hours)
        scores['news'] = news_score
        available['news'] = news_available
        if news_placeholder:
            placeholder_sources.append("news")

        # 3. Reddit sentiment (stubbed -> unavailable until implemented)
        reddit_score, reddit_available, reddit_placeholder = self._get_reddit_sentiment(symbol, lookback_hours)
        scores['reddit'] = reddit_score
        available['reddit'] = reddit_available
        if reddit_placeholder:
            placeholder_sources.append("reddit")

        # 4. Fear & Greed Index (market-wide, crypto only)
        fear_greed_score = 0.0
        fear_greed_available = False
        if 'BTC' in symbol or 'ETH' in symbol:
            fear_greed_score, fear_greed_available = self._get_fear_greed_index()
        scores['fear_greed'] = fear_greed_score
        available['fear_greed'] = fear_greed_available

        # Calculate weighted average using only available sources
        weights = {
            'twitter': self.twitter_weight if twitter_available else 0.0,
            'news': self.news_weight if news_available else 0.0,
            'reddit': self.reddit_weight if reddit_available else 0.0,
            'fear_greed': self.fear_greed_weight if fear_greed_available else 0.0,
        }
        total_weight = sum(weights.values())
        if total_weight > 0:
            combined_score = (
                twitter_score * weights['twitter'] +
                news_score * weights['news'] +
                reddit_score * weights['reddit'] +
                fear_greed_score * weights['fear_greed']
            ) / total_weight
        else:
            combined_score = 0.0
        
        # Classify sentiment
        if combined_score > 0.5:
            sentiment = SentimentScore.VERY_POSITIVE
        elif combined_score > 0.2:
            sentiment = SentimentScore.POSITIVE
        elif combined_score > -0.2:
            sentiment = SentimentScore.NEUTRAL
        elif combined_score > -0.5:
            sentiment = SentimentScore.NEGATIVE
        else:
            sentiment = SentimentScore.VERY_NEGATIVE
        
        details = {
            'combined_score': combined_score,
            'twitter_score': twitter_score,
            'news_score': news_score,
            'reddit_score': reddit_score,
            'fear_greed_score': fear_greed_score,
            'weights': weights,
            'availability': available,
            'sentiment_unavailable': total_weight == 0,
            'placeholders_used': len(placeholder_sources) > 0,
            'placeholder_sources': placeholder_sources,
            'gate_enabled': self.gate_enabled,
        }
        
        return sentiment, combined_score, details
    
    def _get_twitter_sentiment(self, symbol: str, lookback_hours: int) -> Tuple[float, bool, bool]:
        """
        Analyze Twitter sentiment for symbol.
        
        Note: Requires Twitter API v2 (free tier: 500k tweets/month)
        """
        if not self.twitter_api_key:
            logger.debug("Twitter API key not set, returning neutral sentiment")
            return 0.0, False, True
        
        try:
            # Not implemented: Requires Twitter API v2 (API key + approval)
            # To implement: search tweets by symbol/hashtag, analyze sentiment, weight by engagement
            return 0.0, False, True  # Neutral stub when API not configured
            
        except Exception as e:
            logger.error(f"Twitter API error: {e}")
            return 0.0, False, True
    
    def _get_news_sentiment(self, symbol: str, lookback_hours: int) -> Tuple[float, bool, bool]:
        """
        Analyze news sentiment - NewsAPI, Finnhub, Benzinga.
        """
        sym = symbol.split("/")[0].upper()
        if sym in ("BTC", "XBT", "ETH"):
            sym_search = "bitcoin" if "BTC" in sym or "XBT" in sym else "ethereum"
        else:
            sym_search = sym
        if self.news_api_key:
            try:
                import requests
                from_date = (datetime.now(timezone.utc) - timedelta(hours=lookback_hours)).strftime("%Y-%m-%d")
                url = "https://newsapi.org/v2/everything"
                params = {
                    "q": sym_search,
                    "from": from_date,
                    "apiKey": self.news_api_key,
                    "language": "en",
                    "sortBy": "publishedAt",
                    "pageSize": 20,
                }
                r = requests.get(url, params=params, timeout=5)
                if r.status_code == 200:
                    articles = r.json().get("articles", [])
                    if articles:
                        kw = KeywordSentimentAnalyzer()
                        scores = [kw.analyze_text(a.get("title", "") + " " + a.get("description", "")) for a in articles]
                        avg = sum(scores) / len(scores) if scores else 0.0
                        return avg, True, False
            except Exception as e:
                logger.debug("News API error: %s", e)
        if self.finnhub_key:
            try:
                import requests
                url = "https://finnhub.io/api/v1/company-news"
                from_d = (datetime.now(timezone.utc) - timedelta(hours=max(1, lookback_hours))).strftime("%Y-%m-%d")
                to_d = datetime.now(timezone.utc).strftime("%Y-%m-%d")
                r = requests.get(url, params={"symbol": sym, "from": from_d, "to": to_d, "token": self.finnhub_key}, timeout=5)
                if r.status_code == 200:
                    articles = r.json()
                    if articles:
                        kw = KeywordSentimentAnalyzer()
                        scores = [kw.analyze_text(a.get("headline", "")) for a in articles[:15]]
                        avg = sum(scores) / len(scores) if scores else 0.0
                        return avg, True, False
            except Exception as e:
                logger.debug("Finnhub news error: %s", e)
        return 0.0, False, True
    
    def _get_reddit_sentiment(self, symbol: str, lookback_hours: int) -> Tuple[float, bool, bool]:
        """
        Analyze Reddit sentiment for symbol.
        
        Note: Reddit API is free, no key required
        Subreddits to check: r/wallstreetbets, r/CryptoCurrency, r/stocks
        """
        try:
            # Not implemented: Reddit API (PRAW) requires app credentials
            # To implement: search r/CryptoCurrency, r/wallstreetbets; analyze mentions
            return 0.0, False, True  # Neutral stub when not configured
            
        except Exception as e:
            logger.error(f"Reddit API error: {e}")
            return 0.0, False, True
    
    def _get_fear_greed_index(self) -> Tuple[float, bool]:
        """
        Get Crypto Fear & Greed Index (market-wide sentiment).
        
        API: https://api.alternative.me/fng/ (Free, no key needed)
        Returns: 0-100 scale (0=extreme fear, 100=extreme greed)
        """
        try:
            import requests
            
            url = "https://api.alternative.me/fng/"
            response = requests.get(url, timeout=5)
            
            if response.status_code == 200:
                data = response.json()
                value = int(data['data'][0]['value'])
                
                # Convert 0-100 to -1 to +1 scale
                # 0 (extreme fear) = -1
                # 50 (neutral) = 0
                # 100 (extreme greed) = +1
                score = (value - 50) / 50.0
                
                logger.info(f"Fear & Greed Index: {value}/100 (score: {score:.2f})")
                return score, True
            else:
                logger.warning(f"Fear & Greed API returned {response.status_code}")
                return 0.0, False
                
        except Exception as e:
            logger.error(f"Fear & Greed API error: {e}")
            return 0.0, False
    
    def get_analyst_recommendations(self, symbol: str) -> Tuple[Optional[float], List[Dict[str, Any]]]:
        """
        Finnhub analyst recommendations: upgrades/downgrades, price targets.
        Returns: (avg_target_upside_pct or None, list of recs).
        """
        if not self.finnhub_key:
            return None, []
        try:
            import requests
            sym = symbol.split("/")[0].upper()
            r = requests.get(
                "https://finnhub.io/api/v1/stock/recommendation",
                params={"symbol": sym, "token": self.finnhub_key},
                timeout=5,
            )
            if r.status_code != 200:
                return None, []
            data = r.json()
            recs = []
            for m in data[:3]:
                recs.append({
                    "period": m.get("period"),
                    "strongBuy": m.get("strongBuy"),
                    "buy": m.get("buy"),
                    "hold": m.get("hold"),
                    "sell": m.get("sell"),
                    "strongSell": m.get("strongSell"),
                })
            return None, recs  # No price target in this endpoint
        except Exception as e:
            logger.debug("Analyst recs error: %s", e)
            return None, []

    def get_breaking_news_alert(self, symbol: str, lookback_hours: int = 6) -> Optional[str]:
        """
        Check for highly negative news that should pause trading.
        Returns reason string if pause recommended, else None.
        """
        if not self.news_api_key and not self.finnhub_key:
            return None
        sym = symbol.split("/")[0]
        neg_keywords = ["bankruptcy", "fraud", "sec investigation", "recall", "layoff", "lawsuit", "default"]
        try:
            import requests
            if self.finnhub_key:
                url = "https://finnhub.io/api/v1/company-news"
                from_d = (datetime.now(timezone.utc) - timedelta(hours=lookback_hours)).strftime("%Y-%m-%d")
                to_d = datetime.now(timezone.utc).strftime("%Y-%m-%d")
                r = requests.get(url, params={"symbol": sym, "from": from_d, "to": to_d, "token": self.finnhub_key}, timeout=5)
                if r.status_code == 200:
                    for a in r.json()[:10]:
                        h = (a.get("headline") or "").lower()
                        for kw in neg_keywords:
                            if kw in h:
                                return f"Negative news: {a.get('headline', '')[:80]}"
        except Exception:
            pass
        return None

    def should_trade(
        self,
        symbol: str,
        min_sentiment_score: float = -0.3
    ) -> Tuple[bool, str]:
        """
        Determine if sentiment allows trading.
        
        Args:
            symbol: Trading symbol
            min_sentiment_score: Minimum acceptable sentiment (-1 to +1)
        
        Returns:
            (allowed, reason)
        """
        sentiment, score, details = self.get_combined_sentiment(symbol)
        if not self.gate_enabled:
            return True, "Sentiment gate disabled (SENTIMENT_GATE_ENABLED=0)"
        if details.get("sentiment_unavailable") or details.get("placeholders_used"):
            return True, "Sentiment unavailable (placeholder sources)"
        
        _sv = sentiment.value if hasattr(sentiment, "value") else str(sentiment)
        if score < min_sentiment_score:
            return False, f"Negative sentiment: {_sv} ({score:.2f})"
        return True, f"Sentiment OK: {_sv} ({score:.2f})"


class KeywordSentimentAnalyzer:
    """
    Simple keyword-based sentiment analysis (no API required).
    Analyzes text for positive/negative keywords.
    """
    
    def __init__(self):
        self.positive_keywords = [
            'bullish', 'bull', 'moon', 'pump', 'rally', 'breakout', 'surge',
            'strong', 'buy', 'long', 'profit', 'gain', 'rise', 'up', 'high',
            'growth', 'winner', 'success', 'boom', 'explosion', 'rocket'
        ]
        
        self.negative_keywords = [
            'bearish', 'bear', 'dump', 'crash', 'drop', 'fall', 'down', 'low',
            'sell', 'short', 'loss', 'decline', 'weak', 'fail', 'collapse',
            'plunge', 'tank', 'bloodbath', 'rekt', 'panic'
        ]
    
    def analyze_text(self, text: str) -> float:
        """
        Analyze sentiment of text using keyword matching.
        
        Args:
            text: Text to analyze (lowercase)
        
        Returns:
            Sentiment score -1 to +1
        """
        if not text:
            return 0.0
        
        text_lower = text.lower()
        
        # Count positive and negative keywords
        positive_count = sum(1 for kw in self.positive_keywords if kw in text_lower)
        negative_count = sum(1 for kw in self.negative_keywords if kw in text_lower)
        
        total_count = positive_count + negative_count
        
        if total_count == 0:
            return 0.0  # Neutral
        
        # Calculate sentiment score
        score = (positive_count - negative_count) / total_count
        
        return score


# =============================================================================
# Testing / Example Usage
# =============================================================================

if __name__ == "__main__":
    # Example 1: Combined sentiment (requires API keys)
    analyzer = SentimentAnalyzer()
    
    symbol = "BTC"
    sentiment, score, details = analyzer.get_combined_sentiment(symbol)
    
    print(f"=== Sentiment Analysis for {symbol} ===")
    print(f"Overall Sentiment: {sentiment.value}")
    print(f"Score: {score:.2f} (-1 to +1)")
    print(f"Twitter: {details['twitter_score']:.2f}")
    print(f"News: {details['news_score']:.2f}")
    print(f"Reddit: {details['reddit_score']:.2f}")
    print(f"Fear & Greed: {details['fear_greed_score']:.2f}")
    print()
    
    # Example 2: Keyword-based sentiment (no API needed)
    keyword_analyzer = KeywordSentimentAnalyzer()
    
    texts = [
        "Bitcoin is looking very bullish! Strong breakout expected.",
        "Market crash incoming, bearish trend, sell everything!",
        "Neutral market, sideways trading continues."
    ]
    
    print("=== Keyword Sentiment Analysis ===")
    for text in texts:
        score = keyword_analyzer.analyze_text(text)
        print(f"Text: {text}")
        print(f"Score: {score:.2f}")
        print()
