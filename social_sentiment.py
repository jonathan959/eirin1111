"""
Social Sentiment Integration - Twitter, Reddit, StockTwits.
Score: -10 to +10 (very bearish to very bullish).
"""
import os
import logging
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# Config
ENABLE_SOCIAL_SENTIMENT = os.getenv("ENABLE_SOCIAL_SENTIMENT", "0").strip().lower() in ("1", "true", "yes")
TWITTER_BEARER_TOKEN = os.getenv("TWITTER_BEARER_TOKEN", "")
REDDIT_CLIENT_ID = os.getenv("REDDIT_CLIENT_ID", "")
REDDIT_CLIENT_SECRET = os.getenv("REDDIT_CLIENT_SECRET", "")


@dataclass
class SocialSentimentResult:
    score: float  # -10 to +10
    mention_volume: int = 0
    sentiment_polarity: float = 0.0  # -1 to +1
    velocity_24h: float = 0.0
    sources_used: List[str] = field(default_factory=list)
    details: Dict[str, Any] = field(default_factory=dict)


def _normalize_symbol(symbol: str) -> str:
    """AAPL, BTC/USD -> AAPL, BTC."""
    s = symbol.split("/")[0].upper()
    return s.replace("XBT", "BTC")


def _get_twitter_sentiment(symbol: str, lookback_hours: int = 24) -> Tuple[float, int, float, bool]:
    """
    Twitter API v2 - mentions, sentiment, velocity.
    Returns: (polarity -1..1, volume, velocity, available).
    """
    if not TWITTER_BEARER_TOKEN:
        return 0.0, 0, 0.0, False
    try:
        import requests
        sym = _normalize_symbol(symbol)
        headers = {"Authorization": f"Bearer {TWITTER_BEARER_TOKEN}"}
        # Search recent tweets (recent 7 days on free tier)
        params = {
            "query": f"${sym} OR #{sym}",
            "max_results": 100,
            "tweet.fields": "created_at,public_metrics",
            "user.fields": "public_metrics",
        }
        # Not implemented: Twitter API v2 requires project approval
        # url = "https://api.twitter.com/2/tweets/search/recent"
        # r = requests.get(url, headers=headers, params=params, timeout=10)
        return 0.0, 0, 0.0, False
    except Exception as e:
        logger.debug("Twitter sentiment unavailable: %s", e)
        return 0.0, 0, 0.0, False


def _get_reddit_sentiment(symbol: str, lookback_hours: int = 24) -> Tuple[float, int, float, bool]:
    """
    Reddit API - r/wallstreetbets, r/cryptocurrency, r/stocks.
    Returns: (polarity, volume, velocity, available).
    """
    if not REDDIT_CLIENT_ID or not REDDIT_CLIENT_SECRET:
        return 0.0, 0, 0.0, False
    try:
        import requests
        sym = _normalize_symbol(symbol)
        subreddits = ["wallstreetbets", "CryptoCurrency", "stocks"]
        auth = requests.auth.HTTPBasicAuth(REDDIT_CLIENT_ID, REDDIT_CLIENT_SECRET)
        data = {"grant_type": "client_credentials"}
        headers = {"User-Agent": "TradingBot/1.0"}
        token_r = requests.post(
            "https://www.reddit.com/api/v1/access_token",
            auth=auth, data=data, headers=headers, timeout=5
        )
        if token_r.status_code != 200:
            return 0.0, 0, 0.0, False
        token = token_r.json().get("access_token")
        headers["Authorization"] = f"bearer {token}"
        total_mentions = 0
        scores = []
        for sub in subreddits:
            r = requests.get(
                f"https://oauth.reddit.com/r/{sub}/search?q={sym}&limit=25&sort=relevance",
                headers=headers, timeout=5
            )
            if r.status_code != 200:
                continue
            data = r.json().get("data", {}).get("children", [])
            for child in data:
                post = child.get("data", {})
                total_mentions += 1
                # Heuristic: upvote ratio as proxy for sentiment
                up = post.get("ups", 0)
                down = post.get("downs", 0)
                if up + down > 0:
                    scores.append((up - down) / (up + down))
        if not scores:
            return 0.0, total_mentions, 0.0, total_mentions > 0
        avg = sum(scores) / len(scores)
        return avg, total_mentions, len(scores) / 24.0 if lookback_hours else 0.0, True
    except Exception as e:
        logger.debug("Reddit sentiment unavailable: %s", e)
        return 0.0, 0, 0.0, False


def _get_stocktwits_sentiment(symbol: str) -> Tuple[float, int, bool]:
    """
    StockTwits API (free, no key) - stream sentiment.
    Returns: (polarity -1..1, volume, available).
    """
    try:
        import requests
        sym = _normalize_symbol(symbol)
        url = f"https://api.stocktwits.com/api/2/streams/symbol/{sym}.json"
        r = requests.get(url, timeout=5)
        if r.status_code != 200:
            return 0.0, 0, False
        data = r.json()
        msgs = data.get("messages", [])
        bull = sum(1 for m in msgs if (m.get("entities", {}).get("sentiment", {}) or {}).get("basic") == "Bullish")
        bear = sum(1 for m in msgs if (m.get("entities", {}).get("sentiment", {}) or {}).get("basic") == "Bearish")
        total = bull + bear
        if total == 0:
            return 0.0, len(msgs), len(msgs) > 0
        polarity = (bull - bear) / total
        return polarity, len(msgs), True
    except Exception as e:
        logger.debug("StockTwits unavailable: %s", e)
        return 0.0, 0, False


def get_social_sentiment(symbol: str, lookback_hours: int = 24) -> SocialSentimentResult:
    """
    Get combined social sentiment score (-10 to +10).
    Weight: Twitter 40%, Reddit 35%, StockTwits 25%.
    """
    if not ENABLE_SOCIAL_SENTIMENT:
        return SocialSentimentResult(0.0, sources_used=["disabled"])
    symbol = _normalize_symbol(symbol)
    scores = []
    weights = []
    sources = []
    details = {}
    # Twitter
    tw_pol, tw_vol, tw_vel, tw_ok = _get_twitter_sentiment(symbol, lookback_hours)
    if tw_ok:
        scores.append(tw_pol * 10)  # scale to -10..+10
        weights.append(0.40)
        sources.append("twitter")
        details["twitter"] = {"polarity": tw_pol, "volume": tw_vol, "velocity": tw_vel}
    # Reddit
    rd_pol, rd_vol, rd_vel, rd_ok = _get_reddit_sentiment(symbol, lookback_hours)
    if rd_ok:
        scores.append(rd_pol * 10)
        weights.append(0.35)
        sources.append("reddit")
        details["reddit"] = {"polarity": rd_pol, "volume": rd_vol, "velocity": rd_vel}
    # StockTwits
    st_pol, st_vol, st_ok = _get_stocktwits_sentiment(symbol)
    if st_ok:
        scores.append(st_pol * 10)
        weights.append(0.25)
        sources.append("stocktwits")
        details["stocktwits"] = {"polarity": st_pol, "volume": st_vol}
    if not scores:
        return SocialSentimentResult(0.0, sources_used=[])
    total_w = sum(weights)
    weighted = [s * w / total_w for s, w in zip(scores, weights)]
    final_score = sum(weighted)
    mention_vol = details.get("twitter", {}).get("volume", 0) + details.get("reddit", {}).get("volume", 0) + details.get("stocktwits", {}).get("volume", 0)
    polarity = final_score / 10.0
    return SocialSentimentResult(
        score=round(final_score, 1),
        mention_volume=mention_vol,
        sentiment_polarity=polarity,
        velocity_24h=details.get("reddit", {}).get("velocity", 0) or 0,
        sources_used=sources,
        details=details,
    )
