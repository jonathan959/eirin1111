"""
Unified Alternative Data Score - combines social, news, insider, options, on-chain.
Returns: score 0-100, confidence, sources used.
Config: ENABLE_SOCIAL_SENTIMENT, ENABLE_OPTIONS_FLOW, ENABLE_ONCHAIN, NEWS_API_KEY, etc.
"""
import os
import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


@dataclass
class AlternativeDataResult:
    score: float  # 0-100
    confidence: float  # 0-1
    sources_used: List[str] = field(default_factory=list)
    details: Dict[str, Any] = field(default_factory=dict)
    alerts: List[str] = field(default_factory=list)
    short_squeeze_candidate: bool = False
    high_short_interest_risk: bool = False


def _is_crypto(symbol: str) -> bool:
    s = symbol.upper().split("/")[0]
    return s in ("BTC", "XBT", "ETH") or "/" in symbol


def _is_stock(symbol: str) -> bool:
    s = symbol.upper()
    return len(s) <= 5 and "/" not in s and s not in ("BTC", "ETH")


def calculate_alternative_data_score(symbol: str) -> AlternativeDataResult:
    """
    Combines: social sentiment, news, insider, options (stocks), on-chain (crypto).
    Social weight: 10% of total recommendation.
    """
    score = 50.0  # Neutral base
    confidence = 0.0
    sources = []
    details = {}
    alerts = []
    short_squeeze = False
    high_short = False
    is_crypto = _is_crypto(symbol)
    is_stock = _is_stock(symbol)

    # 1. Social sentiment (-10 to +10 -> map to 0-100 contribution)
    try:
        from social_sentiment import get_social_sentiment
        r = get_social_sentiment(symbol, 24)
        if r.sources_used and "disabled" not in str(r.sources_used).lower():
            contrib = 50 + (r.score * 5)  # -10..+10 -> 0..100
            score = 0.6 * score + 0.4 * contrib
            confidence += 0.25
            sources.append("social")
            details["social"] = {"score": r.score, "polarity": r.sentiment_polarity}
    except Exception as e:
        logger.debug("Social sentiment: %s", e)

    # 2. News sentiment (SentimentAnalyzer - NewsAPI, Finnhub)
    try:
        from sentiment_analyzer import SentimentAnalyzer
        analyzer = SentimentAnalyzer()
        news_score, news_ok, _ = analyzer._get_news_sentiment(symbol, 24)
        if news_ok:
            contrib = 50 + (news_score * 50)
            score = 0.7 * score + 0.3 * contrib
            confidence += 0.2
            sources.append("news")
            details["news"] = {"score": news_score}
        break_alert = analyzer.get_breaking_news_alert(symbol)
        if break_alert:
            alerts.append(break_alert)
    except Exception as e:
        logger.debug("News sentiment: %s", e)

    # 3. Insider transactions (stocks only)
    if is_stock:
        try:
            from insider_tracker import get_insider_score
            ins_score, ins_reasons = get_insider_score(symbol)
            if ins_reasons:
                contrib = 50 + (ins_score * 10)
                score = 0.85 * score + 0.15 * contrib
                confidence += 0.15
                sources.append("insider")
                details["insider"] = {"score_delta": ins_score, "reasons": ins_reasons}
        except Exception as e:
            logger.debug("Insider: %s", e)

    # 4. Options flow (stocks only)
    if is_stock:
        try:
            from options_flow import get_options_flow
            r = get_options_flow(symbol)
            if r.details and not r.details.get("skipped"):
                contrib = 50 + r.score * 5
                score = 0.85 * score + 0.15 * contrib
                confidence += 0.1
                sources.append("options")
                details["options"] = {"score": r.score, "alerts": r.alerts}
                alerts.extend(r.alerts[:2])
        except Exception as e:
            logger.debug("Options flow: %s", e)

    # 5. Short interest + analyst ratings (stocks)
    if is_stock:
        try:
            import requests
            key = os.getenv("FINNHUB_API_KEY", os.getenv("FINNHUB_TOKEN", ""))
            sym = symbol.split("/")[0].upper()
            if key:
                r = requests.get(
                    "https://finnhub.io/api/v1/stock/short-interest",
                    params={"symbol": sym, "token": key},
                    timeout=5,
                )
                if r.status_code == 200:
                    raw = r.json()
                    data = raw[0] if isinstance(raw, list) and raw else (raw if isinstance(raw, dict) else {})
                    ratio = float(data.get("shortInterestRatio", data.get("ratio", 0)) or 0)
                    if ratio > 3.0:
                        high_short = True
                        details["short_interest_ratio"] = ratio
                    if ratio > 2.0 and score > 55:
                        short_squeeze = True
                        alerts.append(f"Short squeeze candidate (ratio {ratio:.1f})")
        except Exception as e:
            logger.debug("Short interest: %s", e)
        # Analyst ratings (TRACK_ANALYST_RATINGS)
        try:
            from analyst_ratings_tracker import analyst_score_contribution, TRACK_ANALYST_RATINGS
            if TRACK_ANALYST_RATINGS:
                contrib, reason = analyst_score_contribution(symbol)
                if contrib != 0:
                    score = 0.9 * score + 0.1 * (50 + contrib)  # Blend analyst into score
                    confidence += 0.05
                    sources.append("analyst")
                    details["analyst_contrib"] = contrib
                    if reason:
                        details["analyst_reason"] = reason
        except Exception as e:
            logger.debug("Analyst ratings: %s", e)

    # 6. On-chain (crypto only, 25% weight via ONCHAIN_WEIGHT_CRYPTO)
    if is_crypto:
        try:
            from onchain_analyzer import get_onchain_metrics, ONCHAIN_WEIGHT_CRYPTO
            r = get_onchain_metrics(symbol)
            if r.details and not r.details.get("skipped"):
                w = ONCHAIN_WEIGHT_CRYPTO
                score = (1 - w) * score + w * r.score
                confidence += w
                sources.append("onchain")
                details["onchain"] = {"score": r.score, "mvrv": r.mvrv_ratio, "nvt": r.nvt_ratio}
        except Exception as e:
            logger.debug("On-chain: %s", e)

    if not sources:
        confidence = 0.0
    else:
        confidence = min(1.0, confidence)

    return AlternativeDataResult(
        score=round(min(100, max(0, score)), 1),
        confidence=confidence,
        sources_used=sources,
        details=details,
        alerts=alerts[:5],
        short_squeeze_candidate=short_squeeze,
        high_short_interest_risk=high_short,
    )


class AlternativeDataIntegrator:
    """
    Phase 4C: Wrapper for alternative data integration.
    Used by UltimateIntelligenceLayer.
    """

    def get_alternative_data_signal(self, symbol: str) -> AlternativeDataResult:
        """Get combined alternative data score for symbol."""
        return calculate_alternative_data_score(symbol)
