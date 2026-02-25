"""
Intermarket analysis - broader market context. (12.md Part 4)
VIX, USD, yields, gold, SPY, BTC.
"""
import logging
import os
from typing import Dict, List

logger = logging.getLogger(__name__)
ENABLED = os.getenv("ENABLE_INTERMARKET_ANALYSIS", "0").strip().lower() in ("1", "true", "yes")

INDICATORS = {"vix": "^VIX", "dxy": "DX-Y.NYB", "tnx": "^TNX", "gold": "GC=F", "spy": "SPY", "btc": "BTC-USD"}


def _fetch_yf(ticker: str) -> float:
    try:
        import yfinance as yf
        d = yf.Ticker(ticker)
        h = d.history(period="5d")
        if h is not None and not h.empty:
            return float(h["Close"].iloc[-1])
    except ImportError:
        logger.debug("yfinance not installed for intermarket")
    except Exception as e:
        logger.debug("yf %s: %s", ticker, e)
    return 0.0


def get_market_regime() -> Dict:
    """Determine overall market regime."""
    if not ENABLED:
        return {"regime": "neutral", "risk_score": 0.5, "signals": {}, "recommendations": []}
    try:
        vix = _fetch_yf(INDICATORS["vix"])
        recommendations = []
        risk = 0.5
        if vix > 30:
            risk = 0.2
            recommendations.append("VIX elevated - reduce position sizes")
        elif vix > 20:
            risk = 0.4
            recommendations.append("Elevated VIX - caution")
        elif vix < 15:
            risk = 0.7
            recommendations.append("Low VIX - watch for reversal")
        regime = "risk_off" if risk < 0.3 else ("risk_on" if risk > 0.65 else "neutral")
        return {
            "regime": regime,
            "risk_score": risk,
            "signals": {"vix": {"value": vix}},
            "recommendations": recommendations,
        }
    except Exception as e:
        logger.debug("intermarket: %s", e)
        return {"regime": "neutral", "risk_score": 0.5, "signals": {}, "recommendations": []}


def should_reduce_risk(regime: Dict) -> Dict:
    """Determine if risk should be reduced."""
    vix_sig = regime.get("signals", {}).get("vix", {})
    vix_val = vix_sig.get("value", 0)
    if vix_val > 30:
        return {"reduce_risk": True, "severity": "high", "reduction_pct": 0.5, "reasons": [f"VIX at {vix_val:.1f}"]}
    if regime.get("risk_score", 0.5) < 0.3:
        return {"reduce_risk": True, "severity": "medium", "reduction_pct": 0.3, "reasons": ["Risk-off regime"]}
    return {"reduce_risk": False, "severity": "none"}
