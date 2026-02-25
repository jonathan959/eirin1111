"""
Options Flow Analysis - unusual options activity for stocks.
APIs: Unusual Whales, Tradytics (require subscription).
"""
import os
import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

ENABLE_OPTIONS_FLOW = os.getenv("ENABLE_OPTIONS_FLOW", "0").strip().lower() in ("1", "true", "yes")
UNUSUAL_WHALES_API_KEY = os.getenv("UNUSUAL_WHALES_API_KEY", "")
FINNHUB_API_KEY = os.getenv("FINNHUB_API_KEY", "")  # Free tier has some options data


@dataclass
class OptionsFlowResult:
    score: float  # -10 to +10 (bearish to bullish)
    unusual_call_volume: float = 0.0
    unusual_put_volume: float = 0.0
    put_call_ratio: Optional[float] = None
    alerts: List[str] = field(default_factory=list)
    details: Dict[str, Any] = field(default_factory=dict)


def _normalize_symbol(symbol: str) -> str:
    return symbol.split("/")[0].upper().replace("XBT", "BTC")


def get_options_flow(symbol: str) -> OptionsFlowResult:
    """
    Get options flow signal. Stocks only.
    Large call buys = bullish, large put buys = bearish.
    """
    if not ENABLE_OPTIONS_FLOW:
        return OptionsFlowResult(0.0, details={"enabled": False})
    sym = _normalize_symbol(symbol)
    if sym in ("BTC", "ETH", "USDT"):
        return OptionsFlowResult(0.0, details={"asset": "crypto", "skipped": True})
    try:
        import requests
        alerts = []
        call_vol = 0.0
        put_vol = 0.0
        # Finnhub: aggregate options volume (free tier)
        if FINNHUB_API_KEY:
            url = f"https://finnhub.io/api/v1/stock/option-chain?symbol={sym}&token={FINNHUB_API_KEY}"
            r = requests.get(url, timeout=5)
            if r.status_code == 200:
                data = r.json()
                for item in data.get("data", [])[:50]:
                    vol = float(item.get("volume", 0) or 0)
                    oi = float(item.get("openInterest", 0) or 0)
                    opt_type = str(item.get("type", "call")).lower()
                    if "call" in opt_type:
                        call_vol += vol
                    else:
                        put_vol += vol
        # Unusual Whales (paid)
        if UNUSUAL_WHALES_API_KEY:
            # Not implemented - requires subscription
            pass
        # Score: call/put ratio heuristic
        score = 0.0
        if put_vol > 0:
            ratio = call_vol / put_vol
            if ratio > 1.5:
                score = min(10, (ratio - 1) * 5)
                alerts.append(f"Unusual call bias: {ratio:.1f}x puts")
            elif ratio < 0.67:
                score = max(-10, (1 - ratio) * -5)
                alerts.append(f"Unusual put bias: {1/ratio:.1f}x calls")
            if call_vol > 1e6 or put_vol > 1e6:
                alerts.append(f"High options activity: ${max(call_vol, put_vol)/1e6:.1f}M")
        return OptionsFlowResult(
            score=round(score, 1),
            unusual_call_volume=call_vol,
            unusual_put_volume=put_vol,
            put_call_ratio=put_vol / call_vol if call_vol > 0 else None,
            alerts=alerts,
            details={"call_volume": call_vol, "put_volume": put_vol},
        )
    except Exception as e:
        logger.debug("Options flow unavailable: %s", e)
        return OptionsFlowResult(0.0, details={"error": str(e)})


class OptionsFlowAnalyzer:
    """
    Phase 4B: Wraps get_options_flow for UltimateIntelligenceLayer.
    Exposes get_options_signals() returning signal, put_call_ratio, unusual_calls, unusual_puts, reasoning.
    """

    def __init__(self) -> None:
        pass

    def get_options_signals(self, symbol: str) -> Dict[str, Any]:
        """
        Get options flow signals for symbol.

        Returns:
        {
            'signal': 'bullish' | 'bearish' | 'neutral',
            'put_call_ratio': float | None,
            'unusual_calls': float,
            'unusual_puts': float,
            'reasoning': str,
        }
        """
        r = get_options_flow(symbol)
        score = r.score
        call_vol = r.unusual_call_volume
        put_vol = r.unusual_put_volume
        pcr = r.put_call_ratio
        alerts = r.alerts or []
        if score > 3:
            signal = "bullish"
            reasoning = "; ".join(alerts[:2]) or f"Options flow score +{score:.1f}"
        elif score < -3:
            signal = "bearish"
            reasoning = "; ".join(alerts[:2]) or f"Options flow score {score:.1f}"
        else:
            signal = "neutral"
            reasoning = "Mixed or no options flow" if not alerts else "; ".join(alerts[:2])
        return {
            "signal": signal,
            "put_call_ratio": pcr,
            "unusual_calls": call_vol,
            "unusual_puts": put_vol,
            "reasoning": reasoning,
        }
