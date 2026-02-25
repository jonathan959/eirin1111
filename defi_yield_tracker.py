"""
DeFi yield opportunities - staking/lending APY tracking.

Compare trading returns vs passive yield.
Alert: "USDC lending at 8% APY - higher than your bot's 6% return"
"""
import logging
import os
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


def get_defi_yields() -> List[Dict[str, Any]]:
    """
    Fetch top DeFi yields from DefiLlama (free, no key).
    Returns: [{ protocol, symbol, apy, tvl }]
    """
    try:
        import requests
        url = "https://yields.llama.fi/pools"
        r = requests.get(url, timeout=10)
        if r.status_code != 200:
            return []
        data = r.json().get("data", [])[:100]
        out = []
        for p in data:
            apy = p.get("apy")
            if apy is None:
                continue
            apy = float(apy)
            if apy < 1:
                continue
            out.append({
                "protocol": p.get("project", ""),
                "symbol": p.get("symbol", ""),
                "apy": round(apy, 2),
                "tvl_usd": p.get("tvlUsd"),
            })
        return sorted(out, key=lambda x: -float(x.get("apy", 0)))[:20]
    except Exception as e:
        logger.debug("DefiLlama yields: %s", e)
        return []


def yield_vs_bot_alert(bot_return_pct: float) -> Optional[Dict[str, Any]]:
    """
    If any DeFi yield exceeds bot return, return alert.
    """
    yields_list = get_defi_yields()
    for y in yields_list:
        apy = float(y.get("apy", 0) or 0)
        if apy > bot_return_pct and apy > 5:
            return {
                "alert": f"{y.get('protocol')} {y.get('symbol')} at {apy:.1f}% APY - higher than bot's {bot_return_pct:.1f}%",
                "protocol": y.get("protocol"),
                "apy": apy,
                "bot_return_pct": bot_return_pct,
            }
    return None
