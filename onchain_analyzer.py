"""
Crypto On-Chain Metrics - Glassnode, CryptoQuant, Nansen.
Metrics: MVRV, NVT, active addresses, exchange flows, whale movements, stablecoin supply.
"""
import os
import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

ENABLE_ONCHAIN = os.getenv("ENABLE_ONCHAIN", "0").strip().lower() in ("1", "true", "yes")
ENABLE_ONCHAIN_ANALYSIS = os.getenv("ENABLE_ONCHAIN_ANALYSIS", "").strip().lower() in ("1", "true", "yes") or ENABLE_ONCHAIN
GLASSNODE_API_KEY = os.getenv("GLASSNODE_API_KEY", "").strip()
CRYPTOQUANT_API_KEY = os.getenv("CRYPTOQUANT_API_KEY", "").strip()
NANSEN_API_KEY = os.getenv("NANSEN_API_KEY", "").strip()
ONCHAIN_WEIGHT_CRYPTO = float(os.getenv("ONCHAIN_WEIGHT_CRYPTO", "0.25"))


@dataclass
class OnchainResult:
    score: float  # 0-100
    active_addresses_trend: str = "unknown"
    active_addresses_growth_pct: Optional[float] = None
    exchange_netflow: float = 0.0  # negative = outflow (bullish)
    exchange_inflow: Optional[float] = None
    exchange_outflow: Optional[float] = None
    mvrv_ratio: Optional[float] = None
    nvt_ratio: Optional[float] = None
    whale_transfer_count: Optional[int] = None
    stablecoin_supply_change_pct: Optional[float] = None
    details: Dict[str, Any] = field(default_factory=dict)


def _normalize_symbol(symbol: str) -> str:
    s = symbol.split("/")[0].upper()
    if s == "XBT":
        return "BTC"
    return s


def get_onchain_metrics(symbol: str) -> OnchainResult:
    """
    Get on-chain metrics for crypto. Weight ~25% of crypto recommendation (ONCHAIN_WEIGHT_CRYPTO).
    Metrics: MVRV, NVT, active addresses growth, exchange flows, whale movements, stablecoin supply.
    """
    if not ENABLE_ONCHAIN and not ENABLE_ONCHAIN_ANALYSIS:
        return OnchainResult(50.0, details={"enabled": False})
    sym = _normalize_symbol(symbol)
    if sym not in ("BTC", "ETH"):
        return OnchainResult(50.0, details={"asset": sym, "skipped": "major only"})
    try:
        import requests
        score = 50.0
        details = {}
        # Glassnode (requires API key)
        if GLASSNODE_API_KEY:
            asset = "bitcoin" if sym == "BTC" else "ethereum"
            base = f"https://api.glassnode.com/v1/metrics"
            headers = {"X-Apn-Api-Key": GLASSNODE_API_KEY}
            # Active addresses
            r = requests.get(
                f"{base}/addresses/active_count",
                params={"a": asset, "i": "1d"},
                headers=headers,
                timeout=10,
            )
            if r.status_code == 200:
                data = r.json()
                if data:
                    recent = [x["v"] for x in data[-30:] if x.get("v")]
                    if len(recent) >= 2:
                        trend = "up" if recent[-1] > recent[0] else "down"
                        details["active_addresses_trend"] = trend
                        growth = ((recent[-1] - recent[0]) / recent[0] * 100) if recent[0] else 0
                        details["active_addresses_growth_pct"] = round(growth, 2)
                        if trend == "up":
                            score += 5
                        else:
                            score -= 5
            # Exchange netflow
            r2 = requests.get(
                f"{base}/transactions/transfers_volume_exchanges_net",
                params={"a": asset, "i": "1d"},
                headers=headers,
                timeout=10,
            )
            if r2.status_code == 200:
                data2 = r2.json()
                if data2:
                    net = data2[-1].get("v", 0)
                    details["exchange_netflow"] = net
                    if net < -1000:  # outflow = bullish
                        score += 10
                    elif net > 1000:  # inflow = bearish
                        score -= 5
            # MVRV (market cap / realized cap) - Glassnode market/mvrv
            r3 = requests.get(
                f"{base}/market/mvrv",
                params={"a": asset, "i": "1d"},
                headers=headers,
                timeout=10,
            )
            if r3.status_code == 200:
                data3 = r3.json()
                if data3 and len(data3) >= 1:
                    mvrv = float(data3[-1].get("v", 0) or 0)
                    details["mvrv"] = round(mvrv, 3)
                    if 0.8 < mvrv < 1.2:
                        score += 8
                    elif mvrv > 3.0:
                        score -= 10
                    elif mvrv < 0.8:
                        score += 5
            # NVT (network value / transactions)
            r4 = requests.get(
                f"{base}/transactions/transfers_volume_nvt",
                params={"a": asset, "i": "1d"},
                headers=headers,
                timeout=10,
            )
            if r4.status_code == 200:
                data4 = r4.json()
                if data4:
                    nvt = data4[-1].get("v")
                    if nvt is not None:
                        details["nvt"] = round(float(nvt), 2)
            # Large transfers (whale movements)
            r5 = requests.get(
                f"{base}/transactions/transfers_volume_sum",
                params={"a": asset, "i": "1d", "e": "exchange"},
                headers=headers,
                timeout=10,
            )
            if r5.status_code == 200:
                data5 = r5.json()
                if data5 and len(data5) >= 7:
                    last_7 = [x.get("v") for x in data5[-7:] if x.get("v")]
                    if last_7:
                        details["whale_volume_7d"] = sum(last_7)
        # CryptoQuant (alternative)
        if CRYPTOQUANT_API_KEY and sym == "BTC":
            url = f"https://api.cryptoquant.com/v1/btc/exchange-flows/exchange-netflow"
            r = requests.get(url, headers={"Authorization": f"Bearer {CRYPTOQUANT_API_KEY}"}, timeout=10)
            if r.status_code == 200:
                data = r.json().get("result", {}).get("data", [])
                if data:
                    details["cryptoquant_netflow"] = data[-1]
        return OnchainResult(
            score=round(min(100, max(0, score)), 1),
            active_addresses_trend=details.get("active_addresses_trend", "unknown"),
            active_addresses_growth_pct=details.get("active_addresses_growth_pct"),
            exchange_netflow=details.get("exchange_netflow", 0),
            exchange_inflow=details.get("exchange_inflow"),
            exchange_outflow=details.get("exchange_outflow"),
            mvrv_ratio=details.get("mvrv"),
            nvt_ratio=details.get("nvt"),
            whale_transfer_count=details.get("whale_volume_7d"),
            stablecoin_supply_change_pct=details.get("stablecoin_supply_change_pct"),
            details=details,
        )
    except Exception as e:
        logger.debug("On-chain analysis unavailable: %s", e)
        return OnchainResult(50.0, details={"error": str(e)})
