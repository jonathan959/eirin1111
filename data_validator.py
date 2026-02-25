"""
Multi-source price validation for crypto.

- Fetches prices from Kraken, Coinbase, Binance (via ccxt)
- Uses median price (reduces manipulation risk)
- Alerts if prices diverge >2%
- Fallback: CryptoCompare (free, no key for basic)
"""
import logging
import os
import statistics
import time
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

ENABLE_MULTI_SOURCE = os.getenv("ENABLE_MULTI_SOURCE_VALIDATION", "0").strip().lower() in ("1", "true", "yes")
PRICE_DIVERGENCE_ALERT_PCT = float(os.getenv("PRICE_DIVERGENCE_ALERT_PCT", "2.0"))
BACKUP_DATA_SOURCES = os.getenv("BACKUP_DATA_SOURCES", "1").strip().lower() in ("1", "true", "yes")


def _kraken_price(symbol: str, kc) -> Optional[Tuple[float, str]]:
    try:
        t = kc.fetch_ticker(symbol)
        p = float(t.get("last") or t.get("close") or 0)
        return (p, "kraken") if p > 0 else None
    except Exception as e:
        logger.debug("Kraken price %s: %s", symbol, e)
        return None


def _ccxt_exchange_price(symbol: str, exchange_name: str) -> Optional[Tuple[float, str]]:
    try:
        import ccxt
        ex = getattr(ccxt, exchange_name, None)
        if not ex:
            return None
        e = ex({"enableRateLimit": True, "timeout": 5000})
        t = e.fetch_ticker(symbol)
        p = float(t.get("last") or t.get("close") or 0)
        return (p, exchange_name) if p > 0 else None
    except Exception as e:
        logger.debug("%s price %s: %s", exchange_name, symbol, e)
        return None


def _cryptocompare_price(symbol: str) -> Optional[Tuple[float, str]]:
    """CryptoCompare free API. Symbol like BTC/USD -> BTC."""
    try:
        import requests
        base = (symbol.split("/")[0] or symbol).strip()
        if not base:
            return None
        url = f"https://min-api.cryptocompare.com/data/price?fsym={base}&tsyms=USD"
        r = requests.get(url, timeout=5)
        r.raise_for_status()
        data = r.json()
        p = float(data.get("USD") or 0)
        return (p, "cryptocompare") if p > 0 else None
    except Exception as e:
        logger.debug("CryptoCompare %s: %s", symbol, e)
        return None


def get_validated_price(
    symbol: str,
    kraken_client=None,
    sources: Optional[List[str]] = None,
) -> Tuple[Optional[float], Dict[str, Any]]:
    """
    Fetch price from multiple sources and return median (or primary if single source).
    Returns (median_price, metadata) where metadata has: prices, sources, divergence_alert, median, primary_price.
    """
    if not ENABLE_MULTI_SOURCE and kraken_client:
        try:
            p = kraken_client.fetch_ticker_last(symbol)
            return (p, {"primary_price": p, "sources": ["kraken"], "median": p}) if p and p > 0 else (None, {})
        except Exception:
            return (None, {})

    prices_with_src: List[Tuple[float, str]] = []
    if kraken_client:
        v = _kraken_price(symbol, kraken_client)
        if v:
            prices_with_src.append(v)

    extra = sources or (["coinbase", "binance"] if BACKUP_DATA_SOURCES else [])
    for name in extra:
        v = _ccxt_exchange_price(symbol, name)
        if v:
            prices_with_src.append(v)

    if not prices_with_src and BACKUP_DATA_SOURCES:
        v = _cryptocompare_price(symbol)
        if v:
            prices_with_src.append(v)

    if not prices_with_src:
        return (None, {"error": "no_prices", "sources": []})

    values = [p[0] for p in prices_with_src]
    median = float(statistics.median(values))
    primary = prices_with_src[0][0] if prices_with_src else 0
    max_val = max(values)
    min_val = min(values)
    divergence_pct = ((max_val - min_val) / median * 100) if median > 0 else 0

    metadata = {
        "median": median,
        "primary_price": primary,
        "sources": [p[1] for p in prices_with_src],
        "prices": {p[1]: p[0] for p in prices_with_src},
        "divergence_pct": round(divergence_pct, 2),
        "divergence_alert": divergence_pct > PRICE_DIVERGENCE_ALERT_PCT,
    }

    if metadata["divergence_alert"]:
        try:
            from db import log_data_quality
            log_data_quality(
                source="data_validator",
                issue_type="price_divergence",
                severity="warning",
                details={"symbol": symbol, "divergence_pct": divergence_pct, "prices": metadata["prices"]},
            )
        except Exception:
            pass

    return (median, metadata)


def get_arbitrage_opportunity(
    symbol: str,
    kraken_client=None,
    min_diff_usd: float = 100.0,
    min_diff_pct: float = 0.5,
) -> Optional[Dict[str, Any]]:
    """
    CEX arbitrage detection: alert when price diff between exchanges exceeds threshold.
    Returns: { cheap_exchange, expensive_exchange, diff_usd, diff_pct, opportunity } or None
    """
    price, meta = get_validated_price(symbol, kraken_client, sources=["coinbase", "binance"])
    prices = meta.get("prices", {})
    if len(prices) < 2:
        return None
    items = [(k, v) for k, v in prices.items() if v and v > 0]
    if len(items) < 2:
        return None
    cheap = min(items, key=lambda x: x[1])
    expensive = max(items, key=lambda x: x[1])
    diff_pct = (expensive[1] - cheap[1]) / cheap[1] * 100 if cheap[1] else 0
    diff_usd = expensive[1] - cheap[1]
    if diff_usd >= min_diff_usd or diff_pct >= min_diff_pct:
        return {
            "symbol": symbol,
            "cheap_exchange": cheap[0],
            "expensive_exchange": expensive[0],
            "diff_usd": round(diff_usd, 2),
            "diff_pct": round(diff_pct, 2),
            "opportunity": f"{symbol} ${diff_usd:.0f} cheaper on {cheap[0]} than {expensive[0]}",
        }
    return None


def is_data_quality_degraded(minutes: int = 15, threshold: int = 5) -> bool:
    """True if data quality issues exceed threshold (auto-pause trading)."""
    try:
        from db import get_recent_data_quality_count
        return get_recent_data_quality_count(minutes=minutes, min_severity="warning") >= threshold
    except Exception:
        return False


def get_validated_crypto_price(primary_price: float, symbol: str, kraken_client=None) -> Tuple[float, bool]:
    """
    Legacy interface: validate primary price against other sources.
    Returns (validated_price, divergence_alert).
    When ENABLE_MULTI_SOURCE=0, returns (primary_price, False).
    """
    if not ENABLE_MULTI_SOURCE or not BACKUP_DATA_SOURCES:
        return (primary_price, False)
    try:
        price, meta = get_validated_price(symbol, kraken_client)
        if price is not None and price > 0:
            return (float(price), bool(meta.get("divergence_alert", False)))
        return (primary_price, False)
    except Exception:
        return (primary_price, False)


def get_ticker_validated(
    symbol: str,
    kraken_client=None,
) -> Dict[str, Any]:
    """
    Returns ticker dict with validated 'last' price when ENABLE_MULTI_SOURCE=1.
    Otherwise returns primary source ticker.
    """
    if not kraken_client:
        return {"symbol": symbol, "last": 0.0, "bid": 0.0, "ask": 0.0}

    if not ENABLE_MULTI_SOURCE:
        try:
            return kraken_client.fetch_ticker(symbol)
        except Exception:
            return {"symbol": symbol, "last": 0.0, "bid": 0.0, "ask": 0.0}

    price, meta = get_validated_price(symbol, kraken_client)
    try:
        base = kraken_client.fetch_ticker(symbol)
        if price is not None:
            base["last"] = price
            base["validated_median"] = price
            base["validation_metadata"] = meta
        return base
    except Exception:
        return {
            "symbol": symbol,
            "last": price or 0.0,
            "bid": price or 0.0,
            "ask": price or 0.0,
            "validated_median": price,
            "validation_metadata": meta,
        }
