"""
Crypto correlation analysis - rolling correlation to BTC.

Most altcoins correlate with BTC. When BTC dumps, correlated coins dump.
Diversify: find low-correlation crypto assets.
"""
import logging
import os
import statistics
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


def rolling_correlation(
    series_a: List[float],
    series_b: List[float],
    window: int = 30,
) -> Optional[float]:
    """Pearson correlation over last `window` values. Returns None if insufficient data."""
    if len(series_a) < window or len(series_b) < window:
        return None
    a = series_a[-window:]
    b = series_b[-window:]
    n = len(a)
    ma = sum(a) / n
    mb = sum(b) / n
    cov = sum((a[i] - ma) * (b[i] - mb) for i in range(n))
    std_a = (sum((x - ma) ** 2 for x in a) / n) ** 0.5
    std_b = (sum((x - mb) ** 2 for x in b) / n) ** 0.5
    if std_a == 0 or std_b == 0:
        return 0.0
    return cov / (n * std_a * std_b)


def returns_from_prices(prices: List[float]) -> List[float]:
    """Compute log returns from price series."""
    out = []
    for i in range(1, len(prices)):
        if prices[i - 1] and prices[i - 1] > 0:
            out.append((prices[i] - prices[i - 1]) / prices[i - 1])
    return out


def btc_correlation(
    alt_prices: List[float],
    btc_prices: List[float],
    window_days: int = 30,
) -> Dict[str, Any]:
    """
    Compute rolling correlation of alt returns vs BTC returns.
    Returns: { correlation, window_days, interpretation }
    """
    if len(alt_prices) < window_days + 1 or len(btc_prices) < window_days + 1:
        return {"correlation": None, "window_days": window_days, "interpretation": "insufficient_data"}
    alt_ret = returns_from_prices(alt_prices)
    btc_ret = returns_from_prices(btc_prices)
    min_len = min(len(alt_ret), len(btc_ret))
    corr = rolling_correlation(alt_ret[-min_len:], btc_ret[-min_len:], window=min(window_days, min_len))
    if corr is None:
        return {"correlation": None, "window_days": window_days, "interpretation": "insufficient_data"}
    if corr > 0.8:
        interp = "high_correlation"
    elif corr > 0.5:
        interp = "moderate_correlation"
    elif corr > 0.2:
        interp = "low_correlation"
    else:
        interp = "uncorrelated"
    return {
        "correlation": round(corr, 3),
        "window_days": window_days,
        "interpretation": interp,
    }
