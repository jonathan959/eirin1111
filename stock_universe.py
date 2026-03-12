"""
Expanded stock universe for recommendation scanner: S&P 500, NASDAQ 100, major ETFs.
Ensures the screener cycles through 500+ stocks instead of a small fixed set.
"""
import logging
from typing import List, Set

logger = logging.getLogger(__name__)

# Major ETFs - always include (liquid, broad market)
MAJOR_ETFS = [
    "SPY", "QQQ", "IWM", "DIA", "VOO", "VTI", "VEA", "VWO", "EFA", "EEM",
    "XLK", "XLF", "XLE", "XLV", "XLY", "XLP", "XLI", "XLB", "XLRE", "XLU",
    "GLD", "SLV", "TLT", "HYG", "LQD", "BND", "AGG", "TIP", "VNQ", "IYR",
    "TQQQ", "SQQQ", "SOXL", "SOXS", "UPRO", "SPXL", "ARKK", "ARKG", "ARKW",
    "SMH", "XBI", "KRE", "KIE", "XHB", "ITB", "XRT", "XOP", "OIH", "XME",
]

# NASDAQ 100 constituents (representative set - ~100 largest non-financial NASDAQ)
NASDAQ_100 = [
    "AAPL", "MSFT", "AMZN", "NVDA", "META", "GOOGL", "GOOG", "TSLA", "AVGO", "COST",
    "PEP", "NFLX", "ADBE", "CSCO", "AMD", "INTC", "CMCSA", "TMUS", "TXN", "INTU",
    "AMGN", "QCOM", "ISRG", "VRTX", "BKNG", "LRCX", "SBUX", "ADP", "GILD", "REGN",
    "PANW", "AMAT", "ADI", "MDLZ", "KLAC", "SNPS", "CDNS", "MAR", "MRVL", "CTAS",
    "ORLY", "WDAY", "DXCM", "CPRT", "MELI", "ABNB", "FTNT", "MNST", "CHTR",
    "ASML", "CRWD", "PCAR", "PAYX", "AEP", "KDP", "FAST", "EXC", "XEL", "EA",
    "CTSH", "FANG", "GEHC", "MCHP", "ODFL", "AZN", "DASH", "TEAM", "ZS", "VRSK",
    "CCEP", "IDXX", "WBD", "DLTR", "CSGP", "BKR", "TTD", "CDW", "CPT",
    "GDDY", "MDB", "ROST", "CEG", "AON", "DDOG", "NXPI", "TTWO", "CAG",
    "VTRS", "EXPE", "PAYC", "FAST", "HON", "LULU", "KHC", "MDLZ", "PYPL", "SBUX",
]

# S&P 500 - fetch from Wikipedia when possible; fallback to this curated list
# This is a representative subset; full list fetched dynamically
_SP500_FALLBACK = [
    "AAPL", "MSFT", "AMZN", "NVDA", "META", "GOOGL", "GOOG", "BRK-B", "JPM", "V",
    "PG", "JNJ", "UNH", "MA", "HD", "DIS", "XOM", "BAC", "CVX", "PEP",
    "KO", "AVGO", "ADBE", "WMT", "CRM", "CSCO", "ACN", "MCD", "ABT", "TMO",
    "NEE", "DHR", "VZ", "NFLX", "CMCSA", "INTC", "WFC", "PM", "TXN", "AMD",
    "UPS", "IBM", "RTX", "HON", "QCOM", "INTU", "AMGN", "SPGI", "CAT", "AMT",
    "LOW", "SBUX", "GS", "AXP", "BLK", "DE", "PLD", "BKNG", "LMT", "ADI",
    "GILD", "SYK", "MDT", "REGN", "CVS", "CI", "C", "SO", "DUK", "BDX",
    "BSX", "SLB", "EOG", "MMC", "PGR", "ZTS", "MO", "APD", "CL", "CB",
    "USB", "ITW", "PEP", "NOC", "WM", "FCX", "CME", "COST", "APTV", "AON",
    "SHW", "KLAC", "SNPS", "CDNS", "EMR", "EQIX", "PSA", "ORCL", "MSI",
    "MAR", "MMC", "PCAR", "AIG", "AJG", "MET", "AFL", "TRP", "ECL", "APD",
    "GM", "F", "BA", "GE", "LEN", "DHI", "NVR", "PHM", "TOL", "MTH",
    "COIN", "MSTR", "MARA", "RIOT", "CLSK", "HUT", "HOOD", "PLTR", "SOFI",
    "UBER", "LYFT", "DKNG", "AFRM", "RBLX", "SNOW", "DDOG", "NET", "CRWD",
    "ZS", "OKTA", "PANW", "FTNT", "TEAM", "WDAY", "CRM", "NOW", "VEEV",
    "ROKU", "GME", "AMC", "BBBY", "BABA", "JD", "PDD", "NIO", "XPEV", "LI",
]

_cached_sp500: List[str] = []
_cached_nasdaq100: List[str] = []


def _fetch_sp500_from_wikipedia() -> List[str]:
    """Fetch S&P 500 tickers from Wikipedia. Returns empty list on failure."""
    try:
        import pandas as pd
        url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
        tables = pd.read_html(url)
        df = tables[0]
        if "Symbol" in df.columns:
            tickers = df["Symbol"].astype(str).str.strip().tolist()
            # Filter valid tickers (no dots except BRK.B, BF.B)
            out = [t for t in tickers if t and len(t) <= 5 and t != "nan"]
            return out
        if 0 in df.columns:
            out = df[0].astype(str).str.strip().tolist()
            return [t for t in out if t and len(t) <= 5 and t != "nan"]
    except Exception as e:
        logger.debug("stock_universe: Wikipedia fetch failed: %s", e)
    return []


def get_sp500_tickers() -> List[str]:
    """Return S&P 500 tickers (from cache or fetch)."""
    global _cached_sp500
    if _cached_sp500:
        return _cached_sp500
    tickers = _fetch_sp500_from_wikipedia()
    if tickers:
        _cached_sp500 = tickers
        logger.info("stock_universe: loaded %d S&P 500 tickers from Wikipedia", len(tickers))
        return tickers
    _cached_sp500 = _SP500_FALLBACK
    logger.info("stock_universe: using fallback S&P 500 list (%d tickers)", len(_SP500_FALLBACK))
    return _SP500_FALLBACK


def get_nasdaq100_tickers() -> List[str]:
    """Return NASDAQ 100 tickers."""
    return NASDAQ_100


def get_major_etfs() -> List[str]:
    """Return major ETF tickers."""
    return MAJOR_ETFS


def get_expanded_stock_universe() -> List[str]:
    """
    Return combined stock universe: S&P 500 + NASDAQ 100 + major ETFs.
    Deduplicated, 500+ symbols.
    """
    seen: Set[str] = set()
    out: List[str] = []
    for t in get_major_etfs() + get_nasdaq100_tickers() + get_sp500_tickers():
        t = (t or "").strip().upper()
        if not t or t in seen:
            continue
        # Allow BRK.B, BF.B; skip other dotted symbols (often warrants)
        if "." in t and t not in ("BRK.B", "BF.B"):
            continue
        seen.add(t)
        out.append(t)
    return out
