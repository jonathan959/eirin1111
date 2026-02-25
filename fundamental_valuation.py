"""
Fundamental Valuation Engine for long-term investing mode.

Calculates P/E, P/B, PEG, dividend metrics, DCF, FCF for stocks.
Produces ValuationScore (0-100) separate from technical score.
Extend with Polygon/Finnhub/Alpha Vantage API for live data.
"""
import os
import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

LONG_TERM_MODE = os.getenv("LONG_TERM_MODE", "0").strip().lower() in ("1", "true", "yes", "y", "on")


@dataclass
class FundamentalData:
    """Raw fundamental metrics (from API or cache)."""
    symbol: str
    pe_ratio: Optional[float] = None
    pb_ratio: Optional[float] = None
    peg_ratio: Optional[float] = None
    dividend_yield: Optional[float] = None
    payout_ratio: Optional[float] = None
    dividend_growth_5y: Optional[float] = None
    fcf: Optional[float] = None
    revenue_growth_yoy: Optional[float] = None
    sector_pe_avg: Optional[float] = None
    sector_pb_avg: Optional[float] = None
    eps: Optional[float] = None
    book_value_per_share: Optional[float] = None
    price: Optional[float] = None
    market_cap: Optional[float] = None


@dataclass
class ValuationScore:
    """0-100 score from fundamental valuation only."""
    score: float
    components: Dict[str, float] = field(default_factory=dict)
    dcf_intrinsic: Optional[float] = None
    dcf_margin: Optional[float] = None
    flags: List[str] = field(default_factory=list)
    raw: Dict[str, Any] = field(default_factory=dict)


# Sector average P/E and P/B (approximate; replace with API)
SECTOR_AVG_PE: Dict[str, float] = {
    "Technology": 28.0, "Financial": 14.0, "Healthcare": 22.0,
    "Consumer Cyclical": 18.0, "Consumer Defensive": 20.0,
    "Energy": 12.0, "Industrial": 16.0, "Communication": 15.0,
    "ETF": 20.0, "Utilities": 18.0,
}
SECTOR_AVG_PB: Dict[str, float] = {
    "Technology": 6.0, "Financial": 1.5, "Healthcare": 4.0,
    "Consumer Cyclical": 4.0, "Consumer Defensive": 4.5,
    "Energy": 1.5, "Industrial": 3.0, "Communication": 3.0,
    "ETF": 3.0, "Utilities": 1.8,
}


def _fetch_fundamental_data(symbol: str, sector: Optional[str] = None) -> FundamentalData:
    """
    Fetch fundamental data. Override via env FUNDAMENTAL_API=polygon|finnhub|mock.
    Mock returns placeholder for development.
    """
    try:
        api = os.getenv("FUNDAMENTAL_API", "mock").strip().lower()
        if api == "polygon":
            return _fetch_polygon_fundamentals(symbol)
        if api == "finnhub":
            return _fetch_finnhub_fundamentals(symbol)
    except Exception as e:
        logger.warning("Fundamental fetch failed for %s: %s", symbol, e)
    return _mock_fundamentals(symbol, sector)


def _mock_fundamentals(symbol: str, sector: Optional[str]) -> FundamentalData:
    """Placeholder data when no API configured."""
    sector = sector or "Technology"
    pe = SECTOR_AVG_PE.get(sector, 20.0) * (0.9 + (hash(symbol) % 20) / 100)
    return FundamentalData(
        symbol=symbol,
        pe_ratio=pe,
        pb_ratio=SECTOR_AVG_PB.get(sector, 3.0),
        peg_ratio=1.5 if pe else None,
        dividend_yield=0.02,
        payout_ratio=0.35,
        dividend_growth_5y=0.05,
        fcf=1e9,
        revenue_growth_yoy=0.08,
        sector_pe_avg=SECTOR_AVG_PE.get(sector, 20.0),
        sector_pb_avg=SECTOR_AVG_PB.get(sector, 3.0),
        price=100.0,
    )


def _fetch_polygon_fundamentals(symbol: str) -> FundamentalData:
    """Polygon.io fundamental data - stub for future implementation."""
    return _mock_fundamentals(symbol, None)


def _fetch_finnhub_fundamentals(symbol: str) -> FundamentalData:
    """Finnhub fundamental data - stub for future implementation."""
    return _mock_fundamentals(symbol, None)


def _pe_score(pe: float, sector_pe_avg: float) -> float:
    """0-10: lower P/E vs sector = higher score (undervalued)."""
    if not pe or pe <= 0:
        return 5.0
    if not sector_pe_avg or sector_pe_avg <= 0:
        sector_pe_avg = 20.0
    ratio = pe / sector_pe_avg
    if ratio <= 0.7:
        return 9.0
    if ratio <= 0.9:
        return 8.0
    if ratio <= 1.0:
        return 7.0
    if ratio <= 1.2:
        return 5.0
    if ratio <= 1.5:
        return 3.0
    return 1.0


def _pb_score(pb: float, sector_pb_avg: float) -> float:
    """0-10: lower P/B = undervalued."""
    if not pb or pb <= 0:
        return 5.0
    if not sector_pb_avg or sector_pb_avg <= 0:
        sector_pb_avg = 3.0
    ratio = pb / sector_pb_avg
    if ratio <= 0.6:
        return 9.0
    if ratio <= 0.8:
        return 8.0
    if ratio <= 1.0:
        return 6.0
    if ratio <= 1.5:
        return 4.0
    return 2.0


def _peg_score(peg: float) -> float:
    """0-10: PEG < 1 = growth at reasonable price."""
    if not peg or peg <= 0:
        return 5.0
    if peg <= 0.5:
        return 9.0
    if peg <= 1.0:
        return 8.0
    if peg <= 1.5:
        return 6.0
    if peg <= 2.0:
        return 4.0
    return 2.0


def _dividend_score(dy: float, payout: float, growth: float) -> float:
    """0-10: sustainable yield + growth."""
    if not dy or dy <= 0:
        return 5.0
    score = 5.0
    if 0.02 <= dy <= 0.06:
        score += 2.0
    elif dy > 0.06:
        score += 1.0
    if payout and 0.2 <= payout <= 0.6:
        score += 2.0
    if growth and growth > 0.05:
        score += 1.0
    return min(10.0, score)


def _dcf_estimate(fcf: float, growth: float, discount: float = 0.10, years: int = 10) -> float:
    """Simple DCF: FCF * sum((1+g)^t / (1+r)^t) for t=1..years + terminal."""
    if not fcf or fcf <= 0:
        return 0.0
    g = growth or 0.05
    r = discount
    pv = 0.0
    cf = fcf
    for t in range(1, years + 1):
        cf *= (1 + g)
        pv += cf / ((1 + r) ** t)
    terminal = cf * (1 + g) / (r - g) if r > g else cf * 10
    pv += terminal / ((1 + r) ** years)
    return pv


def calculate_valuation_score(
    symbol: str,
    price: Optional[float] = None,
    sector: Optional[str] = None,
    fundamental_data: Optional[FundamentalData] = None,
) -> ValuationScore:
    """
    Produce ValuationScore (0-100) from fundamental metrics.
    Separate from technical score for long-term mode.
    """
    if fundamental_data is None:
        fundamental_data = _fetch_fundamental_data(symbol, sector)
    fd = fundamental_data
    price = price or fd.price
    components: Dict[str, float] = {}
    flags: List[str] = []

    pe_score_val = _pe_score(fd.pe_ratio, fd.sector_pe_avg)
    components["pe"] = pe_score_val
    if fd.pe_ratio and fd.sector_pe_avg and fd.pe_ratio > fd.sector_pe_avg * 2:
        flags.append("PE_EXTREME_HIGH")

    pb_score_val = _pb_score(fd.pb_ratio, fd.sector_pb_avg)
    components["pb"] = pb_score_val

    peg_score_val = _peg_score(fd.peg_ratio)
    components["peg"] = peg_score_val

    div_score_val = _dividend_score(
        fd.dividend_yield or 0,
        fd.payout_ratio or 0,
        fd.dividend_growth_5y or 0,
    )
    components["dividend"] = div_score_val

    fcf_score_val = 5.0
    if fd.fcf and fd.fcf > 0:
        fcf_score_val = min(10.0, 5.0 + (fd.fcf / 1e9) * 0.5)
    components["fcf"] = fcf_score_val

    dcf_intrinsic = None
    dcf_margin = None
    if fd.fcf and fd.fcf > 0 and price and price > 0:
        growth = fd.revenue_growth_yoy or 0.05
        dcf_intrinsic = _dcf_estimate(fd.fcf, growth)
        if dcf_intrinsic > 0:
            dcf_margin = (dcf_intrinsic - price) / price
            if dcf_margin > 0.3:
                components["dcf"] = 9.0
            elif dcf_margin > 0.1:
                components["dcf"] = 7.0
            elif dcf_margin > 0:
                components["dcf"] = 6.0
            else:
                components["dcf"] = max(0, 5.0 + dcf_margin * 10)
        else:
            components["dcf"] = 5.0

    weights = {"pe": 0.25, "pb": 0.20, "peg": 0.20, "dividend": 0.15, "fcf": 0.10, "dcf": 0.10}
    total = 0.0
    for k, w in weights.items():
        total += components.get(k, 5.0) * w
    score = min(100.0, max(0.0, (total / 0.9) * 10))

    return ValuationScore(
        score=round(score, 1),
        components=components,
        dcf_intrinsic=dcf_intrinsic,
        dcf_margin=dcf_margin,
        flags=flags,
        raw={
            "pe": fd.pe_ratio, "pb": fd.pb_ratio, "peg": fd.peg_ratio,
            "dividend_yield": fd.dividend_yield, "payout_ratio": fd.payout_ratio,
        },
    )


def conviction_score(valuation_score: float, technical_score: float) -> int:
    """
    Conviction 1-10 from fundamental + technical alignment.
    Higher = wider stops, larger size for long-term mode.
    """
    align = (valuation_score + technical_score) / 2.0
    if align >= 85:
        return 9
    if align >= 75:
        return 8
    if align >= 65:
        return 7
    if align >= 55:
        return 6
    if align >= 45:
        return 5
    if align >= 35:
        return 4
    if align >= 25:
        return 3
    if align >= 15:
        return 2
    return 1
