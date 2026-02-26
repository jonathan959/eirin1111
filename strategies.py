# strategies.py
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple
from abc import ABC, abstractmethod
import logging
import math
import statistics

logger = logging.getLogger(__name__)


@dataclass
class DcaConfig:
    # Quote amounts (e.g., USD)
    base_quote: float
    safety_quote: float

    # Safety orders
    max_safety: int
    first_dev: float        # e.g. 0.015 for 1.5%
    step_mult: float        # e.g. 1.2 (each next deviation grows by this factor)

    # Take profit (as fraction of avg entry)
    tp: float               # e.g. 0.012 for 1.2%

    # Optional trend filter
    trend_filter: bool = False
    trend_sma: int = 200

    # Optional safety constraints (not in DB yet; defaults keep behavior stable)
    min_step_mult: float = 1.0
    max_step_mult: float = 10.0


# =========================
# Math helpers
# =========================
def clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


def safe_float(x, default: float = 0.0) -> float:
    try:
        return float(x)
    except Exception:
        return float(default)


# =========================
# Indicators
# =========================
def sma(values: List[float], n: int) -> Optional[float]:
    if n <= 0 or len(values) < n:
        return None
    return sum(values[-n:]) / n


def ema(values: List[float], n: int) -> Optional[float]:
    """
    Simple EMA for future use. Not required for DCA, but useful infrastructure.
    """
    if n <= 0 or len(values) < n:
        return None
    k = 2.0 / (n + 1.0)
    e = values[0]
    for v in values[1:]:
        e = (v * k) + (e * (1.0 - k))
    return float(e)


def ema_series(values: List[float], n: int) -> List[float]:
    if n <= 0 or not values:
        return []
    k = 2.0 / (n + 1.0)
    out = []
    e = values[0]
    out.append(float(e))
    for v in values[1:]:
        e = (v * k) + (e * (1.0 - k))
        out.append(float(e))
    return out


def rsi(values: List[float], n: int = 14) -> Optional[float]:
    if n <= 0 or len(values) < n + 1:
        return None
    gains = []
    losses = []
    for i in range(-n, 0):
        diff = values[i] - values[i - 1]
        if diff >= 0:
            gains.append(diff)
            losses.append(0.0)
        else:
            gains.append(0.0)
            losses.append(-diff)
    avg_gain = sum(gains) / n
    avg_loss = sum(losses) / n
    if avg_loss == 0:
        return 100.0
    rs = avg_gain / avg_loss
    return 100.0 - (100.0 / (1.0 + rs))


def true_range(high: float, low: float, prev_close: float) -> float:
    return max(high - low, abs(high - prev_close), abs(low - prev_close))


def adx(candles: List[List[float]], n: int = 14) -> Optional[float]:
    if len(candles) < n + 2:
        return None
    trs = []
    plus_dm = []
    minus_dm = []
    for i in range(1, len(candles)):
        _, o, h, l, c = candles[i][:5]
        _, _, prev_h, prev_l, prev_c = candles[i - 1][:5]
        tr = true_range(float(h), float(l), float(prev_c))
        trs.append(tr)
        up_move = float(h) - float(prev_h)
        down_move = float(prev_l) - float(l)
        plus_dm.append(up_move if up_move > down_move and up_move > 0 else 0.0)
        minus_dm.append(down_move if down_move > up_move and down_move > 0 else 0.0)
    if len(trs) < n:
        return None
    tr_n = sum(trs[-n:]) / n
    plus_n = sum(plus_dm[-n:]) / n
    minus_n = sum(minus_dm[-n:]) / n
    if tr_n <= 0:
        return None
    plus_di = 100.0 * (plus_n / tr_n)
    minus_di = 100.0 * (minus_n / tr_n)
    dx = 100.0 * abs(plus_di - minus_di) / max(1e-9, plus_di + minus_di)
    return dx


def bollinger(values: List[float], n: int = 20, k: float = 2.0) -> Optional[Tuple[float, float, float, float]]:
    if n <= 0 or len(values) < n:
        return None
    window = values[-n:]
    mid = sum(window) / n
    st = statistics.pstdev(window)
    upper = mid + (k * st)
    lower = mid - (k * st)
    bandwidth = (upper - lower) / mid if mid else 0.0
    return lower, mid, upper, bandwidth


def donchian(candles: List[List[float]], n: int = 20) -> Optional[Tuple[float, float]]:
    if len(candles) < n:
        return None
    highs = [float(c[2]) for c in candles[-n:]]
    lows = [float(c[3]) for c in candles[-n:]]
    return max(highs), min(lows)


def volume_ma(candles: List[List[float]], n: int = 20) -> Optional[float]:
    if len(candles) < n:
        return None
    vols = [float(c[5]) for c in candles[-n:]]
    return sum(vols) / n


def volume_ratio(candles: List[List[float]], n: int = 20) -> Optional[float]:
    if len(candles) < n + 1:
        return None
    vma = volume_ma(candles, n)
    if not vma:
        return None
    return float(candles[-1][5]) / vma


def trend_ok(closes: List[float], n: int) -> bool:
    s = sma(closes, n)
    if s is None:
        return False
    return closes[-1] >= s


# =========================
# DCA ladder helpers
# =========================
def _validate_cfg(cfg: DcaConfig) -> DcaConfig:
    """
    Defensive normalization to prevent impossible ladders that cause instability.
    """
    cfg.base_quote = max(0.0, safe_float(cfg.base_quote))
    cfg.safety_quote = max(0.0, safe_float(cfg.safety_quote))
    cfg.max_safety = int(max(0, cfg.max_safety))

    cfg.first_dev = clamp(safe_float(cfg.first_dev, 0.01), 0.0001, 0.95)
    cfg.step_mult = clamp(safe_float(cfg.step_mult, 1.2), cfg.min_step_mult, cfg.max_step_mult)
    cfg.tp = clamp(safe_float(cfg.tp, 0.01), 0.0001, 1.0)

    cfg.trend_sma = int(max(5, cfg.trend_sma))
    return cfg


def safety_levels(entry_price: float, cfg: DcaConfig) -> List[float]:
    """
    Returns a list of price levels at which safety orders should trigger.

    Example:
      entry=100, first_dev=0.02, step_mult=1.2, max_safety=3
      devs: 2%, 2.4%, 2.88%
      levels: 98.0, 97.6, 97.12

    Safeguards:
    - Always monotonic decreasing levels
    - Prevent dev from exceeding 95% (would be nonsense)
    """
    cfg = _validate_cfg(cfg)
    entry = max(0.00000001, safe_float(entry_price, 0.0))

    levels: List[float] = []
    dev = cfg.first_dev

    last = entry
    for _ in range(cfg.max_safety):
        dev = clamp(dev, 0.0001, 0.95)
        lvl = entry * (1.0 - dev)
        # ensure strictly decreasing
        if lvl >= last:
            lvl = last * 0.999
        last = lvl
        levels.append(float(lvl))
        dev *= cfg.step_mult

    return levels


def total_planned_spend(cfg: DcaConfig) -> float:
    """
    Total quote spend if all safety orders are used.
    """
    cfg = _validate_cfg(cfg)
    return float(cfg.base_quote + (cfg.safety_quote * cfg.max_safety))


def implied_max_deviation(cfg: DcaConfig) -> float:
    """
    Approximate maximum ladder deviation (sum-like, but here we return the last deviation).
    Useful for UI warnings later.
    """
    cfg = _validate_cfg(cfg)
    dev = cfg.first_dev
    for _ in range(max(0, cfg.max_safety - 1)):
        dev *= cfg.step_mult
    return float(clamp(dev, 0.0, 0.95))


def next_safety_trigger(entry_price: float, cfg: DcaConfig, safety_used: int) -> Optional[float]:
    """
    Convenience: get the next safety trigger price based on safety_used count.
    """
    lvls = safety_levels(entry_price, cfg)
    if safety_used < 0 or safety_used >= len(lvls):
        return None
    return float(lvls[safety_used])


# =========================
# Smart DCA strategy
# =========================
@dataclass
class RegimeResult:
    regime: str
    confidence: float
    why: List[str]
    snapshot: Dict[str, Any]
    scores: Dict[str, float] = None
    # Back-compat fields (legacy UI/logic)
    sma200: Optional[float] = None
    ema200: Optional[float] = None
    atr14: Optional[float] = None
    slope: Optional[float] = None
    vol_ratio: Optional[float] = None
    thresholds: Dict[str, float] = None
    legacy_regime: Optional[str] = None


@dataclass
class Decision:
    action: str
    reason: str
    order: Optional[Dict[str, Any]]
    debug: Dict[str, Any]
    strategy: Optional[str] = None


@dataclass
class DealState:
    avg_entry: Optional[float]
    position_size: float
    safety_used: int
    tp_price: Optional[float]
    spent_quote: float = 0.0
    trailing_active: bool = False
    trailing_price: Optional[float] = None


@dataclass
class AccountSnapshot:
    total_usd: float
    free_usd: float
    used_usd: float
    positions_usd: float


@dataclass
class PerformanceStats:
    realized_today: float
    drawdown: float
    open_deals: int


@dataclass
class SmartDcaConfig:
    base_quote: float
    safety_quote: float
    max_safety: int
    tp: float
    atr_mult: float = 1.2
    atr_trail_mult: float = 1.0
    trail_start_mult: float = 1.2
    vol_tp_mult: float = 2.0
    vol_gap_mult: float = 1.0
    min_gap_pct: float = 0.003
    max_gap_pct: float = 0.06
    base_quote_mult: float = 1.0
    safety_quote_mult: float = 1.0
    max_exposure_pct: float = 0.15
    max_spend_quote: float = 0.0
    max_daily_loss: float = 25.0
    max_drawdown: float = 0.20
    cooldown_sec: int = 300
    safety_cooldown_sec: int = 120
    max_open_deals: int = 3
    vol_pause_ratio: float = 0.05
    mean_rev_pct: float = 0.02
    stop_loss_pct: float = 0.08
    max_drawdown_pct: float = 0.0  # Per-position: halt when unrealized loss >= this (0 = disabled)
    # Swing: scale-in over days, time stop, breakeven exit
    scale_in_days: int = 0
    scale_in_amounts: str = "1.0"
    max_days_no_progress: int = 0
    breakeven_exit_enabled: bool = False


def _atr(candles: List[List[float]], n: int = 14) -> Optional[float]:
    if not candles or len(candles) < n + 1:
        return None
    trs: List[float] = []
    for i in range(1, len(candles)):
        _, o, h, l, c = candles[i][:5]
        _, _, _, _, prev_c = candles[i - 1][:5]
        tr = max(float(h) - float(l), abs(float(h) - float(prev_c)), abs(float(l) - float(prev_c)))
        trs.append(tr)
    if len(trs) < n:
        return None
    return sum(trs[-n:]) / float(n)


def _slope(values: List[float], n: int = 20) -> Optional[float]:
    if len(values) < n:
        return None
    ys = values[-n:]
    xs = list(range(n))
    x_mean = sum(xs) / n
    y_mean = sum(ys) / n
    num = sum((x - x_mean) * (y - y_mean) for x, y in zip(xs, ys))
    den = sum((x - x_mean) ** 2 for x in xs) or 1.0
    return num / den


def rolling_return(values: List[float], periods: int) -> Optional[float]:
    if periods <= 0 or len(values) < periods + 1:
        return None
    start = float(values[-(periods + 1)])
    end = float(values[-1])
    if start == 0:
        return None
    return (end / start) - 1.0


def momentum_score(closes: List[float], n: int = 14) -> float:
    """
    Composite momentum score in [-1, 1]. Positive = bullish, negative = bearish.
    Uses rolling return and slope; used for dynamic TP/SL scaling.
    """
    if not closes or len(closes) < max(n, 5):
        return 0.0
    ret = rolling_return(closes, n) or 0.0
    slp = _slope(closes, min(20, len(closes))) or 0.0
    price = float(closes[-1]) if closes else 1.0
    slope_pct = (slp / price) * 100.0 if price and price > 0 else 0.0
    slope_norm = clamp(slope_pct * 2.0, -0.5, 0.5)
    return clamp(ret * 4.0 + slope_norm, -1.0, 1.0)


def max_drawdown(values: List[float], lookback: Optional[int] = None) -> Optional[float]:
    if not values:
        return None
    series = values[-lookback:] if lookback and len(values) >= lookback else values
    peak = series[0]
    max_dd = 0.0
    for v in series:
        if v > peak:
            peak = v
        dd = (peak - v) / peak if peak > 0 else 0.0
        if dd > max_dd:
            max_dd = dd
    return float(max_dd)


def current_drawdown(values: List[float], lookback: Optional[int] = None) -> Optional[float]:
    if not values:
        return None
    series = values[-lookback:] if lookback and len(values) >= lookback else values
    peak = max(series) if series else 0.0
    if peak <= 0:
        return None
    return float((peak - series[-1]) / peak)


def lower_lows_persistence(values: List[float], window: int = 5, periods: int = 8) -> bool:
    if len(values) < window + periods:
        return False
    mins = []
    for i in range(-periods, 0):
        start = i - window + 1
        segment = values[start:i + 1]
        if not segment:
            return False
        mins.append(min(segment))
    return all(earlier > later for earlier, later in zip(mins, mins[1:]))


def base_formation(values: List[float], window: int = 12, max_range_pct: float = 0.25) -> bool:
    if len(values) < window:
        return False
    segment = values[-window:]
    low = min(segment)
    high = max(segment)
    mid = (high + low) / 2 if (high + low) > 0 else 0.0
    if mid <= 0:
        return False
    range_pct = (high - low) / mid
    slope_val = _slope(segment, min(10, len(segment))) or 0.0
    half = window // 2
    first_low = min(segment[:half]) if half else low
    last_low = min(segment[half:]) if half else low
    higher_low = last_low > first_low
    return range_pct <= max_range_pct and abs(slope_val) <= 0.0005 and higher_low


def detect_regime(candles: List[List[float]]) -> RegimeResult:
    confidence = 0.0  # Always initialized; never leave undefined
    closes = [float(c[4]) for c in candles if len(c) >= 5]
    if not closes:
        return RegimeResult(
            regime="RISK_OFF",
            confidence=0.0,
            why=["no_data"],
            snapshot={},
            scores={
                "uptrend_score": 0.0,
                "downtrend_score": 0.0,
                "range_score": 0.0,
                "high_vol_score": 0.0,
            },
            legacy_regime="HIGH_VOLATILITY",
        )

    try:
        return _detect_regime_impl(closes, candles)
    except Exception as e:
        logger.exception("detect_regime: computation failed: %s", e)
        return RegimeResult(
            regime="range",
            confidence=0.0,
            why=["exception"],
            snapshot={},
            scores={
                "uptrend_score": 0.0,
                "downtrend_score": 0.0,
                "range_score": 0.0,
                "high_vol_score": 0.0,
            },
            legacy_regime="RANGE",
        )


def _detect_regime_impl(closes: List[float], candles: List[List[float]]) -> RegimeResult:
    price = closes[-1]
    sma200 = sma(closes, 200)
    ema200 = ema(closes, 200)
    atr14 = _atr(candles, 14)
    slope20 = _slope(closes, 20)
    adx14 = adx(candles, 14)
    bb = bollinger(closes, 20, 2.0)
    don = donchian(candles, 20)
    volr = volume_ratio(candles, 20)
    vol_ratio = (atr14 / price) if (atr14 is not None and price > 0) else None

    thresholds = {
        "adx_trend": 22.0,
        "adx_range": 22.0,
        "bb_bw_low": 0.05,
        "bb_bw_high": 0.09,
        "atr_high": 0.04,
        "vol_breakout": 1.5,
        "slope_min": 0.0,
        "slope_flat": 0.0001,
    }

    why: List[str] = []
    confidence = 0.5
    regime = "RANGING"
    legacy = "RANGE"

    # Score vector (0..1)
    uptrend_score = 0.0
    downtrend_score = 0.0
    range_score = 0.0
    high_vol_score = 0.0
    ema_fast = ema(closes, 12) or 0.0
    ema_slow = ema(closes, 26) or 0.0

    if ema_fast > ema_slow:
        uptrend_score += 0.25
    if ema_fast < ema_slow:
        downtrend_score += 0.25

    if slope20 is not None and slope20 > 0:
        uptrend_score += 0.2
    if slope20 is not None and slope20 < 0:
        downtrend_score += 0.2

    if sma200 is not None and price > sma200:
        uptrend_score += 0.15
    if sma200 is not None and price < sma200:
        downtrend_score += 0.15

    if ema200 is not None and price > ema200:
        uptrend_score += 0.1
    if ema200 is not None and price < ema200:
        downtrend_score += 0.1

    if adx14 is not None and adx14 >= thresholds["adx_trend"]:
        if uptrend_score >= downtrend_score:
            uptrend_score += 0.2
        else:
            downtrend_score += 0.2

    if adx14 is not None and adx14 <= thresholds["adx_range"]:
        range_score += 0.4
    if bb and bb[3] <= thresholds["bb_bw_low"]:
        range_score += 0.3
    if slope20 is not None and abs(slope20) <= thresholds["slope_flat"]:
        range_score += 0.3

    if vol_ratio is not None:
        if vol_ratio >= thresholds["atr_high"]:
            high_vol_score = clamp((vol_ratio - thresholds["atr_high"]) / thresholds["atr_high"], 0.0, 1.0)
        else:
            high_vol_score = clamp(vol_ratio / max(1e-9, thresholds["atr_high"]), 0.0, 1.0)

    uptrend_score = clamp(uptrend_score, 0.0, 1.0)
    downtrend_score = clamp(downtrend_score, 0.0, 1.0)
    range_score = clamp(range_score, 0.0, 1.0)
    high_vol_score = clamp(high_vol_score, 0.0, 1.0)

    # High volatility / risk off
    if vol_ratio is not None and vol_ratio >= thresholds["atr_high"]:
        regime = "HIGH_VOL_RISK"
        legacy = "HIGH_VOLATILITY"
        why.append("atr_high")
        confidence = 0.75

    # Breakout detection
    if bb and don and volr is not None:
        _, bb_mid, _, bb_bw = bb
        don_high, don_low = don
        if bb_bw >= thresholds["bb_bw_high"] and volr >= thresholds["vol_breakout"]:
            if price >= don_high:
                regime = "BREAKOUT_UP"
                legacy = "UPTREND"
                why.extend(["bb_expand", "donchian_up", "vol_confirm"])
                confidence = 0.8
            elif price <= don_low:
                regime = "BREAKOUT_DOWN"
                legacy = "DOWNTREND"
                why.extend(["bb_expand", "donchian_down", "vol_confirm"])
                confidence = 0.8

    # Trend detection
    if adx14 is not None and adx14 >= thresholds["adx_trend"]:
        if sma200 is not None and price > sma200 and (slope20 or 0) > thresholds["slope_min"]:
            regime = "TREND_UP"
            legacy = "UPTREND"
            why.extend(["adx_trend", "price_above_ma", "slope_up"])
            confidence = max(confidence, 0.72)
        elif sma200 is not None and price < sma200 and (slope20 or 0) < -thresholds["slope_min"]:
            regime = "TREND_DOWN"
            legacy = "DOWNTREND"
            why.extend(["adx_trend", "price_below_ma", "slope_down"])
            confidence = max(confidence, 0.72)

    # Range detection
    if adx14 is not None and adx14 <= thresholds["adx_range"]:
        if bb and bb[3] <= thresholds["bb_bw_low"]:
            regime = "RANGING"
            legacy = "RANGE"
            why.extend(["adx_low", "bb_squeeze"])
            confidence = max(confidence, 0.68)

    # Risk off if extreme drawdown or missing indicators
    if atr14 is None or sma200 is None:
        regime = "RISK_OFF"
        legacy = "HIGH_VOLATILITY"
        why.append("insufficient_history")
        confidence = min(confidence, 0.4)
        high_vol_score = max(high_vol_score, 0.6)

    snapshot = {
        "price": price,
        "sma200": sma200,
        "ema200": ema200,
        "atr14": atr14,
        "slope20": slope20,
        "adx14": adx14,
        "bb_bw": bb[3] if bb else None,
        "don_high": don[0] if don else None,
        "don_low": don[1] if don else None,
        "vol_ratio": vol_ratio,
        "vol_confirm": volr,
    }

    scores = {
        "uptrend_score": uptrend_score,
        "downtrend_score": downtrend_score,
        "range_score": range_score,
        "high_vol_score": high_vol_score,
    }
    try:
        confidence = max(scores.values()) if scores else 0.0
    except Exception as e:
        logger.exception("detect_regime: confidence from scores failed: %s", e)
        confidence = 0.0
        regime = "unknown" if regime not in ("RISK_OFF", "RANGING", "TREND_UP", "TREND_DOWN") else regime

    return RegimeResult(
        regime=regime,
        confidence=confidence,
        why=why,
        snapshot=snapshot,
        scores=scores,
        sma200=sma200,
        ema200=ema200,
        atr14=atr14,
        slope=slope20,
        vol_ratio=vol_ratio,
        thresholds=thresholds,
        legacy_regime=legacy,
    )


def _safe_cfg_float(cfg: Dict[str, Any], key: str, default: float) -> float:
    v = cfg.get(key)
    if v is None or (isinstance(v, str) and v.strip() == ""):
        return default
    try:
        return float(v)
    except (TypeError, ValueError):
        return default


def _safe_cfg_int(cfg: Dict[str, Any], key: str, default: int) -> int:
    v = cfg.get(key)
    if v is None or (isinstance(v, str) and v.strip() == ""):
        return default
    try:
        return int(v)
    except (TypeError, ValueError):
        return default


def _safe_cfg_str(cfg: Dict[str, Any], key: str, default: str) -> str:
    v = cfg.get(key)
    if v is None:
        return default
    return str(v).strip() if str(v).strip() else default


def build_smart_dca_config(ctx_or_cfg: Any, overrides: Optional[Dict[str, Any]] = None) -> SmartDcaConfig:
    """
    Build SmartDcaConfig from ctx.cfg or bot config dict.
    Passes through risk caps from bot config; overrides merges on top for strategy-specific tweaks.
    """
    if hasattr(ctx_or_cfg, "cfg"):
        cfg = dict(ctx_or_cfg.cfg or {})
    elif isinstance(ctx_or_cfg, dict):
        cfg = dict(ctx_or_cfg)
    else:
        cfg = {}
    if overrides:
        cfg = {**cfg, **overrides}

    import os
    return SmartDcaConfig(
        base_quote=_safe_cfg_float(cfg, "base_quote", 20.0),
        safety_quote=_safe_cfg_float(cfg, "safety_quote", 20.0),
        max_safety=max(0, _safe_cfg_int(cfg, "max_safety", 5)),
        tp=_safe_cfg_float(cfg, "tp", 0.01),
        atr_mult=float(os.getenv("SMART_ATR_MULT", "1.2")),
        atr_trail_mult=float(os.getenv("SMART_TRAIL_ATR_MULT", "1.0")),
        trail_start_mult=float(os.getenv("SMART_TRAIL_START_MULT", "1.2")),
        vol_tp_mult=_safe_cfg_float(cfg, "tp_vol_mult", 1.0),
        vol_gap_mult=_safe_cfg_float(cfg, "vol_gap_mult", 1.0),
        min_gap_pct=_safe_cfg_float(cfg, "min_gap_pct", 0.003),
        max_gap_pct=_safe_cfg_float(cfg, "max_gap_pct", 0.06),
        base_quote_mult=_safe_cfg_float(cfg, "base_quote_mult", 1.0),
        safety_quote_mult=_safe_cfg_float(cfg, "safety_quote_mult", 1.0),
        max_exposure_pct=_safe_cfg_float(cfg, "max_total_exposure_pct", 0.15) or float(os.getenv("SMART_MAX_EXPOSURE_PCT", "0.15")),
        max_spend_quote=_safe_cfg_float(cfg, "max_spend_quote", 0.0),
        max_daily_loss=_safe_cfg_float(cfg, "max_daily_loss", float(os.getenv("SMART_MAX_DAILY_LOSS", "25"))),
        max_drawdown=_safe_cfg_float(cfg, "max_drawdown", 0.20) or float(os.getenv("SMART_MAX_DRAWDOWN", "0.20")),
        cooldown_sec=(_safe_cfg_int(cfg, "min_minutes_between_entries", 5) * 60) if "min_minutes_between_entries" in cfg else int(os.getenv("SMART_COOLDOWN_SEC", "300")),
        safety_cooldown_sec=(_safe_cfg_int(cfg, "min_minutes_between_safety_orders", 2) * 60) if "min_minutes_between_safety_orders" in cfg else int(os.getenv("SMART_SAFETY_COOLDOWN_SEC", "120")),
        max_open_deals=_safe_cfg_int(cfg, "max_concurrent_deals", 3) or _safe_cfg_int(cfg, "max_open_orders", 3) or int(os.getenv("SMART_MAX_OPEN_DEALS", "3")),
        vol_pause_ratio=float(os.getenv("SMART_VOL_PAUSE_RATIO", "0.05")),
        mean_rev_pct=float(os.getenv("SMART_MEAN_REV_PCT", "0.02")),
        stop_loss_pct=_safe_cfg_float(cfg, "stop_loss_pct", 0.08) or float(os.getenv("SMART_STOP_LOSS_PCT", "0.08")),
        max_drawdown_pct=_safe_cfg_float(cfg, "max_drawdown_pct", 0.0),
        # Swing trading: scale-in, time stop, breakeven
        scale_in_days=_safe_cfg_int(cfg, "SCALE_IN_DAYS", 0),
        scale_in_amounts=_safe_cfg_str(cfg, "SCALE_IN_AMOUNTS", "1.0"),
        max_days_no_progress=_safe_cfg_int(cfg, "MAX_DAYS_NO_PROGRESS", 0),
        breakeven_exit_enabled=bool(_safe_cfg_int(cfg, "BREAKEVEN_EXIT_ENABLED", 0)),
    )


def risk_check(
    cfg: SmartDcaConfig,
    account: AccountSnapshot,
    perf: PerformanceStats,
    position_value: float,
    spent_quote: float,
    now_ts: int,
    cooldown_until: int,
) -> Optional[str]:
    if now_ts < cooldown_until:
        return "Cooldown active. Pausing new actions."
    if account.total_usd > 0 and position_value / account.total_usd > cfg.max_exposure_pct:
        return "Exposure limit exceeded."
    if cfg.max_spend_quote > 0 and spent_quote > cfg.max_spend_quote:
        return "Max spend limit exceeded."
    if perf.open_deals >= cfg.max_open_deals:
        return "Max open deals reached."
    if perf.realized_today <= -abs(cfg.max_daily_loss):
        return "Daily loss limit reached."
    if perf.drawdown >= cfg.max_drawdown:
        return "Drawdown limit reached."
    return None


def _parse_scale_in_amounts(s: str) -> List[float]:
    """Parse SCALE_IN_AMOUNTS e.g. '0.25,0.25,0.5' into list of fractions."""
    try:
        parts = [float(x.strip()) for x in str(s or "1.0").split(",") if x.strip()]
        if not parts:
            return [1.0]
        total = sum(parts)
        if total <= 0:
            return [1.0]
        return [p / total for p in parts]
    except Exception:
        return [1.0]


def smart_decide(
    candles: List[List[float]],
    last_price: float,
    cfg: SmartDcaConfig,
    deal: DealState,
    account: AccountSnapshot,
    perf: PerformanceStats,
    now_ts: int,
    cooldown_until: int,
    deal_opened_at: Optional[int] = None,
    scale_in_tranche_index: int = 0,
    scale_in_last_add_ts: Optional[int] = None,
) -> Decision:
    regime = detect_regime(candles)
    atr = regime.atr14 or 0.0
    vol_ratio = regime.vol_ratio or 0.0
    position_value = float(deal.position_size) * float(last_price or 0.0)
    base_quote = float(cfg.base_quote) * float(cfg.base_quote_mult or 1.0)
    safety_quote = float(cfg.safety_quote) * float(cfg.safety_quote_mult or 1.0)
    tranches = _parse_scale_in_amounts(cfg.scale_in_amounts) if cfg.scale_in_days > 0 else [1.0]
    tranche_pct = tranches[min(scale_in_tranche_index, len(tranches) - 1)] if tranches else 1.0
    scaled_base = base_quote * tranche_pct

    risk_reason = risk_check(cfg, account, perf, position_value, float(deal.spent_quote or 0.0), now_ts, cooldown_until)
    if risk_reason:
        return Decision("PAUSE", risk_reason, None, {"regime": regime})

    if vol_ratio and vol_ratio >= cfg.vol_pause_ratio:
        return Decision("PAUSE", "Volatility spike detected. Pausing.", None, {"regime": regime})

    # Entry logic
    if deal.position_size <= 0:
        if regime.regime in ("TREND_DOWN", "HIGH_VOL_RISK", "RISK_OFF", "BREAKOUT_DOWN"):
            # mean reversion allowance
            if regime.sma200 and last_price < regime.sma200 * (1.0 - cfg.mean_rev_pct):
                return Decision(
                    "ENTER",
                    "Mean reversion entry: price below SMA200.",
                    {"side": "buy", "type": "market", "size_quote": scaled_base, "scale_in_tranche": scale_in_tranche_index},
                    {"regime": regime},
                    "smart_dca",
                )
            return Decision("HOLD", f"Regime {regime.regime} not suitable for entry.", None, {"regime": regime}, "smart_dca")
        return Decision(
            "ENTER",
            f"Regime {regime.regime} allows entry.",
            {"side": "buy", "type": "market", "size_quote": scaled_base, "scale_in_tranche": scale_in_tranche_index},
            {"regime": regime},
            "smart_dca",
        )

    # In position
    # Swing failure: exit at breakeven if no progress after MAX_DAYS_NO_PROGRESS
    if cfg.max_days_no_progress > 0 and cfg.breakeven_exit_enabled and deal_opened_at and deal.avg_entry and deal.position_size > 0:
        days_held = (now_ts - deal_opened_at) / 86400.0
        entry = float(deal.avg_entry)
        progress_pct = (last_price - entry) / entry if entry > 0 else 0.0
        if days_held >= cfg.max_days_no_progress and progress_pct < 0.01:
            return Decision(
                "EXIT",
                f"Time stop: no progress after {int(days_held)} days, exit at breakeven.",
                {"side": "sell", "type": "limit", "price": entry, "size_base": deal.position_size},
                {"regime": regime},
                "smart_dca",
            )

    # Scale-in: add next tranche after 24h if confirmation (price near or above entry)
    # scale_in_tranche_index = completed tranches (0=first entry done, 1=second done, ...)
    if cfg.scale_in_days > 0 and scale_in_tranche_index < len(tranches) - 1 and scale_in_tranche_index < cfg.scale_in_days - 1:
        last_add = scale_in_last_add_ts or deal_opened_at or 0
        sec_since = now_ts - last_add
        if sec_since >= 86400 and deal.avg_entry:  # 24h minimum between tranches
            entry = float(deal.avg_entry)
            confirmation_ok = last_price >= entry * 0.98
            if confirmation_ok and regime.regime not in ("TREND_DOWN", "RISK_OFF", "BREAKOUT_DOWN"):
                next_idx = scale_in_tranche_index + 1
                next_pct = tranches[next_idx]
                add_quote = base_quote * next_pct
                return Decision(
                    "SCALE_IN",
                    f"Scale-in tranche {next_idx + 1}/{len(tranches)} after 24h confirmation.",
                    {"side": "buy", "type": "market", "size_quote": add_quote, "scale_in_tranche": next_idx},
                    {"regime": regime},
                    "smart_dca",
                )
    # Max drawdown halt: sell if unrealized loss exceeds threshold (per-position)
    spent = float(deal.spent_quote or 0.0)
    if cfg.max_drawdown_pct > 0 and spent > 0 and deal.position_size > 0 and last_price > 0:
        current_value = float(deal.position_size) * float(last_price)
        unrealized_loss_pct = (spent - current_value) / spent
        if unrealized_loss_pct >= cfg.max_drawdown_pct:
            return Decision(
                "STOP_LOSS",
                f"Max drawdown halt: unrealized loss {unrealized_loss_pct*100:.1f}% >= {cfg.max_drawdown_pct*100:.0f}%",
                {"side": "sell", "type": "market", "size_base": deal.position_size},
                {"regime": regime},
                "smart_dca",
            )

    scores = regime.scores or {}
    trend_adj = 1.0 + (scores.get("uptrend_score", 0.0) * 0.4) - (scores.get("range_score", 0.0) * 0.3)
    vol_adj = 1.0 + (vol_ratio * float(cfg.vol_tp_mult))
    base_tp = float(cfg.tp)
    mom = momentum_score([c[4] for c in candles] if candles and all(len(c) >= 5 for c in candles) else [], 14)
    if mom > 0.2:
        momentum_tp_mult = 1.15
        momentum_sl_mult = 0.95
    elif mom > 0.1:
        momentum_tp_mult = 1.08
        momentum_sl_mult = 1.0
    elif mom < -0.15:
        momentum_tp_mult = 0.92
        momentum_sl_mult = 1.2
    elif mom < -0.05:
        momentum_tp_mult = 0.97
        momentum_sl_mult = 1.1
    else:
        momentum_tp_mult = 1.0
        momentum_sl_mult = 1.0
    tp_pct = max(0.002, base_tp * trend_adj * vol_adj * momentum_tp_mult)
    tp_price = float(deal.avg_entry or last_price) * (1.0 + tp_pct)

    if last_price >= tp_price:
        return Decision(
            "TAKE_PROFIT",
            "Take profit target reached.",
            {"side": "sell", "type": "limit", "price": tp_price, "size_base": deal.position_size},
            {"regime": regime, "tp_price": tp_price},
            "smart_dca",
        )

    # Stop-loss guard (cooldown handled by caller); dynamic SL from volatility + momentum
    if deal.avg_entry and deal.position_size > 0:
        sl_pct = max(0.001, float(cfg.stop_loss_pct) * (1.0 + vol_ratio) * momentum_sl_mult)
        sl_price = float(deal.avg_entry) * (1.0 - sl_pct)
        if last_price <= sl_price:
            return Decision(
                "STOP_LOSS",
                "Stop-loss triggered.",
                {"side": "sell", "type": "market", "size_base": deal.position_size},
                {"regime": regime, "sl_price": sl_price},
                "smart_dca",
            )

    if cfg.atr_trail_mult > 0 and atr > 0:
        trail_start = float(deal.avg_entry or last_price) * (1.0 + tp_pct * cfg.trail_start_mult)
        if last_price >= trail_start:
            trail_price = last_price - (atr * cfg.atr_trail_mult)
            return Decision(
                "TRAIL_TP_UPDATE",
                "Trailing TP update.",
                {"trail_price": trail_price},
                {"regime": regime, "trail_price": trail_price},
                "smart_dca",
            )

    # Safety order
    if deal.safety_used < cfg.max_safety and atr > 0 and last_price > 0:
        gap_pct = (atr / last_price) * float(cfg.vol_gap_mult)
        gap_pct = clamp(gap_pct, float(cfg.min_gap_pct), float(cfg.max_gap_pct))
        gap_adj = 1.0 + (scores.get("high_vol_score", 0.0) * 0.6) - (scores.get("range_score", 0.0) * 0.3)
        gap_pct = clamp(gap_pct * gap_adj, float(cfg.min_gap_pct), float(cfg.max_gap_pct))
        gap = last_price * gap_pct * (1.0 + deal.safety_used * 0.5)
        next_price = float(deal.avg_entry or last_price) - gap
        if last_price <= next_price:
            return Decision(
                "SAFETY_ORDER",
                "Safety order triggered by ATR gap.",
                {
                    "side": "buy",
                    "type": "market",
                    "size_quote": safety_quote,
                    "cooldown_sec": int(cfg.safety_cooldown_sec or 0),
                },
                {"regime": regime, "next_price": next_price},
                "smart_dca",
            )

    return Decision(
        "HOLD",
        f"Waiting for TP {tp_price:.2f} or next safety.",
        None,
        {"regime": regime, "tp_price": tp_price},
        "smart_dca",
    )


@dataclass
class StrategyContext:
    symbol: str
    last_price: float
    candles_5m: List[List[float]]
    candles_15m: List[List[float]]
    candles_1h: List[List[float]]
    candles_4h: List[List[float]]
    deal: DealState
    account: AccountSnapshot
    perf: PerformanceStats
    now_ts: int
    cooldown_until: int
    cfg: Dict[str, Any]
    regime: RegimeResult
    candles_1m: Optional[List[List[float]]] = None
    deal_opened_at: Optional[int] = None


class StrategyBase(ABC):
    name: str = "base"

    @abstractmethod
    def propose_orders(self, ctx: StrategyContext) -> Decision:
        raise NotImplementedError

    def risk_checks(self, ctx: StrategyContext) -> Optional[str]:
        return None

    def explain_decision(self, decision: Decision) -> Dict[str, Any]:
        return {
            "strategy": decision.strategy or self.name,
            "action": decision.action,
            "reason": decision.reason,
            "regime": decision.debug.get("regime") if decision.debug else None,
        }

    def decide(self, ctx: StrategyContext) -> Decision:
        reason = self.risk_checks(ctx)
        if reason:
            return Decision("PAUSE", reason, None, {"regime": ctx.regime}, self.name)
        decision = self.propose_orders(ctx)
        decision.strategy = self.name
        return decision


class SmartDcaStrategy(StrategyBase):
    name = "smart_dca"

    def propose_orders(self, ctx: StrategyContext) -> Decision:
        cfg = build_smart_dca_config(ctx)
        scale_in_idx = int(ctx.cfg.get("scale_in_tranche_index", 0) or 0)
        scale_in_last = ctx.cfg.get("scale_in_last_add_ts")
        scale_in_last = int(scale_in_last) if scale_in_last is not None else None
        logger.debug("SmartDcaStrategy effective config: max_exposure=%.2f max_daily_loss=%.1f max_drawdown=%.2f stop_loss=%.2f", cfg.max_exposure_pct, cfg.max_daily_loss, cfg.max_drawdown, cfg.stop_loss_pct)
        return smart_decide(
            candles=ctx.candles_15m,
            last_price=ctx.last_price,
            cfg=cfg,
            deal=ctx.deal,
            account=ctx.account,
            perf=ctx.perf,
            now_ts=ctx.now_ts,
            cooldown_until=ctx.cooldown_until,
            deal_opened_at=ctx.deal_opened_at,
            scale_in_tranche_index=scale_in_idx,
            scale_in_last_add_ts=scale_in_last,
        )


class ClassicDcaStrategy(StrategyBase):
    name = "classic_dca"

    def propose_orders(self, ctx: StrategyContext) -> Decision:
        cfg = DcaConfig(
            base_quote=float(ctx.cfg.get("base_quote", 0.0)),
            safety_quote=float(ctx.cfg.get("safety_quote", 0.0)),
            max_safety=int(ctx.cfg.get("max_safety", 0)),
            first_dev=float(ctx.cfg.get("first_dev", 0.01)),
            step_mult=float(ctx.cfg.get("step_mult", 1.2)),
            tp=float(ctx.cfg.get("tp", 0.01)),
            trend_filter=bool(int(ctx.cfg.get("trend_filter", 0) or 0)),
            trend_sma=int(ctx.cfg.get("trend_sma", 200)),
        )
        closes = [float(c[4]) for c in ctx.candles_15m] if ctx.candles_15m else []

        if ctx.deal.position_size <= 0:
            if cfg.trend_filter and closes:
                if not trend_ok(closes, cfg.trend_sma):
                    return Decision("HOLD", "Trend filter blocking entry.", None, {"regime": ctx.regime}, self.name)
            return Decision(
                "ENTER",
                "Classic DCA entry.",
                {"side": "buy", "type": "market", "size_quote": cfg.base_quote},
                {"regime": ctx.regime},
                self.name,
            )

        tp_price = float(ctx.deal.avg_entry or ctx.last_price) * (1.0 + float(cfg.tp))
        if ctx.last_price >= tp_price:
            return Decision(
                "TAKE_PROFIT",
                "Classic TP reached.",
                {"side": "sell", "type": "limit", "price": tp_price, "size_base": ctx.deal.position_size},
                {"regime": ctx.regime, "tp_price": tp_price},
                self.name,
            )

        next_price = next_safety_trigger(float(ctx.deal.avg_entry or ctx.last_price), cfg, ctx.deal.safety_used)
        if next_price and ctx.last_price <= next_price:
            return Decision(
                "SAFETY_ORDER",
                "Classic safety order.",
                {"side": "buy", "type": "market", "size_quote": cfg.safety_quote},
                {"regime": ctx.regime, "next_price": next_price},
                self.name,
            )

        return Decision("HOLD", "Classic DCA holding.", None, {"regime": ctx.regime}, self.name)


class GridStrategy(StrategyBase):
    name = "grid"

    def propose_orders(self, ctx: StrategyContext) -> Decision:
        candles = ctx.candles_15m
        closes = [float(c[4]) for c in candles] if candles else []
        if len(closes) < 50:
            return Decision("HOLD", "Not enough data for grid.", None, {"regime": ctx.regime}, self.name)
        atr_val = _atr(candles, 14) or 0.0
        recent_high = max(closes[-50:])
        recent_low = min(closes[-50:])
        pad = atr_val * 1.5
        grid_low = max(0.0000001, recent_low - pad)
        grid_high = recent_high + pad
        if grid_high <= grid_low:
            return Decision("HOLD", "Grid range invalid.", None, {"regime": ctx.regime}, self.name)

        levels = int(ctx.cfg.get("grid_levels", 6) or 6)
        levels = max(4, min(12, levels))
        step = (grid_high - grid_low) / levels
        if step <= 0:
            return Decision("HOLD", "Grid step invalid.", None, {"regime": ctx.regime}, self.name)

        return Decision(
            "GRID_MAINTAIN",
            "Maintain grid orders.",
            {
                "grid_low": grid_low,
                "grid_high": grid_high,
                "levels": levels,
                "step": step,
                "side": "both",
            },
            {"regime": ctx.regime},
            self.name,
        )


class TrendStrategy(StrategyBase):
    name = "trend"

    def propose_orders(self, ctx: StrategyContext) -> Decision:
        closes = [float(c[4]) for c in ctx.candles_1h] if ctx.candles_1h else []
        if len(closes) < 60:
            return Decision("HOLD", "Not enough data for trend.", None, {"regime": ctx.regime}, self.name)
        ema_fast = ema(closes, 12) or 0.0
        ema_slow = ema(closes, 26) or 0.0
        adx_val = adx(ctx.candles_1h, 14) or 0.0
        slope_val = _slope(closes, 20) or 0.0

        if ctx.deal.position_size <= 0:
            if ema_fast > ema_slow and adx_val >= 20 and slope_val > 0:
                return Decision(
                    "ENTER",
                    "EMA trend entry.",
                    {"side": "buy", "type": "market", "size_quote": float(ctx.cfg.get("base_quote", 0.0))},
                    {"regime": ctx.regime, "ema_fast": ema_fast, "ema_slow": ema_slow, "adx": adx_val},
                    self.name,
                )
            return Decision("HOLD", "No trend entry signal.", None, {"regime": ctx.regime}, self.name)

        if ema_fast < ema_slow or adx_val < 18:
            return Decision(
                "EXIT",
                "Trend exit: cross-back or weak trend.",
                {"side": "sell", "type": "market", "size_base": ctx.deal.position_size},
                {"regime": ctx.regime},
                self.name,
            )
        return Decision("HOLD", "Trend intact.", None, {"regime": ctx.regime}, self.name)


class TrendFollowStrategy(StrategyBase):
    name = "trend_follow"

    def propose_orders(self, ctx: StrategyContext) -> Decision:
        closes_1h = [float(c[4]) for c in ctx.candles_1h] if ctx.candles_1h else []
        closes_4h = [float(c[4]) for c in ctx.candles_4h] if ctx.candles_4h else []
        if len(closes_1h) < 60 or len(closes_4h) < 60:
            return Decision("HOLD", "Not enough data for trend follow.", None, {"regime": ctx.regime}, self.name)

        ema_fast = ema(closes_1h, 12) or 0.0
        ema_slow = ema(closes_1h, 26) or 0.0
        slope_val = _slope(closes_1h, 20) or 0.0
        adx_val = adx(ctx.candles_1h, 14) or 0.0
        ema_fast_htf = ema(closes_4h, 12) or 0.0
        ema_slow_htf = ema(closes_4h, 26) or 0.0
        atr_val = _atr(ctx.candles_1h, 14) or 0.0

        trend_ok_htf = ema_fast_htf > ema_slow_htf
        trend_ok_ltf = ema_fast > ema_slow and slope_val > 0 and adx_val >= 18

        if ctx.deal.position_size <= 0:
            if trend_ok_htf and trend_ok_ltf:
                return Decision(
                    "ENTER",
                    "Trend follow entry: HTF/LTF aligned.",
                    {"side": "buy", "type": "market", "size_quote": float(ctx.cfg.get("base_quote", 0.0))},
                    {"regime": ctx.regime, "ema_fast": ema_fast, "ema_slow": ema_slow, "adx": adx_val},
                    self.name,
                )
            return Decision("HOLD", "Trend follow entry not confirmed.", None, {"regime": ctx.regime}, self.name)

        if ema_fast < ema_slow or slope_val < 0:
            return Decision(
                "EXIT",
                "Trend break detected.",
                {"side": "sell", "type": "market", "size_base": ctx.deal.position_size},
                {"regime": ctx.regime},
                self.name,
            )

        if atr_val > 0:
            trail_price = ctx.last_price - (atr_val * 1.5)
            if trail_price > 0:
                return Decision(
                    "TRAIL_TP_UPDATE",
                    "Trend trailing stop update.",
                    {"trail_price": trail_price},
                    {"regime": ctx.regime, "trail_price": trail_price},
                    self.name,
                )

        return Decision("HOLD", "Trend follow position held.", None, {"regime": ctx.regime}, self.name)


class RangeMeanReversionStrategy(StrategyBase):
    name = "range_mean_reversion"

    def propose_orders(self, ctx: StrategyContext) -> Decision:
        candles = ctx.candles_15m
        closes = [float(c[4]) for c in candles] if candles else []
        if len(closes) < 40:
            return Decision("HOLD", "Not enough data for range.", None, {"regime": ctx.regime}, self.name)
        bb = bollinger(closes, 20, 2.0)
        sma20 = sma(closes, 20)
        rsi_val = rsi(closes, 14) or 50.0
        scores = ctx.regime.scores or {}
        if scores.get("downtrend_score", 0.0) > 0.6 or scores.get("high_vol_score", 0.0) > 0.6:
            return Decision("PAUSE", "Range invalidated by trend/vol.", None, {"regime": ctx.regime}, self.name)

        if ctx.deal.position_size <= 0:
            if bb and bb[0] and ctx.last_price <= bb[0]:
                return Decision(
                    "ENTER",
                    "Range entry near lower band.",
                    {"side": "buy", "type": "market", "size_quote": float(ctx.cfg.get("base_quote", 0.0))},
                    {"regime": ctx.regime, "bb_low": bb[0]},
                    self.name,
                )
            if sma20 and rsi_val <= 35:
                return Decision(
                    "ENTER",
                    "Range entry on oversold/MA deviation.",
                    {"side": "buy", "type": "market", "size_quote": float(ctx.cfg.get("base_quote", 0.0))},
                    {"regime": ctx.regime, "rsi": rsi_val},
                    self.name,
                )
            return Decision("HOLD", "No range entry.", None, {"regime": ctx.regime}, self.name)

        tp_pct = max(0.003, float(ctx.cfg.get("tp", 0.01)) * 0.6)
        tp_price = float(ctx.deal.avg_entry or ctx.last_price) * (1.0 + tp_pct)
        if ctx.last_price >= tp_price:
            return Decision(
                "TAKE_PROFIT",
                "Range TP reached.",
                {"side": "sell", "type": "limit", "price": tp_price, "size_base": ctx.deal.position_size},
                {"regime": ctx.regime, "tp_price": tp_price},
                self.name,
            )

        return Decision("HOLD", "Range position held.", None, {"regime": ctx.regime}, self.name)


class HighVolDefensiveStrategy(StrategyBase):
    name = "high_vol_defensive"

    def propose_orders(self, ctx: StrategyContext) -> Decision:
        cfg = build_smart_dca_config(ctx, overrides={
            "max_safety": max(0, int(ctx.cfg.get("max_safety", 0) or 0) - 1),
            "base_quote_mult": 0.5,
            "safety_quote_mult": 0.4,
            "vol_gap_mult": 1.6,
            "min_gap_pct": 0.004,
            "max_gap_pct": 0.08,
        })
        return smart_decide(
            candles=ctx.candles_15m,
            last_price=ctx.last_price,
            cfg=cfg,
            deal=ctx.deal,
            account=ctx.account,
            perf=ctx.perf,
            now_ts=ctx.now_ts,
            cooldown_until=ctx.cooldown_until,
        )


class BreakoutStrategy(StrategyBase):
    name = "breakout"

    def propose_orders(self, ctx: StrategyContext) -> Decision:
        candles = ctx.candles_15m
        if len(candles) < 40:
            return Decision("HOLD", "Not enough data for breakout.", None, {"regime": ctx.regime}, self.name)
        don = donchian(candles, 20)
        bb = bollinger([float(c[4]) for c in candles], 20, 2.0)
        volr = volume_ratio(candles, 20) or 0.0
        last = ctx.last_price
        if not don or not bb:
            return Decision("HOLD", "Indicators unavailable.", None, {"regime": ctx.regime}, self.name)
        d_high, d_low = don
        _, _, _, bb_bw = bb

        if ctx.deal.position_size <= 0:
            if last >= d_high and bb_bw >= 0.07 and volr >= 1.3:
                return Decision(
                    "ENTER",
                    "Breakout up entry.",
                    {"side": "buy", "type": "market", "size_quote": float(ctx.cfg.get("base_quote", 0.0))},
                    {"regime": ctx.regime, "don_high": d_high, "volr": volr},
                    self.name,
                )
            if last <= d_low and bb_bw >= 0.07 and volr >= 1.3:
                return Decision(
                    "ENTER",
                    "Breakout down entry (mean reversion buy).",
                    {"side": "buy", "type": "market", "size_quote": float(ctx.cfg.get("base_quote", 0.0))},
                    {"regime": ctx.regime, "don_low": d_low, "volr": volr},
                    self.name,
                )
            return Decision("HOLD", "No breakout entry.", None, {"regime": ctx.regime}, self.name)

        atr_val = _atr(candles, 14) or 0.0
        if atr_val > 0 and last <= (ctx.deal.avg_entry or last) - (atr_val * 1.5):
            return Decision(
                "EXIT",
                "ATR stop hit.",
                {"side": "sell", "type": "market", "size_base": ctx.deal.position_size},
                {"regime": ctx.regime},
                self.name,
            )
        return Decision("HOLD", "Breakout position held.", None, {"regime": ctx.regime}, self.name)


try:
    from scalping_strategy import ScalpingStrategy
    _SCALPING_AVAILABLE = True
except ImportError:
    _SCALPING_AVAILABLE = False
    ScalpingStrategy = None

try:
    from long_term_strategies import BuyAndHoldWithGuardrails, SectorRotationStrategy
    _LONG_TERM_AVAILABLE = True
except ImportError:
    _LONG_TERM_AVAILABLE = False
    BuyAndHoldWithGuardrails = None
    SectorRotationStrategy = None

STRATEGY_REGISTRY: Dict[str, StrategyBase] = {
    "smart": SmartDcaStrategy(),
    "smart_dca": SmartDcaStrategy(),
    "classic_dca": ClassicDcaStrategy(),
    "grid": GridStrategy(),
    "trend": TrendStrategy(),
    "trend_follow": TrendFollowStrategy(),
    "range_mean_reversion": RangeMeanReversionStrategy(),
    "high_vol_defensive": HighVolDefensiveStrategy(),
    "breakout": BreakoutStrategy(),
}
if _SCALPING_AVAILABLE and ScalpingStrategy:
    STRATEGY_REGISTRY["scalping"] = ScalpingStrategy()
if _LONG_TERM_AVAILABLE and BuyAndHoldWithGuardrails:
    STRATEGY_REGISTRY["buy_and_hold"] = BuyAndHoldWithGuardrails()
if _LONG_TERM_AVAILABLE and SectorRotationStrategy:
    STRATEGY_REGISTRY["sector_rotation"] = SectorRotationStrategy()


def get_strategy(name: str) -> StrategyBase:
    return STRATEGY_REGISTRY.get(str(name or "").lower(), SmartDcaStrategy())


def dominant_regime(scores: Optional[Dict[str, float]]) -> str:
    if not scores:
        return "range"
    key = max(scores.items(), key=lambda kv: kv[1])[0]
    if key == "uptrend_score":
        return "uptrend"
    if key == "downtrend_score":
        return "downtrend"
    if key == "high_vol_score":
        return "high_vol"
    return "range"


@dataclass
class SelectorState:
    active: str = "smart_dca"
    last_switch_ts: int = 0


def select_strategy(
    regime: RegimeResult,
    current: str,
    last_switch_ts: int,
    now_ts: int,
    forced: Optional[str] = None,
    min_hold_sec: int = 300,
    min_conf: float = 0.55,
    vol_ratio: Optional[float] = None,
) -> Tuple[str, bool, str]:
    import os
    if forced and forced in STRATEGY_REGISTRY:
        return forced, False, "forced"
    router_v1 = os.getenv("ROUTER_V1", "1").strip().lower() in ("1", "true", "yes", "y", "on")
    if not router_v1:
        return current, False, "router_disabled"
    if now_ts - last_switch_ts < min_hold_sec:
        return current, False, "hold_window"
    if regime.confidence < min_conf:
        return current, False, "low_confidence"

    vr = vol_ratio if vol_ratio is not None else (regime.vol_ratio or 0.0)
    if vr >= 0.08 or regime.regime in ("RISK_OFF", "HIGH_VOL_RISK"):
        target = "high_vol_defensive"
        switched = target != current
        return target, switched, "volatility_override"
    if vr >= 0.05:
        override = "high_vol_defensive"
    else:
        override = None

    label = dominant_regime(regime.scores)
    target = current
    if label == "range":
        target = "range_mean_reversion"
    elif label == "uptrend":
        target = "trend_follow"
    elif label == "downtrend":
        target = "smart_dca"
    elif label == "high_vol":
        target = "high_vol_defensive"
    if regime.regime in ("RISK_OFF", "HIGH_VOL_RISK"):
        target = "high_vol_defensive"
    if override and override != target:
        target = override
        label = "high_vol"

    switched = target != current
    return target, switched, f"regime:{label}"
