"""
Explore signal engine — five named chart/strategy patterns + conviction scoring.

Used by the scan pipeline (live) and explore_backtest (historical).
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple

from strategies import bollinger, ema, rsi, sma


def _btc_bearish(btc_ctx: Optional[Dict[str, Any]]) -> bool:
    if not btc_ctx:
        return False
    r = str(
        btc_ctx.get("regime")
        or btc_ctx.get("regime_label")
        or (btc_ctx.get("labels") or {}).get("1d", "")
    ).upper()
    if r in ("BEAR", "STRONG_BEAR", "DOWNTREND", "WEAK_BEAR"):
        return True
    try:
        down = float(btc_ctx.get("btc_down") or 0)
        hv = float(btc_ctx.get("btc_hv") or 0)
        if down >= 0.6 and hv >= 0.5:
            return True
    except (TypeError, ValueError):
        pass
    return False

from explore_scorer import compute_conviction, status_from_conviction

logger = logging.getLogger(__name__)

REJECTION_REASONS = {
    "rsi_overbought": "RSI ≥ 80 (overbought)",
    "bear_regime": "Bear/Strong Bear regime",
    "low_volume": "Volume below minimum threshold",
    "score_below_watch": "Score below watch threshold",
    "stale_data": "Insufficient candle history",
    "spread_too_wide": "Bid-ask spread too wide",
    "btc_correlation_block": "BTC in freefall — altcoin blocked",
    "price_freefall": "Price dropping >6% — momentum negative",
    "no_pattern_match": "No qualifying chart pattern / strategy match",
    "unknown": "Rejected by quality gate",
}


def infer_rejection_code(reason: str) -> str:
    """Map free-text rejection to a stable machine code for DB/API."""
    t = (reason or "").lower()
    if "insufficient" in t or "history" in t or "candle" in t:
        return "stale_data"
    if "spread" in t:
        return "spread_too_wide"
    if "volume" in t and ("low" in t or "thin" in t or "below" in t):
        return "low_volume"
    if "btc" in t and ("freefall" in t or "block" in t or "correlation" in t):
        return "btc_correlation_block"
    if "bear" in t and "regime" in t:
        return "bear_regime"
    if "rsi" in t and ("80" in t or "overbought" in t):
        return "rsi_overbought"
    if "no qualifying" in t or "pattern" in t and "match" in t:
        return "no_pattern_match"
    if "6%" in t or "freefall" in t or "momentum" in t and "negative" in t:
        return "price_freefall"
    return "score_below_watch"


STRATEGY_LABELS = {
    "momentum_breakout": "Momentum Breakout",
    "pullback_support": "Pullback to Support",
    "oversold_reversal": "Oversold Reversal",
    "trend_continuation": "Trend Continuation",
    "crypto_momentum": "Crypto Momentum",
    "oversold_bounce": "Oversold Bounce",
    "volume_capitulation": "Volume Capitulation",
    "relative_strength_bear": "Relative Strength (Bear)",
    "oversold_extreme_fear": "Oversold + Extreme Fear",
}


def _closes_highs_lows_opens_vols(candles: List[List[float]]) -> Tuple[List[float], ...]:
    c, h, l, o, v = [], [], [], [], []
    for row in candles:
        if len(row) < 6:
            continue
        try:
            o.append(float(row[1]))
            h.append(float(row[2]))
            l.append(float(row[3]))
            c.append(float(row[4]))
            v.append(float(row[5]))
        except (TypeError, ValueError):
            continue
    return c, h, l, o, v


def volume_avg_and_ratio_from_candles(
    candles_1d: List[List[float]],
) -> Tuple[Optional[float], Optional[float]]:
    """
    Current bar volume vs mean of the prior 20 daily volumes (excludes today from the average).
    Returns (volume_ratio, avg_volume_prior20) or (None, None) if not computable.
    """
    _c, _h, _l, _o, v = _closes_highs_lows_opens_vols(candles_1d)
    if len(v) < 21:
        return None, None
    try:
        vc = float(v[-1])
        prev = [float(x) for x in v[-21:-1]]
    except (TypeError, ValueError):
        return None, None
    avg = sum(prev) / 20.0 if prev else 0.0
    if avg <= 0 or vc < 0:
        return None, None
    return vc / avg, avg


def _mean(xs: List[float]) -> float:
    if not xs:
        return 0.0
    return sum(xs) / len(xs)


def detect_strategies_at(
    candles: List[List[float]],
    end_idx: int,
    *,
    is_crypto: bool,
    btc_ctx: Optional[Dict[str, Any]] = None,
) -> List[Tuple[str, str, Dict[str, Any]]]:
    """
    Run all strategies as-of bar end_idx (inclusive).
    Returns list of (strategy_id, human_reason, facts).
    """
    if end_idx < 0 or end_idx >= len(candles):
        return []
    window = candles[: end_idx + 1]
    closes, highs, lows, opens, vols = _closes_highs_lows_opens_vols(window)
    if len(closes) < 60:
        return []
    i = len(closes) - 1
    c = closes[i]
    h = highs[i]
    l = lows[i]
    o = opens[i] if i < len(opens) else c
    vc = vols[i] if i < len(vols) else 0.0

    matches: List[Tuple[str, str, Dict[str, Any]]] = []

    # --- 1 Momentum Breakout (stocks + crypto) ---
    if i >= 21:
        resist = max(highs[i - 20 : i])
        if resist > 0 and c > resist * 1.001:
            prev_vol_avg = _mean(vols[i - 20 : i]) if i >= 20 else 0.0
            vol_mult = (vc / prev_vol_avg) if prev_vol_avg > 0 else 0.0
            if vol_mult >= 1.5:
                clearance = (c - resist) / resist * 100.0
                reason = (
                    f"Breaking out above ${resist:.2f} resistance on {vol_mult:.1f}x avg volume"
                )
                matches.append(
                    (
                        "momentum_breakout",
                        reason,
                        {
                            "volume_mult": vol_mult,
                            "trend_aligned": c > (sma(closes, 20) or c),
                            "near_key_level": True,
                            "breakout_clearance_pct": clearance / 100.0,
                        },
                    )
                )

    # --- 2 Pullback to Support ---
    if i >= 55:
        s50_now = sma(closes, 50)
        s50_then = sma(closes[:-5], 50) if len(closes) > 55 else None
        rising_50 = bool(s50_now and s50_then and s50_now > s50_then * 1.002)
        hi_20 = max(highs[max(0, i - 19) : i + 1])
        if hi_20 > 0 and s50_now:
            pull = (hi_20 - c) / hi_20
            near_ma = abs(c - s50_now) / c < 0.035 if c else False
            touched_ma = l <= s50_now * 1.015 and c >= s50_now * 0.97
            if rising_50 and 0.08 <= pull <= 0.16 and (near_ma or touched_ma):
                reason = (
                    f"Pulled back {pull*100:.0f}% to 50MA support, uptrend intact"
                )
                matches.append(
                    (
                        "pullback_support",
                        reason,
                        {
                            "volume_mult": (vc / _mean(vols[max(0, i - 19) : i])) if i >= 1 else 1.0,
                            "trend_aligned": True,
                            "near_key_level": near_ma or touched_ma,
                            "pullback_pct": pull,
                        },
                    )
                )

    # --- 3 Oversold Reversal ---
    rsi_v = rsi(closes, 14)
    bb = bollinger(closes, 20, 2.0)
    if rsi_v is not None and bb is not None:
        lo, mid, up, _bw = bb
        band_w = up - lo
        near_lower = band_w > 0 and c <= lo + 0.18 * band_w
        if rsi_v < 35 and near_lower and i >= 1:
            bull = c > closes[i - 1] and c > o
            if bull:
                reason = (
                    f"RSI {rsi_v:.0f}, bouncing off lower BB, early reversal signal"
                )
                short_up = i >= 5 and c > closes[i - 5]
                matches.append(
                    (
                        "oversold_reversal",
                        reason,
                        {
                            "volume_mult": (vc / max(1e-9, _mean(vols[max(0, i - 9) : i]))) if i >= 1 else 1.0,
                            "trend_aligned": short_up,
                            "near_key_level": True,
                            "rsi": rsi_v,
                        },
                    )
                )

    # --- 4 Trend Continuation (strict weekly-style + relaxed + crypto variant) ---
    btc_bear = _btc_bearish(btc_ctx)
    if i >= 200:
        s50 = sma(closes, 50)
        s200 = sma(closes, 200)
        if s50 and s200 and c > s50 > s200 and s50 > s200 * 1.001:
            sub_h = highs[max(1, i - 7) : i]
            sub_l = lows[max(1, i - 7) : i]
            if sub_h and sub_l:
                rng = max(sub_h) - min(sub_l)
                tight = (rng / c) < 0.08 if c else False
                prior_high = max(highs[max(0, i - 10) : i]) if i >= 1 else h
                breaking = c > prior_high * 1.0005
                if tight and breaking:
                    days_rng = min(10, max(5, len(sub_h)))
                    reason = (
                        f"Uptrend confirmed, {days_rng}-day consolidation breaking upward"
                    )
                    matches.append(
                        (
                            "trend_continuation",
                            reason,
                            {
                                "volume_mult": (vc / _mean(vols[i - 10 : i])) if i >= 10 else 1.0,
                                "trend_aligned": True,
                                "near_key_level": False,
                                "consolidation_tight": tight,
                            },
                        )
                    )

    if i >= 55:
        s50b = sma(closes, 50)
        e20b = ema(closes, 20)
        e50b = ema(closes, 50)
        rsi_rel = rsi(closes, 14)
        if s50b and e20b and e50b and c > s50b and e20b > e50b and rsi_rel is not None and 45.0 <= rsi_rel <= 70.0:
            matches.append(
                (
                    "trend_continuation",
                    f"Uptrend continuation: above 50 SMA, EMA20>EMA50, RSI {rsi_rel:.0f}",
                    {
                        "volume_mult": (vc / _mean(vols[max(0, i - 10) : i])) if i >= 1 else 1.0,
                        "trend_aligned": True,
                        "near_key_level": False,
                        "rsi": rsi_rel,
                    },
                )
            )

    if is_crypto and i >= 55 and not btc_bear:
        e50c = ema(closes, 50)
        avg14 = _mean(vols[max(0, i - 14) : i]) if i >= 15 else 0.0
        if e50c and c > e50c and avg14 > 0 and vc >= 1.5 * avg14:
            matches.append(
                (
                    "trend_continuation",
                    "Crypto uptrend: price above 50 EMA with elevated volume; BTC not in bear regime",
                    {
                        "volume_mult": vc / avg14,
                        "trend_aligned": True,
                        "near_key_level": False,
                    },
                )
            )

    # --- 5 Crypto Momentum ---
    if is_crypto and i >= 25:
        avg7 = _mean(vols[max(0, i - 7) : i])
        s20 = sma(closes, 20)
        rsi_v2 = rsi(closes, 14)
        if avg7 > 0 and vc >= 2.0 * avg7 and s20 and c > s20 and rsi_v2 is not None:
            if 50.0 <= rsi_v2 <= 70.0:
                reason = (
                    f"Volume surge {vc/avg7:.1f}x average, price momentum building"
                )
                matches.append(
                    (
                        "crypto_momentum",
                        reason,
                        {
                            "volume_mult": vc / avg7,
                            "trend_aligned": True,
                            "near_key_level": False,
                            "rsi": rsi_v2,
                        },
                    )
                )

    # --- 6 Oversold Bounce (Range/Bear optimized — proven 60% WR in 30d backtest) ---
    # Entry: RSI 22–35, price near a support level, above 200MA (or within 8% for crypto),
    # volume ≥ 0.8× 20-day avg (not a dead/thinly-traded market).
    if i >= 55:
        rsi_ob = rsi(closes, 14)
        s50_ob = sma(closes, 50)
        s200_ob = sma(closes, 200) if i >= 200 else None
        avg_vol_20_ob = _mean(vols[max(0, i - 20) : i]) if i >= 20 else 0.0
        _vol_ok = avg_vol_20_ob > 0 and vc >= 0.8 * avg_vol_20_ob

        if rsi_ob is not None and 22.0 <= rsi_ob <= 35.0 and _vol_ok:
            # Price near a known support level (within 3%)
            swing_low_20 = min(lows[max(0, i - 20) : i + 1]) if i >= 1 else c
            _supports = [x for x in [s50_ob, s200_ob, swing_low_20] if x and x > 0]
            near_support_ob = any(c > 0 and abs(c - sup) / c < 0.035 for sup in _supports)

            # Not in a hard downtrend: above 200MA or within tolerated distance
            _max_below = 0.08 if is_crypto else 0.05
            if s200_ob and s200_ob > 0:
                _above_200 = c >= s200_ob * (1.0 - _max_below)
            elif s50_ob and s50_ob > 0:
                _above_200 = c >= s50_ob * 0.92
            else:
                _above_200 = True

            if near_support_ob and _above_200:
                _vol_mult_ob = vc / avg_vol_20_ob if avg_vol_20_ob > 0 else 1.0
                matches.append((
                    "oversold_bounce",
                    f"RSI {rsi_ob:.0f} near support — oversold bounce setup (range/bear optimized)",
                    {
                        "volume_mult": _vol_mult_ob,
                        "trend_aligned": False,
                        "near_key_level": near_support_ob,
                        "rsi": rsi_ob,
                        "near_support": near_support_ob,
                    },
                ))

    # --- 7 Volume Capitulation (selling climax + reversal bar; multi-day bearish context)
    if i >= 21:
        vol_20_avg = _mean(vols[-21:-1]) if len(vols) >= 21 else _mean(vols[:-1] if len(vols) > 1 else vols)
        price_drop_5 = (c - closes[max(0, i - 5)]) / closes[max(0, i - 5)] * 100 if i >= 5 and closes[max(0, i - 5)] > 0 else 0.0
        rsi_last = float(rsi(closes, 14) or 50.0)
        candle_body_pct = abs(c - o) / (h - l + 1e-10)
        prior_bear = False
        if i >= 10 and closes[max(0, i - 10)] > 0:
            prior_bear = c < closes[max(0, i - 10)] * 0.98
        if (
            vol_20_avg > 0
            and vc > 3 * vol_20_avg
            and price_drop_5 < -8
            and rsi_last < 28
            and candle_body_pct > 0.6
            and prior_bear
        ):
            matches.append(
                (
                    "volume_capitulation",
                    f"Volume capitulation: {vc/vol_20_avg:.1f}x avg volume, RSI {rsi_last:.0f}, "
                    f"price {price_drop_5:.1f}% over 5 bars",
                    {
                        "volume_ratio": vc / vol_20_avg,
                        "rsi": rsi_last,
                        "price_drop_5": price_drop_5,
                        "candle_body_pct": candle_body_pct,
                    },
                )
            )

    # --- 8 Relative Strength Bear ---
    # Fires when BTC is in a down regime AND this asset's 20d return
    # beats BTC's 20d return by >= 5 percentage points AND RSI < 70.
    if i >= 22 and is_crypto:
        _rs_regime = str((btc_ctx or {}).get("regime") or "").upper()
        _btc_is_down = _rs_regime in ("BEAR", "STRONG_BEAR", "DOWNTREND", "WEAK_BEAR")
        if _btc_is_down:
            btc_closes = list((btc_ctx or {}).get("closes") or [])
            if len(btc_closes) >= 21 and btc_closes[-21] > 0:
                btc_ret_20 = (btc_closes[-1] - btc_closes[-21]) / btc_closes[-21] * 100
                asset_ret_20 = (c - closes[i - 20]) / closes[i - 20] * 100 if closes[i - 20] > 0 else 0
                outperf = asset_ret_20 - btc_ret_20
                _rsi_rs = rsi(closes, 14)
                if outperf >= 5.0 and (_rsi_rs is None or _rsi_rs < 70):
                    avg_vol_rs = _mean(vols[max(0, i - 20):i]) if i >= 1 else 0.0
                    matches.append((
                        "relative_strength_bear",
                        f"Outperforming BTC by {outperf:.1f}% over 20 days in bear market — relative strength winner",
                        {
                            "volume_mult": (vc / avg_vol_rs) if avg_vol_rs > 0 else 1.0,
                            "trend_aligned": True,
                            "near_key_level": False,
                            "outperformance_pct": outperf,
                            "asset_ret_20d": asset_ret_20,
                            "btc_ret_20d": btc_ret_20,
                        },
                    ))

    # --- 9 Oversold + Extreme Fear ---
    # Fires when Fear & Greed <= 20 AND RSI <= 30.
    # Works for both crypto and stocks — contrarian reversal in peak panic.
    if i >= 14:
        _ef_fg = int((btc_ctx or {}).get("fear_greed_value") or (btc_ctx or {}).get("fear_greed") or 100)
        _rsi_ef = rsi(closes, 14)
        if _ef_fg <= 20 and _rsi_ef is not None and _rsi_ef <= 30:
            avg_vol_ef = _mean(vols[max(0, i - 14):i]) if i >= 1 else 0.0
            _not_crashing = (c >= o * 0.97) if o > 0 else True
            if _not_crashing:
                matches.append((
                    "oversold_extreme_fear",
                    f"RSI {_rsi_ef:.0f} + Fear & Greed {_ef_fg} (Extreme Fear) — historically strong reversal zone",
                    {
                        "volume_mult": (vc / avg_vol_ef) if avg_vol_ef > 0 else 1.0,
                        "trend_aligned": False,
                        "near_key_level": False,
                        "rsi": _rsi_ef,
                        "fear_greed": _ef_fg,
                        "vol_spike": avg_vol_ef > 0 and vc >= 1.5 * avg_vol_ef,
                    },
                ))

    return matches


def _pick_best_match(
    matches: List[Tuple[str, str, Dict[str, Any]]],
    *,
    is_crypto: bool,
    sector_etf_ok: Optional[bool],
    market_cap_tier: Optional[str],
) -> Optional[Tuple[str, str, Dict[str, Any], int, str]]:
    if not matches:
        return None
    scored: List[Tuple[int, int, str, str, Dict[str, Any], str]] = []
    for sid, reason, facts in matches:
        conv = compute_conviction(
            sid,
            facts,
            is_crypto=is_crypto,
            sector_etf_ok=sector_etf_ok,
            market_cap_tier=market_cap_tier,
        )
        st = status_from_conviction(conv, sid)
        # Sort: highest conviction first; on tie, prefer crypto_momentum for crypto pairs.
        pref = 0 if (is_crypto and sid == "crypto_momentum") else 1
        scored.append((conv, pref, sid, reason, facts, st))
    scored.sort(key=lambda t: (-t[0], t[1]))
    conv, _pref, sid, reason, facts, st = scored[0]
    return (sid, reason, facts, conv, st)


def evaluate_explore(
    symbol: str,
    candles_1d: List[List[float]],
    *,
    is_crypto: bool,
    horizon: str,
    market_breadth: Optional[Dict[str, Any]] = None,
    btc_context: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Full evaluation for one symbol at the latest candle.
    Returns dict: status, strategy, strategy_label, reason, conviction, signal_ts, detail
    """
    mb = market_breadth or {}
    sector_ok = mb.get("sector_etf_ok")
    if sector_ok is None:
        sector_ok = True
    cap_tier = mb.get("market_cap_tier") or mb.get("tier")

    if not candles_1d or len(candles_1d) < 60:
        return {
            "status": "rejected",
            "strategy": "",
            "strategy_label": "",
            "reason": "Insufficient daily history for pattern detection",
            "conviction": 0,
            "signal_ts": int(candles_1d[-1][0]) if candles_1d else 0,
            "detail": {"symbol": symbol, "horizon": horizon},
        }

    end_idx = len(candles_1d) - 1
    matches = detect_strategies_at(
        candles_1d, end_idx, is_crypto=is_crypto, btc_ctx=btc_context
    )
    picked = _pick_best_match(
        matches,
        is_crypto=is_crypto,
        sector_etf_ok=bool(sector_ok) if sector_ok is not None else None,
        market_cap_tier=str(cap_tier) if cap_tier else None,
    )

    ts = int(float(candles_1d[end_idx][0]))

    if not picked:
        return {
            "status": "rejected",
            "strategy": "",
            "strategy_label": "",
            "reason": "No qualifying chart pattern / strategy match",
            "conviction": 0,
            "signal_ts": ts,
            "detail": {
                "symbol": symbol,
                "horizon": horizon,
                "candidates": len(matches),
            },
        }

    sid, reason, facts, conv, st = picked
    # Horizon tilt: longer horizons slightly easier to qualify as buy
    h = (horizon or "short").lower()
    adj = 0
    if h.startswith("m"):
        adj = 2
    elif h.startswith("l"):
        adj = 4
    conv2 = max(0, min(100, conv + adj))
    st2 = status_from_conviction(conv2, sid)

    detail = {
        "symbol": symbol,
        "horizon": horizon,
        "strategy_id": sid,
        "facts": facts,
        "candidates": [{"id": m[0], "reason": m[1]} for m in matches],
    }

    return {
        "status": st2,
        "strategy": sid,
        "strategy_label": STRATEGY_LABELS.get(sid, sid),
        "reason": reason,
        "conviction": conv2,
        "signal_ts": ts,
        "detail": detail,
    }


# ---------------------------------------------------------------------------
# Hybrid-style evaluate_signal (recommendations / gates / human strategy reason)
# ---------------------------------------------------------------------------


def _closes_from_candles(candles: List[List[float]]) -> List[float]:
    out: List[float] = []
    for row in candles or []:
        if len(row) >= 5:
            try:
                out.append(float(row[4]))
            except (TypeError, ValueError):
                pass
    return out


def _highs_from_candles(candles: List[List[float]]) -> List[float]:
    out: List[float] = []
    for row in candles or []:
        if len(row) >= 5:
            try:
                out.append(float(row[2]))
            except (TypeError, ValueError):
                pass
    return out


def _lows_from_candles(candles: List[List[float]]) -> List[float]:
    out: List[float] = []
    for row in candles or []:
        if len(row) >= 5:
            try:
                out.append(float(row[3]))
            except (TypeError, ValueError):
                pass
    return out


def _stoch_k_last(highs: List[float], lows: List[float], closes: List[float], n: int = 14) -> Optional[float]:
    if len(closes) < n or len(highs) < n or len(lows) < n:
        return None
    hh = max(highs[-n:])
    ll = min(lows[-n:])
    c = closes[-1]
    if hh <= ll:
        return 50.0
    return 100.0 * (c - ll) / (hh - ll)


def apply_fundamental_gates(
    fundamentals: Dict[str, Any],
    market_breadth: Dict[str, Any],
) -> Tuple[bool, Optional[str], Dict[str, Any]]:
    """
    Gates 11 / 16 / 17 — loosened per product spec. Returns (passed, reject_reason, flags).
    """
    flags: Dict[str, Any] = {}
    mb = market_breadth or {}
    fd = fundamentals or {}
    sector = str(fd.get("sector") or mb.get("sector") or "").upper()
    is_tech_health = (
        "TECHNOLOGY" in sector
        or "HEALTHCARE" in sector
        or sector == "TECHNOLOGY"
        or sector == "HEALTHCARE"
        or "TECH" in sector
        or "HEALTH" in sector
    )

    pe = fd.get("pe_ratio", fd.get("pe"))
    if pe is not None:
        try:
            pe_f = float(pe)
            if pe_f < 0:
                return False, "negative_pe", flags
            cap = 100.0 if is_tech_health else 80.0
            if pe_f > cap:
                return False, f"pe_too_high_{pe_f:.1f}", flags
        except (TypeError, ValueError):
            pass

    peg = fd.get("peg_ratio", fd.get("peg"))
    if peg is not None:
        try:
            peg_f = float(peg)
            if is_tech_health and peg_f > 3.0:
                flags["elevated_peg"] = True
            if peg_f > 5.0:
                return False, f"peg_too_high_{peg_f:.2f}", flags
        except (TypeError, ValueError):
            pass

    pos_q_raw = fd.get("positive_cf_quarters", fd.get("positive_cashflow_quarters"))
    pos_q = int(pos_q_raw) if pos_q_raw is not None and str(pos_q_raw).strip() != "" else 0
    fcf = fd.get("free_cashflow", fd.get("freeCashflow"))
    rev_g = fd.get("revenue_growth", fd.get("revenue_growth_yoy"))

    has_cf_data = any(x is not None for x in (pos_q_raw, fcf, rev_g))
    ok_a = pos_q >= 2
    ok_b = False
    ok_c = False
    try:
        ok_b = fcf is not None and float(fcf) > 0
    except (TypeError, ValueError):
        pass
    try:
        ok_c = rev_g is not None and float(rev_g) > 0
    except (TypeError, ValueError):
        pass
    if has_cf_data and not (ok_a or ok_b or ok_c):
        return False, "operating_cashflow_gate", flags

    return True, None, flags


def detect_strategy(
    technical_signals: Dict[str, Any],
    factor_scores: Dict[str, Any],
    fundamentals: Dict[str, Any],
    market_breadth: Dict[str, Any],
) -> str:
    """
    Priority-ordered strategy label for recommendations (human-facing names).
    """
    ts = dict(technical_signals or {})
    fs = dict(factor_scores or {})
    mb = dict(market_breadth or {})
    fun = dict(fundamentals or {})

    def _f(key: str, default: float = 0.0) -> float:
        v = fs.get(key)
        try:
            return float(v) if v is not None else default
        except (TypeError, ValueError):
            return default

    mom = _f("momentum", 0.0)
    val = _f("value", 0.0)
    qual = _f("quality", 0.0)
    sent = _f("sentiment", 0.0)

    p52 = float(ts.get("price_to_52wk_high") or 0.0)
    vol_ratio = float(ts.get("volume_ratio") or 0.0)
    near_52 = bool(ts.get("near_52wk_high"))
    stoch_k = float(ts.get("stoch_k") or 0.0)
    ema_bull = bool(ts.get("ema_9_27_bullish"))
    rsi2 = float(ts.get("rsi_2") or 999.0)
    rsi14 = float(ts.get("rsi_14") or 50.0)
    rsi2_entry = bool(ts.get("rsi2_entry_signal"))
    macd_combo = bool(ts.get("macd_combo_entry"))
    above_200 = bool(ts.get("above_200sma"))
    cardwell = str(ts.get("cardwell_regime") or "").upper()
    regime = str(ts.get("regime_label") or mb.get("regime") or "").upper()
    stoch_raw = ts.get("stoch_k")

    # 1 Momentum Breakout (highest priority)
    if (p52 > 0.92 and mom > 65 and vol_ratio > 1.3) or (
        near_52 and stoch_k > 70 and ema_bull
    ):
        return "Momentum Breakout"

    # 2 Oversold Reversal
    if rsi2 < 15 and rsi2_entry and above_200:
        return "Oversold Reversal"

    # 3 Mean Reversion
    if macd_combo and 15.0 <= rsi2 <= 40.0:
        return "Mean Reversion"

    # 4 Pullback to Support
    uptrend_card = cardwell in ("UPTREND", "STRONG_BULL", "BULL")
    try:
        sk_ok = stoch_raw is not None and float(stoch_raw) < 40.0
    except (TypeError, ValueError):
        sk_ok = False
    if above_200 and uptrend_card and 35.0 <= rsi14 <= 55.0 and sk_ok:
        return "Pullback to Support"

    # 5 Earnings Play
    ed = mb.get("earnings_days", fun.get("earnings_days"))
    try:
        ed_i = int(float(ed)) if ed is not None else 999
    except (TypeError, ValueError):
        ed_i = 999
    if 0 <= ed_i <= 14 and sent > 55:
        return "Earnings Play"

    # 6 Value Compounder
    if val > 70 and qual > 70 and mom > 30:
        return "Value Compounder"

    # 7 Defensive
    beta = float(ts.get("beta") or mb.get("beta") or fun.get("beta") or 1.0)
    if beta < 0.5 and qual > 60:
        return "Defensive"

    # 8 Trend Follow
    uptrend_reg = regime in ("BULL", "STRONG_BULL", "TREND_UP", "WEAK_BULL", "BREAKOUT")
    if above_200 and uptrend_reg and mom > 50:
        return "Trend Follow"

    return "Trend Follow"


def build_strategy_reason(
    strategy: str,
    technical_signals: Dict[str, Any],
    factor_scores: Dict[str, Any],
    fundamentals: Dict[str, Any],
    market_breadth: Dict[str, Any],
) -> str:
    """One-sentence explanation with numbers for the UI."""
    ts = dict(technical_signals or {})
    fs = dict(factor_scores or {})
    mb = dict(market_breadth or {})
    fun = dict(fundamentals or {})

    def _f(key: str, default: float = 0.0) -> float:
        v = fs.get(key)
        try:
            return float(v) if v is not None else default
        except (TypeError, ValueError):
            return default

    if strategy == "Momentum Breakout":
        p52 = float(ts.get("price_to_52wk_high") or 0.0) * 100.0
        vr = float(ts.get("volume_ratio") or 0.0)
        return f"Within {p52:.0f}% of 52-week high, volume {vr:.1f}x average, momentum {_f('momentum'):.0f}."
    if strategy == "Mean Reversion":
        r2 = float(ts.get("rsi_2") or 0.0)
        return f"RSI(2) {r2:.0f} oversold pullback, MACD combo confirming reversal."
    if strategy == "Trend Follow":
        pct = float(ts.get("pct_above_200") or 0.0)
        if pct <= 0 and ts.get("above_200sma"):
            pct = 3.0
        if pct < 0:
            return f"BELOW 200-day MA by {abs(pct):.0f}%, factor momentum {_f('momentum'):.0f}."
        return f"Above 200-day MA (~{pct:.0f}% above MA), factor momentum {_f('momentum'):.0f}."
    if strategy == "Value Compounder":
        pe = float(fun.get("pe_ratio") or fun.get("pe") or 0.0)
        pe_s = f"{pe:.1f}" if pe > 0 else "n/a"
        return f"Value {_f('value'):.0f} + Quality {_f('quality'):.0f} above 70, P/E {pe_s}."
    if strategy == "Earnings Play":
        ed = mb.get("earnings_days", fun.get("earnings_days", "?"))
        return f"Earnings in {ed}d, analyst sentiment {_f('sentiment'):.0f}."
    if strategy == "Defensive":
        b = float(ts.get("beta") or mb.get("beta") or 0.0)
        return f"Beta {b:.2f}, quality {_f('quality'):.0f}, low-volatility setup."
    if strategy == "Oversold Reversal":
        r2 = float(ts.get("rsi_2") or 0.0)
        return f"RSI(2) {r2:.0f} extreme pullback with RSI2 entry, still above 200MA (uptrend)."
    if strategy == "Pullback to Support":
        r14 = float(ts.get("rsi_14") or 0.0)
        sk = float(ts.get("stoch_k") or 0.0)
        return f"Uptrend pullback: RSI(14) {r14:.0f}, Stoch {sk:.0f} — support zone."
    return f"{strategy}: momentum {_f('momentum'):.0f}, volume {float(ts.get('volume_ratio') or 0):.1f}x."


def evaluate_signal(
    symbol: str,
    asset_type: str,
    price: float,
    candles_1d: List[List[float]],
    market_breadth: Dict[str, Any],
    volume_24h: float,
    fear_greed: int,
    btc_context: Dict[str, Any],
    existing_score: float,
    metrics: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Returns signal 'reject' | 'accept', detected_strategy, strategy_reason, flags.
    Integrates Fear & Greed, BTC context, and volume into scoring.
    """
    metrics = dict(metrics or {})
    _fg = int(fear_greed or 50)
    _btc = dict(btc_context or {})
    mb = dict(market_breadth or {})

    closes = _closes_from_candles(candles_1d)
    highs = _highs_from_candles(candles_1d)
    lows = _lows_from_candles(candles_1d)

    technical: Dict[str, Any] = {}
    for k in (
        "rsi_2",
        "price_to_52wk_high",
        "macd_combo_entry",
        "rsi2_entry_signal",
        "cardwell_regime",
        "volume_ratio",
        "beta",
    ):
        if metrics.get(k) is not None:
            technical[k] = metrics[k]

    if closes:
        if technical.get("rsi_2") is None and len(closes) >= 3:
            r2 = rsi(closes, 2)
            if r2 is not None:
                technical["rsi_2"] = r2
        if technical.get("rsi_14") is None:
            r14 = rsi(closes, 14)
            if r14 is not None:
                technical["rsi_14"] = r14
        if len(closes) >= 200:
            s200 = sma(closes, 200)
            if s200:
                technical["above_200sma"] = closes[-1] > s200
                technical["pct_above_200"] = (closes[-1] - s200) / s200 * 100.0
        if len(highs) >= 252 and closes:
            hh = max(highs[-252:])
            if hh > 0:
                technical["price_to_52wk_high"] = closes[-1] / hh
                technical["near_52wk_high"] = closes[-1] / hh > 0.92
        sk = _stoch_k_last(highs, lows, closes, 14)
        if sk is not None:
            technical["stoch_k"] = sk
        if len(closes) >= 30:
            e9 = ema(closes, 9)
            e27 = ema(closes, 27)
            if e9 is not None and e27 is not None:
                technical["ema_9_27_bullish"] = e9 > e27
    rg = metrics.get("regime") or (metrics.get("regime_label"))
    if rg:
        technical["regime_label"] = str(rg).upper()

    factor_scores = metrics.get("factor_scores") or {}
    if not isinstance(factor_scores, dict):
        factor_scores = {}

    fundamentals: Dict[str, Any] = {
        "pe_ratio": metrics.get("pe_ratio") or metrics.get("trailing_pe"),
        "peg_ratio": metrics.get("peg_ratio") or metrics.get("peg"),
        "positive_cf_quarters": metrics.get("positive_cf_quarters")
        or metrics.get("positive_cashflow_quarters"),
        "free_cashflow": metrics.get("free_cashflow") or metrics.get("freeCashflow"),
        "revenue_growth": metrics.get("revenue_growth") or metrics.get("revenue_growth_yoy"),
        "sector": metrics.get("sector") or mb.get("sector"),
        "earnings_days": mb.get("earnings_days") or metrics.get("earnings_days"),
    }

    flags: Dict[str, Any] = {}
    if str(asset_type).lower() in ("crypto", "digital", "cryptocurrency"):
        passed_gates = True
        reject_reason = None
    else:
        passed_gates, reject_reason, fflags = apply_fundamental_gates(fundamentals, mb)
        flags.update(fflags)

    if not passed_gates:
        return {
            "signal": "reject",
            "rejection_reason": reject_reason or "fundamental_gate",
            "passed_gates": False,
            "detected_strategy": "",
            "strategy_reason": "",
            "flags": flags,
        }

    strat = detect_strategy(technical, factor_scores, fundamentals, mb)
    reason = build_strategy_reason(strat, technical, factor_scores, fundamentals, mb)

    # Apply macro context: extreme fear + crypto = flag risk
    is_crypto = str(asset_type).lower() in ("crypto", "digital", "cryptocurrency")
    if _fg <= 12 and is_crypto:
        flags["extreme_fear_crypto"] = True
        flags["macro_warning"] = f"Extreme Fear ({_fg}) — crypto entries very risky"
    elif _fg <= 25:
        flags["fear_environment"] = True

    _btc_regime = str(_btc.get("regime") or _btc.get("regime_label") or "").upper()
    if is_crypto and _btc_regime in ("BEAR", "STRONG_BEAR"):
        flags["btc_downtrend"] = True
        flags["macro_warning"] = flags.get("macro_warning", "") + f" BTC regime: {_btc_regime}"

    return {
        "signal": "accept",
        "rejection_reason": None,
        "passed_gates": True,
        "detected_strategy": strat,
        "strategy_reason": reason,
        "flags": flags,
    }
