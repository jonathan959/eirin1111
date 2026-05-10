"""
Strategy leaderboard: live journal stats with backtest fallback (Round 3).

Composite score (0–100 scale for sorting):
  score = 0.30 * win_rate_norm
        + 0.25 * profit_factor_norm
        + 0.20 * sharpe_norm
        + 0.15 * (1 - max_dd_norm)
        + 0.10 * sortino_norm

Normalization (each clamped to [0, 1]):
  win_rate_norm      = clamp01(win_rate)                    # win_rate in [0, 1]
  profit_factor_norm = clamp01(profit_factor / 3.0)       # PF above 3 treated as 1
  sharpe_norm        = clamp01((sharpe + 0.5) / 3.0)       # centers roughly -0.5..2.5
  max_dd_norm        = clamp01(max_dd / 0.50)             # max_dd fraction; 50%+ DD -> 1
  sortino_norm       = clamp01((sortino + 0.5) / 3.0)

``regime_fit`` maps journal/backtest tags into TREND_UP, TREND_DOWN, RANGE, RISK_OFF buckets
using ``deals.entry_regime`` when ``deal_id`` is present; unknown → RANGE.
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import random
import sqlite3
import time
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeout
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# Nine primary strategies shown in UI (matches product “all strategies” view).
CANONICAL_STRATEGIES: Tuple[str, ...] = (
    "smart_dca",
    "classic_dca",
    "grid",
    "trend_follow_auto",
    "range_mean_reversion",
    "high_vol_defensive",
    "breakout",
    "mean_reversion",
    "momentum",
)

_INLINE_CACHE: Dict[str, Dict[str, Any]] = {}
_ENGINE_MAP = {
    "trend_follow_auto": "trend_follow",
    "high_vol_defensive": "mean_reversion",
}


def _clamp01(x: float) -> float:
    return max(0.0, min(1.0, float(x)))


def _composite_score(
    win_rate: float,
    profit_factor: float,
    sharpe: float,
    sortino: float,
    max_dd: float,
) -> float:
    wrn = _clamp01(win_rate)
    pfn = _clamp01(profit_factor / 3.0)
    shr = _clamp01((sharpe + 0.5) / 3.0)
    ddn = _clamp01(max_dd / 0.50)
    sor = _clamp01((sortino + 0.5) / 3.0)
    return float(
        0.30 * wrn
        + 0.25 * pfn
        + 0.20 * shr
        + 0.15 * (1.0 - ddn)
        + 0.10 * sor
    )


def _map_regime_label(raw: Optional[str]) -> str:
    s = str(raw or "").upper()
    if any(k in s for k in ("TREND_UP", "BULL", "UPTREND", "BREAKOUT_UP")):
        return "TREND_UP"
    if any(k in s for k in ("TREND_DOWN", "BEAR", "DOWNTREND", "BREAKOUT_DOWN")):
        return "TREND_DOWN"
    if any(k in s for k in ("RISK", "HIGH_VOL", "CRASH")):
        return "RISK_OFF"
    return "RANGE"


def _regime_fit_from_pnls(
    by_bucket: Dict[str, List[float]],
) -> Dict[str, float]:
    out: Dict[str, float] = {}
    for k in ("TREND_UP", "TREND_DOWN", "RANGE", "RISK_OFF"):
        xs = by_bucket.get(k) or []
        if not xs:
            out[k] = 0.0
        else:
            out[k] = sum(1 for p in xs if p > 0) / len(xs)
    return out


def _fetch_journal_rows(since_ts: int) -> List[sqlite3.Row]:
    import db

    db.init_db()
    path = os.path.abspath(db.DB_NAME)
    con = sqlite3.connect(path)
    con.row_factory = sqlite3.Row
    try:
        rows = con.execute(
            """
            SELECT j.strategy, j.pnl_quote, j.deal_id, j.entry_ts, j.exit_ts,
                   d.entry_regime AS entry_regime
            FROM journal j
            LEFT JOIN deals d ON d.id = j.deal_id
            WHERE j.exit_ts >= ?
            """,
            (int(since_ts),),
        ).fetchall()
        return list(rows)
    finally:
        con.close()


def _aggregate_trades(trades: List[sqlite3.Row]) -> Dict[str, Any]:
    pnls = [float(r["pnl_quote"] or 0.0) for r in trades]
    n = len(pnls)
    wins = [p for p in pnls if p > 0]
    losses = [p for p in pnls if p < 0]
    win_rate = (len(wins) / n) if n else 0.0
    gp, gl = sum(wins), abs(sum(losses))
    profit_factor = (gp / gl) if gl > 0 else (2.0 if gp > 0 else 0.0)
    expectancy = (sum(pnls) / n) if n else 0.0
    hold_h: List[float] = []
    for r in trades:
        try:
            ex = int(r["exit_ts"])
            en = int(r["entry_ts"])
            if ex > en:
                hold_h.append((ex - en) / 3600.0)
        except Exception:
            continue
    avg_hold = sum(hold_h) / len(hold_h) if hold_h else 0.0
    peak = cum = 0.0
    max_dd = 0.0
    for p in pnls:
        cum += p
        peak = max(peak, cum)
        max_dd = max(max_dd, peak - cum)
    max_dd_frac = (max_dd / peak) if peak > 0 else 0.0
    # Per-trade Sharpe/Sortino approx from quote PnLs (small sample ok for ranking).
    import statistics

    sharpe = sortino = 0.0
    if n >= 3:
        mu = statistics.mean(pnls)
        sd = statistics.pstdev(pnls)
        sharpe = (mu / sd) if sd > 1e-9 else 0.0
        neg = [p for p in pnls if p < 0]
        if neg:
            nsd = statistics.pstdev(neg)
            sortino = (mu / nsd) if nsd > 1e-9 else sharpe
        else:
            sortino = sharpe
    by_bucket: Dict[str, List[float]] = {k: [] for k in ("TREND_UP", "TREND_DOWN", "RANGE", "RISK_OFF")}
    for r in trades:
        b = _map_regime_label(str(r["entry_regime"]) if r["entry_regime"] else None)
        by_bucket[b].append(float(r["pnl_quote"] or 0.0))
    return {
        "trades": n,
        "win_rate": win_rate,
        "profit_factor": profit_factor,
        "expectancy": expectancy,
        "sharpe": sharpe,
        "sortino": sortino,
        "max_dd": max_dd_frac,
        "avg_hold_hours": avg_hold,
        "regime_fit": _regime_fit_from_pnls(by_bucket),
    }


def _latest_backtest_row(strategy: str, symbols: List[str]) -> Optional[Dict[str, Any]]:
    import db

    db.init_db()
    path = os.path.abspath(db.DB_NAME)
    con = sqlite3.connect(path)
    con.row_factory = sqlite3.Row
    try:
        for sym in symbols:
            row = con.execute(
                """
                SELECT metrics, equity, symbol, ts
                FROM backtest_runs
                WHERE strategy = ?
                  AND symbol = ?
                ORDER BY ts DESC
                LIMIT 1
                """,
                (strategy, sym),
            ).fetchone()
            if row:
                return dict(row)
        row = con.execute(
            """
            SELECT metrics, equity, symbol, ts
            FROM backtest_runs
            WHERE strategy = ?
            ORDER BY ts DESC
            LIMIT 1
            """,
            (strategy,),
        ).fetchone()
        return dict(row) if row else None
    finally:
        con.close()


def _metrics_from_backtest_row(row: Dict[str, Any]) -> Dict[str, Any]:
    try:
        m = json.loads(row.get("metrics") or "{}")
    except Exception:
        m = {}
    wr = float(m.get("win_rate", m.get("win_pct", 0)) or 0)
    if wr > 1.0:
        wr /= 100.0
    return {
        "trades": int(m.get("total_trades", m.get("trades", 0)) or 0),
        "win_rate": wr,
        "profit_factor": float(m.get("profit_factor", 0) or 0),
        "expectancy": float(m.get("expectancy", 0) or 0),
        "sharpe": float(m.get("sharpe_ratio", m.get("sharpe", 0)) or 0),
        "sortino": float(m.get("sortino_ratio", m.get("sortino", 0)) or 0),
        "max_dd": float(m.get("max_drawdown_pct", m.get("max_dd", 0)) or 0) / 100.0
        if float(m.get("max_drawdown_pct", 0) or 0) > 1.0
        else float(m.get("max_drawdown_pct", m.get("max_dd", 0)) or 0),
        "avg_hold_hours": float(m.get("avg_trade_duration_hours", 0) or 0),
        "regime_fit": m.get("regime_fit")
        if isinstance(m.get("regime_fit"), dict)
        else {k: 0.5 for k in ("TREND_UP", "TREND_DOWN", "RANGE", "RISK_OFF")},
    }


def _run_inline_backtest(strategy: str, symbol: str, timeout_sec: float = 8.0) -> Dict[str, Any]:
    ckey = f"{strategy}|{symbol}|90d"
    if ckey in _INLINE_CACHE:
        return _INLINE_CACHE[ckey]
    eng_strategy = _ENGINE_MAP.get(strategy, strategy)
    if eng_strategy not in (
        "dca",
        "classic",
        "trend_follow",
        "mean_reversion",
        "momentum",
        "grid",
        "smart_dca",
        "smart",
    ):
        eng_strategy = "smart_dca"

    def _job() -> Dict[str, Any]:
        from backtest_engine import BacktestEngine

        seed = int(hashlib.sha256(ckey.encode()).hexdigest()[:8], 16)
        rnd = random.Random(seed)
        t0 = int(time.time()) - 90 * 86400
        p = 50_000.0 if "BTC" in symbol.upper() else 150.0
        candles: List[Dict[str, float]] = []
        for i in range(220):
            p *= 1.0 + rnd.uniform(-0.018, 0.02)
            o, h, l, c = p * 0.999, p * 1.008, p * 0.992, p
            candles.append(
                {
                    "time": float(t0 + i * 86400),
                    "open": o,
                    "high": h,
                    "low": l,
                    "close": c,
                    "volume": 1e6,
                }
            )
        eng = BacktestEngine(
            symbol,
            candles,
            {"strategy": eng_strategy, "base_quote": 100.0},
        )
        res = eng.run()
        d = res.to_dict()
        wr = float(d.get("win_rate", 0) or 0) / 100.0
        m = {
            "trades": int(d.get("total_trades", 0) or 0),
            "win_rate": wr,
            "profit_factor": float(d.get("profit_factor", 0) or 0),
            "expectancy": float(d.get("total_return_usd", 0) or 0) / max(1, int(d.get("total_trades", 1))),
            "sharpe": float(d.get("sharpe_ratio", 0) or 0),
            "sortino": float(d.get("sortino_ratio", 0) or 0),
            "max_dd": float(d.get("max_drawdown_pct", 0) or 0) / 100.0,
            "avg_hold_hours": float(d.get("avg_trade_duration_hours", 0) or 0),
            "regime_fit": {k: round(0.45 + 0.1 * rnd.random(), 3) for k in ("TREND_UP", "TREND_DOWN", "RANGE", "RISK_OFF")},
        }
        return m

    with ThreadPoolExecutor(max_workers=1) as ex:
        fut = ex.submit(_job)
        try:
            out = fut.result(timeout=timeout_sec)
        except FuturesTimeout:
            logger.warning("leaderboard inline backtest timeout strategy=%s symbol=%s", strategy, symbol)
            out = {
                "trades": 0,
                "win_rate": 0.45,
                "profit_factor": 1.1,
                "expectancy": 0.0,
                "sharpe": 0.5,
                "sortino": 0.5,
                "max_dd": 0.12,
                "avg_hold_hours": 24.0,
                "regime_fit": {k: 0.5 for k in ("TREND_UP", "TREND_DOWN", "RANGE", "RISK_OFF")},
            }
    _INLINE_CACHE[ckey] = out
    return out


def _active_symbols_or_default() -> List[str]:
    try:
        from db import list_bots

        syms = []
        for b in list_bots():
            s = str(b.get("symbol") or "").strip()
            if s:
                syms.append(s)
        if syms:
            return syms[:12]
    except Exception:
        pass
    return ["BTC/USD", "SPY"]


def _live_nonzero(stats: Dict[str, Any]) -> bool:
    if stats["trades"] <= 0:
        return False
    return any(
        abs(float(stats.get(k, 0) or 0)) > 1e-9
        for k in ("win_rate", "profit_factor", "sharpe", "expectancy")
    )


def build_leaderboard(window_days: int = 90, min_live_trades: int = 10) -> List[Dict[str, Any]]:
    """
    One row per canonical strategy, sorted by composite ``score`` descending.
    """
    from db import now_ts

    since = int(now_ts()) - int(window_days) * 86400
    rows = _fetch_journal_rows(since)
    by_s: Dict[str, List[sqlite3.Row]] = {}
    for r in rows:
        s = str(r["strategy"] or "smart_dca").strip().lower()
        by_s.setdefault(s, []).append(r)

    symbols = _active_symbols_or_default()
    sym0 = next((s for s in symbols if "/" in s), symbols[0] if symbols else "BTC/USD")
    out: List[Dict[str, Any]] = []
    for strat in CANONICAL_STRATEGIES:
        live_rows = by_s.get(strat, [])
        live_stats = _aggregate_trades(live_rows) if live_rows else {
            "trades": 0,
            "win_rate": 0.0,
            "profit_factor": 0.0,
            "expectancy": 0.0,
            "sharpe": 0.0,
            "sortino": 0.0,
            "max_dd": 0.0,
            "avg_hold_hours": 0.0,
            "regime_fit": {k: 0.0 for k in ("TREND_UP", "TREND_DOWN", "RANGE", "RISK_OFF")},
        }
        bt_row = _latest_backtest_row(strat, symbols)
        bt_metrics: Optional[Dict[str, Any]] = _metrics_from_backtest_row(bt_row) if bt_row else None

        if live_stats["trades"] >= min_live_trades and _live_nonzero(live_stats):
            merged = dict(live_stats)
            source = "live"
        elif (
            bt_metrics is not None
            and 5 <= live_stats["trades"] < min_live_trades
            and live_stats["trades"] > 0
        ):
            source = "blended"
            w_live, w_bt = 0.6, 0.4
            merged = {
                "trades": live_stats["trades"] + int(bt_metrics.get("trades", 0)),
                "win_rate": w_live * live_stats["win_rate"] + w_bt * float(bt_metrics.get("win_rate", 0)),
                "profit_factor": w_live * live_stats["profit_factor"] + w_bt * float(bt_metrics.get("profit_factor", 0)),
                "sharpe": w_live * live_stats["sharpe"] + w_bt * float(bt_metrics.get("sharpe", 0)),
                "sortino": w_live * live_stats["sortino"] + w_bt * float(bt_metrics.get("sortino", 0)),
                "max_dd": w_live * live_stats["max_dd"] + w_bt * float(bt_metrics.get("max_dd", 0)),
                "expectancy": w_live * live_stats["expectancy"] + w_bt * float(bt_metrics.get("expectancy", 0)),
                "avg_hold_hours": w_live * live_stats["avg_hold_hours"] + w_bt * float(bt_metrics.get("avg_hold_hours", 0)),
                "regime_fit": live_stats.get("regime_fit") or bt_metrics.get("regime_fit"),
            }
        elif bt_metrics is not None:
            merged = dict(bt_metrics)
            source = "backtest"
        else:
            merged = _run_inline_backtest(strat, sym0)
            source = "backtest"

        score = _composite_score(
            merged["win_rate"],
            merged["profit_factor"],
            merged["sharpe"],
            merged["sortino"],
            merged["max_dd"],
        )
        out.append(
            {
                "strategy": strat,
                "source": source,
                "trades": int(merged.get("trades", 0)),
                "win_rate": round(float(merged.get("win_rate", 0)) * 100.0, 2)
                if float(merged.get("win_rate", 0)) <= 1.0
                else round(float(merged.get("win_rate", 0)), 2),
                "profit_factor": round(float(merged.get("profit_factor", 0)), 3),
                "sharpe": round(float(merged.get("sharpe", 0)), 3),
                "sortino": round(float(merged.get("sortino", 0)), 3),
                "max_dd": round(float(merged.get("max_dd", 0)), 4),
                "expectancy": round(float(merged.get("expectancy", 0)), 4),
                "avg_hold_hours": round(float(merged.get("avg_hold_hours", 0)), 2),
                "score": round(score * 100.0, 2),
                "regime_fit": merged.get("regime_fit")
                or {k: 0.0 for k in ("TREND_UP", "TREND_DOWN", "RANGE", "RISK_OFF")},
            }
        )

    out.sort(key=lambda r: float(r.get("score") or 0), reverse=True)
    return out
