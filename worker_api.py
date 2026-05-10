# worker_api.py  (TOP OF FILE)

import os
import socket
import sqlite3
import time
import threading
import re
import json
import logging
import gc
import hashlib
import math
import signal
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, List, Optional, Tuple

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.templating import Jinja2Templates

logger = logging.getLogger(__name__)

# =========================================================
# Graceful shutdown handling
# =========================================================
_shutdown_event = threading.Event()

def _handle_shutdown_signal(signum, frame):
    """Handle SIGINT and SIGTERM for graceful shutdown."""
    sig_name = "SIGINT" if signum == signal.SIGINT else "SIGTERM"
    logger.info(f"Graceful shutdown initiated by {sig_name}...")
    _shutdown_event.set()

# Register signal handlers
try:
    signal.signal(signal.SIGINT, _handle_shutdown_signal)
    signal.signal(signal.SIGTERM, _handle_shutdown_signal)
    logger.info("Signal handlers registered for graceful shutdown (SIGINT, SIGTERM)")
except Exception as e:
    logger.warning(f"Could not register signal handlers: {e}")

# =========================================================
# Memory monitoring for scan stability
# =========================================================
def _memory_usage_mb() -> float:
    """Return current process RSS in MB (prefers psutil for Windows + Linux accuracy)."""
    try:
        import psutil
        return float(psutil.Process().memory_info().rss) / (1024.0 * 1024.0)
    except Exception:
        pass
    try:
        import resource
        ru = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        # Linux: KB; macOS: bytes — heuristic
        if ru > 10**9:
            return float(ru) / (1024.0 * 1024.0)
        return float(ru) / 1024.0
    except Exception:
        pass
    try:
        with open("/proc/self/status") as f:
            for line in f:
                if line.startswith("VmRSS:"):
                    return int(line.split()[1]) / 1024.0
    except Exception:
        pass
    return 0.0


# Baseline RSS for this app (TensorFlow + scans) is often 600–1200MB. A single "400MB" ceiling
# aborts every scan and leaves medium/long never run + 502s when the worker is wedged.
# Soft = pause + GC only; Hard = abort scan (OOM protection).
_SCAN_MEMORY_SOFT_MB = float(os.getenv("SCAN_MEMORY_SOFT_MB", "1600"))
_SCAN_MEMORY_HARD_MB = float(os.getenv("SCAN_MEMORY_HARD_MB", "3200"))
# Legacy SCAN_MEMORY_LIMIT_MB → hard limit only if value is sane (old "400" deploys would abort every scan)
if os.getenv("SCAN_MEMORY_LIMIT_MB", "").strip():
    try:
        _lim = float(os.getenv("SCAN_MEMORY_LIMIT_MB", ""))
        if _lim >= 800:
            _SCAN_MEMORY_HARD_MB = _lim
    except ValueError:
        pass

_SCAN_CHECKPOINT_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".scan_checkpoint.json")
# Single scan slot: manual Rescan endpoint shares this to prevent UI-triggered duplicates.
_RECO_SCAN_SEM = threading.Semaphore(1)
# True while any multi-step scan thread (_scan_async) is running — stays True between horizons.
_RECO_SCAN_ACTIVE = False
_RECO_SCAN_ACTIVE_LOCK = threading.Lock()
# Per-horizon scan locks — allow all 3 horizons to run in parallel without blocking each other.
_HORIZON_SCANNING: Dict[str, bool] = {"short": False, "medium": False, "long": False}
_HORIZON_SCAN_LOCKS: Dict[str, threading.Lock] = {
    "short": threading.Lock(), "medium": threading.Lock(), "long": threading.Lock(),
}
# One-shot threads that are expected to exit: exclude from watchdog restart + report as "completed".
_ONE_SHOT_THREADS = frozenset({"autostart", "bots_summary_prewarm", "explore_startup_scan", "startup_backtest"})


def _scan_checkpoint_read() -> Dict[str, Any]:
    try:
        import json
        if os.path.isfile(_SCAN_CHECKPOINT_PATH):
            with open(_SCAN_CHECKPOINT_PATH, "r", encoding="utf-8") as f:
                return json.load(f)
    except Exception:
        pass
    return {}


def _scan_checkpoint_write(horizon: str, fingerprint: str, next_batch_start: int) -> None:
    try:
        import json
        with open(_SCAN_CHECKPOINT_PATH, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "horizon": horizon,
                    "fingerprint": fingerprint,
                    "next_batch_start": int(next_batch_start),
                    "ts": int(time.time()),
                },
                f,
            )
    except Exception:
        pass


def _scan_checkpoint_clear_for_horizon(horizon: str) -> None:
    try:
        ck = _scan_checkpoint_read()
        if ck.get("horizon") == horizon and os.path.isfile(_SCAN_CHECKPOINT_PATH):
            os.remove(_SCAN_CHECKPOINT_PATH)
    except Exception:
        pass

# =========================================================
# Env loader: MUST RUN BEFORE importing KrakenClient/BotManager
# =========================================================
from env_utils import load_env, get_last_load_result
load_env()

# NOW import project modules that may read env vars
from symbol_classifier import classify_symbol, is_stock_symbol, is_crypto_symbol, validate_symbol_type

from db import (
    init_db,
    now_ts,
    update_bots_by_type,
    add_log,
    create_bot,
    patch_bot_risk_after_create,
    update_bot,
    delete_bot,
    get_bot,
    list_bots,
    list_logs,
    list_logs_since,
    list_deals,
    list_all_deals,
    list_closed_deals_for_journal,
    list_journal_entries,
    get_deal,
    pnl_summary,
    bot_deal_stats,
    latest_open_deal,
    bot_pnl_series,
    bot_drawdown_series,
    bot_performance_stats,
    latest_regime,
    list_strategy_decisions,
    add_order_event,
    save_recommendation_snapshot,
    mark_explore_signals_pending,
    upsert_explore_feed_row,
    list_explore_feed,
    list_explore_rejected,
    get_latest_explore_backtest,
    save_explore_backtest_results,
    list_recommendations,
    count_recommendations_by_horizon,
    count_closed_deals,
    get_recommendation,
    link_recommendation_to_bot,
    get_recommendation_performance_stats,
    delete_recommendations_for_blocklist,
    save_signal_outcome,
    update_explore_signal_outcome,
    list_explore_outcomes_pending_old,
    get_strategy_win_rates,
    set_setting,
    get_setting,
    get_trade_journal,
    upsert_trade_journal,
    list_trade_journals_for_deals,
    get_intelligence_decisions,
    find_stale_ghost_deals,
    cancel_ghost_deal,
    explore_signals_max_updated_ts,
    list_signal_accuracy_baselines,
    list_portfolio_equity_curve,
    get_latest_regime_for_symbols,
)


def get_strategy_leaderboard(window_days: int = 90):
    """Journal + backtest fallback leaderboard (exposed for templates / safety_checklist)."""
    from services.leaderboard import build_leaderboard

    return build_leaderboard(window_days=int(window_days), min_live_trades=10)


from kraken_client import KrakenClient
from alpaca_client import AlpacaClient
from bot_manager import BotManager, ALLOW_LIVE_TRADING
from alpaca_adapter import AlpacaAdapter
from explore_scorer import signal_age_penalty, price_confirmation_score
from explore_rank import (
    apply_smart_rank as _explore_apply_smart_rank,
    bt_lookup as _explore_bt_lookup,
    row_detail_json as _explore_row_detail_json,
)


_STRATEGY_PERF_CACHE: Dict[Tuple[str, int], Tuple[float, Dict[str, Any]]] = {}
_EXPLORE_OUTCOMES_LAST_TS: float = 0.0


def _explore_signal_status_from_snap(snap: Dict[str, Any]) -> Tuple[str, str]:
    """
    Forced explore_signals row when evaluate_signal rejects or EXPLORE_V2_GATE is set.
    Returns ("", "") when evaluate_explore should decide buy/watch/rejected.
    """
    metrics = snap.get("metrics") or {}
    risk_flags = snap.get("risk_flags") or []
    if str(metrics.get("_evaluate_signal") or "") == "reject":
        return "rejected", str(metrics.get("_explore_reject_reason") or "evaluate_signal_reject")
    if any(str(f).startswith("EXPLORE_V2_GATE:") for f in risk_flags):
        return "rejected", "explore_v2_gate"
    return "", ""


def _reco_rows_as_explore_feed_rows(
    horizon: str,
    market_type: str,
    statuses: List[str],
    limit: int,
) -> List[Dict[str, Any]]:
    """Fallback: shape recommendations_snapshots rows like explore_signals for the Explore feed."""
    raw = list_recommendations(horizon, limit=min(500, max(int(limit), 100)))
    if not raw:
        return []
    want_crypto = market_type == "crypto"
    want_stocks = market_type == "stocks"
    out: List[Dict[str, Any]] = []
    for s in raw:
        sym = str(s.get("symbol") or "")
        if not sym:
            continue
        try:
            metrics = json.loads(s.get("metrics_json") or "{}")
        except Exception:
            metrics = {}
        mt = (metrics.get("market_type") or "").strip().lower()
        if mt == "stock":
            mt = "stocks"
        if not mt:
            mt = "crypto" if "/" in sym else "stocks"
        if want_crypto and mt != "crypto":
            continue
        if want_stocks and mt != "stocks":
            continue
        sc = float(s.get("score") or 0)
        st = "buy" if sc >= 70 else "watch"
        if st not in statuses:
            continue
        rs: List[Any] = []
        try:
            rs = json.loads(s.get("reasons_json") or "[]")
        except Exception:
            rs = []
        reason0 = str(rs[0]) if rs else ""
        out.append(
            {
                "symbol": sym,
                "status": st,
                "conviction_score": sc,
                "reason": reason0,
                "strategy": str(
                    metrics.get("detected_strategy") or metrics.get("recommended_strategy") or ""
                ),
                "signal_ts": int(s.get("created_ts") or 0),
                "updated_ts": int(s.get("created_ts") or 0),
                "market_type": "crypto" if mt == "crypto" else "stocks",
                "price": metrics.get("price"),
                "change_24h": metrics.get("change_24h"),
                "detail_json": str(s.get("metrics_json") or "{}"),
                "rejection_reason": "",
            }
        )
        if len(out) >= int(limit):
            break
    return out


def _explore_feed_as_recommendation_rows(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Shape explore_signals rows like list_recommendations for /api/recommendations fallback."""
    out: List[Dict[str, Any]] = []
    for r in rows or []:
        sym = str(r.get("symbol") or "")
        if not sym:
            continue
        try:
            detail = json.loads(r.get("detail_json") or "{}")
        except Exception:
            detail = {}
        if not isinstance(detail, dict):
            detail = {}
        metrics = {
            **detail,
            "market_type": r.get("market_type") or ("crypto" if "/" in sym else "stocks"),
            "price": r.get("price"),
            "change_24h": r.get("change_24h"),
            "data_source": "explore_signals",
        }
        out.append(
            {
                "symbol": sym,
                "score": float(r.get("conviction_score") or 0),
                "metrics_json": json.dumps(metrics),
                "reasons_json": json.dumps([str(r.get("reason") or "")]),
                "risk_flags_json": "[]",
                "regime_json": json.dumps({}),
                "score_breakdown_json": "{}",
                "created_ts": int(r.get("updated_ts") or 0),
                "metrics": metrics,
            }
        )
    return out


def _explore_strict_buy_enabled() -> bool:
    return os.getenv("EXPLORE_STRICT_BUY", "1").strip().lower() not in ("0", "false", "no", "off")


def _explore_strict_buy_demote_reason(
    snap: Dict[str, Any],
    metrics: Dict[str, Any],
    is_crypto: bool,
) -> str:
    """
    When evaluate_explore says 'buy' but real-money / execution hygiene is weak,
    demote to 'watch' with a short machine reason (also stored in detail_json).
    """
    if not snap.get("eligible", True):
        return "scanner_not_eligible"
    flags = [str(f) for f in (snap.get("risk_flags") or [])]
    bad_exact = {"DATA_ERROR", "DATA_INVALID", "ROUTING_ERROR", "NO_CRYPTO_PROVIDER"}
    for f in flags:
        if f in bad_exact:
            return f"risk_flag:{f}"
        if f.startswith("EXPLORE_V2_GATE:"):
            return f"risk_flag:{f[:56]}"

    max_ed = int(os.getenv("EXPLORE_STRICT_EARNINGS_DAYS", "3"))
    ed = metrics.get("earnings_days")
    if not is_crypto and max_ed >= 0 and isinstance(ed, (int, float)) and 0 <= float(ed) <= float(max_ed):
        return f"earnings_within_{max_ed}d"

    if is_crypto:
        try:
            max_sp = float(os.getenv("EXPLORE_STRICT_MAX_SPREAD_PCT", "0.012"))
        except (TypeError, ValueError):
            max_sp = 0.012
        spr = metrics.get("spread_pct")
        try:
            spf = float(spr) if spr is not None else None
        except (TypeError, ValueError):
            spf = None
        if spf is not None and spf > max_sp:
            return f"wide_spread_{spf*100:.2f}pct_gt_{max_sp*100:.2f}"

        volq = metrics.get("volume_24h_quote")
        try:
            vq = float(volq) if volq is not None else None
        except (TypeError, ValueError):
            vq = None
        if vq is not None and vq > 0:
            try:
                min_v = float(os.getenv("EXPLORE_STRICT_MIN_CRYPTO_QUOTE_VOL", str(RECO_MIN_VOLUME_24H)))
            except Exception:
                min_v = float(RECO_MIN_VOLUME_24H)
            if vq < min_v:
                return "low_crypto_liquidity"

    return ""


def _merge_evaluate_signal_into_snap(snap: Dict[str, Any], horizon: str, btc_ctx: Dict[str, Any]) -> None:
    """Run explore_signals.evaluate_signal and merge strategy fields into snap['metrics'] for DB + API."""
    _ = horizon
    try:
        from explore_signals import evaluate_signal

        metrics = dict(snap.get("metrics") or {})
        sym = str(snap.get("symbol") or "")
        is_crypto = (metrics.get("market_type") or "").lower() == "crypto" or "/" in sym
        candles = list(snap.get("_candles_1d") or [])
        price = metrics.get("price")
        try:
            price_f = float(price) if price is not None else 0.0
        except (TypeError, ValueError):
            price_f = 0.0
        try:
            vol = float(metrics.get("volume_24h_quote") or metrics.get("volume_24h") or 0.0)
        except (TypeError, ValueError):
            vol = 0.0
        fg = int(_FEAR_GREED_CACHE.get("value") or 50)
        try:
            existing = float(metrics.get("composite_score") or snap.get("score") or 0.0)
        except (TypeError, ValueError):
            existing = 0.0
        ev = evaluate_signal(
            sym,
            "crypto" if is_crypto else "stock",
            price_f,
            candles,
            dict(metrics),
            vol,
            fg,
            dict(btc_ctx or {}),
            existing,
            metrics,
        )
        metrics["_evaluate_signal"] = ev.get("signal")
        metrics["_explore_reject_reason"] = ev.get("rejection_reason")
        if ev.get("detected_strategy"):
            metrics["detected_strategy"] = ev.get("detected_strategy")
        metrics["strategy_reason"] = ev.get("strategy_reason") or ""
        snap["metrics"] = metrics
    except Exception as e:
        logger.debug("_merge_evaluate_signal_into_snap %s: %s", snap.get("symbol"), e)


def _get_cached_strategy_win_rates(horizon: str, lookback_days: int = 90) -> Dict[str, Any]:
    key = (horizon, int(lookback_days))
    now = time.time()
    hit = _STRATEGY_PERF_CACHE.get(key)
    if hit and (now - hit[0]) < 600:
        return hit[1]
    data = get_strategy_win_rates(horizon, lookback_days=lookback_days)
    _STRATEGY_PERF_CACHE[key] = (now, data)
    return data


def _candle_ts_sec(c: List[float]) -> int:
    try:
        return int(float(c[0]))
    except Exception:
        return 0


def _weekly_ohlcv_from_daily(daily: List[List[float]], max_weeks: int = 200) -> List[List[float]]:
    """Synthesize weekly OHLCV from daily rows when native '1w' candles are missing (Kraken/CCXT edge cases)."""
    if not daily or len(daily) < 7:
        return []
    from datetime import datetime, timezone

    weekly: Dict[int, List[float]] = {}
    for row in daily:
        if len(row) < 6:
            continue
        try:
            ts_ms = int(float(row[0]))
            dt = datetime.fromtimestamp(ts_ms / 1000.0, tz=timezone.utc)
            y, w, _ = dt.isocalendar()
            wk_key = int(y) * 100 + int(w)
            o, h, l, c, v = float(row[1]), float(row[2]), float(row[3]), float(row[4]), float(row[5])
        except Exception:
            continue
        if wk_key not in weekly:
            weekly[wk_key] = [float(ts_ms), o, h, l, c, v]
        else:
            agg = weekly[wk_key]
            agg[2] = max(agg[2], h)
            agg[3] = min(agg[3], l)
            agg[4] = c
            agg[5] += v
    keys = sorted(weekly.keys())
    return [weekly[k] for k in keys[-max_weeks:]]


def _forward_prices_from_candles(
    candles: List[List[float]],
    signal_ts: int,
) -> Tuple[Optional[float], Optional[float], Optional[float]]:
    if not candles or not signal_ts:
        return None, None, None
    idx = None
    for i, c in enumerate(candles):
        if _candle_ts_sec(c) >= signal_ts:
            idx = i
            break
    if idx is None:
        return None, None, None

    def _cls(off: int) -> Optional[float]:
        j = idx + off
        if j < len(candles):
            try:
                return float(candles[j][4])
            except Exception:
                return None
        return None

    return _cls(5), _cls(10), _cls(20)


def _fetch_1d_for_explore_outcome(sym: str) -> List[List[float]]:
    if "/" in sym:
        if not kc:
            return []
        try:
            return kc.fetch_ohlcv(sym, timeframe="1d", limit=400)
        except Exception:
            return []
    ac = alpaca_live or alpaca_paper
    if not ac:
        return []
    try:
        return ac.get_ohlcv(sym, timeframe="1d", limit=400)
    except Exception:
        return []


def _run_explore_outcomes_update_batch() -> None:
    """Every ~4h from recommendations thread: fill 5d/10d/20d closes for pending explore outcomes."""
    global _EXPLORE_OUTCOMES_LAST_TS
    now = time.time()
    if now - _EXPLORE_OUTCOMES_LAST_TS < 4 * 3600:
        return
    _EXPLORE_OUTCOMES_LAST_TS = now
    rows = list_explore_outcomes_pending_old(min_age_sec=5 * 86400, limit=200)
    if not rows:
        return
    updated = 0
    for row in rows:
        try:
            sym = str(row.get("symbol") or "")
            if not sym:
                continue
            candles = _fetch_1d_for_explore_outcome(sym)
            p5, p10, p20 = _forward_prices_from_candles(candles, int(row.get("signal_ts") or 0))
            if p5 is None and p10 is None and p20 is None:
                continue
            update_explore_signal_outcome(int(row["id"]), price_5d=p5, price_10d=p10, price_20d=p20)
            updated += 1
        except Exception as e:
            logger.debug("explore outcome row %s: %s", row.get("id"), e)
    if updated:
        logger.info("Explore outcomes: updated %d pending rows (forward prices)", updated)


def _persist_explore_feed_from_snap(
    symbol: str,
    horizon: str,
    snap: Dict[str, Any],
    candles_1d: List[List[float]],
    btc_ctx: Optional[Dict[str, Any]] = None,
) -> None:
    """Final Explore write: conviction strategies + UPSERT explore_signals (buy/watch/rejected)."""
    try:
        from explore_signals import STRATEGY_LABELS, evaluate_explore, infer_rejection_code

        metrics = snap.get("metrics") or {}
        is_crypto = (metrics.get("market_type") or "").lower() == "crypto" or "/" in str(symbol)
        ev = evaluate_explore(
            str(symbol),
            list(candles_1d or []),
            is_crypto=is_crypto,
            horizon=horizon,
            market_breadth=dict(metrics),
            btc_context=dict(btc_ctx or {}),
        )
        forced_st, forced_reason = _explore_signal_status_from_snap(snap)
        _bear_strategies = {"relative_strength_bear", "oversold_extreme_fear", "volume_capitulation", "oversold_bounce"}
        _ev_strategy = str(ev.get("strategy") or "")
        _ev_has_bear_match = _ev_strategy in _bear_strategies and ev.get("status") != "rejected"
        if forced_st and not _ev_has_bear_match:
            row_status = forced_st
            row_reason = forced_reason
            row_conv = 0.0
        elif forced_st and _ev_has_bear_match:
            row_status = str(ev.get("status") or "watch")
            row_reason = str(ev.get("reason") or "")
            row_conv = float(ev.get("conviction") or 0)
            if row_status == "buy":
                row_status = "watch"
                row_reason += " | Gate bypassed for bear strategy (demoted to watch)"
        else:
            row_status = str(ev.get("status") or "rejected")
            row_reason = str(ev.get("reason") or "")
            row_conv = float(ev.get("conviction") or 0)

        strict_demote = ""
        if row_status == "buy" and not forced_st and _explore_strict_buy_enabled():
            strict_demote = _explore_strict_buy_demote_reason(snap, metrics, is_crypto)
            if strict_demote:
                row_status = "watch"
                row_reason = (row_reason + " | " if row_reason else "") + "Strict buy gate: " + strict_demote

        price = metrics.get("price")
        try:
            price_f = float(price) if price is not None and float(price) > 0 else None
        except (TypeError, ValueError):
            price_f = None
        chg = metrics.get("change_24h")
        try:
            chg_f = float(chg) if chg is not None else None
        except (TypeError, ValueError):
            chg_f = None
        # Prefer chart-pattern name (evaluate_explore) over hybrid scanner label (evaluate_signal).
        # detect_strategy() often falls through to "Trend Follow" for most names — that hid real patterns.
        _pat_key = str(ev.get("strategy") or "").strip()
        _pattern_label = STRATEGY_LABELS.get(_pat_key, _pat_key) if _pat_key else ""
        strat_human = (
            _pattern_label
            or str(metrics.get("detected_strategy") or "").strip()
            or _pat_key
        )
        detail = dict(ev.get("detail") or {})
        if strict_demote:
            detail["strict_buy_demotion"] = strict_demote
        # Persist vol_24h so the feed can render the volume column
        _vol = metrics.get("vol_24h") or metrics.get("volume_24h") or metrics.get("volume")
        if _vol and not detail.get("vol_24h"):
            try:
                detail["vol_24h"] = float(_vol)
            except (TypeError, ValueError):
                pass
        detail["evaluate_signal"] = {
            "signal": metrics.get("_evaluate_signal"),
            "detected_strategy": metrics.get("detected_strategy"),
            "strategy_reason": metrics.get("strategy_reason"),
        }
        # Volume vs 20d average for Explore table (× multiplier column)
        try:
            from explore_signals import volume_avg_and_ratio_from_candles

            _facts = detail.get("facts") if isinstance(detail.get("facts"), dict) else {}
            _vr_fact = _facts.get("volume_ratio")
            if _vr_fact is None:
                _vr_fact = _facts.get("volume_mult")
            if _vr_fact is not None:
                try:
                    detail["volume_ratio"] = round(float(_vr_fact), 4)
                except (TypeError, ValueError):
                    pass
            _vr_c, _avg_c = volume_avg_and_ratio_from_candles(list(candles_1d or []))
            if detail.get("volume_ratio") is None and _vr_c is not None:
                detail["volume_ratio"] = round(float(_vr_c), 4)
            if _avg_c is not None and float(_avg_c) > 0:
                detail["avg_volume_20d"] = float(_avg_c)
        except Exception:
            pass
        _rej_code = infer_rejection_code(row_reason) if row_status == "rejected" else None
        upsert_explore_feed_row(
            str(symbol),
            horizon,
            row_status,
            row_conv,
            row_reason,
            strat_human[:128] if strat_human else str(ev.get("strategy") or ""),
            int(ev.get("signal_ts") or 0),
            json.dumps(detail),
            price_f,
            chg_f,
            "crypto" if is_crypto else "stocks",
            rejection_reason=_rej_code,
        )
        if row_status == "buy" and price_f:
            try:
                cs = metrics.get("composite_score")
                cs_f = float(cs) if cs is not None else None
            except (TypeError, ValueError):
                cs_f = None
            cg = metrics.get("conviction_grade")
            save_signal_outcome(
                str(symbol),
                str(horizon),
                str(strat_human or metrics.get("detected_strategy") or "Trend Follow")[:200],
                int(ev.get("signal_ts") or now_ts()),
                float(price_f),
                cs_f,
                str(cg)[:8] if cg else None,
            )
    except Exception as e:
        logger.warning("persist explore feed failed for %s: %s", symbol, e)


USE_UNIFIED_ALPACA = os.getenv("USE_UNIFIED_ALPACA", "1").strip().lower() in ("1", "true", "yes", "y", "on")
try:
    from unified_alpaca_client import UnifiedAlpacaClient, ALPACA_PY_AVAILABLE
    _UNIFIED_AVAILABLE = ALPACA_PY_AVAILABLE
except ImportError:
    UnifiedAlpacaClient = None
    _UNIFIED_AVAILABLE = False
from strategies import (
    detect_regime,
    select_strategy,
    DcaConfig,
    sma,
    ema,
    ema_series,
    rsi,
    adx,
    macd,
    _atr,
    rolling_return,
    max_drawdown,
    current_drawdown,
    lower_lows_persistence,
    base_formation,
    clamp,
)

from intelligence_layer import IntelligenceLayer, IntelligenceContext
# Global Intelligence Layer instance
intelligence_layer = IntelligenceLayer()


# =========================================================
# App + globals
# =========================================================
app = FastAPI()
app.add_middleware(GZipMiddleware, minimum_size=1000)

_globals_lock = threading.RLock()
_thread_started: Dict[str, bool] = {}
_thread_start_lock = threading.Lock()
_last_portfolio_ts: float = 0.0
_last_reco_short_ts: float = 0.0
_last_reco_medium_ts: float = 0.0
_last_reco_long_ts: float = 0.0

kc: Optional[KrakenClient] = None
alpaca_paper: Optional[AlpacaClient] = None
alpaca_live: Optional[AlpacaClient] = None
bm: Optional[BotManager] = None
KRAKEN_READY: bool = False
KRAKEN_ERROR: str = ""
ALPACA_PAPER_READY: bool = False
ALPACA_LIVE_READY: bool = False
ALPACA_ERROR: str = ""
LIVE_TRADING_ENABLED: bool = os.getenv("LIVE_TRADING_ENABLED", "0").strip().lower() in (
    "1",
    "true",
    "yes",
    "y",
    "on",
)
LIVE_ENDPOINTS_DISABLED: bool = False
LIVE_ENDPOINTS_DISABLED_REASON: str = ""

_APP_START_TIME: float = time.time()

_STARTUP_STATUS: Dict[str, Any] = {
    "flask_ready": False, "db_ready": False, "db_bots": 0,
    "alpaca_ready": False, "alpaca_buying_power": 0.0,
    "websocket_status": "not_ready", "autopilot_enabled": False, "autopilot_bots": 0,
    "candle_test": None,
    "env_loaded_paths": [], "db_path": None, "kraken_ready": False, "bm_ready": False,
    "last_startup_error": None, "timestamp": None,
}

PORT_HISTORY: List[Dict[str, Any]] = []
PORT_EVERY_SEC = int(os.getenv("PORT_EVERY_SEC", "60"))
_last_portfolio_cleanup_ts: float = 0.0

_MARKETS_CACHE: Dict[str, Any] = {"ts": 0.0, "markets": None}
MARKETS_TTL_SEC = int(os.getenv("MARKETS_TTL_SEC", "300"))  # 5 minutes
_TICKERS_CACHE: Dict[str, Dict[str, Any]] = {}
DISCORD_WEBHOOK_URL = os.getenv("DISCORD_WEBHOOK_URL", "").strip()
DISCORD_STATUS_WEBHOOK_URL = os.getenv("DISCORD_STATUS_WEBHOOK_URL", "").strip()
DISCORD_STATUS_MSG_FILE = os.getenv(
    "DISCORD_STATUS_MSG_FILE", "/home/ubuntu/botdata/discord_status_msg_id.txt"
)
DISCORD_STATUS_LOG = os.getenv(
    "DISCORD_STATUS_LOG", "/home/ubuntu/botdata/discord_status.log"
)
AUTO_START_ENABLED = os.getenv("AUTO_START_ENABLED", "1").strip().lower() in (
    "1",
    "true",
    "yes",
    "y",
    "on",
)
HEALTH_WATCHDOG_SEC = int(os.getenv("HEALTH_WATCHDOG_SEC", "30"))
RECO_SYMBOLS = [
    s.strip() for s in os.getenv(
        "RECO_SYMBOLS",
        ",".join([
            # Top tier (ccxt Kraken uses BTC/USD, not XBT/USD)
            "BTC/USD","ETH/USD","SOL/USD","XRP/USD","ADA/USD","DOGE/USD","AVAX/USD","LINK/USD",
            "LTC/USD","BCH/USD","DOT/USD","ATOM/USD","XLM/USD","ETC/USD","UNI/USD","AAVE/USD",
            "MATIC/USD","ALGO/USD","TRX/USD","EOS/USD","ICP/USD","FTM/USD","SAND/USD","MANA/USD",
            "GRT/USD","APE/USD","FIL/USD","NEAR/USD","XTZ/USD","HBAR/USD","EGLD/USD","FLOW/USD",
            "KSM/USD","QNT/USD","CRV/USD","COMP/USD","SNX/USD","MKR/USD","ZEC/USD","DASH/USD",
            # Additional popular cryptos
            "BNB/USD","RNDR/USD","INJ/USD","TIA/USD","SUI/USD","SEI/USD","ARB/USD","OP/USD",
            "STRK/USD","IMX/USD","LDO/USD","RUNE/USD","FET/USD","AGIX/USD","OCEAN/USD","AXS/USD",
            "APT/USD","TON/USD","VET/USD","THETA/USD","CHZ/USD","GALA/USD","ENJ/USD","ONE/USD",
            "ZIL/USD","SUSHI/USD","YFI/USD","BAT/USD","CELR/USD","JASMY/USD","WOO/USD","BLUR/USD",
            "PEPE/USD","SHIB/USD","FLOKI/USD","BONK/USD","WIF/USD","BOME/USD","PEOPLE/USD","LUNC/USD",
            # Layer 1/2 and DeFi
            "ROSE/USD","KAVA/USD","MINA/USD","CELO/USD","WAVES/USD","ANT/USD","SRM/USD","OMG/USD",
            "STX/USD","GLMR/USD","MOVR/USD","KLAY/USD","CFX/USD","API3/USD","1INCH/USD","MASK/USD",
            # Gaming and metaverse
            "ILV/USD","ALICE/USD","TLM/USD","YGG/USD","GHST/USD","XYO/USD","PRIME/USD","BIGTIME/USD",
            # AI and data
            "RENDER/USD","GRT/USD","NMR/USD","LPT/USD","BAL/USD","STORJ/USD","AR/USD","KNC/USD",
        ])
    ).split(",")
    if s.strip()
]

# Append Stocks if Alpaca keys are present
_ALPACA_KEYS_PRESENT = bool(os.getenv("ALPACA_API_KEY_LIVE") or os.getenv("ALPACA_API_KEY_PAPER"))
if _ALPACA_KEYS_PRESENT:
    RECO_SYMBOLS.extend([
        # Tech / growth
        "AAPL", "MSFT", "GOOGL", "AMZN", "TSLA", "NVDA", "META", "NFLX", "AMD", "INTC", "QCOM", "CRM", "ADBE", "AVGO", "TXN", "PLTR", "ROKU", "SHOP", "PYPL", "SQ",
        # Financials
        "JPM", "BAC", "V", "MA", "WFC", "GS", "MS", "BLK", "C", "AXP",
        # ETFs
        "SPY", "QQQ", "IWM", "DIA", "XLK", "XLF", "XLE", "XLV", "XLY", "XLP", "XLI", "GLD", "SLV", "TQQQ", "SQQQ", "SOXL", "ARKK",
        # Crypto proxies / miners
        "COIN", "MSTR", "MARA", "RIOT", "CLSK", "HUT", "BITF", "HOOD",
        # Consumer / retail / travel
        "DIS", "KO", "PEP", "WMT", "TGT", "COST", "HD", "LOW", "MCD", "UBER", "LYFT", "DKNG", "AFRM", "UPST", "CVNA", "GME", "AMC", "SOFI",
        # Healthcare / pharma
        "JNJ", "PG", "PFE", "MRK", "UNH", "T", "VZ", "ABBV", "LLY",
        # Industrial / energy
        "BA", "F", "GM", "XOM", "CVX",
    ])
RECO_MAX_SYMBOLS = int(os.getenv("RECO_MAX_SYMBOLS", "600"))
# Per-scan crypto cap (stocks fill the rest up to RECO_MAX_SYMBOLS). Was hardcoded 50 → mostly stock signals.
RECO_SCAN_CRYPTO_CAP = int(os.getenv("RECO_SCAN_CRYPTO_CAP", "140"))
RECO_SHORT_EVERY_SEC = int(os.getenv("RECO_SHORT_EVERY_SEC", "1800"))   # 30m
RECO_MEDIUM_EVERY_SEC = int(os.getenv("RECO_MEDIUM_EVERY_SEC", "3600")) # 60m
RECO_LONG_EVERY_SEC = int(os.getenv("RECO_LONG_EVERY_SEC", "7200"))    # 2hr
# OHLCV cache TTL used during scans — long enough for medium/long to reuse short's fetched data.
_SCAN_OHLCV_TTL = int(os.getenv("SCAN_OHLCV_TTL", "7200"))  # 2hr
RECO_SHORT_MIN_DAYS = int(os.getenv("RECO_SHORT_MIN_DAYS", "90"))
RECO_MEDIUM_MIN_DAYS = int(os.getenv("RECO_MEDIUM_MIN_DAYS", "135"))
RECO_LONG_MIN_DAYS = int(os.getenv("RECO_LONG_MIN_DAYS", "180"))
RECO_LONG_MIN_WEEKS = int(os.getenv("RECO_LONG_MIN_WEEKS", "52"))
RECO_CRYPTO_TOP_30_ONLY = os.getenv("RECO_CRYPTO_TOP_30_ONLY", "0").strip().lower() in ("1", "true", "yes", "y", "on")
RECO_MAX_SPREAD_PCT = float(os.getenv("RECO_MAX_SPREAD_PCT", "0.004"))
RECO_MAX_ATR_PCT_SHORT = float(os.getenv("RECO_MAX_ATR_PCT_SHORT", "0.06"))
RECO_ATR_PCT_MODERATE = float(os.getenv("RECO_ATR_PCT_MODERATE", "0.035"))
_RECO_STATE_DEFAULT = {"last_run_ts": 0, "error": "", "scanned": 0, "eligible": 0, "total": 0, "scanning": False}
_RECO_STATE: Dict[str, Dict[str, Any]] = {
    "short": dict(_RECO_STATE_DEFAULT),
    "medium": dict(_RECO_STATE_DEFAULT),
    "long": dict(_RECO_STATE_DEFAULT),
}

# /api/bots/summary TTL cache — avoids N serial snapshot() calls per poll
_BOTS_SUMMARY_CACHE: Dict[str, Any] = {"result": None, "ts": 0.0}
_BOTS_SUMMARY_TTL = 15  # seconds

# /api/prices response cache — keyed by sorted symbols string, 20s TTL
_PRICES_CACHE: Dict[str, Dict[str, Any]] = {}
_PRICES_CACHE_TTL = 20  # seconds

# TTL caches for other frequently-polled endpoints (reduces CPU from 130%+ at idle)
_PORTFOLIO_CACHE: Dict[str, Any] = {"result": None, "ts": 0.0}
_PORTFOLIO_CACHE_TTL = 20  # seconds
_HEALTH_CACHE: Dict[str, Any] = {"result": None, "ts": 0.0}
_HEALTH_CACHE_TTL = 30  # seconds
_NOTIF_UNREAD_CACHE: Dict[str, Any] = {"result": None, "ts": 0.0}
_NOTIF_UNREAD_CACHE_TTL = 30  # seconds
_SCAN_PROGRESS_CACHE: Dict[str, Any] = {"result": None, "ts": 0.0}
_SCAN_PROGRESS_CACHE_TTL = 15  # seconds — always invalidated when a scan state changes

# Fear & Greed Index cache
_FEAR_GREED_CACHE: Dict[str, Any] = {"value": 50, "label": "Neutral", "ts": 0, "error": None}
_FEAR_GREED_TTL = 4 * 3600  # 4 hours
_RECO_OHLCV_CACHE: Dict[str, Dict[str, Any]] = {}
_OHLCV_CACHE_MAX_ENTRIES = int(os.getenv("OHLCV_CACHE_MAX_ENTRIES", "500"))
_OHLCV_CACHE_EVICT_AGE_SEC = 1800  # 30 min, was 60
_SCAN_PROGRESS: Dict[str, Any] = {
    "current_symbol": "",
    "current_horizon": "",
    "scan_start_ts": 0,
    "recent_errors": [],
    "buy_signals_found": 0,
    "scan_history": [],
}
_SCAN_HISTORY_MAX = 10
# Parallel recommendation scans: I/O-bound (Alpaca/Kraken/Yahoo) — use >1 worker per vCPU.
_CPU_FOR_SCAN = int(os.cpu_count() or 4)
_SCAN_PARALLEL_DEFAULT = 3  # Hard cap — server has limited CPU
SCAN_PARALLEL_WORKERS = min(
    3,  # Hard cap at 3 — server has limited RAM/CPU
    int(os.getenv("SCAN_PARALLEL_WORKERS", "3"))
)
# Submitted futures per batch = workers * mult (fewer batches → less throttle sleep overhead).
SCAN_BATCH_SIZE_MULT = max(2, int(os.getenv("SCAN_BATCH_SIZE_MULT", "2")))
# Inter-batch pause (rate-limit protection). Much lower than legacy 1–2s; spikes when Kraken/Alpaca 429.
SCAN_BATCH_SLEEP_STOCK_SEC = float(os.getenv("SCAN_BATCH_SLEEP_STOCK_SEC", "0.2"))
SCAN_BATCH_SLEEP_CRYPTO_SEC = float(os.getenv("SCAN_BATCH_SLEEP_CRYPTO_SEC", "0.06"))
SCAN_BATCH_SLEEP_RL_MID_SEC = float(os.getenv("SCAN_BATCH_SLEEP_RL_MID_SEC", "3.0"))
SCAN_BATCH_SLEEP_RL_HIGH_SEC = float(os.getenv("SCAN_BATCH_SLEEP_RL_HIGH_SEC", "10.0"))
_SCAN_EXECUTOR: Optional[ThreadPoolExecutor] = None

# Defensive assets — bought even in bear/risk-off markets (tend to RISE when stocks/crypto fall)
DEFENSIVE_ASSETS = {
    "GLD", "SLV", "TLT", "XLU", "XLP", "XLV",
    "BND", "VNQ", "IYR", "AGG",
}

# Exclude fiat FX pairs from crypto universe (e.g., AUD/USD, EUR/USD)
FIAT_BASES = {
    "USD", "USDT", "USDC", "EUR", "GBP", "AUD", "CAD", "JPY", "CHF", "NZD",
    "CNY", "HKD", "SGD", "SEK", "NOK", "MXN", "BRL", "ZAR", "TRY", "INR",
    "KRW", "PLN", "CZK", "DKK",
}

# Data freshness tracking
_kraken_last_candle_ts: Optional[float] = None
_alpaca_last_candle_ts: Optional[float] = None

# Stablecoins and pegged assets - never appear as buy picks (no price upside)
STABLECOINS = {"DAI", "BUSD", "TUSD", "USDP", "FRAX", "GUSD", "USDD", "LUSD", "sUSD", "CUSD"}
CRYPTO_BLOCKLIST_STABLECOINS = STABLECOINS

# Top 50 cryptocurrencies - expanded for better coverage
# Kraken uses XBT for Bitcoin; we include both BTC and XBT in bases so both pass.
_TOP_30_BASES_RAW = [
    "XBT", "ETH", "SOL", "XRP", "ADA", "AVAX", "DOGE", "DOT", "LINK", "MATIC",
    "UNI", "LTC", "ATOM", "BCH", "ALGO", "XLM", "ICP", "FIL", "VET", "SAND",
    "MANA", "AXS", "THETA", "EOS", "AAVE", "MKR", "SNX", "COMP", "YFI", "SUSHI",
    "NEAR", "APT", "ARB", "OP", "IMX", "GRT", "FTM", "CRV", "LDO", "RPL",
    "RUNE", "INJ", "TIA", "SEI", "SUI", "RENDER", "FET", "OCEAN", "AGIX", "TAO",
    "AKT", "CHZ", "KAS", "ENA", "BLUR", "PYTH",
]
TOP_30_CRYPTO = [f"{b}/USD" for b in _TOP_30_BASES_RAW]
# Include BTC as alias for XBT (e.g. BTC/USD from other exchanges)
TOP_30_CRYPTO_BASES = frozenset(b.upper() for b in _TOP_30_BASES_RAW) | {"BTC"}


def _crypto_base_from_symbol(sym: str) -> str:
    """Extract base from BTC/USD, XBT/USD, BTCUSD, BTC-USD, or plain BTC."""
    s = (sym or "").strip().upper()
    if not s:
        return ""
    if "/" in s:
        return (s.split("/")[0] or "").strip()
    if "-" in s:
        return (s.split("-")[0] or "").strip()
    for suffix in ("USD", "USDT", "USDC"):
        if s.endswith(suffix) and len(s) > len(suffix):
            return s[: -len(suffix)].strip()
    return s

# Min liquidity filters for picks (env overridable)
RECO_MIN_MARKET_CAP = float(os.getenv("RECO_MIN_MARKET_CAP", "50000000"))   # $50M
RECO_MIN_VOLUME_24H = float(os.getenv("RECO_MIN_VOLUME_24H", "100000"))    # $100K
RECO_MIN_SCORE_SHORT = float(os.getenv("RECO_MIN_SCORE_SHORT", "62"))
RECO_MIN_SCORE_MEDIUM = float(os.getenv("RECO_MIN_SCORE_MEDIUM", "58"))
RECO_MIN_SCORE_LONG = float(os.getenv("RECO_MIN_SCORE_LONG", "60"))

# Crypto bases never recommended (not on Kraken spot or problematic)
# STABLE: L1 token with long downtrend, poor profit potential; stablecoin-like names
# Extend via env: RECO_CRYPTO_BLOCKLIST=TOKEN1,TOKEN2,TOKEN3
# BLOCK_MEME_COINS=1 adds meme coins; DEGEN_MODE=1 disables meme blocking
_default_blocklist = {"STABLE", "UST", "USTC", "LUNA2", "LUNA"} | STABLECOINS  # downtrend / dead / misleading
_env_blocklist = os.getenv("RECO_CRYPTO_BLOCKLIST", "")
_meme_block = set()
if os.getenv("BLOCK_MEME_COINS", "0").strip().lower() in ("1", "true", "yes") and os.getenv("DEGEN_MODE", "0").strip().lower() not in ("1", "true", "yes"):
    _meme_block = {"SHIB", "PEPE", "FLOKI", "BONK", "WIF", "MEME", "BRETT", "POPCAT", "TURBO", "WOJAK", "BOME"}
CRYPTO_BLOCKLIST: set = _default_blocklist | _meme_block | {x.strip().upper() for x in _env_blocklist.split(",") if x.strip()}

INVALID_KRAKEN_BASES = frozenset({
    "HYPE", "SKY", "JUP", "WIF", "VIRTUAL", "PENGU",
    "ICNT", "BOME", "PEOPLE", "LUNC",
})

# Optional API protection (recommended for real money)
# If set, all /api/* routes require header: X-API-Key: <token>
WORKER_API_TOKEN = os.getenv("WORKER_API_TOKEN", "").strip()

# Rate limiting: max requests per window per IP for /api/* (0 = disabled)
RATE_LIMIT_REQUESTS = int(os.getenv("RATE_LIMIT_REQUESTS", "0"))
RATE_LIMIT_WINDOW_SEC = int(os.getenv("RATE_LIMIT_WINDOW_SEC", "60"))
_RATE_LIMIT_STORE: Dict[str, List[float]] = {}
_RATE_LIMIT_LOCK = threading.Lock()

RECO_SCAN_MAX_PER_RUN = int(os.getenv("RECO_SCAN_MAX_PER_RUN", "400"))
RECO_SCAN_ERROR_LIMIT = int(os.getenv("RECO_SCAN_ERROR_LIMIT", "15"))
RECO_SCAN_SYMBOL_SLEEP_SEC = float(os.getenv("RECO_SCAN_SYMBOL_SLEEP_SEC", "0.02"))

# Scan profile: conservative (stricter) | balanced | aggressive (looser). Affects thresholds when not overridden by env.
RECO_PROFILE = (os.getenv("RECO_PROFILE", "balanced") or "balanced").strip().lower()
if RECO_PROFILE not in ("conservative", "balanced", "aggressive"):
    RECO_PROFILE = "balanced"

def _reco_buy_threshold_stocks() -> float:
    v = os.getenv("RECO_BUY_THRESHOLD_STOCKS", "").strip()
    if v and v.replace(".", "").replace("-", "").isdigit():
        return float(v)
    # Balanced at 65: produces more high-quality opportunities while regime gate + RSI gate
    # still filter out low-conviction signals. Conservative keeps 72 for strict risk control.
    return {"conservative": 72.0, "balanced": 65.0, "aggressive": 55.0}.get(RECO_PROFILE, 65.0)

def _reco_buy_threshold_crypto() -> float:
    v = os.getenv("RECO_BUY_THRESHOLD_CRYPTO", "").strip()
    if v and v.replace(".", "").replace("-", "").isdigit():
        return float(v)
    # Balanced at 65: crypto signals align with stocks for consistent "All Assets" view.
    return {"conservative": 70.0, "balanced": 65.0, "aggressive": 55.0}.get(RECO_PROFILE, 65.0)

def _reco_watch_threshold() -> float:
    v = os.getenv("RECO_WATCH_THRESHOLD", "").strip()
    if v and v.replace(".", "").replace("-", "").isdigit():
        return float(v)
    return {"conservative": 52.0, "balanced": 48.0, "aggressive": 40.0}.get(RECO_PROFILE, 48.0)

# Legacy names for code that still references them
RECO_BUY_THRESHOLD_CRYPTO = _reco_buy_threshold_crypto()
RECO_BUY_THRESHOLD_STOCKS = _reco_buy_threshold_stocks()
RECO_WATCH_THRESHOLD = _reco_watch_threshold()

_ALLOWED_TFS = {"1m", "5m", "15m", "30m", "1h", "4h", "1d", "1w", "1M"}


_background_threads: Dict[str, Dict[str, Any]] = {}

def _start_background_thread(name: str, target) -> None:
    """Start a daemon thread only if not already started (avoids duplicate on hot reload)."""
    with _thread_start_lock:
        if _thread_started.get(name):
            logger.warning("thread %s already started, skipping", name)
            return
        _thread_started[name] = True
    def _wrapped():
        try:
            target()
        except Exception:
            logger.exception("background thread %s crashed — will be restarted by watchdog", name)
            with _thread_start_lock:
                _thread_started[name] = False
    t = threading.Thread(target=_wrapped, daemon=True, name=name)
    t.start()
    _background_threads[name] = {"thread": t, "target": target, "started_at": time.time()}
    logger.info("started background thread: %s", name)


def _fetch_fear_greed() -> Dict[str, Any]:
    """Fetch Crypto Fear & Greed Index. Returns {value, label, ts}."""
    global _FEAR_GREED_CACHE
    now = time.time()
    if now - _FEAR_GREED_CACHE.get("ts", 0) < _FEAR_GREED_TTL and _FEAR_GREED_CACHE.get("value") is not None:
        return _FEAR_GREED_CACHE
    try:
        import urllib.request
        req = urllib.request.Request("https://api.alternative.me/fng/?limit=1", headers={"User-Agent": "EirinBot/2.0"})
        with urllib.request.urlopen(req, timeout=5) as resp:
            data = json.loads(resp.read().decode())
        if data.get("data"):
            entry = data["data"][0]
            val = int(entry.get("value", 50))
            label = str(entry.get("value_classification", "Neutral"))
            _FEAR_GREED_CACHE = {"value": val, "label": label, "ts": now, "error": None}
            logger.info("Fear & Greed updated: %d (%s)", val, label)
    except Exception as e:
        logger.warning("Fear & Greed fetch failed: %s — using cached/default", e)
        _FEAR_GREED_CACHE["error"] = str(e)[:100]
    return _FEAR_GREED_CACHE


def _fear_greed_score_adjustment() -> float:
    """Return score adjustment based on Fear & Greed Index."""
    fg = _FEAR_GREED_CACHE.get("value", 50)
    if fg <= 25:
        return 4.0
    elif fg <= 45:
        return 2.0
    elif fg <= 55:
        return 0.0
    elif fg <= 75:
        return -3.0
    else:
        return -6.0


# News Sentiment Cache
_NEWS_CACHE: Dict[str, Dict[str, Any]] = {}
_NEWS_CACHE_TTL = 7200  # 2 hours

# Consecutive buy signal tracking: {symbol: {"count": int, "last_scan_ts": float}}
_CONSECUTIVE_BUY_TRACKER: Dict[str, Dict[str, Any]] = {}

# Sector momentum cache: updated after each stock scan
_SECTOR_MOMENTUM: Dict[str, Dict[str, Any]] = {}
# {sector: {"avg_return_5d": float, "stocks_count": int, "rank": int, "tier": "top"|"middle"|"bottom"}}


def _update_sector_momentum(items: list) -> None:
    """Recalculate sector momentum from scanned stock items."""
    global _SECTOR_MOMENTUM
    sector_returns: Dict[str, list] = {}
    for item in items:
        if item.get("market_type") != "stocks":
            continue
        sector = item.get("sector") or STOCK_SECTORS.get(str(item.get("symbol", "")).replace("/USD", "").upper(), "")
        if not sector or sector == "ETF":
            continue
        chg = item.get("change_24h")
        if chg is not None:
            sector_returns.setdefault(sector, []).append(float(chg))
    if len(sector_returns) < 4:
        return
    sector_avgs = {s: sum(r) / len(r) for s, r in sector_returns.items() if r}
    ranked = sorted(sector_avgs.items(), key=lambda x: -x[1])
    new_cache = {}
    for i, (sector, avg_ret) in enumerate(ranked):
        if i < 2:
            tier = "top"
        elif i >= len(ranked) - 2:
            tier = "bottom"
        else:
            tier = "middle"
        new_cache[sector] = {
            "avg_return_5d": round(avg_ret, 2),
            "stocks_count": len(sector_returns[sector]),
            "rank": i + 1,
            "total_sectors": len(ranked),
            "tier": tier,
        }
    _SECTOR_MOMENTUM = new_cache
    logger.info("Sector momentum updated: %d sectors — top: %s, bottom: %s",
                len(new_cache),
                [s for s, d in new_cache.items() if d["tier"] == "top"],
                [s for s, d in new_cache.items() if d["tier"] == "bottom"])

STOCK_SECTORS = {
    "AAPL": "Technology", "MSFT": "Technology", "GOOGL": "Technology", "AMZN": "Technology",
    "TSLA": "Consumer Cyclical", "NVDA": "Technology", "META": "Technology", "NFLX": "Communication",
    "AMD": "Technology", "INTC": "Technology", "QCOM": "Technology", "CRM": "Technology",
    "ADBE": "Technology", "AVGO": "Technology", "TXN": "Technology", "PLTR": "Technology",
    "ROKU": "Technology", "SHOP": "Technology", "PYPL": "Financial", "SQ": "Financial",
    "JPM": "Financial", "BAC": "Financial", "V": "Financial", "MA": "Financial",
    "WFC": "Financial", "GS": "Financial", "MS": "Financial", "BLK": "Financial", "C": "Financial", "AXP": "Financial",
    "COIN": "Financial", "MSTR": "Technology", "MARA": "Technology", "RIOT": "Technology",
    "HOOD": "Financial", "SOFI": "Financial", "AFRM": "Financial", "UPST": "Financial",
    "CLSK": "Technology", "HUT": "Technology", "BITF": "Technology",
    "JNJ": "Healthcare", "PFE": "Healthcare", "MRK": "Healthcare", "UNH": "Healthcare",
    "ABBV": "Healthcare", "LLY": "Healthcare", "PG": "Consumer Defensive",
    "DIS": "Communication", "KO": "Consumer Defensive", "PEP": "Consumer Defensive",
    "WMT": "Consumer Defensive", "TGT": "Consumer Cyclical", "COST": "Consumer Defensive",
    "HD": "Consumer Cyclical", "LOW": "Consumer Cyclical", "MCD": "Consumer Cyclical",
    "UBER": "Technology", "LYFT": "Technology", "DKNG": "Consumer Cyclical",
    "BA": "Industrials", "F": "Consumer Cyclical", "GM": "Consumer Cyclical",
    "XOM": "Energy", "CVX": "Energy", "T": "Communication", "VZ": "Communication",
    "GME": "Consumer Cyclical", "AMC": "Communication", "CVNA": "Consumer Cyclical",
    "SPY": "ETF", "QQQ": "ETF", "IWM": "ETF", "DIA": "ETF",
    "XLK": "ETF", "XLF": "ETF", "XLE": "ETF", "XLV": "ETF", "XLY": "ETF",
    "XLP": "ETF", "XLI": "ETF", "GLD": "ETF", "SLV": "ETF",
    "TQQQ": "ETF", "SQQQ": "ETF", "SOXL": "ETF", "ARKK": "ETF",
}

_NEGATIVE_STRONG = {"hack", "exploit", "crash", "ban", "fraud", "sec", "lawsuit", "bankrupt", "rug", "scam", "indictment"}
_NEGATIVE_MILD = {"bearish", "decline", "sell-off", "concern", "warning", "downgrade", "dumping", "plunge"}
_POSITIVE = {"partnership", "adoption", "etf", "upgrade", "launch", "bullish", "all-time high", "ath", "rally", "breakthrough"}


def _fetch_news_sentiment(symbol: str) -> Dict[str, Any]:
    """Fetch news sentiment for a symbol using CryptoCompare (free, no key needed)."""
    cache_key = symbol.upper()
    cached = _NEWS_CACHE.get(cache_key)
    if cached and (time.time() - cached.get("ts", 0)) < _NEWS_CACHE_TTL:
        return cached
    result = {"sentiment": "Neutral", "score_adj": 0, "headlines": 0, "ts": time.time()}
    try:
        base = symbol.split("/")[0].upper() if "/" in symbol else symbol.upper()
        if base == "XBT":
            base = "BTC"
        import urllib.request
        url = f"https://min-api.cryptocompare.com/data/v2/news/?lang=EN&categories={base}"
        req = urllib.request.Request(url, headers={"User-Agent": "EirinBot/3.0"})
        with urllib.request.urlopen(req, timeout=3) as resp:
            data = json.loads(resp.read().decode())
        articles = (data.get("Data") or [])[:5]
        result["headlines"] = len(articles)
        adj = 0
        for article in articles:
            title = (article.get("title") or "").lower()
            body = (article.get("body") or "")[:200].lower()
            text = title + " " + body
            for kw in _NEGATIVE_STRONG:
                if kw in text:
                    adj -= 8
                    break
            for kw in _NEGATIVE_MILD:
                if kw in text:
                    adj -= 3
                    break
            for kw in _POSITIVE:
                if kw in text:
                    adj += 3
                    break
        adj = max(-16, min(6, adj))
        if adj <= -6:
            result["sentiment"] = "Negative"
        elif adj >= 3:
            result["sentiment"] = "Positive"
        else:
            result["sentiment"] = "Neutral"
        result["score_adj"] = adj
    except Exception as e:
        logger.debug("News sentiment fetch failed for %s: %s", symbol, e)
    _NEWS_CACHE[cache_key] = result
    return result


def _fear_greed_loop() -> None:
    """Background loop: fetch Fear & Greed every 4 hours."""
    time.sleep(10)
    _fetch_fear_greed()
    while True:
        try:
            time.sleep(_FEAR_GREED_TTL)
            _fetch_fear_greed()
        except Exception as e:
            logger.warning("fear_greed_loop error: %s", e)
            time.sleep(300)


def _performance_outcomes_loop() -> None:
    """Background task: check open signal outcomes every 6 hours."""
    time.sleep(60)
    while True:
        try:
            from db import get_open_signal_outcomes, update_signal_outcome
            signals = get_open_signal_outcomes(limit=200)
            now = time.time()
            checked = 0
            wins_24 = 0
            wins_72 = 0
            losses_24 = 0
            losses_72 = 0
            for sig in signals:
                age_h = (now - (sig.get("created_ts") or 0)) / 3600.0
                if age_h < 24:
                    continue
                sym = sig.get("symbol") or ""
                entry_p = sig.get("entry_price")
                if not entry_p:
                    try:
                        m = json.loads(sig.get("metrics_json") or "{}")
                        entry_p = float(m.get("price") or 0)
                    except Exception:
                        entry_p = 0
                    if entry_p and entry_p > 0:
                        update_signal_outcome(sig["id"], entry_price=entry_p)
                if not entry_p or entry_p <= 0:
                    update_signal_outcome(sig["id"], outcome_checked=2)
                    continue
                current_price = None
                try:
                    if kc and "/" in sym and "USD" in sym:
                        current_price = kc.fetch_ticker_last(sym)
                    elif (alpaca_paper or alpaca_live):
                        ac = alpaca_paper or alpaca_live
                        ticker = sym.replace("/USD", "").replace("/", "")
                        current_price = float(ac.get_last_trade(ticker).price)
                except Exception:
                    pass
                if not current_price or current_price <= 0:
                    continue
                up = current_price > entry_p
                update_fields = {}
                if age_h >= 24 and sig.get("outcome_24h") is None:
                    update_fields["price_24h"] = current_price
                    update_fields["outcome_24h"] = "WIN" if up else "LOSS"
                    if up:
                        wins_24 += 1
                    else:
                        losses_24 += 1
                if age_h >= 72 and sig.get("outcome_72h") is None:
                    update_fields["price_72h"] = current_price
                    update_fields["outcome_72h"] = "WIN" if up else "LOSS"
                    update_fields["outcome_checked"] = 2
                    if up:
                        wins_72 += 1
                    else:
                        losses_72 += 1
                elif age_h >= 72:
                    update_fields["outcome_checked"] = 2
                elif age_h >= 24 and sig.get("outcome_24h") is not None:
                    update_fields["outcome_checked"] = 1
                if update_fields:
                    update_signal_outcome(sig["id"], **update_fields)
                    checked += 1
            if checked > 0:
                logger.info("Performance tracker: checked %d signals, 24h W/L %d/%d, 72h W/L %d/%d", checked, wins_24, losses_24, wins_72, losses_72)
        except Exception as e:
            logger.warning("performance_outcomes_loop error: %s", e)
        time.sleep(6 * 3600)


_THREAD_RESTART_COUNTS: Dict[str, int] = {}
_THREAD_LAST_ALIVE: Dict[str, float] = {}

# ---------------------------------------------------------------------------
# Background-loop health (Phase 1.2c)
# ---------------------------------------------------------------------------
# Per-loop health state surfaced in /health/full. Loops call
# _loop_health_ok(name)  on every successful iteration and
# _loop_health_err(name, exc)  on every iteration that raises. The watchdog
# stays in worker_api.py; this just gives the loop body a way to tell the
# operator that something is wrong without going silent.
#
# Schema: { name: { "last_ok_ts": float, "last_err_ts": float,
#                   "last_err": str, "consecutive_failures": int } }
_BACKGROUND_LOOP_HEALTH: Dict[str, Dict[str, Any]] = {}
_loop_health_lock = threading.Lock()


def _loop_health_ok(name: str) -> None:
    """Mark a background loop iteration as successful (resets failure streak)."""
    now = time.time()
    with _loop_health_lock:
        st = _BACKGROUND_LOOP_HEALTH.setdefault(name, {})
        st["last_ok_ts"] = now
        st["consecutive_failures"] = 0


def _loop_health_err(name: str, exc: BaseException) -> None:
    """Record a background loop iteration failure with the truncated repr."""
    now = time.time()
    err_repr = f"{type(exc).__name__}: {exc}"[:240]
    with _loop_health_lock:
        st = _BACKGROUND_LOOP_HEALTH.setdefault(name, {})
        st["last_err_ts"] = now
        st["last_err"] = err_repr
        st["consecutive_failures"] = int(st.get("consecutive_failures", 0)) + 1


def _thread_watchdog_loop() -> None:
    """Periodically check background threads, restart crashed ones, track failures."""
    while True:
        time.sleep(60)
        try:
            now = time.time()
            for name, info in list(_background_threads.items()):
                t = info.get("thread")
                if t and t.is_alive():
                    _THREAD_LAST_ALIVE[name] = now
                elif t and not t.is_alive():
                    # One-shot threads exit normally — skip restart
                    if name in _ONE_SHOT_THREADS:
                        continue
                    _THREAD_RESTART_COUNTS[name] = _THREAD_RESTART_COUNTS.get(name, 0) + 1
                    count = _THREAD_RESTART_COUNTS[name]
                    if count >= 5:
                        logger.error("WATCHDOG: %s has crashed %d times — persistent failure", name, count)
                    else:
                        logger.warning("WATCHDOG: %s is dead (restart #%d), restarting", name, count)
                    with _thread_start_lock:
                        _thread_started[name] = False
                    _start_background_thread(name, info["target"])

                # Check for stalled threads (alive but not producing output)
                last_alive = _THREAD_LAST_ALIVE.get(name, info.get("started_at", now))
                stale_sec = now - last_alive
                expected_intervals = {
                    "recommendations": 1800, "autopilot": 14400, "screener_outcomes": 3600,
                    "portfolio": 120, "health_watchdog": 120, "fear_greed": _FEAR_GREED_TTL,
                }
                expected = expected_intervals.get(name, 300)
                if stale_sec > expected * 2.5 and t and t.is_alive():
                    logger.warning("WATCHDOG: %s has not reported activity in %.0fs (expected every %ds)", name, stale_sec, expected)
        except Exception:
            logger.exception("thread_watchdog_loop: iteration failed")


# =========================================================
# Auth helper
# =========================================================
def _has_live_bots() -> bool:
    try:
        bots = list_bots()
        return any(not bool(b.get("dry_run", 1)) for b in (bots or []))
    except Exception:
        return False


def _alpaca_any_ready() -> bool:
    return bool(ALPACA_PAPER_READY or ALPACA_LIVE_READY)


def _bm_not_ready_reason() -> Optional[str]:
    """Return human-readable reason why BotManager is not ready (for 503 responses)."""
    if bm is not None:
        return None
    if not KRAKEN_READY and not _alpaca_any_ready():
        parts = []
        if not KRAKEN_READY:
            parts.append(f"Kraken: {KRAKEN_ERROR or 'API keys missing or init failed'}")
        if not _alpaca_any_ready() and os.getenv("ENABLE_ALPACA", "").strip().lower() in ("1", "true", "yes"):
            parts.append(f"Alpaca: {ALPACA_ERROR or 'API keys missing or init failed'}")
        return "; ".join(parts) if parts else "No exchange client (Kraken or Alpaca) available"
    err = _STARTUP_STATUS.get("last_startup_error")
    if err:
        return err
    return "BotManager initialization failed or not yet complete"


def _alpaca_market_open() -> bool:
    """Check if US stock market is open. Returns True on error (don't block)."""
    client = alpaca_live if alpaca_live else alpaca_paper
    if not client:
        return True
    try:
        return bool(client.get_clock().get("is_open", True))
    except Exception:
        return True


def _is_live_endpoint(path: str, method: str) -> bool:
    if not path.startswith("/api/"):
        return False
    if method not in ("POST", "PUT", "DELETE"):
        return False
    return False


def _require_api_key(
    path: str, api_key: Optional[str], client_host: Optional[str] = None, force: bool = False,
    referer: Optional[str] = None, host_header: Optional[str] = None,
) -> Tuple[bool, Optional[str], int]:
    if not WORKER_API_TOKEN:
        if force:
            return False, "Live endpoints disabled: WORKER_API_TOKEN is required.", 503
        return True, None, 200
    if not path.startswith("/api/"):
        return True, None, 200
    host = (client_host or "").strip()
    if not force and host in ("127.0.0.1", "::1", "localhost"):
        return True, None, 200
    # Same-origin bypass: allow when Referer/Host match (UI page fetching from own API)
    if not force and (referer or host_header):
        try:
            from urllib.parse import urlparse
            ref_host = urlparse(referer or "").netloc.split(":")[0] if referer else ""
            req_host = (host_header or "").split(":")[0].strip()
            allowed = os.getenv("API_ALLOWED_HOSTS", "localhost,127.0.0.1,::1,3.151.143.63")
            allowed_set = {h.strip().lower() for h in allowed.split(",") if h.strip()}
            for h in (ref_host, req_host):
                if h and h.lower() in allowed_set:
                    return True, None, 200
        except Exception:
            pass
    if not api_key or api_key.strip() != WORKER_API_TOKEN:
        return False, "Unauthorized (missing/invalid X-API-Key)", 401
    return True, None, 200


def _rate_limit_check(ip: str) -> Optional[str]:
    """Return error msg if rate limited, else None."""
    if RATE_LIMIT_REQUESTS <= 0:
        return None
    now = time.time()
    cutoff = now - RATE_LIMIT_WINDOW_SEC
    with _RATE_LIMIT_LOCK:
        timestamps = _RATE_LIMIT_STORE.get(ip, [])
        timestamps = [t for t in timestamps if t > cutoff]
        if len(timestamps) >= RATE_LIMIT_REQUESTS:
            return "Rate limit exceeded. Try again later."
        timestamps.append(now)
        _RATE_LIMIT_STORE[ip] = timestamps[-500:]  # cap memory
    return None


@app.middleware("http")
async def api_key_middleware(request: Request, call_next):
    path = request.url.path
    client_host = request.client.host if request.client else ""
    if path.startswith("/api/") and RATE_LIMIT_REQUESTS > 0:
        err = _rate_limit_check(client_host)
        if err:
            return _json({"ok": False, "error": err}, 429)
    api_key = request.headers.get("X-API-Key")
    referer = request.headers.get("Referer")
    host_header = request.headers.get("Host")
    ok, msg, status = _require_api_key(
        path, api_key, client_host=client_host, referer=referer, host_header=host_header
    )
    if not ok:
        return _json({"ok": False, "error": msg or "Unauthorized"}, status)
    return await call_next(request)


@app.post("/api/bots")
async def api_create_bot(request: Request):
    """Single handler for bot creation. Validates symbol, applies caps, returns full bot."""
    if not bm:
        reason = _bm_not_ready_reason() or "BotManager not initialized"
        return _json({"ok": False, "error": "BotManager not initialized", "reason": reason}, 503)
    payload = await request.json()
    if not isinstance(payload, dict):
        return _json({"ok": False, "error": "Invalid payload"}, 400)

    raw_sym = str(payload.get("symbol") or "").strip()
    detected_type = classify_symbol(raw_sym) if raw_sym else "crypto"
    market_type_val = "stocks" if detected_type == "stock" else "crypto"
    if market_type_val == "crypto" and raw_sym:
        resolved, err = _validate_crypto_symbol(raw_sym)
        if err:
            return _json({"ok": False, "error": err}, 400)
        raw_sym = resolved or raw_sym
    symbol = _resolve_symbol(raw_sym)
    name = str(payload.get("name") or f"Bot {symbol}")

    base_quote = float(payload.get("base_quote") or 20.0)
    safety_quote = float(payload.get("safety_quote") or 20.0)
    max_safety = int(payload.get("max_safety") or 5)
    max_spend_quote = float(payload.get("max_spend_quote") or (base_quote + (safety_quote * max_safety)))
    if base_quote > max_spend_quote * 0.5 or base_quote > 100:
        base_quote = min(max(5.0, max_spend_quote * 0.15), 100.0)
    if safety_quote > max_spend_quote * 0.3 or safety_quote > 75:
        safety_quote = min(max(5.0, max_spend_quote * 0.10), 75.0)

    if payload.get("base_order_quote") is not None:
        try:
            base_quote = float(payload.get("base_order_quote"))
        except (TypeError, ValueError):
            pass

    data = {
        "name": name,
        "symbol": symbol,
        "enabled": int(payload.get("enabled", 1)),
        "dry_run": int(payload.get("dry_run", 1)) if "dry_run" in payload else 1,
        "base_quote": base_quote,
        "safety_quote": safety_quote,
        "max_safety": max_safety,
        "first_dev": float(payload.get("first_dev") or 0.015),
        "step_mult": float(payload.get("step_mult") or 1.2),
        "tp": float(payload.get("tp") or 0.015),
        "max_spend_quote": max_spend_quote,
        "strategy_mode": str(payload.get("strategy_mode") or "auto"),
        "forced_strategy": str(payload.get("forced_strategy") or ""),
        "max_open_orders": int(payload.get("max_open_orders") or 6),
        "market_type": market_type_val,
        "alpaca_mode": str(payload.get("alpaca_mode") or "paper"),
        "max_total_exposure_pct": float(payload.get("max_total_exposure_pct") or 0.50),
        "per_symbol_exposure_pct": float(payload.get("per_symbol_exposure_pct") or payload.get("per_symbol_pct") or 0.15),
    }
    for _k in ("stop_loss_pct", "max_hold_hours"):
        if _k in payload:
            data[_k] = payload[_k]
    _sanitize_bot_numbers(data)
    try:
        from bot_config_validator import validate_bot_config
        data, validation_issues = validate_bot_config(data)
        fatal = [i for i in validation_issues if i.startswith("ERROR:")]
        if fatal:
            return _json({"ok": False, "error": fatal[0], "validation_issues": validation_issues}, 400)
    except ImportError:
        validation_issues = []
    from services.exposure_cap import build_exposure_cap_error

    pv_cap = _portfolio_value_usd_for_exposure()
    cap_err = build_exposure_cap_error(
        pv_cap,
        float(data.get("base_quote") or 0),
        float(data.get("per_symbol_exposure_pct") or 0),
        float(data.get("max_total_exposure_pct") or 0.5),
    )
    if cap_err:
        return _json(cap_err, 422)
    try:
        bot_id = create_bot(data)
        sl_a = data.get("stop_loss_pct")
        mh_a = data.get("max_hold_hours")
        if sl_a is not None or mh_a is not None:
            try:
                patch_bot_risk_after_create(
                    int(bot_id),
                    stop_loss_pct=float(sl_a) if sl_a is not None else None,
                    max_hold_hours=int(mh_a) if mh_a is not None else None,
                )
            except Exception as _risk_patch_e:
                logger.warning("patch_bot_risk_after_create failed bot_id=%s: %s", bot_id, _risk_patch_e)
        bot = get_bot(int(bot_id))
        resp: Dict[str, Any] = {"ok": True, "bot": bot}
        if validation_issues:
            resp["validation_warnings"] = validation_issues
        return _json(resp)
    except Exception as e:
        return _json({"ok": False, "error": str(e)}, 500)


@app.get("/api/bots")
def api_bots():
    try:
        bots = list_bots()
        if bm is not None:
            for b in bots:
                bot_id = int(b.get("id") or 0)
                try:
                    snap = bm.snapshot(bot_id)
                    if snap:
                        is_running = bool(snap.get("running"))
                        has_pos = float(snap.get("base_pos") or 0) > 0
                        risk = snap.get("risk_state")
                        if risk:
                            state_label = f"RISK_{risk}"
                        elif has_pos:
                            state_label = "MANAGING_POSITION"
                        elif is_running:
                            state_label = "WAITING_FOR_SIGNAL"
                        else:
                            state_label = "STOPPED"
                        b["state"] = state_label
                        b["last_event"] = snap.get("last_event")
                        b["last_tick_ts"] = snap.get("last_tick_ts")
                        b["unrealized_pnl_pct"] = snap.get("unrealized_pnl_pct")
                        b["decision_action"] = snap.get("decision_action")
                    else:
                        b["state"] = "STOPPED" if not int(b.get("enabled", 0)) else "UNKNOWN"
                except Exception:
                    b["state"] = "UNKNOWN"
                try:
                    od = latest_open_deal(bot_id)
                    b["open_deal"] = od
                except Exception:
                    b["open_deal"] = None
        return _json({
            "ok": True,
            "bots": bots,
            "kraken_ready": _kraken_ready(),
            "kraken_error": KRAKEN_ERROR,
        })
    except Exception as e:
        logger.exception("api_bots failed")
        return _json({"ok": False, "error": str(e), "bots": []}, 500)


@app.get("/api/bots/stream")
async def api_bots_stream():
    """SSE endpoint for real-time bot status updates (11.md Part 7)."""
    import asyncio

    async def event_gen():
        while True:
            try:
                bots = list_bots()
                data = json.dumps(bots)
                yield f"data: {data}\n\n"
            except Exception as e:
                logger.debug("bots/stream: %s", e)
            await asyncio.sleep(5)

    return StreamingResponse(
        event_gen(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


def _scheduled_db_cleanup() -> None:
    """
    DB maintenance: prune high-growth tables every 6 hours.
    Runs immediately on startup (no initial sleep).
    """
    while True:
        try:
            from db import db_maintenance_cleanup
            results = db_maintenance_cleanup()
            total = sum(v for v in results.values() if isinstance(v, int) and v > 0)
            if total > 0:
                logger.info("[DB_CLEANUP] Pruned %d rows: %s", total, results)
        except Exception as e:
            logger.warning("[DB_CLEANUP] Failed: %s", e)
        time.sleep(21600)  # Every 6 hours


def _prewarm_bots_summary() -> None:
    """Pre-warm /api/bots/summary cache 5 s after startup so the first browser request is instant."""
    time.sleep(5)
    try:
        bots = list_bots()
        running = sum(1 for b in bots if int(b.get("last_running", 0)) == 1)
        result = {"ok": True, "total": len(bots), "running": running, "paused": bool(_pause_state())}
        _BOTS_SUMMARY_CACHE["result"] = result
        _BOTS_SUMMARY_CACHE["ts"] = time.time()
        logger.info("bots_summary pre-warm: %d total, %d running", len(bots), running)
    except Exception as e:
        logger.warning("bots_summary pre-warm failed: %s", e)


@app.get("/api/bots/summary")
def api_bots_summary():
    now = time.time()
    if _BOTS_SUMMARY_CACHE["result"] is not None and now - _BOTS_SUMMARY_CACHE["ts"] < _BOTS_SUMMARY_TTL:
        return _json(_BOTS_SUMMARY_CACHE["result"])
    try:
        bots = list_bots()
    except Exception as e:
        logger.exception("api_bots_summary list_bots failed")
        return _json({"ok": False, "error": str(e), "total": 0, "running": 0, "paused": False}, 500)
    total = len(bots)
    running = 0
    if bm is not None:
        # Acquire with timeout so a contended lock during cold-start never stalls the response
        _lock_acquired = bm._lock.acquire(timeout=1.5)
        if _lock_acquired:
            try:
                running = sum(1 for r in bm._bots.values() if getattr(r.state, "running", False))
            finally:
                bm._lock.release()
        else:
            # Lock busy (startup race) — fall back to last_running flag persisted in DB
            running = sum(1 for b in bots if int(b.get("last_running", 0)) == 1)
    paused = bool(_pause_state())
    result = {"ok": True, "total": total, "running": running, "paused": paused}
    _BOTS_SUMMARY_CACHE["result"] = result
    _BOTS_SUMMARY_CACHE["ts"] = now
    return _json(result)


@app.get("/api/bots/export")
def api_bots_export():
    """Export all bot configurations as JSON array."""
    try:
        bots = list_bots()
        # Remove internal fields and timestamps for cleaner export
        export_list = []
        for bot in bots:
            export_bot = dict(bot)
            # Keep essential config but remove runtime/internal fields
            export_bot.pop("id", None)
            export_bot.pop("created_at", None)
            export_list.append(export_bot)

        return _json({"ok": True, "bots": export_list, "count": len(export_list)})
    except Exception as e:
        logger.exception("api_bots_export failed")
        return _json({"ok": False, "error": str(e), "bots": []}, 500)


@app.post("/api/bots/import")
async def api_bots_import(request: Request):
    """Import bot configurations from JSON array."""
    try:
        body = await request.json() or {}
        bots_to_import = body.get("bots") or []

        if not isinstance(bots_to_import, list):
            return _json({"ok": False, "error": "Expected 'bots' as array"}, 400)

        created = []
        failed = []

        for bot_config in bots_to_import:
            try:
                # Sanitize numeric fields
                _sanitize_bot_numbers(bot_config)
                # Ensure disabled state for import
                bot_config["enabled"] = 0
                # Create the bot
                new_id = create_bot(bot_config)
                created.append({"id": new_id, "name": bot_config.get("name")})
            except Exception as e:
                failed.append({"name": bot_config.get("name"), "error": str(e)})
                logger.warning("Import bot failed: %s", e)

        try:
            _discord_notify(f"📥 Imported {len(created)} bot(s)")
        except Exception:
            pass

        return _json({
            "ok": True,
            "created": created,
            "failed": failed,
            "total_imported": len(created)
        })
    except Exception as e:
        logger.exception("api_bots_import failed")
        return _json({"ok": False, "error": str(e)}, 500)


# =========================================================
# Helpers
# =========================================================

def _portfolio_value_usd_for_exposure() -> float:
    """Live aggregate portfolio (USD) for per-symbol cap checks on bot save."""
    try:
        if bm:
            v = float(bm.get_portfolio_total())
            if math.isfinite(v) and v > 0:
                return v
    except Exception:
        pass
    return 0.0


def _sanitize_bot_numbers(data: Dict[str, Any]) -> None:
    """Clamp critical bot numeric fields to sane ranges (mutates data in place)."""
    def clamp(key: str, lo: float, hi: float, default: float) -> None:
        if key not in data:
            return
        try:
            v = float(data[key])
        except (TypeError, ValueError):
            data[key] = default
            return
        data[key] = round(max(lo, min(hi, v)), 8)

    _round_keys = ("base_quote", "safety_quote", "max_spend_quote", "vol_gap_mult",
                    "tp_vol_mult", "regime_switch_threshold", "max_total_exposure_pct",
                    "per_symbol_exposure_pct", "min_free_cash_pct", "hard_sl_pct")
    for k in _round_keys:
        if k in data:
            try:
                data[k] = round(float(data[k]), 8)
            except (TypeError, ValueError):
                pass

    clamp("first_dev", 0.001, 0.5, 0.015)
    clamp("step_mult", 1.0, 10.0, 1.2)
    clamp("tp", 0.001, 0.5, 0.015)
    clamp("daily_loss_limit_pct", 0.01, 0.25, 0.06)
    clamp("max_drawdown_pct", 0.0, 0.99, 0.0)
    clamp("trailing_activation_pct", 0.001, 0.5, 0.02)
    clamp("trailing_distance_pct", 0.001, 0.2, 0.01)
    clamp("spread_guard_pct", 0.0005, 0.05, 0.003)
    clamp("min_gap_pct", 0.001, 0.1, 0.003)
    clamp("max_gap_pct", 0.01, 0.3, 0.06)
    if "max_safety" in data:
        try:
            v = int(data["max_safety"])
            data["max_safety"] = max(1, min(20, v))
        except (TypeError, ValueError):
            data["max_safety"] = 5
    if "poll_seconds" in data:
        try:
            v = int(data["poll_seconds"])
            data["poll_seconds"] = max(5, min(300, v))
        except (TypeError, ValueError):
            data["poll_seconds"] = 10
    clamp("stop_loss_pct", 0.01, 0.50, 0.08)
    if "max_hold_hours" in data:
        try:
            v = int(float(data["max_hold_hours"]))
            data["max_hold_hours"] = max(0, min(2160, v))
        except (TypeError, ValueError):
            data["max_hold_hours"] = 0


def _kraken_ready() -> bool:
    return bool(kc is not None and bm is not None and KRAKEN_READY)


def _json(payload: Dict[str, Any], status_code: int = 200) -> JSONResponse:
    return JSONResponse(payload, status_code=status_code, headers={'Cache-Control': 'no-store'})


def _serialize_order(o: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "id": o.get("id"),
        "symbol": o.get("symbol"),
        "side": o.get("side"),
        "type": o.get("type"),
        "price": o.get("price"),
        "amount": o.get("amount"),
        "remaining": o.get("remaining"),
        "status": o.get("status"),
        "timestamp": o.get("timestamp"),
        "client_order_id": o.get("clientOrderId") or o.get("client_id"),
    }


def _midnight_local_ts() -> int:
    lt = time.localtime()
    return int(time.mktime((lt.tm_year, lt.tm_mon, lt.tm_mday, 0, 0, 0, lt.tm_wday, lt.tm_yday, lt.tm_isdst)))


def _markets() -> Dict[str, Any]:
    if not _kraken_ready():
        return {}
    now = time.time()
    with _globals_lock:
        cached = _MARKETS_CACHE.get("markets")
        if cached is None or (now - float(_MARKETS_CACHE.get("ts", 0.0))) > MARKETS_TTL_SEC:
            try:
                _MARKETS_CACHE["markets"] = kc.load_markets()
                _MARKETS_CACHE["ts"] = now
            except Exception as e:
                logger.exception("_markets: load_markets failed")
                if _MARKETS_CACHE["markets"] is None:
                    _MARKETS_CACHE["markets"] = {}
                _MARKETS_CACHE["ts"] = now
        return _MARKETS_CACHE["markets"] or {}


def _strategy_display_name(mode: Optional[str]) -> str:
    """Map internal strategy_mode to Explore-friendly label."""
    if not mode or not str(mode).strip():
        return "DCA"
    m = str(mode).strip().lower()
    if m == "trend_follow":
        return "Trend Follow"
    if m == "range_mean_reversion":
        return "Range / Mean Reversion"
    if m == "high_vol_defensive":
        return "High Vol Defensive"
    if m in ("smart_dca", "smart dca"):
        return "Smart DCA"
    if m in ("classic_dca", "classic", "dca"):
        return "DCA"
    return mode.replace("_", " ").title()


def _normalize_symbol(user_symbol: str) -> str:
    """
    Accepts:
      btcusd, BTCUSD, BTC-USD, BTC/USD, xbtusd, XBTUSD, etc.
    Produces:
      BTC/USD, ETH/USD, etc. (ccxt Kraken now uses BTC not XBT)
    Note: XBT/USD input is left as XBT/USD and _resolve_symbol() handles the BTC↔XBT swap.
    """
    s = (user_symbol or "").strip().upper().replace(" ", "")
    s = s.replace("-", "/")
    if "/" not in s and len(s) >= 6:
        s = f"{s[:-3]}/{s[-3:]}"
    parts = s.split("/", 1)
    base = parts[0] if len(parts) > 0 else ""
    quote = parts[1] if len(parts) > 1 else ""
    # Do NOT convert BTC->XBT here; _resolve_symbol handles market resolution
    return f"{base}/{quote}" if base and quote else s


def _resolve_symbol(symbol_from_db_or_user: str) -> str:
    """
    Resolve normalized symbol to a valid Kraken market key when possible.
    For stock-like symbols (short, no slash), return as-is without Kraken validation.
    """
    s = _normalize_symbol(symbol_from_db_or_user or "")
    
    # Stock symbols: short (< 6 chars) and no slash -> return as-is, skip Kraken check
    if len(s) < 6 and "/" not in s:
        return s
    
    mk = _markets()
    if not mk:
        return s

    if s in mk:
        return s

    if s.startswith("BTC/"):
        alt = s.replace("BTC/", "XBT/", 1)
        if alt in mk:
            return alt
    if s.startswith("XBT/"):
        alt = s.replace("XBT/", "BTC/", 1)
        if alt in mk:
            return alt

    return s


def _validate_crypto_symbol(symbol: str) -> tuple:
    """
    Validate crypto symbol against Kraken markets.
    Returns (resolved_symbol, None) if valid, or (None, error_message) if invalid.
    """
    s = _normalize_symbol(symbol or "")
    if not s or "/" not in s:
        return None, "Symbol must be in format BASE/USD (e.g. XBT/USD, ETH/USD)"
    mk = _markets()
    if not mk:
        return s, None  # Skip validation if Kraken not ready
    if s in mk:
        return s, None
    if s.startswith("BTC/"):
        alt = s.replace("BTC/", "XBT/", 1)
        if alt in mk:
            return alt, None
    if s.startswith("XBT/"):
        alt = s.replace("XBT/", "BTC/", 1)
        if alt in mk:
            return alt, None
    # Build suggestions from popular Kraken USD pairs
    popular = ["BTC/USD", "ETH/USD", "SOL/USD", "AVAX/USD", "LINK/USD", "DOT/USD"]
    avail = [k for k in mk if str(mk.get(k, {}).get("quote", "")).upper() == "USD"][:8]
    suggestions = [p for p in popular if p in mk] or (avail[:5] if avail else ["BTC/USD", "ETH/USD"])
    return None, f"Symbol not found on Kraken: {s}. Try: {', '.join(suggestions)}"


def _sanitize_tf(tf: str) -> str:
    t = (tf or "5m").strip()
    return t if t in _ALLOWED_TFS else "5m"


def _tf_seconds(tf: str) -> int:
    if tf.endswith("m"):
        return int(tf[:-1]) * 60
    if tf.endswith("h"):
        return int(tf[:-1]) * 3600
    if tf.endswith("d"):
        return int(tf[:-1]) * 86400
    if tf.endswith("w"):
        return int(tf[:-1]) * 604800
    if tf.endswith("M"):
        return int(tf[:-1]) * 2592000
    return 300


def _safe_last_price(symbol: str) -> Optional[float]:
    """
    Get last price for a CRYPTO symbol from Kraken.
    
    GUARDRAIL: This function is for CRYPTO symbols only.
    Use AlpacaClient.get_latest_quote() for stock symbols.
    """
    # Guardrail: Prevent stock symbols from being routed to Kraken
    validate_symbol_type(symbol, "crypto", "_safe_last_price")
    
    if not _kraken_ready():
        return None
    try:
        s = _resolve_symbol(symbol)
        mk = _markets()
        if mk and s in mk:
            return float(kc.fetch_ticker_last(s))
        return None
    except Exception:
        logger.exception("_safe_last_price: fetch failed symbol=%s", symbol)
        return None


def _ticker_cached(symbol: str, ttl_sec: int = 30) -> Optional[Dict[str, Any]]:
    now = time.time()
    with _globals_lock:
        cached = _TICKERS_CACHE.get(symbol)
        if cached and (now - float(cached.get("ts", 0.0))) < ttl_sec:
            return cached.get("data")
        cached_save = cached

    data = None
    try:
        if _kraken_ready():
            mk = _markets()
            if (mk and symbol in mk) or "/" in symbol:
                t = kc.fetch_ticker(symbol)
                last_raw = float(t.get("last") or t.get("close") or 0)
                data = {
                    "symbol": symbol,
                    "last": last_raw,
                    "bid": t.get("bid"), "ask": t.get("ask"),
                    "percentage": t.get("percentage"), "change": t.get("change"),
                    "baseVolume": t.get("baseVolume"), "quoteVolume": t.get("quoteVolume"),
                }
                try:
                    from data_validator import get_validated_crypto_price
                    validated, alert = get_validated_crypto_price(last_raw, symbol, kc)
                    if validated > 0:
                        data["last"] = validated
                    if alert:
                        logger.warning("Price divergence %s >2%%", symbol)
                except ImportError:
                    pass
    except Exception as e:
        logger.exception("_ticker_cached: Kraken fetch failed symbol=%s", symbol)
        try:
            from circuit_breaker import record_api_failure, check_and_trigger_emergency
            record_api_failure(bot_id=None, source="kraken_ticker")
            check_and_trigger_emergency()
        except ImportError:
            pass

    if not data and (alpaca_paper or alpaca_live):
        try:
            client = alpaca_live if alpaca_live else alpaca_paper
            snaps = client.get_snapshots([symbol])
            snap = snaps.get(symbol)
            if snap:
                daily = snap.get("dailyBar") or {}
                prev = snap.get("prevDailyBar") or {}
                latest = snap.get("latestTrade") or {}
                last_price = float(latest.get("p") or daily.get("c") or 0.0)
                prev_close = float(prev.get("c") or 0.0)
                change = last_price - prev_close if last_price and prev_close else 0.0
                pct = (change / prev_close) * 100.0 if prev_close else 0.0
                vol_shares = float(daily.get("v") or 0.0)
                data = {
                    "symbol": symbol, "last": last_price,
                    "bid": float(snap.get("latestQuote", {}).get("bp") or 0.0),
                    "ask": float(snap.get("latestQuote", {}).get("ap") or 0.0),
                    "percentage": pct, "change": change,
                    "baseVolume": vol_shares, "quoteVolume": vol_shares * last_price,
                }
            else:
                t = client.get_ticker(symbol)
                if t.get("last"):
                    data = {"symbol": symbol, "last": t.get("last"), "bid": t.get("bid"),
                            "ask": t.get("ask"), "percentage": 0.0, "change": 0.0,
                            "baseVolume": 0.0, "quoteVolume": 0.0}
        except Exception as e:
            logger.exception("_ticker_cached: Alpaca fetch failed symbol=%s", symbol)
            try:
                from circuit_breaker import record_api_failure, check_and_trigger_emergency
                record_api_failure(bot_id=None, source="alpaca_ticker")
                check_and_trigger_emergency()
            except ImportError:
                pass

    with _globals_lock:
        if data:
            bid = float(data.get("bid") or 0)
            ask = float(data.get("ask") or 0)
            last = float(data.get("last") or 0)
            mid = (bid + ask) / 2 if bid and ask else last
            if mid > 0 and bid and ask:
                spread_pct = abs(ask - bid) / mid * 100
                if spread_pct > 2.0:
                    try:
                        from db import log_data_quality
                        log_data_quality("ticker", "extreme_spread", "warning", {"symbol": symbol, "spread_pct": round(spread_pct, 2)})
                    except Exception:
                        pass
            _TICKERS_CACHE[symbol] = {"ts": now, "data": data}
            return data
        if cached_save:
            cached_ts = float(cached_save.get("ts", 0))
            if now - cached_ts > 300:
                try:
                    from db import log_data_quality
                    log_data_quality("ticker", "stale_price", "warning", {"symbol": symbol, "age_sec": int(now - cached_ts)})
                except Exception:
                    pass
            return cached_save.get("data")
    return None


_TICKERS_BATCH_CACHE: Dict[str, Any] = {"ts": 0.0, "data": {}}

# One SPY / BTC daily series per scan wave — avoids hundreds of identical Alpaca/Kraken calls.
_BENCHMARK_1D_CACHE: Dict[str, Any] = {}


def _benchmark_ohlcv_cached(cache_key: str, ttl_sec: float, fetch_fn) -> List[List[float]]:
    """Fetch benchmark OHLCV once per TTL (shared across all symbols in a scan)."""
    now = time.time()
    with _globals_lock:
        ent = _BENCHMARK_1D_CACHE.get(cache_key)
        if ent and (now - float(ent.get("ts", 0.0))) < ttl_sec:
            return list(ent.get("rows") or [])
    rows: List[List[float]] = []
    try:
        rows = fetch_fn() or []
    except Exception:
        rows = []
    with _globals_lock:
        _BENCHMARK_1D_CACHE[cache_key] = {"ts": now, "rows": rows}
    return rows


def _tickers_batch_cached(ttl_sec: int = 15) -> Dict[str, Dict[str, Any]]:
    """Fetch all Kraken tickers once, cache briefly. Use for batch price lookups (Explore)."""
    now = time.time()
    with _globals_lock:
        if (now - _TICKERS_BATCH_CACHE.get("ts", 0.0)) < ttl_sec and _TICKERS_BATCH_CACHE.get("data"):
            return _TICKERS_BATCH_CACHE["data"]
    out: Dict[str, Dict[str, Any]] = {}
    try:
        if _kraken_ready() and kc:
            try:
                raw = kc.ex.fetch_tickers()
            except Exception as _ft_err:
                if "safeMarket" in str(_ft_err) or "disambiguate" in str(_ft_err):
                    logger.warning("_tickers_batch_cached: safeMarket error (ccxt), trying individual pairs: %s", _ft_err)
                    raw = {}
                else:
                    raise
            for sym, t in (raw or {}).items():
                if not isinstance(t, dict):
                    continue
                out[sym] = {"symbol": sym, "last": t.get("last") or t.get("close"),
                            "percentage": t.get("percentage"), "quoteVolume": t.get("quoteVolume")}
            with _globals_lock:
                _TICKERS_BATCH_CACHE["ts"] = now
                _TICKERS_BATCH_CACHE["data"] = out
    except Exception as e:
        logger.warning("_tickers_batch_cached: fetch failed: %s", e)
    return out


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


_OHLCV_FETCH_LOCKS: Dict[str, threading.Lock] = {}
_OHLCV_FETCH_LOCKS_LOCK = threading.Lock()


def _scan_ohlcv_get(symbol: str, timeframe: str, limit: int) -> Optional[List]:
    """Return cached OHLCV if still within the 2hr scan TTL, else None."""
    key = f"{symbol}|{timeframe}|{limit}"
    with _globals_lock:
        cached = _RECO_OHLCV_CACHE.get(key)
    if cached and (time.time() - float(cached.get("ts", 0))) < _SCAN_OHLCV_TTL:
        return cached.get("data") or []
    return None


def _scan_ohlcv_put(symbol: str, timeframe: str, limit: int, data: list) -> None:
    """Write fetched candles to _RECO_OHLCV_CACHE so medium/long horizons reuse them."""
    if not data:
        return
    key = f"{symbol}|{timeframe}|{limit}"
    with _globals_lock:
        _RECO_OHLCV_CACHE[key] = {"ts": time.time(), "data": data}


def _ohlcv_cached(symbol: str, timeframe: str, limit: int, ttl_sec: int) -> List[List[float]]:
    """
    Fetch OHLCV data for a CRYPTO symbol from Kraken with caching.
    Thread-safe: uses per-key locks to prevent duplicate API calls
    when multiple threads request the same data simultaneously.
    
    GUARDRAIL: This function is for CRYPTO symbols only.
    Use AlpacaClient.get_ohlcv() for stock symbols.
    """
    validate_symbol_type(symbol, "crypto", "_ohlcv_cached")

    if not _kraken_ready():
        return []
    key = f"{symbol}|{timeframe}|{limit}"
    now = time.time()
    with _globals_lock:
        cached = _RECO_OHLCV_CACHE.get(key)
        if cached and (now - float(cached.get("ts", 0.0))) < ttl_sec:
            return cached.get("data") or []

    with _OHLCV_FETCH_LOCKS_LOCK:
        if key not in _OHLCV_FETCH_LOCKS:
            _OHLCV_FETCH_LOCKS[key] = threading.Lock()
        fetch_lock = _OHLCV_FETCH_LOCKS[key]

    with fetch_lock:
        now = time.time()
        with _globals_lock:
            cached = _RECO_OHLCV_CACHE.get(key)
            if cached and (now - float(cached.get("ts", 0.0))) < ttl_sec:
                return cached.get("data") or []

        data = []
        try:
            is_crypto = "/" in symbol
            if not is_crypto and (alpaca_live or alpaca_paper):
                client = alpaca_live if alpaca_live else alpaca_paper
                data = client.get_ohlcv(symbol, timeframe=timeframe, limit=limit)
            if not data and kc:
                data = kc.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)
        except Exception:
            logger.exception("_ohlcv_cached: fetch failed symbol=%s tf=%s", symbol, timeframe)

        with _globals_lock:
            _RECO_OHLCV_CACHE[key] = {"ts": now, "data": data}
        return data


def _safe_spread_pct(symbol: str) -> Optional[float]:
    try:
        t = kc.fetch_ticker(symbol)
        bid = float(t.get("bid") or 0.0)
        ask = float(t.get("ask") or 0.0)
        mid = (bid + ask) / 2 if bid and ask else 0.0
        if mid <= 0:
            return None
        return float((ask - bid) / mid)
    except Exception:
        return None


def _btc_context() -> Dict[str, Any]:
    ctx = {"risk_off": False, "labels": {}, "scores": {}, "return_30d": -0.20, "fear_greed": 50}
    try:
        ctx["fear_greed"] = int(_FEAR_GREED_CACHE.get("value") or 50)
    except Exception:
        pass
    try:
        sym = _resolve_symbol("XBT/USD")
        c4h = _ohlcv_cached(sym, "4h", 200, 300)
        c1d = _ohlcv_cached(sym, "1d", 400, 900)
        r4h = detect_regime(c4h)
        r1d = detect_regime(c1d)
        ctx["labels"] = {"4h": r4h.regime, "1d": r1d.regime}
        ctx["scores"] = {"4h": r4h.scores or {}, "1d": r1d.scores or {}}
        ctx["regime"] = r1d.regime
        ctx["regime_label"] = r1d.regime
        down = max((r1d.scores or {}).get("downtrend_score", 0.0), (r4h.scores or {}).get("downtrend_score", 0.0))
        hv = max((r1d.scores or {}).get("high_vol_score", 0.0), (r4h.scores or {}).get("high_vol_score", 0.0))
        if (down >= 0.8 and hv >= 0.5) or r1d.regime in ("HIGH_VOL_RISK",):
            ctx["risk_off"] = True
        ctx["btc_down"] = down
        ctx["btc_hv"] = hv
        if c1d and len(c1d) >= 31:
            btc_now = float(c1d[-1][4])
            btc_30d_ago = float(c1d[-31][4])
            if btc_30d_ago > 0:
                ctx["return_30d"] = (btc_now - btc_30d_ago) / btc_30d_ago
        ctx["fear_greed_value"] = int(_FEAR_GREED_CACHE.get("value") or 50)
        if c1d and len(c1d) >= 21:
            ctx["closes"] = [float(row[4]) for row in c1d[-60:] if len(row) >= 5]
    except Exception:
        pass
    return ctx


def _evict_ohlcv_cache() -> int:
    """Evict stale entries from OHLCV cache to prevent memory leak."""
    now = time.time()
    evicted = 0
    with _globals_lock:
        stale_keys = [
            k for k, v in _RECO_OHLCV_CACHE.items()
            if (now - float(v.get("ts", 0))) > _OHLCV_CACHE_EVICT_AGE_SEC
        ]
        for k in stale_keys:
            del _RECO_OHLCV_CACHE[k]
            evicted += 1
        if len(_RECO_OHLCV_CACHE) > _OHLCV_CACHE_MAX_ENTRIES:
            sorted_keys = sorted(
                _RECO_OHLCV_CACHE.keys(),
                key=lambda k: float(_RECO_OHLCV_CACHE[k].get("ts", 0))
            )
            excess = len(_RECO_OHLCV_CACHE) - _OHLCV_CACHE_MAX_ENTRIES
            for k in sorted_keys[:excess]:
                del _RECO_OHLCV_CACHE[k]
                evicted += 1
    return evicted


_PREFILTER_CRYPTO_VOLUME = os.getenv("PREFILTER_CRYPTO_VOLUME", "0").strip().lower() in ("1", "true", "yes")


def _prefilter_crypto_symbol(symbol: str) -> Tuple[bool, str]:
    """
    Cheap pre-filter for crypto symbols before full scan pipeline.
    Static checks (stablecoin, fiat) always run. Volume check requires fetch_ticker
    — disabled by default to avoid rate limit pressure (PREFILTER_CRYPTO_VOLUME=1 to enable).
    """
    base = (symbol.split("/")[0] or "").upper()
    if base in STABLECOINS or base in CRYPTO_BLOCKLIST_STABLECOINS:
        return False, "stablecoin"
    if base in FIAT_BASES:
        return False, "fiat_pair"
    if base in INVALID_KRAKEN_BASES:
        return False, "not_on_kraken"
    if not _PREFILTER_CRYPTO_VOLUME:
        return True, ""
    if not _kraken_ready():
        return True, ""
    try:
        t = kc.fetch_ticker(symbol) if kc else {}
        vol_quote = float(t.get("quoteVolume") or 0)
        last_price = float(t.get("last") or 0)
        if vol_quote > 0 and vol_quote < RECO_MIN_VOLUME_24H * 0.5:
            return False, f"low_volume_{vol_quote/1e6:.1f}M"
        if last_price <= 0:
            return False, "no_price_data"
    except Exception:
        pass
    return True, ""


def _prefilter_stock_symbol(symbol: str) -> Tuple[bool, str]:
    """Cheap pre-filter for stock symbols."""
    if len(symbol) > 5:
        return False, "symbol_too_long"
    return True, ""


def _get_scan_executor() -> ThreadPoolExecutor:
    """Get or create the shared thread pool for parallel scanning."""
    global _SCAN_EXECUTOR
    if _SCAN_EXECUTOR is None:
        _SCAN_EXECUTOR = ThreadPoolExecutor(
            max_workers=SCAN_PARALLEL_WORKERS,
            thread_name_prefix="scan_worker"
        )
    return _SCAN_EXECUTOR


def _trend_age(ema_fast: List[float], ema_slow: List[float], max_weeks: int = 52) -> int:
    if not ema_fast or not ema_slow or len(ema_fast) != len(ema_slow):
        return 0
    age = 0
    for a, b in zip(reversed(ema_fast[-max_weeks:]), reversed(ema_slow[-max_weeks:])):
        if a > b:
            age += 1
        else:
            break
    return age


def _scan_symbol(symbol: str, horizon: str, btc_ctx: Dict[str, Any]) -> Dict[str, Any]:
    """
    Scan a symbol using the Intelligence Layer.
    Routes to appropriate provider based on symbol type (stock → Alpaca, crypto → Kraken).
    """
    global _kraken_last_candle_ts, _alpaca_last_candle_ts
    market_type = classify_symbol(symbol)
    spread_pct: Optional[float] = None

    # Route based on symbol type
    if market_type == "stock":
        # Stock path - Alpaca preferred; Yahoo Finance fallback when Alpaca not configured (minimal mode)
        client = alpaca_live if alpaca_live else alpaca_paper
        try:
            try:
                import yfinance as yf  # type: ignore
            except ModuleNotFoundError:
                yf = None  # type: ignore

            def _yf_candles(sym: str, interval: str, period: str) -> list:
                if yf is None:
                    return []
                try:
                    t = yf.Ticker(sym)
                    hist = t.history(period=period, interval=interval)
                    if hist is None or hist.empty:
                        return []
                    out = []
                    for ts, row in hist.iterrows():
                        try:
                            out.append([
                                int(ts.timestamp()),
                                float(row["Open"]), float(row["High"]),
                                float(row["Low"]), float(row["Close"]),
                                float(row["Volume"]),
                            ])
                        except Exception:
                            continue
                    return out
                except Exception:
                    return []

            def _phase2_candles(sym: str, tf: str, periods: int) -> list:
                try:
                    from phase2_data_fetcher import fetch_recent_candles
                    return fetch_recent_candles(sym, timeframe=tf, periods=int(periods)) or []
                except Exception:
                    return []

            candles_1h = _scan_ohlcv_get(symbol, "1h", 300) or _yf_candles(symbol, "1h", "60d")
            if not candles_1h or len(candles_1h) < 8:
                candles_1h = candles_1h or _phase2_candles(symbol, "1h", 300)
            if candles_1h:
                _scan_ohlcv_put(symbol, "1h", 300, candles_1h)

            candles_4h = _scan_ohlcv_get(symbol, "4h", 300) or _yf_candles(symbol, "4h", "180d")
            if not candles_4h or len(candles_4h) < 8:
                candles_4h = candles_4h or _phase2_candles(symbol, "4h", 300)
            if candles_4h:
                _scan_ohlcv_put(symbol, "4h", 300, candles_4h)

            candles_1d = _scan_ohlcv_get(symbol, "1d", 500) or _yf_candles(symbol, "1d", "2y")
            if not candles_1d or len(candles_1d) < 20:
                candles_1d = candles_1d or _phase2_candles(symbol, "1d", 500)
            if candles_1d:
                _scan_ohlcv_put(symbol, "1d", 500, candles_1d)

            candles_1w = _scan_ohlcv_get(symbol, "1w", 300) or _yf_candles(symbol, "1wk", "5y")
            if not candles_1w or len(candles_1w) < 8:
                candles_1w = candles_1w or _phase2_candles(symbol, "1w", 300)
            if candles_1w:
                _scan_ohlcv_put(symbol, "1w", 300, candles_1w)

            if candles_1h or candles_1d:
                _alpaca_last_candle_ts = time.time()
                try:
                    from health_monitor import update_candle_ts
                    update_candle_ts("alpaca", time.time())
                except Exception:
                    pass

            if not candles_1w or len(candles_1w) < 8:
                candles_1w = _weekly_ohlcv_from_daily(candles_1d or [], max_weeks=120)
            if not candles_1d or len(candles_1d) < 20:
                return {
                    "symbol": symbol,
                    "score": 0.0,
                    "eligible": False,
                    "data_ok": False,
                    "reasons": ["Insufficient candle data (Yahoo Finance)"],
                    "risk_flags": ["DATA_ERROR"],
                    "metrics": {"market_type": "stocks"},
                    "regime": {}
                }
            # Get price: from Alpaca ticker if available, else from last daily candle
            price = 0.0
            if client:
                try:
                    ticker_data = client.get_ticker(symbol)
                    price = float(ticker_data.get("last", 0.0))
                except Exception:
                    pass
            if not price and candles_1d and len(candles_1d[-1]) >= 5:
                price = float(candles_1d[-1][4])
            spread_pct = 0.001  # Stocks typically have tight spreads (~0.1%)
            
            # Stock-specific metadata for intelligence scoring
            from stock_metadata import get_sector, get_liquidity_tier
            sector = get_sector(symbol)
            vol_24h = 0.0
            if candles_1d and len(candles_1d[-1]) >= 6:
                vol_24h = float(candles_1d[-1][5])
            liquidity_tier = get_liquidity_tier(price, vol_24h) if price and vol_24h else "unknown"
            stock_market_breadth = {
                "is_stock": True,
                "sector": sector,
                "liquidity_tier": liquidity_tier,
                "volume_24h": vol_24h,
            }
            # Earnings, analyst, sector ETF, market cap, IPO
            try:
                from earnings_calendar import days_until_earnings
                earnings_days = days_until_earnings(symbol)
                stock_market_breadth["earnings_days"] = earnings_days
            except Exception:
                pass
            try:
                from analyst_ratings_tracker import get_analyst_ratings, analyst_score_contribution
                ar = get_analyst_ratings(symbol)
                stock_market_breadth["analyst_consensus"] = ar.get("consensus")
                stock_market_breadth["analyst_score_delta"] = ar.get("score_delta", 0)
                contrib, _ = analyst_score_contribution(symbol)
                stock_market_breadth["analyst_score_contrib"] = contrib
            except Exception:
                pass
            try:
                from sector_etf_correlation import sector_etf_trend_ok
                sector_ok, sector_reason = sector_etf_trend_ok(symbol, candles_1d=candles_1d)
                stock_market_breadth["sector_etf_ok"] = sector_ok
                stock_market_breadth["sector_etf_reason"] = sector_reason
            except Exception:
                stock_market_breadth["sector_etf_ok"] = True
            try:
                from dividend_calendar import days_to_exdiv
                dte, amt, yld = days_to_exdiv(symbol, price)
                stock_market_breadth["days_to_exdiv"] = dte
                stock_market_breadth["dividend_yield_pct"] = yld
            except Exception:
                pass
            # Compute change_24h for stocks from daily candles (Alpaca has no percentage field)
            stock_change_24h: Optional[float] = None
            if candles_1d and len(candles_1d) >= 2:
                try:
                    _prev_close = float(candles_1d[-2][4])  # index 4 = close
                    _curr_close = float(price if price > 0 else candles_1d[-1][4])
                    if _prev_close > 0 and _curr_close > 0:
                        stock_change_24h = round((_curr_close - _prev_close) / _prev_close * 100.0, 3)
                except Exception:
                    pass
            try:
                from stock_profile import get_stock_profile, passes_market_cap_filter, is_recent_ipo
                p = get_stock_profile(symbol)
                stock_market_breadth["market_cap"] = p.get("market_cap")
                stock_market_breadth["market_cap_b"] = p.get("market_cap_b")
                stock_market_breadth["market_cap_tier"] = p.get("tier")
                stock_market_breadth["days_listed"] = p.get("days_listed")
                mc_ok, mc_reason = passes_market_cap_filter(symbol)
                stock_market_breadth["market_cap_ok"] = mc_ok
                stock_market_breadth["market_cap_reason"] = mc_reason
                recent_ipo, days = is_recent_ipo(symbol)
                stock_market_breadth["recent_ipo"] = recent_ipo
                stock_market_breadth["ipo_days"] = days
            except Exception:
                stock_market_breadth["market_cap_ok"] = True
                stock_market_breadth["recent_ipo"] = False
            
        except Exception as e:
            logger.error(f"Error fetching stock data for {symbol}: {e}")
            return {
                "symbol": symbol,
                "score": 0.0,
                "eligible": False,
                "data_ok": False,
                "reasons": [f"Data fetch error: {str(e)[:100]}"],
                "risk_flags": ["DATA_ERROR"],
                "metrics": {"market_type": "stocks"},
                "regime": {}
            }
        market_breadth = stock_market_breadth
    else:
        # Crypto path - use Kraken (with guardrails now in place)
        if not _kraken_ready():
            logger.warning(f"Crypto symbol {symbol} scanned but Kraken not ready")
            return {
                "symbol": symbol,
                "score": 0.0,
                "eligible": False,
                "data_ok": False,
                "reasons": ["Kraken not available"],
                "risk_flags": ["NO_CRYPTO_PROVIDER"],
                "metrics": {"market_type": "crypto"},
                "regime": {}
            }
        
        sym = _resolve_symbol(symbol)
        market_breadth = {}
        crypto_volume_24h: Optional[float] = None
        crypto_change_24h: Optional[float] = None

        # Fetch crypto data from Kraken (guardrails will catch any misrouted stocks)
        try:
            # Use 2hr scan TTL so medium/long horizons reuse data fetched by short scan
            candles_4h = _ohlcv_cached(sym, "4h", 300, _SCAN_OHLCV_TTL)
            candles_1d = _ohlcv_cached(sym, "1d", 500, _SCAN_OHLCV_TTL)
            candles_1w = _ohlcv_cached(sym, "1w", 300, _SCAN_OHLCV_TTL)
            candles_1h = _ohlcv_cached(sym, "1h", 300, _SCAN_OHLCV_TTL)
            if candles_1h or candles_1d:
                _kraken_last_candle_ts = time.time()
                try:
                    from health_monitor import update_candle_ts
                    update_candle_ts("kraken", time.time())
                except Exception:
                    pass
            if not candles_1w or len(candles_1w) < 8:
                candles_1w = _weekly_ohlcv_from_daily(candles_1d or [], max_weeks=120)
                if candles_1w:
                    logger.debug("Using synthesized weekly candles for %s (%d bars)", sym, len(candles_1w))

            price = _safe_last_price(sym) or 0.0
            spread_pct = _safe_spread_pct(sym)
            try:
                tb_map = _tickers_batch_cached(ttl_sec=15)
                ti = tb_map.get(sym) if tb_map else None
                if ti is not None and (ti.get("quoteVolume") is not None or ti.get("last") is not None):
                    crypto_volume_24h = float(ti.get("quoteVolume") or ti.get("volume") or 0)
                    _pct = ti.get("percentage")
                    if _pct is not None:
                        crypto_change_24h = float(_pct)
                else:
                    t = kc.ex.fetch_ticker(sym) if kc else {}
                    crypto_volume_24h = float(t.get("quoteVolume") or t.get("volume") or 0)
                    _pct = t.get("percentage")
                    if _pct is not None:
                        crypto_change_24h = float(_pct)
            except Exception:
                pass
        except ValueError as e:
            # Guardrail caught a routing error
            logger.error(f"Routing error for {symbol}: {e}")
            return {
                "symbol": symbol,
                "score": 0.0,
                "eligible": False,
                "data_ok": False,
                "reasons": [f"Routing error: {str(e)[:100]}"],
                "risk_flags": ["ROUTING_ERROR"],
                "metrics": {"market_type": "crypto"},
                "regime": {}
            }
        except Exception as e:
            logger.error(f"Error fetching crypto data for {symbol}: {e}")
            return {
                "symbol": symbol,
                "score": 0.0,
                "eligible": False,
                "data_ok": False,
                "reasons": [f"Data fetch error: {str(e)[:100]}"],
                "risk_flags": ["DATA_ERROR"],
                "metrics": {"market_type": "crypto"},
                "regime": {}
            }
    
    # Create Intelligence Context (same for both stock and crypto)
    # Note: For recommendations we don't have a specific bot config or account state,
    # so we pass defaults or "research mode" values.
    # market_breadth: for stocks includes sector, liquidity_tier, earnings_days, analyst, sector_etf, etc.
    bot_config = {}
    if market_type == "stock" and market_breadth:
        ed = market_breadth.get("earnings_days")
        if ed is not None:
            bot_config["earnings_days"] = ed
    context = IntelligenceContext(
        symbol=symbol,  # Use original symbol, not resolved
        last_price=price,
        spread_pct=spread_pct,
        candles_1h=candles_1h or [],
        candles_4h=candles_4h or [],
        candles_1d=candles_1d or [],
        candles_1w=candles_1w or [],
        btc_context=btc_ctx,
        market_breadth=market_breadth,
        bot_config=bot_config,
        now_ts=int(time.time()),
        last_price_ts=int(time.time()),  # Assuming fresh since we just fetched
        dry_run=True,
    )
    
    # Generate Recommendation via Intelligence Layer
    recommendation = intelligence_layer.generate_recommendation(context, horizon)

    # Scan logging for diagnostics
    _r_score = recommendation.get("score", 0)
    _r_regime = (recommendation.get("metrics") or {}).get("regime", "?")
    _r_eligible = recommendation.get("eligible", True)
    _r_reasons = (recommendation.get("reasons") or [])[:3]
    _r_flags = (recommendation.get("risk_flags") or [])[:3]
    logger.info("SCAN %s [%s] score=%.1f regime=%s eligible=%s reasons=%s flags=%s",
                symbol, horizon, _r_score, _r_regime, _r_eligible, _r_reasons, _r_flags)
    
    if "metrics" in recommendation and isinstance(recommendation["metrics"], dict):
        recommendation["metrics"]["market_type"] = "stocks" if market_type == "stock" else "crypto"
        if market_type == "crypto":
            if crypto_change_24h is not None:
                recommendation["metrics"]["change_24h"] = crypto_change_24h
            if crypto_volume_24h is not None:
                recommendation["metrics"]["volume_24h_quote"] = crypto_volume_24h
                if crypto_volume_24h < RECO_MIN_VOLUME_24H:
                    recommendation["eligible"] = False
                    recommendation.setdefault("risk_flags", []).append(
                        f"Low volume: ${crypto_volume_24h/1e6:.1f}M < ${RECO_MIN_VOLUME_24H/1e6:.0f}M min"
                    )
        if market_type == "stock" and market_breadth:
            # Inject stock change_24h computed from daily candles (Alpaca has no percentage ticker field)
            try:
                if stock_change_24h is not None:
                    recommendation["metrics"]["change_24h"] = stock_change_24h
            except Exception:
                pass
            mc = market_breadth.get("market_cap")
            if mc is None and market_breadth.get("market_cap_b"):
                mc = float(market_breadth["market_cap_b"]) * 1e9
            if mc and mc < RECO_MIN_MARKET_CAP:
                recommendation["eligible"] = False
                recommendation.setdefault("risk_flags", []).append(
                    f"Market cap ${mc/1e9:.2f}B < ${RECO_MIN_MARKET_CAP/1e9:.0f}B min"
                )
        if market_type == "crypto":
            try:
                from funding_rate_tracker import get_funding_rate
                from crypto_cycle_detector import get_cycle_phase
                fr = get_funding_rate(symbol)
                if fr:
                    recommendation["metrics"]["funding_rate"] = fr.get("rate")
                    recommendation["metrics"]["funding_signal"] = fr.get("signal")
                    # Warning when funding extremely high (overleveraged long)
                    rate_val = fr.get("rate") or 0
                    if isinstance(rate_val, (int, float)) and rate_val >= 0.001:
                        recommendation.setdefault("risk_flags", []).append(
                            f"High funding rate: {rate_val*100:.3f}% (overleveraged)"
                        )
                cyc = get_cycle_phase()
                recommendation["metrics"]["cycle_phase"] = cyc.get("phase")
            except Exception:
                pass

    # Benchmark & competitive analysis (SPY for stocks, BTC for crypto)
    try:
        from benchmark_analyzer import enrich_recommendation_with_benchmark
        candles_1d = context.candles_1d if hasattr(context, "candles_1d") else []
        benchmark_candles = None
        if len(candles_1d or []) >= 30:
            _bench_ttl = float(os.getenv("SCAN_BENCHMARK_CACHE_TTL_SEC", "120"))
            if market_type == "stock" and (alpaca_live or alpaca_paper):
                _client = alpaca_live or alpaca_paper

                def _fetch_spy():
                    try:
                        return _client.get_ohlcv("SPY", "1d", 200)
                    except Exception:
                        return []

                benchmark_candles = _benchmark_ohlcv_cached("SPY_1d_200", _bench_ttl, _fetch_spy)
            elif market_type == "crypto" and _kraken_ready():
                _btc_sym = _resolve_symbol("BTC/USD")

                def _fetch_btc():
                    try:
                        return _ohlcv_cached(_btc_sym, "1d", 200, 300)
                    except Exception:
                        return []

                benchmark_candles = _benchmark_ohlcv_cached(f"BTC_1d_200_{_btc_sym}", _bench_ttl, _fetch_btc)
            sector = (context.market_breadth or {}).get("sector") if hasattr(context, "market_breadth") else None
            enriched = enrich_recommendation_with_benchmark(
                symbol, price, candles_1d=candles_1d, benchmark_candles=benchmark_candles, sector=sector
            )
            for k, v in enriched.items():
                if v is not None and v != "":
                    recommendation["metrics"][k] = v
            # Score boost for top-quartile peer performers
            if enriched.get("peer_quartile") == "top":
                base = float(recommendation.get("score") or 0)
                recommendation["score"] = min(98.0, base + 3.0)
                recommendation.setdefault("reasons", []).append("Top-quartile in sector")
            # Add benchmark comparison to reasons when available
            if enriched.get("benchmark_vs"):
                recommendation.setdefault("reasons", []).append(enriched["benchmark_vs"])
    except Exception as e:
        logger.debug("Benchmark enrichment failed for %s: %s", symbol, e)
    
    # Explore V2: hard gates (when enabled)
    try:
        from explore_v2 import apply_universe_gates, enhance_score, is_enabled as explore_v2_enabled
        if explore_v2_enabled():
            metrics = recommendation.get("metrics") or {}
            spread_bps = float(spread_pct or 0) * 10000.0
            vol_pct = metrics.get("volatility_pct") or metrics.get("atr_pct")
            vol_avg = metrics.get("volatility_avg_pct")
            volume_24h = metrics.get("volume_24h_quote")
            pass_gate, fail_reason = apply_universe_gates(
                symbol, volume_24h_quote=volume_24h,
                spread_bps=spread_bps if spread_bps > 0 else None,
                volatility_pct=float(vol_pct) if vol_pct is not None else None,
                volatility_avg_pct=float(vol_avg) if vol_avg is not None else None,
            )
            if not pass_gate:
                recommendation["eligible"] = False
                recommendation["score"] = 0.0
                recommendation.setdefault("risk_flags", []).append(f"EXPLORE_V2_GATE:{fail_reason}")
            else:
                base_score = float(recommendation.get("score") or 0)
                adj_score, extra_reasons = enhance_score(
                    base_score, recommendation,
                    regime=str(metrics.get("regime") or ""),
                    spread_bps=spread_bps if spread_bps > 0 else None,
                    volatility_pct=float(vol_pct) if vol_pct is not None else None,
                )
                recommendation["score"] = max(0.0, min(95.0, adj_score))
                recommendation["reasons"] = (recommendation.get("reasons") or []) + extra_reasons
    except ImportError:
        pass

    # Explore chart-pattern feed is written separately in _persist_explore_feed_from_snap (DB UPSERT).

    try:
        recommendation["_candles_1d"] = [list(x) for x in (candles_1d or [])]
    except Exception:
        recommendation["_candles_1d"] = []

    try:
        if spread_pct is not None:
            recommendation.setdefault("metrics", {})["spread_pct"] = float(spread_pct)
    except Exception:
        pass

    return recommendation


def _analyze_market_data(
    symbol: str, 
    horizon: str, 
    btc_ctx: Dict[str, Any], 
    candles_1h: List[List[float]],
    candles_4h: List[List[float]],
    candles_1d: List[List[float]],
    candles_1w: List[List[float]]
) -> Dict[str, Any]:
    """
    Legacy / Helper: Kept for signature compatibility or direct analysis calls if needed.
    Now just routes to IntelligenceLayer as well.
    """
    # Simply wrap into _scan_symbol logic which builds context
    # Use the provided candles instead of fetching again
    price = 0.0
    if candles_1d: price = float(candles_1d[-1][4])
    elif candles_4h: price = float(candles_4h[-1][4])
    
    context = IntelligenceContext(
        symbol=symbol,
        last_price=price,
        candles_1h=candles_1h,
        candles_4h=candles_4h,
        candles_1d=candles_1d,
        candles_1w=candles_1w,
        btc_context=btc_ctx,
        now_ts=int(time.time()),
        last_price_ts=int(time.time()),
        dry_run=True
    )
    return intelligence_layer.generate_recommendation(context, horizon)


# Global Cache for Universe
_UNIVERSE_CACHE = {"ts": 0, "symbols": []}
UNIVERSE_TTL = 600  # 10 minutes

# Momentum-based universe filtering (stocks)
_MOMENTUM_FILTER_ENABLED = os.getenv("RECO_MOMENTUM_FILTER", "0").strip().lower() in (
    "1",
    "true",
    "yes",
    "y",
    "on",
)
_MOMENTUM_FILTER_MIN_SCORE = float(os.getenv("RECO_MOMENTUM_MIN_SCORE", "60"))
_MOMENTUM_FILTER_TOP_N_STOCKS = int(os.getenv("RECO_MOMENTUM_TOP_N_STOCKS", "500"))


def _reco_symbols(quote: str = "USD") -> List[str]:
    with _globals_lock:
        if time.time() - _UNIVERSE_CACHE["ts"] < UNIVERSE_TTL and _UNIVERSE_CACHE["symbols"]:
            return list(_UNIVERSE_CACHE["symbols"])

    symbols = set()

    # Use the new dynamic universe builder for large, comprehensive universe
    try:
        from universe_builder import get_equity_universe, get_crypto_universe
        equity_universe = get_equity_universe()
        crypto_universe = get_crypto_universe()
        for s in equity_universe:
            symbols.add(s)
        for s in crypto_universe:
            symbols.add(s)
        logger.info("[UNIVERSE] Dynamic universe: %d equities + %d crypto = %d total",
                    len(equity_universe), len(crypto_universe), len(symbols))
    except Exception as _ub_err:
        logger.warning("[UNIVERSE] Dynamic universe builder failed (%s), falling back to legacy", _ub_err)
        # Legacy fallback: use the old approach
        mk = _markets() if _kraken_ready() else {}
        for s in RECO_SYMBOLS:
            if "/" in s and mk:
                base = (s.split("/")[0] or "").upper()
                if base in CRYPTO_BLOCKLIST:
                    continue
                resolved, err = _validate_crypto_symbol(s)
                if resolved:
                    symbols.add(resolved)
            elif "/" not in s:
                symbols.add(s)

        try:
            if kc and _kraken_ready():
                tickers = {}
                try:
                    tickers = kc.ex.fetch_tickers()
                except Exception:
                    pass
                _min_crypto_volume = float(os.getenv("RECO_CRYPTO_MIN_VOLUME", "500000"))
                _max_crypto = int(os.getenv("RECO_CRYPTO_TOP_N", "50"))
                if tickers:
                    sorted_tickers = sorted(
                        [(s, float(t.get("quoteVolume") or 0))
                         for s, t in tickers.items()
                         if "/USD" in s and float(t.get("quoteVolume") or 0) >= _min_crypto_volume
                         and (s.split("/")[0] or "").upper() not in FIAT_BASES],
                        key=lambda x: x[1], reverse=True
                    )
                    for s, _ in sorted_tickers[:_max_crypto]:
                        base = (s.split("/")[0] or "").upper()
                        if base not in CRYPTO_BLOCKLIST:
                            symbols.add(s)
        except Exception as e:
            logger.error("Error fetching crypto universe: %s", e)

        try:
            from stock_universe import get_expanded_stock_universe
            for s in get_expanded_stock_universe():
                symbols.add(s)
        except Exception as e:
            logger.warning("stock_universe import failed: %s", e)

        try:
            client = alpaca_live if alpaca_live else alpaca_paper
            if client:
                for a in (client.get_active_assets() or [])[:200]:
                    sym = (a.get("symbol") or "").strip()
                    if sym and "." not in sym and "$" not in sym and "-" not in sym and len(sym) <= 5:
                        symbols.add(sym)
        except Exception:
            pass

    # Filter out fiat FX pairs and blocked crypto bases
    filtered = []
    for s in symbols:
        if "/" in s:
            base = (s.split("/")[0] or "").upper()
            if base in FIAT_BASES or base in CRYPTO_BLOCKLIST:
                continue
        filtered.append(s)
    final_list = sorted(filtered)
    with _globals_lock:
        _UNIVERSE_CACHE["ts"] = time.time()
        _UNIVERSE_CACHE["symbols"] = final_list
    return final_list


def _apply_momentum_filter_to_universe(symbols: List[str]) -> List[str]:
    """
    Optional: focus stock universe on strong-momentum names.

    - Only applies to stocks (plain symbols without "/").
    - Crypto universe is kept as-is.
    - Controlled via:
        RECO_MOMENTUM_FILTER           (default: 1 = enabled)
        RECO_MOMENTUM_MIN_SCORE        (default: 60)
        RECO_MOMENTUM_TOP_N_STOCKS     (default: 150)
    """
    if not _MOMENTUM_FILTER_ENABLED:
        return symbols

    stock_symbols = [s for s in symbols if len(s) < 6 and "/" not in s]
    if not stock_symbols:
        return symbols

    try:
        from momentum_ranking import MomentumRanker
    except Exception as e:
        logger.debug("Momentum universe filter disabled (MomentumRanker import failed): %s", e)
        return symbols

    try:
        ranker = MomentumRanker()
        ranked = ranker.rank_universe(stock_symbols)
    except Exception as e:
        logger.debug("Momentum universe filter disabled (ranking error): %s", e)
        return symbols

    min_score = float(_MOMENTUM_FILTER_MIN_SCORE)
    top_n = max(1, int(_MOMENTUM_FILTER_TOP_N_STOCKS))

    picked: List[str] = []
    for r in ranked:
        try:
            score = float(r.get("score") or 0.0)
        except Exception:
            score = 0.0
        if score < min_score:
            continue
        sym = str(r.get("symbol") or "").strip()
        if not sym:
            continue
        picked.append(sym)
        if len(picked) >= top_n:
            break

    if not picked:
        logger.info(
            "Momentum universe filter produced 0 symbols (min_score=%.1f). Keeping original universe.",
            min_score,
        )
        return symbols

    crypto = [s for s in symbols if s not in stock_symbols]
    out = crypto + picked
    logger.info(
        "Momentum universe filter applied: %s -> %s total (%s stocks kept, min_score=%.1f, top_n=%s)",
        len(symbols),
        len(out),
        len(picked),
        min_score,
        top_n,
    )
    return out


# Optional: focus crypto universe on strong-momentum pairs (RECO_CRYPTO_MOMENTUM_FILTER=1, RECO_CRYPTO_MOMENTUM_TOP_N=80)
_CRYPTO_MOMENTUM_FILTER_ENABLED = os.getenv("RECO_CRYPTO_MOMENTUM_FILTER", "0").strip().lower() in ("1", "true", "yes", "y", "on")
_CRYPTO_MOMENTUM_TOP_N = int(os.getenv("RECO_CRYPTO_MOMENTUM_TOP_N", "80"))


def _apply_crypto_momentum_filter(symbols: List[str]) -> List[str]:
    """
    Optional: keep only top N crypto symbols by momentum (5d/20d/60d).
    Controlled by RECO_CRYPTO_MOMENTUM_FILTER (default 0=off), RECO_CRYPTO_MOMENTUM_TOP_N (default 80).
    """
    if not _CRYPTO_MOMENTUM_FILTER_ENABLED:
        return symbols

    crypto_symbols = [s for s in symbols if "/" in s and (s.split("/")[0] or "").upper() not in FIAT_BASES]
    if not crypto_symbols:
        return symbols

    try:
        from momentum_ranking import MomentumRanker
    except Exception as e:
        logger.debug("Crypto momentum filter disabled (MomentumRanker import failed): %s", e)
        return symbols

    try:
        ranker = MomentumRanker()
        ranked = ranker.rank_universe(crypto_symbols)
    except Exception as e:
        logger.debug("Crypto momentum filter disabled (ranking error): %s", e)
        return symbols

    top_n = max(1, _CRYPTO_MOMENTUM_TOP_N)
    picked = [str(r.get("symbol") or "").strip() for r in ranked[:top_n] if str(r.get("symbol") or "").strip()]
    if not picked:
        return symbols

    stocks = [s for s in symbols if s not in crypto_symbols]
    out = stocks + picked
    logger.info(
        "Crypto momentum filter applied: %s crypto -> %s kept (top_n=%s)",
        len(crypto_symbols),
        len(picked),
        top_n,
    )
    return out


def _scan_one_symbol(sym: str, horizon: str, btc_ctx: Dict[str, Any]) -> Dict[str, Any]:
    """
    Scan a single symbol — designed to run inside a thread pool.
    Returns a result dict with status info, or None-equivalent on skip.
    """
    result = {"symbol": sym, "status": "skipped", "snap": None, "error": None}
    try:
        from stock_universe import JUNK_TICKERS
        _sym_upper = sym.upper().split("/")[0] if "/" in sym else sym.upper()
        if _sym_upper in JUNK_TICKERS:
            result["status"] = "junk_ticker"
            return result
        if "/" in sym:
            base = (sym.split("/")[0] or "").upper()
            if base in CRYPTO_BLOCKLIST:
                result["status"] = "blocklisted"
                return result
            resolved, _ = _validate_crypto_symbol(sym)
            if not resolved:
                result["status"] = "invalid_symbol"
                return result
        snap = _scan_symbol(sym, horizon, btc_ctx)
        if not snap:
            result["status"] = "no_data"
            return result
        if snap.get("data_ok") is False:
            result["status"] = "data_not_ok"
            return result
        risk_flags = snap.get("risk_flags") or []
        if "DATA_INVALID" in risk_flags or "DATA_ERROR" in risk_flags or "ROUTING_ERROR" in risk_flags:
            result["status"] = "data_error"
            return result
        metrics = snap.get("metrics") or {}
        is_stock = len(sym) < 6 and "/" not in sym
        metrics["market_type"] = "stocks" if is_stock else "crypto"
        if not is_stock:
            try:
                from meme_coin_detector import should_block_crypto
                if should_block_crypto(sym, metrics):
                    result["status"] = "meme_blocked"
                    return result
            except Exception:
                pass
        snap_eligible = snap.get("eligible", False)
        metrics["eligible"] = snap_eligible
        snap_score = float(snap.get("score") or 0.0)
        buy_thresh = _reco_buy_threshold_stocks() if is_stock else _reco_buy_threshold_crypto()
        derived_signal = "buy" if snap_score >= buy_thresh else ("watch" if snap_score >= _reco_watch_threshold() else "wait")
        metrics["signal"] = derived_signal

        # Collect human-readable rejection reasons with specific indicator values
        rejection_reasons = []
        try:
            _rsi14 = metrics.get("rsi_14") or metrics.get("rsi")
            if _rsi14 is not None and float(_rsi14) > 70:
                rejection_reasons.append(f"RSI too high: {float(_rsi14):.0f}")
            if _rsi14 is not None and float(_rsi14) < 20:
                rejection_reasons.append(f"RSI extremely low: {float(_rsi14):.0f} (capitulation risk)")
            _pct_200 = metrics.get("pct_above_200")
            if _pct_200 is not None and float(_pct_200) < -10:
                rejection_reasons.append(f"Below 200MA by {abs(float(_pct_200)):.0f}%")
            _vol_ratio = metrics.get("volume_ratio")
            if _vol_ratio is not None and float(_vol_ratio) < 0.5:
                rejection_reasons.append(f"Low volume: {float(_vol_ratio):.1f}x average")
            if snap_score < buy_thresh and snap_score >= _reco_watch_threshold():
                rejection_reasons.append(f"Score {snap_score:.0f} below buy threshold {buy_thresh}")
            elif snap_score < _reco_watch_threshold():
                rejection_reasons.append(f"Score {snap_score:.0f} too low (watch={_reco_watch_threshold()}, buy={buy_thresh})")
            _rej = metrics.get("_explore_reject_reason")
            if _rej:
                rejection_reasons.append(str(_rej).replace("_", " "))
        except Exception:
            pass
        metrics["rejection_reasons"] = rejection_reasons
        snap["metrics"] = metrics

        result["snap"] = snap
        result["status"] = "ok"
        result["eligible"] = snap_eligible
        result["signal"] = derived_signal
        result["score"] = snap_score
        result["rejection_reasons"] = rejection_reasons
    except Exception as e:
        err_str = str(e).lower()
        # 429/rate-limit and "no data" errors are expected transient failures — count as skip,
        # not error, so they don't trigger the RECO_SCAN_ERROR_LIMIT abort.
        if "429" in err_str or "too many request" in err_str or "rate limit" in err_str:
            result["status"] = "rate_limited"
            result["error"] = str(e)[:200]
        elif "no data" in err_str or "insufficient data" in err_str or "not found" in err_str:
            result["status"] = "no_data"
        else:
            result["status"] = "error"
            result["error"] = str(e)[:200]
    return result


def _build_symbols_to_scan(horizon: str) -> List[str]:
    """Build the filtered, prioritized symbol list for a scan pass."""
    symbols = _reco_symbols(quote="USD")
    logger.warning("[SCAN] _reco_symbols returned %d symbols (crypto=%d stocks=%d)",
        len(symbols),
        len([s for s in symbols if "/" in s]),
        len([s for s in symbols if "/" not in s and len(s) < 6]))
    symbols = _apply_momentum_filter_to_universe(symbols)
    logger.warning("[SCAN] after momentum filter: %d symbols", len(symbols))
    symbols = _apply_crypto_momentum_filter(symbols)
    logger.warning("[SCAN] after crypto momentum filter: %d symbols for %s horizon", len(symbols), horizon)

    crypto_symbols = [s for s in symbols if ("/" in s or len(s) > 6) and (s.split("/")[0] or "").upper() not in CRYPTO_BLOCKLIST]
    stock_symbols = [s for s in symbols if len(s) < 6 and "/" not in s]
    logger.warning("[SCAN] split: crypto=%d stocks=%d", len(crypto_symbols), len(stock_symbols))

    if RECO_CRYPTO_TOP_30_ONLY:
        crypto_selected = [s for s in crypto_symbols if _crypto_base_from_symbol(s) in TOP_30_CRYPTO_BASES]
        blocked = [s for s in crypto_symbols if s not in crypto_selected]
        if blocked:
            logger.info("RECO_CRYPTO_TOP_30_ONLY: blocked %d, kept %d", len(blocked), len(crypto_selected))
    else:
        priority_crypto = [
            "BTC/USD", "ETH/USD", "SOL/USD", "XRP/USD",
            "TAO/USD", "FET/USD", "AKT/USD", "OCEAN/USD",
            "KAS/USD", "CHZ/USD", "ENA/USD",
            "ADA/USD", "DOGE/USD", "LTC/USD", "AVAX/USD",
            "LINK/USD", "DOT/USD", "ATOM/USD", "NEAR/USD",
            "APT/USD", "ARB/USD", "MATIC/USD", "UNI/USD",
            "INJ/USD", "RUNE/USD", "TIA/USD",
        ]
        crypto_set = set(crypto_symbols)
        crypto_selected = [pc for pc in priority_crypto if pc in crypto_set]
        crypto_selected += [s for s in crypto_symbols if s not in set(priority_crypto)]

    try:
        from stock_universe import get_sp500_tickers, get_nasdaq100_tickers, get_major_etfs
        _priority_tickers = get_major_etfs() + get_nasdaq100_tickers() + get_sp500_tickers()
    except Exception:
        _priority_tickers = ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA", "NVDA", "META", "NFLX", "SPY", "QQQ"]
    stock_set = set(stock_symbols)
    _prio_set = set()
    stock_selected = []
    for ps in _priority_tickers:
        ps = ps.strip().upper()
        if ps in stock_set and ps not in _prio_set:
            stock_selected.append(ps)
            _prio_set.add(ps)
    stock_selected += [s for s in stock_symbols if s not in _prio_set]

    n_max = min(RECO_MAX_SYMBOLS, 600)
    n_crypto = min(max(40, RECO_SCAN_CRYPTO_CAP), len(crypto_selected))
    n_stocks = min(max(1, n_max - n_crypto), len(stock_selected))
    symbols_to_scan = crypto_selected[:n_crypto] + stock_selected[:n_stocks]
    symbols_to_scan = [
        s for s in symbols_to_scan if not ("/" in s and (s.split("/")[0] or "").upper() in FIAT_BASES)
    ]
    logger.warning("[SCAN] Final list: %d symbols. First 15 stocks: %s",
                   len(symbols_to_scan), [s for s in stock_selected[:15]])

    # Guarantee refresh of active BUY/WATCH signals that age out of the universe.
    # Symbols discovered via movers/actives may drop off the dynamic list after one scan.
    # Re-include stocks and crypto from recommendations_latest with score >= 65.
    try:
        from db import list_recommendations
        _stale_rows = list_recommendations(horizon, limit=200)
        _scan_set = set(symbols_to_scan)
        _stale_added = []
        for _row in _stale_rows:
            _sym = str(_row.get("symbol") or "")
            if not _sym or _sym in _scan_set:
                continue
            if float(_row.get("score") or 0) < 65:
                continue
            _is_crypto = "/" in _sym
            if _is_crypto:
                _base = (_sym.split("/")[0] or "").upper()
                if _base in CRYPTO_BLOCKLIST or _base in FIAT_BASES:
                    continue
            else:
                if len(_sym) >= 6:
                    continue
                if "$" in _sym or "." in _sym or "-" in _sym:
                    continue
            symbols_to_scan.append(_sym)
            _scan_set.add(_sym)
            _stale_added.append(_sym)
        if _stale_added:
            logger.info("[SCAN] Re-added %d stale high-score symbols to universe: %s", len(_stale_added), _stale_added[:15])
    except Exception as _stale_err:
        logger.warning("[SCAN] Failed to add stale BUY symbols: %s", _stale_err)

    if RECO_SCAN_MAX_PER_RUN > 0:
        symbols_to_scan = symbols_to_scan[: min(len(symbols_to_scan), RECO_SCAN_MAX_PER_RUN)]

    # Pre-filter: skip junk symbols cheaply before full pipeline
    prefilter_start = time.time()
    filtered = []
    prefilter_skipped = {"crypto": 0, "stock": 0}
    prefilter_reasons: Dict[str, int] = {}
    for sym in symbols_to_scan:
        is_crypto = "/" in sym or len(sym) > 6
        if is_crypto:
            passes, reason = _prefilter_crypto_symbol(sym)
            if not passes:
                prefilter_skipped["crypto"] += 1
                prefilter_reasons[reason] = prefilter_reasons.get(reason, 0) + 1
                continue
        else:
            passes, reason = _prefilter_stock_symbol(sym)
            if not passes:
                prefilter_skipped["stock"] += 1
                prefilter_reasons[reason] = prefilter_reasons.get(reason, 0) + 1
                continue
        filtered.append(sym)
    prefilter_dur = time.time() - prefilter_start
    logger.warning("[SCAN] Pre-filter: %d -> %d symbols in %.1fs (skipped crypto=%d stocks=%d reasons=%s)",
        len(symbols_to_scan), len(filtered), prefilter_dur,
        prefilter_skipped["crypto"], prefilter_skipped["stock"],
        dict(list(prefilter_reasons.items())[:5]))

    return filtered


def _scan_recommendations_impl(horizon: str) -> None:
    global _last_reco_short_ts, _last_reco_medium_ts, _last_reco_long_ts, _kraken_last_candle_ts, _alpaca_last_candle_ts
    logger.warning("[SCAN-DEBUG] _scan_recommendations_impl ENTRY horizon=%s Kraken=%s Alpaca=%s",
        horizon, _kraken_ready(), bool(alpaca_live or alpaca_paper))
    has_clients = _kraken_ready() or (alpaca_live or alpaca_paper)
    if not has_clients:
        try:
            syms = _reco_symbols(quote="USD")
            stock_count = len([s for s in syms if "/" not in s and len(s) < 6])
            if stock_count == 0:
                logger.warning("[RECO_DEBUG] SCAN SKIPPED: No trading clients and no stock symbols.")
                with _globals_lock:
                    _RECO_STATE[horizon] = {
                        "last_run_ts": now_ts(), "error": "No clients",
                        "btc_ctx": {}, "scanned": 0, "eligible": 0, "total": 0, "scanning": False,
                        "started_at": time.time(),
                    }
                return
            logger.info("[RECO_DEBUG] No Kraken/Alpaca - scanning %d stocks via Yahoo Finance fallback", stock_count)
        except Exception as e:
            logger.warning("[RECO_DEBUG] SCAN SKIPPED: No clients and symbol check failed: %s", e)
            with _globals_lock:
                _RECO_STATE[horizon] = {
                    "last_run_ts": now_ts(), "error": str(e)[:120],
                    "btc_ctx": {}, "scanned": 0, "eligible": 0, "total": 0, "scanning": False,
                    "started_at": time.time(),
                }
            return

    now = now_ts()
    error = ""
    btc_ctx = _btc_context()
    scanned = 0
    eligible = 0
    total_to_scan = 0
    _scan_started_at = time.time()
    symbols_to_scan: List[str] = []
    try:
        symbols_to_scan = _build_symbols_to_scan(horizon)
    except Exception as build_err:
        logger.error("%s scan: _build_symbols_to_scan failed: %s", horizon, build_err, exc_info=True)
        with _globals_lock:
            _RECO_STATE[horizon] = {
                "last_run_ts": now, "error": f"{type(build_err).__name__}: {build_err}"[:200],
                "btc_ctx": btc_ctx, "scanned": 0, "eligible": 0, "total": 0, "scanning": False,
                "started_at": _scan_started_at,
            }
            if horizon == "short":
                _last_reco_short_ts = time.time()
            elif horizon == "medium":
                _last_reco_medium_ts = time.time()
            else:
                _last_reco_long_ts = time.time()
        return

    if len(symbols_to_scan) == 0:
        logger.warning("No symbols to scan for recommendations (horizon=%s)", horizon)
        with _globals_lock:
            _RECO_STATE[horizon] = {
                "last_run_ts": now, "error": "No symbols available",
                "btc_ctx": btc_ctx, "scanned": 0, "eligible": 0, "total": 0, "scanning": False,
                "started_at": _scan_started_at,
            }
            if horizon == "short":
                _last_reco_short_ts = time.time()
            elif horizon == "medium":
                _last_reco_medium_ts = time.time()
            else:
                _last_reco_long_ts = time.time()
        return

    _full_syms = list(symbols_to_scan)
    total_to_scan = len(_full_syms)
    _fp = hashlib.sha256(",".join(_full_syms).encode("utf-8")).hexdigest()[:24]
    _resume_at = 0
    try:
        _ck = _scan_checkpoint_read()
        if _ck.get("horizon") == horizon and _ck.get("fingerprint") == _fp:
            _resume_at = int(_ck.get("next_batch_start", 0))
            if _resume_at >= total_to_scan:
                _resume_at = 0
            elif _resume_at > 0:
                logger.warning(
                    "[SCAN] Resuming %s from symbol index %d / %d (checkpoint)",
                    horizon, _resume_at, total_to_scan,
                )
    except Exception:
        pass

    logger.info("Starting %s scan with %d symbols", horizon, total_to_scan)
    _raw_bs = max(SCAN_PARALLEL_WORKERS * 2, SCAN_PARALLEL_WORKERS * SCAN_BATCH_SIZE_MULT)
    batch_size = max(4, _raw_bs // 2)
    logger.warning(
        "[SCAN-DEBUG] starting %s horizon scan, total symbols: %d | workers=%d batch_size=%d resume_from=%d",
        horizon, total_to_scan, SCAN_PARALLEL_WORKERS, batch_size, _resume_at,
    )

    try:

        with _globals_lock:
            _BENCHMARK_1D_CACHE.clear()

        try:
            mark_explore_signals_pending(horizon, now_ts())
        except Exception as _pend_e:
            logger.warning("mark_explore_signals_pending failed: %s", _pend_e)

        with _globals_lock:
            _prev_run_ts = (_RECO_STATE.get(horizon) or {}).get("last_run_ts", 0)
            _RECO_STATE[horizon] = {
                "last_run_ts": _prev_run_ts or now, "error": "", "btc_ctx": btc_ctx,
                "scanned": 0, "eligible": 0, "total": total_to_scan, "scanning": True,
                "started_at": _scan_started_at,
            }

        error_count = 0
        buy_signals = 0
        skip_counts: Dict[str, int] = {}
        scan_start = time.time()
        with _globals_lock:
            _SCAN_PROGRESS["scan_start_ts"] = scan_start
            _SCAN_PROGRESS["current_horizon"] = horizon
            _SCAN_PROGRESS["current_symbol"] = ""
            _SCAN_PROGRESS["buy_signals_found"] = 0

        executor = _get_scan_executor()
        _batch_num = 0
        _rate_limited_count = 0
        for batch_start in range(_resume_at, total_to_scan, batch_size):
            if error_count >= RECO_SCAN_ERROR_LIMIT:
                error = f"Stopped after {error_count} errors (rate-limit protection)"
                logger.error(error)
                break

            batch = _full_syms[batch_start:batch_start + batch_size]
            _batch_num += 1
            with _globals_lock:
                _SCAN_PROGRESS["current_symbol"] = f"batch {_batch_num} ({len(batch)} symbols)"

            gc.collect()
            mem_mb = _memory_usage_mb()
            if mem_mb > _SCAN_MEMORY_HARD_MB:
                error = (
                    f"Memory pressure abort: {mem_mb:.0f}MB > {_SCAN_MEMORY_HARD_MB:.0f}MB hard limit "
                    f"(raise SCAN_MEMORY_HARD_MB if this is a false positive)"
                )
                logger.error("[SCAN] %s", error)
                break
            if mem_mb > _SCAN_MEMORY_SOFT_MB:
                logger.warning(
                    "[SCAN] Memory soft pressure: %.0f MB > %.0f MB — pausing 10s for GC (no abort).",
                    mem_mb, _SCAN_MEMORY_SOFT_MB,
                )
                gc.collect()
                time.sleep(10)
                gc.collect()

            # Warm Kraken ticker map once per batch (reduces per-symbol fetch_ticker pressure).
            if any("/" in s for s in batch):
                try:
                    _tickers_batch_cached(ttl_sec=20)
                except Exception:
                    pass

            if _batch_num > 1:
                _has_stocks = any("/" not in s for s in batch)
                if _rate_limited_count > 5:
                    _sleep_s = SCAN_BATCH_SLEEP_RL_HIGH_SEC
                elif _rate_limited_count > 2:
                    _sleep_s = SCAN_BATCH_SLEEP_RL_MID_SEC
                elif _has_stocks:
                    _sleep_s = SCAN_BATCH_SLEEP_STOCK_SEC
                else:
                    _sleep_s = SCAN_BATCH_SLEEP_CRYPTO_SEC
                if _sleep_s > 0:
                    time.sleep(_sleep_s)

            futures = {
                executor.submit(_scan_one_symbol, sym, horizon, btc_ctx): sym
                for sym in batch
            }

            for future in as_completed(futures):
                sym = futures[future]
                try:
                    result = future.result(timeout=90)
                except Exception as e:
                    error_count += 1
                    logger.warning("Scan future error %s: %s", sym, e)
                    with _globals_lock:
                        errs = _SCAN_PROGRESS.get("recent_errors") or []
                        errs.append({"ts": int(time.time()), "symbol": sym, "error": str(e)[:200]})
                        _SCAN_PROGRESS["recent_errors"] = errs[-10:]
                    continue

                status = result.get("status", "unknown")
                if status == "rate_limited":
                    _rate_limited_count += 1
                    skip_counts[status] = skip_counts.get(status, 0) + 1
                    continue
                elif status == "error":
                    error_count += 1
                    logger.warning("Error scanning %s: %s", sym, result.get("error"))
                    with _globals_lock:
                        errs = _SCAN_PROGRESS.get("recent_errors") or []
                        errs.append({"ts": int(time.time()), "symbol": sym, "error": result.get("error", "")})
                        _SCAN_PROGRESS["recent_errors"] = errs[-10:]
                    continue
                elif status != "ok":
                    skip_counts[status] = skip_counts.get(status, 0) + 1
                    continue

                snap = result["snap"]
                if not snap:
                    continue

                if result.get("signal") == "buy":
                    buy_signals += 1
                if result.get("eligible"):
                    eligible += 1
                # Ensure change_24h is in metrics for explore_scorer at serve time
                metrics = dict(snap.get("metrics") or {})
                if snap.get("change_24h") is not None:
                    metrics["change_24h"] = snap["change_24h"]
                # Persist top-level recommendation fields into metrics_json so they
                # survive in the DB and are available at API serve time for ranking.
                for _persist_key in ("entry_quality", "rsi_value", "volume_ratio",
                                     "confidence_band", "volume_anomaly"):
                    _val = snap.get(_persist_key)
                    if _val is not None:
                        metrics[_persist_key] = _val
                snap["metrics"] = metrics
                _merge_evaluate_signal_into_snap(snap, horizon, btc_ctx)
                metrics = dict(snap.get("metrics") or {})
                _candles_1d = snap.get("_candles_1d") or []
                _snap_id = save_recommendation_snapshot(
                    symbol=snap["symbol"],
                    horizon=horizon,
                    score=float(snap.get("score") or 0.0),
                    regime_json=json.dumps(snap.get("regime") or {}),
                    metrics_json=json.dumps(metrics),
                    reasons_json=json.dumps(snap.get("reasons") or []),
                    risk_flags_json=json.dumps(snap.get("risk_flags") or []),
                    score_breakdown_json=snap.get("score_breakdown_json") or json.dumps(snap.get("score_breakdown") or {}),
                    composite_score=None,
                    confidence_score=None,
                    conviction_grade=None,
                    factor_scores_json="",
                    signal_flags_json="",
                )
                _persist_explore_feed_from_snap(snap["symbol"], horizon, snap, _candles_1d, btc_ctx)
                scanned += 1
                with _globals_lock:
                    s = _RECO_STATE.get(horizon) or {}
                    s["scanned"] = scanned
                    s["eligible"] = eligible
                    _RECO_STATE[horizon] = s
                    _SCAN_PROGRESS["buy_signals_found"] = buy_signals

            _scan_checkpoint_write(horizon, _fp, min(batch_start + batch_size, total_to_scan))
            time.sleep(0.5)

        if not error:
            _scan_checkpoint_clear_for_horizon(horizon)

        scan_dur = time.time() - scan_start
        avg_per_sym = scan_dur / max(scanned, 1)
        logger.info("[SCAN] %s complete: %d/%d scanned, %d eligible, %d buy signals, "
            "%d errors in %.1fs (%.2fs/symbol). Skip reasons: %s",
            horizon, scanned, total_to_scan, eligible, buy_signals,
            error_count, scan_dur, avg_per_sym, skip_counts)

        global _kraken_last_candle_ts, _alpaca_last_candle_ts
        _now_ts = time.time()
        try:
            if scanned > 0:
                _kraken_last_candle_ts = _now_ts
                _alpaca_last_candle_ts = _now_ts
                logger.info("Updated _kraken_last_candle_ts = %s (scanned=%d)", _now_ts, scanned)
            else:
                logger.warning("Scan completed with 0 scanned — candle ts NOT updated")
        except Exception as _te:
            logger.warning("Failed to update candle ts: %s", _te)
        if total_to_scan - scanned - error_count - sum(skip_counts.values()) > 5:
            logger.warning("[SCAN] %s: %d symbols unaccounted for (total=%d scanned=%d errors=%d skipped=%d)",
                horizon, total_to_scan - scanned - error_count - sum(skip_counts.values()),
                total_to_scan, scanned, error_count, sum(skip_counts.values()))

        # Calibration logging: distribution check after scan
        try:
            rows = list_recommendations(horizon=horizon, limit=500)
            cal_buy = sum(1 for r in rows if float(r.get("score") or 0) >= 70)
            cal_watch = sum(1 for r in rows if 45 <= float(r.get("score") or 0) < 70)
            cal_avoid = sum(1 for r in rows if float(r.get("score") or 0) < 45)
            cal_total = len(rows)
            buy_pct = (cal_buy / cal_total * 100) if cal_total else 0
            watch_pct = (cal_watch / cal_total * 100) if cal_total else 0
            logger.info(
                "CALIBRATION [%s]: %d symbols — BUY(70+): %d (%.1f%%) WATCH(45-69): %d (%.1f%%) AVOID(<45): %d (%.1f%%) "
                "Target: 5-15%% BUY, 10-25%% WATCH",
                horizon, cal_total, cal_buy, buy_pct, cal_watch, watch_pct, cal_avoid,
                (cal_avoid / cal_total * 100) if cal_total else 0,
            )
            if cal_total > 20 and buy_pct < 2:
                logger.warning("CALIBRATION WARNING [%s]: <2%% BUY signals — scoring may be too strict", horizon)
            elif cal_total > 20 and buy_pct > 30:
                logger.warning("CALIBRATION WARNING [%s]: >30%% BUY signals — scoring may be too loose", horizon)
        except Exception as ce:
            logger.debug("Calibration logging failed: %s", ce)

    except Exception as e:
        error = f"{type(e).__name__}: {e}"
        logger.error("%s scan failed: %s", horizon, error, exc_info=True)

    scan_end = time.time()
    with _globals_lock:
        _RECO_STATE[horizon] = {
            "last_run_ts": now, "error": error, "btc_ctx": btc_ctx,
            "scanned": scanned, "eligible": eligible,
            "total": total_to_scan, "scanning": False,
            "started_at": _scan_started_at,
        }
        _RECO_RESULT_CACHE.clear()  # Force fresh scoring on next Explore load
        if horizon == "short": _last_reco_short_ts = time.time()
        elif horizon == "medium": _last_reco_medium_ts = time.time()
        else: _last_reco_long_ts = time.time()
        _SCAN_PROGRESS["current_symbol"] = ""
        _SCAN_PROGRESS["current_horizon"] = ""
        history = _SCAN_PROGRESS.get("scan_history") or []
        history.append({
            "horizon": horizon,
            "ts": int(scan_end),
            "scanned": scanned,
            "eligible": eligible,
            "buy_signals": buy_signals if 'buy_signals' in dir() else 0,
            "duration_sec": round(scan_end - (scan_start if 'scan_start' in dir() else scan_end), 1),
            "error": error,
        })
        _SCAN_PROGRESS["scan_history"] = history[-_SCAN_HISTORY_MAX:]

    # Auto-trigger backtest after every completed scan:
    #   - always on first scan (no stored result)
    #   - daily refresh when stored result is >24h old
    if not error:
        try:
            existing = get_latest_explore_backtest(horizon)
            _needs_bt = (
                not existing
                or (time.time() - float(existing.get("computed_ts", 0))) > 86400
            )
            if _needs_bt:
                _bt_thread_name = f"auto_backtest_{horizon}"
                # Don't stack: only start if no backtest thread already running for this horizon
                _bt_running = any(
                    t.name == _bt_thread_name
                    for t in threading.enumerate()
                    if t.is_alive()
                )
                if not _bt_running:
                    threading.Thread(
                        target=_run_missing_backtests,
                        kwargs={"horizons": (horizon,), "force": True},
                        daemon=True,
                        name=_bt_thread_name,
                    ).start()
                    logger.info("[BACKTEST] Auto-triggered %s backtest (first=%s, stale=%s)",
                        horizon, not existing,
                        bool(existing and (time.time() - float(existing.get("computed_ts", 0))) > 86400))
        except Exception as _bt_trigger_err:
            logger.warning("auto_backtest trigger failed: %s", _bt_trigger_err)

    # Periodic DB cleanup: delete old recommendation snapshots (older than 7 days)
    # to keep the DB from growing unbounded. Only run once per day.
    try:
        _now_cleanup = time.time()
        _last_cleanup = _RECO_STATE.get("_last_snapshot_cleanup_ts", 0)
        if _now_cleanup - _last_cleanup > 86400:
            from db import cleanup_old_recommendation_snapshots
            _deleted = cleanup_old_recommendation_snapshots(keep_days=7)
            if _deleted > 0:
                logger.info("Cleaned up %d old recommendation snapshots (>7 days)", _deleted)
            with _globals_lock:
                _RECO_STATE["_last_snapshot_cleanup_ts"] = _now_cleanup
    except Exception as _cleanup_err:
        logger.warning("recommendation snapshot cleanup failed: %s", _cleanup_err)

    try:
        gc.collect()
    except Exception:
        pass


def _scan_recommendations(horizon: str) -> None:
    """Single-horizon scan. Uses per-horizon lock so horizons run in parallel."""
    with _HORIZON_SCAN_LOCKS[horizon]:
        if _HORIZON_SCANNING.get(horizon):
            logger.warning("[SCAN] Skipped %s: already scanning", horizon)
            return
        _HORIZON_SCANNING[horizon] = True
    try:
        _scan_recommendations_impl(horizon)
        if horizon == "short":
            try:
                _warm_explore_feed_cache()
            except Exception as _wfc_err:
                logger.warning("explore feed cache warm after scan failed: %s", _wfc_err)
    except Exception as e:
        logger.error("[SCAN] %s scan failed: %s", horizon, e, exc_info=True)
    finally:
        _HORIZON_SCANNING[horizon] = False


def _fire_horizon_scan(horizon: str) -> bool:
    """
    Fire a scan for one horizon in a background thread.
    Returns True if a thread was started, False if already scanning.
    """
    with _HORIZON_SCAN_LOCKS[horizon]:
        if _HORIZON_SCANNING.get(horizon):
            logger.debug("[SCAN] %s already scanning — skip fire", horizon)
            return False
        _HORIZON_SCANNING[horizon] = True

    def _run():
        try:
            while True:
                try:
                    _scan_recommendations_impl(horizon)
                    break
                except Exception as e:
                    logger.error(
                        "Scan worker [%s] crashed: %s. Restarting in 30s.",
                        horizon,
                        e,
                        exc_info=True,
                    )
                    time.sleep(30)
            if horizon == "short":
                try:
                    _warm_explore_feed_cache()
                except Exception as _wfc_err:
                    logger.warning("explore feed cache warm after scan failed: %s", _wfc_err)
        except Exception as e:
            logger.error("[SCAN] %s horizon thread failed: %s", horizon, e, exc_info=True)
        finally:
            _HORIZON_SCANNING[horizon] = False

    threading.Thread(target=_run, daemon=True, name=f"scan_{horizon}").start()
    logger.info("[SCAN] Fired parallel scan thread for %s horizon", horizon)
    return True


def _scan_all_horizons(horizons: List[str]) -> set:
    """
    Fire each horizon in its own background thread (parallel, not sequential).
    Returns the set of horizons where a thread was successfully started.
    """
    fired: set = set()
    for h in horizons:
        if _fire_horizon_scan(h):
            fired.add(h)
    return fired


def _backtest_fetch_candles(sym: str) -> list:
    """Shared OHLCV fetcher for explore backtests (yfinance for stocks, Kraken for crypto)."""
    try:
        if "/" not in sym:
            try:
                import yfinance as yf
                hist = yf.Ticker(sym).history(period="1y", interval="1d")
                if hist is not None and not hist.empty:
                    return [[int(ts.timestamp()), float(r["Open"]), float(r["High"]),
                             float(r["Low"]), float(r["Close"]), float(r["Volume"])]
                            for ts, r in hist.iterrows()]
            except Exception:
                pass
            return []
        else:
            if kc is None:
                return []
            now_ms = int(time.time() * 1000)
            return list(kc.fetch_ohlcv_range(sym, "1d", now_ms - 365 * 86400 * 1000, now_ms) or [])
    except Exception:
        return []


def _run_missing_backtests(horizons: tuple = ("short", "medium", "long"),
                           force: bool = False) -> None:
    """Run explore backtest for specified horizons.
    force=False (default): skip horizons that already have a stored result.
    force=True: always refresh (used after a fresh scan).
    """
    try:
        from explore_backtest import default_universe_symbols, run_explore_backtest
        stocks, crypto = default_universe_symbols()
    except Exception as e:
        logger.warning("[BACKTEST] import failed, skipping: %s", e)
        return
    for h in horizons:
        try:
            if not force and get_latest_explore_backtest(h):
                continue  # already have a result; skip unless forced
            logger.info("[BACKTEST] Running %s horizon backtest (force=%s)", h, force)
            res = run_explore_backtest(fetch_candles=_backtest_fetch_candles,
                                       stock_symbols=stocks, crypto_symbols=crypto, horizon=h)
            save_explore_backtest_results(h, res)
            logger.info("[BACKTEST] %s horizon backtest saved (win_rate=%.1f%%)",
                        h, float(res.get("win_rate_90d") or 0) * 100)
        except Exception as _e:
            logger.warning("[BACKTEST] %s horizon failed: %s", h, _e)


def _warm_explore_feed_cache() -> None:
    """Pre-populate _EXPLORE_FEED_CACHE with lightweight explore_signals data."""
    items = _get_explore_feed_items(horizon="short", limit=80, signal_filter="all", market_type="all")
    now_i = int(time.time())
    _ef_response = {
        "ok": True,
        "status": "ready",
        "reason": "ok",
        "message": "Explore feed (cached at startup)",
        "horizon": "short",
        "items": items,
        "count": len(items),
        "has_more": False,
        "last_scan_ts": now_i,
        "scan_age_sec": 0,
        "last_scan_by_horizon": {},
        "explore_rejected": [],
        "cache_ts": now_i,
        "explore_smart_rank": False,
        "explore_disclaimer": "",
        "market_conditions": {},
    }
    _ef_cache_key = "short|all|all|0"
    _EXPLORE_FEED_CACHE[_ef_cache_key] = (time.time(), _ef_response)
    logger.info("explore feed cache warmed: %d items", len(items))


def _explore_startup_sequential_scan() -> None:
    """After boot, fire all 3 horizons in parallel so all populate without user action."""
    time.sleep(22.0)
    try:
        _warm_explore_feed_cache()
    except Exception as e:
        logger.warning("explore feed cache warm failed: %s", e)
    logger.warning("[SCAN-DEBUG] startup scan: firing short + medium + long in parallel")
    try:
        n = delete_recommendations_for_blocklist(list(CRYPTO_BLOCKLIST))
        if n > 0:
            logger.warning("Purged %d blocklisted recommendation(s) before startup scan", n)
    except Exception:
        pass
    for h in ("short", "medium", "long"):
        _fire_horizon_scan(h)
    # Backtest for all horizons — force refresh to include new strategies
    _start_background_thread(
        "startup_backtest",
        lambda: _run_missing_backtests(
            horizons=("short", "medium", "long"),
            force=True  # force refresh to include new strategies
        )
    )


_BACKOFF_BASE = 1.0
_BACKOFF_MAX = 30.0
_CIRCUIT_BREAKER_THRESHOLD = 15


def _retry_with_backoff(func, max_retries: int = 3, base_delay: float = 0.5):
    """
    Retry a function call with exponential backoff.
    Useful for rate-limited or transient Alpaca API calls.

    Args:
        func: Callable that performs the API call
        max_retries: Max attempts (3 = try up to 4 times)
        base_delay: Initial delay in seconds

    Returns:
        Result of func if successful, or raises the last exception if all retries fail
    """
    last_exception = None
    delay = base_delay

    for attempt in range(max_retries + 1):
        try:
            return func()
        except Exception as e:
            last_exception = e
            if attempt < max_retries:
                logger.debug(f"API call failed (attempt {attempt + 1}/{max_retries + 1}), retrying in {delay}s: {e}")
                time.sleep(delay)
                delay = min(_BACKOFF_MAX, delay * 2)  # Exponential backoff
            else:
                logger.warning(f"API call failed after {max_retries + 1} attempts: {e}")

    raise last_exception if last_exception else RuntimeError("Unexpected retry failure")


def _recommendations_loop() -> None:
    """
    Scheduler loop: check every 30s whether any horizon is due, fire parallel scan threads.
    Startup scan already populates all 3 horizons; this loop keeps them fresh on schedule.
    Sleeps 30s between checks (was 5s) to reduce idle CPU from ~130% to baseline.
    """
    logger.info("[RECO_DEBUG] _recommendations_loop STARTED")
    last_short = 0
    last_medium = 0
    last_long = 0
    _last_outcomes_ts = 0.0
    _last_evict_ts = 0.0
    fail_count = 0
    while True:
        try:
            now = int(time.time())

            # Fire each due horizon in its own background thread (parallel, non-blocking)
            if now - last_short >= RECO_SHORT_EVERY_SEC:
                if _fire_horizon_scan("short"):
                    last_short = now
                elif not _HORIZON_SCANNING.get("short"):
                    last_short = now  # already-scanning case — advance to prevent spin

            if now - last_medium >= RECO_MEDIUM_EVERY_SEC:
                if _fire_horizon_scan("medium"):
                    last_medium = now
                elif not _HORIZON_SCANNING.get("medium"):
                    last_medium = now

            if now - last_long >= RECO_LONG_EVERY_SEC:
                if _fire_horizon_scan("long"):
                    last_long = now
                elif not _HORIZON_SCANNING.get("long"):
                    last_long = now

            # Housekeeping — only once per hour to avoid constant GC/IO overhead
            now_f = time.time()
            if now_f - _last_evict_ts >= 3600:
                evicted = _evict_ohlcv_cache()
                if evicted > 0:
                    logger.info("[SCAN] Evicted %d stale OHLCV entries (remaining: %d)",
                        evicted, len(_RECO_OHLCV_CACHE))
                gc.collect()
                _last_evict_ts = now_f
            if now_f - _last_outcomes_ts >= 3600:
                try:
                    _run_explore_outcomes_update_batch()
                except Exception as _eo_err:
                    logger.warning("explore outcomes batch: %s", _eo_err)
                _last_outcomes_ts = now_f

            fail_count = 0
        except Exception:
            logger.exception("_recommendations_loop: iteration failed")
            fail_count += 1
            if fail_count >= _CIRCUIT_BREAKER_THRESHOLD and DISCORD_WEBHOOK_URL:
                try:
                    import requests
                    requests.post(DISCORD_WEBHOOK_URL, json={"content": f"⚠️ Recommendations loop: {fail_count} consecutive failures. Degraded."}, timeout=3)
                except Exception:
                    pass
        time.sleep(30)  # 30s idle check — reduces CPU from ~130% to baseline


def _ml_retrain_loop() -> None:
    """Weekly ML retrain (walk-forward, deploy only if validation >60%).
    Also auto-collects training data from explore signal outcomes when training set is too small.
    """
    import os
    freq_days = int(os.getenv("ML_RETRAIN_FREQUENCY", "7"))
    interval_sec = max(86400, freq_days * 86400)
    last_run = 0
    while True:
        try:
            time.sleep(3600)
            now = int(time.time())
            if now - last_run >= interval_sec:
                try:
                    from ml_ensemble import get_ml_ensemble
                    ensemble = get_ml_ensemble()

                    # Auto-collect training data from signal outcomes if training set is small
                    n_data = len(getattr(ensemble, "_training_data", []))
                    if n_data < 100:
                        try:
                            from db import get_explore_signal_outcomes_for_training
                            outcomes = get_explore_signal_outcomes_for_training(limit=500)
                            if outcomes and len(outcomes) > 0:
                                for row in outcomes:
                                    try:
                                        ensemble.add_training_sample(row)
                                    except Exception:
                                        pass
                                logger.info("ML auto-collected %d training samples from signal outcomes (had %d)",
                                           len(outcomes), n_data)
                        except ImportError:
                            logger.debug("get_explore_signal_outcomes_for_training not available")
                        except Exception as _atd_err:
                            logger.debug("ML auto-collect training data: %s", _atd_err)

                    n_data = len(getattr(ensemble, "_training_data", []))
                    min_samples = int(os.getenv("ML_MIN_TRAINING_SAMPLES", "50"))
                    if n_data >= min_samples:
                        success = ensemble.train(force=True)
                        if success:
                            last_run = now
                            min_acc = float(os.getenv("ML_MIN_ACCURACY", "0.55"))
                            best = max(
                                getattr(ensemble._model_performance.get("xgb"), "recent_accuracy", 0) or 0,
                                getattr(ensemble._model_performance.get("rf"), "recent_accuracy", 0) or 0,
                            )
                            if best >= min_acc:
                                from db import save_ml_model_version
                                save_ml_model_version("ensemble", f"v{now}", best, deployed=True)
                                logger.info("ML retrain: deployed ensemble v%s (acc=%.2f%%)", now, best * 100)
                            else:
                                logger.info("ML retrain: accuracy %.2f%% below threshold %.2f%%, not deploying",
                                           best * 100, min_acc * 100)
                    else:
                        logger.debug("ML retrain: only %d training samples (need %d), skipping", n_data, min_samples)
                except Exception as e:
                    logger.debug("ML retrain loop: %s", e)
        except Exception as e:
            logger.debug("ML retrain loop error: %s", e)


def _ml_outcomes_loop() -> None:
    """Daily job: update ML predictions with actual 7d/30d outcomes."""
    last_run = 0
    while True:
        try:
            time.sleep(3600)
            now = int(time.time())
            if now - last_run >= 86400:
                try:
                    from ml_prediction_tracker import update_outcomes_job
                    update_outcomes_job()
                    last_run = now
                except Exception as e:
                    logger.debug("ML outcomes job: %s", e)
        except Exception:
            pass


def _screener_outcomes_loop() -> None:
    """
    Hourly job: track screener recommendation accuracy.
    For each snapshot >= 24h old without an outcome, fetch current price,
    compute return, and write win/loss to recommendation_performance.
    Only tracks 'buy' signal recs with score > 50.

    Phase 1.2c migration:
      * Stopped opening a raw sqlite3 connection (no PRAGMAs, bypassed
        write_txn). Reads now go through db._conn() (per-thread cached
        with WAL/busy_timeout pragmas applied), inserts go through
        db.write_txn(None, _do, name=...).
      * Inner ``except Exception as e: logger.debug(...)`` swallow has
        been replaced with logger.exception + _loop_health_err so the
        operator can see persistent failures via /health/full.
      * Outer ``except Exception: pass`` (which masked everything,
        including the OperationalError loop) is gone — the loop body
        records health and continues.
    """
    import json as _js
    LOOP = "screener_outcomes"
    last_run = 0
    while True:
        try:
            time.sleep(1800)  # run every 30 min
            now = int(time.time())
            if now - last_run < 3600:
                continue
            last_run = now
            if not _kraken_ready():
                # Not failing — Kraken not ready is a known operational
                # condition; just record liveness and loop.
                _loop_health_ok(LOOP)
                continue

            cutoff_lo = now - 7 * 86400   # don't go older than 7d
            cutoff_hi = now - 24 * 3600   # must be at least 24h old

            from db import _conn as _read_conn, write_txn  # local: avoid import cycle at module top

            con = _read_conn()
            rows = con.execute(
                """
                SELECT id, symbol, score, regime_json, metrics_json, created_ts
                FROM recommendations_snapshots
                WHERE created_ts BETWEEN ? AND ?
                ORDER BY id DESC LIMIT 500
                """,
                (cutoff_lo, cutoff_hi),
            ).fetchall()
            existing_rows = con.execute(
                "SELECT symbol, recommendation_date FROM recommendation_performance "
                "WHERE outcome IN ('price_up','price_down') AND recommendation_date >= ?",
                (cutoff_lo,),
            ).fetchall()
            existing = {
                (er["symbol"], er["recommendation_date"] // 3600 * 3600)
                for er in existing_rows
            }

            # Build the candidate list outside the write_txn so price
            # fetches (network I/O) don't hold the global write lock.
            candidates: List[Tuple[Any, ...]] = []
            for row in rows:
                sym = row["symbol"]
                rec_ts = int(row["created_ts"] or 0)
                bucket = rec_ts // 3600 * 3600
                if (sym, bucket) in existing:
                    continue
                try:
                    metrics = _js.loads(row["metrics_json"] or "{}")
                except Exception:
                    logger.exception("screener_outcomes: bad metrics_json for %s", sym)
                    continue
                rec_price = float(metrics.get("price") or 0)
                signal = metrics.get("signal", "")
                score = float(row["score"] or 0)
                if score < 50 or signal not in ("buy",) or rec_price <= 0:
                    continue
                try:
                    cur_price = float(kc.fetch_ticker_last(sym))
                except Exception:
                    # Per-symbol fetch failures are expected (delisted,
                    # rate-limited). Skip the row but DO NOT mark the
                    # whole loop as failed.
                    logger.debug("screener_outcomes: ticker fetch failed for %s", sym)
                    continue
                ret_pct = (cur_price - rec_price) / rec_price * 100 if rec_price > 0 else 0
                outcome = "price_up" if ret_pct > 0 else "price_down"
                regime = ""
                try:
                    rj = _js.loads(row["regime_json"] or "{}")
                    regime = rj.get("label", "")
                except Exception:
                    logger.exception("screener_outcomes: bad regime_json for %s", sym)
                candidates.append((
                    sym, rec_ts, score, regime,
                    rec_price, cur_price,
                    round(ret_pct, 4),
                    round((now - rec_ts) / 86400, 2),
                    outcome,
                    f"auto-tracked {round(ret_pct, 2)}% return",
                    now,
                    sym, bucket,  # for existing-set update
                ))

            inserted = 0
            if candidates:
                def _do(con) -> int:
                    n = 0
                    for c in candidates:
                        con.execute(
                            """
                            INSERT OR IGNORE INTO recommendation_performance(
                                symbol, recommendation_date, score_at_recommendation,
                                regime_at_recommendation, entry_price, exit_price,
                                pnl_realized, days_held, outcome, notes, created_at
                            ) VALUES (?,?,?,?,?,?,?,?,?,?,?)
                            """,
                            c[:11],
                        )
                        n += 1
                    return n
                inserted = write_txn(None, _do, name="screener_outcomes_insert")

            if inserted:
                logger.info("Screener outcome tracker: recorded %d outcomes", inserted)
            _loop_health_ok(LOOP)
        except Exception as e:
            logger.exception("_screener_outcomes_loop: iteration failed")
            _loop_health_err(LOOP, e)


def _portfolio_snapshot() -> Dict[str, Any]:
    """
    Best-effort portfolio estimation in USD from Kraken balances.
    Safe to call even when Kraken isn't ready.
    """
    if not _kraken_ready():
        with _globals_lock:
            latest = PORT_HISTORY[-1]["total_usd"] if PORT_HISTORY else 0.0
        return {
            "total_usd": float(latest),
            "free_usd": 0.0,
            "used_usd": 0.0,
            "positions_usd": 0.0,
            "holdings": [],
            "error": KRAKEN_ERROR or "Kraken not ready",
        }

    try:
        bal = kc.fetch_balance()
        total = (bal.get("total", {}) or {})
        free = (bal.get("free", {}) or {})
        used = (bal.get("used", {}) or {})
        mk = _markets()

        symbols_usd = set(
            s for s, m in mk.items()
            if m.get("spot") and m.get("active") and m.get("quote") == "USD"
        )

        def price_for_asset(asset: str) -> Optional[float]:
            if asset in ("USD", "ZUSD"):
                return 1.0
            # Normalize Kraken internal names: XXBT->XBT, XETH->ETH, etc.
            a = asset
            if a == "BTC":
                a = "XBT"
            elif a.startswith("X") and len(a) == 4 and a != "XBT":
                a = a[1:]  # XETH -> ETH, XLTC -> LTC
            elif a.startswith("Z") and len(a) == 4:
                a = a[1:]  # ZUSD -> USD (already handled above)
            sym = f"{a}/USD"
            if sym in symbols_usd:
                try:
                    return float(kc.fetch_ticker_last(sym))
                except Exception:
                    return None
            # Fallback: try BTC/USD for BTC variants (ccxt Kraken uses BTC not XBT)
            if a in ("BTC", "XBT", "XXBT"):
                try:
                    return float(kc.fetch_ticker_last(_resolve_symbol("BTC/USD")))
                except Exception:
                    return None
            return None

        holdings: List[Dict[str, Any]] = []
        total_usd = 0.0
        free_usd = 0.0
        used_usd = 0.0
        positions_usd = 0.0

        for asset, amt in total.items():
            a_total = float(amt or 0.0)
            a_free = float(free.get(asset, 0.0) or 0.0)
            a_used = float(used.get(asset, 0.0) or 0.0)
            if a_total <= 0 and a_free <= 0 and a_used <= 0:
                continue

            p = price_for_asset(str(asset))
            usd_total = a_total * p if p is not None else None
            usd_free = a_free * p if p is not None else None
            usd_used = a_used * p if p is not None else None

            if usd_total is not None:
                total_usd += float(usd_total)
            if usd_free is not None:
                free_usd += float(usd_free)
            if usd_used is not None:
                used_usd += float(usd_used)

            if str(asset) != "USD" and usd_total is not None:
                positions_usd += float(usd_total)

            holdings.append(
                {
                    "asset": str(asset),
                    "amount": a_total,
                    "free": a_free,
                    "used": a_used,
                    "usd_value": usd_total,
                    "usd_free": usd_free,
                    "usd_used": usd_used,
                }
            )

        holdings.sort(key=lambda x: (x["usd_value"] or 0.0), reverse=True)
        return {
            "total_usd": float(total_usd),
            "free_usd": float(free_usd),
            "used_usd": float(used_usd),
            "positions_usd": float(positions_usd),
            "holdings": holdings,
            "error": "",
        }
    except Exception as e:
        logger.exception("_portfolio_snapshot: Kraken/balance fetch failed")
        with _globals_lock:
            latest = PORT_HISTORY[-1]["total_usd"] if PORT_HISTORY else 0.0
        return {
            "total_usd": float(latest),
            "free_usd": 0.0,
            "used_usd": 0.0,
            "positions_usd": 0.0,
            "holdings": [],
            "error": f"{type(e).__name__}: {e}",
        }


def _portfolio_loop():
    """Portfolio sampler — appends to in-memory history and writes to
    portfolio_snapshots once per iteration.

    Phase 1.2c migration:
      * INSERT now goes through db.write_txn(None, ...) so the global
        write lock is acquired and a per-iteration retry/backoff
        applies on 'database is locked'.
      * The ``except Exception: pass`` swallow around the DB write is
        gone; failures log via logger.exception and update
        _BACKGROUND_LOOP_HEALTH so /health/full surfaces them.
      * cleanup_old_portfolio_snapshots is called via the chunked path
        added in Phase 1.2b step 8 — the inner try/except still bounds
        cleanup failures so a janitor failure doesn't take the
        sampler down.
    """
    global _last_portfolio_ts
    LOOP = "portfolio"
    fail_count = 0
    backoff = _BACKOFF_BASE
    while True:
        try:
            snap = _portfolio_snapshot()
            total_usd = float(snap.get("total_usd") or 0.0)
            ts = now_ts()
            with _globals_lock:
                PORT_HISTORY.append({"ts": ts, "total_usd": total_usd})
                if len(PORT_HISTORY) > 2000:
                    del PORT_HISTORY[:200]
                _last_portfolio_ts = time.time()

            try:
                from db import write_txn, cleanup_old_portfolio_snapshots
                positions_count = int(snap.get("positions_count", 0))

                def _do(con) -> None:
                    con.execute(
                        "INSERT INTO portfolio_snapshots (total_value, total_pnl, active_positions, unrealized_pnl) "
                        "VALUES (?, ?, ?, ?)",
                        (total_usd, 0.0, positions_count, 0.0),
                    )
                write_txn(None, _do, name="portfolio_snapshot_insert")

                # Run cleanup once per day to keep last 90 days. Bounded
                # try so a janitor failure doesn't stop the sampler.
                now_float = time.time()
                if now_float - _last_portfolio_cleanup_ts >= 86400:
                    try:
                        cleanup_old_portfolio_snapshots(keep_days=90)
                        globals()["_last_portfolio_cleanup_ts"] = now_float
                    except Exception:
                        logger.exception("_portfolio_loop: cleanup_old_portfolio_snapshots failed")
            except Exception as e:
                # The DB write failed. We do NOT silently continue —
                # surface it to the loop-level health and let the
                # outer except retry with backoff.
                logger.exception("_portfolio_loop: snapshot persist failed")
                _loop_health_err(LOOP, e)
                fail_count += 1
                backoff = min(_BACKOFF_MAX, backoff * 2)
                time.sleep(max(5, PORT_EVERY_SEC, int(backoff)))
                continue

            fail_count = 0
            backoff = _BACKOFF_BASE
            _loop_health_ok(LOOP)
        except Exception as e:
            logger.exception("_portfolio_loop: iteration failed")
            _loop_health_err(LOOP, e)
            fail_count += 1
            backoff = min(_BACKOFF_MAX, backoff * 2)
        time.sleep(max(5, PORT_EVERY_SEC, int(backoff)))


def _discord_notify(message: str) -> None:
    if not DISCORD_WEBHOOK_URL:
        return
    try:
        import requests
        requests.post(DISCORD_WEBHOOK_URL, json={"content": message}, timeout=3)
    except Exception:
        logger.exception("_discord_notify: webhook post failed")


def _discord_status_update(message: str) -> None:
    if not DISCORD_STATUS_WEBHOOK_URL:
        return
    try:
        import requests
        try:
            os.makedirs(os.path.dirname(DISCORD_STATUS_MSG_FILE), exist_ok=True)
        except Exception:
            pass
        try:
            with open(DISCORD_STATUS_LOG, "a", encoding="utf-8") as f:
                f.write(f"[status] update start\n")
        except Exception:
            pass
        msg_id = None
        try:
            if os.path.exists(DISCORD_STATUS_MSG_FILE):
                with open(DISCORD_STATUS_MSG_FILE, "r", encoding="utf-8") as f:
                    msg_id = f.read().strip() or None
        except Exception:
            msg_id = None

        if msg_id:
            url = f"{DISCORD_STATUS_WEBHOOK_URL}/messages/{msg_id}"
            r = requests.patch(url, json={"content": message}, timeout=3)
            if r.ok:
                try:
                    with open(DISCORD_STATUS_LOG, "a", encoding="utf-8") as f:
                        f.write(f"[status] patched {msg_id}\n")
                except Exception:
                    pass
                return
            if r.status_code != 404:
                try:
                    with open(DISCORD_STATUS_LOG, "a", encoding="utf-8") as f:
                        f.write(f"[status] patch failed {r.status_code}\n")
                except Exception:
                    pass
                return
            try:
                if os.path.exists(DISCORD_STATUS_MSG_FILE):
                    os.remove(DISCORD_STATUS_MSG_FILE)
            except Exception:
                pass

        post_url = f"{DISCORD_STATUS_WEBHOOK_URL}?wait=true"
        r = requests.post(post_url, json={"content": message}, timeout=3)
        if r.ok:
            data = r.json()
            mid = str(data.get("id") or "")
            if mid:
                try:
                    with open(DISCORD_STATUS_MSG_FILE, "w", encoding="utf-8") as f:
                        f.write(mid)
                except Exception:
                    pass
                try:
                    with open(DISCORD_STATUS_LOG, "a", encoding="utf-8") as f:
                        f.write(f"[status] posted {mid}\n")
                except Exception:
                    pass
    except Exception as e:
        logger.exception("_discord_status_update failed: %s", e)


def _discord_status_loop() -> None:
    last_state: Dict[int, bool] = {}
    fail_count = 0
    backoff = _BACKOFF_BASE
    # initial summary
    try:
        bots = list_bots()
        running_ids = []
        if bm is not None:
            for b in bots:
                snap = bm.snapshot(int(b.get("id")))
                is_running = bool(snap.get("running"))
                last_state[int(b.get("id"))] = is_running
                if is_running:
                    running_ids.append(str(b.get("name") or b.get("id")))
        lines = []
        for b in bots:
            bid = int(b.get("id"))
            name = b.get("name") or bid
            state = "🟢 live" if last_state.get(bid) else "⚪ idle"
            lines.append(f"{name}: {state}")
            _discord_status_update("**Bot status**\n" + "\n".join(lines) if lines else "**Bot status**\n(no bots)")
    except Exception:
        logger.exception("_discord_status_loop: initial summary failed")

    while True:
        try:
            if bm is None:
                time.sleep(5)
                continue
            bots = list_bots()
            for b in bots:
                bot_id = int(b.get("id"))
                snap = bm.snapshot(bot_id)
                is_running = bool(snap.get("running"))
                if bot_id not in last_state or last_state[bot_id] != is_running:
                    last_state[bot_id] = is_running
                    # Optional: send individual start/stop notifications (default off to reduce spam)
                    if os.getenv("DISCORD_STATUS_NOTIFY_CHANGES", "0").strip().lower() in ("1", "true", "yes"):
                        name = b.get("name") or bot_id
                        msg = f"✅ {name} is running." if is_running else f"🛑 {name} stopped."
                        _discord_notify(msg)
            lines = []
            for b in bots:
                bid = int(b.get("id"))
                name = b.get("name") or bid
                state = "🟢 live" if last_state.get(bid) else "⚪ idle"
                lines.append(f"{name}: {state}")
            _discord_status_update("**Bot status**\n" + "\n".join(lines) if lines else "**Bot status**\n(no bots)")
            fail_count = 0
            backoff = _BACKOFF_BASE
        except Exception:
            logger.exception("_discord_status_loop: iteration failed")
            fail_count += 1
            backoff = min(_BACKOFF_MAX, backoff * 1.5)
        time.sleep(max(5, int(backoff)))


def _pause_state() -> bool:
    env = os.getenv("PAUSE_ALL_BOTS", "").strip().lower()
    if env in ("1", "true", "yes", "y", "on"):
        return True
    try:
        v = get_setting("global_pause", "0")
        if str(v).strip().lower() in ("1", "true", "yes", "y", "on"):
            until = get_setting("global_pause_until", "0")
            try:
                until_ts = int(until or 0)
            except Exception:
                until_ts = 0
            if until_ts and until_ts <= int(time.time()):
                set_setting("global_pause", "0")
                set_setting("global_pause_until", "0")
                return False
            return True
        return False
    except Exception:
        return False


def _kill_switch_state() -> bool:
    env = os.getenv("KILL_SWITCH", "").strip().lower()
    if env in ("1", "true", "yes", "y", "on"):
        return True
    try:
        v = get_setting("kill_switch", "0")
        return str(v).strip().lower() in ("1", "true", "yes", "y", "on")
    except Exception:
        return False


def _should_autostart(bot: Dict[str, Any]) -> bool:
    """Return True if the bot is eligible for auto-start on server boot.

    Bug 8 fix: This is checked once at startup, not on every watchdog cycle.
    """
    try:
        if int(bot.get("last_running", 0)) == 1:
            return True
        if AUTO_START_ENABLED and int(bot.get("enabled", 0)) == 1:
            return True
    except Exception:
        return False
    return False


def _watchdog_should_restart(bot: Dict[str, Any]) -> bool:
    """Return True only when the watchdog is allowed to (re)start a stopped bot.

    Bug 8 fix: when ``auto_restart`` is explicitly disabled the watchdog must
    NOT restart the bot every cycle. We still allow a single recovery start at
    boot via the autostart loop. After that, we only restart bots that have
    opted in via ``auto_restart=1``.
    """
    try:
        if int(bot.get("auto_restart", 0)) == 1 and int(bot.get("enabled", 0)) == 1:
            return True
    except Exception:
        pass
    return False


def _autostart_loop() -> None:
    # Wait for BotManager initialization before attempting autostart
    time.sleep(15)
    attempts = 0
    max_attempts = 36
    while attempts < max_attempts:
        try:
            if bm is None:
                time.sleep(5)
                attempts += 1
                continue

            bots = list_bots()
            pending_live = False
            if _pause_state():
                time.sleep(5)
                attempts += 1
                continue
            started_count = 0
            for b in bots:
                if not _should_autostart(b):
                    continue
                bot_id = int(b.get("id"))
                try:
                    snap = bm.snapshot(bot_id)
                    if bool(snap.get("running")):
                        continue
                except Exception as bot_err:
                    logger.debug("autostart: snapshot failed for bot %s: %s", bot_id, bot_err)
                    continue
                # Safety: skip bots with CRITICAL risk flag
                if snap and snap.get("risk_level") == "CRITICAL":
                    logger.info("autostart: skipping bot %d — CRITICAL risk flag active", bot_id)
                    continue
                # Safety: skip bots with $0 budget
                _budget = float(b.get("base_order_size") or b.get("base_quote") or 0)
                if _budget <= 0:
                    logger.info("autostart: skipping bot %d — no budget allocated", bot_id)
                    continue
                ok, reason = _can_start_bot_live(b)
                if not ok:
                    pending_live = True
                    if attempts < 3:
                        logger.info("autostart: waiting for readiness bot_id=%s reason=%s", bot_id, reason)
                    continue
                try:
                    bm.start(bot_id, silent=True)
                    started_count += 1
                except Exception as start_err:
                    logger.warning("autostart: failed to start bot %s: %s", bot_id, start_err)
            if started_count > 0:
                logger.info("autostart: started %d bots on attempt %d", started_count, attempts + 1)
            if not pending_live:
                logger.info("autostart: complete — all eligible bots started")
                return
        except Exception as e:
            logger.warning("autostart: iteration %d failed: %s", attempts + 1, e)
        time.sleep(5)
        attempts += 1
    logger.info("autostart: finished after %d attempts (some bots may still be pending)", max_attempts)


_WATCHDOG_STICKY_ERRORS: Dict[int, Dict[str, Any]] = {}

# Bug 8: Minimum gap between watchdog restarts of the same bot. Default 5min.
WATCHDOG_MIN_RESTART_INTERVAL_SEC = int(os.getenv("WATCHDOG_MIN_RESTART_INTERVAL_SEC", "300"))


def _health_watchdog_loop() -> None:
    global _WATCHDOG_STICKY_ERRORS
    while True:
        try:
            if bm is None:
                time.sleep(HEALTH_WATCHDOG_SEC)
                continue
            if _pause_state():
                time.sleep(HEALTH_WATCHDOG_SEC)
                continue
            bots = list_bots()
            now = int(time.time())
            for b in bots:
                # Bug 8: Only restart bots that opted in via auto_restart=1.
                # The previous behaviour restarted any enabled bot every ~60s
                # which thrashed open deals.
                if not _watchdog_should_restart(b):
                    continue
                bot_id = int(b.get("id"))
                snap = bm.snapshot(bot_id)
                if not bool(snap.get("running")):
                    bot = get_bot(bot_id)
                    if bot and int(bot.get("enabled", 0)) == 1:
                        last_restart = getattr(_health_watchdog_loop, f"_last_restart_{bot_id}", 0)
                        fail_count = getattr(_health_watchdog_loop, f"_fail_count_{bot_id}", 0)
                        if (now - last_restart) > WATCHDOG_MIN_RESTART_INTERVAL_SEC:
                            ok, reason = _can_start_bot_live(b)
                            if ok:
                                try:
                                    last_err = str(snap.get("last_error") or "").strip()
                                    bm.start(bot_id, silent=True)
                                    setattr(_health_watchdog_loop, f"_last_restart_{bot_id}", now)
                                    setattr(_health_watchdog_loop, f"_fail_count_{bot_id}", 0)
                                    _WATCHDOG_STICKY_ERRORS.pop(bot_id, None)
                                    detail = f" Reason: {last_err}" if last_err else ""
                                    # add_log can hit "database is locked" during a startup
                                    # write storm; swallow that so we don't take the watchdog
                                    # down. journalctl still records the restart.
                                    try:
                                        add_log(bot_id, "WARN", f"Watchdog restarted bot (auto_restart=1).{detail}", "SYSTEM")
                                    except Exception as log_err:
                                        logger.warning("watchdog add_log skipped (locked?): %s", log_err)
                                except Exception as restart_err:
                                    fail_count += 1
                                    setattr(_health_watchdog_loop, f"_fail_count_{bot_id}", fail_count)
                                    _WATCHDOG_STICKY_ERRORS[bot_id] = {
                                        "error": f"Watchdog restart failed: {restart_err}",
                                        "fail_count": fail_count, "last_attempt_ts": now,
                                    }
                                    try:
                                        add_log(bot_id, "ERROR", f"Watchdog restart failed ({fail_count}x): {restart_err}", "SYSTEM")
                                    except Exception as log_err:
                                        logger.warning("watchdog add_log skipped (locked?): %s", log_err)
                            else:
                                fail_count += 1
                                setattr(_health_watchdog_loop, f"_fail_count_{bot_id}", fail_count)
                                _WATCHDOG_STICKY_ERRORS[bot_id] = {
                                    "error": f"Cannot start: {reason}",
                                    "fail_count": fail_count, "last_attempt_ts": now,
                                }
                                if fail_count <= 3:
                                    try:
                                        add_log(bot_id, "WARN", f"Watchdog: cannot restart — {reason}", "SYSTEM")
                                    except Exception as log_err:
                                        logger.warning("watchdog add_log skipped (locked?): %s", log_err)
                                logger.warning(
                                    "watchdog: start blocked bot_id=%s market_type=%s reason=%s fails=%d",
                                    bot_id, b.get("market_type"), reason, fail_count,
                                )
                    continue
                last_tick = int(snap.get("last_tick_ts") or 0)
                bot_row = get_bot(bot_id) or b
                db_polled = int(bot_row.get("last_polled_at") or 0)
                last_ref = max(last_tick, db_polled)
                poll_s = max(1, int(bot_row.get("poll_seconds") or 10))
                stale_threshold = max(120, 2 * poll_s)
                if last_ref and (now - last_ref) > stale_threshold:
                    if bool(snap.get("running")):
                        bot = get_bot(bot_id)
                        is_stock = bot and (len(str(bot.get("symbol", ""))) < 6 and "/" not in str(bot.get("symbol", "")))
                        if is_stock and not _alpaca_market_open():
                            continue

                        # Bug 1: NEVER cancel an open deal here. Only cancel a
                        # *ghost* deal — one with no recorded entry — and only
                        # if it has been stuck like that for hours. Real deals
                        # with entry_avg are preserved across restart.
                        _wd_deal = latest_open_deal(bot_id)
                        if _wd_deal and _wd_deal.get("entry_avg") is None:
                            _deal_age = now - int(_wd_deal.get("opened_at", now))
                            if _deal_age > 7200:
                                try:
                                    cancel_ghost_deal(int(_wd_deal["id"]))
                                    add_log(bot_id, "WARN",
                                            f"Watchdog cancelled ghost deal #{_wd_deal['id']} ({_deal_age // 3600}h old, no entry).",
                                            "SYSTEM")
                                except Exception:
                                    pass

                        # Bug 8: Throttle stalled-bot restarts via the same
                        # per-bot interval used for cold-start recovery.
                        _last_restart = getattr(_health_watchdog_loop, f"_last_restart_{bot_id}", 0)
                        if (now - _last_restart) < WATCHDOG_MIN_RESTART_INTERVAL_SEC:
                            continue
                        # Only restart if TRULY stale (>15 min for stocks, >10 min for crypto)
                        _restart_threshold = max(stale_threshold, 900 if is_stock else stale_threshold)
                        if (now - last_ref) > _restart_threshold:
                            bm.stop(bot_id, silent=True)
                            time.sleep(1)
                            ok, reason = _can_start_bot_live(b)
                            if ok:
                                bm.start(bot_id, silent=True)
                                setattr(_health_watchdog_loop, f"_last_restart_{bot_id}", now)
                                try:
                                    add_log(bot_id, "WARN",
                                            f"Watchdog restarted stalled bot (last activity: {now - last_ref}s ago).",
                                            "SYSTEM")
                                except Exception as log_err:
                                    logger.warning("watchdog add_log skipped (locked?): %s", log_err)
                            else:
                                logger.warning(
                                    "watchdog: restart blocked bot_id=%s market_type=%s reason=%s",
                                    bot_id, b.get("market_type"), reason,
                                )
            # Ghost deal detector: close deals that are OPEN with no entry for >2 hours
            try:
                ghost_deals = find_stale_ghost_deals(max_age_sec=7200)
                for gd in ghost_deals:
                    gd_id = int(gd["id"])
                    gd_bot_id = int(gd["bot_id"])
                    gd_age_hr = (now - int(gd.get("opened_at", now))) / 3600.0
                    cancel_ghost_deal(gd_id)
                    add_log(gd_bot_id, "WARN",
                            f"Ghost deal #{gd_id} auto-cancelled: OPEN for {gd_age_hr:.1f}h with no entry (entry_avg=NULL).",
                            "SYSTEM")
                    logger.warning("watchdog: cancelled ghost deal #%d for bot %d (%.1fh old, no entry)",
                                   gd_id, gd_bot_id, gd_age_hr)
            except Exception as _gd_err:
                logger.debug("Ghost deal check failed: %s", _gd_err)

            # Reconciliation: compare local positions vs exchange (RECONCILIATION_ENABLED=1 to activate)
            try:
                from reconciliation import run_reconciliation
                recon_snapshots = []
                if bm:
                    for b in bots:
                        _bid = int(b.get("id"))
                        _snap = bm.snapshot(_bid)
                        if _snap and _snap.get("running") and float(_snap.get("base_pos") or 0.0) > 0:
                            recon_snapshots.append({
                                "bot_id": _bid,
                                "symbol": b.get("symbol", ""),
                                "base_pos": float(_snap.get("base_pos") or 0.0),
                                "market_type": b.get("market_type", "crypto"),
                            })

                def _recon_client_fn(b_snap):
                    mt = b_snap.get("market_type", "crypto")
                    if mt in ("stocks", "stock"):
                        _client = alpaca_live or alpaca_paper
                        if _client:
                            from alpaca_adapter import AlpacaAdapter
                            return (AlpacaAdapter(_client), False)
                        return (None, False)
                    return (kc, True)

                recon_warnings = run_reconciliation(recon_snapshots, _recon_client_fn)
                for w in recon_warnings:
                    logger.warning("Reconciliation: %s", w)
            except ImportError:
                pass
            except Exception:
                logger.debug("Reconciliation check failed (non-critical)", exc_info=True)
        except Exception:
            logger.exception("_health_watchdog_loop: iteration failed")
        time.sleep(HEALTH_WATCHDOG_SEC)


def _comprehensive_health_check() -> Tuple[bool, List[str]]:
    """Check all critical systems. Returns (ok, issues)."""
    issues = []
    try:
        client = alpaca_live or alpaca_paper
        if client and hasattr(client, "check_websocket_health"):
            ok, ws_issues = client.check_websocket_health()
            if not ok:
                issues.extend(ws_issues)
        elif client and hasattr(client, "get_stats"):
            stats = client.get_stats()
            ws = stats.get("websocket", {})
            if not ws.get("running", True):
                issues.append("WebSocket not running")
        autopilot_on = False
        try:
            import autopilot
            autopilot_on = autopilot.is_autopilot_enabled()
            if not autopilot_on:
                issues.append("Autopilot is disabled")
        except Exception:
            pass
        bots = list_bots() or []
        active = [b for b in bots if int(b.get("enabled", 0)) == 1]
        if len(active) == 0 and autopilot_on:
            issues.append("No active bots (autopilot enabled but no bots)")
        if client:
            try:
                client.get_account()
            except Exception as e:
                issues.append(f"API connection error: {e}")
        if bm:
            snaps = [bm.snapshot(int(b["id"])) for b in active]
            paused = [b for b, s in zip(active, snaps) if str(s.get("status", "") or "").upper() == "PAUSE"]
            if paused:
                issues.append(f"{len(paused)} bots paused")
        if issues:
            logger.error("Health check FAILED: %s", issues)
        else:
            logger.info("Health check PASSED - All systems operational")
        return len(issues) == 0, issues
    except Exception as e:
        logger.exception("comprehensive_health_check error: %s", e)
        return False, [str(e)]


def _health_comprehensive_loop() -> None:
    """Run comprehensive health check and WebSocket stats every 5 minutes."""
    while True:
        try:
            time.sleep(300)
            _comprehensive_health_check()  # (ok, issues) - logs internally
            client = alpaca_live or alpaca_paper
            if client and hasattr(client, "print_stats"):
                try:
                    client.print_stats()
                except Exception:
                    pass
        except Exception:
            logger.exception("_health_comprehensive_loop error")


# =========================================================
# Startup (fully fault-tolerant: never raise; always bind)
# =========================================================
@app.on_event("startup")
def startup():
    global kc, alpaca_paper, alpaca_live, bm, KRAKEN_READY, KRAKEN_ERROR, ALPACA_PAPER_READY, ALPACA_LIVE_READY, ALPACA_ERROR
    try:
        _startup_impl()
    except Exception as e:
        logger.exception("startup: unhandled error")
        logger.error("worker_api startup: UNHANDLED ERROR (%s). Running in minimal mode.", e)


@app.on_event("shutdown")
def shutdown():
    """Graceful shutdown: stop all bots, close websockets, clean up resources."""
    global alpaca_paper, alpaca_live, bm, _shutdown_event

    logger.info("Shutdown event triggered - waiting up to 30 seconds for operations to complete...")
    _shutdown_event.set()

    try:
        from db import stop_wal_checkpoint_thread
        stop_wal_checkpoint_thread(timeout_sec=5.0)
    except Exception:
        logger.exception("shutdown: stop_wal_checkpoint_thread failed")

    # Stop all running bots
    if bm:
        try:
            logger.info("Stopping all bots...")
            for b in list_bots():
                try:
                    bm.stop(int(b.get("id")), silent=True)
                except Exception as e:
                    logger.debug("Stop bot %s failed: %s", b.get("id"), e)
        except Exception as e:
            logger.warning("Error stopping bots: %s", e)

    # Wait for running operations to complete (up to 30 seconds)
    start_time = time.time()
    timeout_sec = 30
    while time.time() - start_time < timeout_sec:
        if bm and bm._running_tasks and len([t for t in bm._running_tasks if t.get("active")]) > 0:
            active_tasks = len([t for t in bm._running_tasks if t.get("active")])
            logger.info(f"Waiting for {active_tasks} task(s) to complete... ({int(time.time() - start_time)}s)")
            time.sleep(1)
        else:
            break

    # Close Alpaca clients
    for client in [alpaca_paper, alpaca_live]:
        if client and hasattr(client, "shutdown"):
            try:
                client.shutdown()
                logger.info("UnifiedAlpacaClient shutdown complete")
            except Exception as e:
                logger.warning("Client shutdown: %s", e)

    logger.info("Shutdown complete")


def _start_websocket_with_timeout(ws_manager, timeout_sec: int = 10):
    """Start WebSocket in background - never block startup."""
    try:
        if ws_manager and hasattr(ws_manager, "start"):
            ws_manager.start()
        _STARTUP_STATUS["websocket_status"] = "running"
    except Exception as e:
        logger.warning("WebSocket start failed (non-blocking): %s", e)
        _STARTUP_STATUS["websocket_status"] = "not_ready"


def _init_alpaca_and_bm_sync():
    """Initialize Alpaca (if ENABLE_ALPACA and keys present) and BotManager synchronously. No background thread."""
    global alpaca_paper, alpaca_live, bm, ALPACA_PAPER_READY, ALPACA_LIVE_READY, ALPACA_ERROR, _STARTUP_STATUS
    _has_alpaca_keys = bool(os.getenv("ALPACA_API_KEY_PAPER") and os.getenv("ALPACA_API_SECRET_PAPER"))
    _enable_alpaca = os.getenv("ENABLE_ALPACA", "1" if _has_alpaca_keys else "0").strip().lower() in ("1", "true", "yes", "y", "on")
    if not _enable_alpaca or not _has_alpaca_keys:
        logger.warning("WARNING: Stock signals are enabled but Alpaca is not configured — stock recommendations will appear but bots cannot be created for them. Add Alpaca API keys in Settings to enable stock trading.")
        logger.info("worker_api startup: Alpaca skipped (ENABLE_ALPACA=%s, keys=%s); crypto-only mode.", _enable_alpaca, "set" if _has_alpaca_keys else "missing")
        # Still create BotManager with Kraken only
        with _globals_lock:
            bm = None
            if kc:
                try:
                    bm = BotManager(kc, None, None)
                    if bm and hasattr(bm, "subscribe_all_symbols"):
                        threading.Thread(target=bm.subscribe_all_symbols, daemon=True).start()
                    logger.info("worker_api startup: BotManager OK (Crypto only)")
                except Exception as e:
                    _STARTUP_STATUS["last_startup_error"] = f"BotManager init failed: {e}"
                    logger.exception("worker_api startup: BotManager init failed")
        return

    pk = "SET" if os.getenv("ALPACA_API_KEY_PAPER") else "blank"
    sk = "SET" if os.getenv("ALPACA_API_SECRET_PAPER") else "blank"
    lk = "SET" if os.getenv("ALPACA_API_KEY_LIVE") else "blank"
    ls = "SET" if os.getenv("ALPACA_API_SECRET_LIVE") else "blank"
    logger.info("worker_api startup: ALPACA_API_KEY_PAPER=%s ALPACA_API_SECRET_PAPER=%s ALPACA_API_KEY_LIVE=%s ALPACA_API_SECRET_LIVE=%s",
                pk, sk, lk, ls)
    try:
        with _globals_lock:
            try:
                if USE_UNIFIED_ALPACA and _UNIFIED_AVAILABLE and UnifiedAlpacaClient:
                    alpaca_paper = UnifiedAlpacaClient(mode="paper", auto_start_websocket=False)
                    if alpaca_paper and hasattr(alpaca_paper, "websocket"):
                        t = threading.Thread(target=_start_websocket_with_timeout, args=(alpaca_paper.websocket, 10), daemon=True)
                        t.start()
                    logger.info("worker_api startup: Alpaca PAPER initialized (Unified, websocket deferred).")
                else:
                    alpaca_paper = AlpacaClient(mode="paper")
                    logger.info("worker_api startup: Alpaca PAPER initialized (Legacy).")
                ALPACA_PAPER_READY = True
            except Exception as e:
                try:
                    alpaca_paper = AlpacaClient(mode="paper")
                    ALPACA_PAPER_READY = True
                    logger.info("worker_api startup: Alpaca PAPER fallback to Legacy: %s", e)
                except Exception as e2:
                    alpaca_paper = None
                    ALPACA_PAPER_READY = False
                    ALPACA_ERROR = f"Paper: {e2}"
                    logger.warning("worker_api startup: Alpaca PAPER failed: %s", e2)

        # Skip live Alpaca init when live trading disabled (saves memory + connections)
        if LIVE_TRADING_ENABLED:
            with _globals_lock:
                try:
                    if USE_UNIFIED_ALPACA and _UNIFIED_AVAILABLE and UnifiedAlpacaClient:
                        alpaca_live = UnifiedAlpacaClient(mode="live", auto_start_websocket=False)
                        if alpaca_live and hasattr(alpaca_live, "websocket"):
                            t = threading.Thread(target=_start_websocket_with_timeout, args=(alpaca_live.websocket, 10), daemon=True)
                            t.start()
                        logger.info("worker_api startup: Alpaca LIVE initialized (Unified, websocket deferred).")
                    else:
                        alpaca_live = AlpacaClient(mode="live")
                        logger.info("worker_api startup: Alpaca LIVE initialized (Legacy).")
                    ALPACA_LIVE_READY = True
                except Exception as e:
                    try:
                        alpaca_live = AlpacaClient(mode="live")
                        ALPACA_LIVE_READY = True
                        logger.info("worker_api startup: Alpaca LIVE fallback to Legacy")
                    except Exception as e2:
                        alpaca_live = None
                        ALPACA_LIVE_READY = False
                        ALPACA_ERROR = (ALPACA_ERROR or "") + f" | Live: {e2}"
                        logger.warning("worker_api startup: Alpaca LIVE failed: %s", e2)
        else:
            alpaca_live = None
            ALPACA_LIVE_READY = False
            logger.info("worker_api startup: Alpaca LIVE skipped (set LIVE_TRADING_ENABLED=1 in .env for live)")

        bp = 0.0
        if alpaca_paper:
            try:
                bp = float((_retry_with_backoff(lambda: alpaca_paper.get_account() or {}, max_retries=2, base_delay=0.5)).get("buying_power", 0))
            except Exception as e:
                logger.debug(f"Failed to fetch Alpaca buying power on startup: {e}")
                pass
        _STARTUP_STATUS["alpaca_ready"] = ALPACA_PAPER_READY or ALPACA_LIVE_READY
        _STARTUP_STATUS["alpaca_buying_power"] = bp

        with _globals_lock:
            bm = None
            if kc or alpaca_paper or alpaca_live:
                try:
                    bm = BotManager(kc, alpaca_paper, alpaca_live)
                    if bm and hasattr(bm, "subscribe_all_symbols"):
                        threading.Thread(target=bm.subscribe_all_symbols, daemon=True).start()
                    logger.info("worker_api startup: BotManager OK (Crypto: %s, Alpaca paper: %s, Alpaca live: %s)",
                        KRAKEN_READY, ALPACA_PAPER_READY, ALPACA_LIVE_READY)
                except Exception as e:
                    logger.exception("worker_api startup: BotManager init failed")
                    logger.warning("worker_api startup: BotManager failed (%s). Degraded mode.", e)

        # Candle pre-warm test
        if alpaca_paper and bm:
            try:
                c = _retry_with_backoff(lambda: alpaca_paper.get_ohlcv("AAPL", "1h", 50), max_retries=2, base_delay=0.5)
                _STARTUP_STATUS["candle_test"] = f"fetched {len(c)} candles for AAPL" if c else "failed"
            except Exception as ex:
                _STARTUP_STATUS["candle_test"] = f"error: {ex}"
    except Exception as e:
        _STARTUP_STATUS["last_startup_error"] = f"Alpaca/BotManager init failed: {e}"
        logger.exception("_init_alpaca_and_bm_sync failed: %s", e)


def _retry_alpaca_init_if_keys_present() -> bool:
    """
    Attempt to re-initialize Alpaca if keys are present but ALPACA_PAPER_READY is False.
    Returns True if Alpaca is now ready, False otherwise.
    """
    global alpaca_paper, alpaca_live, ALPACA_PAPER_READY, ALPACA_LIVE_READY, ALPACA_ERROR

    _has_alpaca_keys = bool(os.getenv("ALPACA_API_KEY_PAPER") and os.getenv("ALPACA_API_SECRET_PAPER"))

    # Only retry if keys exist but neither is ready
    if not _has_alpaca_keys or (ALPACA_PAPER_READY or ALPACA_LIVE_READY):
        return ALPACA_PAPER_READY or ALPACA_LIVE_READY

    logger.info("Attempting to re-initialize Alpaca (keys present, but not ready)...")

    try:
        with _globals_lock:
            try:
                if USE_UNIFIED_ALPACA and _UNIFIED_AVAILABLE and UnifiedAlpacaClient:
                    alpaca_paper = UnifiedAlpacaClient(mode="paper", auto_start_websocket=False)
                    logger.info("Alpaca PAPER re-initialized (Unified)")
                else:
                    alpaca_paper = AlpacaClient(mode="paper")
                    logger.info("Alpaca PAPER re-initialized (Legacy)")
                ALPACA_PAPER_READY = True
                ALPACA_ERROR = ""
            except Exception as e:
                try:
                    alpaca_paper = AlpacaClient(mode="paper")
                    ALPACA_PAPER_READY = True
                    ALPACA_ERROR = ""
                    logger.info("Alpaca PAPER fallback to Legacy succeeded")
                except Exception as e2:
                    alpaca_paper = None
                    ALPACA_PAPER_READY = False
                    ALPACA_ERROR = f"Paper re-init failed: {e2}"
                    logger.warning("Alpaca PAPER re-init failed: %s", e2)
    except Exception as e:
        logger.warning("Unexpected error during Alpaca re-init: %s", e)

    return ALPACA_PAPER_READY or ALPACA_LIVE_READY


def _validate_config() -> None:
    """Validate configuration at startup. Check API keys and connectivity."""
    errs = []
    warnings = []

    # Numeric range validation
    ttl_val = os.getenv("MARKETS_TTL_SEC", "300").strip()
    if ttl_val:
        try:
            ttl = int(ttl_val)
            if not (60 <= ttl <= 3600):
                errs.append(f"MARKETS_TTL_SEC must be 60-3600, got {ttl}")
        except (ValueError, TypeError):
            errs.append("MARKETS_TTL_SEC must be an integer")

    port_val = os.getenv("PORT_EVERY_SEC", "60").strip()
    if port_val:
        try:
            port_every = int(port_val)
            if not (1 <= port_every <= 3600):
                errs.append(f"PORT_EVERY_SEC must be 1-3600, got {port_every}")
        except (ValueError, TypeError):
            errs.append("PORT_EVERY_SEC must be an integer")

    # Check API key presence
    kraken_key = os.getenv("KRAKEN_API_KEY", "").strip()
    kraken_secret = os.getenv("KRAKEN_API_SECRET", "").strip()

    if not kraken_key:
        warnings.append("KRAKEN_API_KEY not set; crypto trading disabled")
    elif len(kraken_key) < 20:
        warnings.append(f"KRAKEN_API_KEY seems too short ({len(kraken_key)} chars)")

    if not kraken_secret:
        warnings.append("KRAKEN_API_SECRET not set; crypto trading disabled")
    elif len(kraken_secret) < 20:
        warnings.append(f"KRAKEN_API_SECRET seems too short ({len(kraken_secret)} chars)")

    # Check Alpaca keys if enabled
    enable_alpaca = os.getenv("ENABLE_ALPACA", "1").strip().lower() in ("1", "true", "yes", "y", "on")
    if enable_alpaca:
        alpaca_key_paper = os.getenv("ALPACA_API_KEY_PAPER", "").strip()
        alpaca_secret_paper = os.getenv("ALPACA_API_SECRET_PAPER", "").strip()

        if not alpaca_key_paper:
            warnings.append("ALPACA_API_KEY_PAPER not set; paper trading disabled")
        if not alpaca_secret_paper:
            warnings.append("ALPACA_API_SECRET_PAPER not set; paper trading disabled")

        alpaca_key_live = os.getenv("ALPACA_API_KEY_LIVE", "").strip()
        alpaca_secret_live = os.getenv("ALPACA_API_SECRET_LIVE", "").strip()
        live_trading = os.getenv("LIVE_TRADING_ENABLED", "0").strip().lower() in ("1", "true", "yes", "y", "on")

        if live_trading:
            if not alpaca_key_live or not alpaca_secret_live:
                warnings.append("LIVE_TRADING_ENABLED but Alpaca live keys not set; live trading disabled")

    # Log warnings
    for w in warnings:
        logger.warning(f"[STARTUP VALIDATION] {w}")

    # Log key validation results
    _STARTUP_STATUS["validation_warnings"] = warnings
    _STARTUP_STATUS["kraken_key_present"] = bool(kraken_key)
    _STARTUP_STATUS["kraken_secret_present"] = bool(kraken_secret)

    if errs:
        msg = "Config validation failed: " + "; ".join(errs)
        logger.error(msg)
        _STARTUP_STATUS["validation_errors"] = errs
        raise ValueError(msg)

    logger.info(f"[STARTUP VALIDATION] Config valid. {len(warnings)} warning(s).")


def _startup_impl():
    global kc, alpaca_paper, alpaca_live, bm, KRAKEN_READY, KRAKEN_ERROR, ALPACA_PAPER_READY, ALPACA_LIVE_READY, ALPACA_ERROR
    global LIVE_ENDPOINTS_DISABLED, LIVE_ENDPOINTS_DISABLED_REASON

    _validate_config()

    def _safe_init_db():
        try:
            init_db()
            return True
        except Exception as e:
            logger.exception("startup: init_db failed")
            logger.warning("worker_api startup: init_db failed (%s). DB features degraded.", e)
            return False

    _safe_init_db()
    _STARTUP_STATUS["db_ready"] = True
    # Clean up stale/invalid recommendation scores (100-score artifacts, old data)
    try:
        from db import cleanup_invalid_scores
        cleaned = cleanup_invalid_scores()
        if cleaned > 0:
            logger.info("startup: cleaned %d invalid/stale recommendation records", cleaned)
    except Exception as e:
        logger.debug("startup: cleanup_invalid_scores: %s", e)
    try:
        _STARTUP_STATUS["db_bots"] = len(list_bots())
    except Exception:
        pass
    from db import DB_NAME
    _STARTUP_STATUS["db_path"] = DB_NAME
    _STARTUP_STATUS["db_path_abs"] = os.path.abspath(DB_NAME)
    _STARTUP_STATUS["timestamp"] = int(time.time())
    try:
        from db import start_wal_checkpoint_thread
        start_wal_checkpoint_thread()
        logger.info(
            "startup: WAL checkpoint daemon started (interval BOT_WAL_CHECKPOINT_INTERVAL_SEC)"
        )
    except Exception:
        logger.exception("startup: WAL checkpoint daemon failed to start")

    try:
        env_res = get_last_load_result()
        _STARTUP_STATUS["env_loaded_paths"] = env_res.get("loaded_paths") or []
    except Exception:
        _STARTUP_STATUS["env_loaded_paths"] = []
    _db_abs = os.path.abspath(DB_NAME)
    logger.info("Database: %s (abs=%s) (%s bots)", DB_NAME, _db_abs, _STARTUP_STATUS.get("db_bots", 0))

    # Purge blocklisted symbols from recommendations (Explore tab)
    try:
        n = delete_recommendations_for_blocklist(list(CRYPTO_BLOCKLIST))
        if n > 0:
            logger.info("Purged %d blocklisted recommendation(s) (STABLE, etc.)", n)
    except Exception as e:
        logger.warning("Purge blocklist failed: %s", e)

    try:
        t = WORKER_API_TOKEN
        if t and (len(t) < 16 or t.lower() in ("dev", "test", "secret")):
            logger.warning("WORKER_API_TOKEN looks weak or default. Use a long random secret for production.")
        elif not t:
            logger.warning("WORKER_API_TOKEN not set. API auth disabled; fine for localhost only.")
    except Exception as e:
        logger.warning("startup: token check failed: %s", e)

    # --- KRAKEN DIAGNOSTIC: Check .env and env before init ---
    _env_paths = [os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env"), os.path.join(os.getcwd(), ".env")]
    _env_exists = any(os.path.exists(p) for p in _env_paths)
    _kkey = os.getenv("KRAKEN_API_KEY", "")
    _ksec = os.getenv("KRAKEN_API_SECRET", "")
    _kkey_in_file = False
    _ksec_in_file = False
    _kkey_raw = ""
    _ksec_raw = ""
    for _p in _env_paths:
        if os.path.exists(_p):
            try:
                with open(_p, "r", encoding="utf-8") as _f:
                    for _line in _f:
                        if "KRAKEN_API_KEY" in _line and "=" in _line:
                            _kkey_in_file = True
                            _parts = _line.split("=", 1)
                            if len(_parts) == 2:
                                _kkey_raw = _parts[1].strip().strip('"').strip("'")
                        if "KRAKEN_API_SECRET" in _line and "=" in _line:
                            _ksec_in_file = True
                            _parts = _line.split("=", 1)
                            if len(_parts) == 2:
                                _ksec_raw = _parts[1].strip().strip('"').strip("'")
            except Exception:
                pass
            break
    logger.info(
        "[KRAKEN_DEBUG] .env exists=%s | KRAKEN_API_KEY in file=%s in env=%s len=%d first4=%s | KRAKEN_API_SECRET in file=%s in env=%s len=%d first4=%s",
        _env_exists, _kkey_in_file, bool(_kkey), len(_kkey), (_kkey[:4] + "..." if len(_kkey) >= 4 else ("(empty)" if not _kkey else _kkey[:4])),
        _ksec_in_file, bool(_ksec), len(_ksec), (_ksec[:4] + "..." if len(_ksec) >= 4 else ("(empty)" if not _ksec else _ksec[:4])),
    )
    if _kkey_raw and _kkey_raw != _kkey:
        logger.warning("[KRAKEN_DEBUG] KRAKEN_API_KEY in .env differs from os.environ (env may have been set before load_env, or key was skipped because k in os.environ)")
    if _ksec_raw and _ksec_raw != _ksec:
        logger.warning("[KRAKEN_DEBUG] KRAKEN_API_SECRET in .env differs from os.environ")
    # --- END KRAKEN DIAGNOSTIC ---

    # === KRAKEN DIAGNOSTIC (startup) ===
    _base = os.path.dirname(os.path.abspath(__file__))
    _env_paths = [
        os.path.join(_base, ".env"),
        os.path.join(os.getcwd(), ".env"),
    ]
    if os.getenv("ENV_FILE"):
        _env_paths.insert(0, os.getenv("ENV_FILE"))
    _env_exists = any(os.path.exists(p) for p in _env_paths)
    _env_loaded = _STARTUP_STATUS.get("env_loaded_paths") or []
    _kk = os.getenv("KRAKEN_API_KEY", "")
    _ks = os.getenv("KRAKEN_API_SECRET", "")
    _kk_preview = (_kk[:4] + "…") if _kk and len(_kk) >= 4 else ("EMPTY" if not _kk else "len=" + str(len(_kk)))
    _ks_preview = (_ks[:4] + "…") if _ks and len(_ks) >= 4 else ("EMPTY" if not _ks else "len=" + str(len(_ks)))
    # Check .env raw lines for KRAKEN (format issues: spaces, quotes)
    _kraken_lines_raw = []
    for _p in _env_paths:
        if os.path.exists(_p):
            try:
                with open(_p, "r", encoding="utf-8") as _f:
                    for _ln in _f:
                        if "KRAKEN" in _ln.upper() and "=" in _ln and not _ln.strip().startswith("#"):
                            _kraken_lines_raw.append(repr(_ln.rstrip()))
            except Exception:
                pass
            break
    logger.info(
        "[KRAKEN_DIAG] .env exists=%s | loaded_paths=%s | KRAKEN_API_KEY in env=%s (preview=%s) | KRAKEN_API_SECRET in env=%s (preview=%s) | .env KRAKEN lines: %s",
        _env_exists, _env_loaded, bool(_kk), _kk_preview, bool(_ks), _ks_preview, _kraken_lines_raw[:4]
    )

    # === KRAKEN DIAGNOSTIC (startup) ===
    _base = os.path.dirname(os.path.abspath(__file__))
    _env_paths = [os.path.join(_base, ".env"), os.path.join(os.getcwd(), ".env")]
    _env_exists = any(os.path.exists(p) for p in _env_paths)
    _env_loaded = os.path.abspath(_env_paths[0]) if os.path.exists(_env_paths[0]) else (os.path.abspath(_env_paths[1]) if os.path.exists(_env_paths[1]) else None)
    _kk = os.getenv("KRAKEN_API_KEY", "")
    _ks = os.getenv("KRAKEN_API_SECRET", "")
    _kk_first4 = (_kk[:4] + "...") if _kk and len(_kk) >= 4 else ("EMPTY" if not _kk else ("len=" + str(len(_kk))))
    _ks_first4 = (_ks[:4] + "...") if _ks and len(_ks) >= 4 else ("EMPTY" if not _ks else ("len=" + str(len(_ks))))
    print(f"[KRAKEN_DIAG] .env exists: {_env_exists} (checked: {_env_paths})")
    print(f"[KRAKEN_DIAG] .env loaded from: {_env_loaded or 'NONE'}")
    print(f"[KRAKEN_DIAG] KRAKEN_API_KEY in os.environ: {bool(_kk)} first4={_kk_first4}")
    print(f"[KRAKEN_DIAG] KRAKEN_API_SECRET in os.environ: {bool(_ks)} first4={_ks_first4}")
    if _env_loaded:
        try:
            with open(_env_loaded, "r", encoding="utf-8") as _f:
                for _line in _f:
                    _l = _line.rstrip()
                    if "KRAKEN" in _l and not _l.strip().startswith("#"):
                        _has_spaces = " = " in _l or _l.startswith(" ") or "= " in _l.split("=", 1)[0]
                        _has_quotes = '"' in _l or "'" in _l
                        print(f"[KRAKEN_DIAG] .env line format: key_part={repr(_l.split('=')[0])} has_spaces_around_eq={_has_spaces} has_quotes={_has_quotes}")
        except Exception as _e:
            print(f"[KRAKEN_DIAG] .env read error: {_e}")
    # === END KRAKEN DIAGNOSTIC ===

    # === KRAKEN DIAGNOSTIC (startup) ===
    _base = os.path.dirname(os.path.abspath(__file__))
    _env_paths = [os.path.join(_base, ".env"), os.path.join(os.getcwd(), ".env")]
    _env_exists = any(os.path.exists(p) for p in _env_paths)
    _env_loaded = _STARTUP_STATUS.get("env_loaded_paths") or []
    _kk = os.getenv("KRAKEN_API_KEY", "")
    _ks = os.getenv("KRAKEN_API_SECRET", "")
    _kk_preview = (_kk[:4] + "...") if _kk and len(_kk) >= 4 else ("EMPTY" if not _kk else "***")
    _ks_preview = (_ks[:4] + "...") if _ks and len(_ks) >= 4 else ("EMPTY" if not _ks else "***")
    print(
        f"[KRAKEN_DEBUG] .env exists: {_env_exists} | paths tried: {_env_paths} | loaded: {_env_loaded} | "
        f"KRAKEN_API_KEY in env: {bool(_kk)} (first4={_kk_preview}) | "
        f"KRAKEN_API_SECRET in env: {bool(_ks)} (first4={_ks_preview})"
    )
    # Check .env raw lines for format issues (spaces, quotes)
    for _p in _env_paths:
        if os.path.exists(_p):
            try:
                with open(_p, "r", encoding="utf-8") as _f:
                    for _line in _f:
                        _line = _line.rstrip("\n\r")
                        if "KRAKEN" in _line and not _line.strip().startswith("#"):
                            _has_spaces = " = " in _line or _line.startswith(" ") or " =" in _line
                            _has_quotes = '"' in _line or "'" in _line
                            print(f"[KRAKEN_DEBUG] .env line: {repr(_line[:80])} (spaces_around_eq={_has_spaces}, has_quotes={_has_quotes})")
            except Exception as _e:
                print(f"[KRAKEN_DEBUG] Could not read .env: {_e}")
            break

    with _globals_lock:
        try:
            kc = KrakenClient()
            KRAKEN_READY = True
            KRAKEN_ERROR = ""
            logger.info("worker_api startup: Kraken client initialized.")
        except Exception as e:
            kc = None
            KRAKEN_READY = False
            KRAKEN_ERROR = str(e)
            logger.warning("worker_api startup: Kraken NOT initialized: %s", e)

    try:
        from db import repair_closed_deals_missing_entry

        repair_closed_deals_missing_entry(kc)
    except Exception as e:
        logger.warning("startup: repair_closed_deals_missing_entry failed: %s", e)

    _STARTUP_STATUS["kraken_ready"] = KRAKEN_READY
    # ENABLE_ALPACA: 0 = crypto-only (skip Alpaca); 1 = require Alpaca for stocks. Default 0 if keys missing.
    _has_alpaca_keys = bool(os.getenv("ALPACA_API_KEY_PAPER") and os.getenv("ALPACA_API_SECRET_PAPER"))
    _enable_alpaca_val = os.getenv("ENABLE_ALPACA", "1" if _has_alpaca_keys else "0").strip().lower() in ("1", "true", "yes", "y", "on")
    if _enable_alpaca_val and not _has_alpaca_keys:
        _STARTUP_STATUS["last_startup_error"] = "ENABLE_ALPACA=1 but ALPACA_API_KEY_PAPER or ALPACA_API_SECRET_PAPER missing; set keys in .env or ENABLE_ALPACA=0 for crypto-only"
        logger.warning("worker_api startup: %s", _STARTUP_STATUS["last_startup_error"])

    _STARTUP_STATUS["flask_ready"] = True
    # Deterministic startup: Alpaca + BotManager init synchronously (no background thread)
    _init_alpaca_and_bm_sync()
    _STARTUP_STATUS["alpaca_ready"] = ALPACA_PAPER_READY or ALPACA_LIVE_READY
    _STARTUP_STATUS["bm_ready"] = bm is not None

    # Autopilot status (from DB)
    try:
        import autopilot
        autopilot.ensure_autopilot_config_defaults()
        _STARTUP_STATUS["autopilot_enabled"] = autopilot.is_autopilot_enabled()
        ap_bots = [b for b in list_bots() if str(b.get("bot_type") or "").lower() == "autopilot"]
        _STARTUP_STATUS["autopilot_bots"] = len(ap_bots)
    except Exception:
        pass

    # API auth disabled by user request
    LIVE_ENDPOINTS_DISABLED = False
    LIVE_ENDPOINTS_DISABLED_REASON = ""

    for name, target in [
        ("portfolio", _portfolio_loop),
        ("discord_status", _discord_status_loop),
        ("autostart", _autostart_loop),
        ("health_watchdog", _health_watchdog_loop),
        ("health_comprehensive", _health_comprehensive_loop),
        ("explore_startup_scan", _explore_startup_sequential_scan),
        ("recommendations", _recommendations_loop),
        ("ml_retrain", _ml_retrain_loop),
        ("ml_outcomes", _ml_outcomes_loop),
        ("screener_outcomes", _screener_outcomes_loop),
        ("autopilot", _autopilot_loop),
        ("fear_greed", _fear_greed_loop),
        ("perf_outcomes", _performance_outcomes_loop),
        ("thread_watchdog", _thread_watchdog_loop),
        ("bots_summary_prewarm", _prewarm_bots_summary),
        ("db_cleanup", _scheduled_db_cleanup),
    ]:
        try:
            _start_background_thread(name, target)
        except Exception as e:
            logger.exception("startup: failed to start thread %s", name)
            logger.warning("worker_api startup: thread %s failed (%s). Continuing.", name, e)
    try:
        _comprehensive_health_check()  # returns (ok, issues)
    except Exception:
        pass

    # Startup validation
    _startup_checks = []
    try:
        init_db()
        _cnt = len(list_bots())
        _startup_checks.append(f"DB ✓ ({_cnt} bots)")
    except Exception as e:
        _startup_checks.append(f"DB ✗ ({e})")
        logger.error("STARTUP VALIDATION FAILED: Database unreachable: %s", e)

    if KRAKEN_READY:
        _startup_checks.append("Kraken ✓")
    else:
        _startup_checks.append(f"Kraken ✗ ({KRAKEN_ERROR or 'not configured'})")

    if ALPACA_PAPER_READY:
        _startup_checks.append("Alpaca Paper ✓")
    elif ALPACA_LIVE_READY:
        _startup_checks.append("Alpaca Live ✓")
    elif _ALPACA_KEYS_PRESENT:
        _startup_checks.append(f"Alpaca ✗ ({ALPACA_ERROR or 'connection failed'})")
    else:
        _startup_checks.append("Alpaca ✗ (not configured)")

    logger.info("Startup validation: %s", " | ".join(_startup_checks))
    if ALLOW_LIVE_TRADING:
        logger.info(
            "\U0001f7e2 LIVE TRADING ENABLED \u2014 ALLOW_LIVE_TRADING=1 \u2014 "
            "Real money orders will be placed on Kraken/Alpaca"
        )
    else:
        logger.warning(
            "\U0001f534 LIVE TRADING DISABLED \u2014 ALLOW_LIVE_TRADING=0 \u2014 "
            "All real orders will be blocked"
        )
    logger.info("worker_api startup: complete.")
    logger.info(
        "Startup diagnostics: Flask=%s DB=%s (bots=%s) Alpaca=%s WebSocket=%s Autopilot=%s (bots=%s) CandleTest=%s",
        _STARTUP_STATUS.get("flask_ready"),
        _STARTUP_STATUS.get("db_ready"),
        _STARTUP_STATUS.get("db_bots"),
        ALPACA_PAPER_READY or ALPACA_LIVE_READY,
        _STARTUP_STATUS.get("websocket_status"),
        _STARTUP_STATUS.get("autopilot_enabled"),
        _STARTUP_STATUS.get("autopilot_bots"),
        _STARTUP_STATUS.get("candle_test"),
    )
    logger.info(
        "Trading readiness: Kraken=%s (err=%s) AlpacaPaper=%s AlpacaLive=%s (err=%s) AUTO_START_ENABLED=%s HEALTH_WATCHDOG_SEC=%s",
        KRAKEN_READY,
        KRAKEN_ERROR or "none",
        ALPACA_PAPER_READY,
        ALPACA_LIVE_READY,
        ALPACA_ERROR or "none",
        AUTO_START_ENABLED,
        HEALTH_WATCHDOG_SEC,
    )


@app.put("/api/bots/{bot_id}")
async def api_update_bot(bot_id: int, request: Request):
    """Update bot settings. Merges payload with existing bot for partial updates."""
    payload = await request.json()
    if not isinstance(payload, dict):
        return _json({"ok": False, "error": "Invalid payload"}, 400)

    b = get_bot(int(bot_id))
    if not b:
        return _json({"ok": False, "error": "Bot not found"}, 404)

    # Live-promotion guard runs BEFORE the bm-not-initialized check so we never
    # have a window where dry_run could be silently flipped to 0. Any 1->0 flip
    # on dry_run must satisfy the same safety contract as POST
    # /api/go_live/confirm. Without this, a stray PUT (or the older /bots Go
    # Live modal) could promote a bot to live with one click and zero checks.
    # The typed-LIVE confirmation is required so even a successful PUT cannot
    # silently flip dry_run.
    if "dry_run" in payload:
        try:
            requested_dry = int(payload.get("dry_run"))
        except (TypeError, ValueError):
            requested_dry = 1
        current_dry = int(b.get("dry_run", 1) or 1)
        if current_dry == 1 and requested_dry == 0:
            confirm_str = str(payload.get("confirm", "")).strip().upper()
            if confirm_str != "LIVE":
                return _json({
                    "ok": False,
                    "error": "Live promotion requires confirm=\"LIVE\" in body. Use POST /api/go_live/confirm.",
                }, 400)
            # Strict gate: same shared function as /api/safety/checklist and
            # /api/go_live/confirm. The gate is on the dry_run 1->0 transition
            # itself, so it applies regardless of caller identity (browser
            # session, worker token, automation script).
            from services import safety_checklist as _sc
            check = _sc.compute_live_readiness()
            if not check.get("live_ready"):
                return _json({
                    "ok": False,
                    "error": "Live promotion blocked: safety checklist not live_ready",
                    "blocking_reasons": check.get("blocking_reasons", []),
                    "flags": check.get("flags", {}),
                }, 409)

    # All non-promotion edits still require BM to be up.
    if not bm:
        reason = _bm_not_ready_reason() or "BotManager not initialized"
        return _json({"ok": False, "error": "BotManager not initialized", "reason": reason}, 503)

    # Merge: existing bot as base, overlay payload (partial updates supported)
    raw_sym = str(payload.get("symbol") or b.get("symbol") or "").strip()
    detected_type = classify_symbol(raw_sym) if raw_sym else "crypto"
    market_type = "stocks" if detected_type == "stock" else "crypto"
    if market_type == "crypto" and raw_sym:
        resolved, err = _validate_crypto_symbol(raw_sym)
        if err:
            return _json({"ok": False, "error": err}, 400)
        raw_sym = resolved or raw_sym
    symbol = _resolve_symbol(raw_sym)

    def _ov(key: str, default: Any, cast=None):
        v = payload.get(key)
        if v is None or (isinstance(v, str) and v.strip() == ""):
            v = b.get(key, default)
        if cast:
            try:
                v = cast(v)
            except (TypeError, ValueError):
                v = cast(default)
        return v

    settings = {
        "name": str(_ov("name", b.get("name") or f"Bot {symbol}")),
        "symbol": symbol,
        "enabled": int(_ov("enabled", 1, lambda x: int(x) if x is not None else 1)),
        "dry_run": int(_ov("dry_run", 1, lambda x: int(x) if x is not None else 1)),
        "base_quote": float(_ov("base_quote", 20.0, lambda x: float(x) if x is not None else 20.0)),
        "safety_quote": float(_ov("safety_quote", 20.0, lambda x: float(x) if x is not None else 20.0)),
        "max_safety": int(_ov("max_safety", 5, lambda x: int(x) if x is not None else 5)),
        "first_dev": float(_ov("first_dev", 0.015, lambda x: float(x) if x is not None else 0.015)),
        "step_mult": float(_ov("step_mult", 1.2, lambda x: float(x) if x is not None else 1.2)),
        "tp": float(_ov("tp", 0.015, lambda x: float(x) if x is not None else 0.015)),
        "market_type": market_type,
        "strategy_mode": str(_ov("strategy_mode", "auto")),
        "forced_strategy": str(_ov("forced_strategy", "")),
        "alpaca_mode": str(_ov("alpaca_mode", "paper")),
        "max_spend_quote": float(_ov("max_spend_quote", 0.0, lambda x: float(x) if x is not None else 0.0)),
        "poll_seconds": int(_ov("poll_seconds", 10, lambda x: int(x) if x is not None else 10)),
        "trend_filter": int(_ov("trend_filter", 0, lambda x: int(x) if x is not None else 0)),
        "trend_sma": int(min(500, max(10, int(_ov("trend_sma", 200, lambda x: int(x) if x is not None else 200))))),
        "max_open_orders": int(min(50, max(1, int(_ov("max_open_orders", 6, lambda x: int(x) if x is not None else 6))))),
        "daily_loss_limit_pct": float(_ov("daily_loss_limit_pct", 0.06, lambda x: float(x) if x is not None else 0.06)),
        "pause_hours": int(_ov("pause_hours", 6, lambda x: int(x) if x is not None else 6)),
        "auto_restart": int(_ov("auto_restart", 1, lambda x: int(x) if x is not None else 1)),
        "vol_gap_mult": float(_ov("vol_gap_mult", 1.0, lambda x: float(x) if x is not None else 1.0)),
        "tp_vol_mult": float(_ov("tp_vol_mult", 1.0, lambda x: float(x) if x is not None else 1.0)),
        "min_gap_pct": float(_ov("min_gap_pct", 0.003, lambda x: float(x) if x is not None else 0.003)),
        "max_gap_pct": float(_ov("max_gap_pct", 0.06, lambda x: float(x) if x is not None else 0.06)),
        "regime_hold_candles": int(_ov("regime_hold_candles", 2, lambda x: int(x) if x is not None else 2)),
        "regime_switch_ticks": int(_ov("regime_switch_ticks", 2, lambda x: int(x) if x is not None else 2)),
        "regime_switch_threshold": float(_ov("regime_switch_threshold", 0.6, lambda x: float(x) if x is not None else 0.6)),
        "max_total_exposure_pct": float(_ov("max_total_exposure_pct", 0.50, lambda x: float(x) if x is not None else 0.50)),
        "per_symbol_exposure_pct": float(_ov("per_symbol_exposure_pct", 0.15, lambda x: float(x) if x is not None else 0.15)),
        "min_free_cash_pct": float(_ov("min_free_cash_pct", 0.1, lambda x: float(x) if x is not None else 0.1)),
        "max_concurrent_deals": int(_ov("max_concurrent_deals", 6, lambda x: int(x) if x is not None else 6)),
        "spread_guard_pct": float(_ov("spread_guard_pct", 0.003, lambda x: float(x) if x is not None else 0.003)),
        "limit_timeout_sec": int(_ov("limit_timeout_sec", 45, lambda x: int(x) if x is not None else 45)),
        "max_drawdown_pct": float(_ov("max_drawdown_pct", 0.0, lambda x: float(x) if x is not None else 0.0)),
        "hard_sl_pct": float(_ov("hard_sl_pct", 0.0, lambda x: float(x) if x is not None else 0.0)),
    }
    if payload.get("base_order_quote") is not None:
        try:
            settings["base_quote"] = float(payload.get("base_order_quote"))
        except (TypeError, ValueError):
            pass
    if payload.get("per_symbol_pct") is not None:
        try:
            settings["per_symbol_exposure_pct"] = float(payload.get("per_symbol_pct"))
        except (TypeError, ValueError):
            pass
    if settings["max_spend_quote"] <= 0:
        settings["max_spend_quote"] = settings["base_quote"] + settings["safety_quote"] * settings["max_safety"]

    _sanitize_bot_numbers(settings)
    from services.exposure_cap import build_exposure_cap_error

    pv_cap = _portfolio_value_usd_for_exposure()
    cap_err = build_exposure_cap_error(
        pv_cap,
        float(settings.get("base_quote") or 0),
        float(settings.get("per_symbol_exposure_pct") or 0),
        float(settings.get("max_total_exposure_pct") or 0.5),
    )
    if cap_err:
        return _json(cap_err, 422)
    try:
        update_bot(int(bot_id), settings)
        # Extra fields not covered by the rigid update_bot SQL go through
        # the partial helper: adaptive_limit, max_slippage_pct, stop_loss_pct,
        # max_hold_hours, trailing_*, bot_type.
        extras: Dict[str, Any] = {}
        for k in ("adaptive_limit", "max_slippage_pct", "stop_loss_pct",
                  "max_hold_hours", "trailing_stop_enabled",
                  "trailing_activation_pct", "trailing_distance_pct",
                  "risk_profile", "bot_type"):
            if k in payload and payload[k] is not None:
                extras[k] = payload[k]
        if extras:
            try:
                from db import update_bot_fields as _ubf
                _ubf(int(bot_id), extras)
            except Exception as _eex:
                logger.warning("update_bot_fields extras failed: %s", _eex)
        return _json({"ok": True, "bot": get_bot(int(bot_id))})
    except Exception as e:
        return _json({"ok": False, "error": str(e)}, 500)


def _alpaca_client_for_bot(bot: Dict[str, Any]) -> Optional[AlpacaClient]:
    mode = str(bot.get("alpaca_mode") or "paper").lower()
    return alpaca_live if mode == "live" else alpaca_paper


def _alpaca_live_block_reason() -> str:
    """User-facing reason when Alpaca live is requested but not available."""
    if alpaca_live is not None:
        return ""
    if not LIVE_TRADING_ENABLED:
        return (
            "Alpaca live trading client not initialized. "
            "Set LIVE_TRADING_ENABLED=1 in .env and add Alpaca live API keys, then restart the app. "
            "Use Paper mode until then."
        )
    return (
        "Alpaca live trading client not initialized (startup failed). "
        "Check Alpaca live API keys in .env and app logs, then restart. Use Paper mode until then."
    )


def _can_start_bot_live(bot: Dict[str, Any]) -> Tuple[bool, str]:
    if bool(bot.get("dry_run", 1)):
        return True, ""
    market_type = str(bot.get("market_type") or "").strip().lower()
    if market_type == "crypto":
        if not KRAKEN_READY or kc is None:
            return False, KRAKEN_ERROR or "Kraken not ready"
        return True, ""
    if market_type == "stocks":
        cl = _alpaca_client_for_bot(bot)
        if cl is None:
            reason = _alpaca_live_block_reason() if (str(bot.get("alpaca_mode") or "paper").lower() == "live") else "Alpaca client not initialized"
            return False, reason
        try:
            adp = AlpacaAdapter(cl)
            adp.ensure_market(str(bot.get("symbol") or ""))
        except Exception as e:
            return False, f"Alpaca market check failed: {e}"
        return True, ""
    return False, f"Unknown market_type: {market_type}"


def _get_bot_client(bot: Dict[str, Any]):
    """
    Returns (client, is_kraken).
    If stocks, returns (AlpacaAdapter, False).
    If crypto, returns (KrakenClient, True).
    """
    market_type = str(bot.get("market_type") or "").strip().lower()
    if market_type not in ("stocks", "crypto"):
        raise HTTPException(status_code=400, detail=f"Unknown market_type: {market_type}")

    if market_type == "stocks":
        cl = _alpaca_client_for_bot(bot)
        if cl is None:
            logger.warning("_get_bot_client: Alpaca client missing bot_id=%s", bot.get("id"))
            raise HTTPException(status_code=503, detail="Alpaca client not available for stocks")
        try:
            adp = AlpacaAdapter(cl)
            sym = str(bot.get("symbol") or "")
            adp.ensure_market(sym)
        except Exception as e:
            logger.warning("_get_bot_client: Alpaca ensure_market failed bot_id=%s err=%s", bot.get("id"), e)
            raise HTTPException(status_code=503, detail=f"Alpaca market check failed: {e}")
        return adp, False

    if kc is None or not KRAKEN_READY:
        logger.warning("_get_bot_client: Kraken not ready bot_id=%s", bot.get("id"))
        raise HTTPException(status_code=503, detail=KRAKEN_ERROR or "Kraken not ready")
    return kc, True


# =========================================================
# Health
# =========================================================
@app.get("/health")
def health():
    """Ultra-fast liveness probe: returns in <50 ms regardless of DB / exchange state.

    nginx and the deploy script call this every few seconds and MUST never 504. We
    deliberately do not touch the DB, exchanges, or per-bot state here. Use
    `/health/full` (formerly the heavy version) or `/api/health` for the rich
    dashboard payload.
    """
    try:
        return {"status": "ok", "ok": True, "timestamp": int(time.time())}
    except Exception:
        # Last-ditch fallback — must always return 200.
        return {"status": "degraded", "ok": True, "timestamp": 0}


@app.get("/health/full")
def health_full():
    """Detailed health (DB, Kraken, Alpaca, disk, bots, threads). Wrapped in a 2 s
    timeout: if anything blocks (e.g. SQLite lock storm) we return `degraded`/200
    so the upstream never bubbles a 5xx.
    """
    deadline = time.time() + 2.0
    try:
        ts = now_ts()
    except Exception:
        ts = int(time.time())
    try:
        db_ok = True
        db_latency = 0
        db_start = time.time()
        try:
            init_db()
            db_latency = int((time.time() - db_start) * 1000)
        except Exception as e:
            db_ok = False
            logger.debug("DB health check failed: %s", e)

        if time.time() > deadline:
            return {"ok": True, "status": "degraded", "reason": "db_slow", "time": ts}

        status = "healthy" if db_ok else "degraded"

        # Check Kraken
        kr = False
        kraken_latency = 0
        try:
            kraken_start = time.time()
            kr = bool(_kraken_ready())
            kraken_latency = int((time.time() - kraken_start) * 1000)
        except Exception as e:
            logger.debug("Kraken health check failed: %s", e)

        # Check Alpaca (simplified latency check)
        alpaca_latency = 0

        # Check disk space
        disk_free_gb = 0.0
        try:
            import shutil
            disk_stat = shutil.disk_usage("/")
            disk_free_gb = round(disk_stat.free / (1024**3), 1)
        except Exception as e:
            logger.debug("Disk space check failed: %s", e)

        uptime_sec = int(time.time() - _APP_START_TIME) if _APP_START_TIME else 0
        last_autopilot_heartbeat = 0
        try:
            last_autopilot_heartbeat = int(get_setting("autopilot_last_heartbeat_ts", "0") or 0)
        except Exception:
            pass

        # Scanner status
        scanner_status = "idle"
        last_scan_ago = None
        try:
            _sp = _SCAN_PROGRESS
            if _sp.get("scan_start_ts", 0) > 0:
                scan_age = int(time.time()) - int(_sp["scan_start_ts"])
                last_scan_ago = f"{scan_age // 60} min ago" if scan_age > 60 else f"{scan_age}s ago"
                scanner_status = "running" if _sp.get("current_symbol") else "idle"
        except Exception:
            pass

        # Active bots count (time-budgeted: health must stay fast for nginx/upstream)
        bots_active = 0
        try:
            if bm:
                _t_budget = time.time() + 0.2
                for b in list_bots():
                    if time.time() > _t_budget:
                        break
                    try:
                        snap = bm.snapshot(int(b["id"]))
                        if snap and snap.get("running"):
                            bots_active += 1
                    except Exception:
                        pass
        except Exception:
            pass

        # Background thread health
        thread_health = {}
        try:
            for name, info in list(_background_threads.items()):
                t = info.get("thread")
                if t and t.is_alive():
                    thread_health[name] = "alive"
                elif name in _ONE_SHOT_THREADS:
                    thread_health[name] = "completed"  # expected to exit
                else:
                    thread_health[name] = "dead"
        except Exception:
            logger.exception("health/full: thread roll-up failed")

        # Background-loop health (Phase 1.2c).
        # Per-loop iteration outcome surfaced as 'ok' / 'degraded' /
        # 'failing' so /health/full doesn't lie when a loop is silently
        # erroring (the bug the brief explicitly called out).
        loop_health: Dict[str, Any] = {}
        try:
            with _loop_health_lock:
                snap = {k: dict(v) for k, v in _BACKGROUND_LOOP_HEALTH.items()}
            for name, st in snap.items():
                fails = int(st.get("consecutive_failures", 0) or 0)
                last_ok = float(st.get("last_ok_ts", 0) or 0)
                last_err = float(st.get("last_err_ts", 0) or 0)
                if fails == 0:
                    sub = "ok"
                elif fails < 3:
                    sub = "degraded"
                else:
                    sub = "failing"
                    status = "degraded"
                loop_health[name] = {
                    "status": sub,
                    "consecutive_failures": fails,
                    "last_ok_age_sec": int(time.time() - last_ok) if last_ok else None,
                    "last_err_age_sec": int(time.time() - last_err) if last_err else None,
                    "last_err": st.get("last_err"),
                }
        except Exception:
            logger.exception("health/full: loop-health roll-up failed")

        return {
            "ok": True,
            "status": status,
            "scanner": scanner_status,
            "last_scan": last_scan_ago,
            "bots_active": bots_active,
            "checks": {
                "database": {
                    "status": "ok" if db_ok else "error",
                    "latency_ms": db_latency
                },
                "kraken": {
                    "status": "ok" if kr else "error",
                    "latency_ms": kraken_latency,
                    "connected": kr,
                },
                "alpaca": {
                    "status": "ok" if ALPACA_PAPER_READY or ALPACA_LIVE_READY else ("not_configured" if not _ALPACA_KEYS_PRESENT else "error"),
                    "latency_ms": alpaca_latency,
                    "paper_ready": ALPACA_PAPER_READY,
                    "live_ready": ALPACA_LIVE_READY,
                },
                "disk_space": {
                    "status": "ok" if disk_free_gb > 1.0 else "warning",
                    "free_gb": disk_free_gb
                }
            },
            "threads": thread_health,
            "thread_restarts": {k: v for k, v in _THREAD_RESTART_COUNTS.items() if v > 0},
            "loops": loop_health,
            "uptime_seconds": uptime_sec,
            "version": "3.0.0",
            "time": ts,
            "fear_greed": {"value": _FEAR_GREED_CACHE.get("value", 50), "label": _FEAR_GREED_CACHE.get("label", "Neutral")},
            "last_autopilot_heartbeat_ts": last_autopilot_heartbeat if last_autopilot_heartbeat else None,
            "kraken_ready": kr,
            "alpaca_paper_ready": ALPACA_PAPER_READY,
            "alpaca_live_ready": ALPACA_LIVE_READY,
        }
    except Exception as e:
        logger.exception("health/full handler error")
        # Always return 200 so the upstream proxy never sees a 5xx from /health/full.
        return {"ok": True, "status": "degraded", "error": str(e)[:200], "time": ts}


@app.get("/api/debug/db_info")
def api_debug_db_info():
    """Return DB path, bot count, recommendation count, and cwd for persistence verification."""
    try:
        from db import DB_NAME
        init_db()
        bots = list_bots()
        bot_count = len(bots)
        last_bot_id = max((int(b.get("id") or 0) for b in bots), default=0)
        autopilot_bots = [b for b in bots if str(b.get("bot_type") or "").lower() == "autopilot"]
        enabled_bots = [b for b in bots if int(b.get("enabled", 0)) == 1]
        reco_count = 0
        try:
            from db import list_recommendations
            reco_count = len(list_recommendations("long", limit=500))
        except Exception:
            pass
        return _json({
            "ok": True,
            "db_path": DB_NAME,
            "db_path_abs": os.path.abspath(DB_NAME),
            "bot_count": bot_count,
            "autopilot_bot_count": len(autopilot_bots),
            "enabled_bot_count": len(enabled_bots),
            "recommendation_count": reco_count,
            "last_bot_id": last_bot_id if bot_count else None,
            "cwd": os.getcwd(),
        })
    except Exception as e:
        try:
            from db import DB_NAME as _db_path
        except Exception:
            _db_path = None
        return _json({"ok": False, "error": str(e)[:200], "db_path": _db_path})


@app.get("/api/debug/bm_ready")
def api_debug_bm_ready():
    """Return whether BotManager is initialized, with reason if not."""
    reason = _bm_not_ready_reason()
    return _json({
        "ok": True,
        "bm_ready": bm is not None,
        "reason": reason,
        "kraken_ready": KRAKEN_READY,
        "kraken_error": KRAKEN_ERROR or None,
        "alpaca_paper_ready": ALPACA_PAPER_READY,
        "alpaca_live_ready": ALPACA_LIVE_READY,
        "alpaca_error": ALPACA_ERROR or None,
    })


@app.get("/api/debug/watchdog_errors")
def api_debug_watchdog_errors():
    """Sticky errors from the health watchdog — bots that are enabled but failed to restart."""
    return _json({"ok": True, "errors": dict(_WATCHDOG_STICKY_ERRORS)})


@app.get("/api/debug/startup_status")
def api_debug_startup_status():
    """Full startup diagnostics for debugging autopilot and bot creation issues."""
    db_path = _STARTUP_STATUS.get("db_path")
    if db_path is None:
        try:
            from db import DB_NAME
            db_path = DB_NAME
        except Exception:
            pass
    paused = False
    kill_switch = False
    try:
        paused = bool(_pause_state())
        kill_switch = str(get_setting("kill_switch", "0")).strip().lower() in ("1", "true")
    except Exception:
        pass
    s = {
        "env_loaded_paths": _STARTUP_STATUS.get("env_loaded_paths") or [],
        "db_path": db_path,
        "db_path_abs": os.path.abspath(db_path) if db_path else None,
        "kraken_ready": _STARTUP_STATUS.get("kraken_ready", KRAKEN_READY),
        "kraken_error": KRAKEN_ERROR or None,
        "alpaca_ready": _STARTUP_STATUS.get("alpaca_ready", ALPACA_PAPER_READY or ALPACA_LIVE_READY),
        "alpaca_error": ALPACA_ERROR or None,
        "bm_ready": _STARTUP_STATUS.get("bm_ready", bm is not None),
        "bm_not_ready_reason": _bm_not_ready_reason(),
        "last_startup_error": _STARTUP_STATUS.get("last_startup_error"),
        "timestamp": _STARTUP_STATUS.get("timestamp"),
        "paused": paused,
        "kill_switch": kill_switch,
        "cwd": os.getcwd(),
        "uptime_sec": int(time.time() - _APP_START_TIME),
    }
    return _json({"ok": True, "startup_status": s})


@app.get("/api/bots/{bot_id}/data_health")
def api_bot_data_health(bot_id: int):
    """Per-bot data health: last ticker, last candle per TF, stale flags, spread, provider errors."""
    b = get_bot(int(bot_id))
    if not b:
        return _json({"ok": False, "error": "Bot not found"}, 404)
    symbol = str(b.get("symbol", ""))
    market = str(b.get("market_type", "crypto")).lower()
    result: Dict[str, Any] = {
        "ok": True, "bot_id": bot_id, "symbol": symbol,
        "ticker": None, "candles": {}, "gate": None, "provider_errors": [],
    }
    try:
        if market != "stocks" and kc:
            t = kc.fetch_ticker(symbol)
            bid = float(t.get("bid") or 0)
            ask = float(t.get("ask") or 0)
            last = float(t.get("last") or t.get("c") or 0)
            mid = (bid + ask) / 2 if bid and ask else 0
            spread_pct = ((ask - bid) / mid) if mid > 0 else None
            ts = t.get("timestamp") or t.get("ts")
            ticker_ts = (float(ts) / 1000.0 if float(ts) > 1e12 else float(ts)) if ts else None
            age = (time.time() - ticker_ts) if ticker_ts else None
            result["ticker"] = {
                "bid": bid or None, "ask": ask or None, "last": last or None,
                "spread_pct": round(spread_pct * 100, 4) if spread_pct is not None else None,
                "ticker_ts": ticker_ts, "age_sec": round(age, 1) if age is not None else None,
                "stale": age is not None and age > 120,
            }
        elif market == "stocks":
            client = alpaca_live or alpaca_paper
            if client:
                try:
                    t = client.get_ticker(symbol)
                    result["ticker"] = {"last": float(t.get("last") or 0), "source": "alpaca"}
                except Exception as e:
                    result["provider_errors"].append(f"Alpaca ticker: {e}")
    except Exception as e:
        result["provider_errors"].append(f"Ticker fetch: {e}")
    try:
        if bm:
            snap = bm.snapshot(bot_id)
            if snap.get("gate_details"):
                result["gate"] = snap["gate_details"]
            result["last_tick_ts"] = snap.get("last_tick_ts")
            result["running"] = snap.get("running", False)
            result["risk_state"] = snap.get("risk_state")
    except Exception as e:
        result["provider_errors"].append(f"Snapshot: {e}")
    try:
        for tf in ["1h", "4h", "1d"]:
            if market != "stocks" and bm and hasattr(bm, "ohlcv_cached"):
                candles = bm.ohlcv_cached(symbol, tf, 5, ttl_sec=600)
                if candles and len(candles) > 0:
                    last_ts = float(candles[-1][0]) / 1000.0 if float(candles[-1][0]) > 1e12 else float(candles[-1][0])
                    age = time.time() - last_ts
                    tf_sec = {"1h": 3600, "4h": 14400, "1d": 86400}.get(tf, 3600)
                    result["candles"][tf] = {
                        "count": len(candles),
                        "last_candle_ts": last_ts,
                        "age_sec": round(age, 1),
                        "stale": age > tf_sec * 3,
                    }
                else:
                    result["candles"][tf] = {"count": 0, "stale": True}
    except Exception as e:
        result["provider_errors"].append(f"OHLCV: {e}")
    return _json(result)


@app.get("/api/bots/{bot_id}/execution_gate")
def api_bot_execution_gate(bot_id: int):
    """Run the execution gate check for a bot and return the result (diagnostic only)."""
    b = get_bot(int(bot_id))
    if not b:
        return _json({"ok": False, "error": "Bot not found"}, 404)
    symbol = str(b.get("symbol", ""))
    try:
        from execution_gate import check_execution_gate, fetch_gate_inputs
        if kc:
            inputs = fetch_gate_inputs(kc, symbol, b)
        else:
            inputs = {"error": "No trading client available"}
        if inputs.get("error"):
            return _json({"ok": False, "error": inputs["error"], "inputs": inputs})
        gate = check_execution_gate(
            symbol=symbol,
            bid=inputs.get("bid"),
            ask=inputs.get("ask"),
            last_price=inputs.get("last_price"),
            ticker_ts=inputs.get("ticker_ts"),
            volume_24h=inputs.get("volume_24h"),
            bot_spread_guard_pct=float(b.get("spread_guard_pct", 0.003)),
            dry_run=bool(b.get("dry_run", 1)),
        )
        return _json({"ok": True, "gate": gate.to_dict(), "inputs": inputs})
    except Exception as e:
        return _json({"ok": False, "error": str(e)[:200]})


@app.get("/ready")
def ready():
    """Readiness: DB + Alpaca + Kraken. Fast probe for deploy/k8s."""
    checks = {"db": False, "alpaca": False, "kraken": False}
    try:
        init_db()
        checks["db"] = True
    except Exception as e:
        return _json({"ok": False, "checks": checks, "error": f"db: {e}"}, 503)
    try:
        if ALPACA_PAPER_READY and alpaca_paper:
            try:
                _ = _retry_with_backoff(lambda: alpaca_paper.get_account() or {}, max_retries=2, base_delay=0.5)
                checks["alpaca"] = True
            except Exception as e:
                logger.warning(f"Alpaca health check failed (will retry): {e}")
                checks["alpaca"] = False
        elif not os.getenv("ALPACA_API_KEY_PAPER"):
            checks["alpaca"] = True  # Not configured, skip
        else:
            checks["alpaca"] = ALPACA_PAPER_READY
    except Exception as e:
        return _json({"ok": False, "checks": checks, "error": f"alpaca: {e}"}, 503)
    try:
        checks["kraken"] = _kraken_ready() if os.getenv("KRAKEN_API_KEY") else True
    except Exception:
        checks["kraken"] = False
    ok = checks["db"] and (checks["alpaca"] or not os.getenv("ALPACA_API_KEY_PAPER"))
    return _json({"ok": ok, "checks": checks})


# =========================================================
# API: global helpers for UI
# =========================================================
@app.get("/api/startup_status")
def api_startup_status():
    """Startup diagnostics - flask, db, alpaca, websocket, autopilot, env_loaded_paths, db_path, bm_ready, last_startup_error, timestamp."""
    s = dict(_STARTUP_STATUS)
    s["alpaca_ready"] = ALPACA_PAPER_READY or ALPACA_LIVE_READY
    s["kraken_ready"] = s.get("kraken_ready", KRAKEN_READY)
    s["bm_ready"] = s.get("bm_ready", bm is not None)
    if alpaca_paper:
        try:
            s["alpaca_buying_power"] = float((_retry_with_backoff(lambda: alpaca_paper.get_account() or {}, max_retries=2, base_delay=0.5)).get("buying_power", 0))
        except Exception as e:
            logger.debug(f"Failed to fetch Alpaca buying power in status endpoint: {e}")
            pass
    try:
        import autopilot
        s["autopilot_enabled"] = autopilot.is_autopilot_enabled()
        s["autopilot_bots"] = len([b for b in list_bots() if str(b.get("bot_type") or "").lower() == "autopilot"])
    except Exception:
        pass
    return _json({"ok": True, "startup": s})


@app.get("/api/health")
def api_health():
    """Fast liveness for monitors: SQLite + version only (typically <500ms)."""
    t0 = time.time()
    db_ok = True
    try:
        init_db()
    except Exception:
        db_ok = False
    db_ms = int((time.time() - t0) * 1000)
    bots_running = 0
    bots_total = 0
    try:
        _rows = list_bots()
        bots_total = len(_rows)
        bots_running = sum(1 for x in _rows if int(x.get("last_running", 0)) == 1)
    except Exception:
        pass
    return _json({
        "ok": db_ok,
        "status": "healthy" if db_ok else "degraded",
        "db_ok": db_ok,
        "db_latency_ms": db_ms,
        "version": "3.0.0",
        "ts": now_ts(),
        "bots": {"running": bots_running, "total": bots_total},
    })


@app.get("/api/health/deep")
def api_health_deep():
    """Deep health: brokers, bot manager, background threads (cached)."""
    _now = time.time()
    with _globals_lock:
        _cached = _HEALTH_CACHE.get("result")
        _cached_ts = _HEALTH_CACHE.get("ts", 0.0)
    if _cached is not None and (_now - _cached_ts) < _HEALTH_CACHE_TTL:
        return _json(_cached)
    try:
        db_ok = True
        try:
            init_db()
        except Exception:
            db_ok = False
        kr = bool(_kraken_ready())
        expanded: Dict[str, Any] = {}
        try:
            from health_monitor import build_expanded_health
            expanded = build_expanded_health(
                kraken_ready=kr,
                kraken_error=KRAKEN_ERROR or "",
                alpaca_paper_ready=ALPACA_PAPER_READY,
                alpaca_live_ready=ALPACA_LIVE_READY,
                alpaca_error=ALPACA_ERROR or "",
                bot_manager_ready=bm is not None,
                db_ok=db_ok,
                list_bots_fn=list_bots,
                last_portfolio_ts=_last_portfolio_ts,
                last_reco_short_ts=_last_reco_short_ts,
                last_reco_long_ts=_last_reco_long_ts,
            )
        except Exception as e:
            logger.debug("health_monitor expand failed: %s", e)
            expanded = {"ok": db_ok, "status": "healthy" if db_ok else "degraded"}
        with _thread_start_lock:
            expanded["threads_started"] = list(_thread_started.keys())
        thread_health = {}
        for name, info in _background_threads.items():
            t = info.get("thread")
            alive = t.is_alive() if t else False
            status = "alive" if alive else ("completed" if name in _ONE_SHOT_THREADS else "dead")
            thread_health[name] = {
                "alive": alive,
                "status": status,
                "uptime_sec": int(time.time() - info.get("started_at", 0)),
            }
        expanded["thread_health"] = thread_health
        expanded["last_portfolio_ts"] = _last_portfolio_ts
        expanded["last_reco_short_ts"] = _last_reco_short_ts
        expanded["last_reco_long_ts"] = _last_reco_long_ts
        expanded["kraken_last_candle_ts"] = _kraken_last_candle_ts
        expanded["alpaca_last_candle_ts"] = _alpaca_last_candle_ts
        try:
            from db import DB_NAME as _db_name
            expanded["db_path"] = _db_name
            expanded["db_path_abs"] = os.path.abspath(_db_name)
        except Exception:
            expanded.setdefault("db_path", None)
            expanded.setdefault("db_path_abs", None)
        expanded["uptime_sec"] = int(time.time() - _APP_START_TIME) if _APP_START_TIME else 0
        try:
            expanded["last_autopilot_heartbeat_ts"] = int(get_setting("autopilot_last_heartbeat_ts", "0") or 0) or None
        except Exception:
            expanded["last_autopilot_heartbeat_ts"] = None
        expanded["endpoint"] = "deep"
        result = expanded
        with _globals_lock:
            _HEALTH_CACHE["result"] = result
            _HEALTH_CACHE["ts"] = time.time()
        return _json(result)
    except Exception as e:
        return _json({"ok": False, "status": "error", "error": str(e), "ts": now_ts()}, 503)


@app.get("/api/health/metrics")
def api_health_prometheus():
    """Prometheus metrics (optional, ENABLE_PROMETHEUS=1)."""
    try:
        from health_monitor import prometheus_metrics
        out = prometheus_metrics()
        if out:
            from fastapi.responses import PlainTextResponse
            return PlainTextResponse(out, media_type="text/plain; version=0.0.4")
    except Exception:
        pass
    return _json({"ok": False, "message": "Prometheus disabled"})


@app.get("/api/health/detailed")
def api_health_detailed():
    """Enhanced health check with detailed component status."""
    try:
        import time as time_module

        # Determine overall status
        status = "healthy"
        uptime_sec = int(time_module.time() - _APP_START_TIME) if _APP_START_TIME else 0

        # Database check
        db_status = "ok"
        db_latency_ms = 0
        try:
            start = time_module.time()
            init_db()
            db_latency_ms = int((time_module.time() - start) * 1000)
        except Exception as e:
            db_status = "error"
            status = "degraded"

        # Kraken check
        kraken_status = "ok" if KRAKEN_READY and kc else ("error" if KRAKEN_ERROR else "not_configured")
        kraken_message = KRAKEN_ERROR if KRAKEN_ERROR else ("Ready" if KRAKEN_READY else "Not initialized")

        # Alpaca Paper check
        alpaca_paper_status = "ok" if ALPACA_PAPER_READY else ("error" if ALPACA_ERROR else "not_configured")
        alpaca_paper_message = ALPACA_ERROR if ALPACA_ERROR else ("Ready" if ALPACA_PAPER_READY else "Not initialized")

        # Alpaca Live check
        alpaca_live_status = "ok" if ALPACA_LIVE_READY else ("not_configured")
        alpaca_live_message = "Ready" if ALPACA_LIVE_READY else "Not configured/Live trading disabled"

        # Set status to degraded if any exchange is down
        if kraken_status == "error" or alpaca_paper_status == "error":
            status = "degraded"

        checks = {
            "database": {
                "status": db_status,
                "latency_ms": db_latency_ms
            },
            "kraken": {
                "status": kraken_status,
                "message": kraken_message
            },
            "alpaca_paper": {
                "status": alpaca_paper_status,
                "message": alpaca_paper_message
            },
            "alpaca_live": {
                "status": alpaca_live_status,
                "message": alpaca_live_message
            }
        }

        return _json({
            "status": status,
            "version": "2.0.0",
            "uptime_seconds": uptime_sec,
            "checks": checks,
            "timestamp": int(time_module.time())
        })
    except Exception as e:
        logger.exception("api_health_detailed failed")
        return _json({
            "status": "error",
            "version": "2.0.0",
            "uptime_seconds": 0,
            "checks": {},
            "error": str(e)
        }, 503)


@app.get("/api/comprehensive_health")
def api_comprehensive_health():
    """Run comprehensive health check and return issues (if any)."""
    try:
        ok, issues = _comprehensive_health_check()
        return _json({"ok": ok, "issues": issues, "message": "All systems operational" if ok else f"{len(issues)} issue(s) found"})
    except Exception as e:
        return _json({"ok": False, "issues": [str(e)], "error": str(e)}, 500)


@app.get("/api/diag/network")
def api_diag_network():
    """Diagnose Alpaca connectivity from this server: DNS resolution + HTTPS HEAD."""
    result: Dict[str, Any] = {
        "ok": False,
        "dns": {},
        "https_head": {},
        "alpaca_data_feed": os.getenv("ALPACA_DATA_FEED", "").strip() or "(default)",
        "alpaca_mode": "paper",
    }
    host = "paper-api.alpaca.markets"
    # DNS resolution
    try:
        addrs = socket.getaddrinfo(host, 443, socket.AF_UNSPEC, socket.SOCK_STREAM)
        result["dns"] = {"ok": True, "host": host, "resolved": [a[4][0] for a in addrs]}
    except Exception as e:
        result["dns"] = {"ok": False, "host": host, "error": str(e)}
    # HTTPS HEAD
    try:
        import urllib.request
        req = urllib.request.Request("https://" + host, method="HEAD")
        with urllib.request.urlopen(req, timeout=10) as resp:
            result["https_head"] = {"ok": True, "status_code": resp.status}
    except Exception as e:
        result["https_head"] = {"ok": False, "error": str(e)}
    result["ok"] = result["dns"].get("ok", False) and result["https_head"].get("ok", False)
    return _json(result)


@app.get("/api/diag/scan_full")
def api_diag_scan_full():
    """
    Diagnostic: scan pipeline state for Market Screener troubleshooting.
    Returns: RECO_STATE, SCAN_PROGRESS, cache sizes, thread count, intervals, DB counts.
    """
    now_ts = int(time.time())
    with _globals_lock:
        short_s = (_RECO_STATE.get("short") or {}).copy()
        medium_s = (_RECO_STATE.get("medium") or {}).copy()
        long_s = (_RECO_STATE.get("long") or {}).copy()
        prog = dict(_SCAN_PROGRESS)
    try:
        counts = count_recommendations_by_horizon()
    except Exception as e:
        counts = {"error": str(e)}
    thread_count = threading.active_count()
    ohlcv_cache_size = len(_RECO_OHLCV_CACHE)
    ohlcv_fetch_locks = len(_OHLCV_FETCH_LOCKS)
    ram_mb = None
    try:
        import resource
        rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        ram_mb = round(rss / 1024.0, 2)
    except ImportError:
        try:
            import psutil
            proc = psutil.Process()
            ram_mb = round(proc.memory_info().rss / (1024 * 1024), 2)
        except Exception:
            pass
    except Exception:
        pass
    short_last = short_s.get("last_run_ts", 0)
    medium_last = medium_s.get("last_run_ts", 0)
    long_last = long_s.get("last_run_ts", 0)
    return _json({
        "ok": True,
        "timestamp": now_ts,
        "reco_state": {"short": short_s, "medium": medium_s, "long": long_s},
        "scan_progress": prog,
        "intervals_sec": {
            "short": RECO_SHORT_EVERY_SEC, "medium": RECO_MEDIUM_EVERY_SEC, "long": RECO_LONG_EVERY_SEC,
        },
        "ages_sec": {
            "short": now_ts - short_last if short_last else None,
            "medium": now_ts - medium_last if medium_last else None,
            "long": now_ts - long_last if long_last else None,
        },
        "db_counts": counts,
        "thread_count": thread_count,
        "ohlcv_cache_entries": ohlcv_cache_size,
        "ohlcv_fetch_locks": ohlcv_fetch_locks,
        "ram_mb": ram_mb,
    })


@app.get("/api/stats")
def api_stats():
    """UnifiedAlpacaClient stats: cache hits, rate limit, WebSocket subscriptions."""
    client = alpaca_live or alpaca_paper
    if not client:
        return _json({"ok": False, "message": "Alpaca not configured"})
    if not hasattr(client, "get_stats"):
        return _json({"ok": True, "message": "Using legacy AlpacaClient (no stats)", "client": "legacy"})
    try:
        stats = client.get_stats()
        return _json({"ok": True, "client": "unified", **stats})
    except Exception as e:
        return _json({"ok": False, "error": str(e)}, 500)


@app.get("/api/websocket_stats")
def api_websocket_stats():
    """WebSocket, cache, and rate limiter stats (alias for /api/stats when using UnifiedAlpacaClient)."""
    return api_stats()


@app.post("/api/notifications/test")
def api_notifications_test():
    """Send a test Discord notification to verify webhook configuration."""
    webhook = os.getenv("DISCORD_WEBHOOK_URL", "").strip()
    if not webhook:
        return _json({"ok": False, "error": "DISCORD_WEBHOOK_URL not configured"}, 400)
    try:
        import requests as _req
        resp = _req.post(webhook, json={"content": "✅ Eirin Bot: test notification (Discord webhook working)"}, timeout=5)
        if resp.status_code in (200, 204):
            return _json({"ok": True, "message": "Test notification sent"})
        return _json({"ok": False, "error": f"Discord returned {resp.status_code}: {resp.text[:200]}"}, 502)
    except Exception as e:
        return _json({"ok": False, "error": str(e)}, 500)


# NOTE: GET /api/safety_check was removed. It returned a looser live_ready
# (only checked api_auth + allow_live + exchange ready + kill_switch_state) and
# diverged from /api/safety/checklist's strict 10-item gate. All callers now
# use /api/safety/checklist, which calls services.safety_checklist
# .compute_live_readiness() — the single source of truth for live readiness.


@app.get("/api/pnl")
def api_pnl():
    """
    Used by dashboard / UI to show today and total realized PnL.
    """
    try:
        today = pnl_summary(_midnight_local_ts())
        total = pnl_summary(0)
        return _json({"ok": True, "today": today, "total": total})
    except Exception as e:
        return _json({"ok": False, "error": f"{type(e).__name__}: {e}", "today": {}, "total": {}}, 500)


@app.get("/api/symbols")
def api_symbols(quote: str = "USD"):
    if not _kraken_ready():
        return _json({"ok": False, "error": KRAKEN_ERROR or "Kraken not ready", "symbols": []}, 503)
    try:
        symbols = kc.list_spot_symbols(quote=quote)
        return _json({"ok": True, "symbols": symbols})
    except Exception as e:
        return _json({"ok": False, "error": f"{type(e).__name__}: {e}", "symbols": []}, 500)


@app.get("/api/notifications")
def api_notifications(limit: int = 50, unread_only: bool = False):
    """Get recent notifications for in-app display."""
    try:
        from notification_manager import get_notifications, get_unread_count

        notifications = get_notifications(limit=limit, unread_only=unread_only)
        unread_count = get_unread_count()

        return _json({
            "ok": True,
            "notifications": notifications,
            "unread_count": unread_count,
        })
    except Exception as e:
        logger.error(f"Notifications API error: {type(e).__name__}: {e}")
        return _json({"ok": False, "error": f"{type(e).__name__}: {e}", "notifications": [], "unread_count": 0}, 500)


@app.post("/api/notifications/{notification_id}/read")
def api_mark_notification_read(notification_id: int):
    """Mark a notification as read."""
    try:
        from notification_manager import mark_notification_read

        success = mark_notification_read(notification_id)
        return _json({"ok": success, "notification_id": notification_id})
    except Exception as e:
        logger.error(f"Mark notification read error: {type(e).__name__}: {e}")
        return _json({"ok": False, "error": f"{type(e).__name__}: {e}"}, 500)


@app.get("/api/notifications/unread_count")
def api_notifications_unread_count():
    """Get count of unread notifications."""
    _now = time.time()
    with _globals_lock:
        _cached = _NOTIF_UNREAD_CACHE.get("result")
        _cached_ts = _NOTIF_UNREAD_CACHE.get("ts", 0.0)
    if _cached is not None and (_now - _cached_ts) < _NOTIF_UNREAD_CACHE_TTL:
        return _json(_cached)
    try:
        from notification_manager import get_unread_count
        count = get_unread_count()
        result = {"ok": True, "unread_count": count}
        with _globals_lock:
            _NOTIF_UNREAD_CACHE["result"] = result
            _NOTIF_UNREAD_CACHE["ts"] = _now
        return _json(result)
    except Exception as e:
        logger.error(f"Unread count error: {type(e).__name__}: {e}")
        return _json({"ok": False, "error": f"{type(e).__name__}: {e}", "unread_count": 0}, 500)


@app.get("/api/alpaca/symbols")
def api_alpaca_symbols():
    """Get list of tradeable stock symbols from Alpaca"""
    try:
        # Use paper trading client to get symbols (same symbols for both paper and live)
        if not alpaca_paper:
            return _json({"ok": False, "error": "Alpaca not initialized", "symbols": []}, 503)
        
        # Get top tradeable stocks
        assets = _retry_with_backoff(lambda: alpaca_paper.search_assets(query="", asset_class="us_equity"), max_retries=2, base_delay=0.5)
        
        # Filter to tradeable and sort by common popularity
        popular_symbols = [
            "AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META", "TSLA", "JPM", "V", "JNJ",
            "WMT", "JPM", "MA", "PG", "UNH", "DIS", "HD", "PYPL", "BAC", "VZ",
            "ADBE", "NFLX", "CRM", "NKE", "CMCSA", "PFE", "T", "INTC", "CSCO", "ABT",
            "KO", "PEP", "MRK", "AVGO", "TMO", "COST", "ABBV", "ACN", "TXN", "NEE",
            "DHR", "LLY", "MDT", "UNP", "BMY", "PM", "QCOM", "HON", "UPS", "LOW"
        ]
        
        # Create symbols list with names
        symbols = []
        for asset in assets[:500]:  # Limit to 500 most common
            symbol = asset.get("symbol", "")
            name = asset.get("name", symbol)
            
            # Prioritize popular symbols
            if symbol in popular_symbols:
                symbols.insert(0, {"symbol": symbol, "name": name})
            else:
                symbols.append({"symbol": symbol, "name": name})
        
        return _json({"ok": True, "symbols": symbols})
    except Exception as e:
        logger.error(f"Alpaca symbols error: {type(e).__name__}: {e}")
        return _json({"ok": False, "error": f"{type(e).__name__}: {e}", "symbols": []}, 500)


@app.get("/api/prices")
async def api_prices(symbols: str = "", market_type: str = "all"):
    """
    Batch price fetch: crypto (Kraken bulk ticker) + stocks (Alpaca snapshots
    w/ latestTrade) in parallel, 3s per venue, optional partial payload.
    Per-symbol TTL cache (10s) in services.prices_fetch.
    """
    from services import prices_fetch as _pf

    payload = await _pf.fetch_prices_async(symbols or "", market_type or "all", timeout_sec=3.0)
    return _json(payload)


@app.get("/api/icons/map")
def api_icons_map():
    """Base-symbol → CoinGecko id for client-side throttled icon loads."""
    from services.icon_map import SYMBOL_TO_COINGECKO_ID

    return _json({"ok": True, "map": SYMBOL_TO_COINGECKO_ID})


@app.get("/api/market/ticker")
def api_market_ticker(symbol: str):
    """Get ticker data for a symbol, routing to appropriate provider"""
    market_type = classify_symbol(symbol)
    
    if market_type == "stock":
        # Route to Alpaca
        client = alpaca_live if alpaca_live else alpaca_paper
        if not client:
            return _json({"ok": False, "error": "Alpaca not configured for stock symbols"}, 503)
        
        try:
            ticker = client.get_ticker(symbol)
            return _json({"ok": True, **ticker})
        except Exception as e:
            return _json({"ok": False, "error": f"{type(e).__name__}: {e}"}, 500)
    
    else:
        # Crypto path - existing Kraken logic
        if not _kraken_ready():
            return _json({"ok": False, "error": KRAKEN_ERROR or "Kraken not ready"}, 503)
        sym = _resolve_symbol(symbol)
        mk = _markets()
        if mk and sym not in mk:
            return _json({"ok": False, "error": f"Symbol not found on Kraken: {sym}"}, 400)
        data = _ticker_cached(sym, ttl_sec=30) or {}
        return _json({"ok": True, **data})


@app.get("/api/market/ohlcv")
def api_market_ohlcv(symbol: str, tf: str = "1h", limit: int = 500):
    """Get OHLCV data, routing to appropriate provider"""
    market_type = classify_symbol(symbol)
    
    # 1. STOCK ROUTING
    if market_type == "stock":
        try:
            client = alpaca_live if alpaca_live else alpaca_paper
            if client:
                safe_limit = int(max(10, min(int(limit), 1000)))
                # Alpaca get_ohlcv(symbol, timeframe, limit)
                candles = client.get_ohlcv(symbol, tf, safe_limit)
                return _json({"ok": True, "symbol": symbol, "tf": tf, "candles": candles})
            else:
                 return _json({"ok": False, "error": "Alpaca not ready", "candles": []}, 503)
        except Exception as e:
            return _json({"ok": False, "error": str(e), "candles": []}, 500)

    # 2. CRYPTO ROUTING (Existing Logic)
    if not _kraken_ready():
        return _json({"ok": False, "error": KRAKEN_ERROR or "Kraken not ready", "candles": []}, 503)
    sym = _resolve_symbol(symbol)
    mk = _markets()
    if mk and sym not in mk:
        return _json({"ok": False, "error": f"Symbol not found on Kraken: {sym}", "candles": []}, 400)
    safe_tf = _sanitize_tf(tf)
    safe_limit = int(max(10, min(int(limit), 1000)))
    ttl = max(30, min(300, _tf_seconds(safe_tf)))
    candles = _ohlcv_cached(sym, safe_tf, safe_limit, ttl)
    return _json({"ok": True, "symbol": sym, "tf": safe_tf, "candles": candles})


def _check_trading_allowed(bot_id: Optional[int] = None) -> Optional[Dict[str, Any]]:
    """Returns error dict if trading blocked (kill switch, global pause, circuit breaker, data quality). None if OK."""
    if _pause_state():
        return {"ok": False, "error": "Trading paused (global pause). Resume from Safety or Pause page."}
    if _kill_switch_state():
        return {"ok": False, "error": "Kill switch is on. Turn it off on the Safety page to trade."}
    try:
        from circuit_breaker import is_emergency_stop_active, is_bot_circuit_open, get_bot_pause_until
        if is_emergency_stop_active():
            return {"ok": False, "error": "Emergency stop active. Exchange errors persist. Check /api/health."}
        if bot_id is not None and is_bot_circuit_open(int(bot_id)):
            until = get_bot_pause_until(int(bot_id))
            return {"ok": False, "error": f"Circuit breaker: bot paused until errors clear (until ts {until})"}
    except ImportError:
        pass
    try:
        from data_validator import is_data_quality_degraded
        if is_data_quality_degraded():
            return {"ok": False, "error": "Data quality degraded (5+ issues in 15 min). Trading paused."}
    except ImportError:
        pass
    return None


@app.post("/api/orders/buy")
async def api_orders_buy(request: Request):
    try:
        payload = await request.json()
    except Exception:
        # Handle case where body might be already read or invalid
        return _json({"ok": False, "error": "Invalid payload or body stream consumed"}, 400)
        
    if not isinstance(payload, dict):
        return _json({"ok": False, "error": "Invalid payload"}, 400)

    dry_run = bool(payload.get("dry_run", True))
    if not dry_run:
        block = _check_trading_allowed(bot_id=None)
        if block:
            return _json(block, 503)
    
    raw_symbol = str(payload.get("symbol") or "")
    market_type = classify_symbol(raw_symbol)
    quote_usd = float(payload.get("quote_usd") or 0.0)
    limit_price = float(payload.get("limit_price") or 0.0)
    dry_run = bool(payload.get("dry_run", True))

    if quote_usd <= 0 or limit_price <= 0:
        return _json({"ok": False, "error": "quote_usd and limit_price must be > 0"}, 400)

    # 1. STOCK ROUTING
    if market_type == "stock":
        client = alpaca_live if alpaca_live else alpaca_paper
        if not client:
            return _json({"ok": False, "error": "Alpaca not configured for stocks"}, 503)
            
        # CRITICAL: Calculate and validate amount before placing order
        if limit_price <= 0:
            return _json({"ok": False, "error": f"Invalid limit_price: {limit_price}"}, 400)
        
        base_amount = quote_usd / limit_price
        
        # CRITICAL: Reject zero or invalid amounts
        import math
        if base_amount <= 0 or math.isnan(base_amount) or math.isinf(base_amount):
            return _json({
                "ok": False, 
                "error": f"Order size invalid: amount={base_amount}, quote_usd={quote_usd}, limit_price={limit_price}. Order skipped."
            }, 400)
        
        if dry_run:
             return _json({
                "ok": True, 
                "message": "Dry run: stock limit buy simulated.",
                "order": {
                    "symbol": raw_symbol, "side": "buy", "type": "limit",
                    "price": limit_price, "amount": base_amount, "market_type": "stock"
                }
            })
            
        try:
            # Alpaca place_limit_order(symbol, qty, limit_price, side, time_in_force)
            # Check if client has this method or similar
            if hasattr(client, "place_limit_order"):
                order = client.place_limit_order(raw_symbol, base_amount, limit_price, "buy")
            else:
                 return _json({"ok": False, "error": "Alpaca client missing place_limit_order"}, 500)
                 
            return _json({"ok": True, "message": "Stock limit buy placed.", "order": order})
        except Exception as e:
            return _json({"ok": False, "error": f"Alpaca Order Failed: {e}"}, 500)

    # 2. CRYPTO ROUTING (Existing Logic)
    if not dry_run and not _kraken_ready():
        return _json({"ok": False, "error": KRAKEN_ERROR or "Kraken not ready"}, 503)

    symbol = _resolve_symbol(raw_symbol)
    mk = _markets()
    if mk and symbol not in mk:
        return _json({"ok": False, "error": f"Symbol not found on Kraken: {symbol}"}, 400)

    spread = _safe_spread_pct(symbol)
    if spread is not None and spread > (RECO_MAX_SPREAD_PCT * 2):
        return _json({"ok": False, "error": f"Spread too wide ({spread:.2%})"}, 400)

    # CRITICAL: Calculate and validate amount before placing order
    if limit_price <= 0:
        return _json({"ok": False, "error": f"Invalid limit_price: {limit_price}"}, 400)
    
    base_amount = quote_usd / limit_price
    
    # CRITICAL: Reject zero or invalid amounts
    import math
    if base_amount <= 0 or math.isnan(base_amount) or math.isinf(base_amount):
        return _json({
            "ok": False, 
            "error": f"Order size invalid: amount={base_amount}, quote_usd={quote_usd}, limit_price={limit_price}. Order skipped."
        }, 400)
    
    if dry_run:
        return _json(
            {
                "ok": True,
                "message": "Dry run: limit buy simulated.",
                "order": {
                    "symbol": symbol,
                    "side": "buy",
                    "type": "limit",
                    "price": limit_price,
                    "amount": base_amount,
                },
            }
        )

    try:
        order = kc.create_limit_buy_base(symbol, base_amount, limit_price)
        return _json({"ok": True, "message": "Limit buy placed.", "order": _serialize_order(order)})
    except Exception as e:
        return _json({"ok": False, "error": f"{type(e).__name__}: {e}"}, 500)


@app.get("/api/market/overview")
def api_market_overview(quote: str = "USD", limit: int = 50, market_type: str = "crypto"):
    """
    Returns categorized market data for the Explore dashboard.
    Structure: { "ok": true, "gainers": [], "losers": [], "hot": [], "trending": [] }
    """
    # 1. Stocks Mode
    if market_type == "stocks":
        try:
            if alpaca_paper or alpaca_live:
                client = alpaca_live if alpaca_live else alpaca_paper
                data = client.get_top_movers() 
                return _json({
                    "ok": True,
                    "gainers": data.get("gainers", []),
                    "losers": data.get("losers", []),
                    "hot": data.get("hot", []),
                    "trending": data.get("hot", []),
                    "market_type": "stocks"
                })
            else:
                 return _json({"ok": False, "error": "Alpaca not ready"}, 503)
        except Exception as e:
            logger.error(f"Alpaca overview failed: {e}")
            return _json({"ok": False, "error": str(e)}, 500)

    # 2. Crypto Mode (Default)
    if not _kraken_ready():
        return _json({"ok": False, "error": "No market data available"}, 503)

    try:
        tickers = _tickers_batch_cached(ttl_sec=30)
        if not tickers:
            try:
                tickers = kc.ex.fetch_tickers()
            except Exception as _ft_err:
                if "safeMarket" in str(_ft_err) or "disambiguate" in str(_ft_err):
                    logger.warning("market_overview: Kraken fetch_tickers safeMarket error, using cached batch: %s", _ft_err)
                    tickers = {}
                else:
                    raise

        parsed = []
        q_upper = quote.upper()

        for sym, t in (tickers or {}).items():
            if not isinstance(t, dict):
                continue
            if f"/{q_upper}" not in sym:
                continue

            try:
                vol = float(t.get("quoteVolume") or 0)
                if vol < 50000:
                    continue
                change = float(t.get("percentage") or 0)
                close = float(t.get("last") or t.get("close") or 0)
            except (TypeError, ValueError):
                continue

            parsed.append({
                "symbol": sym,
                "last": close,
                "percentage": change,
                "quoteVolume": vol
            })

        parsed.sort(key=lambda x: x["percentage"], reverse=True)
        gainers = parsed[:6]
        losers = sorted(parsed, key=lambda x: x["percentage"])[:6]

        parsed.sort(key=lambda x: x["quoteVolume"], reverse=True)
        hot = parsed[:6]

        return _json({
            "ok": True,
            "gainers": gainers,
            "losers": losers,
            "hot": hot,
            "trending": hot
        })

    except Exception as e:
        logger.error("market_overview crypto failed: %s", e)
        return _json({"ok": False, "error": str(e)}, 500)


@app.get("/api/portfolio")
def api_portfolio():
    _now = time.time()
    with _globals_lock:
        _cached = _PORTFOLIO_CACHE.get("result")
        _cached_ts = _PORTFOLIO_CACHE.get("ts", 0.0)
    if _cached is not None and (_now - _cached_ts) < _PORTFOLIO_CACHE_TTL:
        return _json(_cached)
    snap = _portfolio_snapshot()
    with _globals_lock:
        history = list(PORT_HISTORY[-500:])
    result = {"ok": True, "portfolio": snap, "history": history}
    with _globals_lock:
        _PORTFOLIO_CACHE["result"] = result
        _PORTFOLIO_CACHE["ts"] = _now
    return _json(result)


@app.get("/api/dashboard/pnl-history")
def api_dashboard_pnl_history(days: int = 7):
    """Daily realized PnL for the last N days."""
    try:
        import sqlite3 as _sq
        db_path = os.environ.get("DB_PATH", "botdb.sqlite3")
        con = _sq.connect(db_path)
        con.row_factory = _sq.Row
        _days = max(1, min(90, int(days)))
        cutoff = int(time.time()) - _days * 86400
        rows = con.execute(
            """SELECT date(closed_at, 'unixepoch') as day, SUM(realized_pnl_quote) as pnl, COUNT(*) as deals
               FROM deals WHERE closed_at >= ? AND closed_at > 0 AND realized_pnl_quote IS NOT NULL
               GROUP BY day ORDER BY day""",
            (cutoff,),
        ).fetchall()
        con.close()
        result = [{"date": r["day"], "pnl": round(float(r["pnl"] or 0), 2), "deals": int(r["deals"] or 0)} for r in rows]
        return _json({"ok": True, "days": result})
    except Exception as e:
        return _json({"ok": True, "days": [], "error": str(e)[:100]})


@app.get("/api/portfolio/performance")
def api_portfolio_performance(timeframe: str = "1D"):
    """Portfolio performance for charts. Uses PORT_HISTORY."""
    try:
        snap = _portfolio_snapshot()
        total_usd = float(snap.get("total_usd") or 0)
        with _globals_lock:
            history = list(PORT_HISTORY[-500:])
        cutoff_ts = 0
        if timeframe == "1H":
            cutoff_ts = now_ts() - 3600
        elif timeframe == "4H":
            cutoff_ts = now_ts() - 14400
        elif timeframe == "1D":
            cutoff_ts = now_ts() - 86400
        elif timeframe == "1W":
            cutoff_ts = now_ts() - 604800
        elif timeframe == "1M":
            cutoff_ts = now_ts() - 2592000
        elif timeframe == "3M":
            cutoff_ts = now_ts() - 7776000
        elif timeframe == "1Y":
            cutoff_ts = now_ts() - 31536000
        filtered = [h for h in history if (h.get("ts") or 0) >= cutoff_ts] if cutoff_ts else history
        series = [
            {"timestamp": time.strftime("%Y-%m-%d %H:%M", time.localtime(h.get("ts", 0))), "value": float(h.get("total_usd") or 0), "pnl_pct": 0}
            for h in filtered
        ]
        if not series and total_usd > 0:
            series = [{"timestamp": time.strftime("%Y-%m-%d %H:%M", time.localtime()), "value": total_usd, "pnl_pct": 0}]
        pos_usd = float(snap.get("positions_usd") or 0)
        free = float(snap.get("free_usd") or 0)
        allocation = [
            {"name": "Crypto", "value": pos_usd * 0.6},
            {"name": "Stocks", "value": pos_usd * 0.4},
            {"name": "Cash", "value": free},
        ]
        return _json({"ok": True, "series": series, "allocation": allocation, "timeframe": timeframe})
    except Exception as e:
        logger.exception("portfolio/performance: %s", e)
        return _json({"ok": True, "series": [], "allocation": [], "timeframe": timeframe, "error": str(e)})


@app.get("/api/tax_optimization_suggestions")
def api_tax_optimization_suggestions(min_loss_pct: float = 5.0):
    """Tax-loss harvesting suggestions for positions with unrealized loss >= min_loss_pct."""
    try:
        from tax_optimizer import tax_harvest_suggestions, ENABLE_TAX_HARVESTING
    except ImportError:
        return _json({"ok": False, "error": "Tax optimizer not available", "suggestions": []})
    if not ENABLE_TAX_HARVESTING:
        return _json({"ok": True, "enabled": False, "suggestions": [], "message": "Tax harvesting disabled"})
    positions = []
    if bm:
        try:
            for bot in list_bots():
                od = latest_open_deal(int(bot.get("id") or 0))
                if od and od.get("state") == "OPEN":
                    sym = bot.get("symbol") or od.get("symbol")
                    entry = float(od.get("entry_avg") or 0)
                    if sym and entry > 0:
                        price = _ticker_cached(sym, ttl_sec=60)
                        cur = float(price.get("last", 0) or price.get("c", 0) or 0) if price else 0
                        if cur > 0:
                            positions.append({
                                "symbol": sym,
                                "entry_price": entry,
                                "current_price": cur,
                                "avg_entry": entry,
                                "last_price": cur,
                            })
        except Exception as e:
            logger.warning("tax_optimization positions fetch: %s", e)
    suggestions = tax_harvest_suggestions(positions, min_loss_pct=float(min_loss_pct))
    return _json({"ok": True, "enabled": True, "suggestions": suggestions})


@app.get("/api/portfolio/rebalance_suggestions")
def api_rebalance_suggestions():
    """Portfolio rebalancing suggestions based on TARGET_ALLOCATIONS vs current sector allocation."""
    try:
        from sector_rotation import get_rotation_suggestions
        from stock_metadata import get_sector
    except ImportError:
        return _json({"ok": False, "error": "Sector rotation not available", "suggestions": []})
    sector_alloc = {}
    if bm:
        try:
            for bot in list_bots():
                od = latest_open_deal(int(bot.get("id") or 0))
                if od and bot.get("symbol"):
                    sym = bot.get("symbol")
                    sector = get_sector(sym)
                    if sector:
                        sector_alloc[sector] = sector_alloc.get(sector, 0.0) + 1.0
        except Exception as e:
            logger.warning("rebalance sector alloc: %s", e)
    suggestions = get_rotation_suggestions(sector_alloc)
    return _json({"ok": True, "suggestions": suggestions, "current_allocations": sector_alloc})


# =========================================================
# API: bots list/detail (DB only)
# =========================================================
@app.get("/api/bots/{bot_id}")
def api_bot(bot_id: int):
    b = get_bot(int(bot_id))
    if not b:
        return _json({"ok": False, "error": "Bot not found"}, 404)
    return _json({"ok": True, "bot": b})


# =========================================================
# API: bot runtime control (start/stop defined later with full snap response)
# =========================================================
@app.delete("/api/bots/{bot_id}")
def api_bots_delete(bot_id: int):
    bid = int(bot_id)
    b = get_bot(bid)
    if not b:
        return _json({"ok": False, "error": "Bot not found"}, 404)
    if bm:
        try:
            bm.stop(bid)
        except Exception as e:
            logger.warning("api_bots_delete: stop failed for bot %s: %s", bid, e)
    try:
        _discord_notify(f"🗑️ {b.get('name') or bid} deleted.")
    except Exception:
        pass
    try:
        delete_bot(bid)
        return _json({"ok": True, "message": "Bot deleted"})
    except Exception as e:
        logger.exception("api_bots_delete: delete_bot failed for bot %s", bid)
        return _json({"ok": False, "error": f"Delete failed: {e}"}, 500)


@app.post("/api/bots/{bot_id}/clone")
def api_bots_clone(bot_id: int):
    """Create a copy of a bot with (Copy) suffix in name."""
    bid = int(bot_id)
    original = get_bot(bid)
    if not original:
        return _json({"ok": False, "error": "Bot not found"}, 404)

    try:
        # Create a copy of the bot config
        copy_data = dict(original)
        # Remove ID so it gets a new one
        copy_data.pop("id", None)
        # Append (Copy) to name
        original_name = copy_data.get("name", "Bot")
        copy_data["name"] = f"{original_name} (Copy)"
        # Ensure it starts disabled
        copy_data["enabled"] = 0

        # Create the new bot
        new_bot_id = create_bot(copy_data)
        new_bot = get_bot(new_bot_id)

        try:
            _discord_notify(f"✅ Bot '{original_name}' cloned as '{copy_data['name']}'")
        except Exception:
            pass

        return _json({"ok": True, "bot": new_bot, "message": f"Bot cloned with ID {new_bot_id}"})
    except Exception as e:
        logger.exception("api_bots_clone: clone failed for bot %s", bid)
        return _json({"ok": False, "error": f"Clone failed: {e}"}, 500)


@app.get("/api/bots/{bot_id}/dealstats")
def api_bot_dealstats(bot_id: int):
    b = get_bot(int(bot_id))
    if not b:
        return _json({"ok": False, "error": "Bot not found"}, 404)
    try:
        stats = bot_deal_stats(int(bot_id))
        od = latest_open_deal(int(bot_id))
        return _json({"ok": True, "stats": stats, "open_deal": od})
    except Exception as e:
        return _json({"ok": False, "error": f"{type(e).__name__}: {e}", "stats": {}}, 500)


@app.get("/api/bots/{bot_id}/pnl_series")
def api_bot_pnl_series(bot_id: int, limit: int = 500):
    b = get_bot(int(bot_id))
    if not b:
        return _json({"ok": False, "error": "Bot not found"}, 404)
    try:
        series = bot_pnl_series(int(bot_id), limit=int(max(10, min(5000, int(limit)))))
        return _json({"ok": True, "series": series})
    except Exception as e:
        return _json({"ok": False, "error": f"{type(e).__name__}: {e}", "series": []}, 500)


@app.get("/api/bots/{bot_id}/metrics")
def api_bot_metrics(bot_id: int, limit: int = 500):
    b = get_bot(int(bot_id))
    if not b:
        return _json({"ok": False, "error": "Bot not found"}, 404)
    try:
        perf = bot_performance_stats(int(bot_id))
        dd = bot_drawdown_series(int(bot_id), limit=int(max(10, min(5000, int(limit)))))
        return _json({"ok": True, "perf": perf, "drawdown_series": dd})
    except Exception as e:
        return _json({"ok": False, "error": f"{type(e).__name__}: {e}", "perf": {}, "drawdown_series": []}, 500)


@app.get("/api/bots/{bot_id}/performance")
def api_bot_performance(bot_id: int):
    """Comprehensive bot performance: total P&L, win rate, avg duration, deal count."""
    b = get_bot(int(bot_id))
    if not b:
        return _json({"ok": False, "error": "Bot not found"}, 404)
    try:
        perf = bot_performance_stats(int(bot_id))
        stats = bot_deal_stats(int(bot_id))
        snap = None
        if bm:
            try:
                snap = bm.snapshot(int(bot_id))
            except Exception:
                pass
        unrealized = float(snap.get("unrealized_pnl", 0)) if snap else 0
        realized = float(stats.get("realized_total", 0)) if stats else 0
        return _json({
            "ok": True,
            "total_pnl": round(realized + unrealized, 2),
            "realized_pnl": round(realized, 2),
            "unrealized_pnl": round(unrealized, 2),
            "deals_closed": perf.get("total", 0),
            "wins": perf.get("wins", 0),
            "losses": perf.get("losses", 0),
            "win_rate": round(perf.get("win_rate", 0), 1),
            "avg_duration_sec": perf.get("avg_duration_sec", 0),
            "avg_profit_pct": round(perf.get("avg_profit_pct", 0), 2),
        })
    except Exception as e:
        return _json({"ok": False, "error": str(e)}, 500)


@app.get("/api/pause")
def api_pause_state():
    return _json({"ok": True, "paused": bool(_pause_state())})


@app.post("/api/pause")
async def api_pause_set(request: Request):
    payload = await request.json()
    paused = bool(payload.get("paused"))
    pause_hours_raw = payload.get("pause_hours", os.getenv("DEFAULT_GLOBAL_PAUSE_HOURS", "6"))
    try:
        pause_hours = max(0.0, float(pause_hours_raw))
    except Exception:
        pause_hours = 6.0
    try:
        if paused:
            set_setting("global_pause", "1")
            if pause_hours > 0:
                until_ts = int(time.time()) + int(pause_hours * 3600)
                set_setting("global_pause_until", str(until_ts))
            else:
                # Explicit indefinite pause when pause_hours=0.
                set_setting("global_pause_until", "0")
        else:
            set_setting("global_pause", "0")
            set_setting("global_pause_until", "0")
    except Exception:
        pass
    return _json({
        "ok": True,
        "paused": bool(_pause_state()),
        "pause_hours": pause_hours,
        "global_pause_until": get_setting("global_pause_until", "0"),
    })


@app.get("/api/risk/kill")
def api_kill_state():
    return _json({"ok": True, "kill_switch": bool(_kill_switch_state())})


@app.post("/api/risk/kill")
async def api_kill_set(request: Request):
    payload = await request.json()
    enabled = bool(payload.get("enabled"))
    try:
        set_setting("kill_switch", "1" if enabled else "0")
        # Remember that the user has toggled the kill switch at least once -
        # powers the "Kill switch tested" item on the Safety checklist.
        set_setting("kill_switch_tested", "1")
    except Exception:
        pass
    return _json({"ok": True, "kill_switch": bool(_kill_switch_state())})


def _fetch_status_last_price(b: Dict[str, Any]) -> Any:
    """Best-effort last price for status row; may call exchange APIs."""
    symbol = (b.get("symbol") or "").strip()
    if not symbol:
        return None
    try:
        market_type = classify_symbol(symbol)
        if market_type == "stock":
            client = alpaca_live if alpaca_live else alpaca_paper
            if client:
                ticker = client.get_ticker(symbol)
                return ticker.get("last") if ticker else None
        ticker = _ticker_cached(symbol, ttl_sec=60) or {}
        return ticker.get("last") or ticker.get("close")
    except Exception:
        return None


@app.get("/api/bots/{bot_id}/status")
def api_bot_status(bot_id: int):
    b = get_bot(int(bot_id))
    if not b:
        return _json({"ok": False, "error": "Bot not found"}, 404)
    if bm is None:
        # Degraded mode: return minimal status so UI shows bot info instead of "Worker not initialized"
        snap = {
            "running": False,
            "last_event": "Worker not initialized. Check Kraken/Alpaca API keys and restart the service.",
            "last_price": None,
            "avg_entry": None,
            "base_pos": None,
        }
        return _json({
            "ok": True,
            "bot": b,
            "snap": snap,
            "regime": None,
            "kraken_ready": _kraken_ready(),
            "kraken_error": KRAKEN_ERROR,
            "paused": bool(_pause_state()),
            "data_health": None,
            "worker_degraded": True,
        })

    try:
        snap = bm.snapshot(int(bot_id))
    except Exception as e:
        logger.warning("api_bot_status snapshot failed bot_id=%s: %s", bot_id, e)
        # Return valid response so UI shows error instead of "Status unavailable"
        snap = {
            "running": False,
            "last_event": f"Error: {str(e)[:80]}",
            "last_price": None,
            "avg_entry": None,
            "base_pos": None,
        }

    if snap.get("running") and not snap.get("last_event"):
        snap["last_event"] = "Running."
    if not snap.get("running") and not snap.get("last_event"):
        snap["last_event"] = "Stopped."
    
    # Get last price if not in snapshot (bounded wait so one slow exchange call cannot stall the worker)
    if snap.get("last_price") is None:
        holder: List[Any] = [None]

        def _run_lp():
            try:
                holder[0] = _fetch_status_last_price(b)
            except Exception:
                holder[0] = None

        th = threading.Thread(target=_run_lp, daemon=True)
        th.start()
        th.join(timeout=6.0)
        if holder[0] is not None:
            snap["last_price"] = holder[0]

    try:
        regime = latest_regime(int(bot_id))
    except Exception:
        regime = None

    data_health = None
    try:
        router = getattr(bm, "_md_router", None)
        if router:
            symbol = b.get("symbol", "")
            mt = b.get("market_type", "crypto")
            data_health = router.get_data_health(symbol, mt, required_tfs=["1h", "4h", "1d"], min_candles=20)
    except Exception as e:
        logger.debug("data_health fetch failed: %s", e)

    last_decisions = []
    try:
        from db import list_logs as _list_bot_logs
        logs = _list_bot_logs(int(bot_id), limit=10)
        for lg in (logs or []):
            msg = lg.get("message") or ""
            if any(k in msg for k in ("STRATEGY:", "Decision:", "INTELLIGENCE:", "Risk blocked:", "Order placed")):
                last_decisions.append({"ts": lg.get("ts"), "level": lg.get("level"), "message": msg})
                if len(last_decisions) >= 5:
                    break
    except Exception:
        pass

    # Compute a human-readable display_state so the Bot row can replace the
    # indefinite "Checking..." with a concrete status (IDLE / WAITING_FOR_FILL /
    # FILLED / MANAGING / CLOSING / ERROR / PAUSED / COOLDOWN / STOPPED).
    display_state = None
    try:
        from services.bot_display import compute_display_state
        current_deal: Dict[str, Any] = {}
        try:
            deals = list_deals(int(bot_id), limit=1) or []
            if deals:
                current_deal = deals[0] or {}
        except Exception:
            current_deal = {}
        display_state = compute_display_state(snap, current_deal)
    except Exception:
        display_state = None

    # Live-value enrichment: realized_pnl + position_pct_of_capital
    try:
        stats = bot_deal_stats(int(bot_id)) or {}
        snap["realized_pnl"] = float(stats.get("realized_total") or 0.0)
    except Exception:
        snap["realized_pnl"] = 0.0
    try:
        lp = float(snap.get("last_price") or 0.0)
        bp = float(snap.get("base_pos") or 0.0)
        position_quote = lp * bp if lp and bp else 0.0
        # Use cached portfolio snapshot if available; otherwise leave None.
        total_usd = 0.0
        try:
            from worker_api import _PORTFOLIO_CACHE as _PC
            cached = _PC.get("result") or {}
            pobj = cached.get("portfolio") or {}
            total_usd = float(pobj.get("total_usd") or 0.0)
        except Exception:
            total_usd = 0.0
        snap["position_pct_of_capital"] = round((position_quote / total_usd) * 100.0, 2) if total_usd > 0 else None
    except Exception:
        snap["position_pct_of_capital"] = None

    try:
        if b.get("last_poll_error"):
            snap["last_poll_error"] = str(b.get("last_poll_error") or "")[:2000]
        _lpa = b.get("last_polled_at")
        if _lpa is not None and str(_lpa).strip():
            try:
                snap["last_polled_at_ts"] = int(_lpa)
            except (TypeError, ValueError):
                pass
    except Exception:
        pass

    return _json({
        "ok": True,
        "bot": b,
        "snap": snap,
        "display_state": display_state,
        "regime": regime,
        "kraken_ready": _kraken_ready(),
        "kraken_error": KRAKEN_ERROR,
        "paused": bool(_pause_state()),
        "data_health": data_health,
        "last_decisions": last_decisions,
    })


@app.get("/api/positions/{bot_id}")
def api_positions_bot(bot_id: int):
    """Get positions for a specific bot (production-ready)."""
    b = get_bot(int(bot_id))
    if not b:
        return _json({"ok": False, "error": "Bot not found"}, 404)
    snap = {}
    if bm:
        try:
            snap = bm.snapshot(int(bot_id)) or {}
        except Exception:
            pass
    symbol = b.get("symbol", "")
    last_price = float(snap.get("last_price") or 0)
    if last_price <= 0 and symbol:
        try:
            if classify_symbol(symbol) == "stock" and (alpaca_live or alpaca_paper):
                client = alpaca_live or alpaca_paper
                t = client.get_ticker(symbol)
                last_price = float(t.get("last") or 0)
            else:
                tc = _ticker_cached(symbol, ttl_sec=60)
                if tc:
                    last_price = float(tc.get("last") or tc.get("c") or 0)
        except Exception:
            pass
    avg_entry = float(snap.get("avg_entry") or 0)
    base_pos = float(snap.get("base_pos") or 0)
    if avg_entry <= 0 and base_pos > 0 and last_price > 0:
        avg_entry = last_price
    position_value = base_pos * last_price if last_price > 0 else base_pos
    unrealized_pnl = 0.0
    unrealized_pnl_pct = 0.0
    if avg_entry > 0 and base_pos > 0:
        unrealized_pnl = (last_price - avg_entry) * base_pos
        unrealized_pnl_pct = ((last_price - avg_entry) / avg_entry) * 100
    pos = {
        "bot_id": int(bot_id),
        "symbol": symbol,
        "strategy": b.get("strategy_mode", "classic"),
        "avg_entry_price": avg_entry,
        "current_price": last_price,
        "position_value": position_value,
        "quantity": base_pos,
        "unrealized_pnl": unrealized_pnl,
        "unrealized_pnl_pct": unrealized_pnl_pct,
        "take_profit_price": snap.get("tp_price"),
        "stop_loss_price": None,
    }
    return _json({"ok": True, "positions": [pos] if base_pos > 0 else []})


@app.get("/api/bots/{bot_id}/regime")
def api_bot_regime(bot_id: int):
    b = get_bot(int(bot_id))
    if not b:
        return _json({"ok": False, "error": "Bot not found"}, 404)
    return _json({"ok": True, "regime": latest_regime(int(bot_id))})


@app.get("/api/bots/{bot_id}/recommendation")
def api_bot_recommendation(bot_id: int):
    b = get_bot(int(bot_id))
    if not b:
        return _json({"ok": False, "error": "Bot not found"}, 404)
    b = get_bot(int(bot_id))
    if not b:
        return _json({"ok": False, "error": "Bot not found"}, 404)

    try:
        client, is_kraken = _get_bot_client(b)
        if not client:
             return _json({"ok": False, "error": "Trading client not available"}, 503)
        if is_kraken and not _kraken_ready():
             return _json({"ok": False, "error": KRAKEN_ERROR or "Kraken not ready"}, 503)

        symbol = _resolve_symbol(b.get("symbol", ""))
        
        # Validation
        if is_kraken:
            mk = _markets()
            if mk and symbol not in mk:
                return _json({"ok": False, "error": f"Symbol not found on Kraken: {symbol}"}, 400)
    except Exception as e:
        return _json({"ok": False, "error": str(e)}, 500)

    try:
        # Use cached OHLCV from BotManager if available
        if bm:
            candles = bm.ohlcv_cached(symbol, "15m", limit=300)
        else:
            # Fallback if worker not running
            candles = client.fetch_ohlcv(symbol, timeframe="15m", limit=200)
            
        regime = detect_regime(candles)
        target, switched, reason = select_strategy(
            regime=regime,
            current="smart_dca",
            last_switch_ts=0,
            now_ts=now_ts(),
            forced=None,
            vol_ratio=regime.vol_ratio,
        )
        note = ""
        mode = str(b.get("strategy_mode") or "").lower()
        forced = str(b.get("forced_strategy") or "").lower()
        if forced:
            note = f"Bot is forcing '{forced}'. Recommendation ignores forced."
        elif mode and mode not in ("auto", "router"):
            note = f"Bot strategy mode is '{mode}'. Recommendation ignores manual mode."

        return _json(
            {
                "ok": True,
                "symbol": symbol,
                "regime": {"label": regime.regime, "confidence": regime.confidence, "why": regime.why},
                "recommended": target,
                "reason": reason,
                "note": note,
            }
        )
    except Exception as e:
        return _json({"ok": False, "error": f"{type(e).__name__}: {e}"}, 500)


@app.get("/api/bots/{bot_id}/decisions")
def api_bot_decisions(bot_id: int, limit: int = 100):
    b = get_bot(int(bot_id))
    if not b:
        return _json({"ok": False, "error": "Bot not found"}, 404)
    decisions = list_strategy_decisions(int(bot_id), limit=int(max(1, min(int(limit), 500))))
    return _json({"ok": True, "decisions": decisions})


@app.get("/api/strategies/leaderboard")
def api_strategies_leaderboard(window: Optional[int] = None, window_days: int = 90):
    """Strategy leaderboard: live journal + backtest / inline fallback (see services.leaderboard)."""
    wd = int(window) if window is not None else int(window_days)
    wd = int(min(365, max(7, wd)))
    rows = get_strategy_leaderboard(window_days=wd)
    return _json({"ok": True, "strategies": rows, "window_days": wd})


@app.get("/api/recommendations/scan_status")
def api_recommendations_scan_status():
    """Return current scan progress for short/medium/long horizons. Used during Rescan."""
    with _globals_lock:
        short_state = (_RECO_STATE.get("short") or {}).copy()
        medium_state = (_RECO_STATE.get("medium") or {}).copy()
        long_state = (_RECO_STATE.get("long") or {}).copy()
    return _json({
        "ok": True,
        "short": {"scanned": short_state.get("scanned", 0), "total": short_state.get("total", 0), "scanning": short_state.get("scanning", False)},
        "medium": {"scanned": medium_state.get("scanned", 0), "total": medium_state.get("total", 0), "scanning": medium_state.get("scanning", False)},
        "long": {"scanned": long_state.get("scanned", 0), "total": long_state.get("total", 0), "scanning": long_state.get("scanning", False)},
    })


@app.get("/api/recommendations/scan_progress")
def api_recommendations_scan_progress():
    """Rich progress data for the live status panel: per-horizon state, current symbol, errors, history, ETA."""
    now_t = time.time()
    # Skip cache while any horizon is actively scanning (progress updates are live)
    _any_active = any(_HORIZON_SCANNING.get(h) for h in ("short", "medium", "long"))
    if not _any_active:
        with _globals_lock:
            _sp_cached = _SCAN_PROGRESS_CACHE.get("result")
            _sp_ts = _SCAN_PROGRESS_CACHE.get("ts", 0.0)
        if _sp_cached is not None and (now_t - _sp_ts) < _SCAN_PROGRESS_CACHE_TTL:
            return _json(_sp_cached)
    _SCAN_STALE_THRESHOLD = 25 * 60  # 25 minutes: force-reset a stuck scanning=True
    with _globals_lock:
        prog = {k: v for k, v in _SCAN_PROGRESS.items()}
        horizons = {}
        any_scanning = False
        for h in ("short", "medium", "long"):
            st = (_RECO_STATE.get(h) or {}).copy()
            scanning = bool(st.get("scanning"))
            # Watchdog: if scanning=True but started more than 25 min ago, force-reset the lock
            if scanning:
                started_at = st.get("started_at") or 0
                if started_at and (now_t - started_at) > _SCAN_STALE_THRESHOLD:
                    logger.warning(
                        "[SCAN-WATCHDOG] %s scan stuck scanning=True for %.0f min — force-resetting",
                        h, (now_t - started_at) / 60,
                    )
                    _RECO_STATE[h] = {
                        **st,
                        "scanning": False,
                        "error": f"Watchdog reset after {int((now_t - started_at) / 60)}min stall",
                        "last_run_ts": int(now_t),
                    }
                    scanning = False
            if scanning:
                any_scanning = True
            last_ts = st.get("last_run_ts", 0)
            horizons[h] = {
                "scanning": scanning,
                "scanned": st.get("scanned", 0),
                "total": st.get("total", 0),
                "eligible": st.get("eligible", 0),
                "error": st.get("error", ""),
                "last_run_ts": last_ts,
                "last_run_ago_sec": int(now_t - last_ts) if last_ts else None,
            }
    with _RECO_SCAN_ACTIVE_LOCK:
        scan_running = _RECO_SCAN_ACTIVE or any_scanning
    eta_sec = None
    if any_scanning:
        cur_h = prog.get("current_horizon", "")
        st = horizons.get(cur_h, {})
        done = st.get("scanned", 0)
        total = st.get("total", 0)
        elapsed = now_t - prog.get("scan_start_ts", now_t)
        if done > 0 and total > done and elapsed > 0:
            rate = done / elapsed
            eta_sec = int((total - done) / rate)
    universe_size = {"equity": 0, "crypto": 0, "total": 0}
    try:
        from universe_builder import get_universe_stats, get_universe_age_seconds
        _ust = get_universe_stats()
        universe_size["equity"] = (_ust.get("equity") or {}).get("total", 0)
        universe_size["crypto"] = (_ust.get("crypto") or {}).get("total", 0)
        universe_size["total"] = universe_size["equity"] + universe_size["crypto"]
        universe_size["age_sec"] = get_universe_age_seconds()
    except Exception:
        pass

    _sp_result = {
        "ok": True,
        "any_scanning": any_scanning,
        "scan_running": scan_running,
        "current_symbol": prog.get("current_symbol", ""),
        "current_horizon": prog.get("current_horizon", ""),
        "buy_signals_found": prog.get("buy_signals_found", 0),
        "eta_sec": eta_sec,
        "horizons": horizons,
        "recent_errors": (prog.get("recent_errors") or [])[-5:],
        "scan_history": (prog.get("scan_history") or [])[-5:],
        "server_ts": int(now_t),
        "universe_size": universe_size,
    }
    if not _any_active:
        with _globals_lock:
            _SCAN_PROGRESS_CACHE["result"] = _sp_result
            _SCAN_PROGRESS_CACHE["ts"] = now_t
    return _json(_sp_result)


@app.get("/api/explore/scan_status")
def api_explore_scan_status():
    """Per-horizon scan progress for Explore UI (alias-style payload; see also /api/recommendations/scan_progress).

    progress_pct semantics:
      - status == "scanning"      → live percent (scanned/total * 100), 0 if total unknown
      - status == "idle" w/ run   → 100 (the last scan completed; bar should read "done", not stuck)
      - status == "idle" no run   → 0   (never run; bar should read empty)

    This keeps the bar in lockstep with the frontend's idle/scanning/never-run buckets even
    when scanned/total were never cleared back to 0 after the previous scan finished.
    """
    try:
        from explore_composite_scorer import HORIZON_CONFIGS
    except Exception:
        HORIZON_CONFIGS = {}
    now_t = time.time()
    now_i = int(now_t)
    with _globals_lock:
        prog = dict(_SCAN_PROGRESS)
        body: Dict[str, Any] = {"ok": True}
        for h in ("short", "medium", "long"):
            st = (_RECO_STATE.get(h) or {}).copy()
            scanning = bool(st.get("scanning"))
            last_ts = int(st.get("last_run_ts") or 0)
            scanned = int(st.get("scanned") or 0)
            total = int(st.get("total") or 0)
            interval_sec = int(HORIZON_CONFIGS.get(h, {}).get("scan_interval_minutes", 60)) * 60
            if scanning:
                pct = int(min(100, scanned * 100 / max(1, total))) if total else 0
            elif last_ts > 0:
                pct = 100
            else:
                pct = 0
            cur_sym = ""
            if scanning and str(prog.get("current_horizon") or "") == h:
                cur_sym = str(prog.get("current_symbol") or "")
            eta_sec = None
            if scanning and scanned > 0 and total > scanned:
                elapsed = max(0.001, now_t - float(prog.get("scan_start_ts", now_t)))
                rate = scanned / elapsed
                if rate > 0:
                    eta_sec = int((total - scanned) / rate)
            entry: Dict[str, Any] = {
                "status": "scanning" if scanning else "idle",
                "current_symbol": cur_sym,
                "progress_pct": pct,
                "batch_current": 0,
                "batch_total": 0,
                "last_completed": last_ts,
                "eta_seconds": eta_sec,
            }
            if not scanning and last_ts > 0:
                entry["next_run_seconds"] = max(0, last_ts + interval_sec - now_i)
            elif not scanning:
                entry["next_run_seconds"] = None
            body[h] = entry
        return _json(body)


@app.get("/api/market/regime")
def api_market_regime():
    """Latest persisted regime snapshots for key symbols."""
    want = ["BTC/USD", "ETH/USD", "SPY"]
    rows = get_latest_regime_for_symbols(want + ["XBT/USD"])
    out: Dict[str, Any] = {}
    for label, keys in (("BTC/USD", ("BTC/USD", "XBT/USD")), ("ETH/USD", ("ETH/USD",)), ("SPY", ("SPY",))):
        picked = None
        for k in keys:
            if k in rows and rows[k]:
                picked = rows[k]
                break
        if picked:
            out[label] = picked
    return _json({"ok": True, "symbols": out, "timestamp": now_ts()})


@app.get("/api/explore/signal_accuracy")
def api_explore_signal_accuracy():
    return _json({"ok": True, "baselines": list_signal_accuracy_baselines()})


@app.get("/api/portfolio/equity_curve")
def api_portfolio_equity_curve(days: int = 30):
    d = max(1, min(int(days), 730))
    return _json({"ok": True, "days": d, "points": list_portfolio_equity_curve(d)})


@app.get("/api/deals")
def api_deals_list(status: Optional[str] = None, limit: int = 50):
    """Cross-bot deals for dashboards; includes realized_pnl_pct when present."""
    lim = max(1, min(int(limit), 500))
    st = (status or "").strip().upper() or None
    if st and st not in ("OPEN", "CLOSED"):
        st = None
    deals = list_all_deals(state=st, limit=lim)
    return _json({"ok": True, "deals": deals, "count": len(deals)})


@app.get("/api/universe/stats")
def api_universe_stats():
    """Return universe builder statistics."""
    try:
        from universe_builder import get_universe_stats, get_universe_age_seconds, get_equity_universe, get_crypto_universe
        stats = get_universe_stats()
        ages = get_universe_age_seconds()
        eq = get_equity_universe()
        cr = get_crypto_universe()
        return _json({
            "ok": True,
            "equity_count": len(eq),
            "crypto_count": len(cr),
            "total": len(eq) + len(cr),
            "stats": stats,
            "age_seconds": ages,
            "equity_sample": eq[:20],
            "crypto_sample": cr[:20],
        })
    except Exception as e:
        return _json({"ok": False, "error": str(e)})


@app.post("/api/universe/rebuild")
def api_universe_rebuild():
    """Force rebuild the scanning universe."""
    try:
        from universe_builder import get_full_universe, get_universe_stats
        equities, crypto = get_full_universe(force_rebuild=True)
        return _json({
            "ok": True,
            "equity_count": len(equities),
            "crypto_count": len(crypto),
            "total": len(equities) + len(crypto),
            "stats": get_universe_stats(),
        })
    except Exception as e:
        return _json({"ok": False, "error": str(e)})


@app.get("/api/recommendations/diagnose")
def api_recommendations_diagnose(symbols: str = "XBT/USD,ETH/USD,SOL/USD,XRP/USD,ADA/USD", horizon: str = "short"):
    """Diagnostic: scan specific symbols and show raw scores, filter decisions, and why they pass/fail."""
    sym_list = [s.strip() for s in symbols.split(",") if s.strip()][:10]
    btc_ctx = _btc_context()
    buy_thresh_crypto = _reco_buy_threshold_crypto()
    buy_thresh_stocks = _reco_buy_threshold_stocks()
    watch_thresh = _reco_watch_threshold()
    results = []
    for sym in sym_list:
        entry = {"symbol": sym, "raw_score": 0, "final_score": 0, "eligible": False,
                 "signal": "wait", "filters_passed": [], "filters_failed": [], "reasons": [],
                 "risk_flags": [], "regime": "", "error": None}
        try:
            snap = _scan_symbol(sym, horizon, btc_ctx)
            if not snap:
                entry["error"] = "No data returned"
                results.append(entry)
                continue
            score = float(snap.get("score") or 0)
            entry["raw_score"] = score
            entry["final_score"] = score
            entry["eligible"] = bool(snap.get("eligible", False))
            entry["reasons"] = snap.get("reasons") or []
            entry["risk_flags"] = snap.get("risk_flags") or []
            regime = snap.get("regime") or {}
            entry["regime"] = str(regime.get("regime") or regime.get("label") or regime)
            metrics = snap.get("metrics") or {}
            entry["allowed_action"] = metrics.get("allowed_action", "")
            is_stock = len(sym) < 6 and "/" not in sym
            buy_t = buy_thresh_stocks if is_stock else buy_thresh_crypto
            entry["buy_threshold"] = buy_t
            entry["watch_threshold"] = watch_thresh
            entry["signal"] = "buy" if score >= buy_t else ("watch" if score >= watch_thresh else "wait")
            if snap.get("data_ok") is False:
                entry["filters_failed"].append("DATA: data_ok=False")
            else:
                entry["filters_passed"].append("DATA: OK")
            rflags = snap.get("risk_flags") or []
            if "btc_risk_off" in rflags:
                entry["filters_failed"].append("MACRO: btc_risk_off (score -20)")
            else:
                entry["filters_passed"].append("MACRO: no risk-off penalty")
            if any("EXPLORE_V2_GATE" in str(f) for f in rflags):
                gate_msg = [f for f in rflags if "EXPLORE_V2_GATE" in str(f)]
                entry["filters_failed"].append(f"EXPLORE_V2: {gate_msg}")
            else:
                entry["filters_passed"].append("EXPLORE_V2: passed gates")
            vol = metrics.get("volume_24h_quote")
            if vol is not None and vol < RECO_MIN_VOLUME_24H:
                entry["filters_failed"].append(f"VOLUME: {vol:.0f} < {RECO_MIN_VOLUME_24H:.0f}")
            else:
                entry["filters_passed"].append(f"VOLUME: {vol}")
            if score < buy_t:
                entry["filters_failed"].append(f"SCORE: {score:.1f} < buy_threshold {buy_t}")
            else:
                entry["filters_passed"].append(f"SCORE: {score:.1f} >= buy_threshold {buy_t}")
        except Exception as e:
            entry["error"] = f"{type(e).__name__}: {str(e)[:200]}"
        results.append(entry)
    return _json({
        "ok": True,
        "horizon": horizon,
        "btc_context": btc_ctx,
        "thresholds": {
            "buy_crypto": buy_thresh_crypto,
            "buy_stocks": buy_thresh_stocks,
            "watch": watch_thresh,
            "reco_min_volume_24h": RECO_MIN_VOLUME_24H,
            "reco_min_market_cap": RECO_MIN_MARKET_CAP,
            "explore_v2_enabled": os.getenv("EXPLORE_V2", "1"),
        },
        "results": results,
    })


def _active_symbol_set() -> Tuple[set, Dict[str, List[str]]]:
    """Build set of normalized symbols that are already active (enabled bot, running bot, open deal). Returns (symbols_set, symbol -> [reasons])."""
    active = set()
    reasons: Dict[str, List[str]] = {}
    try:
        for b in list_bots():
            if int(b.get("enabled", 0)) != 1:
                continue
            sym = str(b.get("symbol") or "").strip()
            if not sym:
                continue
            norm = _normalize_symbol(sym)
            active.add(norm)
            reasons.setdefault(norm, []).append("enabled_bot")
            if int(b.get("last_running", 0)) == 1 and "running_bot" not in (reasons.get(norm) or []):
                reasons.setdefault(norm, []).append("running_bot")
        for d in list_all_deals(state="OPEN", limit=500):
            sym = str(d.get("symbol") or "").strip()
            if not sym:
                continue
            norm = _normalize_symbol(sym)
            active.add(norm)
            if "open_deal" not in (reasons.get(norm) or []):
                reasons.setdefault(norm, []).append("open_deal")
    except Exception as e:
        logger.debug("_active_symbol_set: %s", e)
    return active, reasons


# ─── Scanner readiness enrichment for Explore items ──────────────────────────
_SCANNER_CACHE: Dict[str, Tuple[float, Dict[str, Any]]] = {}
_SCANNER_CACHE_TTL = 300  # 5 minutes

def _get_scanner_fields_for_item(symbol: str, market_type: str, metrics: Dict[str, Any]) -> Dict[str, Any]:
    """
    Return scanner-derived fields for a recommendation item.
    Uses a lightweight cache to avoid re-computing on every API call.
    Non-blocking: returns empty defaults if scanner unavailable.
    """
    defaults = {
        "ready_now": None,
        "entry_type_scanner": None,
        "edge_score": None,
        "expected_move_pct": None,
        "invalidation_level": None,
        "target_levels": None,
        "time_horizon_scanner": None,
        "evidence": None,
        "trigger_conditions": None,
    }

    now = time.time()
    cached = _SCANNER_CACHE.get(symbol)
    if cached:
        ts, fields = cached
        if (now - ts) < _SCANNER_CACHE_TTL:
            return fields

    try:
        from db import get_watchlist_entry
        wl = get_watchlist_entry(symbol, status="watching")
        if wl:
            setup = json.loads(wl.get("setup_json") or "{}")
            fields = {
                "ready_now": setup.get("ready_now", False),
                "entry_type_scanner": setup.get("entry_type"),
                "edge_score": setup.get("edge_score"),
                "expected_move_pct": setup.get("expected_move_pct"),
                "invalidation_level": setup.get("invalidation_level"),
                "target_levels": setup.get("target_levels"),
                "time_horizon_scanner": setup.get("time_horizon"),
                "evidence": setup.get("evidence"),
                "trigger_conditions": setup.get("trigger_conditions"),
            }
            _SCANNER_CACHE[symbol] = (now, fields)
            return fields
    except Exception:
        pass

    return defaults


_RECO_RESULT_CACHE: Dict[str, Tuple[float, Any]] = {}
_RECO_RESULT_CACHE_TTL = 60  # 60 seconds — invalidated on new scan

_EXPLORE_FEED_CACHE: Dict[str, Tuple[float, Any]] = {}
_EXPLORE_FEED_CACHE_TTL = 2400  # 40-minute TTL — covers full 30-min scan interval with margin


@app.get("/api/explore/feed")
def api_explore_feed(
    horizon: str = "short",
    market_type: str = "all",
    signal: str = "all",
    limit: int = 80,
    show_already_active: int = 0,
):
    """
    Explore tab: read-only from explore_signals (buy/watch).

    When EXPLORE_SMART_RANK is on (default), fetches a wider candidate set, re-ranks by
    conviction + recency + 24h action + 90d backtest quality, and diversifies strategies.
    """
    _h = str(horizon).lower().strip()
    h = "long" if _h.startswith("l") else ("medium" if _h.startswith("m") else "short")
    lim = max(1, min(int(limit), 200))

    _ef_cache_key = f"{h}|{market_type}|{signal}|{show_already_active}"
    _ef_cached = _EXPLORE_FEED_CACHE.get(_ef_cache_key)
    if _ef_cached:
        _ef_ts, _ef_result = _ef_cached
        if (time.time() - _ef_ts) < _EXPLORE_FEED_CACHE_TTL:
            _cached_copy = dict(_ef_result)
            _cached_copy["items"] = _cached_copy.get("items", [])[:lim]
            _cached_copy["count"] = len(_cached_copy["items"])
            _cached_copy.setdefault("data_source", "live")
            # Refresh the scan timestamps from live state so the
            # freshness badge ("Screener scan (X): Nm ago") stays in lockstep with
            # the /api/explore/scan_status badge — both are sourced from _RECO_STATE.
            _now_i_live = int(time.time())
            with _globals_lock:
                _short_st_live = (_RECO_STATE.get("short") or {}).copy()
                _med_st_live = (_RECO_STATE.get("medium") or {}).copy()
                _long_st_live = (_RECO_STATE.get("long") or {}).copy()
            _live_last_scan_by_horizon = {
                "short": {
                    "ts": _short_st_live.get("last_run_ts"),
                    "age_sec": (_now_i_live - int(_short_st_live.get("last_run_ts") or 0)) if _short_st_live.get("last_run_ts") else None,
                },
                "medium": {
                    "ts": _med_st_live.get("last_run_ts"),
                    "age_sec": (_now_i_live - int(_med_st_live.get("last_run_ts") or 0)) if _med_st_live.get("last_run_ts") else None,
                },
                "long": {
                    "ts": _long_st_live.get("last_run_ts"),
                    "age_sec": (_now_i_live - int(_long_st_live.get("last_run_ts") or 0)) if _long_st_live.get("last_run_ts") else None,
                },
            }
            _cached_copy["last_scan_by_horizon"] = _live_last_scan_by_horizon
            _hor_live = _live_last_scan_by_horizon.get(h) or {}
            _cached_copy["last_scan_ts"] = _hor_live.get("ts")
            _cached_copy["scan_age_sec"] = _hor_live.get("age_sec")
            return _json(_cached_copy)

    sig_f = str(signal or "all").lower().strip()
    if sig_f == "buy":
        statuses = ["buy"]
    elif sig_f == "watch_only":
        statuses = ["watch"]
    elif sig_f in ("buy_watch", "watch"):
        # UI historically used value="watch" for the label "Buy + Watch" — that wrongly fetched watch-only rows.
        # buy_watch = both; watch kept for backward-compatible API clients that meant "both".
        statuses = ["buy", "watch"]
    else:
        statuses = ["buy", "watch"]

    from explore_signals import STRATEGY_LABELS

    bt_row = get_latest_explore_backtest(h)
    bt_90: Dict[str, Any] = {}
    bt_60: Dict[str, Any] = {}
    bt_30: Dict[str, Any] = {}
    if bt_row and isinstance(bt_row.get("results"), dict):
        _bt_windows = bt_row["results"].get("windows") or {}
        bt_90 = _bt_windows.get("90d") or {}
        bt_60 = _bt_windows.get("60d") or {}
        bt_30 = _bt_windows.get("30d") or {}

    fetch_mult = max(2, min(12, int(os.getenv("EXPLORE_FEED_FETCH_MULT", "6"))))
    fetch_cap = min(500, max(lim * fetch_mult, lim + 80))
    rows = list_explore_feed(h, market_type=market_type, statuses=statuses, limit=fetch_cap)
    feed_data_source = "live"
    if not rows:
        try:
            rows = _reco_rows_as_explore_feed_rows(h, market_type, statuses, fetch_cap)
            if rows:
                feed_data_source = "cached"
        except Exception as _exfb:
            logger.debug("explore feed fallback from recommendations: %s", _exfb)
            rows = []

    now_i = int(time.time())
    use_smart = os.getenv("EXPLORE_SMART_RANK", "1").strip().lower() not in ("0", "false", "no", "off")
    pick_scores: Dict[str, float] = {}
    live_perf: Dict[str, Any] = {}
    if use_smart and rows and os.getenv("EXPLORE_LIVE_STRAT_RANK", "1").strip().lower() not in (
        "0",
        "false",
        "no",
        "off",
    ):
        try:
            live_perf = _get_cached_strategy_win_rates(h, lookback_days=90)
        except Exception:
            live_perf = {}
    if use_smart and rows:
        rows, pick_scores = _explore_apply_smart_rank(rows, bt_90, lim, now_i, live_perf=live_perf or None)
    else:
        rows = rows[:lim]
        pick_scores = {}

    active_symbols, active_reason = _active_symbol_set()
    with _globals_lock:
        short_state = (_RECO_STATE.get("short") or {}).copy()
        medium_state = (_RECO_STATE.get("medium") or {}).copy()
        long_state = (_RECO_STATE.get("long") or {}).copy()
    last_scan_by_horizon = {
        "short": {"ts": short_state.get("last_run_ts"), "age_sec": now_i - short_state.get("last_run_ts", 0) if short_state.get("last_run_ts") else None},
        "medium": {"ts": medium_state.get("last_run_ts"), "age_sec": now_i - medium_state.get("last_run_ts", 0) if medium_state.get("last_run_ts") else None},
        "long": {"ts": long_state.get("last_run_ts"), "age_sec": now_i - long_state.get("last_run_ts", 0) if long_state.get("last_run_ts") else None},
    }

    # Composite scorer integration: enrich each signal with multi-layer analysis
    try:
        from explore_composite_scorer import (
            strategy_quality_gate,
            macro_environment_score,
            assign_grade,
            fix_200ma_display,
            filter_correlated_signals,
            apply_safety_filters,
            generate_composite_summary,
            technical_score as compute_technical_score,
            buy_threshold_from_fear_greed,
            compute_regime_adjusted_threshold,
            regime_strategy_bonus,
            SECTOR_MAP,
            BEAR_STRATEGY_IDS,
            position_size_pct_for_grade,
        )
        _composite_available = True
    except Exception:
        _composite_available = False

    _fg_val = int(_FEAR_GREED_CACHE.get("value") or 50)
    _btc_ctx_feed = _btc_context()

    try:
        _live_closed_deals = count_closed_deals()
    except Exception:
        _live_closed_deals = 0
    # Weighted win-rate: recent 30d matters most (50%), 60d secondary (30%), 90d base (20%)
    # A strategy qualifies as "active" only if BOTH conditions hold:
    #   (a) 90d average return ≥ 0  (no money-losing strategies, even if recent WR is high), AND
    #   (b) weighted_win_rate ≥ 38%  OR  30d win_rate ≥ 50%  (passes a quality bar).
    # This keeps the backend "active" flag in sync with the UI verdict — a strategy with a
    # negative 90d avg return is "⛔ DO NOT USE" in the backtest table, so it must not be
    # marked active here either.
    _active_strategies = 0
    _total_strategies = 0
    _pending_strategies = 0
    _strategy_health: List[Dict[str, Any]] = []
    _all_strat_keys = set(bt_90) | set(bt_60) | set(bt_30)
    # Bug 11: minimum closed-trade count required to consider a strategy mature.
    # Below this we mark the strategy as "pending" (collecting data) instead of
    # "inactive" so new dry-run setups don't look broken to users.
    _PENDING_TRADES_THRESHOLD = 10
    for _sk in sorted(_all_strat_keys):
        _sv90 = bt_90.get(_sk) or {}
        _sv60 = bt_60.get(_sk) or {}
        _sv30 = bt_30.get(_sk) or {}
        if not any(isinstance(x, dict) for x in (_sv90, _sv60, _sv30)):
            continue
        _total_strategies += 1
        _wr90 = float(_sv90.get("win_rate") or 0)
        _wr60 = float(_sv60.get("win_rate") or 0)
        _wr30 = float(_sv30.get("win_rate") or 0)
        _ar90 = float(_sv90.get("avg_return") or 0)
        _trades_90 = int(_sv90.get("trades") or _sv90.get("total_trades") or _sv90.get("count") or 0)
        _trades_30 = int(_sv30.get("trades") or _sv30.get("total_trades") or _sv30.get("count") or 0)
        _w_wr = _wr30 * 0.5 + _wr60 * 0.3 + _wr90 * 0.2
        _passes_wr = (_w_wr >= 38) or (_wr30 >= 50)
        _active = (_ar90 >= 0) and _passes_wr
        # Bug 11: a strategy is *pending* (not inactive) when there isn't enough
        # closed-trade history yet to judge it. Treat synthetic backtest data
        # as legitimate pre-warm: trades_90 reflects backtest sample size.
        _pending = (not _active) and (_trades_90 < _PENDING_TRADES_THRESHOLD) and (_live_closed_deals < _PENDING_TRADES_THRESHOLD)
        if _active:
            _active_strategies += 1
        if _pending:
            _pending_strategies += 1
        _lab = STRATEGY_LABELS.get(_sk, _sk.replace("_", " ").title())
        if _active:
            _status = "active"
            _st_reason = "Active: 90d avg return ≥ 0% and (weighted WR ≥ 38% or 30d WR ≥ 50%)."
        elif _pending:
            _status = "pending"
            _needed = max(0, _PENDING_TRADES_THRESHOLD - max(_trades_90, _live_closed_deals))
            _st_reason = (
                f"Pending: collecting closed-trade data ({_trades_90} backtest trades, "
                f"{_live_closed_deals} live closed deals). Needs {_needed} more to qualify."
            )
        elif _ar90 < 0:
            _status = "inactive"
            _st_reason = (
                f"Inactive: negative 90d average return ({_ar90:.2f}%). "
                "Never marked active even if recent WR is high — protects against bleeding strategies."
            )
        elif not _passes_wr:
            _status = "inactive"
            _st_reason = (
                f"Inactive: weighted win rate {_w_wr:.0f}% (need ≥38%) and 30d WR {_wr30:.0f}% (need ≥50%)."
            )
        else:
            _status = "inactive"
            _st_reason = "Inactive: does not meet Strategy Health thresholds."
        _strategy_health.append(
            {
                "id": _sk,
                "label": _lab,
                "active": bool(_active),
                "status": _status,
                "pending_progress": {
                    "trades_90d": _trades_90,
                    "trades_30d": _trades_30,
                    "live_closed_deals": _live_closed_deals,
                    "threshold": _PENDING_TRADES_THRESHOLD,
                },
                "reason": _st_reason,
                "weighted_win_rate": round(_w_wr, 1),
                "win_rate_90d": round(_wr90, 1),
                "win_rate_30d": round(_wr30, 1),
                "avg_return_90d": round(_ar90, 2),
            }
        )

    items = []
    for r in rows:
        sym = str(r.get("symbol") or "")
        norm_sym = _normalize_symbol(sym)
        already_active = norm_sym in active_symbols
        if not show_already_active and already_active:
            continue
        sid = str(r.get("strategy") or "")
        detail = _explore_row_detail_json(r)
        pat_id = str(detail.get("strategy_id") or "").strip()
        evs = detail.get("evaluate_signal") if isinstance(detail.get("evaluate_signal"), dict) else {}
        ds = str(evs.get("detected_strategy") or "").strip()
        pat_label = STRATEGY_LABELS.get(pat_id, pat_id) if pat_id else ""
        strat_display = pat_label or ds or (STRATEGY_LABELS.get(sid, sid) if sid else "—")
        if strat_display == "—" or not str(strat_display).strip():
            strat_display = sid or "—"
        pr = str(r.get("reason") or "").strip()
        hr = str(evs.get("strategy_reason") or "").strip()
        if pr and hr and hr not in pr and pr not in hr:
            reason_out = f"{pr} · Scanner: {hr}"
        elif pr:
            reason_out = pr
        else:
            reason_out = hr
        bt_info = _explore_bt_lookup(bt_90, detail, r)
        wr90 = float(bt_info["win_rate"]) if bt_info and bt_info.get("win_rate") is not None else None
        conv = float(r.get("conviction_score") or 0)
        st = str(r.get("status") or "watch")
        pick_sc = float(pick_scores.get(sym, conv)) if pick_scores else float(conv)

        is_crypto_item = "/" in sym or str(r.get("market_type") or "").lower() == "crypto"
        asset_type_str = "crypto" if is_crypto_item else "stock"

        # Hard classification rule — "/" means crypto, always
        _item_mt = "crypto" if "/" in sym else "stocks"
        if market_type == "stocks" and _item_mt != "stocks":
            continue
        if market_type == "crypto" and _item_mt != "crypto":
            continue

        # --- Composite scoring overlay ---
        composite_grade = ""
        composite_score_val = None
        composite_signal = st
        composite_summary = ""
        composite_rr = None
        composite_rr_display = "—"
        composite_rr_color = "gray"
        macro_label = "neutral"
        strategy_gate_warn = False
        safety_flags_out = []
        composite_reasons = []
        composite_flags = []
        pct_above_200_val = None
        chg24 = None
        try:
            chg24 = float(r.get("change_24h")) if r.get("change_24h") is not None else None
        except (TypeError, ValueError):
            pass

        # Sparkline: initialise before composite block so it's always defined
        _sparkline: List[float] = []

        # Populate sparkline from cached OHLCV regardless of composite availability
        try:
            _spark_candles: list = []
            if is_crypto_item and "/" in sym:
                _spark_sym = _resolve_symbol(sym)
                _spark_candles = _ohlcv_cached(_spark_sym, "1d", 10, 1800)
            else:
                # Stock: scan writes key "{sym}|1d|500" to _RECO_OHLCV_CACHE; also check
                # short-form key written by the fallback fetch below
                _spark_key = f"{sym}|1d|500"
                _spark_fb_key = f"{sym}|spark|1d"
                with _globals_lock:
                    _spark_cached = _RECO_OHLCV_CACHE.get(_spark_key) or _RECO_OHLCV_CACHE.get(_spark_fb_key)
                if _spark_cached and _spark_cached.get("data"):
                    _spark_candles = _spark_cached["data"]
                elif alpaca_live or alpaca_paper:
                    # Scan hasn't run yet — fetch directly from Alpaca and cache for 30 min
                    try:
                        _alpaca_cl = alpaca_live if alpaca_live else alpaca_paper
                        _fetched = _alpaca_cl.get_ohlcv(sym, timeframe="1d", limit=10) or []
                        if _fetched:
                            _spark_candles = _fetched
                            with _globals_lock:
                                _RECO_OHLCV_CACHE[_spark_fb_key] = {"ts": time.time(), "data": _fetched}
                    except Exception:
                        pass
            if _spark_candles:
                _spark_closes = [float(c[4]) for c in _spark_candles if isinstance(c, (list, tuple)) and len(c) > 4]
                if len(_spark_closes) >= 3:
                    _sparkline = _spark_closes[-7:]
        except Exception:
            pass

        if _composite_available and (bt_90 or bt_30):
            try:
                _strat_key = pat_id or sid or ""
                # Use weighted backtest data for the quality gate:
                # merge 30d/60d/90d win rates so the gate benefits from recent performance
                _bt_for_gate = {}
                for _btsrc in (bt_90, bt_60, bt_30):
                    for _k, _v in (_btsrc or {}).items():
                        if _k not in _bt_for_gate and isinstance(_v, dict):
                            _bt_for_gate[_k] = _v
                # Prefer 30d data if it has ≥10 signals (more recent)
                if bt_30.get(_strat_key, {}).get("signals", 0) >= 10:
                    _bt_for_gate[_strat_key] = bt_30[_strat_key]
                elif bt_60.get(_strat_key, {}).get("signals", 0) >= 10:
                    _bt_for_gate[_strat_key] = bt_60[_strat_key]
                _gate = strategy_quality_gate(_strat_key, _bt_for_gate)
                strategy_gate_warn = _gate.get("warn", False)

                _btc_labels = _btc_ctx_feed.get("labels") or {}
                _btc_regime = str(
                    _btc_ctx_feed.get("regime") or _btc_ctx_feed.get("regime_label")
                    or _btc_labels.get("1d") or _btc_labels.get("4h") or ""
                ).upper()
                _btc_dt = float(_btc_ctx_feed.get("downtrend_score") or _btc_ctx_feed.get("dt_score") or _btc_ctx_feed.get("btc_down") or 0)
                _btc_hv_val = float(_btc_ctx_feed.get("hv") or _btc_ctx_feed.get("btc_hv") or 0)
                _macro = macro_environment_score(_fg_val, _btc_regime, _btc_dt, _btc_hv_val, asset_type_str)
                macro_label = _macro.get("macro_label", "neutral")
                composite_flags = _macro.get("flags", [])

                _item_price = float(r.get("price") or 0)
                _safety = apply_safety_filters(sym, chg24, None, None, None, [], asset_type_str, current_price=_item_price)
                safety_flags_out = _safety.get("flags", [])

                if not is_crypto_item:
                    _earnings_days = None
                    try:
                        _earnings_days = detail.get("earnings_days")
                        if _earnings_days is None:
                            from earnings_calendar import days_until_earnings
                            _earnings_days = days_until_earnings(sym)
                    except Exception:
                        pass
                    if _earnings_days is not None:
                        try:
                            _ed_int = int(_earnings_days)
                            if 0 <= _ed_int <= 3:
                                safety_flags_out.append(
                                    f"⚠️ Earnings in {_ed_int} days — "
                                    f"high volatility risk"
                                )
                            elif 4 <= _ed_int <= 7:
                                safety_flags_out.append(
                                    f"📅 Earnings in {_ed_int} days — "
                                    f"potential catalyst"
                                )
                        except (TypeError, ValueError):
                            pass

                # Try to get OHLCV data for technical scoring & R:R
                _candles_1d = []
                try:
                    if is_crypto_item and "/" in sym:
                        _resolved_sym = _resolve_symbol(sym)
                        _candles_1d = _ohlcv_cached(_resolved_sym, "1d", 200, 1800)
                    elif not is_crypto_item:
                        _cached_key = f"{sym}|1d|500"
                        with _globals_lock:
                            _ck = _RECO_OHLCV_CACHE.get(_cached_key)
                        if _ck and _ck.get("data"):
                            _candles_1d = _ck["data"]
                        if not _candles_1d:
                            try:
                                import yfinance as yf
                                def _yf_candles_feed(s: str, interval: str, period: str) -> list:
                                    try:
                                        t = yf.Ticker(s)
                                        hist = t.history(period=period, interval=interval)
                                        if hist is None or hist.empty:
                                            return []
                                        out = []
                                        for ts_idx, row in hist.iterrows():
                                            try:
                                                out.append([int(ts_idx.timestamp()), float(row["Open"]), float(row["High"]), float(row["Low"]), float(row["Close"]), float(row["Volume"])])
                                            except Exception:
                                                continue
                                        return out
                                    except Exception:
                                        return []
                                _candles_1d = _yf_candles_feed(sym, "1d", "1y") if "/" not in sym else []
                            except Exception:
                                pass
                except Exception as _ohlcv_err:
                    logger.debug("Feed OHLCV fetch failed for %s: %s", sym, _ohlcv_err)
                _closes = [float(c[4]) for c in _candles_1d if isinstance(c, (list, tuple)) and len(c) > 4]
                _highs = [float(c[2]) for c in _candles_1d if isinstance(c, (list, tuple)) and len(c) > 2]
                _lows = [float(c[3]) for c in _candles_1d if isinstance(c, (list, tuple)) and len(c) > 3]
                _vols = [float(c[5]) for c in _candles_1d if isinstance(c, (list, tuple)) and len(c) > 5]
                # Populate 7-day sparkline from daily closes
                if len(_closes) >= 3:
                    _sparkline = _closes[-7:]

                _tech = {"score": conv, "risk_reward_ratio": None, "trend_direction": "neutral",
                         "rsi": None, "volume_ratio": None, "pct_above_200": None}
                if len(_closes) >= 50:
                    try:
                        _tech = compute_technical_score(_closes, _highs, _lows, _vols, _strat_key)
                        pct_above_200_val = _tech.get("pct_above_200")
                        _rr = _tech.get("risk_reward_ratio")
                        if _rr is not None:
                            composite_rr = _rr
                            composite_rr_display = f"{_rr:.1f}:1"
                            composite_rr_color = "green" if _rr >= 2 else ("yellow" if _rr >= 1.5 else "red")
                    except Exception:
                        pass

                composite_grade = assign_grade(conv, _gate, _macro, _tech, _safety, strategy_id=_strat_key)

                composite_score_val = conv
                if _gate["max_score_cap"] < conv and _gate["max_score_cap"] > 0:
                    composite_score_val = _gate["max_score_cap"]
                _penalty = _safety.get("score_penalty", 0)
                if _penalty > 0:
                    composite_score_val = max(0, composite_score_val - _penalty)
                composite_score_val = max(0, min(100, composite_score_val + _macro.get("score_adjustment", 0)))
                # Regime-aware per-strategy bonus — replaces the flat -4 RANGE penalty
                try:
                    _regime_bonus = regime_strategy_bonus(_btc_regime, _strat_key)
                    if _regime_bonus != 0:
                        composite_score_val = max(0, min(100, float(composite_score_val) + _regime_bonus))
                except Exception:
                    pass
                _tc_bear = _btc_regime in ("BEAR", "STRONG_BEAR", "DOWNTREND")
                if _strat_key == "trend_continuation" and _fg_val > 40 and not _tc_bear:
                    composite_score_val = min(100.0, float(composite_score_val) + 10.0)

                _buy_cut = compute_regime_adjusted_threshold(_fg_val, asset_type_str, _btc_regime)
                if _strat_key == "trend_continuation" and _fg_val > 40:
                    _buy_cut = max(55, _buy_cut - 5)
                if composite_score_val < _buy_cut and st == "buy":
                    composite_signal = "watch"
                if not _gate["passed"] and st == "buy":
                    composite_signal = "watch"
                _bear_exempt_strategies = {
                    "relative_strength_bear", "oversold_extreme_fear",
                    "volume_capitulation", "oversold_bounce",
                }
                _macro_block = _macro.get("block_buy") and _strat_key not in _bear_exempt_strategies
                if _macro_block and st == "buy":
                    composite_signal = "wait"
                if _safety.get("block_buy") and st == "buy":
                    composite_signal = "wait"
                _rrg = _tech.get("risk_reward_ratio")
                if _rrg is not None and _rrg < 1.5 and composite_signal == "buy":
                    composite_signal = "wait"
                    composite_reasons_pre = [
                        f"Risk:Reward {_rrg:.1f}:1 below minimum 1.5:1 — not actionable as BUY",
                    ]
                else:
                    composite_reasons_pre = []

                composite_reasons = composite_reasons_pre + [_gate["reason"]] + _macro.get("flags", []) + _safety.get("flags", [])
                if _tech.get("reasons"):
                    composite_reasons.extend(_tech["reasons"][:3])
            except Exception as _comp_err:
                logger.debug("Composite scoring failed for %s: %s", sym, _comp_err)

        # Fix 200MA display
        if _composite_available and reason_out:
            try:
                reason_out = fix_200ma_display(pct_above_200_val, reason_out)
            except Exception:
                pass

        bt_warn = bool(bt_info.get("warn")) if isinstance(bt_info, dict) else False
        effective_signal = composite_signal if _composite_available else st
        if bt_warn and effective_signal == "buy" and strategy_gate_warn:
            effective_signal = "watch"

        # Backtest-aware rating
        _BEAR_STRAT_IDS = {
            "relative_strength_bear",
            "oversold_extreme_fear",
            "oversold_bounce",
            "volume_capitulation",
        }
        _is_bear_strat = str(pat_id or sid or "").lower() in _BEAR_STRAT_IDS
        if _is_bear_strat and conv > 0:
            _final_score = conv
        else:
            _final_score = composite_score_val if composite_score_val is not None else conv
        if effective_signal == "buy" and _final_score >= 75 and not strategy_gate_warn:
            rating = "Strong Buy"
        elif effective_signal == "buy":
            rating = "Buy"
        elif effective_signal == "watch":
            rating = "Watch"
        else:
            rating = "Wait"

        # Generate composite summary after all scoring is done
        if _composite_available and composite_score_val is not None:
            try:
                _summary_item = {
                    "explore_strategy": strat_display,
                    "explore_strategy_id": pat_id or sid,
                    "conviction_grade": composite_grade,
                    "macro_label": macro_label,
                    "backtest_win_rate_90d": wr90,
                    "signal": effective_signal,
                    "composite_score": composite_score_val,
                    "score": _final_score,
                    "safety_flags": safety_flags_out,
                    "strategy_gate_warn": strategy_gate_warn,
                }
                composite_summary = generate_composite_summary(_summary_item)
            except Exception:
                pass

        _bt_sigs = int(float((bt_info or {}).get("signals") or 0))
        _bt_avg_ret = float((bt_info or {}).get("avg_return") or 0.0)
        _strategy_evidence_line = ""
        if wr90 is not None and _bt_sigs > 0:
            _strategy_evidence_line = (
                f"Based on {strat_display} which has won {wr90:.0f}% of {_bt_sigs} backtested trades "
                f"with avg return of {_bt_avg_ret:+.1f}%."
            )
        elif wr90 is not None:
            _strategy_evidence_line = (
                f"{strat_display} — {wr90:.0f}% 90d aggregate pattern win rate (see backtest panel for sample size)."
            )

        _facts_d = detail.get("facts") if isinstance(detail.get("facts"), dict) else {}
        _vol_ratio_item = detail.get("volume_ratio")
        if _vol_ratio_item is None and _facts_d:
            _vol_ratio_item = (
                _facts_d.get("volume_ratio")
                if _facts_d.get("volume_ratio") is not None
                else _facts_d.get("volume_mult")
            )
        try:
            _vol_ratio_item = float(_vol_ratio_item) if _vol_ratio_item is not None else None
        except (TypeError, ValueError):
            _vol_ratio_item = None
        try:
            _avg_vol_d = float(detail["avg_volume_20d"]) if detail.get("avg_volume_20d") is not None else None
        except (TypeError, ValueError):
            _avg_vol_d = None
        try:
            _vol_now_d = float(
                detail.get("vol_24h")
                or detail.get("volume_24h")
                or detail.get("volume")
                or r.get("volume_24h")
                or r.get("volume")
                or 0
            )
        except (TypeError, ValueError):
            _vol_now_d = 0.0
        _vol_now_d = _vol_now_d or None
        if (
            (_vol_ratio_item is None or _vol_ratio_item <= 0)
            and _avg_vol_d
            and _avg_vol_d > 0
            and _vol_now_d
            and _vol_now_d > 0
        ):
            _vol_ratio_item = float(_vol_now_d) / float(_avg_vol_d)

        items.append(
            {
                "symbol": sym,
                "explore_feed": True,
                "score": _final_score,
                "conviction_score": conv,
                "composite_score": composite_score_val,
                "pick_score": round(pick_sc, 2),
                "smart_rank_applied": bool(use_smart and pick_scores),
                "signal": effective_signal,
                "horizon": h,
                "market_type": "crypto" if "/" in sym else "stocks",
                "price": r.get("price"),
                "volume": float(
                    detail.get("vol_24h") or detail.get("volume_24h") or detail.get("volume")
                    or r.get("volume_24h") or r.get("volume") or 0
                ) or None,
                "volume_ratio": _vol_ratio_item,
                "avg_volume": _avg_vol_d,
                "change_pct": chg24,
                "explore_strategy": strat_display,
                "explore_strategy_id": pat_id or sid,
                "scanner_strategy": ds or None,
                "scanner_strategy_reason": hr or None,
                "strategy_reason": reason_out,
                "explore_reason": reason_out,
                "backtest_win_rate_90d": wr90,
                "backtest_warn_90d": bt_warn,
                "backtest_aggregate_note": (
                    "Aggregate 90d win rate for this chart pattern across the backtest universe — not this symbol's live P&L or a profit guarantee."
                    if wr90 is not None
                    else None
                ),
                "detail": detail,
                "signal_ts": r.get("signal_ts"),
                "updated_ts": r.get("updated_ts"),
                "rating": rating,
                "sparkline": _sparkline,
                "already_active": already_active,
                "active_reason": list(active_reason.get(norm_sym) or []),
                "factor_scores": (detail.get("facts") if isinstance(detail, dict) else None) or {},
                "conviction_grade": composite_grade,
                "macro_label": macro_label,
                "strategy_gate_warn": strategy_gate_warn,
                "safety_flags": safety_flags_out,
                "composite_reasons": composite_reasons,
                "composite_flags": composite_flags,
                "composite_summary": composite_summary,
                "risk_reward_display": composite_rr_display,
                "risk_reward_color": composite_rr_color,
                "strategy_evidence_line": _strategy_evidence_line,
                "live_unproven_badge": bool(effective_signal == "buy" and _live_closed_deals <= 0),
                "sector": SECTOR_MAP.get(sym, "") if _composite_available else "",
                "suggested_position_pct": (
                    position_size_pct_for_grade(
                        composite_grade, h,
                        is_bear_strategy=(pat_id or sid or "") in BEAR_STRATEGY_IDS,
                    ) if _composite_available and composite_grade else None
                ),
            }
        )

    # Apply cross-signal correlation filter
    if _composite_available and items:
        try:
            items = filter_correlated_signals(items)
        except Exception as _filt_err:
            logger.debug("Cross-signal filter failed: %s", _filt_err)

    try:
        from stock_universe import JUNK_TICKERS
        def _sym_ok(it):
            s = (it.get("symbol") or "").upper()
            base = s.split("/")[0] if "/" in s else s
            if base in JUNK_TICKERS:
                return False
            if "/" in s and base in INVALID_KRAKEN_BASES:
                return False
            return True
        items = [it for it in items if _sym_ok(it)]
    except Exception:
        pass

    # DCA entry plan for buy signals
    def _build_dca_plan(
        symbol: str, price: float,
        suggested_position_pct: float,
        portfolio_total: float,
        horizon: str,
    ) -> Dict[str, Any]:
        total_usd = (suggested_position_pct / 100.0) * portfolio_total
        t1_pct, t2_pct, t3_pct = 0.40, 0.35, 0.25
        drop2, drop3 = 0.03, 0.07
        return {
            "total_usd": round(total_usd, 2),
            "tranche_1": {
                "usd": round(total_usd * t1_pct, 2),
                "trigger": "Buy now",
                "price": round(price, 4),
            },
            "tranche_2": {
                "usd": round(total_usd * t2_pct, 2),
                "trigger": f"Buy if drops {drop2*100:.0f}%",
                "price": round(price * (1 - drop2), 4),
            },
            "tranche_3": {
                "usd": round(total_usd * t3_pct, 2),
                "trigger": f"Buy if drops {drop3*100:.0f}%",
                "price": round(price * (1 - drop3), 4),
            },
            "strategy": "Bear market DCA — scale in on weakness",
        }

    _dca_portfolio = 100.0
    try:
        _bm_dca = globals().get("bm")
        if _bm_dca and hasattr(_bm_dca, "get_portfolio_total"):
            _dca_portfolio = float(_bm_dca.get_portfolio_total() or 100.0)
        if _dca_portfolio <= 0:
            _dca_portfolio = 100.0
    except Exception:
        pass

    for _it in items:
        if _it.get("signal") == "buy":
            _it_price = float(_it.get("price") or 0)
            _it_pct = float(_it.get("suggested_position_pct") or 3.0)
            if _it_price > 0:
                _it["dca_plan"] = _build_dca_plan(
                    _it["symbol"], _it_price, _it_pct,
                    _dca_portfolio, h,
                )

    explore_rejected_payload: List[Dict[str, Any]] = []
    try:
        _rej_raw = list_explore_rejected(h, limit=50)
        for _r in _rej_raw:
            _sid = str(_r.get("strategy") or "")
            explore_rejected_payload.append(
                {
                    **_r,
                    "strategy": STRATEGY_LABELS.get(_sid, _sid) if _sid else "",
                }
            )
    except Exception:
        pass

    with _globals_lock:
        state_h = (_RECO_STATE.get(h) or {}).copy()
    last_scan = (last_scan_by_horizon.get(h) or {}).get("ts") or 0
    scan_age = now_i - int(last_scan) if last_scan else 999999
    status = "ready"
    reason = "ok" if items else "no_matches"
    message = "Explore feed (pattern strategies)" if items else "No buy/watch rows yet — run Rescan or wait for the scanner."
    if not items and not last_scan and not state_h.get("scanning"):
        status = "warming_up"
        reason = "no_scan_yet"
        message = "Generating explore signals… first scan starting."
        import threading

        def _kick():
            try:
                _scan_recommendations(h)
            except Exception as _ke:
                logger.warning("explore feed auto-scan failed: %s", _ke)

        threading.Thread(target=_kick, daemon=True).start()

    _td = max(_total_strategies, 5)
    _buy_thr_fg = 65
    try:
        from explore_composite_scorer import buy_threshold_from_fear_greed as _btfg_disp

        _buy_thr_fg = int(_btfg_disp(_fg_val, "crypto"))
    except Exception:
        pass
    _btc_regime_label = ""
    try:
        _labels = _btc_ctx_feed.get("labels") or {}
        _btc_regime_label = str(
            _btc_ctx_feed.get("regime")
            or _btc_ctx_feed.get("regime_label")
            or _labels.get("1d")
            or _labels.get("4h")
            or ""
        ).upper()
    except Exception:
        pass
    # Bug 11: include pending count + a softer banner for early-stage setups.
    _banner = (
        f"⚠️ Only {_active_strategies} of {_td} strategies are currently treated as active "
        f"(≥40% win rate and non-losing avg return in the 90d backtest). "
    )
    if _pending_strategies > 0 and _live_closed_deals < 10:
        _banner += (
            f"{_pending_strategies} strategies are pending — they will activate automatically "
            f"once {max(0, 10 - _live_closed_deals)} more live trades close. "
        )
    _banner += "Recommendations rely on pattern signals with limited multi-strategy confirmation — use position sizing accordingly."
    _explore_disclaimer = {
        "active_strategies": _active_strategies,
        "pending_strategies": _pending_strategies,
        "total_strategies": _td,
        "completed_closed_deals": _live_closed_deals,
        "trades_needed_for_activation": max(0, 10 - _live_closed_deals),
        "fear_greed": _fg_val,
        "_btc_regime": _btc_regime_label,
        "buy_score_threshold": _buy_thr_fg,
        "strategy_health": _strategy_health,
        "banner": _banner,
    }

    def _build_market_summary(regime: str, fg: int, risk_off: bool) -> str:
        parts = []
        if regime in ("STRONG_BEAR", "BEAR", "TREND_DOWN"):
            parts.append("BTC in downtrend")
        elif regime in ("RANGING", "RANGE"):
            parts.append("BTC ranging sideways")
        elif regime in ("BULL", "STRONG_BULL", "BREAKOUT"):
            parts.append("BTC in uptrend")
        if fg <= 15:
            parts.append(
                f"extreme fear (F&G={fg}) — historically "
                f"near cycle bottoms, contrarian opportunity"
            )
        elif fg <= 30:
            parts.append(f"market fearful (F&G={fg})")
        elif fg >= 75:
            parts.append(f"market greedy (F&G={fg}) — caution")
        if risk_off:
            parts.append(
                "defensive mode active — only relative "
                "strength and oversold signals showing"
            )
        return (
            ". ".join(parts).capitalize() + "."
            if parts else "Neutral market conditions."
        )

    _horizons_for_ui: Dict[str, Any] = {}
    with _globals_lock:
        for _hh in ("short", "medium", "long"):
            _st_err = (_RECO_STATE.get(_hh) or {}).copy()
            _horizons_for_ui[_hh] = {"error": str(_st_err.get("error") or "")}

    _macro_risk_off = _btc_regime_label in ("BEAR", "STRONG_BEAR") and _fg_val <= 20
    _market_conditions = {
        "btc_regime": _btc_regime_label or "UNKNOWN",
        "fear_greed_value": _fg_val,
        "fear_greed_label": _FEAR_GREED_CACHE.get("label", "Neutral"),
        "fear_greed_contrarian": _fg_val <= 20,
        "macro_risk_off": _macro_risk_off,
        "active_strategies": _active_strategies,
        "total_strategies": _td,
        "summary": _build_market_summary(
            str(_btc_regime_label or ""),
            _fg_val,
            bool(_macro_risk_off),
        ),
    }

    _ef_response = {
        "ok": True,
        "status": status,
        "reason": reason,
        "message": message,
        "horizon": h,
        "items": items,
        "count": len(items),
        "has_more": False,
        "last_scan_ts": last_scan,
        "scan_age_sec": scan_age,
        "last_scan_by_horizon": last_scan_by_horizon,
        "explore_rejected": explore_rejected_payload,
        "cache_ts": now_i,
        "explore_smart_rank": use_smart,
        "explore_disclaimer": _explore_disclaimer,
        "market_conditions": _market_conditions,
        "horizons": _horizons_for_ui,
        "data_source": feed_data_source,
    }
    _EXPLORE_FEED_CACHE[_ef_cache_key] = (time.time(), _ef_response)
    if len(_EXPLORE_FEED_CACHE) > 30:
        _oldest = min(_EXPLORE_FEED_CACHE, key=lambda k: _EXPLORE_FEED_CACHE[k][0])
        _EXPLORE_FEED_CACHE.pop(_oldest, None)
    return _json(_ef_response)


@app.post("/api/explore/backtest")
def api_explore_backtest_run(horizon: str = "short"):
    """Run historical validation (sample universe) and persist results."""
    _h = str(horizon).lower().strip()
    h = "long" if _h.startswith("l") else ("medium" if _h.startswith("m") else "short")
    try:
        from explore_backtest import default_universe_symbols, run_explore_backtest

        def _fetch(sym: str):
            try:
                # Stocks: use alpaca or yfinance fallback
                if "/" not in sym:
                    try:
                        import yfinance as yf
                        t = yf.Ticker(sym)
                        hist = t.history(period="1y", interval="1d")
                        if hist is not None and not hist.empty:
                            candles = []
                            for ts, row in hist.iterrows():
                                candles.append([
                                    int(ts.timestamp()),
                                    float(row["Open"]),
                                    float(row["High"]),
                                    float(row["Low"]),
                                    float(row["Close"]),
                                    float(row["Volume"]),
                                ])
                            return candles
                    except Exception:
                        pass
                    return []
                else:
                    # Crypto: use kraken_client
                    if kc is None:
                        return []
                    try:
                        now_ms = int(time.time() * 1000)
                        since_ms = now_ms - (365 * 86400 * 1000)
                        raw = kc.fetch_ohlcv_range(sym, "1d", since_ms, now_ms)
                        return list(raw or [])
                    except Exception:
                        return []
            except Exception:
                return []

        stocks, crypto = default_universe_symbols()
        res = run_explore_backtest(
            fetch_candles=_fetch,
            stock_symbols=stocks,
            crypto_symbols=crypto,
            horizon=h,
        )
        save_explore_backtest_results(h, res)
        return _json({"ok": True, "result": res})
    except Exception as e:
        logger.exception("explore backtest failed: %s", e)
        return _json({"ok": False, "error": str(e)}, 500)


@app.post("/api/explore/backtest/refresh")
def api_explore_backtest_refresh():
    """Force refresh of explore strategy backtests."""
    try:
        import threading
        t = threading.Thread(
            target=_run_missing_backtests,
            kwargs={"force": True},
            daemon=True,
        )
        t.start()
        return _json({
            "ok": True,
            "message": "Backtest refresh started"
        })
    except Exception as e:
        return _json({"ok": False, "error": str(e)}, 500)


@app.get("/api/explore/backtest/latest")
def api_explore_backtest_latest(horizon: str = "short"):
    _h = str(horizon).lower().strip()
    h = "long" if _h.startswith("l") else ("medium" if _h.startswith("m") else "short")
    row = get_latest_explore_backtest(h)
    if not row:
        return _json({"ok": True, "horizon": h, "result": None})
    return _json({"ok": True, "horizon": h, "result": row})


@app.get("/api/explore/strategy-performance")
def api_explore_strategy_performance(horizon: str = "short", days: int = 90):
    """Per-strategy win rates from explore_signal_outcomes (not cached here — use for analytics)."""
    _h = str(horizon or "short").lower().strip()
    h = "long" if _h.startswith("l") else ("medium" if _h.startswith("m") else "short")
    d = max(7, min(int(days), 365))
    strat = get_strategy_win_rates(h, lookback_days=d)
    low_acc = [k for k, v in (strat or {}).items() if v.get("low_accuracy")]
    return _json({
        "ok": True,
        "horizon": h,
        "days": d,
        "strategies": strat or {},
        "low_accuracy_strategies": low_acc,
    })


def _compute_suggested_entry(breakdown: dict, price: float, signal: str) -> Optional[str]:
    """Compute suggested entry text based on S/R proximity."""
    if signal != "buy" or not price or price <= 0:
        return None
    try:
        sr = breakdown.get("support_resistance", {}) if isinstance(breakdown, dict) else {}
        sr_val = float(sr.get("value", 0)) if sr else 0
        if sr_val >= 3.0:
            return "Current price is near support — good entry."
        elif sr_val <= -3.0:
            pullback_target = round(price * 0.985, 4 if price < 10 else 2)
            return f"Consider waiting for pullback to ~${pullback_target}"
        else:
            return None
    except Exception:
        return None

def _stock_market_open() -> bool:
    """Returns True if US stock market is currently open."""
    try:
        import datetime as _dt
        now_et = _dt.datetime.now(_dt.timezone(_dt.timedelta(hours=-5)))
        if now_et.weekday() >= 5:
            return False
        market_open = now_et.replace(hour=9, minute=30, second=0, microsecond=0)
        market_close = now_et.replace(hour=16, minute=0, second=0, microsecond=0)
        return market_open <= now_et <= market_close
    except Exception:
        return True


def _conviction_grade(score: float) -> str:
    if score >= 80:
        return "A"
    if score >= 65:
        return "B"
    if score >= 50:
        return "C"
    return "D"


def _get_explore_feed_items(
    horizon: str = "short",
    limit: int = 20,
    signal_filter: str = "buy",
    market_type: str = "crypto",
) -> List[Dict[str, Any]]:
    """Read directly from explore_signals table (buy/watch only). Never falls back to recommendations."""
    from stock_universe import JUNK_TICKERS
    from explore_signals import STRATEGY_LABELS as _STRAT_LABELS

    try:
        from db import _conn as _db_conn
        con = _db_conn()
        cur = con.execute(
            """
            SELECT symbol, status, conviction_score, strategy, reason,
                   market_type, updated_ts, detail_json, price, change_24h
            FROM explore_signals
            WHERE status IN ('buy', 'watch')
              AND updated_ts > (strftime('%s','now') - 7200)
            ORDER BY conviction_score DESC, updated_ts DESC
            LIMIT ?
            """,
            (limit * 4,),
        )
        rows = cur.fetchall()
    except Exception as _db_err:
        logger.warning("_get_explore_feed_items: DB query failed: %s", _db_err)
        return []

    if not rows:
        return []

    _STRAT_LABEL_TO_ID = {v: k for k, v in _STRAT_LABELS.items()}
    items: List[Dict[str, Any]] = []
    for r in rows:
        sym = str(r["symbol"] or "").strip()
        if not sym:
            continue
        is_crypto = "/" in sym
        base = sym.split("/")[0].upper() if is_crypto else sym.upper()
        if base in JUNK_TICKERS:
            continue
        if is_crypto and base in INVALID_KRAKEN_BASES:
            continue

        item_market = str(r["market_type"] or "").strip().lower() or ("crypto" if is_crypto else "stocks")
        if market_type not in ("all", "") and item_market != market_type:
            continue

        conv = float(r["conviction_score"] or 0)
        st = str(r["status"] or "watch")
        if signal_filter == "buy" and st != "buy":
            continue

        strat_col = str(r["strategy"] or "").strip()
        detail_json_raw = r["detail_json"] or "{}"
        try:
            detail = json.loads(detail_json_raw) if isinstance(detail_json_raw, str) else (detail_json_raw or {})
        except (json.JSONDecodeError, TypeError):
            detail = {}
        strat_id = str(detail.get("strategy_id") or "").strip()
        if not strat_id:
            strat_id = _STRAT_LABEL_TO_ID.get(strat_col, strat_col)
        strat_label = strat_col or _STRAT_LABELS.get(strat_id, strat_id) or "smart_dca"

        reason = str(r["reason"] or "").strip()
        grade = _conviction_grade(conv)

        items.append({
            "symbol": sym,
            "score": conv,
            "signal": st,
            "horizon": horizon,
            "market_type": item_market,
            "price": r["price"],
            "change_pct": r["change_24h"],
            "volume": None,
            "market_cap": None,
            "rating": "Neutral",
            "confidence": conv / 100.0 if conv > 1 else conv,
            "sparkline": [],
            "regime_1d": None,
            "regime_4h": None,
            "regime_label": None,
            "weekly_trend": None,
            "strategy": strat_id or "smart_dca",
            "strategy_id": strat_id or "smart_dca",
            "strategy_mode": strat_label or "smart_dca",
            "suggested_strategy": strat_label or "smart_dca",
            "recommended_strategy": strat_label or "smart_dca",
            "explore_strategy": strat_label or "smart_dca",
            "explore_strategy_id": strat_id or "smart_dca",
            "strategy_reason": reason,
            "volatility": None,
            "risk_flags": [],
            "updated_ts": r["updated_ts"],
            "signal_age_sec": None,
            "eligible": True,
            "research_only": False,
            "reasons": [reason] if reason else [],
            "already_active": False,
            "active_reason": [],
            "conviction_grade": grade,
            "conviction_score": conv,
            "composite_score": float(detail.get("composite_score") or 0) if detail.get("composite_score") is not None else None,
            "explore_feed": True,
        })
    items.sort(key=lambda x: x["score"], reverse=True)
    return items[:limit]


@app.get("/api/recommendations")
def api_recommendations(
    horizon: str = "short",
    min_score: float = 0.0,
    include: str = "",
    exclude: str = "",
    quote: str = "",
    market_type: str = "crypto", # "crypto" or "stocks" or "all"
    include_all: int = 0,
    limit: int = 10,  # Default to 10 for fast loading
    signal: str = "buy",  # "buy" | "watch" | "all", default "buy"
    sort: str = "score",  # "score" | "profit_factor" | "drawdown" | "winrate", default "score"
    offset: int = 0,  # Pagination offset
    volatility: str = "all",  # "all" | "low" | "medium" | "high" - filter by volatility level
    regime: str = "all",  # "all" | "bull" | "breakout" | "range" | "bear" - filter by regime
    sector: str = "all",  # "all" | "Technology" | "Financial" | etc. - stocks only
    show_already_active: int = 0,  # 1 = include symbols that already have a bot/open deal (with badge)
):
    """
    Get market recommendations. Returns structured status for better UX.
    FAST: No network calls, uses cached DB data only. Prices filled async.
    Default: Shows top 10 buy signals, sorted by score.
    
    Volatility filter:
    - low: volatility < 0.02 (2%)
    - medium: 0.02 <= volatility < 0.05 (2-5%)
    - high: volatility >= 0.05 (5%+)
    
    Regime filter: BULL, BREAKOUT, RANGE, BEAR (from metrics.regime).
    """
    _h = str(horizon).lower().strip()
    h = "long" if _h.startswith("l") else ("medium" if _h.startswith("m") else "short")
    if market_type == "crypto" and not quote:
        quote = "USD"
    
    # Enforce reasonable limit to prevent slow responses
    limit = min(int(limit), 100)  # Cap at 100 for pagination support
    offset = max(0, int(offset))  # Ensure non-negative
    signal_filter = str(signal).lower()  # "buy", "watch", or "all"
    # Min Score: Any (0) = show everything. Else use horizon default when not explicitly set.
    _horizon_defaults = {"short": RECO_MIN_SCORE_SHORT, "medium": RECO_MIN_SCORE_MEDIUM, "long": RECO_MIN_SCORE_LONG}
    if min_score <= 0:
        min_score = -1  # Sentinel: ignore score filter (show everything)
    elif signal_filter == "buy" and min_score == 80:
        # UI default 80 - use horizon-specific for better results
        min_score = _horizon_defaults.get(h, 70)
    sort_by = str(sort).lower()  # "score", "profit_factor", "drawdown", "winrate"
    volatility_filter = str(volatility).lower()  # "all", "low", "medium", "high"
    regime_filter = str(regime).lower().strip()  # "all", "bull", "breakout", "range", "bear"
    sector_filter = str(sector).strip()  # "all" or sector name for stocks
    
    # Check client readiness
    kraken_ready = _kraken_ready()
    alpaca_ready = _alpaca_any_ready()

    # Early returns for specific market types
    if market_type == "crypto" and not kraken_ready:
        return _json({
            "ok": False,
            "status": "error",
            "reason": "kraken_not_ready",
            "message": KRAKEN_ERROR or "Kraken not ready",
            "items": []
        }, 503)

    if market_type == "stocks" and not alpaca_ready:
        # Retry Alpaca initialization if keys are present
        alpaca_ready = _retry_alpaca_init_if_keys_present()
        if not alpaca_ready:
            return _json({
                "ok": False,
                "status": "error",
                "reason": "alpaca_not_ready",
                "message": "Alpaca API not configured",
                "items": []
            }, 503)
    
    # For "all" market type, if neither is ready, return empty but don't block
    if market_type == "all" and not kraken_ready and not alpaca_ready:
        return _json({
            "ok": True,
            "status": "ready",
            "reason": "no_clients",
            "message": "No trading clients configured",
            "items": [],
            "count": 0
        })
    
    # Result cache: avoid re-processing on rapid page loads (2-min bucket refreshes stale-watch / age logic)
    _cache_tb = int(time.time()) // 120
    _cache_key = f"{h}|{market_type}|{signal_filter}|{sort_by}|{volatility_filter}|{regime_filter}|{sector_filter}|{limit}|{offset}|{min_score}|{_cache_tb}"
    _cached = _RECO_RESULT_CACHE.get(_cache_key)
    if _cached:
        _ts, _result = _cached
        if (time.time() - _ts) < _RECO_RESULT_CACHE_TTL:
            try:
                _er_fresh = list_explore_rejected(h, limit=50)
                _merged = dict(_result)
                _merged["explore_rejected"] = _er_fresh
            except Exception:
                _merged = dict(_result)
            return _json(_merged)

    # Fetch cached recommendations – over-fetch so Explore has enough per asset type.
    if market_type in ("crypto", "stocks"):
        fetch_limit = 1200
    else:
        fetch_limit = min(max(limit * 4, 200), 800)
    rows = []
    reco_data_source = "recommendations_latest"
    try:
        # Direct call - exclude blocklisted (STABLE, etc.) at source
        rows = list_recommendations(h, limit=fetch_limit, exclude_bases=list(CRYPTO_BLOCKLIST))
        if not rows:
            try:
                ex_fb = list_explore_feed(
                    h, market_type=market_type, statuses=["buy", "watch"], limit=min(fetch_limit, 400),
                )
                rows = _explore_feed_as_recommendation_rows(ex_fb)
                if rows:
                    reco_data_source = "explore_signals"
            except Exception as _rfb:
                logger.debug("recommendations explore_signals fallback: %s", _rfb)
            if not rows:
                rows = []
    except Exception as e:
        logger.error(f"list_recommendations failed: {e}")
        rows = []
        # Don't return error - just return empty list with status

    explore_rejected_payload: List[Dict[str, Any]] = []
    explore_rejected_symbols: set = set()
    try:
        explore_rejected_payload = list_explore_rejected(h, limit=50)
        explore_rejected_symbols = {
            str(x.get("symbol") or "").strip()
            for x in explore_rejected_payload
            if x.get("symbol")
        }
    except Exception as _exr_err:
        logger.debug("explore_rejected load: %s", _exr_err)

    # Log DB counts when medium is empty (diagnostics)
    if h == "medium" and len(rows) == 0:
        try:
            counts = count_recommendations_by_horizon()
            logger.warning("Medium Term has 0 rows. DB counts: short=%d medium=%d long=%d",
                counts.get("short", 0), counts.get("medium", 0), counts.get("long", 0))
        except Exception:
            pass
    
    with _globals_lock:
        state = (_RECO_STATE.get(h) or {}).copy()
        short_state = (_RECO_STATE.get("short") or {}).copy()
        medium_state = (_RECO_STATE.get("medium") or {}).copy()
        long_state = (_RECO_STATE.get("long") or {}).copy()
    last_scan = state.get("last_run_ts", 0)
    now_i = int(time.time())
    last_scan_by_horizon = {
        "short": {"ts": short_state.get("last_run_ts"), "age_sec": now_i - short_state.get("last_run_ts", 0) if short_state.get("last_run_ts") else None},
        "medium": {"ts": medium_state.get("last_run_ts"), "age_sec": now_i - medium_state.get("last_run_ts", 0) if medium_state.get("last_run_ts") else None},
        "long": {"ts": long_state.get("last_run_ts"), "age_sec": now_i - long_state.get("last_run_ts", 0) if long_state.get("last_run_ts") else None},
    }
    scan_error = state.get("error", "")
    now = int(time.time())
    scan_age = now - last_scan if last_scan > 0 else 999999
    
    # If no rows, determine status
    if len(rows) == 0:
        if last_scan == 0:
            # Never scanned - trigger async scan and return warming_up
            status = "warming_up"
            reason = "no_scan_yet"
            message = "Generating recommendations... (first scan in progress)"
            import threading
            def _trigger_scan():
                try:
                    logger.info(f"Auto-triggering initial scan for {h} horizon")
                    _scan_recommendations(h)
                except Exception as e:
                    logger.error(f"Auto-trigger scan failed: {e}")
            threading.Thread(target=_trigger_scan, daemon=True).start()
        elif scan_age > 600:  # > 10 minutes old
            if scan_error:
                status = "error"
                reason = "scan_failed"
                message = f"Last scan failed: {scan_error}"
            else:
                status = "warming_up"
                reason = "scan_stale"
                message = "Recommendations are stale. Refreshing..."
                # Trigger refresh in background
                import threading
                def _trigger_refresh():
                    try:
                        logger.info(f"Auto-refreshing stale scan for {h} horizon")
                        _scan_recommendations(h)
                    except Exception as e:
                        logger.error(f"Auto-refresh failed: {e}")
                threading.Thread(target=_trigger_refresh, daemon=True).start()
        else:
            # Recently scanned but no results - might be filtering issue
            status = "ready"
            reason = "no_matches"
            message = "No recommendations match your filters"
    else:
        status = "ready"
        reason = "ok"
        message = "Recommendations loaded" 
    
    include_set = {s.strip().upper() for s in include.split(",") if s.strip()}
    exclude_set = {s.strip().upper() for s in exclude.split(",") if s.strip()}
    active_symbols, active_reason = _active_symbol_set()
    seen_normalized: set = set()  # Dedupe by normalized symbol
    btc_ctx = _btc_context()
    macro_risk_off = bool(btc_ctx.get("risk_off", False))
    _strategy_perf = _get_cached_strategy_win_rates(h, 90)

    # BTC inter-market regime penalty for altcoins
    _btc_regime_label = ""
    _btc_inter_market_adj = 0.0
    try:
        btc_rec = None
        for _r in rows:
            _s = str(_r.get("symbol") or "").upper()
            if _s in ("XBT/USD", "BTC/USD"):
                btc_rec = _r
                break
        if btc_rec:
            _bm = json.loads(btc_rec.get("metrics_json") or "{}")
            _btc_regime_label = str(_bm.get("regime") or "").upper()
            if _btc_regime_label in ("BEAR", "STRONG_BEAR"):
                _btc_inter_market_adj = -5.0
            elif _btc_regime_label in ("WEAK_BEAR", "RANGING", "RANGE"):
                _btc_inter_market_adj = -2.0
            elif _btc_regime_label in ("BULL", "STRONG_BULL"):
                _btc_inter_market_adj = 3.0
            if _btc_inter_market_adj != 0:
                logger.info("BTC regime: %s — applying %.0f inter-market adjustment to altcoins", _btc_regime_label, _btc_inter_market_adj)
    except Exception:
        pass

    items = []
    eligible_count = 0

    # Process rows with error handling to avoid blocking on malformed data.
    # When filtering by market_type, process many more rows (crypto/stocks mixed in DB).
    max_process = min(len(rows), 800) if market_type != "all" else min(len(rows), limit)
    processed = 0
    start_time = time.time()
    max_process_time = 1.5  # Max 1.5 seconds for processing (reduced for speed)
    
    for r in rows:
        if processed >= max_process:
            break
        # Timeout guard - if processing takes too long, return what we have
        elapsed = time.time() - start_time
        if elapsed > max_process_time:
            logger.warning(f"Recommendations processing timeout after {elapsed:.1f}s, returning {processed} items")
            break
        try:
            sym = str(r.get("symbol") or "")
            if not sym:
                continue
            if "/" in sym and (sym.split("/")[0] or "").upper() in FIAT_BASES:
                continue
            if "/" in sym and (sym.split("/")[0] or "").upper() in CRYPTO_BLOCKLIST:
                continue
            # Fast JSON parse with fallback
            try:
                metrics = json.loads(r.get("metrics_json") or "{}")
            except (json.JSONDecodeError, TypeError):
                metrics = {}
            risk_flags = []
            try:
                _rf_raw = r.get("risk_flags_json")
                if _rf_raw:
                    risk_flags = json.loads(_rf_raw)
            except (json.JSONDecodeError, TypeError):
                pass
            # Exclude symbols explicitly rejected in explore_signals (DB)
            if sym in explore_rejected_symbols:
                continue
            score_breakdown = {}
            try:
                _sbd_raw_early = r.get("score_breakdown_json")
                if _sbd_raw_early:
                    score_breakdown = json.loads(_sbd_raw_early)
            except (json.JSONDecodeError, TypeError):
                pass
            
            # 1. Market Separation — always trust symbol format, not DB
            if "/" in str(sym):
                item_market = "crypto"
            else:
                item_market = "stocks"

            if market_type != "all":
                if market_type == "stocks" and item_market != "stocks":
                    continue
                if market_type == "crypto" and item_market != "crypto":
                    continue

            # Skip stock recommendations when Alpaca is not connected - prevents false signals
            if item_market == "stocks" and not alpaca_ready:
                continue

            # For crypto, only allow USD-quoted pairs and exclude non-USD duplicates
            if item_market == "crypto":
                if "/" not in sym:
                    continue
                base, quote_ccy = (sym.split("/") + [""])[:2]
                if (quote_ccy or "").upper() != "USD":
                    continue
                if (base or "").upper() in CRYPTO_BLOCKLIST:
                    continue
                # TOP 30 whitelist: block small-caps - enforced at API level (handles BTC/USD, XBT/USD, BTCUSD, etc.)
                base_norm = _crypto_base_from_symbol(sym)
                if RECO_CRYPTO_TOP_30_ONLY and base_norm not in TOP_30_CRYPTO_BASES:
                    if logger.isEnabledFor(logging.DEBUG):
                        logger.debug("Whitelist BLOCKED: %s (base=%s not in top30)", sym, base_norm)
                    continue
                # Ensure symbol exists on Kraken (never recommend CC or invalid pairs)
                resolved, err = _validate_crypto_symbol(sym)
                if not resolved:
                    continue
                # Min volume filter: exclude low-liquidity crypto ($10M+ 24h volume)
                vol = float(metrics.get("volume_24h_quote") or 0)
                if vol > 0 and vol < RECO_MIN_VOLUME_24H:
                    continue
            if item_market == "stocks":
                mc = metrics.get("market_cap") or (float(metrics.get("market_cap_b") or 0) * 1e9)
                if mc and mc < RECO_MIN_MARKET_CAP:
                    continue

            # 2. Filters
            if include_set and sym.upper() not in include_set:
                continue
            if exclude_set and sym.upper() in exclude_set:
                continue
            if quote and not sym.upper().endswith(f"/{quote.upper()}"):
                continue

            # 2b. Sector filter (stocks only)
            if sector_filter and sector_filter.lower() != "all" and item_market == "stocks":
                item_sector = (metrics.get("sector") or "").strip()
                if item_sector.lower() != sector_filter.lower():
                    continue

            # 3. Eligibility
            eligible = bool(metrics.get("eligible"))
            research_only = bool(metrics.get("research_only"))
            
            if eligible:
                eligible_count += 1
            
            # 4. Score Filter (min_score < 0 means "Any" - show everything)
            score = float(r.get("score") or 0.0)
            if min_score >= 0 and not include_all:
                if (not eligible or research_only) and score < min_score:
                    continue
                if score < min_score:
                    continue
            
            # 5. Signal Filter (buy/watch/all)
            # Determine signal from score: per-market buy threshold, shared watch threshold
            buy_thresh = _reco_buy_threshold_stocks() if item_market == "stocks" else _reco_buy_threshold_crypto()
            item_signal = "buy" if score >= buy_thresh else ("watch" if score >= _reco_watch_threshold() else "wait")

            # 5-override. Bear-market oversold dip detection for crypto:
            # Even when score=0 (bear regime, ineligible), surface quality cryptos as WATCH
            # when RSI is extremely oversold (< 30) — these are DCA/reversal setup candidates.
            # Only applies in signal=watch or signal=all contexts (never promotes to BUY).
            if item_market == "crypto" and item_signal == "wait" and signal_filter in ("watch", "all"):
                _bear_rsi = None
                try:
                    _bear_rsi = float(metrics.get("rsi_value") or r.get("rsi_value") or 0)
                except (TypeError, ValueError):
                    pass
                _bear_regime = (metrics.get("regime") or "").upper()
                _bear_volume = float(metrics.get("volume_24h_quote") or metrics.get("volume_24h") or 0)
                _min_bear_vol = 500_000  # Only surface liquid crypto for dip setups
                if (_bear_rsi and _bear_rsi <= 30
                        and _bear_regime in ("BEAR", "STRONG_BEAR", "WEAK_BEAR", "RANGE")
                        and _bear_volume >= _min_bear_vol
                        and score == 0.0):
                    # Promote to WATCH with a base score of 52 (just above watch threshold)
                    # so it appears with low priority but is visible for DCA consideration.
                    score = 52.0
                    item_signal = "watch"
                    risk_flags = list(risk_flags) + [f"Oversold dip setup (RSI {_bear_rsi:.0f}) — bear regime, DCA only"]

            if signal_filter == "buy" and item_signal != "buy":
                continue  # Skip non-buy items
            elif signal_filter == "watch" and item_signal not in ("buy", "watch"):
                continue  # Skip wait/sell items
            # "all" includes everything, no filter

            # 5b. Volatility Filter (low/medium/high/all)
            # low: < 2%, medium: 2-5%, high: >= 5%. Use atr_pct if volatility/vol missing.
            if volatility_filter != "all":
                item_volatility = float(
                    metrics.get("volatility") or metrics.get("vol") or metrics.get("atr_pct") or 0.0
                )
                if volatility_filter == "low" and item_volatility >= 0.02:
                    continue
                elif volatility_filter == "medium" and (item_volatility < 0.02 or item_volatility >= 0.05):
                    continue
                elif volatility_filter == "high" and item_volatility < 0.05:
                    continue

            # 5c. Regime Filter (bull/breakout/range/bear/all)
            if regime_filter != "all":
                item_regime = (metrics.get("regime") or "").strip().upper()
                if not item_regime:
                    continue
                want = regime_filter.replace(" ", "").lower()
                ir = item_regime.upper()
                if want == "bull" and ir not in ("BULL",):
                    continue
                if want == "breakout" and ir not in ("BREAKOUT",):
                    continue
                if want == "range" and ir not in ("RANGE",):
                    continue
                if want == "bear" and ir not in ("BEAR", "HIGH_VOL_DEFENSIVE", "RISK_OFF"):
                    continue

            # 5d. Hard regime gate: block BUY only in clearly bearish regimes.
            # RANGE and HIGH_VOL_DEFENSIVE are NOT hard-blocked — stocks frequently trade in
            # these regimes and can still be profitable entries with sufficient score conviction.
            # Only hard-block truly bearish/risk-off conditions.
            _BULLISH_REGIMES = {"BULL", "STRONG_BULL", "BREAKOUT", "TREND_UP", "WEAK_BULL"}
            # Hard block: clearly bearish — no BUY regardless of score
            _BLOCKED_BUY_REGIMES = {"BEAR", "STRONG_BEAR", "RISK_OFF"}
            # Elevated conviction needed: sideways/high-vol — require score >= 72 (not full block)
            _ELEVATED_SCORE_REGIMES = {"RANGE", "RANGING", "HIGH_VOL_DEFENSIVE", "SIDEWAYS", "WEAK_BEAR"}
            if item_signal == "buy":
                _item_regime_gate = (metrics.get("regime") or "").strip().upper()
                if not _item_regime_gate:
                    # No regime data stored — require above-average conviction
                    if score < 75.0:
                        item_signal = "watch"
                elif _item_regime_gate in _BLOCKED_BUY_REGIMES:
                    item_signal = "watch"  # Hard block: bearish/risk-off regime
                elif _item_regime_gate in _ELEVATED_SCORE_REGIMES:
                    # Sideways/high-vol: require elevated score but don't hard-block
                    if score < 72.0:
                        item_signal = "watch"
                elif _item_regime_gate not in _BULLISH_REGIMES:
                    # Unknown/neutral regime — require solid conviction
                    if score < 70.0:
                        item_signal = "watch"
            # 5d-ii. Hard RSI gate: RSI >= 80 should never be a BUY
            if item_signal == "buy":
                _rsi_gate = None
                try:
                    _rsi_gate = float(metrics.get("rsi_value") or r.get("rsi_value") or 0)
                except (TypeError, ValueError):
                    pass
                if _rsi_gate and _rsi_gate >= 80:
                    item_signal = "watch"

            # 5d-ii-b. Extended RSI penalties: RSI > 72 = -5, RSI > 78 = -10 + demote
            if _rsi_gate and _rsi_gate > 72 and _rsi_gate < 80:
                _rsi_penalty = -5 if _rsi_gate <= 78 else -10
                score = max(0, score + _rsi_penalty)
                if _rsi_gate > 78 and item_signal == "buy":
                    item_signal = "watch"

            # 5d-ii-c. Volume conviction: gentle penalty, only block extreme illiquidity
            _vr_for_gate = float(metrics.get("volume_ratio") or r.get("volume_ratio") or 1.0)
            if _vr_for_gate < 0.2:
                score = min(score, 30)
                if item_signal == "buy":
                    item_signal = "watch"
                risk_flags.append("Extreme illiquidity — volume ratio < 0.2")
            elif _vr_for_gate < 1.5:
                _vol_penalty = max(-5.0, (_vr_for_gate - 1.5) * 10.0)
                score = max(0, score + _vol_penalty)
                if _vol_penalty < -2:
                    risk_flags.append("Low volume — use smaller position")

            # 5d-ii-d. Long-term downtrend penalty: -5 per 20% of drawdown (max -15)
            for _rf in risk_flags:
                _rf_s = str(_rf).lower()
                if "down" in _rf_s and "long-term" in _rf_s:
                    import re as _re_mod
                    _dd_match = _re_mod.search(r'-(\d+)%', str(_rf))
                    if _dd_match:
                        _dd_val = int(_dd_match.group(1))
                        _tiers = min(_dd_val // 20, 3)
                        score = max(0, score - _tiers * 5)
                    break

            # 5d-iii. Signal age penalty (Problem 1) + Price confirmation (Problem 3) + Three-gate BUY (Problem 4)
            _created_ts_raw = int(r.get("created_ts") or r.get("updated_ts") or 0)
            _signal_age_minutes = (now - _created_ts_raw) / 60.0 if _created_ts_raw > 0 else None
            _change_pct = None
            try:
                _ch = r.get("change_24h") or metrics.get("change_24h")
                if _ch is not None:
                    _change_pct = float(_ch)
            except (TypeError, ValueError):
                pass
            _tentative_signal = "buy" if score >= buy_thresh else "watch"
            _price_conf_delta = 0.0
            _price_conf_label = ""
            _stale_penalty = 0.0
            _signal_freshness_label = "Fresh"
            try:
                _price_conf_delta, _price_conf_label = price_confirmation_score(_change_pct, _tentative_signal)
                score += _price_conf_delta
                _pen, _reason, _force_watch = signal_age_penalty(_signal_age_minutes)
                if _force_watch:
                    _pre_cap = score
                    score = min(score, 62.0)
                    _stale_penalty = max(0, _pre_cap - 62)
                    item_signal = "watch"
                    _signal_freshness_label = "Previous Session"
                elif _pen > 0:
                    score = max(0, score - _pen)
                    _stale_penalty = _pen
                    _signal_freshness_label = _reason if _pen >= 15 else "Aging"
                if _signal_age_minutes is not None:
                    if _signal_age_minutes > 480:
                        _signal_freshness_label = "Previous Session"
                    elif _signal_age_minutes >= 120:
                        _signal_freshness_label = "Previous Session (-15)"
                    elif _signal_age_minutes >= 60:
                        _signal_freshness_label = "Aging (-7)"
            except Exception as _e:
                logger.warning("explore_scorer failed (row %s): %s", sym, _e)
            # Three-gate BUY: all must pass or status = WATCH
            # Gate 3: aligned with the actual per-market buy threshold so there's no hidden
            # secondary floor that silently demotes valid signals (e.g. stocks scoring 68-69).
            if item_signal == "buy":
                _gate1 = _signal_age_minutes is None or _signal_age_minutes < 480
                _gate2 = _price_conf_delta > -20
                _gate3 = score >= buy_thresh  # Use actual market-specific buy threshold
                if not (_gate1 and _gate2 and _gate3):
                    item_signal = "watch"
            # Build score breakdown audit for UI (Problem 5)
            _regime_val = float(score_breakdown.get("regime", {}).get("value", 0) if isinstance(score_breakdown.get("regime"), dict) else 0)
            _trend_val = float(score_breakdown.get("trend", {}).get("value", 0) if isinstance(score_breakdown.get("trend"), dict) else 0)
            _vol_val = float(score_breakdown.get("volume", {}).get("value", 0) if isinstance(score_breakdown.get("volume"), dict) else 0)
            _score_breakdown_audit = {
                "regime": _regime_val,
                "trend": _trend_val,
                "volume": _vol_val,
                "price_confirmation": _price_conf_delta,
                "price_confirmation_label": _price_conf_label,
                "signal_freshness": _signal_freshness_label,
                "stale_penalty": _stale_penalty,
            }

            # 5d-iv. BTC inter-market penalty for altcoins
            _btc_adj_applied = 0.0
            if _btc_inter_market_adj != 0 and item_market == "crypto":
                _sym_upper = sym.upper()
                if _sym_upper not in ("XBT/USD", "BTC/USD"):
                    score = max(0, score + _btc_inter_market_adj)
                    _btc_adj_applied = _btc_inter_market_adj
                    # Re-check buy threshold after penalty
                    if item_signal == "buy" and score < buy_thresh:
                        item_signal = "watch"

            # 5d-iv-b. Fear & Greed dynamic adjustment (crypto: extreme fear = bonus, extreme greed = penalty)
            _fear_greed_adj = 0.0
            if item_market == "crypto":
                _fear_greed_adj = _fear_greed_score_adjustment()
                if _fear_greed_adj != 0:
                    score = max(0, min(95, score + _fear_greed_adj))
                    if item_signal == "buy" and score < buy_thresh:
                        item_signal = "watch"

            # 5d-iv-c. BTC macro momentum overlay: when BTC probability of down + high vol are both
            # elevated, apply an extra penalty and require higher conviction for BUY.
            # This catches cases where individual symbols look great but the macro backdrop is risky
            # (e.g. BTC RANGING with 75% downward probability and 91% high-vol probability).
            _btc_down_prob = float(btc_ctx.get("btc_down", 0))
            _btc_hv_prob = float(btc_ctx.get("btc_hv", 0))
            if item_market == "crypto" and sym.upper() not in ("XBT/USD", "BTC/USD"):
                if _btc_down_prob >= 0.70 and _btc_hv_prob >= 0.70:
                    # Dangerous macro environment: high downward momentum + high volatility
                    score = max(0, score - 5)
                    if item_signal == "buy" and score < (buy_thresh + 5):
                        item_signal = "watch"
                elif _btc_down_prob >= 0.60 and _btc_hv_prob >= 0.60:
                    # Cautious environment: moderate downward + elevated vol
                    score = max(0, score - 2)
                    if item_signal == "buy" and score < buy_thresh:
                        item_signal = "watch"

            # 5d-v. Momentum confirmation bonus from score breakdown
            _sbd_momentum = score_breakdown.get("momentum", {})
            _sbd_trend = score_breakdown.get("trend", {})
            _mom_val = float(_sbd_momentum.get("value", 0) if isinstance(_sbd_momentum, dict) else 0)
            _trend_val = float(_sbd_trend.get("value", 0) if isinstance(_sbd_trend, dict) else 0)
            if _mom_val > 0 and _trend_val > 0:
                score = min(95, score + 3)

            # 5d-vi. ATR volatility filter: too flat (< 1.5%) or too volatile (> 12%) → demote
            _atr_raw = float(metrics.get("atr_pct") or 0)
            _vol_raw = float(metrics.get("volatility") or metrics.get("vol") or 0)
            _atr_pct = _atr_raw if _atr_raw > 0.5 else (_vol_raw * 100 if _vol_raw > 0 else 0)
            if _atr_pct > 0:
                if _atr_pct < 1.5 or _atr_pct > 12:
                    if item_signal == "buy":
                        item_signal = "watch"

            # 5d-vi-b. Volume ratio gate: require above-average volume for BUY conviction.
            # Breakouts/moves on low relative volume have historically poor follow-through.
            if item_signal == "buy":
                _vol_ratio_gate = float(metrics.get("volume_ratio") or r.get("volume_ratio") or 0)
                if 0 < _vol_ratio_gate < 0.75:
                    # Volume is 25%+ below average — price move lacks institutional confirmation
                    item_signal = "watch"
                    risk_flags = list(risk_flags) + [f"Below-avg volume ({_vol_ratio_gate:.2f}x) — weak BUY conviction"]

            # 5d-vi-c. High Vol Defensive strategy gate: when ATR forces a defensive strategy,
            # require higher score conviction before calling it a BUY.
            # High-vol assets in defensive mode have poor risk/reward unless the signal is very strong.
            if item_signal == "buy":
                _strategy_gate = (metrics.get("strategy") or metrics.get("suggested_strategy") or metrics.get("recommended_strategy") or "").lower()
                if "high_vol" in _strategy_gate and score < 78.0:
                    item_signal = "watch"

            # Re-apply signal filter after regime gate
            if signal_filter == "buy" and item_signal != "buy":
                continue

            # 5e. Macro Risk-Off: block only if asset is also falling
            macro_warning = None
            if macro_risk_off and item_signal == "buy":
                if _change_pct is not None and _change_pct < -2.0:
                    item_signal = "watch"
                    macro_warning = "Risk-off + falling price — demoted to watch."
                elif _change_pct is None:
                    item_signal = "watch"
                    macro_warning = "Risk-off + unknown price change — demoted to watch."
                else:
                    macro_warning = "Risk-off but asset showing relative strength — allowing signal."

            # Override: always allow defensive assets even in risk-off
            _is_defensive = sym.upper() in DEFENSIVE_ASSETS
            if macro_risk_off and item_signal == "watch" and _is_defensive:
                if score >= _reco_watch_threshold():
                    item_signal = "buy"
                    macro_warning = "Defensive asset promoted to buy during risk-off."

            # 5f. Market hours check — demote stock buys outside trading hours
            if item_market == "stocks" and item_signal == "buy":
                if not _stock_market_open():
                    item_signal = "watch"
                    _score_breakdown_audit["market_hours"] = "Market closed — signal valid at open"

            # 6. Populate Ticker Data (FAST - use metrics only, no network calls)
            # Prices will be filled via /api/prices endpoint async
            price_from_metrics = metrics.get("price")
            price = float(price_from_metrics) if price_from_metrics and price_from_metrics > 0 else None
            
            # Compute sort key based on sort parameter
            sort_key = score  # Default to score
            if sort_by == "profit_factor":
                sort_key = float(metrics.get("profit_factor") or metrics.get("expected_return") or score)
            elif sort_by == "drawdown":
                # Lower drawdown is better, so negate for descending sort
                cur_dd = float(metrics.get("cur_dd") or 0.0)
                sort_key = -cur_dd  # Negate so lower drawdown sorts higher
            elif sort_by == "winrate":
                sort_key = float(metrics.get("winrate") or metrics.get("win_rate") or score)
            # else: sort_by == "score", use score as sort_key
            # Tiebreaker: when many picks share the same score (all hit the 92 cap),
            # add a small fractional bonus so best-quality picks rank first.
            # Components: entry quality (A=+0.4, B=+0.2, C=0), volume ratio (capped +0.3), earnings risk (-0.5 if <3 days).
            _eq_tiebreak = {"A": 0.40, "B": 0.20, "C": 0.0}.get(
                str(r.get("entry_quality") or metrics.get("entry_quality") or "C").upper(), 0.0
            )
            _vr_raw = float(metrics.get("volume_ratio") or r.get("volume_ratio") or 1.0)
            _vr_tiebreak = min(0.30, max(0.0, (_vr_raw - 1.0) * 0.15))  # above-avg volume is a bonus
            _ed = metrics.get("earnings_days")
            _earnings_risk = -0.50 if isinstance(_ed, (int, float)) and 0 <= _ed <= 3 else 0.0
            sort_key += _eq_tiebreak + _vr_tiebreak + _earnings_risk

            # Sparkline - only use cache, don't fetch on-demand (too slow)
            sparkline = []
            try:
                c_key_1d = f"{sym}|1d|500"
                c_data = _RECO_OHLCV_CACHE.get(c_key_1d, {}).get("data")
                if not c_data:
                    c_key_4h = f"{sym}|4h|300"
                    c_data = _RECO_OHLCV_CACHE.get(c_key_4h, {}).get("data")
                if c_data and len(c_data) > 0:
                    # Get last 24-30 data points for 7d trend
                    sparkline = [float(c[4]) for c in c_data[-30:]]
            except Exception:
                pass

            # Hybrid screener data from DB
            _db_composite = r.get("composite_score")
            _db_confidence = r.get("confidence_score")
            _db_conviction = r.get("conviction_grade")
            _db_factor_scores = {}
            try:
                _fs_raw = r.get("factor_scores_json")
                if _fs_raw:
                    _db_factor_scores = json.loads(_fs_raw)
            except (json.JSONDecodeError, TypeError):
                pass
            _db_signal_flags = []
            try:
                _sf_raw = r.get("signal_flags_json")
                if _sf_raw:
                    _db_signal_flags = json.loads(_sf_raw)
            except (json.JSONDecodeError, TypeError):
                pass

            # Also read from metrics (populated during scan)
            if _db_composite is None:
                _db_composite = metrics.get("composite_score")
            if _db_confidence is None:
                _db_confidence = metrics.get("confidence_score")
            if _db_conviction is None:
                _db_conviction = metrics.get("conviction_grade")
            if not _db_factor_scores:
                _db_factor_scores = metrics.get("factor_scores") or {}

            # Rating: use conviction grade when available, fall back to score buckets
            if _db_conviction == "A":
                rating = "Strong Buy"
            elif _db_conviction == "B":
                rating = "Buy"
            elif _db_conviction == "C":
                rating = "Watch"
            elif _db_conviction == "D":
                rating = "Avoid"
            elif score >= 85:
                rating = "Strong Buy"
            elif score >= 55:
                rating = "Buy"
            elif score >= 40:
                rating = "Watch"
            else:
                rating = "Avoid"

            regime = {}
            reasons = []
            try:
                regime_raw = r.get("regime_json")
                if regime_raw:
                    parsed = json.loads(regime_raw)
                    if isinstance(parsed, dict) and "label" in parsed and "1d" not in parsed:
                        regime = {"1d": parsed, "4h": parsed}
                    else:
                        regime = parsed
            except (json.JSONDecodeError, TypeError):
                pass
            try:
                reasons_raw = r.get("reasons_json")
                if reasons_raw:
                    reasons = json.loads(reasons_raw)
            except (json.JSONDecodeError, TypeError):
                pass

            norm_sym = _normalize_symbol(sym)
            if norm_sym in seen_normalized:
                continue
            seen_normalized.add(norm_sym)
            already_active = norm_sym in active_symbols
            active_reason_list = list(active_reason.get(norm_sym) or [])
            if not show_already_active and already_active:
                continue

            if not score_breakdown:
                try:
                    from explore_v2 import compute_score_breakdown, is_enabled as _ev2
                    if _ev2():
                        regime_label = (regime.get("1d") or {}).get("label") or ""
                        score_breakdown = compute_score_breakdown(score, metrics, regime_label)
                except ImportError:
                    pass

            # Scanner readiness enrichment (from cached setup if available)
            scanner_fields = _get_scanner_fields_for_item(sym, item_market, metrics)

            # Confidence band from score
            _cb = "NONE"
            if score >= 82:
                _cb = "HIGH"
            elif score >= 70:
                _cb = "MEDIUM"
            elif score >= 62:
                _cb = "LOW"

            # Signal age in seconds
            _created_ts = int(r.get("created_ts") or 0)
            _signal_age_sec = now - _created_ts if _created_ts > 0 else None

            # RSI and volume ratio from metrics
            _rsi_val = metrics.get("rsi_value") or r.get("rsi_value")
            _vol_ratio = metrics.get("volume_ratio") or r.get("volume_ratio")
            _regime_label = (regime.get("1d") or {}).get("label") or metrics.get("regime") or ""
            _entry_quality = r.get("entry_quality") or metrics.get("entry_quality") or "C"
            _change_24h = r.get("change_24h") or metrics.get("change_24h")

            # Global score cap: ensure no score > 95 ever reaches the frontend
            score = max(0.0, min(95.0, score))

            # Re-evaluate signal against FINAL adjusted score (after all penalties/bonuses).
            # An item initially labeled "watch" from the raw DB score could have been penalized
            # down to score=26 — that should become "wait" and be excluded, not shown as "watch".
            _final_buy_thresh = _reco_buy_threshold_stocks() if item_market == "stocks" else _reco_buy_threshold_crypto()
            _final_watch_thresh = _reco_watch_threshold()
            if item_signal == "buy" and score < _final_buy_thresh:
                item_signal = "watch"  # score fell below buy threshold after adjustments
            if item_signal in ("buy", "watch") and score < _final_watch_thresh:
                item_signal = "wait"  # score fell below watch threshold — exclude from explore
            # Re-apply signal filter against corrected signal
            if signal_filter == "buy" and item_signal != "buy":
                continue
            if signal_filter == "watch" and item_signal not in ("buy", "watch"):
                continue
            # Buy+Watch: hide stale session watches (>8h) — use signal=all to see them
            if signal_filter == "watch" and item_signal == "watch":
                if _signal_age_minutes is not None and _signal_age_minutes > 480:
                    continue

            items.append(
                {
                    "symbol": sym,
                    "score": score,
                    "sort_key": sort_key,
                    "signal": item_signal,
                    "horizon": h,
                    "market_type": item_market,
                    "price": price,
                    "change_pct": _change_pct,
                    "volume": float(metrics.get("volume_24h_quote") or metrics.get("volume_24h") or 0) or None,
                    "market_cap": None,
                    "rating": rating,
                    "confidence": scanner_fields.get("confidence") or float(metrics.get("confidence_score") or 0.0),
                    "confidence_band": _cb,
                    "sparkline": sparkline,
                    "regime_1d": (regime.get("1d") or {}).get("label"),
                    "regime_4h": (regime.get("4h") or {}).get("label"),
                    "regime_label": _regime_label,
                    "weekly_trend": metrics.get("weekly_trend"),
                    "strategy_mode": metrics.get("detected_strategy") or metrics.get("strategy") or metrics.get("recommended_strategy") or metrics.get("suggested_strategy") or "smart_dca",
                    "suggested_strategy": metrics.get("detected_strategy") or _strategy_display_name(metrics.get("strategy") or metrics.get("recommended_strategy") or metrics.get("suggested_strategy")),
                    "recommended_strategy": metrics.get("detected_strategy") or _strategy_display_name(metrics.get("strategy") or metrics.get("recommended_strategy") or metrics.get("suggested_strategy")),
                    "volatility": metrics.get("atr_pct"),
                    "risk_flags": risk_flags,
                    "updated_ts": _created_ts,
                    "signal_age_sec": _signal_age_sec,
                    "rsi_value": float(_rsi_val) if _rsi_val is not None else None,
                    "volume_ratio": float(_vol_ratio) if _vol_ratio is not None else None,
                    "eligible": eligible,
                    "research_only": research_only or (item_market == "stocks" and not _alpaca_any_ready()),
                    "reasons": reasons,
                    "sector": metrics.get("sector") if item_market == "stocks" else None,
                    "benchmark_vs": metrics.get("benchmark_vs") or None,
                    "peer_rank": metrics.get("peer_rank") or None,
                    "beta": metrics.get("beta"),
                    "diversify_key": (metrics.get("sector") or "unknown") if item_market == "stocks" else "crypto",
                    "already_active": already_active,
                    "active_reason": active_reason_list,
                    "top_reasons": (reasons or [])[:3],
                    "top_risk_flags": (risk_flags or [])[:3],
                    "score_breakdown": score_breakdown,
                    "macro_risk_off": macro_risk_off,
                    "macro_warning": macro_warning,
                    "ml_confidence": (metrics.get("ml_confidence_pct") or metrics.get("ml_confidence")),
                    "crypto_cycle": metrics.get("crypto_cycle") or metrics.get("cycle_phase"),
                    "funding_rate_warning": (float(metrics.get("funding_rate") or 0) > 0.001) if item_market == "crypto" else None,
                    "earnings_warning": (isinstance(metrics.get("earnings_days"), (int, float)) and 0 <= metrics.get("earnings_days", 99) <= 5) if item_market == "stocks" else None,
                    "earnings_days": metrics.get("earnings_days"),
                    "earnings_flag": metrics.get("earnings_flag"),
                    "entry_quality": _entry_quality,
                    "change_24h": float(_change_24h) if _change_24h is not None else None,
                    "suggested_entry": _compute_suggested_entry(score_breakdown, price, item_signal),
                    "fear_greed_adj": round(_fear_greed_adj, 1) if _fear_greed_adj != 0 else None,
                    "btc_regime_adj": _btc_adj_applied if _btc_adj_applied != 0 else None,
                    "sector": STOCK_SECTORS.get(sym.replace("/USD", "").upper(), "") if item_market == "stocks" else "",
                    "volume_anomaly": bool(metrics.get("volume_anomaly")),
                    "signal_age_minutes": round(_signal_age_minutes, 0) if _signal_age_minutes is not None else None,
                    "score_breakdown_audit": _score_breakdown_audit,
                    "composite_score": float(_db_composite) if _db_composite is not None else None,
                    "confidence_score": float(_db_confidence) if _db_confidence is not None else None,
                    "conviction_grade": _db_conviction or None,
                    "factor_scores": _db_factor_scores or None,
                    "hybrid_flags": _db_signal_flags or None,
                    "strategy_reason": metrics.get("strategy_reason") or "",
                    "strategy_win_rate": ((_strategy_perf.get(metrics.get("detected_strategy") or "") or {}).get("win_rate")),
                    "strategy_signal_count": ((_strategy_perf.get(metrics.get("detected_strategy") or "") or {}).get("signals", 0)),
                    "rsi_2": metrics.get("rsi_2"),
                    "price_to_52wk_high": metrics.get("price_to_52wk_high"),
                    "cardwell_regime": metrics.get("cardwell_regime"),
                    "rsi2_entry_signal": metrics.get("rsi2_entry_signal"),
                    "macd_combo_entry": metrics.get("macd_combo_entry"),
                    **scanner_fields,
                }
            )
            processed += 1
        except Exception:
            continue
            
    # Sort (Problem 6): score desc, signal_age asc (fresher first), volume desc
    items.sort(key=lambda x: (
        0 if x.get("eligible") else 1,
        -float(x.get("sort_key", x.get("score", 0))),
        float(x.get("signal_age_minutes") or 999999),
        -float(x.get("volume") or 0),
    ))

    # Diversify top N by sector (stocks) / crypto so list isn't all one sector
    try:
        from explore_v2 import diversify_picks, is_enabled as explore_v2_enabled
        if explore_v2_enabled() and items and "diversify_key" in (items[0] or {}):
            need = min(len(items), offset + limit)
            items = diversify_picks(items, top_k=max(need, limit), cluster_key="diversify_key")
    except ImportError:
        pass

    # Enrich crypto BUY signals with news sentiment (cached, non-blocking)
    for item in items:
        if item.get("signal") == "buy" and item.get("market_type") == "crypto":
            try:
                ns = _fetch_news_sentiment(item.get("symbol", ""))
                item["news_sentiment"] = ns.get("sentiment", "Neutral")
                item["news_score_adj"] = ns.get("score_adj", 0)
                if ns.get("score_adj", 0) != 0:
                    item["score"] = max(0, min(95, item["score"] + ns["score_adj"]))
            except Exception:
                item["news_sentiment"] = "Neutral"
                item["news_score_adj"] = 0
        else:
            item["news_sentiment"] = None
            item["news_score_adj"] = 0

    # Apply sector momentum to stock items
    _update_sector_momentum(items)
    for item in items:
        if item.get("market_type") == "stocks":
            _sec = item.get("sector") or ""
            _sm = _SECTOR_MOMENTUM.get(_sec)
            if _sm:
                item["sector_momentum"] = _sm
                _tier = _sm.get("tier", "middle")
                _sector_adj = 4.0 if _tier == "top" else (-4.0 if _tier == "bottom" else 0.0)
                if _sector_adj != 0:
                    item["score"] = max(0, min(95, item["score"] + _sector_adj))
                    item["sector_score_adj"] = _sector_adj
                    # Re-check buy threshold
                    if _sector_adj < 0 and item.get("signal") == "buy" and item["score"] < 62:
                        item["signal"] = "watch"
                else:
                    item["sector_score_adj"] = 0
            else:
                item["sector_momentum"] = None
                item["sector_score_adj"] = 0
            # Liquidity participation indicator
            _vol_ratio = item.get("volume_ratio")
            if _vol_ratio is not None:
                item["volume_pct_avg"] = round(float(_vol_ratio) * 100, 0)
                if float(_vol_ratio) < 0.30:
                    if item.get("signal") == "buy":
                        item["signal"] = "watch"
                    item["low_volume_warning"] = True
                else:
                    item["low_volume_warning"] = False
            else:
                item["volume_pct_avg"] = None
                item["low_volume_warning"] = False

    # Relative strength boost — before feed filtering
    # Assets outperforming BTC get a score lift so they reach the feed threshold
    import re as _re2
    for item in items:
        _rs_boost_applied = 0.0
        _rs_pct_found = None
        for _rsn in (item.get("reasons") or []):
            _rsn_str = str(_rsn).lower()
            if "outperform" in _rsn_str and "btc" in _rsn_str:
                try:
                    _rsm = _re2.search(r'\+(\d+\.?\d*)%', str(_rsn))
                    if _rsm:
                        _rs_pct_found = float(_rsm.group(1))
                        if _rs_pct_found >= 50:
                            _rs_boost_applied = 15.0
                        elif _rs_pct_found >= 30:
                            _rs_boost_applied = 10.0
                        elif _rs_pct_found >= 15:
                            _rs_boost_applied = 5.0
                except Exception:
                    pass
                break

        if _rs_boost_applied > 0:
            item["score"] = min(95, float(item.get("score") or 0)
                                + _rs_boost_applied)
            item["relative_strength_pct"] = _rs_pct_found
            item["relative_strength_boost"] = _rs_boost_applied
            _new_score = item["score"]
            if item.get("signal") == "wait" and _new_score >= 48:
                item["signal"] = "watch"
            if item.get("signal") == "watch" and _new_score >= 65:
                item["signal"] = "buy"

    # Fallback reason text for items with empty strategy_reason
    for item in items:
        _has_reason = bool(item.get("strategy_reason"))
        if not _has_reason and item.get("signal") in ("buy", "watch"):
            _strat_id = item.get("strategy_mode") or ""
            _sym = item.get("symbol", "")
            _score = item.get("score", 0)
            _24h = item.get("change_pct") or 0
            _regime = item.get("regime_label", "")

            if "oversold_bounce" in _strat_id:
                item["strategy_reason"] = (
                    f"Oversold bounce setup — RSI recovering from "
                    f"oversold levels, down trend losing momentum. "
                    f"24h: {float(_24h):+.1f}%, regime: {_regime}"
                )
            elif "trend_continuation" in _strat_id:
                item["strategy_reason"] = (
                    f"Trend continuation — strong uptrend with "
                    f"pullback entry. 24h: {float(_24h):+.1f}%"
                )
            elif "momentum_breakout" in _strat_id:
                item["strategy_reason"] = (
                    f"Momentum breakout — price breaking above "
                    f"resistance with volume. 24h: {float(_24h):+.1f}%"
                )
            elif not _strat_id:
                item["strategy_reason"] = (
                    f"Scanner signal — score {float(_score):.0f}, "
                    f"24h: {float(_24h):+.1f}%, regime: {_regime}"
                )

    # Apply pagination
    total_count = len(items)
    paginated_items = items[offset:offset + limit]
    
    # If no items and include_all is set, try to return at least some data with lower threshold
    if len(items) == 0 and include_all:
        # Re-scan with lower score threshold (but still no network calls)
        for r in rows[:min(100, len(rows))]:  # Check limited rows for speed
            try:
                sym = str(r.get("symbol") or "")
                if not sym:
                    continue
                if sym in explore_rejected_symbols:
                    continue
                if "/" in sym and (sym.split("/")[0] or "").upper() in FIAT_BASES:
                    continue
                try:
                    metrics = json.loads(r.get("metrics_json") or "{}")
                except (json.JSONDecodeError, TypeError):
                    metrics = {}
                item_market = (metrics.get("market_type") or "").strip().lower() or None
                if not item_market:
                    item_market = "stocks" if (len(sym) < 6 and "/" not in sym) else "crypto"
                if item_market == "stock":
                    item_market = "stocks"
                want_mt = market_type.lower() if market_type else "all"
                if want_mt != "all" and (
                    (want_mt == "stocks" and item_market != "stocks") or (want_mt == "crypto" and item_market != "crypto")
                ):
                    continue
                score = float(r.get("score") or 0.0)
                if score < -50:
                    continue
                price_from_metrics = metrics.get("price")
                norm_sym = _normalize_symbol(sym)
                already_active = norm_sym in active_symbols
                active_reason_list = list(active_reason.get(norm_sym) or [])
                if not show_already_active and already_active:
                    continue
                items.append({
                    "symbol": sym,
                    "score": score,
                    "horizon": h,
                    "market_type": item_market,
                    "price": float(price_from_metrics) if price_from_metrics and price_from_metrics > 0 else None,
                    "change_pct": None,
                    "volume": None,
                    "market_cap": None,
                    "rating": "Neutral",
                    "confidence": float(metrics.get("confidence_score") or 0.0),
                    "sparkline": [],
                    "regime_1d": None,
                    "regime_4h": None,
                    "weekly_trend": None,
                    "strategy_mode": metrics.get("strategy") or metrics.get("recommended_strategy") or "smart_dca",
                    "suggested_strategy": _strategy_display_name(metrics.get("strategy") or metrics.get("recommended_strategy")),
                    "recommended_strategy": _strategy_display_name(metrics.get("strategy") or metrics.get("recommended_strategy")),
                    "volatility": metrics.get("atr_pct"),
                    "risk_flags": [],
                    "updated_ts": int(r.get("created_ts") or 0),
                    "eligible": bool(metrics.get("eligible")),
                    "research_only": bool(metrics.get("research_only")) or (item_market == "stocks" and not _alpaca_any_ready()),
                    "reasons": [],
                    "already_active": already_active,
                    "active_reason": active_reason_list,
                })
            except Exception:
                continue
        # Update consecutive buy tracker and apply conviction bonus
        now_ts_conv = time.time()
        seen_buy_syms = set()
        for item in items:
            sym_key = item.get("symbol", "").upper()
            if item.get("signal") == "buy":
                seen_buy_syms.add(sym_key)
                tracker = _CONSECUTIVE_BUY_TRACKER.get(sym_key, {"count": 0, "last_scan_ts": 0})
                if (now_ts_conv - tracker["last_scan_ts"]) < 7200:
                    tracker["count"] = tracker["count"] + 1
                else:
                    tracker["count"] = 1
                tracker["last_scan_ts"] = now_ts_conv
                _CONSECUTIVE_BUY_TRACKER[sym_key] = tracker
                consec = tracker["count"]
                conv_bonus = 0
                if consec >= 3:
                    conv_bonus = 6
                elif consec >= 2:
                    conv_bonus = 3
                if conv_bonus > 0:
                    item["score"] = min(95, item["score"] + conv_bonus)
                item["consecutive_buy_count"] = consec
            else:
                if sym_key in _CONSECUTIVE_BUY_TRACKER:
                    _CONSECUTIVE_BUY_TRACKER[sym_key] = {"count": 0, "last_scan_ts": now_ts_conv}
                item["consecutive_buy_count"] = 0
        for sym_key in list(_CONSECUTIVE_BUY_TRACKER.keys()):
            if sym_key not in seen_buy_syms and (now_ts_conv - _CONSECUTIVE_BUY_TRACKER[sym_key].get("last_scan_ts", 0)) > 7200:
                del _CONSECUTIVE_BUY_TRACKER[sym_key]

        items.sort(key=lambda x: x["score"], reverse=True)
        items = items[:limit]
        total_count = len(items)
        paginated_items = items[offset:offset + limit]

    _feed_fallback_used = False
    if not paginated_items:
        try:
            feed_items = _get_explore_feed_items(
                horizon=h, limit=limit, signal_filter=signal_filter, market_type=market_type,
            )
            if not feed_items and signal_filter == "buy":
                feed_items = _get_explore_feed_items(
                    horizon=h, limit=limit, signal_filter="all", market_type=market_type,
                )
            if not feed_items and market_type != "all":
                feed_items = _get_explore_feed_items(
                    horizon=h, limit=limit, signal_filter="all", market_type="all",
                )
            if feed_items:
                paginated_items = feed_items[offset:offset + limit]
                total_count = len(feed_items)
                _feed_fallback_used = True
                status = "ready"
                reason = "explore_feed_fallback"
                message = "Showing explore feed signals (recommendation scan pending)"
                logger.info("recommendations fallback: returning %d items from explore_feed", len(paginated_items))
        except Exception as _fb_err:
            logger.debug("recommendations feed fallback error: %s", _fb_err)

    _result = {
        "ok": status == "ready",
        "status": status,
        "reason": reason,
        "message": message,
        "items": paginated_items,
        "count": len(paginated_items),
        "total_count": total_count,
        "offset": offset,
        "limit": limit,
        "has_more": (offset + len(paginated_items)) < total_count,
        "scan_age_sec": scan_age if last_scan > 0 else None,
        "last_scan_ts": last_scan if last_scan > 0 else None,
        "last_scan_by_horizon": last_scan_by_horizon,
        "macro_risk_off": macro_risk_off,
        "top30_whitelist_active": RECO_CRYPTO_TOP_30_ONLY and market_type in ("crypto", "all"),
        "cache_ts": int(time.time()),
        "fear_greed": {"value": _FEAR_GREED_CACHE.get("value", 50), "label": _FEAR_GREED_CACHE.get("label", "Neutral")},
        "btc_regime": _btc_regime_label if _btc_regime_label else None,
        "btc_inter_market_adj": _btc_inter_market_adj if _btc_inter_market_adj != 0 else None,
        "sector_momentum": _SECTOR_MOMENTUM if _SECTOR_MOMENTUM else None,
        "explore_rejected": explore_rejected_payload,
        "source": "explore_feed_fallback" if _feed_fallback_used else "recommendations",
        "data_source": "explore_feed_fallback" if _feed_fallback_used else reco_data_source,
    }
    _RECO_RESULT_CACHE[_cache_key] = (time.time(), _result)
    if len(_RECO_RESULT_CACHE) > 50:
        oldest_key = min(_RECO_RESULT_CACHE, key=lambda k: _RECO_RESULT_CACHE[k][0])
        _RECO_RESULT_CACHE.pop(oldest_key, None)
    return _json(_result)


@app.post("/api/recommendations/calibrate")
def api_recommendations_calibrate(window_days: int = 30):
    """Run adaptive scoring calibration from closed recommendation outcomes."""
    try:
        from recommendation_validator import run_calibration
        result = run_calibration(window_days=int(max(7, min(90, window_days))))
        return _json(result)
    except Exception as e:
        logger.exception("Calibration failed: %s", e)
        return _json({"ok": False, "error": str(e)}, 500)


@app.post("/api/ml/regime/train")
def api_ml_regime_train():
    """Train ML regime detector on historical data (Phase 2 Advanced)."""
    try:
        from ml_regime_detector import train_regime_detector_on_historical_data
        train_regime_detector_on_historical_data()
        return _json({"ok": True, "message": "ML regime detector trained"})
    except Exception as e:
        logger.exception("ML regime train failed: %s", e)
        return _json({"ok": False, "error": str(e)}, 500)


@app.post("/api/insider/fetch/{symbol}")
def api_insider_fetch(symbol: str, days_back: int = 90):
    """Fetch and store SEC Form 4 insider transactions for a stock (requires FINNHUB_API_KEY)."""
    sym = symbol.upper().split("/")[0]
    if sym in ("BTC", "ETH"):
        return _json({"ok": False, "error": "Insider data is for stocks only"}, 400)
    try:
        from insider_tracker import fetch_and_store_insider_transactions
        n = fetch_and_store_insider_transactions(sym, days_back=int(min(365, max(7, days_back))))
        return _json({"ok": True, "symbol": sym, "new_records": n})
    except Exception as e:
        logger.exception("Insider fetch failed: %s", e)
        return _json({"ok": False, "error": str(e)}, 500)


@app.get("/api/ml/performance")
def api_ml_performance(days: int = 30, symbol: Optional[str] = None):
    """ML model performance: accuracy by symbol, timeframe, precision, recall, F1."""
    try:
        from db import get_ml_model_accuracy, get_ml_predictions
        stats = get_ml_model_accuracy(days_back=int(min(365, max(7, days))))
        preds = get_ml_predictions(symbol=symbol, limit=500, days_back=days)
        by_symbol = {}
        for p in preds:
            sym = p.get("symbol", "?")
            if sym not in by_symbol:
                by_symbol[sym] = {"correct": 0, "total": 0}
            if p.get("actual_outcome_7d") is not None:
                pred_up = str(p.get("predicted_direction", "")).upper() == "UP"
                actual_up = float(p.get("actual_outcome_7d", 0)) > 0
                by_symbol[sym]["total"] += 1
                if pred_up == actual_up:
                    by_symbol[sym]["correct"] += 1
        by_symbol = {k: {"accuracy": v["correct"] / v["total"] if v["total"] > 0 else 0.5, "total": v["total"]} for k, v in by_symbol.items()}
        try:
            from ml_ensemble import get_ml_ensemble
            ensemble = get_ml_ensemble()
            status = ensemble.get_status()
            feature_imp = ensemble.get_feature_importance()
        except Exception:
            status = {}
            feature_imp = {}
        return _json({
            "ok": True,
            "days": days,
            "accuracy": round(stats["accuracy"], 4),
            "precision": round(stats["precision"], 4),
            "recall": round(stats["recall"], 4),
            "f1": round(stats["f1"], 4),
            "total_predictions": stats["total"],
            "by_symbol": by_symbol,
            "ensemble_status": status,
            "feature_importance": feature_imp,
        })
    except Exception as e:
        logger.exception("ML performance failed: %s", e)
        return _json({"ok": False, "error": str(e)}, 500)


@app.get("/api/ml/predictions")
def api_ml_predictions(symbol: Optional[str] = None, limit: int = 50, days_back: int = 30):
    """List recent ML predictions."""
    try:
        from db import get_ml_predictions
        rows = get_ml_predictions(symbol=symbol, limit=int(min(200, limit)), days_back=days_back)
        return _json({"ok": True, "predictions": rows, "count": len(rows)})
    except Exception as e:
        logger.exception("ML predictions list failed: %s", e)
        return _json({"ok": False, "error": str(e)}, 500)


@app.post("/api/ml/retrain")
def api_ml_retrain():
    """Trigger ML model retraining (walk-forward, deploy only if validation >60%)."""
    try:
        from ml_ensemble import get_ml_ensemble
        from db import save_ml_model_version
        import os
        ensemble = get_ml_ensemble()
        if len(ensemble._training_data) < 100:
            return _json({"ok": False, "error": "Insufficient training data (need 100+ samples)"}, 400)
        success = ensemble.train(force=True)
        if not success:
            return _json({"ok": False, "error": "Training failed"}, 500)
        min_acc = float(os.getenv("ML_MIN_ACCURACY", "0.60"))
        best_acc = max(
            ensemble._model_performance.get("xgb", type("O", (), {"recent_accuracy": 0})()).recent_accuracy,
            ensemble._model_performance.get("rf", type("O", (), {"recent_accuracy": 0})()).recent_accuracy,
        )
        deployed = best_acc >= min_acc
        version = f"v{int(time.time())}"
        save_ml_model_version("ensemble", version, best_acc, deployed=deployed)
        return _json({"ok": True, "validation_accuracy": best_acc, "deployed": deployed, "version": version})
    except Exception as e:
        logger.exception("ML retrain failed: %s", e)
        return _json({"ok": False, "error": str(e)}, 500)


@app.post("/api/ml/signal_scorer/retrain")
def api_ml_signal_scorer_retrain():
    """Trigger ML signal scorer retraining with recent trade feedback."""
    try:
        from ml_signal_scorer import MLSignalScorer
        from db import get_trade_feedback
        scorer = MLSignalScorer()
        feedback = get_trade_feedback(limit=5000)
        return _json({"status": "ok", "feedback_count": len(feedback), "message": "Retraining triggered"})
    except Exception as e:
        logger.exception("ML signal scorer retrain failed: %s", e)
        return _json({"status": "error", "message": str(e)}, 500)


@app.get("/api/ml/ensemble")
def api_ml_ensemble(symbol: str = "TRX/USD", timeframe: str = "1h"):
    """Ensemble signal scorer: combined ML + regime + tech + volume + pattern + momentum + screener."""
    try:
        from signal_ensemble import get_ensemble_report, get_weight_diagnostics
        bm_ref = globals().get("bm")
        candles_1h, candles_4h, candles_1d = [], [], []
        if bm_ref and hasattr(bm_ref, "kc"):
            try:
                candles_1h = bm_ref.kc.fetch_ohlcv(symbol, "1h", limit=200) or []
            except Exception:
                pass
            try:
                candles_4h = bm_ref.kc.fetch_ohlcv(symbol, "4h", limit=100) or []
            except Exception:
                pass
            try:
                candles_1d = bm_ref.kc.fetch_ohlcv(symbol, "1d", limit=50) or []
            except Exception:
                pass
        regime = "RANGE"
        try:
            from strategies import detect_regime
            if candles_1h:
                rr = detect_regime(candles_1h, candles_4h or candles_1h, candles_1d or candles_1h)
                regime = str(rr.regime.value if hasattr(rr.regime, "value") else rr.regime)
        except Exception:
            pass
        report = get_ensemble_report(candles_1h, regime=regime, candles_4h=candles_4h,
                                     candles_1d=candles_1d, symbol=symbol)
        return _json(report)
    except Exception as e:
        logger.exception("Ensemble API error: %s", e)
        return _json({"error": str(e)}, 500)


@app.get("/api/portfolio/capital")
def api_portfolio_capital():
    """Portfolio-level capital management: total, reserve, allocation, heat map, CAGR, leverage."""
    try:
        from db import list_bots, list_all_deals, get_bot_recent_streak, bot_performance_stats, bot_deal_stats, all_deal_stats
        from portfolio_manager import (
            compute_cash_reserve,
            get_portfolio_heat_map_data,
            get_portfolio_cagr,
            check_leverage,
        )
        from capital_allocator import get_allocation_mult, AUTO_SCALE_ENABLED
        bots = list_bots()
        deals = list_all_deals(state="OPEN", limit=500)
        portfolio_total = 0.0
        try:
            bm_ref = globals().get("bm")
            if bm_ref and hasattr(bm_ref, "get_portfolio_total"):
                portfolio_total = float(bm_ref.get_portfolio_total())
        except Exception:
            pass
        if portfolio_total <= 0:
            try:
                all_stats = all_deal_stats()
                realized = float(all_stats.get("realized_total", 0))
                portfolio_total = max(1000.0, 1000.0 + realized)
            except Exception:
                portfolio_total = 1000.0
        reserve = compute_cash_reserve(portfolio_total)
        heat_map = get_portfolio_heat_map_data(bots, deals, portfolio_total)
        cagr = get_portfolio_cagr(365)
        leverage_info = check_leverage(portfolio_total, 0)
        per_bot = []
        for b in bots:
            bid = b.get("id")
            if not bid:
                continue
            streak = get_bot_recent_streak(bid, 5)
            perf = bot_performance_stats(bid)
            stats = bot_deal_stats(bid)
            mult = get_allocation_mult(streak, float(perf.get("win_rate", 0.5)), float(stats.get("realized_total", 0)))
            per_bot.append({
                "bot_id": bid,
                "symbol": b.get("symbol"),
                "streak": streak,
                "allocation_mult": mult,
                "realized_total": stats.get("realized_total"),
            })
        return _json({
            "ok": True,
            "portfolio_total": round(portfolio_total, 2),
            "cash_reserve": round(reserve, 2),
            "reserve_pct": round(reserve / portfolio_total * 100, 1) if portfolio_total > 0 else 20,
            "heat_map": heat_map,
            "cagr_1y": round(cagr * 100, 2) if cagr is not None else None,
            "leverage": leverage_info,
            "per_bot_allocation": per_bot,
            "auto_scale_enabled": AUTO_SCALE_ENABLED,
        })
    except Exception as e:
        logger.exception("Portfolio capital API failed: %s", e)
        return _json({"ok": False, "error": str(e)}, 500)


@app.get("/api/execution/quality")
def api_execution_quality(days: int = 30):
    """Return execution quality stats: avg slippage by symbol and by strategy."""
    try:
        from execution_quality_tracker import get_avg_slippage_by_symbol, get_avg_slippage_by_strategy
        by_symbol = get_avg_slippage_by_symbol(days=int(days))
        by_strategy = get_avg_slippage_by_strategy(days=int(days))
        return _json({
            "ok": True,
            "days": int(days),
            "by_symbol": by_symbol,
            "by_strategy": by_strategy,
        })
    except Exception as e:
        logger.exception("Execution quality API failed: %s", e)
        return _json({"ok": False, "error": str(e)}, 500)


@app.get("/explore/fear-greed")
def explore_fear_greed():
    """Return current Fear & Greed index value."""
    fg = _FEAR_GREED_CACHE
    return _json({"ok": True, "value": fg.get("value", 50), "label": fg.get("label", "Neutral"), "score_adj": _fear_greed_score_adjustment(), "updated_ts": int(fg.get("ts", 0)), "error": fg.get("error")})


@app.get("/explore/signal_audit")
def explore_signal_audit(symbol: str = "", grade: str = "", limit: int = 50, days: int = 7):
    """Query hybrid screener signal audit trail."""
    try:
        from db import list_signal_audits
        since_ts = int(time.time()) - (int(days) * 86400)
        audits = list_signal_audits(
            symbol=symbol.strip(),
            conviction_grade=grade.strip().upper() if grade else "",
            limit=min(200, int(limit)),
            since_ts=since_ts,
        )
        for a in audits:
            for json_key in ("factor_scores_json", "gate_results_json",
                             "technical_signals_json", "metadata_json", "flags_json"):
                raw = a.get(json_key)
                if raw:
                    try:
                        a[json_key.replace("_json", "")] = json.loads(raw)
                    except (json.JSONDecodeError, TypeError):
                        pass
        return _json({"ok": True, "audits": audits, "count": len(audits)})
    except Exception as e:
        logger.exception("Signal audit query failed: %s", e)
        return _json({"ok": False, "error": str(e)}, 500)


@app.get("/explore/accuracy")
def explore_accuracy(days: int = 30):
    """Explore page accuracy stats: signal-level 24h/72h win rates + bot-based.

    Wraps the DB calls so that transient SQLite lock contention surfaces as a
    clean 503 (which the UI handles as "Accuracy data temporarily unavailable")
    instead of a 500 with stack trace.
    """
    try:
        from db import get_recommendation_performance_stats, get_signal_accuracy_stats
        bot_stats = get_recommendation_performance_stats(days=int(max(7, min(365, days))))
        sig_stats = get_signal_accuracy_stats(days=int(max(7, min(365, days))))
        return _json({
            "ok": True,
            "total_signals": bot_stats["total_closed"],
            "wins": bot_stats["wins"],
            "losses": bot_stats["losses"],
            "win_rate_pct": round(bot_stats["win_rate"], 1),
            "by_score_range": bot_stats.get("by_score_range", []),
            "enough_data": bot_stats["total_closed"] >= 10 or sig_stats["total_tracked"] >= 10,
            "signal_accuracy": sig_stats,
        })
    except sqlite3.OperationalError as exc:
        msg = str(exc).lower()
        if "locked" in msg or "busy" in msg:
            logger.warning("/explore/accuracy: DB locked, returning 503: %s", exc)
            return _json({"ok": False, "error": "database busy", "transient": True}, 503)
        logger.exception("/explore/accuracy: DB error")
        return _json({"ok": False, "error": str(exc)}, 503)
    except Exception as exc:
        logger.exception("/explore/accuracy failed")
        return _json({"ok": False, "error": str(exc), "transient": True}, 503)


@app.get("/explore/accuracy/symbols")
def explore_accuracy_symbols(symbols: str = "", days: int = 90):
    """Per-symbol win rates for explore cards."""
    from db import get_per_symbol_accuracy
    sym_list = [s.strip() for s in symbols.split(",") if s.strip()] if symbols else []
    if not sym_list:
        return _json({"ok": True, "symbols": {}})
    result = get_per_symbol_accuracy(sym_list[:100], days=int(max(7, min(365, days))))
    return _json({"ok": True, "symbols": result})


@app.get("/api/recommendations/performance")
def api_recommendations_performance(days: int = 30):
    """Return recommendation accuracy metrics: win rate, avg profit, performance by score range, regime, and alpha by strategy."""
    stats = get_recommendation_performance_stats(days=int(max(7, min(365, days))))
    alpha_by_strategy = {}
    try:
        from benchmark_analyzer import get_alpha_by_strategy
        alpha_by_strategy = get_alpha_by_strategy(days=int(days))
    except Exception:
        pass
    return _json({
        "ok": True,
        "days": int(days),
        "total_closed": stats["total_closed"],
        "wins": stats["wins"],
        "losses": stats["losses"],
        "win_rate_pct": round(stats["win_rate"], 1),
        "avg_profit_per_recommendation": stats["avg_profit_per_recommendation"],
        "by_score_range": stats["by_score_range"],
        "by_regime": stats["by_regime"],
        "alpha_by_strategy": alpha_by_strategy,
    })


@app.get("/api/recommendations/{symbol}")
def api_recommendation_symbol(symbol: str, horizon: str = "short"):
    """Get recommendation for a specific symbol. Returns single item."""
    from urllib.parse import unquote
    sym = unquote(symbol)
    if "/" in sym and (sym.split("/")[0] or "").upper() in FIAT_BASES:
        return _json({"ok": False, "error": "Fiat FX pairs are excluded"}, 404)
    if "/" in sym and (sym.split("/")[0] or "").upper() in CRYPTO_BLOCKLIST:
        return _json({"ok": False, "error": "Symbol is blocklisted"}, 404)
    h = "long" if str(horizon).lower().startswith("l") else "short"
    row = get_recommendation(_resolve_symbol(sym), h)
    if not row:
        return _json({"ok": False, "error": "No recommendation found"}, 404)
    
    metrics = json.loads(row.get("metrics_json") or "{}")
    _regime_raw = json.loads(row.get("regime_json") or "{}")
    # Normalize flat format {"label": "X"} → nested {"1d": {"label": "X"}, "4h": {"label": "X"}}
    if isinstance(_regime_raw, dict) and "label" in _regime_raw and "1d" not in _regime_raw:
        regime = {"1d": _regime_raw, "4h": _regime_raw}
    else:
        regime = _regime_raw
    reasons = json.loads(row.get("reasons_json") or "[]")
    risk_flags = json.loads(row.get("risk_flags_json") or "[]")
    
    item_market = (metrics.get("market_type") or "").strip().lower()
    if not item_market:
        item_market = "stocks" if (len(sym) < 6 and "/" not in sym) else "crypto"
    if item_market == "stock":
        item_market = "stocks"
    
    ticker = {}
    price_from_metrics = metrics.get("price")
    if price_from_metrics and price_from_metrics > 0:
        ticker = {"last": price_from_metrics}
    else:
        if item_market == "stocks":
            ticker = _ticker_cached(sym, ttl_sec=120) or {}
        else:
            ticker = _ticker_cached(sym, ttl_sec=120) or {}
    
    score = float(row.get("score") or 0.0)
    _d_conv = row.get("conviction_grade") or metrics.get("conviction_grade")
    if _d_conv == "A": rating = "Strong Buy"
    elif _d_conv == "B": rating = "Buy"
    elif _d_conv == "C": rating = "Watch"
    elif _d_conv == "D": rating = "Avoid"
    elif score >= 85: rating = "Strong Buy"
    elif score >= 55: rating = "Buy"
    elif score >= 40: rating = "Watch"
    else: rating = "Avoid"

    _d_factor_scores = {}
    try:
        _fs = row.get("factor_scores_json")
        if _fs:
            _d_factor_scores = json.loads(_fs)
    except (json.JSONDecodeError, TypeError):
        pass
    if not _d_factor_scores:
        _d_factor_scores = metrics.get("factor_scores") or {}

    return _json({
        "ok": True,
        "item": {
            "symbol": sym,
            "score": score,
            "horizon": h,
            "market_type": item_market,
            "price": float(metrics.get("price") or ticker.get("last") or 0.0) if (metrics.get("price") or ticker.get("last")) else None,
            "change_pct": float(ticker.get("percentage") or 0.0),
            "volume": float(ticker.get("quoteVolume") or 0.0),
            "rating": rating,
            "confidence": float(row.get("confidence_score") or metrics.get("confidence_score") or 0.0),
            "composite_score": float(row.get("composite_score") or metrics.get("composite_score") or 0.0),
            "confidence_score": float(row.get("confidence_score") or metrics.get("confidence_score") or 0.0),
            "conviction_grade": _d_conv,
            "factor_scores": _d_factor_scores,
            "regime_1d": (regime.get("1d") or {}).get("label"),
            "regime_4h": (regime.get("4h") or {}).get("label"),
            "weekly_trend": metrics.get("weekly_trend"),
            "strategy_mode": metrics.get("strategy") or metrics.get("recommended_strategy") or "smart_dca",
            "suggested_strategy": _strategy_display_name(metrics.get("strategy") or metrics.get("recommended_strategy")),
            "recommended_strategy": _strategy_display_name(metrics.get("strategy") or metrics.get("recommended_strategy")),
            "volatility": metrics.get("atr_pct"),
            "risk_flags": risk_flags,
            "updated_ts": int(row.get("created_ts") or 0),
            "eligible": bool(metrics.get("eligible")),
            "research_only": bool(metrics.get("research_only")) or (item_market == "stocks" and not _alpaca_any_ready()),
            "reasons": reasons,
            "benchmark_vs": metrics.get("benchmark_vs") or None,
            "peer_rank": metrics.get("peer_rank") or None,
            "beta": metrics.get("beta"),
            "rsi_2": metrics.get("rsi_2"),
            "price_to_52wk_high": metrics.get("price_to_52wk_high"),
            "rsi2_entry_signal": metrics.get("rsi2_entry_signal"),
            "macd_combo_entry": metrics.get("macd_combo_entry"),
            "hybrid_flags": metrics.get("hybrid_flags") or [],
        }
    })


@app.get("/api/diagnostics")
def api_diagnostics():
    """Get system diagnostics: client status, scan state, recommendation counts."""
    now = int(time.time())
    uptime_sec = int(time.time() - _APP_START_TIME)
    
    # Kraken status
    kraken_ready = _kraken_ready()
    kraken_error = KRAKEN_ERROR or None
    
    # Alpaca status
    alpaca_ready = _alpaca_any_ready()
    alpaca_error = None
    if not alpaca_ready:
        if not os.getenv("ALPACA_API_KEY_PAPER") and not os.getenv("ALPACA_API_KEY_LIVE"):
            alpaca_error = "No Alpaca API keys configured"
        else:
            alpaca_error = "Alpaca client initialization failed"
    
    with _globals_lock:
        short_state = (_RECO_STATE.get("short") or {}).copy()
        medium_state = (_RECO_STATE.get("medium") or {}).copy()
        long_state = (_RECO_STATE.get("long") or {}).copy()

    # Count recommendations in DB
    counts = count_recommendations_by_horizon()
    short_count = counts.get("short", 0)
    medium_count = counts.get("medium", 0)
    long_count = counts.get("long", 0)

    def _scan_info(state: dict) -> dict:
        ts = state.get("last_run_ts")
        return {
            "count": 0,  # filled below
            "last_scan_ts": ts,
            "last_scan_age_sec": now - ts if ts else None,
            "last_error": state.get("error"),
            "scanned": state.get("scanned", 0),
            "eligible": state.get("eligible", 0),
        }

    short_info = _scan_info(short_state)
    short_info["count"] = short_count
    medium_info = _scan_info(medium_state)
    medium_info["count"] = medium_count
    long_info = _scan_info(long_state)
    long_info["count"] = long_count

    return _json({
        "ok": True,
        "uptime_sec": uptime_sec,
        "kraken": {"ready": kraken_ready, "error": kraken_error},
        "alpaca": {"ready": alpaca_ready, "error": alpaca_error},
        "recommendations": {
            "short": short_info,
            "medium": medium_info,
            "long": long_info,
        },
        "timestamp": now
    })


@app.get("/api/logs")
def api_logs(lines: int = 200, service: str = "tradingserver"):
    """Return last N lines of service logs (journalctl). For diagnostics."""
    import subprocess
    try:
        out = subprocess.run(
            ["journalctl", "-u", service, f"-n{min(lines, 2000)}", "--no-pager", "-o", "short-iso"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        if out.returncode == 0 and out.stdout:
            return JSONResponse(
                content={"ok": True, "service": service, "lines": out.stdout.strip().split("\n")[-min(lines, 2000):]},
                headers={"Cache-Control": "no-store"},
            )
        return _json({"ok": False, "error": out.stderr or "journalctl failed"}, 500)
    except FileNotFoundError:
        return _json({"ok": False, "error": "journalctl not available"}, 501)
    except subprocess.TimeoutExpired:
        return _json({"ok": False, "error": "journalctl timed out"}, 504)
    except Exception as e:
        logger.exception("api_logs failed")
        return _json({"ok": False, "error": str(e)[:200]}, 500)


@app.get("/api/activity")
def api_activity(limit: int = 20):
    """Get recent bot activity for the dashboard live feed.
    Returns recent logs from all bots, combining them into a single timeline."""
    try:
        from db import list_all_deals

        limit = min(500, max(1, int(limit)))

        # Get recent deals (trades) from all bots
        all_deals = list_all_deals(limit=limit*2)

        activities = []

        for deal in all_deals:
            if not deal or deal.get("state") not in ("open", "closed"):
                continue

            try:
                bot = get_bot(deal.get("bot_id"))
                if not bot:
                    continue

                action = "BUY"
                details = ""
                deal_type = "trade"
                timestamp = deal.get("opened_at") or now_ts()

                entry_price = deal.get("entry_avg") or 0
                base_amt = deal.get("base_amount") or 0
                realized_pnl = deal.get("realized_pnl_quote") or 0

                if entry_price > 0 and base_amt > 0:
                    details = f"Bought {base_amt:.4f} {deal.get('symbol')} at {entry_price:.2f}"
                    if deal.get("state") == "closed":
                        exit_price = deal.get("exit_avg") or entry_price
                        pnl_pct = ((exit_price - entry_price) / entry_price * 100) if entry_price > 0 else 0
                        action = "TAKE_PROFIT" if realized_pnl > 0 else "LOSS"
                        details = f"Sold {base_amt:.4f} {deal.get('symbol')} at {exit_price:.2f} ({pnl_pct:+.2f}%)"
                        deal_type = "profit" if realized_pnl > 0 else "loss"
                        timestamp = deal.get("closed_at") or timestamp

                activities.append({
                    "timestamp": timestamp,
                    "bot_name": bot.get("name") or f"Bot {bot.get('id')}",
                    "bot_id": bot.get("id"),
                    "symbol": deal.get("symbol"),
                    "action": action,
                    "details": details or f"Deal on {deal.get('symbol')}",
                    "type": deal_type,
                    "pnl": realized_pnl
                })
            except Exception as e:
                logger.debug("Activity feed deal processing error: %s", e)
                continue

        # Sort by timestamp descending (newest first)
        activities.sort(key=lambda x: x.get("timestamp", 0), reverse=True)

        return _json({
            "ok": True,
            "activities": activities[:limit],
            "total": len(activities)
        })
    except Exception as e:
        logger.exception("api_activity failed")
        return _json({"ok": False, "error": str(e), "activities": []}, 500)


@app.post("/api/recommendations/{symbol}/create_bot")
async def api_recommendation_create_bot(symbol: str, request: Request):
    payload = await request.json()
    horizon = str(payload.get("horizon") or "short")
    if "/" in symbol and (symbol.split("/")[0] or "").upper() in CRYPTO_BLOCKLIST:
        return _json({"ok": False, "error": "Symbol is blocklisted"}, 400)
    h = "long" if str(horizon).lower().startswith("l") else ("medium" if str(horizon).lower().startswith("m") else "short")
    row = get_recommendation(_resolve_symbol(symbol), h)
    if not row:
        return _json({"ok": False, "error": "No recommendation found"}, 404)
    metrics = json.loads(row.get("metrics_json") or "{}")
    strategy = str(metrics.get("strategy") or metrics.get("recommended_strategy") or "smart_dca")
    # Derive market_type from recommendation so stock bots use Alpaca, crypto use Kraken
    item_market = (metrics.get("market_type") or "").strip().lower()
    if item_market == "stock":
        item_market = "stocks"
    if not item_market:
        item_market = "stocks" if (len(str(symbol)) < 6 and "/" not in str(symbol)) else "crypto"
    name = str(payload.get("name") or f"Reco {symbol} {horizon.upper()}")
    enabled = int(bool(payload.get("enabled", False)))
    dry_run = int(bool(payload.get("dry_run", True)))
    auto_restart = int(bool(payload.get("auto_restart", True)))
    start_now = bool(payload.get("start_now", False))

    sym_resolved = str(row.get("symbol") or symbol)
    bot_id = create_bot(
        {
            "name": name,
            "symbol": sym_resolved,
            "enabled": enabled,
            "dry_run": dry_run,
            "strategy_mode": strategy,
            "forced_strategy": "",
            "auto_restart": auto_restart,
            "market_type": item_market,
            "alpaca_mode": str(payload.get("alpaca_mode") or "paper"),
            "base_quote": float(payload.get("base_quote") or 25.0),
            "safety_quote": float(payload.get("safety_quote") or 25.0),
            "max_safety": int(payload.get("max_safety") or 3),
            "first_dev": float(payload.get("first_dev") or 0.015),
            "step_mult": float(payload.get("step_mult") or 1.2),
            "tp": float(payload.get("tp") or 0.012),
            "trend_filter": int(payload.get("trend_filter") or 0),
            "trend_sma": int(payload.get("trend_sma") or 200),
            "max_spend_quote": float(payload.get("max_spend_quote") or 250.0),
            "poll_seconds": int(payload.get("poll_seconds") or 10),
            "max_open_orders": int(payload.get("max_open_orders") or 6),
            "max_total_exposure_pct": float(payload.get("max_total_exposure_pct") or 0.50),
            "per_symbol_exposure_pct": float(payload.get("per_symbol_exposure_pct") or (0.05 if horizon.startswith("l") else 0.1)),
            "min_free_cash_pct": float(payload.get("min_free_cash_pct") or 0.2),
            "max_concurrent_deals": int(payload.get("max_concurrent_deals") or 4),
            "spread_guard_pct": float(payload.get("spread_guard_pct") or 0.004),
            "limit_timeout_sec": int(payload.get("limit_timeout_sec") or 45),
            "daily_loss_limit_pct": float(payload.get("daily_loss_limit_pct") or 0.05),
            "pause_hours": int(payload.get("pause_hours") or 6),
        }
    )

    # Link recommendation to bot for performance tracking
    try:
        regime_obj = json.loads(row.get("regime_json") or "{}")
        regime_name = str(regime_obj.get("regime") or regime_obj.get("name") or metrics.get("regime") or "")
        link_recommendation_to_bot(
            bot_id=int(bot_id),
            symbol=sym_resolved,
            recommendation_date=int(row.get("created_ts") or 0),
            score_at_recommendation=float(row.get("score") or 0),
            regime_at_recommendation=regime_name,
            metrics_json=row.get("metrics_json") or "{}",
            reasons_json=row.get("reasons_json") or "[]",
            snapshot_id=int(row["id"]) if row.get("id") else None,
        )
    except Exception as e:
        logger.warning("link_recommendation_to_bot failed for bot_id=%s: %s", bot_id, e)

    if start_now and bm is not None:
        try:
            bot = get_bot(int(bot_id))
            ok, reason = _can_start_bot_live(bot or {})
            if ok:
                bm.start(int(bot_id))
            else:
                logger.warning(
                    "api_recommendation_create_bot: start_now blocked bot_id=%s reason=%s",
                    bot_id,
                    reason,
                )
        except Exception:
            logger.exception("api_recommendation_create_bot: start_now failed bot_id=%s", bot_id)

    return _json({"ok": True, "bot_id": int(bot_id)})


@app.post("/api/recommendations/scan")
def api_recommendations_scan(horizon: str = "short"):
    """Trigger a manual scan. Supports horizon=all for sequential multi-horizon."""
    import threading
    _h = str(horizon).lower().strip()
    logger.warning("[SCAN-DEBUG] scan requested for horizon=%s", _h)

    if _h == "all":
        horizon_list = ["short", "medium", "long"]
    else:
        horizon_list = ["long" if _h.startswith("l") else ("medium" if _h.startswith("m") else "short")]

    if not _RECO_SCAN_SEM.acquire(blocking=False):
        return _json({"ok": False, "error": "Scan already in progress"}, status_code=409)

    global _RECO_SCAN_ACTIVE
    with _RECO_SCAN_ACTIVE_LOCK:
        _RECO_SCAN_ACTIVE = True

    def _scan_async():
        global _RECO_SCAN_ACTIVE
        try:
            for h in horizon_list:
                try:
                    n = delete_recommendations_for_blocklist(list(CRYPTO_BLOCKLIST))
                    if n > 0:
                        logger.warning("Purged %d blocklisted recommendation(s) before %s scan", n, h)
                    logger.warning("[SCAN-DEBUG] starting %s horizon scan (manual/rescan)", h)
                    _scan_recommendations_impl(h)
                    with _globals_lock:
                        state = (_RECO_STATE.get(h) or {}).copy()
                    logger.warning(
                        "[SCAN-DEBUG] %s horizon finished: scanned=%d eligible=%d",
                        h, state.get("scanned", 0), state.get("eligible", 0),
                    )
                except BaseException as e:
                    logger.error("[SCAN] %s scan failed: %s", h, e, exc_info=True)
                    with _globals_lock:
                        _RECO_STATE[h] = {
                            "last_run_ts": int(time.time()), "error": str(e)[:200],
                            "scanned": 0, "eligible": 0, "total": 0, "scanning": False,
                        }
                    if not isinstance(e, Exception):
                        raise  # re-raise KeyboardInterrupt / SystemExit
                time.sleep(2.0)
            # After any scan, refresh backtests for scanned horizons in background
            if horizon_list:
                _bt_horizons = tuple(horizon_list)
                threading.Thread(
                    target=_run_missing_backtests,
                    kwargs={"horizons": _bt_horizons, "force": True},
                    daemon=True, name="postscan_backtest",
                ).start()
        finally:
            with _RECO_SCAN_ACTIVE_LOCK:
                _RECO_SCAN_ACTIVE = False
            _RECO_SCAN_SEM.release()

    threading.Thread(target=_scan_async, daemon=True).start()
    return _json({"ok": True, "message": f"Scan triggered for {', '.join(horizon_list)} horizon(s)"})


def _try_init_bot_manager() -> bool:
    """Lazy-init BotManager if bm is None and we have at least one client. Returns True if bm is now ready."""
    global bm
    if bm is not None:
        return True
    if not (kc or alpaca_paper or alpaca_live):
        return False
    try:
        with _globals_lock:
            if bm is not None:
                return True
            bm = BotManager(kc, alpaca_paper, alpaca_live)
            logger.info("BotManager lazy-init OK (Crypto: %s, Alpaca paper: %s, Alpaca live: %s)",
                       KRAKEN_READY, ALPACA_PAPER_READY, ALPACA_LIVE_READY)
            return True
    except Exception as e:
        logger.warning("BotManager lazy-init failed: %s", e)
        return False


@app.post("/api/bots/{bot_id}/start")
def api_bot_start(bot_id: int):
    b = get_bot(int(bot_id))
    if not b:
        return _json({"ok": False, "error": "Bot not found"}, 404)
    try:
        from circuit_breaker import is_emergency_stop_active, is_bot_circuit_open, get_bot_pause_until
        if is_emergency_stop_active():
            return _json({"ok": False, "error": "Emergency stop active. Exchange errors persist. Check /api/health."}, 503)
        if is_bot_circuit_open(int(bot_id)):
            until = get_bot_pause_until(int(bot_id))
            return _json({"ok": False, "error": f"Circuit breaker: bot paused until errors clear (until ts {until})"}, 503)
    except ImportError:
        pass
    try:
        from data_validator import is_data_quality_degraded
        if is_data_quality_degraded():
            return _json({"ok": False, "error": "Data quality degraded (5+ issues in 15 min). Trading paused."}, 503)
    except ImportError:
        pass
    if bm is None:
        if _try_init_bot_manager():
            pass  # bm now set, continue
        else:
            return _json({"ok": False, "error": "Worker not initialized. Check Kraken/Alpaca API keys in .env and restart."}, 503)

    ok, reason = _can_start_bot_live(b)
    if not ok:
        logger.warning("api_bot_start blocked bot_id=%s reason=%s", bot_id, reason)
        return _json({"ok": False, "error": reason}, 503)

    msg = bm.start(int(bot_id))
    snap = bm.snapshot(int(bot_id))
    return _json({"ok": True, "message": msg, "snap": snap})


@app.post("/api/bots/{bot_id}/clear_risk_flag")
def api_bot_clear_risk_flag(bot_id: int, request: Request):
    """Clear a CRITICAL risk flag after user acknowledgment."""
    b = get_bot(int(bot_id))
    if not b:
        return _json({"ok": False, "error": "Bot not found"}, 404)
    if bm is None:
        return _json({"ok": False, "error": "Worker not initialized"}, 503)
    snap = bm.snapshot(int(bot_id))
    if not snap or snap.get("risk_level") != "CRITICAL":
        return _json({"ok": True, "message": "No CRITICAL risk flag to clear"})
    runner = bm._bots.get(int(bot_id))
    if runner:
        with runner._lock:
            runner.state.risk_state = None
            runner.state.risk_level = "OK"
            runner.state.risk_reason = None
            runner.state.risk_since_ts = 0
        logger.info("Risk flag cleared for bot %d by user", bot_id)
    return _json({"ok": True, "message": "Risk flag cleared — bot will resume on next cycle"})


@app.post("/api/bots/{bot_id}/stop")
def api_bot_stop(bot_id: int):
    b = get_bot(int(bot_id))
    if not b:
        return _json({"ok": False, "error": "Bot not found"}, 404)
    if bm is None:
        if not _try_init_bot_manager():
            return _json({"ok": False, "error": "Worker not initialized. Check Kraken/Alpaca API keys in .env and restart."}, 503)

    msg = bm.stop(int(bot_id))
    snap = bm.snapshot(int(bot_id))
    return _json({"ok": True, "message": msg, "snap": snap})


@app.get("/api/validate-keys")
def api_validate_keys():
    """Pre-live key validation. Calls fetch_balance() on Kraken (and Alpaca if
    configured) so the UI can confirm the API keys are not just *present* but
    actually authorised against the exchange.

    Returns:
        ok=True if at least one configured exchange returned a non-empty balance.
        per-exchange: status (ok|error|not_configured), error message, sample
                      balances (asset -> total) for the top 5 by total value.
    """
    out: Dict[str, Any] = {"ok": False, "checked_at": int(time.time()), "exchanges": {}}

    # --- Kraken ---
    if os.getenv("KRAKEN_API_KEY") and os.getenv("KRAKEN_API_SECRET"):
        kr_info: Dict[str, Any] = {"configured": True}
        if kc is None:
            kr_info.update({"status": "error", "error": "KrakenClient not initialised. Check .env keys + restart."})
        else:
            try:
                bal = kc.fetch_balance() or {}
                totals = (bal.get("total") or {}) if isinstance(bal, dict) else {}
                # Filter out zero balances; sort by amount descending; take top 5.
                non_zero = {k: float(v) for k, v in totals.items()
                            if v not in (None, 0) and isinstance(v, (int, float))}
                top = dict(sorted(non_zero.items(), key=lambda kv: -kv[1])[:5])
                kr_info.update({
                    "status": "ok",
                    "asset_count": len(non_zero),
                    "top_balances": top,
                })
                out["ok"] = True
            except Exception as e:
                kr_info.update({"status": "error", "error": f"{type(e).__name__}: {e}"})
        out["exchanges"]["kraken"] = kr_info
    else:
        out["exchanges"]["kraken"] = {"configured": False, "status": "not_configured"}

    # --- Alpaca paper ---
    if os.getenv("ALPACA_API_KEY_PAPER") and os.getenv("ALPACA_API_SECRET_PAPER"):
        ap_info: Dict[str, Any] = {"configured": True}
        if alpaca_paper is None:
            ap_info.update({"status": "error", "error": "AlpacaClient (paper) not initialised."})
        else:
            try:
                acct = alpaca_paper.get_account() if hasattr(alpaca_paper, "get_account") else {}
                if isinstance(acct, dict) and (acct.get("buying_power") is not None or acct.get("cash") is not None):
                    ap_info.update({
                        "status": "ok",
                        "buying_power": float(acct.get("buying_power", 0) or 0),
                        "cash": float(acct.get("cash", 0) or 0),
                        "equity": float(acct.get("equity", 0) or 0),
                    })
                    out["ok"] = True
                else:
                    ap_info.update({"status": "error", "error": "Empty account response"})
            except Exception as e:
                ap_info.update({"status": "error", "error": f"{type(e).__name__}: {e}"})
        out["exchanges"]["alpaca_paper"] = ap_info
    else:
        out["exchanges"]["alpaca_paper"] = {"configured": False, "status": "not_configured"}

    # --- Alpaca live ---
    if os.getenv("ALPACA_API_KEY_LIVE") and os.getenv("ALPACA_API_SECRET_LIVE"):
        al_info: Dict[str, Any] = {"configured": True}
        if alpaca_live is None:
            al_info.update({"status": "error", "error": "AlpacaClient (live) not initialised."})
        else:
            try:
                acct = alpaca_live.get_account() if hasattr(alpaca_live, "get_account") else {}
                if isinstance(acct, dict) and (acct.get("buying_power") is not None or acct.get("cash") is not None):
                    al_info.update({
                        "status": "ok",
                        "buying_power": float(acct.get("buying_power", 0) or 0),
                        "cash": float(acct.get("cash", 0) or 0),
                        "equity": float(acct.get("equity", 0) or 0),
                    })
                    out["ok"] = True
                else:
                    al_info.update({"status": "error", "error": "Empty account response"})
            except Exception as e:
                al_info.update({"status": "error", "error": f"{type(e).__name__}: {e}"})
        out["exchanges"]["alpaca_live"] = al_info
    else:
        out["exchanges"]["alpaca_live"] = {"configured": False, "status": "not_configured"}

    return _json(out)


@app.post("/api/bots/{bot_id}/test-order")
async def api_bot_test_order(bot_id: int, request: Request):
    """Place a tiny limit order well below market and immediately cancel it.

    This proves the full order-placement path works end-to-end without risking
    an actual fill. The order is placed at 50 % below the current ask (limit
    buy) so it cannot fill on a normal book; we then cancel it within seconds.

    Body params (optional): {"quote_amount": 5.0, "symbol": "<override>"}
    """
    b = get_bot(int(bot_id))
    if not b:
        return _json({"ok": False, "error": "Bot not found"}, 404)
    if bool(b.get("dry_run", 1)):
        return _json({"ok": False, "error": "Bot is in dry-run mode; switch to live first."}, 400)
    if not ALLOW_LIVE_TRADING:
        return _json({"ok": False, "error": "Live trading disabled. Set ALLOW_LIVE_TRADING=1 in .env."}, 403)

    try:
        payload = await request.json()
    except Exception:
        payload = {}
    quote_amount = float(payload.get("quote_amount", 5.0) or 5.0)
    sym_override = str(payload.get("symbol") or "").strip()

    client, is_kraken = _get_bot_client(b)
    if not client:
        return _json({"ok": False, "error": "Trading client not available"}, 503)
    if is_kraken and not _kraken_ready():
        return _json({"ok": False, "error": KRAKEN_ERROR or "Kraken not ready"}, 503)

    symbol = _resolve_symbol(sym_override or b.get("symbol", ""))
    placed: Optional[Dict[str, Any]] = None
    cancelled: Optional[Dict[str, Any]] = None
    error: Optional[str] = None

    try:
        tk = client.fetch_ticker(symbol) if hasattr(client, "fetch_ticker") else {}
        last = float(tk.get("last") or tk.get("close") or 0)
        bid = float(tk.get("bid") or 0)
        if last <= 0 and bid <= 0:
            return _json({"ok": False, "error": f"No reference price for {symbol}"}, 503)
        ref = last or bid
        # 50% below market — well outside the book, so it will rest unfilled.
        limit_price = round(ref * 0.5, 8)
        base_amount = max(quote_amount / max(ref, 1e-9), 0.0)
        if base_amount <= 0:
            return _json({"ok": False, "error": "computed zero base amount"}, 400)

        if hasattr(client, "create_limit_buy_base"):
            placed = client.create_limit_buy_base(symbol, float(base_amount), float(limit_price))
        elif hasattr(client, "create_order"):
            placed = client.create_order(symbol, "limit", "buy", float(base_amount), float(limit_price))
        else:
            return _json({"ok": False, "error": "Client does not support limit buy"}, 501)

        order_id = str((placed or {}).get("id") or "")
        # Brief pause then cancel — confirms cancel path also works.
        time.sleep(1.0)
        if order_id:
            try:
                cancelled = client.cancel_order(order_id, symbol)
            except Exception as ce:
                error = f"cancel failed: {type(ce).__name__}: {ce}"
        else:
            error = "no order id in placement response"

        return _json({
            "ok": error is None,
            "symbol": symbol,
            "ref_price": ref,
            "limit_price": limit_price,
            "base_amount": base_amount,
            "quote_amount": quote_amount,
            "placed": placed,
            "cancelled": bool(cancelled is not None and not error),
            "cancel_response": cancelled,
            "error": error,
        })
    except Exception as e:
        logger.exception("test-order failed bot=%s symbol=%s", bot_id, symbol)
        return _json({
            "ok": False,
            "error": f"{type(e).__name__}: {e}",
            "symbol": symbol,
            "placed": placed,
        }, 500)


@app.post("/api/bots/{bot_id}/reset-error")
def api_bot_reset_error(bot_id: int):
    """Unstick a bot wedged on a fatal error (e.g. "database is locked").

    Clears `last_event`, resets `last_tick_ts`, drops a BLOCKED/ERROR risk
    flag, and re-spawns the runner thread if it died but the bot is still
    enabled. No-op if the bot is healthy.
    """
    b = get_bot(int(bot_id))
    if not b:
        return _json({"ok": False, "error": "Bot not found"}, 404)
    if bm is None:
        if not _try_init_bot_manager():
            return _json({"ok": False, "error": "Worker not initialized."}, 503)
    try:
        result = bm.reset_error(int(bot_id))
    except Exception as e:
        logger.exception("reset-error failed for bot %s", bot_id)
        return _json({"ok": False, "error": f"{type(e).__name__}: {e}"}, 500)
    try:
        add_log(int(bot_id), "INFO",
                f"reset-error invoked: actions={result.get('actions')}", "SYSTEM")
    except Exception:
        pass
    snap = {}
    try:
        snap = bm.snapshot(int(bot_id))
    except Exception:
        pass
    return _json({**result, "snap": snap})


# NOTE: _get_bot_client is defined once at line ~1410 with full AlpacaAdapter support


@app.get("/api/bots/{bot_id}/orders")
def api_bot_orders(bot_id: int):
    b = get_bot(int(bot_id))
    if not b:
        return _json({"ok": False, "error": "Bot not found"}, 404)

    # Bug 9: in dry run there are no real exchange orders to fetch. Return a
    # success response with an empty orders list and a `dry_run` flag so the UI
    # can render an informational box instead of an error toast.
    if bool(b.get("dry_run", 1)):
        return _json({
            "ok": True,
            "dry_run": True,
            "orders": [],
            "message": "This bot is running in dry-run mode. No real exchange orders are placed.",
        })

    client, is_kraken = _get_bot_client(b)
    if not client:
         return _json({"ok": False, "error": "Trading client not available", "orders": []}, 503)
    if is_kraken and not _kraken_ready():
        return _json({"ok": False, "error": KRAKEN_ERROR or "Kraken not ready", "orders": []}, 503)

    symbol = _resolve_symbol(b.get("symbol", ""))
    
    if is_kraken:
        mk = _markets()
        if mk and symbol not in mk:
            return _json({"ok": False, "error": f"Symbol not found on Kraken: {symbol}", "orders": []}, 400)

    try:
        orders = client.fetch_open_orders(symbol)
        return _json({"ok": True, "orders": [_serialize_order(o) for o in (orders or [])]})
    except Exception as e:
        return _json({"ok": False, "error": f"{type(e).__name__}: {e}", "orders": []}, 500)


@app.post("/api/bots/{bot_id}/orders")
async def api_bot_order_create(bot_id: int, request: Request):
    b = get_bot(int(bot_id))
    if not b:
        return _json({"ok": False, "error": "Bot not found"}, 404)
        
    client, is_kraken = _get_bot_client(b)
    if not client:
         return _json({"ok": False, "error": "Trading client not available"}, 503)
    if is_kraken and not _kraken_ready():
        return _json({"ok": False, "error": KRAKEN_ERROR or "Kraken not ready"}, 503)
        
    if bool(b.get("dry_run", 1)):
        return _json({"ok": False, "error": "Manual orders are disabled in dry run mode."}, 400)
    if not ALLOW_LIVE_TRADING:
        return _json({"ok": False, "error": "Live trading is disabled. Set ALLOW_LIVE_TRADING=1 in .env to place real orders."}, 403)
    block = _check_trading_allowed(bot_id=int(bot_id))
    if block:
        return _json(block, 503)

    payload = await request.json()
    action = str(payload.get("action") or "").strip().lower()
    size_quote = payload.get("size_quote")
    size_base = payload.get("size_base")
    price = payload.get("price")

    symbol = _resolve_symbol(b.get("symbol", ""))
    
    if is_kraken:
        mk = _markets()
        if mk and symbol not in mk:
            return _json({"ok": False, "error": f"Symbol not found on Kraken: {symbol}"}, 400)

    try:
        order = None
        side = ""
        ord_type = ""
        if action == "market_buy":
            side = "buy"
            ord_type = "market"
            q = float(size_quote or 0)
            if q <= 0:
                return _json({"ok": False, "error": "size_quote must be > 0 for market buy"}, 400)
            order = client.create_market_buy_quote(symbol, q)
        elif action == "market_sell":
            side = "sell"
            ord_type = "market"
            amt = float(size_base or 0)
            if amt <= 0:
                return _json({"ok": False, "error": "size_base must be > 0 for market sell"}, 400)
            order = client.create_market_sell_base(symbol, amt)
        elif action == "limit_buy":
            side = "buy"
            ord_type = "limit"
            amt = float(size_base or 0)
            px = float(price or 0)
            if amt <= 0 or px <= 0:
                return _json({"ok": False, "error": "size_base and price must be > 0 for limit buy"}, 400)
            order = client.create_limit_buy_base(symbol, amt, px)
        elif action == "limit_sell":
            side = "sell"
            ord_type = "limit"
            amt = float(size_base or 0)
            px = float(price or 0)
            if amt <= 0 or px <= 0:
                return _json({"ok": False, "error": "size_base and price must be > 0 for limit sell"}, 400)
            order = client.create_limit_sell_base(symbol, amt, px)
        else:
            return _json({"ok": False, "error": "Invalid action"}, 400)

        price_val = None
        amount_val = None
        try:
            if price is not None and float(price) > 0:
                price_val = float(price)
        except Exception:
            price_val = None
        try:
            if size_base is not None and float(size_base) > 0:
                amount_val = float(size_base)
        except Exception:
            amount_val = None

        add_order_event(
            bot_id=int(bot_id),
            symbol=symbol,
            side=side,
            ord_type=ord_type,
            price=price_val,
            amount=amount_val,
            order_id=str(order.get("id")) if isinstance(order, dict) else None,
            tag="manual",
            status="submitted",
            reason="manual",
            is_live=0 if bool(b.get("dry_run", 1)) else 1,
        )
        add_log(int(bot_id), "INFO", f"Manual order submitted ({action}).", "ORDER")
        return _json({"ok": True, "order": order})
    except Exception as e:
        return _json({"ok": False, "error": f"{type(e).__name__}: {e}"}, 500)


@app.post("/api/bots/{bot_id}/close_position")
async def api_bot_close_position(bot_id: int):
    """One-click close: sell full position and close the deal."""
    b = get_bot(int(bot_id))
    if not b:
        return _json({"ok": False, "error": "Bot not found"}, 404)
    if not bool(b.get("dry_run", 1)):
        block = _check_trading_allowed(bot_id=int(bot_id))
        if block:
            return _json(block, 503)
    client, is_kraken = _get_bot_client(b)
    if not client:
        return _json({"ok": False, "error": "Trading client not available"}, 503)
    if is_kraken and not _kraken_ready():
        return _json({"ok": False, "error": KRAKEN_ERROR or "Kraken not ready"}, 503)
    if bool(b.get("dry_run", 1)):
        return _json({"ok": False, "error": "Close position is disabled in dry run mode."}, 400)
    if not ALLOW_LIVE_TRADING:
        return _json({"ok": False, "error": "Live trading disabled. Set ALLOW_LIVE_TRADING=1."}, 403)

    symbol = _resolve_symbol(b.get("symbol", ""))
    if is_kraken:
        mk = _markets()
        if mk and symbol not in mk:
            return _json({"ok": False, "error": f"Symbol not found: {symbol}"}, 400)

    try:
        snap = bm.snapshot(int(bot_id)) if bm else {}
        base_pos = float(snap.get("base_pos") or 0)
        if base_pos <= 0:
            return _json({"ok": False, "error": "No open position to close"}, 400)

        order = client.create_market_sell_base(symbol, base_pos)
        add_log(int(bot_id), "INFO", f"Quick close: sold {base_pos} {symbol}", "ORDER")

        od = latest_open_deal(int(bot_id))
        if od:
            deal_id = int(od["id"])
            deal_opened = int(od.get("opened_at") or 0)
            from db import close_deal
            entry_avg = float(od.get("entry_avg") or snap.get("avg_entry") or 0)
            exit_avg = float(snap.get("last_price") or entry_avg)
            close_deal(
                deal_id,
                entry_avg=entry_avg,
                exit_avg=exit_avg,
                base_amount=base_pos,
                realized_pnl_quote=float((exit_avg - entry_avg) * base_pos) if entry_avg > 0 else 0.0,
                hold_sec=int(time.time()) - deal_opened,
                exit_strategy="manual_close",
            )
            add_log(int(bot_id), "INFO", f"Deal {deal_id} closed (manual).", "SYSTEM")

        return _json({"ok": True, "message": f"Sold {base_pos} {symbol}", "order": order})
    except Exception as e:
        logger.exception("close_position failed bot_id=%s: %s", bot_id, e)
        return _json({"ok": False, "error": f"{type(e).__name__}: {e}"}, 500)


# Bug 10: Manual close-deal endpoint. Works in dry run (simulated market sell at
# the last tick price) and in live mode (sells the position). Records the deal
# as CLOSED so the Journal / Analytics / Strategy stats start accumulating.
@app.post("/api/bots/{bot_id}/deals/{deal_id}/close")
async def api_bot_deal_close(bot_id: int, deal_id: int, request: Request):
    b = get_bot(int(bot_id))
    if not b:
        return _json({"ok": False, "error": "Bot not found"}, 404)

    try:
        body = await request.json()
    except Exception:
        body = {}
    requested_price = body.get("exit_price") if isinstance(body, dict) else None
    note = (body.get("note") if isinstance(body, dict) else None) or ""

    target_deal: Optional[Dict[str, Any]] = None
    try:
        target_deal = get_deal(int(deal_id), full=True)
    except TypeError:
        try:
            target_deal = get_deal(int(deal_id))
        except Exception:
            target_deal = None
    except Exception:
        target_deal = None
    if target_deal and int(target_deal.get("bot_id") or 0) != int(bot_id):
        return _json({"ok": False, "error": f"Deal {deal_id} does not belong to bot {bot_id}"}, 400)

    if not target_deal:
        return _json({"ok": False, "error": f"Deal {deal_id} not found for bot {bot_id}"}, 404)

    deal_state = str(target_deal.get("state") or "").upper()
    if deal_state in ("CLOSED", "CANCELLED"):
        return _json({"ok": False, "error": f"Deal {deal_id} already {deal_state}"}, 400)

    snap: Dict[str, Any] = {}
    try:
        snap = bm.snapshot(int(bot_id)) if bm else {}
    except Exception:
        snap = {}

    base_amount = float(target_deal.get("base_amount") or snap.get("base_pos") or 0.0)
    entry_avg = float(target_deal.get("entry_avg") or snap.get("avg_entry") or 0.0)
    if base_amount <= 0 or entry_avg <= 0:
        return _json({
            "ok": False,
            "error": f"Cannot close deal {deal_id}: missing entry data (base_amount={base_amount}, entry_avg={entry_avg}).",
        }, 400)

    is_dry_run = bool(b.get("dry_run", 1))

    # Determine exit price.
    exit_price: Optional[float] = None
    try:
        if requested_price is not None and float(requested_price) > 0:
            exit_price = float(requested_price)
    except (TypeError, ValueError):
        exit_price = None
    if exit_price is None:
        try:
            lp = snap.get("last_price")
            if lp is not None and float(lp) > 0:
                exit_price = float(lp)
        except Exception:
            exit_price = None
    if exit_price is None:
        # Fall back to fetching a fresh ticker.
        try:
            client_pair = _get_bot_client(b)
            if client_pair and client_pair[0] is not None:
                _client = client_pair[0]
                _sym = _resolve_symbol(b.get("symbol", ""))
                tick = _client.fetch_ticker(_sym) if hasattr(_client, "fetch_ticker") else {}
                if isinstance(tick, dict):
                    for k in ("last", "close"):
                        if tick.get(k) is not None:
                            exit_price = float(tick.get(k))
                            break
        except Exception:
            exit_price = None
    if exit_price is None or exit_price <= 0:
        return _json({"ok": False, "error": "No reference price available to close the deal."}, 400)

    realized_pnl = (exit_price - entry_avg) * base_amount
    realized_pnl_pct = ((exit_price - entry_avg) / entry_avg) * 100.0 if entry_avg > 0 else 0.0
    hold_sec = int(time.time()) - int(target_deal.get("opened_at") or int(time.time()))

    # In live mode, place the actual market sell first.
    live_order: Optional[Dict[str, Any]] = None
    if not is_dry_run:
        if not ALLOW_LIVE_TRADING:
            return _json({"ok": False, "error": "Live trading disabled. Set ALLOW_LIVE_TRADING=1 in .env."}, 403)
        block = _check_trading_allowed(bot_id=int(bot_id))
        if block:
            return _json(block, 503)
        try:
            client, is_kraken = _get_bot_client(b)
            if not client:
                return _json({"ok": False, "error": "Trading client not available"}, 503)
            if is_kraken and not _kraken_ready():
                return _json({"ok": False, "error": KRAKEN_ERROR or "Kraken not ready"}, 503)
            symbol = _resolve_symbol(b.get("symbol", ""))
            live_order = client.create_market_sell_base(symbol, base_amount)
        except Exception as e:
            logger.exception("close_deal live order failed bot=%s deal=%s: %s", bot_id, deal_id, e)
            return _json({"ok": False, "error": f"Live close failed: {type(e).__name__}: {e}"}, 500)

    journal_note = note or ("Manual dry-run close at $%.4f" % exit_price if is_dry_run else "Manual live close")
    try:
        if bm:
            result = bm.manual_close_open_deal(
                int(bot_id),
                int(deal_id),
                entry_avg=float(entry_avg),
                exit_avg=float(exit_price),
                base_amount=float(base_amount),
                realized_pnl_quote=float(realized_pnl),
                entry_strategy=target_deal.get("entry_strategy"),
                exit_strategy="manual_close" + ("_dry" if is_dry_run else ""),
                hold_sec=int(hold_sec),
                safety_count=int(target_deal.get("safety_count") or 0),
                journal_exit_reason=journal_note,
                entry_regime=target_deal.get("entry_regime"),
                exit_regime=target_deal.get("exit_regime"),
                mae=target_deal.get("mae"),
                mfe=target_deal.get("mfe"),
            )
        else:
            from db import manual_close_deal_and_journal as _mc

            result = _mc(
                int(deal_id),
                int(bot_id),
                float(entry_avg),
                float(exit_price),
                float(base_amount),
                float(realized_pnl),
                entry_strategy=target_deal.get("entry_strategy"),
                exit_strategy="manual_close" + ("_dry" if is_dry_run else ""),
                hold_sec=int(hold_sec),
                safety_count=int(target_deal.get("safety_count") or 0),
                journal_exit_reason=journal_note,
                entry_regime=target_deal.get("entry_regime"),
                exit_regime=target_deal.get("exit_regime"),
                mae=target_deal.get("mae"),
                mfe=target_deal.get("mfe"),
            )
    except ValueError as e:
        return _json({"ok": False, "error": str(e)}, 400)
    except Exception as e:
        logger.exception("manual_close_open_deal failed bot=%s deal=%s: %s", bot_id, deal_id, e)
        return _json({"ok": False, "error": f"{type(e).__name__}: {e}"}, 500)

    add_log(
        int(bot_id), "INFO",
        f"Manual close deal #{deal_id} @ ${exit_price:.4f} — pnl={realized_pnl:+.2f} ({realized_pnl_pct:+.2f}%)",
        "ORDER",
    )

    rp_pct = float(result.get("realized_pnl_pct", realized_pnl_pct))
    return _json({
        "ok": True,
        "deal_id": int(deal_id),
        "exit_price": float(exit_price),
        "entry_avg": float(entry_avg),
        "base_amount": float(base_amount),
        "realized_pnl_quote": float(realized_pnl),
        "realized_pnl": float(realized_pnl),
        "realized_pnl_pct": rp_pct,
        "hold_sec": int(hold_sec),
        "dry_run": bool(is_dry_run),
        "live_order": live_order,
    })


@app.delete("/api/bots/{bot_id}/orders/{order_id}")
def api_bot_order_cancel(bot_id: int, order_id: str):
    b = get_bot(int(bot_id))
    if not b:
        return _json({"ok": False, "error": "Bot not found"}, 404)
        
    client, is_kraken = _get_bot_client(b)
    if not client:
         return _json({"ok": False, "error": "Trading client not available"}, 503)
    if is_kraken and not _kraken_ready():
        return _json({"ok": False, "error": KRAKEN_ERROR or "Kraken not ready"}, 503)
        
    if bool(b.get("dry_run", 1)):
        return _json({"ok": False, "error": "Manual orders are disabled in dry run mode."}, 400)
    if not ALLOW_LIVE_TRADING:
        return _json({"ok": False, "error": "Live trading is disabled. Set ALLOW_LIVE_TRADING=1 in .env to cancel real orders."}, 403)

    symbol = _resolve_symbol(b.get("symbol", ""))
    if is_kraken:
        mk = _markets()
        if mk and symbol not in mk:
            return _json({"ok": False, "error": f"Symbol not found on Kraken: {symbol}"}, 400)

    try:
        res = client.cancel_order(str(order_id), symbol)
        add_order_event(
            bot_id=int(bot_id),
            symbol=symbol,
            side="",
            ord_type="cancel",
            price=None,
            amount=None,
            order_id=str(order_id),
            tag="manual",
            status="cancelled",
            reason="manual",
            is_live=0 if bool(b.get("dry_run", 1)) else 1,
        )
        add_log(int(bot_id), "INFO", f"Manual order cancelled ({order_id}).", "ORDER")
        return _json({"ok": True, "result": res})
    except Exception as e:
        return _json({"ok": False, "error": f"{type(e).__name__}: {e}"}, 500)


@app.post("/api/bots/{bot_id}/orders/cancel_all")
def api_bot_order_cancel_all(bot_id: int):
    b = get_bot(int(bot_id))
    if not b:
        return _json({"ok": False, "error": "Bot not found"}, 404)
    
    client, is_kraken = _get_bot_client(b)
    if not client:
         return _json({"ok": False, "error": "Trading client not available"}, 503)
    if is_kraken and not _kraken_ready():
        return _json({"ok": False, "error": KRAKEN_ERROR or "Kraken not ready"}, 503)
        
    if bool(b.get("dry_run", 1)):
        return _json({"ok": False, "error": "Manual orders are disabled in dry run mode."}, 400)
    if not ALLOW_LIVE_TRADING:
        return _json({"ok": False, "error": "Live trading is disabled. Set ALLOW_LIVE_TRADING=1 in .env to cancel real orders."}, 403)

    symbol = _resolve_symbol(b.get("symbol", ""))
    if is_kraken:
        mk = _markets()
        if mk and symbol not in mk:
            return _json({"ok": False, "error": f"Symbol not found on Kraken: {symbol}"}, 400)

    try:
        client.cancel_all_open_orders(symbol)
        add_order_event(
            bot_id=int(bot_id),
            symbol=symbol,
            side="",
            ord_type="cancel_all",
            price=None,
            amount=None,
            order_id=None,
            tag="manual",
            status="cancelled",
            reason="manual",
            is_live=0 if bool(b.get("dry_run", 1)) else 1,
        )
        add_log(int(bot_id), "INFO", "Manual cancel-all submitted.", "ORDER")
        return _json({"ok": True})
    except Exception as e:
        return _json({"ok": False, "error": f"{type(e).__name__}: {e}"}, 500)


@app.post("/api/bots/{bot_id}/add_funds")
async def api_bot_add_funds(bot_id: int, request: Request):
    """Add funds to an active bot. Increases investment_amount and scales safety order sizes."""
    b = get_bot(int(bot_id))
    if not b:
        return _json({"ok": False, "error": "Bot not found"}, 404)
    
    try:
        payload = await request.json()
    except Exception:
        return _json({"ok": False, "error": "Invalid JSON payload"}, 400)
    
    add_amount = float(payload.get("amount") or 0)
    add_safety_orders = int(payload.get("add_safety_orders") or 0)
    
    if add_amount < 5:
        return _json({"ok": False, "error": "Amount must be at least $5"}, 400)
    if add_amount > 100000:
        return _json({"ok": False, "error": "Amount exceeds maximum allowed"}, 400)
    
    # Current values
    current_max_spend = float(b.get("max_spend_quote") or b.get("base_quote") or 0)
    current_base_quote = float(b.get("base_quote") or 20)
    current_safety_quote = float(b.get("safety_quote") or 10)
    current_max_safety = int(b.get("max_safety") or 3)
    
    # Calculate new values
    new_max_spend = current_max_spend + add_amount
    
    # Scale safety order size proportionally (optional: keep same or scale)
    scale_factor = new_max_spend / current_max_spend if current_max_spend > 0 else 1
    new_safety_quote = current_safety_quote * scale_factor
    new_base_quote = current_base_quote * scale_factor
    
    # Add safety orders if requested
    new_max_safety = current_max_safety + add_safety_orders
    
    # Update bot config
    try:
        update_payload = {
            "max_spend_quote": new_max_spend,
            "base_quote": new_base_quote,
            "safety_quote": new_safety_quote,
            "max_safety": new_max_safety,
        }
        
        # Merge with existing bot config
        updated_bot = {**b, **update_payload}
        
        # Use update_bot from db module
        from db import update_bot
        update_bot(int(bot_id), updated_bot)
        
        add_log(
            int(bot_id),
            "INFO",
            f"Funds added: +${add_amount:.2f} (new total: ${new_max_spend:.2f}). Safety orders: {current_max_safety} -> {new_max_safety}.",
            "SYSTEM"
        )
        
        return _json({
            "ok": True,
            "message": f"Added ${add_amount:.2f} to bot",
            "new_max_spend": new_max_spend,
            "new_base_quote": new_base_quote,
            "new_safety_quote": new_safety_quote,
            "new_max_safety": new_max_safety,
        })
        
    except Exception as e:
        logger.exception("Failed to add funds to bot %d", bot_id)
        return _json({"ok": False, "error": f"Failed to update bot: {e}"}, 500)


def _bot_live_degraded(b: Dict[str, Any], bot_id: int, logs_limit: int, deals_limit: int, last_event: str, price_error: Optional[str] = None) -> Dict[str, Any]:
    """Return a successful live payload with degraded snap so the UI loads instead of showing 'Live refresh failed'."""
    sym = b.get("symbol", "")
    snap = {
        "running": False,
        "last_event": last_event,
        "last_price": None,
        "avg_entry": None,
        "base_pos": None,
    }
    return {
        "ok": True,
        "bot": b,
        "snap": snap,
        "logs": list_logs(int(bot_id), limit=int(max(1, min(int(logs_limit), 2000)))),
        "deals": list_deals(int(bot_id), limit=int(max(1, min(int(deals_limit), 1000)))),
        "market_type": classify_symbol(sym),
        "kraken_ready": _kraken_ready(),
        "kraken_error": KRAKEN_ERROR,
        "alpaca_paper_ready": ALPACA_PAPER_READY,
        "alpaca_live_ready": ALPACA_LIVE_READY,
        "alpaca_error": ALPACA_ERROR or "",
        "price_error": price_error or last_event,
        "data_health": None,
        "worker_degraded": True,
    }


@app.get("/api/bots/{bot_id}/live")
def api_bot_live(bot_id: int, logs_limit: int = 150, deals_limit: int = 30):
    """
    Single call used by the UI to avoid "Loading..." races.
    """
    try:
        b = get_bot(int(bot_id))
        if not b:
            return _json({"ok": False, "error": "Bot not found"}, 404)
        if bm is None:
            return _json(_bot_live_degraded(
                b, bot_id, logs_limit, deals_limit,
                "Worker not initialized. Check Kraken/Alpaca API keys and restart the service.",
                "Worker not initialized",
            ))

        try:
            snap = bm.snapshot(int(bot_id))
        except ValueError as ve:
            msg = str(ve)
            if "Alpaca live" in msg or "not initialized" in msg:
                return _json(_bot_live_degraded(
                    b, bot_id, logs_limit, deals_limit,
                    msg,
                    msg,
                ))
            raise
        except Exception:
            raise
        if snap.get("running") and not snap.get("last_event"):
            snap["last_event"] = "Running."
        if not snap.get("running") and not snap.get("last_event"):
            snap["last_event"] = "Stopped."
        
        # Try to fetch current price if missing or zero (non-blocking, with timeout)
        sym = b.get("symbol", "")
        market_type = classify_symbol(sym)
        current_price = snap.get("last_price")
        price_error = None
        
        if current_price is None or current_price <= 0:
            # Use cached price if available to avoid blocking
            # Only fetch fresh price if absolutely necessary and do it quickly
            import threading
            import time as time_module
            price_fetched = threading.Event()
            fetched_price = [None]
            fetch_error = [None]
            
            def _fetch_price():
                try:
                    if market_type == "stock":
                        # Stock: Try getting from Alpaca
                        client, _ = _get_bot_client(b)
                        if client:
                            # Use get_ticker with timeout protection
                            t = client.get_ticker(sym)
                            lp = float(t.get("last", 0) or t.get("price", 0) or 0)
                            if lp > 0:
                                fetched_price[0] = lp
                        else:
                            fetch_error[0] = "Alpaca client not available"
                    else:
                        # Crypto: Use Kraken safe price
                        lp = _safe_last_price(sym)
                        if lp is not None and lp > 0:
                            fetched_price[0] = lp
                        elif not _kraken_ready():
                            fetch_error[0] = "Kraken not ready"
                except Exception as e:
                    fetch_error[0] = str(e)
                finally:
                    price_fetched.set()
            
            # Start fetch in background thread with 2 second timeout
            fetch_thread = threading.Thread(target=_fetch_price, daemon=True)
            fetch_thread.start()
            fetch_thread.join(timeout=2.0)  # Max 2 seconds wait
            
            # If we got a price, use it; otherwise keep existing (might be 0 or None)
            if fetched_price[0] is not None and fetched_price[0] > 0:
                snap["last_price"] = fetched_price[0]
            elif fetch_error[0]:
                price_error = fetch_error[0]

        logs = list_logs(int(bot_id), limit=int(max(1, min(int(logs_limit), 2000))))
        deals = list_deals(int(bot_id), limit=int(max(1, min(int(deals_limit), 1000))))

        data_health = None
        try:
            router = getattr(bm, "_md_router", None)
            if router:
                data_health = router.get_data_health(sym, b.get("market_type", "crypto"), required_tfs=["1h", "4h", "1d"], min_candles=20)
        except Exception:
            pass
        
        return _json(
            {
                "ok": True,
                "bot": b,
                "snap": snap,
                "logs": logs,
                "deals": deals,
                "market_type": market_type,
                "kraken_ready": _kraken_ready(),
                "kraken_error": KRAKEN_ERROR,
                "alpaca_paper_ready": ALPACA_PAPER_READY,
                "alpaca_live_ready": ALPACA_LIVE_READY,
                "alpaca_error": ALPACA_ERROR or "",
                "price_error": price_error,
                "data_health": data_health,
            }
        )
    except Exception as e:
        logger.error(f"api_bot_live error for bot {bot_id}: {type(e).__name__}: {e}")
        return _json({"ok": False, "error": f"Live refresh error: {type(e).__name__}: {e}"}, 500)


@app.get("/api/bots/{bot_id}/logs")
def api_bot_logs(bot_id: int, limit: int = 200):
    b = get_bot(int(bot_id))
    if not b:
        raise HTTPException(status_code=404, detail="Bot not found")

    safe_limit = int(max(1, min(int(limit), 2000)))
    logs = list_logs(int(bot_id), limit=safe_limit)

    # logs usually come newest-first. We will compress repeated spam lines.
    compressed = []
    last_msg = None
    prev_ts = 0
    prev_level = "INFO"
    repeat_count = 0

    for row in logs:
        msg = (row.get("message") or "").strip()
        if msg == last_msg:
            repeat_count += 1
            continue

        # flush previous repeated message
        if last_msg is not None:
            if repeat_count > 0:
                compressed.append(
                    {"ts": prev_ts, "level": prev_level, "message": f"{last_msg} (x{repeat_count+1})"}
                )
            else:
                compressed.append({"ts": prev_ts, "level": prev_level, "message": last_msg})

        # start tracking new message
        last_msg = msg
        prev_ts = int(row.get("ts") or 0)
        prev_level = row.get("level") or "INFO"
        repeat_count = 0

    # flush last
    if last_msg is not None:
        if repeat_count > 0:
            compressed.append({"ts": prev_ts, "level": prev_level, "message": f"{last_msg} (x{repeat_count+1})"})
        else:
            compressed.append({"ts": prev_ts, "level": prev_level, "message": last_msg})

    # return newest-first like before
    return {"ok": True, "logs": compressed[:200]}


@app.get("/api/bots/{bot_id}/deals")
def api_bot_deals(bot_id: int, limit: int = 50):
    b = get_bot(int(bot_id))
    if not b:
        return _json({"ok": False, "error": "Bot not found"}, 404)
    return _json({"ok": True, "deals": list_deals(int(bot_id), limit=int(max(1, min(int(limit), 1000))))})


# =========================================================
# API: Trade Journal
# =========================================================
@app.get("/api/journal")
def api_journal_list(
    days: int = 90,
    strategy: Optional[str] = None,
    limit: int = 100,
):
    """List closed deals + optional canonical `journal` rows (backfill / write-on-close)."""
    lim = int(max(1, min(500, int(limit))))
    journal_rows = list_journal_entries(limit=lim, offset=0)
    since = now_ts() - (int(days) * 86400) if days else None
    deals = list_closed_deals_for_journal(since_ts=since, limit=500)
    if strategy and str(strategy).strip():
        strat = str(strategy).strip().lower()
        deals = [d for d in deals if (str(d.get("entry_strategy") or "").lower() == strat or str(d.get("exit_strategy") or "").lower() == strat)]
    deal_ids = [int(d["id"]) for d in deals]
    journals = list_trade_journals_for_deals(deal_ids)
    bot_map = {int(b["id"]): b for b in list_bots()}
    out = []
    for d in deals:
        j = journals.get(int(d["id"]), {})
        bot = bot_map.get(int(d.get("bot_id") or 0), {})
        out.append({
            **d,
            "journal": j,
            "bot_name": bot.get("name"),
            "bot_symbol": bot.get("symbol"),
        })
    return _json({"ok": True, "journal": journal_rows, "deals": out})


@app.get("/api/journal/{deal_id}")
def api_journal_get(deal_id: int):
    deal = get_deal(int(deal_id), full=True)
    if not deal:
        return _json({"ok": False, "error": "Deal not found"}, 404)
    journal = get_trade_journal(int(deal_id))
    bot = get_bot(int(deal.get("bot_id") or 0)) or {}
    return _json({
        "ok": True,
        "deal": deal,
        "journal": journal,
        "bot_name": bot.get("name"),
        "bot_symbol": bot.get("symbol"),
    })


@app.put("/api/journal/{deal_id}")
async def api_journal_put(deal_id: int, request: Request):
    deal = get_deal(int(deal_id))
    if not deal:
        return _json({"ok": False, "error": "Deal not found"}, 404)
    try:
        body = await request.json()
    except Exception:
        body = {}
    upsert_trade_journal(
        int(deal_id),
        entry_reason=body.get("entry_reason"),
        exit_reason=body.get("exit_reason"),
        lessons_learned=body.get("lessons_learned"),
        screenshot_data=body.get("screenshot_data"),
    )
    return _json({"ok": True, "journal": get_trade_journal(int(deal_id))})


# =========================================================
# API: Performance Analytics (Sharpe, Sortino, win rate, etc.)
# =========================================================
@app.get("/api/analytics/performance")
def api_analytics_performance(days: int = 90):
    """Sharpe, Sortino, max drawdown, win rate by strategy/symbol."""
    import math
    since = now_ts() - (int(days) * 86400)
    closed = list_closed_deals_for_journal(since_ts=since, limit=2000)
    closed = [d for d in closed if d.get("closed_at") and int(d.get("closed_at") or 0) >= since]
    pnls = [float(d.get("realized_pnl_quote") or 0) for d in closed]
    wins = sum(1 for p in pnls if p > 0)
    losses = sum(1 for p in pnls if p < 0)
    total_pnl = sum(pnls)
    win_rate = wins / len(pnls) if pnls else 0.0
    n = len(pnls)
    mean_ret = total_pnl / n if n else 0.0
    variance = sum((p - mean_ret) ** 2 for p in pnls) / n if n else 0.0
    std = math.sqrt(variance) if variance > 0 else 0.0
    downside = [p for p in pnls if p < 0]
    downside_var = sum(p ** 2 for p in downside) / len(downside) if downside else 0.0
    downside_std = math.sqrt(downside_var) if downside_var > 0 else 0.0
    sharpe = (mean_ret / std * math.sqrt(252)) if std > 0 else 0.0
    sortino = (mean_ret / downside_std * math.sqrt(252)) if downside_std > 0 else 0.0
    cum = 0.0
    peak = 0.0
    max_dd = 0.0
    for p in pnls:
        cum += p
        peak = max(peak, cum)
        dd = peak - cum if peak > 0 else 0
        if dd > max_dd:
            max_dd = dd
    by_strategy = {}
    by_symbol = {}
    for d in closed:
        s = d.get("entry_strategy") or "unknown"
        sym = d.get("symbol") or "?"
        if s not in by_strategy:
            by_strategy[s] = {"count": 0, "wins": 0, "pnl": 0.0}
        by_strategy[s]["count"] += 1
        by_strategy[s]["wins"] += 1 if float(d.get("realized_pnl_quote") or 0) > 0 else 0
        by_strategy[s]["pnl"] += float(d.get("realized_pnl_quote") or 0)
        if sym not in by_symbol:
            by_symbol[sym] = {"count": 0, "wins": 0, "pnl": 0.0}
        by_symbol[sym]["count"] += 1
        by_symbol[sym]["wins"] += 1 if float(d.get("realized_pnl_quote") or 0) > 0 else 0
        by_symbol[sym]["pnl"] += float(d.get("realized_pnl_quote") or 0)
    from datetime import datetime
    daily_pnl: Dict[str, float] = {}
    for d in closed:
        ts = int(d.get("closed_at") or 0)
        if ts:
            dt = datetime.utcfromtimestamp(ts)
            key = dt.strftime("%Y-%m-%d")
            daily_pnl[key] = daily_pnl.get(key, 0.0) + float(d.get("realized_pnl_quote") or 0)
    # Calculate additional analytics for new enhanced page
    equity_curve = []
    cum = 0.0
    peak = 0.0
    for p in pnls:
        cum += p
        peak = max(peak, cum)
        equity_curve.append({"timestamp": int(now_ts()), "value": cum, "pnl": p})

    # Trade history for streaks
    pnl_history = [p for p in pnls]

    # Trades data for best/worst trades
    trades_list = []
    for d in closed:
        trades_list.append({
            "symbol": d.get("symbol") or "?",
            "entry_price": float(d.get("entry_avg") or 0),
            "exit_price": float(d.get("exit_avg") or 0),
            "quantity": float(d.get("amount") or 0),
            "pnl": float(d.get("realized_pnl_quote") or 0),
            "pnl_pct": float(d.get("profit_percent") or 0),
            "closed_at": int(d.get("closed_at") or 0),
            "notes": d.get("notes") or ""
        })

    # Trade journal
    trade_journal = []
    for d in closed:
        trade_journal.append({
            "symbol": d.get("symbol") or "?",
            "entry_price": float(d.get("entry_avg") or 0),
            "exit_price": float(d.get("exit_avg") or 0),
            "quantity": float(d.get("amount") or 0),
            "pnl": float(d.get("realized_pnl_quote") or 0),
            "pnl_pct": float(d.get("profit_percent") or 0),
            "closed_at": int(d.get("closed_at") or 0),
            "notes": d.get("notes") or ""
        })

    return _json({
        "ok": True,
        "days": days,
        "trades": n,
        "total_pnl": total_pnl,
        "win_rate": round(win_rate, 2),
        "wins": wins,
        "losses": losses,
        "sharpe_ratio": round(sharpe, 2),
        "sortino_ratio": round(sortino, 2),
        "max_drawdown": round(max_dd, 2),
        "by_strategy": by_strategy,
        "by_symbol": by_symbol,
        "daily_pnl": daily_pnl,
        "equity_curve": equity_curve,
        "pnl_history": pnl_history,
        "trades": trades_list,
        "trade_journal": trade_journal,
    })


@app.get("/api/analytics/summary")
def api_analytics_summary(days: int = 90):
    """Return summary of analytics: win rate, avg win/loss, Sharpe, best/worst trades, streaks."""
    import math
    since = now_ts() - (int(days) * 86400)
    closed = list_closed_deals_for_journal(since_ts=since, limit=2000)
    closed = [d for d in closed if d.get("closed_at") and int(d.get("closed_at") or 0) >= since]
    pnls = [float(d.get("realized_pnl_quote") or 0) for d in closed]

    if not pnls:
        return _json({
            "ok": True,
            "win_rate": 0,
            "avg_win": 0,
            "avg_loss": 0,
            "sharpe": 0,
            "best_trade": {"symbol": "N/A", "pnl": 0},
            "worst_trade": {"symbol": "N/A", "pnl": 0},
            "current_win_streak": 0,
            "current_loss_streak": 0,
        })

    wins = [p for p in pnls if p > 0]
    losses = [p for p in pnls if p < 0]
    avg_win = sum(wins) / len(wins) if wins else 0
    avg_loss = sum(losses) / len(losses) if losses else 0

    # Sharpe
    mean_ret = sum(pnls) / len(pnls)
    variance = sum((p - mean_ret) ** 2 for p in pnls) / len(pnls)
    std = math.sqrt(variance) if variance > 0 else 0.0
    sharpe = (mean_ret / std * math.sqrt(252)) if std > 0 else 0.0

    # Best and worst trades
    best_pnl = max(pnls) if pnls else 0
    worst_pnl = min(pnls) if pnls else 0
    best_trade = [d for d in closed if float(d.get("realized_pnl_quote") or 0) == best_pnl]
    worst_trade = [d for d in closed if float(d.get("realized_pnl_quote") or 0) == worst_pnl]

    # Streaks (count from most recent trade backward)
    win_streak = 0
    loss_streak = 0
    for p in reversed(pnls):
        if p > 0:
            if loss_streak > 0:
                break
            win_streak += 1
        elif p < 0:
            if win_streak > 0:
                break
            loss_streak += 1
        else:
            break

    return _json({
        "ok": True,
        "win_rate": round(len(wins) / len(pnls), 2) if pnls else 0,
        "avg_win": round(avg_win, 2),
        "avg_loss": round(avg_loss, 2),
        "sharpe": round(sharpe, 2),
        "best_trade": {"symbol": (best_trade[0].get("symbol") if best_trade else "N/A"), "pnl": round(best_pnl, 2)},
        "worst_trade": {"symbol": (worst_trade[0].get("symbol") if worst_trade else "N/A"), "pnl": round(worst_pnl, 2)},
        "current_win_streak": win_streak,
        "current_loss_streak": loss_streak,
    })


@app.get("/api/analytics/daily_pnl")
def api_analytics_daily_pnl(days: int = 90):
    """Return daily P&L history for charting."""
    import math
    from datetime import datetime
    since = now_ts() - (int(days) * 86400)
    closed = list_closed_deals_for_journal(since_ts=since, limit=2000)

    daily_pnl = {}
    for d in closed:
        ts = int(d.get("closed_at") or 0)
        if ts and ts >= since:
            dt = datetime.utcfromtimestamp(ts)
            key = dt.strftime("%Y-%m-%d")
            daily_pnl[key] = daily_pnl.get(key, 0.0) + float(d.get("realized_pnl_quote") or 0)

    return _json({
        "ok": True,
        "daily_pnl": daily_pnl,
    })


@app.get("/api/analytics/equity_curve")
def api_analytics_equity_curve(days: int = 90):
    """Return portfolio equity curve history."""
    import math
    from datetime import datetime
    since = now_ts() - (int(days) * 86400)
    closed = list_closed_deals_for_journal(since_ts=since, limit=2000)

    # Sort by closed_at
    closed = sorted(closed, key=lambda d: int(d.get("closed_at") or 0))

    equity_curve = []
    cum = 0.0
    for d in closed:
        ts = int(d.get("closed_at") or 0)
        if ts and ts >= since:
            cum += float(d.get("realized_pnl_quote") or 0)
            equity_curve.append({
                "timestamp": ts,
                "value": cum,
                "pnl": float(d.get("realized_pnl_quote") or 0),
            })

    return _json({
        "ok": True,
        "equity_curve": equity_curve,
    })


@app.get("/api/portfolio/history")
def api_portfolio_history():
    """Return portfolio value history for analytics."""
    with _globals_lock:
        history = list(PORT_HISTORY[-500:])

    return _json({
        "ok": True,
        "history": history,
    })


# =========================================================
# API: chart candles + markers
# =========================================================
@app.get("/api/bots/{bot_id}/ohlc")
def api_bot_ohlc(bot_id: int, timeframe: str = "5m", limit: int = 200):
    b = get_bot(int(bot_id))
    if not b:
        return _json({"ok": False, "error": "Bot not found"}, 404)
    
    client, is_kraken = _get_bot_client(b)
    if not client:
         return _json({"ok": False, "error": "Trading client not available", "candles": []}, 503)
    if is_kraken and not _kraken_ready():
        return _json({"ok": False, "error": KRAKEN_ERROR or "Kraken not ready", "candles": []}, 503)

    tf = _sanitize_tf(timeframe)
    lim = int(max(10, min(2000, int(limit))))

    symbol = _resolve_symbol(b.get("symbol", ""))
    if is_kraken:
        mk = _markets()
        if mk and symbol not in mk:
            return _json({"ok": False, "error": f"Symbol not found on Kraken: {symbol}", "candles": []}, 400)

    try:
        # Optimizations: Use BotManager cache for standard timeframes (Kraken only for now)
        # Only if bot is actually running in BM, otherwise BM cache might be stale/empty?
        # Actually BM cache is global for market data, so it's fine if BM is running.
        used_cache = False
        ohlcv = []
        
        if is_kraken and bm and tf in ("5m", "15m", "1h", "4h"):
            # Try cache first
            cached = bm.ohlcv_cached(symbol, tf, limit=lim)
            if cached:
                ohlcv = cached
                used_cache = True
        
        if not used_cache:
            # Direct fetch (Stocks, or Kraken cache miss, or custom TF)
            if is_kraken:
                ohlcv = client.fetch_ohlcv(symbol, timeframe=tf, limit=lim)
            else:
                # Alpaca: use fetch_ohlcv (AlpacaAdapter) or get_ohlcv (AlpacaClient)
                try:
                    if hasattr(client, "fetch_ohlcv"):
                        ohlcv = client.fetch_ohlcv(symbol, timeframe=tf, limit=lim)
                    else:
                        ohlcv = client.get_ohlcv(symbol, tf, lim)
                except Exception as ex:
                    logger.warning("OHLC fetch failed for %s %s: %s", symbol, tf, ex)
                    ohlcv = []
            
        candles = []
        if not ohlcv:
             return _json({"ok": True, "candles": []})

        for row in ohlcv:
            # Handle potential different formats (just in case)
            # Expecting [ts, o, h, l, c, v] where ts might be in seconds or milliseconds
            if len(row) < 5: 
                continue
            
            try:
                ts_raw = float(row[0])
                # Convert to seconds: if timestamp > 1e10, it's in milliseconds
                if ts_raw > 1e10:
                    ts_sec = int(ts_raw // 1000)
                else:
                    ts_sec = int(ts_raw)
                
                candles.append(
                    {
                        "time": ts_sec,
                        "open": float(row[1]),
                        "high": float(row[2]),
                        "low": float(row[3]),
                        "close": float(row[4]),
                    }
                )
            except (ValueError, TypeError, IndexError) as e:
                # Skip invalid rows
                continue
        
        return _json({"ok": True, "candles": candles, "symbol": symbol, "source": "cache" if used_cache else "api"})
    except Exception as e:
        return _json({"ok": False, "error": f"{type(e).__name__}: {e}", "candles": []}, 500)


_BUY_RE = re.compile(r"\bbuy\b", re.IGNORECASE)
_SELL_RE = re.compile(r"\bsell\b", re.IGNORECASE)


@app.get("/api/bots/{bot_id}/markers")
def api_bot_markers(bot_id: int, timeframe: str = "5m", limit: int = 250):
    b = get_bot(int(bot_id))
    if not b:
        return _json({"ok": False, "error": "Bot not found"}, 404)

    tf = _sanitize_tf(timeframe)
    bucket = _tf_seconds(tf)

    logs = list_logs(int(bot_id), limit=int(max(50, min(2000, int(limit)))))
    markers = []

    for x in logs:
        msg = str(x.get("message") or "")
        ts = int(x.get("ts") or 0)
        if ts <= 0:
            continue

        is_buy = bool(_BUY_RE.search(msg))
        is_sell = bool(_SELL_RE.search(msg))
        if not (is_buy or is_sell):
            continue

        t = int((ts // bucket) * bucket)
        markers.append(
            {
                "time": t,
                "position": "belowBar" if is_buy else "aboveBar",
                "color": "#16a34a" if is_buy else "#ef4444",
                "shape": "arrowUp" if is_buy else "arrowDown",
                "text": "BUY" if is_buy else "SELL",
            }
        )

    markers.sort(key=lambda m: int(m.get("time") or 0))
    return _json({"markers": markers})


# =========================================================
# API: stream (optional)
# =========================================================
@app.get("/api/bots/{bot_id}/stream")
def api_bot_stream(bot_id: int):
    """
    Optional SSE feed. UI doesn't need this (polling is safer),
    but it's correct and ready for later.
    """
    b = get_bot(int(bot_id))
    if not b:
        raise HTTPException(status_code=404, detail="Bot not found")
    if bm is None:
        raise HTTPException(status_code=503, detail="Worker not initialized")

    async def gen():
        while True:
            snap = bm.snapshot(int(bot_id))
            payload = {"ts": now_ts(), "snap": snap}
            yield f"data: {json.dumps(payload)}\n\n"
            await _async_sleep(1.0)

    return StreamingResponse(gen(), media_type="text/event-stream", headers={"Cache-Control": "no-store"})



@app.get("/api/bots/{bot_id}/logstream")
def api_bot_logstream(bot_id: int):
    """Server-Sent Events stream of new logs (no page refresh needed)."""
    b = get_bot(int(bot_id))
    if not b:
        raise HTTPException(status_code=404, detail="Bot not found")

    def gen():
        last_id = 0
        last_ping = 0.0
        while True:
            now = time.time()
            if now - last_ping >= 5.0:
                last_ping = now
                yield "event: ping\ndata: {}\n\n"

            try:
                rows = list_logs_since(int(bot_id), int(last_id), limit=200)
            except Exception:
                rows = []

            for r in rows:
                try:
                    last_id = int(r.get("id") or last_id)
                except Exception:
                    pass
                payload = {"log": {"id": r.get("id"), "ts": r.get("ts"), "level": r.get("level"), "message": r.get("message")}}
                yield f"data: {json.dumps(payload)}\n\n"

            time.sleep(1.0)

    return StreamingResponse(gen(), media_type="text/event-stream", headers={"Cache-Control": "no-store"})

async def _async_sleep(sec: float):
    import asyncio
    await asyncio.sleep(sec)


# =========================================================
# API: Intelligence Dashboard
# =========================================================
@app.get("/api/intelligence/decisions")
def api_intelligence_decisions(bot_id: Optional[int] = None, limit: int = 50):
    """Get recent intelligence decisions for dashboard"""
    try:
        if bot_id:
            decisions = get_intelligence_decisions(bot_id, limit=limit)
        else:
            # Get decisions for all bots
            decisions = []
            bots = list_bots()
            for bot in bots[:10]:  # Limit to first 10 bots
                bot_decisions = get_intelligence_decisions(bot.get("id"), limit=10)
                decisions.extend(bot_decisions)
            # Sort by timestamp descending
            decisions.sort(key=lambda d: d.get("ts", 0), reverse=True)
            decisions = decisions[:limit]
        
        return _json({"ok": True, "decisions": decisions})
    except Exception as e:
        logger.error(f"Intelligence decisions error: {type(e).__name__}: {e}")
        return _json({"ok": False, "error": f"{type(e).__name__}: {e}", "decisions": []}, 500)


@app.get("/intelligence")
def ui_intelligence_dashboard(request: Request):
    """Intelligence Dashboard UI"""
    _templates = Jinja2Templates(directory="templates")
    return _templates.TemplateResponse("intelligence_dashboard.html", {"request": request})


def _scan_stock_symbol(symbol: str, horizon: str, btc_ctx: Dict[str, Any]) -> Dict[str, Any]:
    if not alpaca_paper and not alpaca_live:
         return {"symbol": symbol, "score": 0.0, "eligible": False, "reasons": ["Alpaca not ready"]}

    client = alpaca_live or alpaca_paper
    def fetch(tf):
        return client.get_ohlcv(symbol, timeframe=tf, limit=500)

    try:
        candles_1h = fetch("1h")
        candles_4h = fetch("4h")
        candles_1d = fetch("1d")
        candles_1w = fetch("1w")
    except Exception as e:
        return {"symbol": symbol, "score": 0.0, "eligible": False, "reasons": [f"Data fetch error: {e}"]}

    res = _analyze_market_data(symbol, horizon, btc_ctx, candles_1h, candles_4h, candles_1d, candles_1w)
    # Benchmark enrichment (SPY for stocks)
    try:
        from benchmark_analyzer import enrich_recommendation_with_benchmark
        from stock_metadata import get_sector
        if candles_1d and len(candles_1d) >= 30:
            _bench_ttl = float(os.getenv("SCAN_BENCHMARK_CACHE_TTL_SEC", "120"))

            def _fetch_spy_only():
                try:
                    return client.get_ohlcv("SPY", "1d", 200)
                except Exception:
                    return []

            benchmark_candles = _benchmark_ohlcv_cached("SPY_1d_200", _bench_ttl, _fetch_spy_only)
            price = float(candles_1d[-1][4]) if candles_1d else 0.0
            enriched = enrich_recommendation_with_benchmark(
                symbol, price, candles_1d=candles_1d, benchmark_candles=benchmark_candles, sector=get_sector(symbol)
            )
            metrics = res.get("metrics") or {}
            for k, v in enriched.items():
                if v is not None and v != "":
                    metrics[k] = v
            res["metrics"] = metrics
            if enriched.get("peer_quartile") == "top":
                base = float(res.get("score") or 0)
                res["score"] = min(98.0, base + 3.0)
                res.setdefault("reasons", []).append("Top-quartile in sector")
            if enriched.get("benchmark_vs"):
                res.setdefault("reasons", []).append(enriched["benchmark_vs"])
    except Exception as e:
        logger.debug("Benchmark enrichment failed for stock %s: %s", symbol, e)
    try:
        res["_candles_1d"] = [list(x) for x in (candles_1d or [])]
    except Exception:
        res["_candles_1d"] = []
    return res


@app.post("/api/recommendations/scan_stocks")
def api_recommendations_scan_stocks(horizon: str = "short", limit: int = 150):
    if not alpaca_paper and not alpaca_live:
         return _json({"ok": False, "error": "Alpaca not configured"}, 503)
         
    client = alpaca_live if alpaca_live else alpaca_paper
    
    # 1. Build Stock Universe
    universe = []
    
    # Preset: "Mega-Cap + Popular"
    # Ideally we'd fetch active assets, but searching all active is heavy.
    # We'll use a larger static list + some dynamic checks if possible.
    # For now, let's use a robust list of liquid stocks/ETFs.
    
    # Tech / Growth
    universe += ["AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META", "TSLA", "AMD", "NFLX", "INTC", "QCOM", "CRM", "ADBE", "AVGO", "TXN"]
    # Financials
    universe += ["JPM", "BAC", "V", "MA", "WFC", "GS", "MS", "BLK", "C", "AXP"]
    # ETFs (benchmark-only — scanned for data, not emitted as BUY signals)
    _BENCHMARK_ONLY = {"SPY", "QQQ", "IWM", "DIA", "XLK", "XLF", "XLE", "XLV", "XLY", "XLP", "XLI", "GLD", "SLV", "TQQQ", "SQQQ", "SOXL", "ARKK"}
    universe += list(_BENCHMARK_ONLY)
    # Crypto proxies
    universe += ["COIN", "MSTR", "MARA", "RIOT", "CLSK", "HUT", "BITF", "HOOD"]
    # Retail / Meme / High Vol
    universe += ["GME", "AMC", "ROKU", "PLTR", "SOFI", "UBER", "LYFT", "DKNG", "AFRM", "UPST", "CVNA"]
    # Defensive / Value
    universe += ["JNJ", "PG", "KO", "PEP", "WMT", "COST", "TGT", "HD", "LOW", "MCD", "DIS", "T", "VZ", "PFE", "MRK", "UNH"]
    
    # Deduplicate
    universe = list(set(universe))

    # --- EXPANDED UNIVERSE START ---
    # Fill up to 'limit' with random active assets to discover new stocks
    if len(universe) < limit:
        try:
            import random
            all_assets = client.get_active_assets()
            # Filter somewhat for quality (e.g. marginable usually implies better liquidity/status)
            candidates = [
                a["symbol"] for a in all_assets
                if a.get("symbol") not in universe
                and a.get("marginable") # simplistic quality filter
                and "." not in a.get("symbol", "") # avoid weird warrants/classes
                and "-" not in a.get("symbol", "") # BRK-B style tickers break Yahoo Finance
                and len(a.get("symbol", "")) <= 5  # skip long symbols (ETNs, leveraged, etc.)
            ]
            
            needed = limit - len(universe)
            if needed > 0 and candidates:
                # Random sample
                if len(candidates) > needed:
                    fill = random.sample(candidates, needed)
                else:
                    fill = candidates
                universe.extend(fill)
        except Exception as e:
            logger.warning("Error expanding universe: %s", e)
    # --- EXPANDED UNIVERSE END ---

    # Dynamic: Add Top Gainers (if feasible) to ensure we find "Strong Buys" even if market is mixed
    try:
        movers = client.get_top_movers()
        gainers = [x["symbol"] for x in movers.get("gainers", [])]
        universe.extend(gainers)
        # Re-deduplicate
        universe = list(set(universe))
    except Exception:
        pass
    
    results = []
    processed = 0
    errors = 0
    
    # Get BTC context for correlation? Stocks have their own beta.
    # We'll use SPY regime as context.
    spy_ctx = {"risk_off": False}
    try:
        spy_c = client.get_ohlcv("SPY", "1d", 200)
        spy_r = detect_regime(spy_c)
        if spy_r.regime in ("TREND_DOWN", "CRASH") or (spy_r.scores or {}).get("downtrend_score", 0) > 0.65:
            spy_ctx["risk_off"] = True
    except Exception as e:
        logger.debug("SPY regime check failed: %s", e)

    try:
        mark_explore_signals_pending(horizon, now_ts())
    except Exception as _msp:
        logger.warning("mark_explore_signals_pending (scan_stocks): %s", _msp)

    # Batch processing to avoid rate limits
    chunk_size = 50
    for i in range(0, min(len(universe), int(limit)), chunk_size):
        chunk = universe[i:i+chunk_size]
        
        # We can pre-fetch snapshots to check volume/price filter
        try:
            snaps = client.get_snapshots(chunk)
        except Exception as e:
            logger.debug("Batch snapshots failed: %s", e)
            snaps = {}
             
        for sym in chunk:
            try:
                # Pre-filter Check
                snap = snaps.get(sym)
                if snap:
                    price = 0.0
                    vol = 0.0
                    if snap.get("dailyBar"):
                        price = float(snap["dailyBar"].get("c", 0))
                        vol = float(snap["dailyBar"].get("v", 0)) * price
                    elif snap.get("latestTrade"):
                         price = float(snap["latestTrade"].get("p", 0))
                    
                    # Skip penny stocks or illiquid
                    if price < 5.0 or vol < 500000: # Min $5 price, $500k volume
                        continue

                # Benchmark-only: scan for data but cap score so they never appear as BUY
                _is_benchmark = sym in _BENCHMARK_ONLY
                res = _scan_stock_symbol(sym, horizon, spy_ctx)
                
                if _is_benchmark and res.get("score") is not None:
                    res["score"] = min(float(res.get("score", 0)), 30.0)
                    res.setdefault("risk_flags", []).append("BENCHMARK_ONLY")
                
                if res.get("score") is not None:
                     metrics = dict(res.get("metrics") or {})
                     metrics["market_type"] = "stocks"  # TAG AS STOCKS
                     if res.get("change_24h") is not None:
                         metrics["change_24h"] = res["change_24h"]
                     # Persist top-level recommendation fields into metrics_json
                     # so tiebreaker sorting works at API serve time.
                     for _persist_key in ("entry_quality", "rsi_value", "volume_ratio",
                                          "confidence_band", "volume_anomaly"):
                         _val = res.get(_persist_key)
                         if _val is not None:
                             metrics[_persist_key] = _val
                     res["metrics"] = metrics
                     _merge_evaluate_signal_into_snap(res, horizon, spy_ctx)
                     metrics = dict(res.get("metrics") or {})
                     _c1d_st = res.get("_candles_1d") or []
                     _sid_st = save_recommendation_snapshot(
                        symbol=sym,
                        horizon=horizon,
                        score=float(res.get("score") or 0.0),
                        regime_json=json.dumps(res.get("regime") or {}),
                        metrics_json=json.dumps(metrics),
                        reasons_json=json.dumps(res.get("reasons") or []),
                        risk_flags_json=json.dumps(res.get("risk_flags") or []),
                        score_breakdown_json=res.get("score_breakdown_json") or json.dumps(res.get("score_breakdown") or {}),
                        composite_score=None,
                        confidence_score=None,
                        conviction_grade=None,
                        factor_scores_json="",
                        signal_flags_json="",
                    )
                     _persist_explore_feed_from_snap(sym, horizon, res, _c1d_st, spy_ctx)
                     processed += 1
                     results.append(res)
            except Exception:
                errors += 1
                continue
                
    _RECO_RESULT_CACHE.clear()  # Force fresh scoring on next Explore load
    return _json({
        "ok": True, 
        "message": f"Scanned {processed} stocks",
        "processed": processed,
        "errors": errors
    })


# =========================================================
# Autopilot API
# =========================================================
_autopilot_last_run: float = 0.0
_autopilot_next_run: float = 0.0


def _autopilot_loop() -> None:
    """Background loop: run autopilot cycle every scan_interval when enabled."""
    import autopilot
    import autopilot as ap_mod
    global _autopilot_last_run, _autopilot_next_run
    interval = int(getattr(ap_mod, "AUTOPILOT_SCAN_INTERVAL_SEC", 14400))
    try:
        cfg = ap_mod.get_autopilot_config()
        hours = cfg.get("scan_interval_hours")
        if hours is not None and int(hours) > 0:
            interval = int(hours) * 3600
    except Exception:
        pass
    logger.info("autopilot_loop started, interval=%ds", interval)
    while True:
        try:
            time.sleep(min(60, interval // 10))
            enabled = ap_mod.is_autopilot_enabled()
            if not enabled:
                continue
            now = time.time()
            if now < _autopilot_next_run:
                continue
            _autopilot_next_run = now + interval
            _autopilot_last_run = now
            logger.info("autopilot cycle starting (next in %ds)", interval)

            def _create_bot_fn(payload):
                return create_bot(payload)

            def _delete_bot_fn(bot_id):
                delete_bot(int(bot_id))

            def _start_bot_fn(bot_id):
                if bm:
                    try:
                        bot = get_bot(int(bot_id))
                        if bot and _can_start_bot_live(bot)[0]:
                            bm.start(int(bot_id))
                    except Exception as e:
                        logger.warning("autopilot start_bot %s: %s", bot_id, e)

            def _stop_bot_fn(bot_id):
                if bm:
                    try:
                        bm.stop(int(bot_id))
                    except Exception as e:
                        logger.warning("autopilot stop_bot %s: %s", bot_id, e)

            def _get_portfolio_fn():
                try:
                    snap = _portfolio_snapshot()
                    t = float(snap.get("total_usd") or 0)
                    if t > 0:
                        return t
                    if bm and hasattr(bm, "get_portfolio_total"):
                        return float(bm.get_portfolio_total())
                except Exception:
                    pass
                return 0.0

            notify_fn = None
            try:
                from notification_manager import notify
                notify_fn = notify
            except Exception:
                pass

            res = ap_mod.run_autopilot_cycle(
                create_bot_fn=_create_bot_fn,
                delete_bot_fn=_delete_bot_fn,
                start_bot_fn=_start_bot_fn,
                stop_bot_fn=_stop_bot_fn,
                get_portfolio_total_fn=_get_portfolio_fn,
                notify_fn=notify_fn,
            )
            logger.info("autopilot cycle complete: created=%s closed=%s skipped=%s",
                        res.get("created", 0), res.get("closed", 0), len(res.get("skipped", [])))

            # Stale detection: alert if heartbeat >24h or capital mismatch >20%
            try:
                _hb_ts = int(get_setting("autopilot_last_heartbeat_ts", "0") or 0)
                _hb_age = now - _hb_ts if _hb_ts > 0 else None
                _cfg_capital = float(cfg.get("total_capital", 0))
                _actual = _get_portfolio_fn()
                _stale_msgs = []
                if _hb_age is not None and _hb_age > 86400:
                    _stale_msgs.append(f"Heartbeat is {_hb_age / 86400:.1f}d old")
                if _cfg_capital > 0 and _actual > 0 and abs(_cfg_capital - _actual) / _cfg_capital > 0.20:
                    _stale_msgs.append(f"Capital mismatch: cfg ${_cfg_capital:.0f} vs actual ${_actual:.0f}")
                if _stale_msgs and notify_fn:
                    notify_fn(
                        "autopilot_stale",
                        f"Autopilot stale warning: {'; '.join(_stale_msgs)}. Review and re-configure if needed.",
                        level="warning",
                    )
                    logger.warning("Autopilot stale: %s", "; ".join(_stale_msgs))
            except Exception:
                pass

        except Exception as e:
            logger.exception("autopilot loop error: %s", e)


_AUTOPILOT_STALE_ALERTED = False  # one-shot flag to avoid spam


@app.get("/api/autopilot/activity")
def api_autopilot_activity(request: Request):
    """Return latest autopilot audit log entries for the dashboard."""
    try:
        raw = request.query_params.get("limit") if request else None
        limit = min(100, int(raw) if raw not in (None, "") else 50)
    except (TypeError, ValueError):
        limit = 50
    try:
        from db import list_autopilot_audit_log
        rows = list_autopilot_audit_log(limit=limit)
    except Exception as e:
        logger.warning("autopilot activity log failed: %s", e)
        rows = []
    for r in rows:
        if r.get("details_json"):
            try:
                r["details"] = json.loads(r["details_json"])
            except Exception:
                r["details"] = None
    return _json({"ok": True, "items": rows})


@app.get("/api/autopilot/config")
def api_autopilot_config_get():
    """Get autopilot_config from db (Master Upgrade Part 4)."""
    from db import get_autopilot_config_row
    row = get_autopilot_config_row()
    if not row:
        return _json({"ok": True, "config": {}})
    cfg = {k: v for k, v in row.items() if k != "id"}
    return _json({"ok": True, "config": cfg})


@app.post("/api/autopilot/config")
async def api_autopilot_config_save(request: Request):
    """Save autopilot_config (Master Upgrade Part 4)."""
    from db import save_autopilot_config
    body = await request.json()
    save_autopilot_config(body or {})
    return _json({"ok": True})


@app.post("/api/autopilot/config/update")
async def api_autopilot_config_update(request: Request):
    """Merge body into settings-stored autopilot_config (used by dashboard edit)."""
    import autopilot
    body = await request.json() or {}
    cfg = dict(autopilot.get_autopilot_config())
    for k, v in body.items():
        if v is not None and k != "id":
            cfg[k] = v
    set_setting("autopilot_config", json.dumps(cfg))
    return _json({"ok": True, "config": cfg})


@app.get("/api/autopilot/status")
def api_autopilot_status():
    """Always return 200 so UI shows a message instead of 500. On error, ok=False + error."""
    try:
        import autopilot
        enabled = autopilot.is_autopilot_enabled()
        cfg = autopilot.get_autopilot_config()
    except Exception as e:
        logger.exception("api_autopilot_status: autopilot config failed")
        return _json({
            "ok": False,
            "error": str(e)[:300],
            "enabled": False,
            "config": {},
            "last_run_ts": 0,
            "next_scan_in_sec": 0,
            "last_autopilot_heartbeat_ts": None,
            "portfolio_value": 0.0,
            "active_positions": 0,
            "max_positions": 6,
            "total_pnl": 0.0,
        })
    now = time.time()
    next_sec = max(0, int(_autopilot_next_run - now)) if _autopilot_next_run > now else 0
    portfolio_value = 0.0
    active_positions = 0
    total_pnl = 0.0
    try:
        # Use cached portfolio data (no network call) to keep this endpoint fast
        with _globals_lock:
            if PORT_HISTORY:
                portfolio_value = float(PORT_HISTORY[-1].get("total_usd") or 0)
        bots = list_bots()
        active_positions = sum(1 for b in bots if int(b.get("enabled", 0)) == 1)
        from db import all_deal_stats
        stats = all_deal_stats()
        total_pnl = float(stats.get("realized_total", 0) or 0)
    except Exception:
        pass
    try:
        last_heartbeat = int(get_setting("autopilot_last_heartbeat_ts", "0") or 0)
    except Exception:
        last_heartbeat = 0
    last_run_ts = int(_autopilot_last_run)
    if last_run_ts <= 0 and last_heartbeat > 0:
        last_run_ts = last_heartbeat
    # Stale detection: heartbeat >24h or capital mismatch >20%
    stale_warnings = []
    is_stale = False
    _heartbeat_age_sec = int(now - last_heartbeat) if last_heartbeat > 0 else None
    if last_heartbeat > 0 and _heartbeat_age_sec > 86400:
        is_stale = True
        _stale_days = _heartbeat_age_sec / 86400
        stale_warnings.append(f"Autopilot heartbeat is {_stale_days:.1f} days old (>24h)")

    config_capital = float(cfg.get("total_capital", 0))
    if config_capital > 0 and portfolio_value > 0:
        _capital_pct_diff = abs(config_capital - portfolio_value) / config_capital * 100
        if _capital_pct_diff > 20:
            is_stale = True
            stale_warnings.append(
                f"Capital mismatch: configured ${config_capital:.0f} vs actual ${portfolio_value:.0f} ({_capital_pct_diff:.0f}% off)"
            )

    return _json({
        "ok": True,
        "enabled": enabled,
        "config": cfg,
        "total_capital": float(cfg.get("total_capital", 0)),
        "capital_per_bot": float(cfg.get("capital_per_bot", 0)),
        "max_positions": int(cfg.get("max_positions") or 6),
        "last_run_ts": last_run_ts,
        "next_scan_in_sec": next_sec,
        "last_autopilot_heartbeat_ts": last_heartbeat if last_heartbeat else None,
        "heartbeat_age_sec": _heartbeat_age_sec,
        "portfolio_value": portfolio_value,
        "active_positions": active_positions,
        "total_pnl": total_pnl,
        "is_stale": is_stale,
        "stale_warnings": stale_warnings,
    })


@app.get("/api/autopilot/positions")
def api_autopilot_positions():
    """Active bots with live P&L for dashboard (123.md Fix 3)."""
    bots = [b for b in list_bots() if int(b.get("enabled", 0)) == 1]
    out = []
    for b in bots:
        bot_id = int(b.get("id", 0))
        snap = {}
        if bm:
            try:
                snap = bm.snapshot(bot_id) or {}
            except Exception:
                pass
        last_price = float(snap.get("last_price") or 0)
        if last_price <= 0:
            try:
                sym = b.get("symbol", "")
                if sym:
                    if classify_symbol(sym) == "stock" and (alpaca_live or alpaca_paper):
                        client = alpaca_live or alpaca_paper
                        t = client.get_ticker(sym)
                        last_price = float(t.get("last") or 0)
                    else:
                        tc = _ticker_cached(sym, ttl_sec=60)
                        if tc:
                            last_price = float(tc.get("last") or tc.get("c") or 0)
            except Exception:
                pass
        avg_entry = float(snap.get("avg_entry") or 0)
        base_pos = float(snap.get("base_pos") or 0)
        if avg_entry <= 0 and base_pos > 0 and last_price > 0:
            avg_entry = last_price
        position_value = base_pos * last_price if last_price > 0 else base_pos
        unrealized_pnl = 0.0
        unrealized_pnl_pct = 0.0
        if avg_entry > 0 and base_pos > 0:
            unrealized_pnl = (last_price - avg_entry) * base_pos
            unrealized_pnl_pct = ((last_price - avg_entry) / avg_entry) * 100
        out.append({
            "id": bot_id,
            "symbol": b.get("symbol", ""),
            "strategy": b.get("strategy_mode", "classic"),
            "enabled": int(b.get("enabled", 0)),
            "avg_entry_price": avg_entry,
            "current_price": last_price,
            "position_value": position_value,
            "quantity": base_pos,
            "unrealized_pnl": unrealized_pnl,
            "unrealized_pnl_pct": unrealized_pnl_pct,
            "take_profit_price": snap.get("tp_price"),
            "stop_loss_price": None,
            "trading_mode": b.get("trading_mode", "swing_trade"),
        })
    return _json({"ok": True, "positions": out})


@app.post("/api/autopilot/toggle")
def api_autopilot_toggle():
    val = get_setting("autopilot_enabled", "0")
    new_val = "0" if str(val).strip().lower() in ("1", "true", "yes", "y", "on") else "1"
    set_setting("autopilot_enabled", new_val)
    return _json({"ok": True, "enabled": new_val == "1"})


@app.post("/api/autopilot/start")
def api_autopilot_start():
    """Start ALL autopilot bots: enable setting + enable all bot_type='autopilot' bots + start in BotManager."""
    set_setting("autopilot_enabled", "1")
    n = update_bots_by_type("autopilot", 1)
    autopilot_bots = [b for b in list_bots() if str(b.get("bot_type") or "").lower() == "autopilot"]
    started = 0
    for bot in autopilot_bots:
        bot_id = int(bot.get("id", 0))
        if bot_id and bm:
            try:
                if _can_start_bot_live(bot)[0]:
                    bm.start(bot_id)
                    started += 1
            except Exception as e:
                logger.warning("autopilot start bot %s: %s", bot_id, e)
    logger.info("Autopilot STARTED - %d bots enabled, %d started in BotManager", n, started)
    return _json({"ok": True, "enabled": True, "bots_updated": n, "bots_started": started})


@app.post("/api/autopilot/stop")
def api_autopilot_stop():
    """Stop ALL autopilot bots: disable setting + disable all bot_type='autopilot' bots + stop in BotManager."""
    set_setting("autopilot_enabled", "0")
    n = update_bots_by_type("autopilot", 0)
    autopilot_bots = [b for b in list_bots() if str(b.get("bot_type") or "").lower() == "autopilot"]
    stopped = 0
    for bot in autopilot_bots:
        bot_id = int(bot.get("id", 0))
        if bot_id and bm:
            try:
                bm.stop(bot_id)
                stopped += 1
            except Exception as e:
                logger.warning("autopilot stop bot %s: %s", bot_id, e)
    logger.info("Autopilot STOPPED - %d bots disabled, %d stopped in BotManager", n, stopped)
    return _json({"ok": True, "enabled": False, "bots_updated": n, "bots_stopped": stopped})


@app.post("/api/autopilot/run")
def api_autopilot_run():
    """Run one autopilot cycle. Returns detailed diagnostic info so the UI never silently fails."""
    global _autopilot_last_run, _autopilot_next_run
    import autopilot
    import autopilot as ap_mod

    paused = bool(_pause_state())
    kill_switch = bool(get_setting("kill_switch", "0") in ("1", "true"))
    bm_ready = bm is not None
    kraken_ready = _kraken_ready()

    if paused:
        return _json({
            "ok": False, "error": "Trading is paused (global pause active). Resume from Safety page before running autopilot.",
            "created": 0, "closed": 0, "created_bots": [], "skipped": [],
            "bm_ready": bm_ready, "kraken_ready": kraken_ready, "paused": True, "kill_switch": kill_switch,
        })
    if kill_switch:
        return _json({
            "ok": False, "error": "Kill switch is ON. Disable it before running autopilot.",
            "created": 0, "closed": 0, "created_bots": [], "skipped": [],
            "bm_ready": bm_ready, "kraken_ready": kraken_ready, "paused": paused, "kill_switch": True,
        })
    if not bm_ready:
        reason = _bm_not_ready_reason() or "BotManager not initialized"
        return _json({
            "ok": False, "error": f"BotManager not ready: {reason}. Server may still be starting up.",
            "created": 0, "closed": 0, "created_bots": [], "skipped": [],
            "bm_ready": False, "kraken_ready": kraken_ready, "paused": paused, "kill_switch": kill_switch,
            "reason": reason,
        }, 503)

    def _create_bot_fn(payload):
        return create_bot(payload)

    def _delete_bot_fn(bot_id):
        delete_bot(int(bot_id))

    def _start_bot_fn(bot_id):
        if bm:
            try:
                bot = get_bot(int(bot_id))
                if bot and _can_start_bot_live(bot)[0]:
                    bm.start(int(bot_id))
            except Exception as e:
                logger.warning("autopilot start_bot %s: %s", bot_id, e)

    def _stop_bot_fn(bot_id):
        if bm:
            try:
                bm.stop(int(bot_id))
            except Exception as e:
                logger.warning("autopilot stop_bot %s: %s", bot_id, e)

    def _get_portfolio_fn():
        try:
            snap = _portfolio_snapshot()
            t = float(snap.get("total_usd") or 0)
            if t > 0:
                return t
            if bm and hasattr(bm, "get_portfolio_total"):
                return float(bm.get_portfolio_total())
        except Exception:
            pass
        return 0.0

    notify_fn = None
    try:
        from notification_manager import notify
        notify_fn = notify
    except Exception:
        pass

    res = ap_mod.run_autopilot_cycle(
        create_bot_fn=_create_bot_fn,
        delete_bot_fn=_delete_bot_fn,
        start_bot_fn=_start_bot_fn,
        stop_bot_fn=_stop_bot_fn,
        get_portfolio_total_fn=_get_portfolio_fn,
        notify_fn=notify_fn,
        force_run=True,
    )
    _autopilot_last_run = time.time()
    interval = getattr(ap_mod, "AUTOPILOT_SCAN_INTERVAL_SEC", 14400)
    try:
        cfg = ap_mod.get_autopilot_config()
        if cfg.get("scan_interval_hours"):
            interval = int(cfg["scan_interval_hours"]) * 3600
    except Exception:
        pass
    _autopilot_next_run = time.time() + interval

    created_count = res.get("created", 0)
    created_bots = res.get("created_bots", [])
    skipped = res.get("skipped", [])
    candidates = res.get("candidates_considered", 0)
    no_reason = res.get("no_candidates_reason")
    errors = res.get("errors", [])
    closed = res.get("closed", 0)

    success = created_count > 0 or closed > 0 or (candidates == 0 and not errors)
    error_msg = None
    if created_count == 0 and not errors:
        if candidates == 0:
            error_msg = no_reason or "No recommendation candidates found. Run 'Scan recommendations' first, then retry."
        elif len(skipped) > 0:
            reasons = set(s.get("reason", "") for s in skipped[:5])
            error_msg = f"Found {candidates} candidate(s) but all were skipped: {', '.join(reasons)}"
        else:
            error_msg = f"Cycle ran but created 0 bots from {candidates} candidate(s). Check min_score, max_positions, or slot availability."

    return _json({
        "ok": success and not errors,
        "created": created_count,
        "closed": closed,
        "created_bots": created_bots[:20],
        "skipped": skipped[:30],
        "candidates_considered": candidates,
        "errors": errors,
        "error": error_msg,
        "bm_ready": bm_ready,
        "kraken_ready": kraken_ready,
        "paused": paused,
        "kill_switch": kill_switch,
        "slots_used": res.get("slots_used", 0),
        "max_positions": res.get("max_positions", 0),
    })


@app.post("/api/autopilot/create_bots")
async def api_autopilot_create_bots(request: Request):
    """
    Create N bots from top recommendations (portfolio-aware: skips symbols already active).
    Does: create -> enabled=1 -> start. Returns per-bot result for clear error reporting.
    """
    import autopilot
    body = await request.json() or {}
    count = max(1, min(int(body.get("count", 1)), 20))
    dry_run = int(body.get("dry_run", 1))
    horizon = str(body.get("horizon", "long")).lower() or "long"
    if horizon not in ("short", "long"):
        horizon = "long"
    cfg = autopilot.get_autopilot_config()
    min_score = float(body.get("min_score") or cfg.get("min_score") or cfg.get("min_score_threshold") or 75)
    min_score = max(50, min(95, min_score))
    asset_filter = str(body.get("asset_filter") or cfg.get("asset_types") or "both")
    sectors_avoid = cfg.get("sectors_avoid") or []
    if isinstance(sectors_avoid, str):
        sectors_avoid = [s.strip() for s in sectors_avoid.split(",") if s.strip()]

    # Build active symbol set (duplicate-symbol prevention)
    active_symbols, active_reason = _active_symbol_set()
    top = autopilot.get_top_recommendations(
        horizon=horizon,
        min_score=min_score,
        max_count=count * 3,
        asset_filter=asset_filter,
        sectors_avoid=sectors_avoid,
    )
    candidates = []
    skipped = []  # detailed reason per skipped symbol
    for r in top:
        sym = str(r.get("symbol") or "")
        if not sym:
            continue
        norm = _normalize_symbol(sym)
        if norm in active_symbols:
            reasons = active_reason.get(norm) or ["already_active"]
            skipped.append({"symbol": sym, "reason": "already_active", "detail": reasons})
            continue
        candidates.append(r)
        if len(candidates) >= count:
            break

    if not candidates:
        return _json({
            "ok": False,
            "error": "No candidates available. Either no recommendations above min_score or all top symbols already have a bot/open position.",
            "created": [],
            "errors": [],
            "skipped": skipped[:20],
        })

    # Kraken must be ready for crypto bots
    if not _kraken_ready():
        return _json({
            "ok": False,
            "error": f"Kraken not ready: {KRAKEN_ERROR or 'API not configured or unreachable'}",
            "created": [],
            "errors": [],
        })

    if not bm:
        reason = _bm_not_ready_reason() or "BotManager not initialized"
        return _json({
            "ok": False, "error": "BotManager not initialized", "reason": reason,
            "created": [], "errors": [],
        }, 503)

    strategy = str(body.get("strategy_mode") or cfg.get("strategy_mode") or "smart_dca")
    base_quote = float(body.get("base_quote") or cfg.get("capital_per_bot") or 500)
    safety_quote = float(body.get("safety_quote") or base_quote * 0.2)
    max_safety = int(body.get("max_safety") or 3)
    tp = float(body.get("tp") or 0.03)
    market_default = "crypto"
    profile = str(cfg.get("risk_profile") or "balanced").lower()
    _pp = {"conservative": {"tp": 0.02, "stop_loss_pct": 0.05}, "balanced": {"tp": 0.03, "stop_loss_pct": 0.08}, "aggressive": {"tp": 0.05, "stop_loss_pct": 0.12}}
    pp = _pp.get(profile, _pp["balanced"])
    created_list = []
    errors_list = []
    for rec in candidates[:count]:
        sym = str(rec.get("symbol") or "")
        metrics = rec.get("metrics") or {}
        market = (metrics.get("market_type") or "crypto").strip().lower()
        if market == "stock":
            market = "stocks"
        try:
            payload = {
                "name": f"Autopilot {sym}",
                "symbol": sym,
                "bot_type": "autopilot",
                "enabled": 1,
                "dry_run": dry_run,
                "strategy_mode": str(metrics.get("strategy") or metrics.get("recommended_strategy") or strategy),
                "forced_strategy": "",
                "base_quote": base_quote,
                "safety_quote": safety_quote,
                "max_safety": max_safety,
                "first_dev": 0.015,
                "step_mult": 1.2,
                "tp": float(body.get("tp") or pp["tp"] or tp),
                "max_spend_quote": base_quote + safety_quote * max_safety,
                "poll_seconds": 10,
                "max_open_orders": 6,
                "market_type": market,
                "alpaca_mode": str(cfg.get("alpaca_mode") or "paper"),
                "auto_restart": 1,
                "stop_loss_pct": pp.get("stop_loss_pct", 0.08),
            }
            bot_id = create_bot(payload)
            bot = get_bot(int(bot_id))
            if bot and bm and _can_start_bot_live(bot)[0]:
                try:
                    bm.start(int(bot_id))
                except Exception as e:
                    errors_list.append(f"Start {sym}: {e}")
            active_symbols.add(_normalize_symbol(sym))
            created_list.append({"ok": True, "bot_id": bot_id, "symbol": sym})
        except Exception as e:
            err_msg = str(e)[:200]
            created_list.append({"ok": False, "bot_id": None, "symbol": sym, "error": err_msg})
            errors_list.append(f"Create {sym}: {err_msg}")
    return _json({
        "ok": True,
        "created": created_list,
        "errors": errors_list,
        "skipped": skipped[:30],
        "summary": f"chose {len(candidates)} candidate(s); created {len([c for c in created_list if c.get('ok')])} bot(s).",
    })


@app.get("/api/autopilot/top")
def api_autopilot_top():
    import autopilot
    cfg = autopilot.get_autopilot_config()
    min_score = float(cfg.get("min_score") or cfg.get("min_score_threshold") or getattr(autopilot, "AUTOPILOT_MIN_SCORE", 75))
    min_score = max(50, min(95, min_score))
    items = autopilot.get_top_recommendations(
        min_score=min_score,
        max_count=15,
        asset_filter=cfg.get("asset_types"),
        sectors_avoid=cfg.get("sectors_avoid"),
    )
    out = []
    for r in items:
        m = r.get("metrics") or {}
        reason = m.get("explanation") or m.get("recommended_strategy") or m.get("reason") or ""
        out.append({
            "symbol": r.get("symbol"),
            "score": r.get("score"),
            "horizon": "long",
            "reason": (str(reason)[:200] if reason else None),
        })
    return _json({"ok": True, "items": out})


@app.get("/api/opportunities/now")
def api_now_opportunities(max_count: int = 3, asset_filter: Optional[str] = None):
    """
    Return best opportunities right now for radar/autopilot.

    Uses autopilot config, recommendations, and active bots to pick top candidates.
    """
    import autopilot
    try:
        max_count = max(1, min(int(max_count), 10))
    except Exception:
        max_count = 3

    cfg = autopilot.get_autopilot_config()
    opp_defaults = cfg.get("opportunity_defaults") or {}
    if not asset_filter:
        asset_filter = str(opp_defaults.get("asset_filter") or cfg.get("asset_types") or "both")
    min_score = float(
        opp_defaults.get("min_score")
        or cfg.get("min_score")
        or cfg.get("min_score_threshold")
        or getattr(autopilot, "AUTOPILOT_MIN_SCORE", 75)
    )
    min_score = max(50, min(95, min_score))

    items = autopilot.get_now_opportunities(
        asset_filter=asset_filter,
        max_count=max_count,
        min_score=min_score,
    )
    return _json(
        {
            "ok": True,
            "items": items,
            "config": {
                "asset_filter": asset_filter,
                "min_score": min_score,
            },
        }
    )


@app.get("/api/autopilot/watchlist")
def api_autopilot_watchlist():
    import autopilot
    cfg = autopilot.get_autopilot_config()
    items = autopilot.get_watchlist(asset_filter=cfg.get("asset_types"))
    out = []
    for r in items:
        m = r.get("metrics") or {}
        reason = m.get("explanation") or m.get("recommended_strategy") or ""
        out.append({
            "symbol": r.get("symbol"),
            "score": r.get("score"),
            "horizon": "long",
            "reason": (str(reason)[:200] if reason else None),
        })
    return _json({"ok": True, "items": out})


@app.get("/api/scanner/status")
def api_scanner_status():
    """Return scanner status with last scan time, next scan time, and interval.
    Used by frontend countdown timer."""
    now_t = time.time()
    
    # Get last scan time from any horizon
    last_scan_ts = 0
    with _globals_lock:
        for h in ("short", "medium", "long"):
            st = _RECO_STATE.get(h) or {}
            ts = st.get("last_run_ts", 0)
            if ts > last_scan_ts:
                last_scan_ts = ts
    
    # Default scan interval (15 minutes for short horizon)
    scan_intervals = {
        "short": RECO_SHORT_EVERY_SEC,   # 1800s (30 min)
        "medium": RECO_MEDIUM_EVERY_SEC, # 3600s (60 min)
        "long": RECO_LONG_EVERY_SEC,     # 7200s (120 min)
    }
    
    interval_sec = scan_intervals.get("short", 1800)
    
    # Calculate next scan time
    next_scan_ts = last_scan_ts + interval_sec if last_scan_ts > 0 else now_t + interval_sec
    
    # If next scan is in the past, calculate next one
    if next_scan_ts < now_t:
        cycles_elapsed = int((now_t - last_scan_ts) / interval_sec)
        next_scan_ts = last_scan_ts + (cycles_elapsed + 1) * interval_sec
    
    return _json({
        "ok": True,
        "last_scan": int(last_scan_ts) if last_scan_ts > 0 else None,
        "next_scan": int(next_scan_ts),
        "interval_seconds": interval_sec,
        "last_scan_ago_sec": int(now_t - last_scan_ts) if last_scan_ts > 0 else None,
        "next_scan_in_sec": max(0, int(next_scan_ts - now_t)),
    })


@app.get("/api/scanner/watchlist")
def api_scanner_watchlist(limit: int = 50):
    """Get the scanner watchlist: symbols identified as good setups but not yet READY."""
    try:
        import autopilot
        items = autopilot.get_scanner_watchlist(limit=min(int(limit), 100))
        out = []
        for entry in items:
            setup = {}
            try:
                setup = json.loads(entry.get("setup_json") or "{}")
            except Exception:
                pass
            out.append({
                "symbol": entry.get("symbol"),
                "market_type": entry.get("market_type"),
                "regime": entry.get("regime"),
                "entry_type": entry.get("entry_type"),
                "confidence": entry.get("confidence"),
                "edge_score": entry.get("edge_score"),
                "trigger_conditions": entry.get("trigger_conditions"),
                "evidence": setup.get("evidence", []),
                "created_at": entry.get("created_at"),
                "updated_at": entry.get("updated_at"),
                "status": entry.get("status"),
            })
        return _json({"ok": True, "items": out, "count": len(out)})
    except Exception as e:
        logger.error("Scanner watchlist API failed: %s", e)
        return _json({"ok": False, "error": str(e), "items": []}, 500)


@app.post("/api/scanner/watchlist")
async def api_add_to_watchlist(request: Request):
    """Add a symbol to the scanner watchlist (manual save from explore)."""
    try:
        body = await request.json()
        symbol = str(body.get("symbol") or "").strip()
        if not symbol:
            return _json({"ok": False, "error": "symbol required"}, 400)
        market_type = "crypto" if "/" in symbol or len(symbol) > 6 else "stocks"
        from db import upsert_watchlist_entry
        rid = upsert_watchlist_entry(
            symbol=symbol,
            market_type=market_type,
            setup_json="{}",
            trigger_conditions="Manual add from Explore",
            regime="",
            entry_type="manual",
            confidence=0.0,
            edge_score=0.5,
        )
        return _json({"ok": True, "id": rid, "symbol": symbol})
    except Exception as e:
        logger.error("Add to watchlist failed: %s", e)
        return _json({"ok": False, "error": str(e)}, 500)


@app.post("/api/autopilot/setup")
async def api_autopilot_setup(request: Request):
    from portfolio_initializer import compute_optimal_allocation, save_autopilot_config
    body = await request.json()
    total_capital = float(body.get("total_capital") or 10000)
    risk_tolerance = str(body.get("risk_tolerance") or "moderate")
    asset_types = str(body.get("asset_types") or "both")
    max_positions = int(body.get("max_positions") or 10)
    max_bots_per_sector = body.get("max_bots_per_sector")
    if max_bots_per_sector is not None:
        max_bots_per_sector = max(1, min(10, int(max_bots_per_sector)))
    sectors_avoid = body.get("sectors_avoid") or []
    if isinstance(sectors_avoid, str):
        sectors_avoid = [s.strip() for s in sectors_avoid.split(",") if s.strip()]
    dry_run = int(body.get("dry_run", 1))
    min_score = body.get("min_score")
    if min_score is not None:
        min_score = max(50, min(95, float(min_score)))
    scan_interval_hours = body.get("scan_interval_hours")
    if scan_interval_hours is not None:
        scan_interval_hours = max(1, min(168, int(scan_interval_hours)))
    auto_delete_closed = body.get("auto_delete_closed", False)

    alloc = compute_optimal_allocation(
        total_capital=total_capital,
        risk_tolerance=risk_tolerance,
        max_positions=max_positions,
        asset_types=asset_types,
        sectors_avoid=sectors_avoid,
    )
    alloc["dry_run"] = dry_run
    alloc["alpaca_mode"] = "paper" if dry_run else str(body.get("alpaca_mode") or "live")
    if min_score is not None:
        alloc["min_score"] = min_score
    if scan_interval_hours is not None:
        alloc["scan_interval_hours"] = scan_interval_hours
    alloc["auto_delete_closed"] = bool(auto_delete_closed)
    if max_bots_per_sector is not None:
        alloc["max_bots_per_sector"] = max_bots_per_sector
    # Preserve opportunity_defaults when saving portfolio config
    try:
        import autopilot
        existing = autopilot.get_autopilot_config()
        alloc["opportunity_defaults"] = existing.get("opportunity_defaults") or {}
    except Exception:
        alloc["opportunity_defaults"] = {}
    save_autopilot_config(alloc)
    return _json({"ok": True, "config": alloc})


@app.post("/api/autopilot/opportunity-defaults")
async def api_autopilot_opportunity_defaults(request: Request):
    """Save Now opportunities configuration (min score, defaults for Open bot)."""
    from portfolio_initializer import save_autopilot_config
    import autopilot
    body = await request.json() or {}
    min_score = body.get("min_score")
    if min_score is not None:
        min_score = max(50, min(95, float(min_score)))
    defaults = {
        "min_score": min_score,
        "asset_filter": str(body.get("asset_filter") or "both"),
        "mode": str(body.get("mode") or "dry"),
        "capital_per_bot": float(body.get("capital_per_bot") or 250),
        "base_quote": float(body.get("base_quote") or 25),
        "safety_quote": float(body.get("safety_quote") or 25),
        "max_safety": int(body.get("max_safety") or 3),
        "tp_pct": float(body.get("tp_pct") or 1.2),
        "first_dev_pct": float(body.get("first_dev_pct") or 1.5),
        "step_mult": float(body.get("step_mult") or 1.2),
    }
    cfg = dict(autopilot.get_autopilot_config())
    cfg["opportunity_defaults"] = defaults
    save_autopilot_config(cfg)
    return _json({"ok": True, "opportunity_defaults": defaults})


@app.get("/api/notification/prefs")
def api_notification_prefs():
    """Get notification preferences (autopilot, Discord, etc.). Webhook URL is in .env only."""
    try:
        from notification_manager import _get_notification_prefs
        prefs = _get_notification_prefs()
        has_webhook = bool(os.getenv("DISCORD_WEBHOOK_URL", "").strip())
        return _json({"ok": True, "prefs": prefs, "discord_configured": has_webhook})
    except Exception as e:
        return _json({"ok": False, "error": str(e)}, 500)


@app.post("/api/notification/prefs")
async def api_notification_prefs_save(request: Request):
    """Update notification preferences (enabled, discord on/off). Set DISCORD_WEBHOOK_URL in .env for Discord."""
    try:
        body = await request.json() or {}
        from notification_manager import _get_notification_prefs
        prefs = dict(_get_notification_prefs())
        if "enabled" in body:
            prefs["enabled"] = bool(body["enabled"])
        if "discord" in body:
            prefs["discord"] = bool(body["discord"])
        set_setting("notification_prefs", json.dumps(prefs))
        return _json({"ok": True, "prefs": prefs})
    except Exception as e:
        return _json({"ok": False, "error": str(e)}, 500)


@app.post("/api/autopilot/capital/add")
async def api_autopilot_capital_add(request: Request):
    """Add capital to autopilot bots: all (split equally), all_enabled (any enabled bot), single bot, or percentage to each.
    Only updates bot config (base_quote, max_spend_quote). No order is placed. The bot uses
    the extra budget on its next tick when the strategy sees a good entry point."""
    body = await request.json() or {}
    amount_usd = float(body.get("amount_usd") or 0)
    mode = str(body.get("mode") or "all").strip().lower()
    bot_id = body.get("bot_id")
    pct_per_bot = body.get("pct_per_bot")
    if amount_usd <= 0:
        return _json({"ok": False, "error": "amount_usd must be positive"}, 400)
    bots = [b for b in list_bots() if str(b.get("bot_type") or "").lower() == "autopilot"]
    if mode == "all_enabled":
        bots = [b for b in list_bots() if int(b.get("enabled", 0)) == 1]
    if mode == "single":
        if not bot_id:
            return _json({"ok": False, "error": "bot_id required for single"}, 400)
        bots = [b for b in list_bots() if int(b.get("id")) == int(bot_id)]
        if not bots:
            return _json({"ok": False, "error": "Bot not found"}, 404)
    if not bots:
        return _json({"ok": False, "error": "No bots to update. Create bots (Setup Autopilot or Bots page) or use 'All enabled bots' if you have enabled bots."}, 400)
    updated = 0
    for b in bots:
        bid = int(b.get("id"))
        cur = get_bot(bid)
        if not cur:
            continue
        base = float(cur.get("base_quote") or 0)
        spend = float(cur.get("max_spend_quote") or 0)
        if mode in ("all", "all_enabled"):
            add = amount_usd / len(bots)
        elif mode == "single":
            add = amount_usd
        elif pct_per_bot:
            add = (float(pct_per_bot) / 100.0) * base
        else:
            add = amount_usd / max(1, len(bots))
        if add <= 0:
            continue
        data = dict(cur)
        data["base_quote"] = base + add
        data["max_spend_quote"] = spend + add
        try:
            update_bot(bid, data)
            updated += 1
        except Exception as e:
            logger.warning("autopilot capital add bot %s: %s", bid, e)
    msg = f"Added capital to {updated} bot(s)."
    return _json({"ok": True, "updated": updated, "message": msg})


# =========================================================
# Backtest Engine Endpoints
# =========================================================

@app.post("/api/backtest/run")
async def api_backtest_run(request: Request):
    """Run a backtest with given parameters."""
    from backtest_engine import BacktestEngine

    body = await request.json() or {}
    symbol = str(body.get("symbol", "BTC/USD")).strip()
    # Accept strategy_mode as alias for strategy
    strategy = str(body.get("strategy") or body.get("strategy_mode") or "dca").strip().lower()
    days = int(body.get("days", 90))
    params = dict(body.get("params", {}))

    # Normalize param aliases from bot config format → backtest engine format
    # hard_sl_pct (fraction) → sl_pct (percentage)
    if "hard_sl_pct" in params and "sl_pct" not in params:
        params["sl_pct"] = float(params.pop("hard_sl_pct"))
    # tp (fraction like 0.015) → tp_pct (percentage like 1.5)
    if "tp" in params and "tp_pct" not in params:
        tp_raw = float(params.pop("tp"))
        params["tp_pct"] = tp_raw * 100.0 if tp_raw < 1.0 else tp_raw

    if not symbol:
        return _json({"ok": False, "error": "symbol required"}, 400)

    try:
        # Fetch candles from market data
        from kraken_client import KrakenClient
        kc = KrakenClient()

        now_ms = int(time.time() * 1000)
        since_ms = now_ms - (days * 86400 * 1000)

        candles_raw = kc.fetch_ohlcv_range(symbol, "1h", since_ms, now_ms)
        if not candles_raw:
            return _json({"ok": False, "error": f"No candles fetched for {symbol}"}, 400)

        # Convert to dict format (c[0] is ms from CCXT; backtest_engine expects seconds)
        candles = [
            {
                "time": int(c[0]) // 1000,
                "open": float(c[1]),
                "high": float(c[2]),
                "low": float(c[3]),
                "close": float(c[4]),
                "volume": float(c[5]),
            }
            for c in candles_raw
        ]

        # Merge strategy name into params
        params["strategy"] = strategy

        # Run backtest
        engine = BacktestEngine(symbol, candles, params)
        result = engine.run()

        return _json({"ok": True, "result": result.to_dict()})

    except Exception as e:
        logger.exception("backtest run error")
        return _json({"ok": False, "error": str(e)}, 500)


@app.post("/api/backtest/optimize")
async def api_backtest_optimize(request: Request):
    """Run parameter optimization across a grid."""
    from backtest_engine import optimize_parameters

    body = await request.json() or {}
    symbol = str(body.get("symbol", "BTC/USD")).strip()
    strategy = str(body.get("strategy", "dca")).strip().lower()
    days = int(body.get("days", 90))
    param_grid = dict(body.get("param_grid", {}))

    if not symbol or not param_grid:
        return _json({"ok": False, "error": "symbol and param_grid required"}, 400)

    try:
        from kraken_client import KrakenClient
        kc = KrakenClient()

        now_ms = int(time.time() * 1000)
        since_ms = now_ms - (days * 86400 * 1000)

        candles_raw = kc.fetch_ohlcv_range(symbol, "1h", since_ms, now_ms)
        if not candles_raw:
            return _json({"ok": False, "error": f"No candles fetched for {symbol}"}, 400)

        candles = [
            {
                "time": int(c[0]) // 1000,
                "open": float(c[1]),
                "high": float(c[2]),
                "low": float(c[3]),
                "close": float(c[4]),
                "volume": float(c[5]),
            }
            for c in candles_raw
        ]

        # Run optimization
        results = optimize_parameters(symbol, candles, param_grid, strategy)

        # Return top results
        top_results = results[:10]  # Top 10

        return _json({
            "ok": True,
            "strategy": strategy,
            "symbol": symbol,
            "total_combinations": len(results),
            "top_results": [
                {
                    "params": r["params"],
                    "sharpe_ratio": r["sharpe_ratio"],
                    "total_return_pct": r["result"]["total_return_pct"],
                    "win_rate": r["result"]["win_rate"],
                    "max_drawdown_pct": r["result"]["max_drawdown_pct"],
                    "total_trades": r["result"]["total_trades"],
                }
                for r in top_results
            ]
        })

    except Exception as e:
        logger.exception("backtest optimize error")
        return _json({"ok": False, "error": str(e)}, 500)


@app.post("/api/backtest/monte_carlo")
async def api_backtest_monte_carlo(request: Request):
    """Run Monte Carlo simulation on trade sequence."""
    from backtest_engine import monte_carlo_simulation, Trade

    body = await request.json() or {}
    trades_data = body.get("trades", [])
    n_simulations = int(body.get("n_simulations", 1000))

    if not trades_data:
        return _json({"ok": False, "error": "trades required"}, 400)

    try:
        # Convert dict trades back to Trade objects
        trades = [
            Trade(
                entry_time=int(t["entry_time"]),
                exit_time=int(t["exit_time"]),
                entry_price=float(t["entry_price"]),
                exit_price=float(t["exit_price"]),
                pnl_usd=float(t["pnl_usd"]),
                pnl_pct=float(t["pnl_pct"]),
                side=str(t["side"]),
                duration_hours=float(t["duration_hours"]),
                position_size=float(t["position_size"]),
            )
            for t in trades_data
        ]

        # Run Monte Carlo
        results = monte_carlo_simulation(trades, n_simulations)

        return _json({
            "ok": True,
            "n_simulations": n_simulations,
            "results": results,
        })

    except Exception as e:
        logger.exception("backtest monte carlo error")
        return _json({"ok": False, "error": str(e)}, 500)


@app.post("/api/backtest/walk_forward")
async def api_backtest_walk_forward(request: Request):
    """Run Walk-Forward Analysis to prevent curve fitting."""
    from backtest_engine import BacktestEngine

    body = await request.json() or {}
    symbol = str(body.get("symbol", "BTC/USD")).strip()
    strategy = str(body.get("strategy") or body.get("strategy_mode") or "dca").strip().lower()
    days = int(body.get("days", 90))
    initial_capital = float(body.get("initial_capital", 10000.0))
    n_windows = int(body.get("n_windows", 5))
    is_ratio = float(body.get("is_ratio", 0.7))
    param_grid = dict(body.get("param_grid", {}))

    if not symbol:
        return _json({"ok": False, "error": "symbol required"}, 400)

    try:
        from kraken_client import KrakenClient
        kc = KrakenClient()

        now_ms = int(time.time() * 1000)
        since_ms = now_ms - (days * 86400 * 1000)

        candles_raw = kc.fetch_ohlcv_range(symbol, "1h", since_ms, now_ms)
        if not candles_raw:
            return _json({"ok": False, "error": f"No candles fetched for {symbol}"}, 400)

        candles = [
            {
                "time": int(c[0]) // 1000,
                "open": float(c[1]),
                "high": float(c[2]),
                "low": float(c[3]),
                "close": float(c[4]),
                "volume": float(c[5]),
            }
            for c in candles_raw
        ]

        # Create temporary engine for walk_forward_test
        temp_params = {"strategy": strategy}
        engine = BacktestEngine(symbol, candles, temp_params)

        # Run walk-forward test
        result = engine.walk_forward_test(
            candles,
            strategy=strategy,
            initial_capital=initial_capital,
            is_ratio=is_ratio,
            n_windows=n_windows,
            param_grid=param_grid if param_grid else None,
        )

        return _json({"ok": True, "result": result})

    except Exception as e:
        logger.exception("backtest walk_forward error")
        return _json({"ok": False, "error": str(e)}, 500)


@app.get("/api/patterns/{symbol:path}")
async def get_patterns(symbol: str, timeframe: str = "1h", sensitivity: float = 1.0):
    """Detect candlestick patterns for a symbol."""
    if not _kraken_ready():
        return _json({"ok": False, "error": KRAKEN_ERROR or "Kraken not ready"}, 503)

    try:
        from pattern_recognition import analyze_patterns

        # Map timeframe to minutes for fetch_ohlcv
        tf_map = {"15m": "15m", "1h": "1h", "4h": "4h", "1d": "1d", "1w": "1w"}
        tf = tf_map.get(timeframe, "1h")

        # Fetch candles
        candles = kc.fetch_ohlcv(symbol, timeframe=tf, limit=200)
        if not candles:
            return _json({"ok": False, "patterns": [], "error": "No candle data"}, 400)

        result = analyze_patterns(candles, sensitivity=sensitivity)
        return _json({"ok": True, "symbol": symbol, "timeframe": timeframe, "patterns": result}, 200)
    except Exception as e:
        logger.exception("Pattern detection failed for %s", symbol)
        return _json({"ok": False, "error": str(e)}, 500)


@app.get("/api/anomalies/{symbol:path}")
async def get_anomalies(symbol: str, timeframe: str = "1h"):
    """Detect market anomalies for a symbol."""
    if not _kraken_ready():
        return _json({"ok": False, "error": KRAKEN_ERROR or "Kraken not ready"}, 503)

    try:
        from anomaly_detector import assess_market_risk

        # Map timeframe
        tf_map = {"15m": "15m", "1h": "1h", "4h": "4h", "1d": "1d", "1w": "1w"}
        tf = tf_map.get(timeframe, "1h")

        # Fetch candles
        candles = kc.fetch_ohlcv(symbol, timeframe=tf, limit=200)
        if not candles:
            return _json({"ok": False, "anomalies": [], "risk_level": "unknown"}, 400)

        result = assess_market_risk(candles)
        return _json({"ok": True, "symbol": symbol, "timeframe": timeframe, "anomalies": result}, 200)
    except Exception as e:
        logger.exception("Anomaly detection failed for %s", symbol)
        return _json({"ok": False, "error": str(e)}, 500)


@app.get("/api/intelligence/{symbol}")
async def get_intelligence(symbol: str, timeframe: str = "1h"):
    """Combined intelligence: patterns + anomalies + ML score."""
    if not _kraken_ready():
        return _json({"ok": False, "error": KRAKEN_ERROR or "Kraken not ready"}, 503)

    result = {"ok": True, "symbol": symbol, "timeframe": timeframe}
    try:
        # Map timeframe
        tf_map = {"15m": "15m", "1h": "1h", "4h": "4h", "1d": "1d", "1w": "1w"}
        tf = tf_map.get(timeframe, "1h")

        # Fetch candles once
        candles = kc.fetch_ohlcv(symbol, timeframe=tf, limit=200)

        if candles:
            # Try pattern recognition
            try:
                from pattern_recognition import analyze_patterns
                result["patterns"] = analyze_patterns(candles, sensitivity=1.0)
            except Exception as e:
                result["patterns"] = {"error": str(e)}

            # Try anomaly detection
            try:
                from anomaly_detector import assess_market_risk
                result["anomalies"] = assess_market_risk(candles)
            except Exception as e:
                result["anomalies"] = {"error": str(e)}

            # Try ML signal scoring
            try:
                from ml_signal_scorer import MLSignalScorer
                scorer = MLSignalScorer()
                proba = scorer.predict(candles)
                result["ml_score"] = {"probability_up": proba, "model_loaded": scorer.model is not None}
            except Exception as e:
                result["ml_score"] = {"error": str(e)}
        else:
            result["error"] = "No candle data available"
    except Exception as e:
        result["error"] = str(e)

    return _json(result)


@app.get("/api/autopilot/intelligence/{symbol}")
async def autopilot_intelligence_evaluate(symbol: str):
    """Evaluate a symbol for autopilot trading opportunity using intelligence signals.

    Analyzes: pattern recognition, anomaly detection, ML scores, and smart entry filters.
    Returns conviction score (0-1) and recommended action (create_bot, watch, skip).
    """
    try:
        from autopilot import AutopilotEngine

        if not _kraken_ready():
            return _json({"ok": False, "error": KRAKEN_ERROR or "Kraken not ready"}, 503)

        engine = AutopilotEngine()

        # Fetch candles at multiple timeframes
        candles_1h = kc.fetch_ohlcv(symbol, timeframe="1h", limit=200)
        candles_4h = kc.fetch_ohlcv(symbol, timeframe="4h", limit=200)
        candles_1d = kc.fetch_ohlcv(symbol, timeframe="1d", limit=200)

        if not candles_1h:
            return _json({"ok": False, "error": "No candle data available for symbol"}, 400)

        # Evaluate opportunity with intelligence signals
        result = engine.evaluate_opportunity(symbol, candles_1h, candles_4h, candles_1d)
        return _json({
            "ok": True,
            "symbol": symbol,
            "evaluation": result,
        })
    except Exception as e:
        logger.exception("Autopilot intelligence evaluation failed for %s", symbol)
        return _json({"ok": False, "error": str(e)}, 500)


@app.get("/api/autopilot/health")
async def autopilot_health_check():
    """Scan active bots and recommend actions based on P&L and conditions.

    Returns: active bot list with P&L, health recommendations (stop/reduce/hold).
    """
    try:
        from autopilot import AutopilotEngine
        from db import list_bots

        if not bm:
            return _json({"ok": False, "error": "Bot manager not ready"}, 503)

        engine = AutopilotEngine()

        # Get active bots with P&L info
        all_bots = list_bots()
        active_bots = [b for b in all_bots if int(b.get("enabled", 0)) == 1]

        # Enrich with P&L data
        bots_with_pnl = []
        for bot in active_bots:
            bot_id = int(bot.get("id") or 0)
            try:
                pnl_data = bot_pnl_series(bot_id)
                if pnl_data:
                    current_pnl_pct = pnl_data[-1][1] if pnl_data else 0
                else:
                    current_pnl_pct = 0
            except Exception:
                current_pnl_pct = 0

            bots_with_pnl.append({
                "bot_id": bot_id,
                "symbol": bot.get("symbol"),
                "pnl_pct": float(current_pnl_pct),
                "strategy": bot.get("strategy_mode"),
                "started_at": bot.get("created_ts"),
                "enabled": int(bot.get("enabled", 0)),
            })

        # Get health recommendations
        recommendations = engine.scan_portfolio_health(bots_with_pnl)

        return _json({
            "ok": True,
            "active_bot_count": len(bots_with_pnl),
            "bots": bots_with_pnl,
            "recommendations": recommendations,
        })
    except Exception as e:
        logger.exception("Autopilot health check failed")
        return _json({"ok": False, "error": str(e)}, 500)


# NOTE: Duplicate startup event was removed - the main startup() function at line ~1312
# handles all initialization correctly (Kraken, Alpaca, BotManager with proper signatures)
