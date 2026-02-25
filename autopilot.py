"""
Autopilot: Set-and-forget automation.
- Auto-scan recommendations, create bots for top picks
- Close/demote when score drops
- Watchlist for 65-74 scores
"""
import json
import logging
import os
import time
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

AUTOPILOT_SCAN_INTERVAL_SEC = int(os.getenv("AUTOPILOT_SCAN_FREQUENCY", "14400"))  # 4 hours
AUTOPILOT_MIN_SCORE = float(os.getenv("AUTOPILOT_MIN_SCORE", "75"))
AUTOPILOT_DEMOTE_SCORE = float(os.getenv("AUTOPILOT_DEMOTE_SCORE", "60"))
AUTOPILOT_WATCH_SCORE_LO = float(os.getenv("AUTOPILOT_WATCH_SCORE_LO", "65"))
AUTOPILOT_WATCH_SCORE_HI = float(os.getenv("AUTOPILOT_WATCH_SCORE_HI", "74"))

# Profile: conservative (stricter) | balanced | aggressive (looser). Affects defaults when not in config.
AUTOPILOT_PROFILE = (os.getenv("AUTOPILOT_PROFILE", "balanced") or "balanced").strip().lower()
if AUTOPILOT_PROFILE not in ("conservative", "balanced", "aggressive"):
    AUTOPILOT_PROFILE = "balanced"

# Hard cap on total autopilot bots (even if config max_positions is higher)
AUTOPILOT_MAX_BOTS = int(os.getenv("AUTOPILOT_MAX_BOTS", "10"))

# Cooldown: don't add the same symbol again for this many hours
AUTOPILOT_COOLDOWN_HOURS = float(os.getenv("AUTOPILOT_COOLDOWN_HOURS", "6"))

# Max bots per sector (stocks); crypto treated as one "sector"
AUTOPILOT_MAX_BOTS_PER_SECTOR = int(os.getenv("AUTOPILOT_MAX_BOTS_PER_SECTOR", "2"))


def _get_setting(key: str, default: str = "") -> str:
    try:
        from db import get_setting
        return str(get_setting(key, default) or default)
    except Exception:
        return default


def is_autopilot_enabled() -> bool:
    return _get_setting("autopilot_enabled", "0").strip().lower() in ("1", "true", "yes", "y", "on")


def get_autopilot_config() -> Dict[str, Any]:
    try:
        raw = _get_setting("autopilot_config", "{}")
        return json.loads(raw) if raw else {}
    except Exception:
        return {}


def _capital_for_score(score: float) -> float:
    """Allocate % of available capital by score. 90-100: 15%, 80-89: 10%, 75-79: 5%."""
    if score >= 90:
        return 0.15
    if score >= 80:
        return 0.10
    if score >= 75:
        return 0.05
    return 0.0


def get_top_recommendations(
    horizon: str = "long",
    min_score: float = 75,
    max_count: int = 10,
    asset_filter: Optional[str] = None,
    sectors_avoid: Optional[List[str]] = None,
) -> List[Dict[str, Any]]:
    """Get top recommendations above min_score, filtered by asset type and sectors.
    Results are spread across sectors so you get more than just one industry (e.g. not only tech).
    """
    try:
        from db import list_recommendations
        rows = list_recommendations(horizon, limit=100)
        out = []
        for r in rows:
            score = float(r.get("score") or 0)
            if score < min_score:
                continue
            metrics = {}
            try:
                metrics = json.loads(r.get("metrics_json") or "{}")
            except Exception:
                pass
            market = (metrics.get("market_type") or "").strip().lower()
            sector = (metrics.get("sector") or "").strip() or "unknown"
            if market == "crypto" or "/" in str(r.get("symbol") or ""):
                sector = "crypto"
            if asset_filter and asset_filter != "both":
                if asset_filter == "stocks" and market != "stocks":
                    continue
                if asset_filter == "crypto" and market == "stocks":
                    continue
            if sectors_avoid and sector and sector in sectors_avoid:
                continue
            out.append({**dict(r), "metrics": metrics, "sector": sector})
        if not out:
            return []
        # Spread across sectors: take top per sector then sort by score so we don't return only tech
        by_sector: Dict[str, List[Dict[str, Any]]] = {}
        for it in out:
            s = str(it.get("sector") or "unknown")
            by_sector.setdefault(s, []).append(it)
        per_sector = max(1, max_count // max(1, len(by_sector)))
        diversified = []
        for lst in by_sector.values():
            lst.sort(key=lambda x: float(x.get("score") or 0), reverse=True)
            diversified.extend(lst[:per_sector])
        diversified.sort(key=lambda x: float(x.get("score") or 0), reverse=True)
        return diversified[:max_count]
    except Exception as e:
        logger.error("get_top_recommendations failed: %s", e)
        return []


def get_bots_for_symbol(symbol: str) -> List[Dict[str, Any]]:
    """Bots currently trading this symbol."""
    try:
        from db import list_bots
        return [b for b in list_bots() if str(b.get("symbol") or "").upper() == str(symbol).upper()]
    except Exception:
        return []


def get_now_opportunities(
    asset_filter: Optional[str] = None,
    max_count: int = 3,
    min_score: Optional[float] = None,
) -> List[Dict[str, Any]]:
    """
    Curated list of "best opportunities right now" for radar/autopilot UI.

    - Uses existing recommendations (short + long horizon)
    - Applies autopilot config (min_score, asset_types, sectors_avoid)
    - Excludes symbols that already have active bots
    """
    try:
        from db import list_bots
    except Exception:
        return []

    cfg = get_autopilot_config()
    # Min score: explicit override > config > env default
    if min_score is None:
        min_score = float(cfg.get("min_score") or cfg.get("min_score_threshold") or AUTOPILOT_MIN_SCORE)
    try:
        min_score = float(min_score)
    except Exception:
        min_score = AUTOPILOT_MIN_SCORE
    min_score = max(50, min(95, min_score))

    if not asset_filter:
        asset_filter = str(cfg.get("asset_types") or "both")
    sectors_avoid = cfg.get("sectors_avoid") or []
    if isinstance(sectors_avoid, str):
        sectors_avoid = [s.strip() for s in sectors_avoid.split(",") if s.strip()]

    try:
        active_bots = [b for b in list_bots() if int(b.get("enabled", 0)) == 1]
        symbols_with_bots = {str(b.get("symbol") or "").upper() for b in active_bots}
    except Exception:
        active_bots = []
        symbols_with_bots = set()

    candidates: List[Dict[str, Any]] = []

    # Prefer short-term horizon first, then long-term if we still have room.
    for horizon in ("short", "long"):
        rows = get_top_recommendations(
            horizon=horizon,
            min_score=min_score,
            max_count=max_count * 4,
            asset_filter=asset_filter,
            sectors_avoid=sectors_avoid,
        )
        for r in rows:
            sym = str(r.get("symbol") or "")
            if not sym:
                continue
            if sym.upper() in symbols_with_bots:
                # Already have an active bot for this symbol; skip for radar.
                continue
            score = float(r.get("score") or 0)
            if score < min_score:
                continue
            metrics = r.get("metrics") or {}
            market = (metrics.get("market_type") or "").strip().lower()
            if market == "stock":
                market = "stocks"
            if not market:
                market = "stocks" if (len(sym) < 6 and "/" not in sym) else "crypto"
            reason = metrics.get("explanation") or metrics.get("reason") or metrics.get("recommended_strategy") or ""
            strategy = metrics.get("recommended_strategy") or metrics.get("strategy") or "smart_dca"
            created_ts = r.get("created_ts") or r.get("ts") or None

            candidates.append(
                {
                    "symbol": sym,
                    "score": score,
                    "horizon": horizon,
                    "market_type": market,
                    "reason": str(reason)[:200] if reason else None,
                    "strategy": strategy,
                    "created_ts": int(created_ts) if created_ts else None,
                }
            )

    # Sort by score desc, then by recency (if available)
    candidates.sort(key=lambda x: (float(x.get("score") or 0), float(x.get("created_ts") or 0)), reverse=True)
    return candidates[: max(1, int(max_count))]


def run_autopilot_cycle(
    create_bot_fn,
    delete_bot_fn,
    start_bot_fn,
    stop_bot_fn,
    get_portfolio_total_fn,
    notify_fn=None,
    force_run: bool = False,
) -> Dict[str, Any]:
    """
    One autopilot cycle: scan, create bots for top picks, close bots below threshold.
    When force_run=True, run even if autopilot is disabled (e.g. manual trigger).
    """
    if not force_run and not is_autopilot_enabled():
        logger.info("Autopilot check skipped: disabled (set autopilot_enabled=1 in settings to enable)")
        return {"status": "disabled", "created": 0, "closed": 0}
    # LIVE-HARDENED: heartbeat written at end of cycle for watchdog
    logger.info("Autopilot checking... [%s]", time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
    cfg = get_autopilot_config()
    total_capital = float(cfg.get("total_capital", 10000))
    # Profile defaults when not in config: conservative 4/80, balanced 6/75, aggressive 8/70
    _profile_defaults = {
        "conservative": (4, 80.0),
        "balanced": (6, 75.0),
        "aggressive": (8, 70.0),
    }
    _def_slots, _def_score = _profile_defaults.get(AUTOPILOT_PROFILE, (6, 75.0))
    max_positions = int(cfg.get("max_positions") or os.getenv("DEFAULT_MAX_POSITIONS", str(_def_slots)))
    max_positions = min(max_positions, AUTOPILOT_MAX_BOTS)
    asset_filter = cfg.get("asset_types", "both")
    sectors_avoid = cfg.get("sectors_avoid") or []
    if isinstance(sectors_avoid, str):
        sectors_avoid = [s.strip() for s in sectors_avoid.split(",") if s.strip()]

    results = {"created": 0, "closed": 0, "errors": []}
    try:
        from db import list_bots, get_setting, set_setting, get_recommendation
        portfolio = get_portfolio_total_fn() if get_portfolio_total_fn else total_capital
        if portfolio <= 0:
            portfolio = total_capital
        active_bots = [b for b in list_bots() if int(b.get("enabled", 0)) == 1]
        symbols_with_bots = {str(b.get("symbol", "")).upper() for b in active_bots}
        slots_used = len(symbols_with_bots)

        # Min score: config override or profile default or env
        min_score = float(cfg.get("min_score") or cfg.get("min_score_threshold") or AUTOPILOT_MIN_SCORE)
        if (cfg.get("min_score") or cfg.get("min_score_threshold")) is None:
            min_score = _def_score
        min_score = max(50, min(95, min_score))

        # Sector counts from existing active bots (for sector cap)
        sector_counts: Dict[str, int] = {}
        for b in active_bots:
            sym = str(b.get("symbol") or "")
            if not sym:
                continue
            sector = "crypto" if "/" in sym else "unknown"
            if "/" not in sym:
                try:
                    row = get_recommendation(sym, "long")
                    if row:
                        m = json.loads(row.get("metrics_json") or "{}")
                        sector = (m.get("sector") or "unknown").strip()
                except Exception:
                    pass
            sector_counts[sector] = sector_counts.get(sector, 0) + 1

        max_bots_per_sector = int(cfg.get("max_bots_per_sector") or os.getenv("AUTOPILOT_MAX_BOTS_PER_SECTOR", "2"))
        max_bots_per_sector = max(1, min(10, max_bots_per_sector))

        # Cooldown: last add per symbol (JSON map symbol -> ts)
        cooldown_sec = int(AUTOPILOT_COOLDOWN_HOURS * 3600)
        last_add_by_symbol: Dict[str, int] = {}
        try:
            raw = get_setting("autopilot_last_add_by_symbol", "{}")
            if raw:
                last_add_by_symbol = json.loads(raw)
        except Exception:
            pass
        now_ts = int(time.time())

        # 1. Get top recommendations
        top = get_top_recommendations(
            horizon="long",
            min_score=min_score,
            max_count=max_positions * 2,
            asset_filter=asset_filter,
            sectors_avoid=sectors_avoid,
        )
        logger.info(
            "Autopilot: portfolio=%.2f max_positions=%d slots_used=%d recommendations_above_%.0f=%d",
            portfolio, max_positions, slots_used, min_score, len(top)
        )
        if not top:
            logger.info("Autopilot: Signal HOLD — No recommendations with score >= %.0f (run /api/reco/scan to populate)", min_score)

        # 2. Create bots for new top picks (up to max_positions)
        slots_full_logged = False
        for rec in top:
            if slots_used >= max_positions:
                if not slots_full_logged:
                    try:
                        from db import add_autopilot_audit_log
                        add_autopilot_audit_log("skip_slots_full", reason=f"slots_used={slots_used} >= max_positions={max_positions}")
                        slots_full_logged = True
                    except Exception:
                        pass
                break
            sym = str(rec.get("symbol") or "")
            if not sym or sym.upper() in symbols_with_bots:
                if sym:
                    try:
                        from db import add_autopilot_audit_log
                        add_autopilot_audit_log("skip_symbol_has_bot", symbol=sym, reason="already has active bot")
                    except Exception:
                        pass
                continue
            # Cooldown: skip if we added this symbol recently
            if sym.upper() in last_add_by_symbol and (now_ts - last_add_by_symbol.get(sym.upper(), 0)) < cooldown_sec:
                try:
                    from db import add_autopilot_audit_log
                    add_autopilot_audit_log("skip_cooldown", symbol=sym, reason=f"cooldown {AUTOPILOT_COOLDOWN_HOURS}h")
                except Exception:
                    pass
                continue
            metrics = rec.get("metrics") or {}
            sector = "crypto" if "/" in sym else (metrics.get("sector") or "unknown").strip()
            if sector_counts.get(sector, 0) >= max_bots_per_sector:
                try:
                    from db import add_autopilot_audit_log
                    add_autopilot_audit_log("skip_sector_cap", symbol=sym, reason=f"sector {sector} at cap {max_bots_per_sector}")
                except Exception:
                    pass
                continue
            score = float(rec.get("score") or 0)
            config_cap = float(cfg.get("capital_per_bot") or 0)
            per_slot_share = total_capital / max_positions if max_positions else total_capital
            if config_cap >= 0.5:
                capital_per_bot = min(config_cap, max(per_slot_share * 1.5, total_capital))
                capital_per_bot = max(0.5, capital_per_bot)
            else:
                pct = _capital_for_score(score)
                capital_per_bot = portfolio * pct
                floor = max(1.0, per_slot_share)
                capital_per_bot = max(floor, min(capital_per_bot, total_capital / max_positions * 1.5))
                default_cap = float(os.getenv("DEFAULT_CAPITAL_PER_BOT", "500"))
                capital_per_bot = min(capital_per_bot, default_cap)
            base_order = max(0.5, capital_per_bot * 0.15)
            safety_order = max(0.5, capital_per_bot * 0.1)
            metrics = rec.get("metrics") or {}
            strategy = str(metrics.get("strategy") or metrics.get("recommended_strategy") or "smart_dca")
            market = (metrics.get("market_type") or "crypto").strip().lower()
            if market == "stock":
                market = "stocks"
            try:
                bot_id = create_bot_fn({
                    "name": f"Autopilot {sym}",
                    "symbol": sym,
                    "bot_type": "autopilot",
                    "enabled": 1,
                    "dry_run": int(cfg.get("dry_run", 1)),
                    "strategy_mode": strategy,
                    "forced_strategy": "",
                    "base_quote": base_order,
                    "safety_quote": safety_order,
                    "max_safety": 3,
                    "first_dev": 0.015,
                    "step_mult": 1.2,
                    "tp": 0.012,
                    "trend_filter": 0,
                    "trend_sma": 200,
                    "max_spend_quote": capital_per_bot,
                    "poll_seconds": 10,
                    "max_open_orders": 6,
                    "max_total_exposure_pct": float(os.getenv("MAX_TOTAL_EXPOSURE_PCT", "0.50")),
                    "per_symbol_exposure_pct": 0.1,
                    "min_free_cash_pct": 0.2,
                    "max_concurrent_deals": 4,
                    "spread_guard_pct": 0.004,
                    "limit_timeout_sec": 8,
                    "daily_loss_limit_pct": 0.05,
                    "pause_hours": 6,
                    "market_type": market,
                    "alpaca_mode": cfg.get("alpaca_mode", "paper"),
                    "auto_restart": 1,
                })
                if bot_id and start_bot_fn:
                    start_bot_fn(bot_id)
                symbols_with_bots.add(sym.upper())
                slots_used += 1
                results["created"] += 1
                last_add_by_symbol[sym.upper()] = now_ts
                try:
                    set_setting("autopilot_last_add_by_symbol", json.dumps(last_add_by_symbol))
                except Exception:
                    pass
                sector_counts[sector] = sector_counts.get(sector, 0) + 1
                logger.info("Autopilot: Signal BUY — Created bot for %s (score=%.1f bot_id=%s)", sym, score, bot_id)
                try:
                    from db import add_autopilot_audit_log
                    add_autopilot_audit_log("create_bot", symbol=sym, reason=f"score={score:.1f}", details={"bot_id": bot_id, "score": score})
                except Exception:
                    pass
                if notify_fn:
                    notify_fn("autopilot_bot_created", {"symbol": sym, "score": score, "bot_id": bot_id})
            except Exception as e:
                results["errors"].append(f"Create {sym}: {e}")
                logger.warning("Autopilot create bot failed %s: %s", sym, e)
                try:
                    from db import add_autopilot_audit_log, log_error
                    add_autopilot_audit_log("create_bot_failed", symbol=sym, reason=str(e)[:200])
                    log_error("autopilot", "create_bot_failed", f"Create {sym}: {e}", details={"symbol": sym})
                except Exception:
                    pass

        # 3. Check active bots: close if score dropped below threshold
        for bot in active_bots:
            sym = str(bot.get("symbol") or "")
            if not sym:
                continue
            try:
                from db import get_recommendation, list_recommendations
                row = get_recommendation(sym, "long")
                score = float(row.get("score") or 0) if row else 0
                if score < AUTOPILOT_DEMOTE_SCORE and score > 0:
                    bot_id = int(bot.get("id") or 0)
                    logger.info("Autopilot: Signal SELL — Closing %s (score=%.1f < %.0f)", sym, score, AUTOPILOT_DEMOTE_SCORE)
                    try:
                        from db import add_autopilot_audit_log
                        add_autopilot_audit_log("stop_bot", symbol=sym, reason=f"score_dropped score={score:.1f} < {AUTOPILOT_DEMOTE_SCORE}", details={"bot_id": bot_id})
                    except Exception:
                        pass
                    if bot_id and stop_bot_fn:
                        stop_bot_fn(bot_id)
                    if bot_id and delete_bot_fn and cfg.get("auto_delete_closed", False):
                        delete_bot_fn(bot_id)
                    results["closed"] += 1
                    if notify_fn:
                        notify_fn("autopilot_bot_closed", {"symbol": sym, "score": score, "reason": "score_dropped"})
            except Exception as e:
                logger.debug("Autopilot score check %s failed: %s", sym, e)

    except Exception as e:
        logger.exception("Autopilot cycle failed: %s", e)
        results["errors"].append(str(e))
        try:
            from db import add_autopilot_audit_log, log_error
            add_autopilot_audit_log("cycle_failed", reason=str(e)[:200])
            log_error("autopilot", "cycle_failed", str(e))
        except Exception:
            pass
    # LIVE-HARDENED: heartbeat so watchdog can restart autopilot if stale
    try:
        from db import set_setting, add_autopilot_audit_log
        set_setting("autopilot_last_heartbeat_ts", str(int(time.time())))
        add_autopilot_audit_log("cycle_complete", reason="ok", details={"created": results.get("created", 0), "closed": results.get("closed", 0), "errors": len(results.get("errors", []))})
    except Exception:
        pass
    logger.info("Autopilot cycle done: created=%d closed=%d errors=%d", results.get("created", 0), results.get("closed", 0), len(results.get("errors", [])))
    return results


def get_watchlist(
    horizon: str = "long",
    asset_filter: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """Recommendations in watch zone (65-74 score) - monitor, don't trade yet."""
    try:
        from db import list_recommendations
        rows = list_recommendations(horizon, limit=100)
        out = []
        for r in rows:
            score = float(r.get("score") or 0)
            if score < AUTOPILOT_WATCH_SCORE_LO or score > AUTOPILOT_WATCH_SCORE_HI:
                continue
            metrics = {}
            try:
                metrics = json.loads(r.get("metrics_json") or "{}")
            except Exception:
                pass
            market = (metrics.get("market_type") or "").strip().lower()
            if asset_filter and asset_filter != "both":
                if asset_filter == "stocks" and market != "stocks":
                    continue
                if asset_filter == "crypto" and market == "stocks":
                    continue
            out.append({**dict(r), "metrics": metrics})
        return sorted(out, key=lambda x: -float(x.get("score") or 0))
    except Exception as e:
        logger.error("get_watchlist failed: %s", e)
        return []
