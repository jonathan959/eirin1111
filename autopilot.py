"""
Autopilot: Set-and-forget automation with purposed-entry filtering.
- Auto-scan recommendations, create bots ONLY for READY candidates
- WATCH candidates go to watchlist, not live bots
- Close/demote when score drops
- Enforces execution gate preflight before bot creation
- Intelligence-driven with pattern recognition, anomaly detection, and ML signals
"""
import json
import logging
import os
import threading
import time
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# =========================================================
# Autopilot Engine: Intelligence-driven decision making
# =========================================================

class AutopilotEngine:
    """Autonomous trading decision engine with pattern recognition, anomaly detection, and ML signals."""

    def __init__(self):
        self.last_scan_ts = 0.0
        self._pattern_recognizer = None
        self._anomaly_detector = None
        self._ml_scorer = None
        self._smart_entry = None
        self._init_modules()

    def _init_modules(self):
        """Initialize intelligence modules with graceful fallbacks."""
        try:
            from pattern_recognition import PatternRecognizer
            self._pattern_recognizer = PatternRecognizer()
        except ImportError:
            logger.debug("PatternRecognizer not available for autopilot")

        try:
            from anomaly_detector import AnomalyDetector
            self._anomaly_detector = AnomalyDetector()
        except ImportError:
            logger.debug("AnomalyDetector not available for autopilot")

        try:
            from ml_signal_scorer import MLSignalScorer
            self._ml_scorer = MLSignalScorer()
        except ImportError:
            logger.debug("MLSignalScorer not available for autopilot")

        try:
            from smart_entry import SmartEntryFilter
            self._smart_entry = SmartEntryFilter()
        except ImportError:
            logger.debug("SmartEntryFilter not available for autopilot")

    def evaluate_opportunity(self, symbol: str, candles_1h: list, candles_4h: list = None, candles_1d: list = None) -> dict:
        """
        Evaluate a symbol for trading opportunity using multiple intelligence signals.

        Returns:
            {
                "symbol": str,
                "conviction": float (0-1),
                "action": "create_bot" | "skip" | "watch",
                "signals": {
                    "patterns": {...},
                    "anomalies": {...},
                    "ml_score": float or None,
                    "smart_entry": bool or None,
                },
                "recommended_strategy": str,
                "recommended_size_pct": float,
                "reasons": [str],
            }
        """
        result = {
            "symbol": symbol,
            "conviction": 0.0,
            "action": "skip",
            "signals": {},
            "recommended_strategy": "dca",
            "recommended_size_pct": 1.0,
            "reasons": [],
        }

        conviction_scores = []
        min_conviction = float(os.getenv("AUTOPILOT_MIN_CONVICTION", "0.7"))

        # 1. Pattern Recognition
        if self._pattern_recognizer and candles_1h:
            try:
                patterns = self._pattern_recognizer.get_pattern_signals(candles_1h) if hasattr(self._pattern_recognizer, 'get_pattern_signals') else {}
                result["signals"]["patterns"] = patterns
                bullish = patterns.get("bullish_count", 0)
                bearish = patterns.get("bearish_count", 0)
                bias = patterns.get("overall_bias", "neutral")

                if bias == "bullish" and bullish > 0:
                    pattern_score = min(0.3 + bullish * 0.15, 0.9)
                    conviction_scores.append(("patterns", pattern_score))
                    result["reasons"].append(f"{bullish} bullish pattern(s) detected")
                    strongest = patterns.get("strongest_pattern")
                    if strongest:
                        result["reasons"].append(f"Strongest: {strongest.get('pattern', 'unknown')} ({strongest.get('confidence', 0):.0%})")
                elif bias == "bearish":
                    conviction_scores.append(("patterns", 0.2))
                    result["reasons"].append(f"{bearish} bearish pattern(s) - caution")
                else:
                    conviction_scores.append(("patterns", 0.5))
            except Exception as e:
                logger.debug("Pattern analysis failed for %s: %s", symbol, e)

        # 2. Anomaly Detection
        if self._anomaly_detector and candles_1h:
            try:
                risk = self._anomaly_detector.get_risk_assessment(candles_1h) if hasattr(self._anomaly_detector, 'get_risk_assessment') else {}
                result["signals"]["anomalies"] = risk
                risk_level = risk.get("risk_level", "low")
                size_mult = risk.get("position_size_multiplier", 1.0)
                result["recommended_size_pct"] = size_mult

                if risk.get("should_pause_trading"):
                    conviction_scores.append(("anomalies", 0.0))
                    result["reasons"].append("Critical anomalies - trading paused")
                elif risk_level == "high":
                    conviction_scores.append(("anomalies", 0.3))
                    result["reasons"].append(f"High risk: {risk.get('anomaly_count', 0)} anomalies")
                elif risk_level == "medium":
                    conviction_scores.append(("anomalies", 0.6))
                else:
                    conviction_scores.append(("anomalies", 0.8))
            except Exception as e:
                logger.debug("Anomaly analysis failed for %s: %s", symbol, e)

        # 3. ML Signal Score
        if self._ml_scorer and candles_1h:
            try:
                proba = self._ml_scorer.predict(candles_1h)
                result["signals"]["ml_score"] = proba
                if proba is not None:
                    conviction_scores.append(("ml", proba))
                    if proba > 0.65:
                        result["reasons"].append(f"ML model bullish ({proba:.0%} confidence)")
                    elif proba < 0.4:
                        result["reasons"].append(f"ML model bearish ({proba:.0%} confidence)")
            except Exception as e:
                logger.debug("ML scoring failed for %s: %s", symbol, e)

        # 4. Smart Entry Filter
        if self._smart_entry and candles_1h and candles_4h and candles_1d:
            try:
                should_enter, reason = self._smart_entry.should_enter(
                    symbol=symbol,
                    market_type="crypto",
                    candles_1h=candles_1h,
                    candles_4h=candles_4h,
                    candles_1d=candles_1d,
                ) if hasattr(self._smart_entry, 'should_enter') else (False, "not_available")
                result["signals"]["smart_entry"] = should_enter
                if should_enter:
                    conviction_scores.append(("smart_entry", 0.85))
                    result["reasons"].append("Smart entry filter: PASS")
                else:
                    conviction_scores.append(("smart_entry", 0.25))
                    result["reasons"].append(f"Smart entry filter: BLOCKED ({reason})")
            except Exception as e:
                logger.debug("Smart entry check failed for %s: %s", symbol, e)

        # Calculate weighted conviction
        if conviction_scores:
            weights = {"patterns": 0.25, "anomalies": 0.15, "ml": 0.30, "smart_entry": 0.30}
            total_weight = sum(weights.get(name, 0.2) for name, _ in conviction_scores)
            weighted_sum = sum(weights.get(name, 0.2) * score for name, score in conviction_scores)
            result["conviction"] = weighted_sum / total_weight if total_weight > 0 else 0.0

        # Determine action
        conviction = result["conviction"]
        if conviction >= min_conviction:
            result["action"] = "create_bot"
        elif conviction >= 0.5:
            result["action"] = "watch"
            result["reasons"].append("Moderate conviction - adding to watchlist")
        else:
            result["action"] = "skip"
            if not result["reasons"]:
                result["reasons"].append("Low conviction - skipping")

        return result

    def scan_portfolio_health(self, active_bots: List[dict]) -> List[dict]:
        """
        Scan active bots and recommend actions for underperformers.

        Args:
            active_bots: List of bot dicts with keys: bot_id, symbol, pnl_pct, strategy, started_at

        Returns:
            List of action recommendations: {"bot_id": int, "action": "stop"|"reduce"|"hold", "reason": str}
        """
        max_loss_pct = float(os.getenv("AUTOPILOT_MAX_LOSS_PCT", "5.0"))
        recommendations = []

        for bot in active_bots:
            bot_id = bot.get("bot_id")
            pnl_pct = bot.get("pnl_pct", 0)
            symbol = bot.get("symbol", "")

            # Auto-stop losers exceeding max loss
            if pnl_pct < -max_loss_pct:
                recommendations.append({
                    "bot_id": bot_id,
                    "action": "stop",
                    "reason": f"P&L {pnl_pct:.1f}% exceeds max loss threshold ({-max_loss_pct:.1f}%)",
                })
                continue

            # Check for deteriorating conditions
            if pnl_pct < -2.0:
                recommendations.append({
                    "bot_id": bot_id,
                    "action": "reduce",
                    "reason": f"Negative P&L ({pnl_pct:.1f}%) - consider reducing position",
                })
            else:
                recommendations.append({
                    "bot_id": bot_id,
                    "action": "hold",
                    "reason": "Conditions acceptable",
                })

        return recommendations

    def get_status(self) -> dict:
        """Return autopilot engine status."""
        return {
            "enabled": is_autopilot_enabled(),
            "min_conviction": float(os.getenv("AUTOPILOT_MIN_CONVICTION", "0.7")),
            "max_active_bots": int(os.getenv("AUTOPILOT_MAX_BOTS", "10")),
            "max_loss_pct": float(os.getenv("AUTOPILOT_MAX_LOSS_PCT", "5.0")),
            "scan_interval_sec": int(os.getenv("AUTOPILOT_SCAN_FREQUENCY", "14400")),
            "modules": {
                "pattern_recognition": self._pattern_recognizer is not None,
                "anomaly_detection": self._anomaly_detector is not None,
                "ml_scorer": self._ml_scorer is not None,
                "smart_entry": self._smart_entry is not None,
            },
            "last_scan_ts": self.last_scan_ts,
        }

AUTOPILOT_SCAN_INTERVAL_SEC = int(os.getenv("AUTOPILOT_SCAN_FREQUENCY", "14400"))  # 4 hours
AUTOPILOT_MIN_SCORE = float(os.getenv("AUTOPILOT_MIN_SCORE", "80"))
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

# Purposed-entry mode: STRICT_READY | ALLOW_WATCH_DRYRUN
AUTOPILOT_MODE = os.getenv("AUTOPILOT_MODE", "STRICT_READY").strip().upper()
WATCHLIST_ENABLED = os.getenv("WATCHLIST_ENABLED", "1").strip().lower() in ("1", "true", "yes", "y", "on")
MIN_ENTRY_CONFIDENCE = float(os.getenv("MIN_ENTRY_CONFIDENCE", "0.65"))


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
    min_score: float = 80,
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
            # Skip stock opportunities when Alpaca is not connected
            if market == "stocks":
                try:
                    from worker_api import _alpaca_any_ready
                    if not _alpaca_any_ready():
                        continue
                except Exception:
                    pass
            # Skip if score is 100 with no reason - likely a data artifact
            if score >= 99.5 and not reason:
                continue
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


_cycle_lock = threading.Lock()

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
    if not _cycle_lock.acquire(blocking=False):
        logger.info("Autopilot cycle already running, skipping")
        return {"status": "busy", "created": 0, "closed": 0}
    try:
        return _run_autopilot_cycle_impl(create_bot_fn, delete_bot_fn, start_bot_fn, stop_bot_fn, get_portfolio_total_fn, notify_fn, force_run)
    finally:
        _cycle_lock.release()


def _run_autopilot_cycle_impl(create_bot_fn, delete_bot_fn, start_bot_fn, stop_bot_fn, get_portfolio_total_fn, notify_fn, force_run):
    logger.info("Autopilot checking... [%s]", time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
    cfg = get_autopilot_config()
    # Use actual portfolio value if available, otherwise config (never phantom $10000)
    _actual_capital = 0
    if get_portfolio_total_fn:
        try:
            _actual_capital = float(get_portfolio_total_fn() or 0)
        except Exception:
            pass
    config_capital = float(cfg.get("total_capital", 0))
    # Use the smaller of configured capital and actual portfolio to prevent over-leveraging
    if _actual_capital > 0 and config_capital > 0:
        total_capital = min(_actual_capital, config_capital)
    elif _actual_capital > 0:
        total_capital = _actual_capital
    elif config_capital > 0:
        total_capital = config_capital
    else:
        total_capital = 50.0  # Safe minimum default
    _profile_defaults = {
        "conservative": (4, 80.0),
        "balanced": (6, 75.0),
        "aggressive": (8, 70.0),
    }
    _def_slots, _def_score = _profile_defaults.get(AUTOPILOT_PROFILE, (6, 75.0))
    max_positions = int(cfg.get("max_positions") or os.getenv("DEFAULT_MAX_POSITIONS", str(_def_slots)))
    max_positions = min(max_positions, AUTOPILOT_MAX_BOTS)
    min_capital_per = float(os.getenv("MIN_CAPITAL_PER_BOT", "5.0"))
    capital_affordable_slots = max(1, int(total_capital / min_capital_per))
    if max_positions > capital_affordable_slots:
        logger.info("Autopilot: Reducing max_positions from %d to %d (capital $%.0f / $%.0f min per bot)", max_positions, capital_affordable_slots, total_capital, min_capital_per)
        max_positions = capital_affordable_slots
    asset_filter = cfg.get("asset_types", "both")
    sectors_avoid = cfg.get("sectors_avoid") or []
    if isinstance(sectors_avoid, str):
        sectors_avoid = [s.strip() for s in sectors_avoid.split(",") if s.strip()]

    results: Dict[str, Any] = {
        "created": 0, "closed": 0, "errors": [],
        "created_bots": [],
        "skipped": [],
        "candidates_considered": 0,
        "slots_used": 0,
        "max_positions": max_positions,
        "no_candidates_reason": None,
    }
    try:
        from db import list_bots, get_setting, set_setting, get_recommendation
        portfolio = get_portfolio_total_fn() if get_portfolio_total_fn else total_capital
        if portfolio <= 0:
            portfolio = total_capital
        active_bots = [b for b in list_bots() if int(b.get("enabled", 0)) == 1]
        symbols_with_bots = {str(b.get("symbol", "")).upper() for b in active_bots}
        slots_used = len(symbols_with_bots)
        results["slots_used"] = slots_used

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

        max_bots_per_sector = int(cfg.get("max_bots_per_sector") or os.getenv("AUTOPILOT_MAX_BOTS_PER_SECTOR", "5"))
        max_bots_per_sector = max(1, min(20, max_bots_per_sector))

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

        top = get_top_recommendations(
            horizon="long",
            min_score=min_score,
            max_count=max_positions * 2,
            asset_filter=asset_filter,
            sectors_avoid=sectors_avoid,
        )
        results["candidates_considered"] = len(top)
        logger.info(
            "Autopilot: portfolio=%.2f max_positions=%d slots_used=%d recommendations_above_%.0f=%d",
            portfolio, max_positions, slots_used, min_score, len(top)
        )
        if not top:
            results["no_candidates_reason"] = f"No recommendations with score >= {min_score:.0f}. Run a scan first (/api/recommendations/scan) or lower min_score."
            logger.info("Autopilot: Signal HOLD — %s", results["no_candidates_reason"])

        slots_full_logged = False
        for rec in top:
            sym = str(rec.get("symbol") or "")
            if not sym:
                continue
            if slots_used >= max_positions:
                if not slots_full_logged:
                    try:
                        from db import add_autopilot_audit_log
                        add_autopilot_audit_log("skip_slots_full", reason=f"slots_used={slots_used} >= max_positions={max_positions}")
                        slots_full_logged = True
                    except Exception:
                        pass
                results["skipped"].append({"symbol": sym, "reason": f"slots_full ({slots_used}/{max_positions})"})
                continue
            if sym.upper() in symbols_with_bots:
                try:
                    from db import add_autopilot_audit_log
                    add_autopilot_audit_log("skip_symbol_has_bot", symbol=sym, reason="already has active bot")
                except Exception:
                    pass
                results["skipped"].append({"symbol": sym, "reason": "already_has_bot"})
                continue
            if sym.upper() in last_add_by_symbol and (now_ts - last_add_by_symbol.get(sym.upper(), 0)) < cooldown_sec:
                try:
                    from db import add_autopilot_audit_log
                    add_autopilot_audit_log("skip_cooldown", symbol=sym, reason=f"cooldown {AUTOPILOT_COOLDOWN_HOURS}h")
                except Exception:
                    pass
                results["skipped"].append({"symbol": sym, "reason": f"cooldown ({AUTOPILOT_COOLDOWN_HOURS}h)"})
                continue
            metrics = rec.get("metrics") or {}
            sector = "crypto" if "/" in sym else (metrics.get("sector") or "unknown").strip()
            # For unknown sectors, try to look up sector dynamically
            if sector == "unknown" and "/" not in sym:
                try:
                    from stock_metadata import get_sector
                    looked_up = get_sector(sym)
                    if looked_up and looked_up != "unknown":
                        sector = looked_up
                except Exception:
                    pass
            # Crypto and unknown sectors get generous caps
            if sector == "crypto":
                effective_cap = max(max_bots_per_sector * 3, 15)
            elif sector == "unknown":
                effective_cap = max(max_bots_per_sector, 8)
            else:
                effective_cap = max_bots_per_sector
            if sector_counts.get(sector, 0) >= effective_cap:
                try:
                    from db import add_autopilot_audit_log
                    add_autopilot_audit_log("skip_sector_cap", symbol=sym, reason=f"sector {sector} at cap {effective_cap}")
                except Exception:
                    pass
                results["skipped"].append({"symbol": sym, "reason": f"sector_cap ({sector})"})
                continue

            # ── Purposed-entry: scanner readiness check ──────────────
            scanner_setup = _evaluate_candidate_readiness(sym, metrics)
            if scanner_setup and not scanner_setup.get("ready_now", False):
                ready_reason = scanner_setup.get("ready_reason", "not_ready")
                if WATCHLIST_ENABLED:
                    _add_to_watchlist(sym, metrics, scanner_setup)
                if AUTOPILOT_MODE == "ALLOW_WATCH_DRYRUN":
                    logger.info("Autopilot: %s not READY (%s), creating dry-run bot", sym, ready_reason)
                else:
                    try:
                        from db import add_autopilot_audit_log
                        add_autopilot_audit_log("skip_not_ready", symbol=sym,
                                                reason=f"scanner: {ready_reason}",
                                                details={"edge_score": scanner_setup.get("edge_score", 0),
                                                         "confidence": scanner_setup.get("confidence", 0)})
                    except Exception:
                        pass
                    results["skipped"].append({"symbol": sym, "reason": f"not_ready: {ready_reason}"})
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
            # Ensure capital_per_bot is enough for meaningful trades ($5 minimum)
            min_viable_capital = float(os.getenv("MIN_CAPITAL_PER_BOT", "5.0"))
            if capital_per_bot < min_viable_capital:
                capital_per_bot = min_viable_capital
            base_order = max(2.0, capital_per_bot * 0.30)
            safety_order = max(1.0, capital_per_bot * 0.15)
            metrics = rec.get("metrics") or {}
            strategy = str(metrics.get("strategy") or metrics.get("recommended_strategy") or "smart_dca")
            market = (metrics.get("market_type") or "crypto").strip().lower()
            if market == "stock":
                market = "stocks"
            try:
                profile = str(cfg.get("risk_profile") or os.getenv("AUTOPILOT_PROFILE", "balanced")).lower()
                # Default: 5% stop loss, 15% take profit; profile tweaks
                _profile_params = {
                    "conservative": {"tp": 0.10, "stop_loss_pct": 0.05, "trailing_activation_pct": 0.015, "trailing_distance_pct": 0.008, "max_safety": 2, "max_drawdown_pct": 0.10, "daily_loss_limit_pct": 0.03, "per_symbol_exposure_pct": 0.08},
                    "balanced":     {"tp": 0.15, "stop_loss_pct": 0.05, "trailing_activation_pct": 0.020, "trailing_distance_pct": 0.010, "max_safety": 3, "max_drawdown_pct": 0.15, "daily_loss_limit_pct": 0.05, "per_symbol_exposure_pct": 0.10},
                    "aggressive":   {"tp": 0.20, "stop_loss_pct": 0.05, "trailing_activation_pct": 0.030, "trailing_distance_pct": 0.015, "max_safety": 5, "max_drawdown_pct": 0.15, "daily_loss_limit_pct": 0.08, "per_symbol_exposure_pct": 0.15},
                }
                pp = _profile_params.get(profile, _profile_params["balanced"])
                _force_dry = (AUTOPILOT_MODE == "ALLOW_WATCH_DRYRUN"
                              and scanner_setup and not scanner_setup.get("ready_now", True))
                _dry_run_val = 1 if _force_dry else int(cfg.get("dry_run", 1))
                bot_id = create_bot_fn({
                    "name": f"Autopilot {sym}",
                    "symbol": sym,
                    "bot_type": "autopilot",
                    "enabled": 1,
                    "dry_run": _dry_run_val,
                    "strategy_mode": strategy,
                    "forced_strategy": "",
                    "base_quote": base_order,
                    "safety_quote": safety_order,
                    "max_safety": pp["max_safety"],
                    "first_dev": 0.015,
                    "step_mult": 1.2,
                    "tp": pp["tp"],
                    "trend_filter": 0,
                    "trend_sma": 200,
                    "max_spend_quote": capital_per_bot,
                    "poll_seconds": 10,
                    "max_open_orders": 6,
                    "max_total_exposure_pct": float(os.getenv("MAX_TOTAL_EXPOSURE_PCT", "0.50")),
                    "per_symbol_exposure_pct": pp["per_symbol_exposure_pct"],
                    "min_free_cash_pct": 0.2,
                    "max_concurrent_deals": 4,
                    "spread_guard_pct": 0.004,
                    "limit_timeout_sec": 8,
                    "daily_loss_limit_pct": pp["daily_loss_limit_pct"],
                    "pause_hours": 6,
                    "market_type": market,
                    "alpaca_mode": cfg.get("alpaca_mode", "paper"),
                    "auto_restart": 1,
                    "stop_loss_pct": pp["stop_loss_pct"],
                    "trailing_stop_enabled": 1,
                    "trailing_activation_pct": pp["trailing_activation_pct"],
                    "trailing_distance_pct": pp["trailing_distance_pct"],
                    "max_drawdown_pct": pp["max_drawdown_pct"],
                    "use_kelly_sizing": 1,
                    "kelly_fraction": 0.25,
                    "risk_profile": profile,
                    "adaptive_tp_enabled": 1,
                })
                if bot_id and start_bot_fn:
                    start_bot_fn(bot_id)
                symbols_with_bots.add(sym.upper())
                slots_used += 1
                results["created"] += 1
                results["created_bots"].append({"id": bot_id, "symbol": sym, "name": f"Autopilot {sym}", "score": score})
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
                    _notify_payload = {"symbol": sym, "score": score, "bot_id": bot_id}
                    if scanner_setup:
                        _notify_payload["entry_type"] = scanner_setup.get("entry_type", "")
                        _notify_payload["confidence"] = scanner_setup.get("confidence", 0)
                        _notify_payload["evidence"] = (scanner_setup.get("evidence") or [])[:3]
                        _notify_payload["target_levels"] = scanner_setup.get("target_levels", {})
                        _notify_payload["invalidation_level"] = scanner_setup.get("invalidation_level", 0)
                    notify_fn("autopilot_bot_created", _notify_payload)
            except Exception as e:
                results["errors"].append(f"Create {sym}: {e}")
                logger.warning("Autopilot create bot failed %s: %s", sym, e)
                try:
                    from db import add_autopilot_audit_log, log_error
                    add_autopilot_audit_log("create_bot_failed", symbol=sym, reason=str(e)[:200])
                    log_error("autopilot", "create_bot_failed", f"Create {sym}: {e}", details={"symbol": sym})
                except Exception:
                    pass

        # 2b. Recheck watchlist: promote WATCH symbols that are now READY
        try:
            recheck_watchlist(
                create_bot_fn=create_bot_fn,
                start_bot_fn=start_bot_fn,
                notify_fn=notify_fn,
            )
        except Exception as wl_err:
            logger.warning("Autopilot: recheck_watchlist failed: %s", wl_err)

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


def _evaluate_candidate_readiness(symbol: str, metrics: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """
    Run the market scanner on a candidate symbol and return the setup dict.
    Returns None if scanner is not available or fails (graceful degradation).
    """
    try:
        from market_scanner import analyze_symbol, MarketSetup
        import pandas as pd
    except ImportError:
        logger.debug("market_scanner not available, skipping readiness check")
        return None

    try:
        market_type = (metrics.get("market_type") or "crypto").strip().lower()
        if market_type == "stock":
            market_type = "stocks"

        candles = _fetch_candles_for_scanner(symbol, market_type)
        if candles is None or len(candles) < 50:
            return {"ready_now": False, "ready_reason": "insufficient_candle_data",
                    "confidence": 0, "edge_score": 0}

        setup = analyze_symbol(
            symbol=symbol,
            candles_df=candles,
            market_type=market_type,
            run_preflight=True,
            min_confidence=MIN_ENTRY_CONFIDENCE,
        )
        return setup.to_dict()
    except Exception as e:
        logger.debug("Scanner evaluation failed for %s: %s", symbol, e)
        return None


def _fetch_candles_for_scanner(symbol: str, market_type: str) -> Optional["pd.DataFrame"]:
    """Fetch recent candles for scanner analysis using existing exchange adapters."""
    try:
        import pandas as pd
    except ImportError:
        return None

    try:
        if market_type == "crypto":
            try:
                from worker_api import _safe_kraken_client
                kc = _safe_kraken_client()
                if kc:
                    ohlcv = kc.fetch_ohlcv(symbol, timeframe="1h", limit=200)
                    if ohlcv and len(ohlcv) > 50:
                        df = pd.DataFrame(ohlcv, columns=["timestamp", "open", "high", "low", "close", "volume"])
                        return df
            except Exception as e:
                logger.debug("Kraken candle fetch for scanner failed %s: %s", symbol, e)
        else:
            try:
                from worker_api import _safe_alpaca_data_client
                ac = _safe_alpaca_data_client()
                if ac:
                    from datetime import datetime, timedelta
                    end = datetime.utcnow()
                    start = end - timedelta(days=30)
                    bars = ac.get_stock_bars(symbol, timeframe="1Hour", start=start, end=end)
                    if bars:
                        df = pd.DataFrame([{
                            "timestamp": b.timestamp, "open": b.open, "high": b.high,
                            "low": b.low, "close": b.close, "volume": b.volume,
                        } for b in bars])
                        return df if len(df) > 50 else None
            except Exception as e:
                logger.debug("Alpaca candle fetch for scanner failed %s: %s", symbol, e)
    except Exception as e:
        logger.debug("Candle fetch for scanner failed %s: %s", symbol, e)

    return None


def _add_to_watchlist(symbol: str, metrics: Dict[str, Any], scanner_setup: Dict[str, Any]) -> None:
    """Add a WATCH candidate to the scanner watchlist DB table."""
    try:
        from db import upsert_watchlist_entry
        market_type = (metrics.get("market_type") or "crypto").strip().lower()
        if market_type == "stock":
            market_type = "stocks"
        upsert_watchlist_entry(
            symbol=symbol,
            market_type=market_type,
            setup_json=json.dumps(scanner_setup),
            trigger_conditions=scanner_setup.get("trigger_conditions", ""),
            regime=scanner_setup.get("regime", ""),
            entry_type=scanner_setup.get("entry_type", ""),
            confidence=float(scanner_setup.get("confidence", 0)),
            edge_score=float(scanner_setup.get("edge_score", 0)),
        )
        logger.info("Autopilot: Added %s to watchlist (edge=%.3f, reason=%s)",
                     symbol, scanner_setup.get("edge_score", 0),
                     scanner_setup.get("ready_reason", ""))
    except Exception as e:
        logger.debug("Failed to add %s to watchlist: %s", symbol, e)


def recheck_watchlist(
    create_bot_fn=None,
    start_bot_fn=None,
    notify_fn=None,
) -> Dict[str, Any]:
    """
    Re-evaluate watchlist entries: promote to bot if now READY, expire if stale.
    Called during autopilot cycle.
    """
    result = {"promoted": 0, "expired": 0, "still_watching": 0}
    if not WATCHLIST_ENABLED:
        return result

    try:
        from db import list_watchlist, mark_watchlist_triggered, remove_watchlist_entry, cleanup_old_watchlist
        cleanup_old_watchlist(max_age_hours=72)

        entries = list_watchlist(status="watching", limit=50)
        for entry in entries:
            sym = str(entry.get("symbol", ""))
            if not sym:
                continue
            market_type = str(entry.get("market_type", "crypto"))
            setup = _evaluate_candidate_readiness(sym, {"market_type": market_type})
            if setup and setup.get("ready_now", False):
                logger.info("Autopilot watchlist: %s now READY, promoting", sym)
                mark_watchlist_triggered(sym)
                result["promoted"] += 1
                if notify_fn:
                    notify_fn("watchlist_triggered", {
                        "symbol": sym,
                        "confidence": setup.get("confidence", 0),
                        "entry_type": setup.get("entry_type", ""),
                    })
            else:
                result["still_watching"] += 1
    except Exception as e:
        logger.debug("Watchlist recheck failed: %s", e)

    return result


def get_scanner_watchlist(limit: int = 50) -> List[Dict[str, Any]]:
    """Get the current scanner watchlist (symbols waiting for trigger)."""
    try:
        from db import list_watchlist
        return list_watchlist(status="watching", limit=limit)
    except Exception:
        return []


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
