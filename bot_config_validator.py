"""
Bot configuration validation — strict typing + range checks.

Validates bot config before creation/update. Returns cleaned config + list of issues.
Uses plain dataclass validation (no pydantic dependency required).
"""

import logging
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

VALID_STRATEGY_MODES = {
    "classic", "smart_dca", "classic_dca", "grid",
    "trend_follow", "range_mean_reversion", "high_vol_defensive",
    "breakout", "auto", "router", "scalping", "smart",
}

VALID_MARKET_TYPES = {"crypto", "stocks"}
VALID_RISK_PROFILES = {"conservative", "balanced", "aggressive"}
VALID_ALPACA_MODES = {"paper", "live"}


def validate_bot_config(config: Dict[str, Any]) -> Tuple[Dict[str, Any], List[str]]:
    """
    Validate and clean bot config. Returns (cleaned_config, issues).
    Issues are warnings/errors as strings. Fatal issues start with "ERROR:".
    """
    issues: List[str] = []
    c = dict(config)

    def _clamp(key: str, val: Any, lo: float, hi: float, default: float) -> float:
        try:
            v = float(val)
            if v < lo:
                issues.append(f"{key}={v} below min {lo}, clamped")
                return lo
            if v > hi:
                issues.append(f"{key}={v} above max {hi}, clamped")
                return hi
            return v
        except (TypeError, ValueError):
            if val is not None:
                issues.append(f"{key}={val!r} invalid, using default {default}")
            return default

    if "symbol" in c:
        sym = str(c["symbol"]).strip()
        if not sym:
            issues.append("ERROR: symbol is empty")
        c["symbol"] = sym

    if "strategy_mode" in c:
        sm = str(c["strategy_mode"]).strip().lower()
        if sm not in VALID_STRATEGY_MODES:
            issues.append(f"strategy_mode='{sm}' not recognized, defaulting to smart_dca")
            sm = "smart_dca"
        c["strategy_mode"] = sm

    if "market_type" in c:
        mt = str(c["market_type"]).strip().lower()
        if mt == "stock":
            mt = "stocks"
        if mt not in VALID_MARKET_TYPES:
            issues.append(f"market_type='{mt}' invalid, defaulting to crypto")
            mt = "crypto"
        c["market_type"] = mt

    if "risk_profile" in c:
        rp = str(c["risk_profile"]).strip().lower()
        if rp not in VALID_RISK_PROFILES:
            issues.append(f"risk_profile='{rp}' invalid, defaulting to balanced")
            rp = "balanced"
        c["risk_profile"] = rp

    if "alpaca_mode" in c:
        am = str(c["alpaca_mode"]).strip().lower()
        if am not in VALID_ALPACA_MODES:
            issues.append(f"alpaca_mode='{am}' invalid, defaulting to paper")
            am = "paper"
        c["alpaca_mode"] = am

    c["tp"] = _clamp("tp", c.get("tp"), 0.001, 0.50, 0.03)
    c["first_dev"] = _clamp("first_dev", c.get("first_dev"), 0.001, 0.20, 0.015)
    c["step_mult"] = _clamp("step_mult", c.get("step_mult"), 1.0, 5.0, 1.2)
    c["base_quote"] = _clamp("base_quote", c.get("base_quote"), 0.5, 1000000, 10.0)
    c["safety_quote"] = _clamp("safety_quote", c.get("safety_quote"), 0.0, 1000000, 5.0)
    c["max_safety"] = int(_clamp("max_safety", c.get("max_safety"), 0, 20, 3))
    c["max_spend_quote"] = _clamp("max_spend_quote", c.get("max_spend_quote"), 0.0, 10000000, 0.0)
    c["poll_seconds"] = int(_clamp("poll_seconds", c.get("poll_seconds"), 5, 3600, 10))
    c["max_open_orders"] = int(_clamp("max_open_orders", c.get("max_open_orders"), 1, 50, 6))

    c["spread_guard_pct"] = _clamp("spread_guard_pct", c.get("spread_guard_pct"), 0.0005, 0.05, 0.003)
    c["stop_loss_pct"] = _clamp("stop_loss_pct", c.get("stop_loss_pct"), 0.01, 0.50, 0.08)
    c["max_drawdown_pct"] = _clamp("max_drawdown_pct", c.get("max_drawdown_pct"), 0.01, 0.50, 0.15)
    c["daily_loss_limit_pct"] = _clamp("daily_loss_limit_pct", c.get("daily_loss_limit_pct"), 0.01, 0.30, 0.05)

    if c.get("trailing_stop_enabled"):
        c["trailing_activation_pct"] = _clamp("trailing_activation_pct", c.get("trailing_activation_pct"), 0.005, 0.20, 0.02)
        c["trailing_distance_pct"] = _clamp("trailing_distance_pct", c.get("trailing_distance_pct"), 0.002, 0.10, 0.01)

    for bool_key in ("enabled", "dry_run", "auto_restart", "trailing_stop_enabled", "adaptive_tp_enabled", "use_kelly_sizing"):
        if bool_key in c:
            c[bool_key] = int(bool(c[bool_key]))

    return c, issues


# ─── Scanner / autopilot env config validation ──────────────────────────────

VALID_AUTOPILOT_MODES = {"STRICT_READY", "ALLOW_WATCH_DRYRUN"}
VALID_PREDICTION_HORIZONS = {"1h", "4h", "1d", "8h", "12h"}


def validate_scanner_env_config() -> Tuple[Dict[str, Any], List[str]]:
    """
    Validate scanner-related env vars and return (config_dict, issues).
    Non-fatal issues are warnings; fatal issues start with "ERROR:".
    """
    import os
    issues: List[str] = []
    cfg: Dict[str, Any] = {}

    def _env_float(key: str, default: float, lo: float, hi: float) -> float:
        raw = os.getenv(key, str(default))
        try:
            v = float(raw)
            if v < lo or v > hi:
                issues.append(f"{key}={v} out of range [{lo},{hi}], clamped")
                return max(lo, min(hi, v))
            return v
        except (TypeError, ValueError):
            issues.append(f"{key}={raw!r} invalid, using default {default}")
            return default

    cfg["min_entry_confidence"] = _env_float("MIN_ENTRY_CONFIDENCE", 0.65, 0.1, 0.99)
    cfg["min_p_tp_before_sl"] = _env_float("MIN_P_TP_BEFORE_SL", 0.55, 0.1, 0.95)
    cfg["max_allowed_drawdown_entry_pct"] = _env_float("MAX_ALLOWED_DRAWDOWN_ENTRY_PCT", 2.0, 0.5, 10.0)
    cfg["ml_train_lookback_days"] = int(_env_float("ML_TRAIN_LOOKBACK_DAYS", 180, 30, 730))
    cfg["adx_trend_threshold"] = _env_float("ADX_TREND_THRESHOLD", 20, 10, 40)

    watchlist_raw = os.getenv("WATCHLIST_ENABLED", "1").strip().lower()
    cfg["watchlist_enabled"] = watchlist_raw in ("1", "true", "yes", "y", "on")

    autopilot_mode = os.getenv("AUTOPILOT_MODE", "STRICT_READY").strip().upper()
    if autopilot_mode not in VALID_AUTOPILOT_MODES:
        issues.append(f"AUTOPILOT_MODE='{autopilot_mode}' invalid, using STRICT_READY")
        autopilot_mode = "STRICT_READY"
    cfg["autopilot_mode"] = autopilot_mode

    pred_horizon = os.getenv("PREDICTION_HORIZON", "4h").strip().lower()
    if pred_horizon not in VALID_PREDICTION_HORIZONS:
        issues.append(f"PREDICTION_HORIZON='{pred_horizon}' not in {VALID_PREDICTION_HORIZONS}, using 4h")
        pred_horizon = "4h"
    cfg["prediction_horizon"] = pred_horizon

    return cfg, issues
