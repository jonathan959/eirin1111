"""User settings service.

Exposes a single, consistent shape for GET/PATCH /api/settings:

    {
        "kraken":       {"connected": bool, "last_check": int|null},
        "alpaca_paper": {"connected": bool, "last_check": int|null},
        "alpaca_live":  {"connected": bool, "last_check": int|null},
        "defaults":     {"tp_pct": 0.015, "hard_sl_pct": 0.02, ...},
        "notifications":{"enabled": true, "discord": true, ...},
        "ui_prefs":     {"theme": "dark", "accent": "#7c3aed", ...},
        "risk_limits":  {"daily_loss_limit_pct": 0.03, ...},
    }

API keys are *never* returned by this service. They live in `.env` only.
This avoids leaking credentials through an auth-less API and matches
the repo's "env-only" rule.
"""
from __future__ import annotations

import json
import os
from typing import Any, Dict, Optional

try:
    from db import get_setting, set_setting
except Exception:  # pragma: no cover
    def get_setting(_k: str, _d: Optional[str] = None) -> Optional[str]:
        return _d

    def set_setting(_k: str, _v: Any) -> None:
        return None


# -- Defaults --------------------------------------------------------------

DEFAULT_DEFAULTS: Dict[str, Any] = {
    "tp_pct": 0.015,
    "hard_sl_pct": 0.02,
    "base_quote": 25.0,
    "max_safety_orders": 3,
    "first_dev_pct": 0.01,
    "step_multiplier": 1.2,
    "strategy": "trend_follow_auto",
    "adaptive_limit": True,
    "limit_timeout_sec": 45,
    "max_slippage_pct": 0.003,
    "trailing_stop_enabled": True,
}

DEFAULT_NOTIFICATIONS: Dict[str, Any] = {
    "enabled": True,
    "discord": True,
    "trade_executed": True,
    "tp_hit": True,
    "sl_hit": True,
    "bot_error": True,
    "drawdown": True,
    "daily_summary": True,
}

DEFAULT_UI_PREFS: Dict[str, Any] = {
    "theme": "dark",
    "accent": "#7c3aed",
    "default_timeframe": "1H",
    "auto_refresh_sec": 15,
    "show_debug_logs": False,
    "high_conviction_only": True,
    "low_conviction_threshold": 55,
}

DEFAULT_RISK_LIMITS: Dict[str, Any] = {
    "daily_loss_limit_pct": 0.03,
    "max_drawdown_pct": 0.10,
    "max_position_pct": 0.20,
    "max_correlated_pct": 0.50,
    "cooldown_after_loss_min": 30,
    "max_deals_per_day": 5,
    "per_symbol_exposure_pct": 0.15,
    "max_concurrent_per_symbol": 4,
}


SETTINGS_GROUPS = {
    "defaults": DEFAULT_DEFAULTS,
    "notifications": DEFAULT_NOTIFICATIONS,
    "ui_prefs": DEFAULT_UI_PREFS,
    "risk_limits": DEFAULT_RISK_LIMITS,
}


def _load_group(name: str) -> Dict[str, Any]:
    defaults = SETTINGS_GROUPS.get(name, {})
    raw = get_setting(f"settings_{name}", None)
    if not raw:
        return dict(defaults)
    try:
        stored = json.loads(raw) if isinstance(raw, str) else (raw or {})
        if not isinstance(stored, dict):
            return dict(defaults)
        merged = dict(defaults)
        merged.update(stored)
        return merged
    except Exception:
        return dict(defaults)


def _save_group(name: str, data: Dict[str, Any]) -> None:
    if not isinstance(data, dict):
        return
    defaults = SETTINGS_GROUPS.get(name, {})
    merged = _load_group(name)
    for k, v in data.items():
        if k not in defaults:
            # Ignore unknown keys to keep the schema tight.
            continue
        merged[k] = v
    set_setting(f"settings_{name}", json.dumps(merged))


# -- Connection status -----------------------------------------------------

def _connection_block(ready: bool) -> Dict[str, Any]:
    return {
        "connected": bool(ready),
        "last_check": None,  # API layer can fill this in from /health timestamps
    }


def connection_status(kraken_ready: bool, alpaca_paper_ready: bool, alpaca_live_ready: bool) -> Dict[str, Any]:
    return {
        "kraken": _connection_block(kraken_ready),
        "alpaca_paper": _connection_block(alpaca_paper_ready),
        "alpaca_live": _connection_block(alpaca_live_ready),
    }


def snapshot(kraken_ready: bool, alpaca_paper_ready: bool, alpaca_live_ready: bool) -> Dict[str, Any]:
    """Return the full /api/settings GET payload."""
    out: Dict[str, Any] = {
        "ok": True,
        **connection_status(kraken_ready, alpaca_paper_ready, alpaca_live_ready),
        "defaults": _load_group("defaults"),
        "notifications": _load_group("notifications"),
        "ui_prefs": _load_group("ui_prefs"),
        "risk_limits": _load_group("risk_limits"),
        "flags": {
            "allow_live_trading": _truthy(os.getenv("ALLOW_LIVE_TRADING", "")),
            "live_trading_enabled": _truthy(os.getenv("LIVE_TRADING_ENABLED", "")),
            "worker_api_token_set": bool(os.getenv("WORKER_API_TOKEN", "").strip()),
            "discord_webhook_set": bool(os.getenv("DISCORD_WEBHOOK_URL", "").strip()),
        },
    }
    return out


def _truthy(v: Any) -> bool:
    return str(v or "").strip().lower() in ("1", "true", "yes", "on")


def apply_patch(patch: Dict[str, Any]) -> Dict[str, Any]:
    """Apply a partial update to the user settings.

    Accepts any of the top-level keys in SETTINGS_GROUPS. Unknown keys are
    silently ignored (we never want the API to surface arbitrary
    bot_manager internals here).
    """
    if not isinstance(patch, dict):
        return {"ok": False, "error": "patch must be an object"}
    for group_name in SETTINGS_GROUPS:
        group_val = patch.get(group_name)
        if isinstance(group_val, dict):
            _save_group(group_name, group_val)
    return {"ok": True, "applied": list(SETTINGS_GROUPS.keys())}
