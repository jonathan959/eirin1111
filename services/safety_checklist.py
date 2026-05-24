"""Safety checklist service.

Produces the rich list-shape the Safety page wants:

    [
        {"name": "...", "passed": bool, "detail": "...", "fix_url": "..."},
        ...
    ]

Backing inputs are fetched from already-existing worker_api primitives
(keys, Kraken / Alpaca readiness, strategy leaderboard win rates, etc.).
All data access is lazy-imported so this module has zero import-time
side effects — easy to unit test.
"""
from __future__ import annotations

import os
import time
from typing import Any, Callable, Dict, List, Optional


ITEM = Dict[str, Any]


def _item(name: str, passed: bool, detail: str, fix_url: str = "") -> ITEM:
    return {
        "name": name,
        "passed": bool(passed),
        "detail": detail,
        "fix_url": fix_url,
    }


def _env(key: str) -> str:
    return (os.getenv(key) or "").strip()


def _truthy(v: str) -> bool:
    return v.lower() in ("1", "true", "yes", "on", "y")


def compute_live_readiness() -> Dict[str, Any]:
    """One-call gate for live-promotion endpoints. Returns the full checklist
    dict (same shape as build_checklist) plus a `blocking_reasons` list with the
    detail strings of every failed item — so a 409 from PUT /api/bots/{id} (or
    POST /api/go_live/confirm) can show the user exactly what to fix.

    Lazy-imports worker_api globals so unit tests don't pay the import cost.
    """
    try:
        # Late imports so this module stays import-cost-free for tests
        from worker_api import (
            _kraken_ready, alpaca_paper, alpaca_live, _kill_switch_state,
            ALLOW_LIVE_TRADING, LIVE_TRADING_ENABLED, WORKER_API_TOKEN,
            get_strategy_leaderboard,
        )
        from db import list_bots, get_setting
    except Exception as e:
        # If imports fail we cannot grant live-readiness — fail closed.
        return {
            "ok": False, "live_ready": False,
            "blocking_reasons": [f"safety wiring error: {type(e).__name__}: {e}"],
            "flags": {},
        }

    try:
        all_bots = list_bots() or []
    except Exception:
        all_bots = []
    has_live_bots = any(int(b.get("dry_run", 1) or 1) == 0 for b in all_bots)
    try:
        vals = [float(b.get("daily_loss_limit_pct") or 0) for b in all_bots]
        daily_loss_pct = max(vals) if vals else 0.0
    except Exception:
        daily_loss_pct = 0.0
    try:
        kill_switch_tested = str(get_setting("kill_switch_tested", "0")).strip().lower() in ("1", "true", "yes")
        autopilot_configured = bool((get_setting("autopilot_config", "") or "").strip())
    except Exception:
        kill_switch_tested = False
        autopilot_configured = False
    notifications_wired = bool((os.getenv("DISCORD_WEBHOOK_URL") or "").strip())
    try:
        leaderboard = get_strategy_leaderboard(window_days=90)
    except Exception:
        leaderboard = []

    result = build_checklist(
        kraken_ready=bool(_kraken_ready()),
        alpaca_paper_ready=bool(alpaca_paper is not None),
        alpaca_live_ready=bool(alpaca_live is not None),
        worker_api_token_set=bool(WORKER_API_TOKEN),
        kill_switch_on=bool(_kill_switch_state()),
        allow_live_trading=bool(ALLOW_LIVE_TRADING),
        live_trading_enabled=bool(LIVE_TRADING_ENABLED),
        has_live_bots=has_live_bots,
        daily_loss_limit_pct=float(daily_loss_pct or 0),
        kill_switch_tested=bool(kill_switch_tested),
        strategy_leaderboard=leaderboard or [],
        autopilot_configured=bool(autopilot_configured),
        notifications_wired=bool(notifications_wired),
    )
    result["blocking_reasons"] = [it["detail"] for it in result.get("items", []) if not it.get("passed")]
    result["flags"] = result.get("context", {})
    return result


def build_checklist(
    *,
    kraken_ready: bool,
    alpaca_paper_ready: bool,
    alpaca_live_ready: bool,
    worker_api_token_set: bool,
    kill_switch_on: bool,
    allow_live_trading: bool,
    live_trading_enabled: bool,
    has_live_bots: bool,
    daily_loss_limit_pct: float = 0.0,
    kill_switch_tested: bool = False,
    strategy_leaderboard: Optional[List[Dict[str, Any]]] = None,
    autopilot_configured: bool = False,
    notifications_wired: bool = False,
) -> Dict[str, Any]:
    """Build the 8-item checklist.

    Each caller is expected to pull live values from worker_api. The
    signature stays strict / pure-python for ease of testing.
    """
    strategy_leaderboard = strategy_leaderboard or []

    items: List[ITEM] = []
    items.append(_item(
        "API keys present",
        passed=(_env("KRAKEN_API_KEY") != "" or _env("ALPACA_API_KEY_PAPER") != "" or _env("ALPACA_API_KEY_LIVE") != ""),
        detail=(
            "Kraken + Alpaca keys detected in .env"
            if (_env("KRAKEN_API_KEY") or _env("ALPACA_API_KEY_PAPER") or _env("ALPACA_API_KEY_LIVE"))
            else "No exchange API keys found in .env — see Settings → API Keys"
        ),
        fix_url="/settings",
    ))

    items.append(_item(
        "Alpaca live connection",
        passed=alpaca_live_ready,
        detail="Alpaca live API reachable" if alpaca_live_ready else "Alpaca live API not connected",
        fix_url="/settings",
    ))

    items.append(_item(
        "Kraken connection",
        passed=kraken_ready,
        detail="Kraken API reachable" if kraken_ready else "Kraken API not connected",
        fix_url="/settings",
    ))

    items.append(_item(
        "Worker API token",
        passed=worker_api_token_set,
        detail=(
            "WORKER_API_TOKEN is set in .env"
            if worker_api_token_set
            else "Missing WORKER_API_TOKEN — required to call live trade endpoints"
        ),
        fix_url="/safety",
    ))

    items.append(_item(
        "Daily loss limit configured",
        passed=(daily_loss_limit_pct is not None and daily_loss_limit_pct > 0),
        detail=(
            f"Daily loss limit = {daily_loss_limit_pct * 100:.1f}%"
            if daily_loss_limit_pct and daily_loss_limit_pct > 0
            else "Set a non-zero daily loss limit in Settings → Risk Limits"
        ),
        fix_url="/settings",
    ))

    items.append(_item(
        "Kill switch tested",
        passed=bool(kill_switch_tested),
        detail=(
            "Kill switch has been toggled once in the past (good)."
            if kill_switch_tested
            else "Toggle the kill switch on/off on the Safety page at least once."
        ),
        fix_url="/safety",
    ))

    # At least one strategy with >= 40% win rate (10+ deals).
    passable_strats = [
        s for s in strategy_leaderboard
        if int(s.get("trades") or 0) >= 10 and float(s.get("win_rate") or 0) >= 40.0
    ]
    top = sorted(passable_strats, key=lambda s: float(s.get("win_rate") or 0), reverse=True)[:1]
    items.append(_item(
        "Strategy with ≥40% win rate",
        passed=bool(top),
        detail=(
            f"{top[0].get('strategy', '?')} @ {float(top[0].get('win_rate') or 0):.1f}% win rate"
            if top
            else "No strategy has ≥ 10 closed deals AND ≥ 40% win rate yet — run backtests and/or let bots gather data."
        ),
        fix_url="/strategies",
    ))

    items.append(_item(
        "Autopilot configured",
        passed=bool(autopilot_configured),
        detail="Autopilot has a valid config" if autopilot_configured else "Finish /setup-autopilot to enable autopilot",
        fix_url="/setup-autopilot",
    ))

    items.append(_item(
        "Notifications wired",
        passed=bool(notifications_wired),
        detail=(
            "Discord/Telegram webhook is set"
            if notifications_wired
            else "Set DISCORD_WEBHOOK_URL in .env to receive trade notifications"
        ),
        fix_url="/settings",
    ))

    # Negative checks last; these block live trading if not OK.
    items.append(_item(
        "Kill switch OFF",
        passed=not kill_switch_on,
        detail=("Kill switch is disabled" if not kill_switch_on else "Kill switch is ENABLED — live trading blocked"),
        fix_url="/safety",
    ))

    passed = sum(1 for it in items if it["passed"])
    total = len(items)
    all_pass = passed == total

    return {
        "ok": True,
        "items": items,
        "passed": passed,
        "total": total,
        "all_pass": bool(all_pass),
        "live_ready": bool(
            all_pass
            and allow_live_trading
            and live_trading_enabled
            and worker_api_token_set
            and (kraken_ready or alpaca_live_ready)
            and not kill_switch_on
        ),
        "context": {
            "allow_live_trading": bool(allow_live_trading),
            "live_trading_enabled": bool(live_trading_enabled),
            "has_live_bots": bool(has_live_bots),
            "generated_at": int(time.time()),
        },
    }
