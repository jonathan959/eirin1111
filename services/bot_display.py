"""Compute a human-readable bot row *lifecycle* display_state from snapshot + deal.

Separately from ``bot_status`` (signal vs action): this covers IDLE / MANAGING /
WAITING_FOR_FILL — operational state, not conviction."""
from __future__ import annotations

import time
from typing import Any, Dict, Optional


def _is_truthy(x: Any) -> bool:
    if x is None:
        return False
    if isinstance(x, (int, float)):
        return x > 0
    return bool(x)


def compute_display_state(snap: Optional[Dict[str, Any]], deal: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Return {state, label, detail, urgency, spin}."""
    snap = snap or {}
    deal = deal or {}
    now = int(time.time())
    running = bool(snap.get("running"))
    last_event = str(snap.get("last_event") or "").strip()
    base_pos = float(snap.get("base_pos") or 0.0)
    tp_price = snap.get("tp_price")
    errors = snap.get("errors")
    cooldown_until = int(snap.get("cooldown_until") or 0)
    decision_action = str(snap.get("decision_action") or "").upper()
    risk_state = str(snap.get("risk_state") or "").upper()

    if not running:
        if last_event.lower().startswith(("stopped", "manual stop", "idle")):
            return _out("STOPPED", "Stopped", last_event, "normal", False)
        return _out("STOPPED", "Stopped", last_event or "Not running.", "normal", False)

    if risk_state in ("KILL", "KILLED", "PAUSED"):
        return _out("PAUSED", "Paused", snap.get("risk_reason") or "Risk pause", "warn", False)

    if errors and str(errors).strip():
        err = str(errors)
        short = err if len(err) <= 100 else err[:97] + "..."
        return _out("ERROR", "Error", short, "error", False)

    if cooldown_until and cooldown_until > now:
        secs = max(0, cooldown_until - now)
        return _out("COOLDOWN", f"Cooldown ({_fmt_secs(secs)})", "Waiting after recent loss", "normal", False)

    state = (deal.get("state") or "").upper() if deal else ""
    if state == "CLOSING":
        return _out("CLOSING", "Closing", "Closing position", "normal", True)

    if base_pos > 0:
        if tp_price:
            return _out(
                "MANAGING",
                "Managing",
                f"TP @ {tp_price}" if isinstance(tp_price, (int, float)) else "Managing position",
                "normal",
                True,
            )
        return _out("MANAGING", "Managing", "Holding open position", "normal", True)

    if deal and state == "OPEN":
        repost = int(deal.get("repost_count") or 0)
        opened_at = int(deal.get("opened_at") or 0)
        age = max(0, now - opened_at) if opened_at else 0
        detail = f"{age}s, reposts={repost}"
        if repost >= 2:
            return _out("WAITING_FOR_FILL", f"Waiting ({_fmt_secs(age)})", detail, "warn", True)
        return _out("WAITING_FOR_FILL", f"Waiting ({_fmt_secs(age)})", detail, "normal", True)

    if decision_action in ("FILLED", "BUY_FILLED", "SELL_FILLED"):
        return _out("FILLED", "Filled", "Order filled", "normal", False)

    return _out("IDLE", "Idle", last_event or "Scanning…", "normal", True)


def _fmt_secs(secs: int) -> str:
    if secs < 60:
        return f"{secs}s"
    m, s = divmod(secs, 60)
    if m < 60:
        return f"{m}m{s:02d}s"
    h, m = divmod(m, 60)
    return f"{h}h{m:02d}m"


def _out(state: str, label: str, detail: str, urgency: str, spin: bool) -> Dict[str, Any]:
    return {
        "state": state,
        "label": label,
        "detail": detail,
        "urgency": urgency,
        "spin": spin,
    }
