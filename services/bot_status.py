"""Canonical two-axis status for bot cards: *signal* (what the setup looks like) vs
*action_state* (what execution is allowed to do right now).

Legacy / Explore code still uses verbal *ratings* ("Strong Buy", "Avoid", …).
Use :func:`recommendation_rating_legacy` for API payloads that must stay backward-compatible.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Dict, Optional

# --- User-facing signal (strategy / scanner view) ---
SIGNAL_STRONG_BUY = "Strong Buy"
SIGNAL_BUY = "Buy"
SIGNAL_NEUTRAL = "Neutral"
SIGNAL_SELL = "Sell"
SIGNAL_STRONG_SELL = "Strong Sell"

# --- User-facing action (execution posture) ---
ACTION_TRADE = "Trade"
ACTION_WATCH = "Watch"
ACTION_WAIT = "Wait"
ACTION_BLOCKED = "Blocked"


@dataclass
class BotStatus:
    signal: str
    action_state: str
    reason: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "signal": self.signal,
            "action_state": self.action_state,
            "reason": self.reason,
        }


def recommendation_rating_legacy(conviction_grade: Optional[str], score: float) -> str:
    """Backward-compatible single-label rating used by Explore/recommendation JSON."""
    _d = (conviction_grade or "").strip().upper()
    if _d == "A":
        return "Strong Buy"
    if _d == "B":
        return "Buy"
    if _d == "C":
        return "Watch"
    if _d == "D":
        return "Avoid"
    s = float(score or 0.0)
    if s >= 85:
        return "Strong Buy"
    if s >= 55:
        return "Buy"
    if s >= 40:
        return "Watch"
    return "Avoid"


def signal_from_recommendation(latest_signal: Optional[Dict[str, Any]]) -> str:
    """Map scanner row / recommendation dict to the five-level *signal* axis."""
    if not latest_signal:
        return SIGNAL_NEUTRAL
    grade = str(latest_signal.get("conviction_grade") or "").strip().upper()
    score = float(latest_signal.get("score") or 0.0)
    if grade == "A" or score >= 85:
        return SIGNAL_STRONG_BUY
    if grade == "B" or score >= 55:
        return SIGNAL_BUY
    if grade == "C" or (score >= 40 and score < 55):
        return SIGNAL_NEUTRAL
    if grade == "D" or score < 40:
        return SIGNAL_NEUTRAL
    return SIGNAL_NEUTRAL


# Strategy/runtime exit actions from BotRunner (smart_decide / risk exits).
_STRONG_EXIT_DECISION_PREFIXES = ("EXIT",)
_STRONG_EXIT_DECISION_EXACT = frozenset(
    {
        "SELL",
        "CLOSE",
        "EXIT",
        "STOP_LOSS",
        "TAKE_PROFIT",
        "TRAILING_EXIT",
        "PARTIAL_EXIT",
        "TIME_EXIT",
    }
)

# Long-only book: reco score strictly below this while holding → Sell on the *signal* axis.
# Option B (stricter): use 35.0 instead of 40.0 — confirm with ops if too noisy.
POSITION_WEAK_SIGNAL_SCORE_LT = 40.0


def strategy_decision_implies_strong_sell(decision_action: Optional[str]) -> bool:
    """True when the engine's latest strategy decision is an exit / risk-off sell."""
    da = str(decision_action or "").strip().upper()
    if da in _STRONG_EXIT_DECISION_EXACT:
        return True
    for pref in _STRONG_EXIT_DECISION_PREFIXES:
        if pref in da and da != "ENTER":
            return True
    return False


def compute_signal_axis(
    latest_signal: Optional[Dict[str, Any]],
    *,
    base_pos: float = 0.0,
    strategy_decision_action: Optional[str] = None,
) -> str:
    """
    Full signal axis including exits:

    - Explicit exit decisions from the runner → Strong Sell.
    - Long position + weak scanner score → Sell (bottom band default: score < 40).
    - Otherwise recommendation-only mapping via :func:`signal_from_recommendation`.
    """
    if strategy_decision_implies_strong_sell(strategy_decision_action):
        return SIGNAL_STRONG_SELL

    has_long = float(base_pos or 0.0) > 0.0
    score = float(latest_signal.get("score") or 0.0) if latest_signal else 0.0
    if has_long and score < POSITION_WEAK_SIGNAL_SCORE_LT:
        return SIGNAL_SELL

    return signal_from_recommendation(latest_signal)


def load_latest_signal_for_bot(bot: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Best-effort DB lookup of the latest recommendation for the bot symbol."""
    sym = str(bot.get("symbol") or "").strip()
    if not sym:
        return None
    try:
        from db import get_recommendation
    except Exception:
        return None
    for h in ("long", "short"):
        try:
            row = get_recommendation(sym, h)
        except Exception:
            row = None
        if not row:
            continue
        metrics: Dict[str, Any] = {}
        try:
            raw = row.get("metrics_json")
            if raw:
                metrics = json.loads(raw)
        except (json.JSONDecodeError, TypeError, ValueError):
            metrics = {}
        cg = row.get("conviction_grade") or metrics.get("conviction_grade")
        return {
            "score": float(row.get("score") or 0.0),
            "conviction_grade": cg,
            "horizon": h,
        }
    return None


def compute_bot_status(
    bot: Dict[str, Any],
    latest_signal: Optional[Dict[str, Any]],
    intelligence_decision: Optional[Dict[str, Any]],
    risk_ctx: Optional[Dict[str, Any]],
    exec_gate_result: Optional[Dict[str, Any]],
    *,
    global_pause: bool = False,
    kill_switch: bool = False,
    allow_live_trading: bool = True,
    base_pos: float = 0.0,
) -> BotStatus:
    """
    Args:
        bot: full or partial bot row (needs dry_run, symbol, optional conviction_level)
        latest_signal: from :func:`load_latest_signal_for_bot` or tests
        intelligence_decision: decision_action, decision_reason, allowed_actions
        risk_ctx: level, reason
        exec_gate_result: gate to_dict() with allowed: bool, reason, spread_pct, …
    """
    intel = intelligence_decision or {}
    risk = risk_ctx or {}
    gate = exec_gate_result or {}

    signal = compute_signal_axis(
        latest_signal,
        base_pos=float(base_pos or 0.0),
        strategy_decision_action=intel.get("decision_action"),
    )

    dry_run = bool(int(bot.get("dry_run", 1)))
    if dry_run:
        return BotStatus(
            signal=signal,
            action_state=ACTION_WATCH,
            reason="Dry run — orders are simulated; no live broker execution.",
        )

    if kill_switch:
        return BotStatus(signal=signal, action_state=ACTION_BLOCKED, reason="Kill switch is ON.")

    if global_pause:
        return BotStatus(signal=signal, action_state=ACTION_BLOCKED, reason="Server trading pause is active.")

    if not allow_live_trading and not dry_run:
        return BotStatus(
            signal=signal,
            action_state=ACTION_BLOCKED,
            reason="ALLOW_LIVE_TRADING is off — live execution disabled in configuration.",
        )

    lvl = str(risk.get("level") or "").upper()
    rreason = str(risk.get("reason") or "").strip()

    if lvl == "CRITICAL" and rreason:
        return BotStatus(signal=signal, action_state=ACTION_BLOCKED, reason=rreason[:240])

    allowed = gate.get("allowed")
    if allowed is False:
        spread = gate.get("spread_pct")
        extra = ""
        if spread is not None:
            try:
                extra = f" spread {float(spread) * 100:.3f}%"
            except (TypeError, ValueError):
                extra = ""
        gr = str(gate.get("reason") or "Execution gate blocked.")
        return BotStatus(
            signal=signal,
            action_state=ACTION_WATCH,
            reason=(gr + extra).strip()[:240],
        )

    ia = str(intel.get("allowed_actions") or "").upper()
    if ia in ("NO_TRADE",):
        fr = str(intel.get("final_reason") or intel.get("decision_reason") or "Intelligence layer: NO_TRADE.")
        return BotStatus(signal=signal, action_state=ACTION_BLOCKED, reason=fr[:240])

    if ia == "MANAGE_ONLY" and float(base_pos or 0.0) <= 0:
        return BotStatus(
            signal=signal,
            action_state=ACTION_WAIT,
            reason="Manage-only: new entries disabled; existing positions still managed.",
        )

    da = str(intel.get("decision_action") or "").upper()
    dr = str(intel.get("decision_reason") or "")

    if da == "PAUSE":
        return BotStatus(signal=signal, action_state=ACTION_WAIT, reason=(dr or "Strategy paused.")[:240])

    dr_l = dr.lower()
    wait_hints = (
        "confidence",
        "entry",
        "threshold",
        "wait",
        "not suitable",
        "regime",
        "blocked",
        "spread",
    )
    if da in ("HOLD", "NOOP"):
        msg = dr or "Strategy holding — waiting for next actionable tick."
        return BotStatus(signal=signal, action_state=ACTION_WAIT, reason=msg[:240])
    if not da and any(h in dr_l for h in wait_hints) and dr:
        return BotStatus(signal=signal, action_state=ACTION_WAIT, reason=dr[:240])
    if any(h in dr_l for h in wait_hints) and dr:
        return BotStatus(signal=signal, action_state=ACTION_WAIT, reason=dr[:240])

    return BotStatus(
        signal=signal,
        action_state=ACTION_TRADE,
        reason="Live trading permitted; gates report OK for this tick.",
    )
