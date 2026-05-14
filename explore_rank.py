"""Explore feed ranking + backtest row helpers (used by worker_api explore endpoints)."""
from __future__ import annotations

import json
from typing import Any, Dict, List, Optional, Tuple


def row_detail_json(row: Dict[str, Any]) -> Dict[str, Any]:
    raw = row.get("detail_json")
    if raw is None:
        return {}
    if isinstance(raw, dict):
        return raw
    try:
        out = json.loads(str(raw))
    except Exception:
        return {}
    return out if isinstance(out, dict) else {}


def bt_lookup(bt_90: Dict[str, Any], detail: Dict[str, Any], row: Dict[str, Any]) -> Dict[str, Any]:
    sid = (
        detail.get("explore_strategy_id")
        or detail.get("detected_strategy")
        or detail.get("recommended_strategy")
        or row.get("strategy")
    )
    if not sid:
        return {}
    sk = str(sid)
    return bt_90.get(sk) or bt_90.get(sk.lower()) or {}


def apply_smart_rank(
    rows: List[Dict[str, Any]],
    bt_90: Dict[str, Any],
    lim: int,
    now_i: int,
    *,
    live_perf: Optional[Dict[str, Any]] = None,
) -> Tuple[List[Dict[str, Any]], Dict[str, float]]:
    """
    Order explore rows by blended score + recency, then trim to ``lim``.
    ``live_perf`` may contain per-strategy live win rates (optional boost).
    """
    _ = bt_90
    if not rows:
        return [], {}

    lp = live_perf or {}

    def _key(r: Dict[str, Any]) -> Tuple[float, float]:
        base = float(r.get("conviction_score") or r.get("score") or 0.0)
        ts = float(r.get("updated_ts") or r.get("signal_ts") or 0.0)
        strat = str(
            r.get("strategy")
            or row_detail_json(r).get("detected_strategy")
            or ""
        )
        boost = 0.0
        if strat and strat in lp:
            try:
                boost = float(lp[strat].get("win_rate") or 0.0) * 0.01
            except Exception:
                boost = 0.0
        age_pen = max(0.0, min(30.0, (float(now_i) - ts) / 3600.0))
        blended = base + boost - 0.02 * age_pen
        return (blended, ts)

    ranked = sorted(rows, key=_key, reverse=True)
    trimmed = ranked[: max(1, int(lim))]
    pick_scores: Dict[str, float] = {}
    for r in trimmed:
        sym = str(r.get("symbol") or "")
        if sym:
            pick_scores[sym] = _key(r)[0]
    return trimmed, pick_scores
