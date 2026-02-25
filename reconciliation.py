# reconciliation.py
"""
Reconciliation: compare local expected positions/orders vs exchange state.

When RECONCILIATION_ENABLED=1, run periodically to detect mismatches.
Feature flag: RECONCILIATION_ENABLED (default: 0)
"""

import os
import logging
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

_ENABLED = os.getenv("RECONCILIATION_ENABLED", "0").strip().lower() in ("1", "true", "yes", "y", "on")


def reconcile_bot(
    bot_id: int,
    symbol: str,
    expected_base_pos: float,
    client: Any,
    is_kraken: bool,
) -> Tuple[bool, Optional[str]]:
    """
    Compare expected base position vs exchange balance for a bot.
    Returns (match_ok, warning_message).
    """
    if not _ENABLED:
        return True, None
    try:
        if is_kraken:
            parts = (symbol or "").split("/")
            base = (parts[0] or "").strip()
            if base == "XBT":
                base = "XXBT"
            bal = client.fetch_balance()
            free = (bal.get("free") or {}) or {}
            actual = float(free.get(base, 0) or 0)
        else:
            # AlpacaAdapter: use fetch_balance, positions keyed by base (e.g. AAPL)
            base = (symbol.split("/")[0] if "/" in symbol else symbol).strip()
            bal = client.fetch_balance()
            used = (bal.get("used") or {}) or {}
            free = (bal.get("free") or {}) or {}
            actual = float((used.get(base) or 0) + (free.get(base) or 0))
        diff = abs(actual - expected_base_pos)
        if diff > 0.0001:
            msg = f"Reconciliation mismatch bot_id={bot_id} {symbol}: expected={expected_base_pos:.6f} actual={actual:.6f}"
            logger.warning("RECONCILIATION: %s", msg)
            return False, msg
    except Exception as e:
        logger.warning("Reconciliation failed bot_id=%s: %s", bot_id, e)
        return True, None
    return True, None


def run_reconciliation(
    bots_snapshots: List[Dict[str, Any]],
    get_client_fn,
) -> List[str]:
    """
    Run reconciliation for all bots. get_client_fn(bot) -> (client, is_kraken).
    Returns list of warning messages.
    """
    if not _ENABLED:
        return []
    warnings = []
    for snap in bots_snapshots or []:
        bot_id = int(snap.get("bot_id") or 0)
        symbol = str(snap.get("symbol") or "")
        expected = float(snap.get("base_pos") or 0)
        if not symbol or expected <= 0:
            continue
        try:
            client, is_kraken = get_client_fn({"id": bot_id, "symbol": symbol})
            if not client:
                continue
            ok, msg = reconcile_bot(bot_id, symbol, expected, client, is_kraken)
            if not ok and msg:
                warnings.append(msg)
        except Exception as e:
            logger.warning("Reconciliation error bot_id=%s: %s", bot_id, e)
    return warnings
