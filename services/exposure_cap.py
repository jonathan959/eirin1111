"""Per-symbol exposure cap validation for bot saves."""

from __future__ import annotations

import math
from typing import Any, Dict, Optional


def build_exposure_cap_error(
    portfolio_value_usd: float,
    base_order_quote: float,
    per_symbol_pct: float,
    max_total_exposure_pct: float,
) -> Optional[Dict[str, Any]]:
    """Return 422 body dict if base order exceeds per-symbol cap; else None."""
    pv = float(portfolio_value_usd or 0.0)
    base = float(base_order_quote or 0.0)
    psp = float(per_symbol_pct or 0.0)
    if pv <= 0 or base <= 0 or psp <= 0:
        return None
    eff = pv * psp
    if base <= eff + 1e-9:
        return None
    # Smallest pct (fraction) to fit base, rounded up to 0.01 (100 bps)
    raw_pct = base / pv
    min_pct = math.ceil(raw_pct / 0.01) * 0.01
    min_pct = min(min_pct, float(max_total_exposure_pct or 1.0))
    max_base_fit = int(math.floor(eff))
    return {
        "error": "exposure_cap_conflict",
        "message": (
            f"Base order ${base:.2f} exceeds per-symbol cap ${eff:.2f} "
            f"(portfolio ${pv:.2f} x {psp*100:.2f}%)."
        ),
        "suggestions": {
            "per_symbol_pct": round(min_pct, 2),
            "base_order_quote": float(max(1, max_base_fit)),
        },
        "current": {
            "portfolio_value": round(pv, 2),
            "per_symbol_pct": psp,
            "base_order_quote": base,
            "effective_cap_usd": round(eff, 2),
        },
    }
