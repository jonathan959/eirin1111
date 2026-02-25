"""
Institutional-grade execution: TWAP, VWAP, iceberg orders.
"""
import logging
import os
import time
from typing import Any, Callable, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

ENABLE_TWAP = os.getenv("ENABLE_TWAP", "0").strip().lower() in ("1", "true", "yes", "y", "on")
TWAP_DURATION_MINUTES = int(os.getenv("TWAP_DURATION_MINUTES", "30"))
ENABLE_VWAP = os.getenv("ENABLE_VWAP", "0").strip().lower() in ("1", "true", "yes", "y", "on")
TWAP_MIN_NOTIONAL_USD = float(os.getenv("TWAP_MIN_NOTIONAL_USD", "5000"))
VWAP_MIN_NOTIONAL_USD = float(os.getenv("VWAP_MIN_NOTIONAL_USD", "10000"))


def twap_schedule(
    total_size: float,
    duration_minutes: int,
    *,
    side: str = "buy",
    size_in_base: bool = True,
) -> List[Dict[str, Any]]:
    """
    Generate TWAP schedule: equal-sized chunks evenly over time.
    
    Returns: [{"size": X, "delay_sec": Y, "interval_sec": Z}, ...]
    """
    if total_size <= 0 or duration_minutes <= 0:
        return []
    # Minimum 2 chunks, max ~20 to avoid excessive API calls
    n_chunks = max(2, min(20, max(2, duration_minutes // 2)))
    chunk_size = total_size / n_chunks
    interval_sec = (duration_minutes * 60) / n_chunks
    chunks = []
    for i in range(n_chunks):
        chunks.append({
            "size": chunk_size,
            "size_in_base": size_in_base,
            "delay_sec": i * interval_sec,
            "interval_sec": interval_sec,
            "chunk_index": i,
            "total_chunks": n_chunks,
            "side": side,
        })
    return chunks


def vwap_schedule(
    total_size: float,
    volume_profile: List[Tuple[int, float]],
    *,
    side: str = "buy",
    size_in_base: bool = True,
    duration_minutes: int = 60,
) -> List[Dict[str, Any]]:
    """
    Generate VWAP schedule: chunks weighted by historical volume profile.
    volume_profile: [(minute_of_day, vol_pct), ...] e.g. [(540, 0.05), (600, 0.08), ...]
    If empty, falls back to TWAP.
    """
    if total_size <= 0:
        return []
    if not volume_profile or len(volume_profile) < 2:
        return twap_schedule(total_size, duration_minutes, side=side, size_in_base=size_in_base)
    total_vol = sum(v for _, v in volume_profile)
    if total_vol <= 0:
        return twap_schedule(total_size, duration_minutes, side=side, size_in_base=size_in_base)
    chunks = []
    for i, (minute, vol_pct) in enumerate(volume_profile):
        pct = vol_pct / total_vol
        size = total_size * pct
        if size <= 0:
            continue
        delay_sec = minute * 60  # Approximate
        chunks.append({
            "size": size,
            "size_in_base": size_in_base,
            "delay_sec": delay_sec,
            "vol_pct": pct,
            "chunk_index": i,
            "side": side,
        })
    return sorted(chunks, key=lambda x: x["delay_sec"])


def get_default_volume_profile() -> List[Tuple[int, float]]:
    """Typical intraday volume: low open, high mid, lower close."""
    return [
        (0, 0.03), (30, 0.05), (60, 0.06), (90, 0.05), (120, 0.04),
        (180, 0.05), (240, 0.06), (300, 0.07), (360, 0.08), (420, 0.07),
        (480, 0.05), (540, 0.04), (600, 0.03), (660, 0.02),
    ]


def should_use_twap(size_quote: float, size_base: float, price: float) -> bool:
    """Decide if order is large enough for TWAP."""
    if not ENABLE_TWAP:
        return False
    notional = size_quote if size_quote > 0 else (size_base * price if price > 0 else 0)
    return notional >= TWAP_MIN_NOTIONAL_USD


def should_use_vwap(size_quote: float, size_base: float, price: float) -> bool:
    """Decide if order is large enough for VWAP."""
    if not ENABLE_VWAP:
        return False
    notional = size_quote if size_quote > 0 else (size_base * price if price > 0 else 0)
    return notional >= VWAP_MIN_NOTIONAL_USD


def execute_twap_sync(
    symbol: str,
    total_size: float,
    side: str,
    is_quote: bool,
    client: Any,
    duration_minutes: Optional[int] = None,
    dry_run: bool = True,
    on_chunk: Optional[Callable[[Dict[str, Any]], None]] = None,
) -> List[Dict[str, Any]]:
    """
    Execute TWAP: place chunks over time (blocking/sync).
    client: KrakenClient or AlpacaAdapter with create_limit_buy_base, create_market_buy_quote, etc.
    Returns list of placed orders.
    """
    duration = duration_minutes or TWAP_DURATION_MINUTES
    schedule = twap_schedule(total_size, duration, side=side, size_in_base=not is_quote)
    results = []
    for chunk in schedule:
        delay = chunk.get("delay_sec", 0)
        if delay > 0:
            time.sleep(delay)
        size = chunk["size"]
        try:
            if side == "buy":
                if is_quote:
                    ticker = client.fetch_ticker(symbol)
                    price = float(ticker.get("last", 0)) or float(ticker.get("ask", 0))
                    if price <= 0:
                        logger.warning("TWAP: no price for %s, skipping chunk", symbol)
                        continue
                    order = client.create_market_buy_quote(symbol, size, client_order_id=f"twap_{int(time.time())}")
                else:
                    ticker = client.fetch_ticker(symbol)
                    price = float(ticker.get("ask", 0)) or float(ticker.get("last", 0))
                    if price <= 0:
                        continue
                    order = client.create_limit_buy_base(symbol, size, price * 1.001, client_order_id=f"twap_{int(time.time())}")
            else:
                order = client.create_market_sell_base(symbol, size, client_order_id=f"twap_{int(time.time())}")
            if order:
                results.append(order)
                if on_chunk:
                    on_chunk({"order": order, "chunk": chunk})
        except Exception as e:
            logger.warning("TWAP chunk failed: %s", e)
    return results


def iceberg_display_size(total_size: float, min_display_frac: float = 1 / 15) -> float:
    """
    Kraken: display quantity must be at least 1/15 of total.
    Returns recommended display size.
    """
    return max(total_size * min_display_frac, total_size / 15)


# ============================================================
# Phase 2 Advanced Executor Class (spec-compliant API)
# ============================================================

def _get_limit_price(symbol: str, side: str, client: Any) -> float:
    """Get optimal limit price from order book."""
    try:
        ob = client.fetch_order_book(symbol) if hasattr(client, "fetch_order_book") else {}
        bids = ob.get("bids") or []
        asks = ob.get("asks") or []
        if side == "buy" and bids:
            return float(bids[0][0])
        if side == "sell" and asks:
            return float(asks[0][0])
        ticker = client.fetch_ticker(symbol) if hasattr(client, "fetch_ticker") else {}
        return float(ticker.get("ask" if side == "buy" else "bid", 0) or ticker.get("last", 0))
    except Exception:
        return 0.0


def _get_order_status(order_id: str, client: Any, symbol: Optional[str] = None) -> Dict[str, Any]:
    """Get order status. Returns dict with status, filled_size, avg_fill_price."""
    try:
        if hasattr(client, "fetch_order"):
            o = client.fetch_order(order_id, symbol)
            if o:
                filled = float(o.get("filled", 0) or o.get("filled_qty", 0) or o.get("filled_size", 0))
                avg = float(o.get("average", 0) or o.get("avg_fill_price", 0) or o.get("price", 0))
                return {"status": str(o.get("status", "unknown")).lower(), "filled_size": filled, "avg_fill_price": avg}
        return {"status": "unknown", "filled_size": 0, "avg_fill_price": 0}
    except Exception:
        return {"status": "unknown", "filled_size": 0, "avg_fill_price": 0}


class AdvancedExecutor:
    """
    Phase 2: Institutional-grade execution (TWAP, VWAP, Iceberg).
    Wraps client (Kraken/Alpaca) for algo execution.
    """

    def __init__(self, client: Any):
        self.client = client

    def execute_twap(
        self,
        symbol: str,
        side: str,
        total_size: float,
        duration_minutes: int = 30,
        num_slices: int = 10,
        dry_run: bool = True,
    ) -> Dict[str, Any]:
        """
        Time-Weighted Average Price execution.
        Returns status, total_filled, avg_fill_price, slippage_vs_start.
        """
        from phase2_data_fetcher import get_current_price

        start_price = get_current_price(symbol)
        if start_price <= 0 and hasattr(self.client, "fetch_ticker"):
            t = self.client.fetch_ticker(symbol)
            start_price = float(t.get("last", 0) or t.get("ask", 0) or t.get("bid", 0))

        results = execute_twap_sync(
            symbol=symbol,
            total_size=total_size,
            side=side,
            is_quote=False,
            client=self.client,
            duration_minutes=duration_minutes,
            dry_run=dry_run,
        )

        total_filled = sum(float(r.get("filled", 0) or r.get("filled_qty", 0) or r.get("filled_size", 0) or 0) for r in results if isinstance(r, dict))
        total_cost = 0.0
        for r in results:
            if isinstance(r, dict):
                filled = float(r.get("filled", 0) or r.get("filled_qty", 0) or 0)
                avg_px = float(r.get("average", 0) or r.get("avg_fill_price", 0) or 0)
                total_cost += filled * avg_px
        avg_fill = total_cost / total_filled if total_filled > 0 else 0
        slippage = 0.0
        if start_price > 0:
            slippage = ((avg_fill - start_price) / start_price * 100) if side == "buy" else ((start_price - avg_fill) / start_price * 100)

        return {
            "status": "completed" if results else "no_orders",
            "total_filled": total_filled,
            "total_requested": total_size,
            "fill_rate": total_filled / total_size if total_size > 0 else 0,
            "avg_fill_price": avg_fill,
            "start_price": start_price,
            "slippage_vs_start": slippage,
            "execution_time_minutes": duration_minutes,
            "num_child_orders": len(results),
            "child_orders": results,
        }

    def execute_iceberg(
        self,
        symbol: str,
        side: str,
        total_size: float,
        visible_size: Optional[float] = None,
        dry_run: bool = True,
    ) -> Dict[str, Any]:
        """
        Iceberg order: show small portion publicly, replace as filled.
        visible_size: default 10% of total.
        """
        visible = visible_size if visible_size is not None and visible_size > 0 else total_size * 0.10
        visible = max(visible, total_size / 15)  # Kraken minimum

        remaining = total_size
        child_orders = []
        total_filled = 0.0
        total_cost = 0.0
        max_iter = 20

        for _ in range(max_iter):
            if remaining <= 0:
                break
            slice_size = min(visible, remaining)
            if dry_run:
                break
            try:
                price = _get_limit_price(symbol, side, self.client)
                if price <= 0:
                    break
                if side == "buy":
                    order = self.client.create_limit_buy_base(symbol, slice_size, price, client_order_id=f"iceberg_{int(time.time())}")
                else:
                    order = self.client.create_limit_sell_base(symbol, slice_size, price, client_order_id=f"iceberg_{int(time.time())}")
                if order:
                    child_orders.append(order)
                    filled = float(order.get("filled", 0) or order.get("filled_qty", 0) or 0)
                    avg_px = float(order.get("average", 0) or order.get("avg_fill_price", 0) or price)
                    total_filled += filled
                    total_cost += filled * avg_px
                    remaining -= filled
                time.sleep(1)
            except Exception as e:
                logger.warning("Iceberg slice failed: %s", e)
                break

        avg_fill = total_cost / total_filled if total_filled > 0 else 0
        return {
            "status": "completed",
            "total_filled": total_filled,
            "avg_fill_price": avg_fill,
            "visible_size_per_order": visible,
            "num_child_orders": len(child_orders),
            "child_orders": child_orders,
        }
