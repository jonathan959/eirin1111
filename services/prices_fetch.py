"""Split crypto vs stocks price fetch with per-symbol TTL cache and hard timeouts."""

from __future__ import annotations

import asyncio
import logging
import threading
import time
from typing import Any, Callable, Dict, List, Optional, Tuple

from cachetools import TTLCache

logger = logging.getLogger(__name__)

# Process-local, thread-safe symbol cache: key = (normalized_symbol, "crypto"|"stocks")
_SYMBOL_CACHE: TTLCache = TTLCache(maxsize=512, ttl=10.0)
_CACHE_LOCK = threading.Lock()

UPSTREAM_CRYPTO_BATCH_CALLS = 0
UPSTREAM_STOCKS_BATCH_CALLS = 0


def reset_upstream_counters() -> None:
    global UPSTREAM_CRYPTO_BATCH_CALLS, UPSTREAM_STOCKS_BATCH_CALLS
    UPSTREAM_CRYPTO_BATCH_CALLS = 0
    UPSTREAM_STOCKS_BATCH_CALLS = 0


def clear_symbol_cache_for_tests() -> None:
    with _CACHE_LOCK:
        _SYMBOL_CACHE.clear()


def _cache_get(key: Tuple[str, str]) -> Optional[Dict[str, Any]]:
    with _CACHE_LOCK:
        v = _SYMBOL_CACHE.get(key)
        return dict(v) if v else None


def _cache_put(symbol: str, venue: str, price: Any, change: Any, volume: Any) -> None:
    key = (symbol.upper(), venue)
    with _CACHE_LOCK:
        _SYMBOL_CACHE[key] = {
            "price": price,
            "change": change,
            "volume": volume,
            "ts": time.time(),
        }


def default_split_buckets(
    req: List[str], market_type: str
) -> Tuple[List[str], List[str]]:
    from symbol_classifier import is_crypto_symbol, is_stock_symbol

    crypto_syms: List[str] = []
    stock_syms: List[str] = []
    for s in req:
        if market_type == "crypto":
            crypto_syms.append(s)
        elif market_type == "stocks":
            stock_syms.append(s)
        else:
            if is_stock_symbol(s):
                stock_syms.append(s)
            elif is_crypto_symbol(s) or "/" in s:
                crypto_syms.append(s)
            else:
                if len(s) < 6 and "/" not in s:
                    stock_syms.append(s)
                else:
                    crypto_syms.append(s)
    return crypto_syms, stock_syms


split_buckets_fn: Callable[[List[str], str], Tuple[List[str], List[str]]] = default_split_buckets


def default_normalize(s: str) -> str:
    import worker_api as w

    return w._normalize_symbol(s)


def default_resolve(s: str) -> str:
    import worker_api as w

    return w._resolve_symbol(s)


normalize_fn: Callable[[str], str] = default_normalize
resolve_fn: Callable[[str], str] = default_resolve


def default_kraken_batch() -> Dict[str, Dict[str, Any]]:
    global UPSTREAM_CRYPTO_BATCH_CALLS
    import worker_api as w

    UPSTREAM_CRYPTO_BATCH_CALLS += 1
    return w._tickers_batch_cached(ttl_sec=15)


def default_stock_snapshots(batch: List[str]) -> Dict[str, Any]:
    global UPSTREAM_STOCKS_BATCH_CALLS
    import worker_api as w

    UPSTREAM_STOCKS_BATCH_CALLS += 1
    client = w.alpaca_live if w.alpaca_live else w.alpaca_paper
    if not client or not batch:
        return {"snapshots": {}}
    return client.get_snapshots(batch[:100]) or {"snapshots": {}}


kraken_batch_fn: Callable[[], Dict[str, Dict[str, Any]]] = default_kraken_batch
stocks_snapshots_fn: Callable[[List[str]], Dict[str, Any]] = default_stock_snapshots


def apply_kraken_to_bucket(
    crypto_symbols: List[str],
    batch_map: Dict[str, Dict[str, Any]],
    out: Dict[str, Optional[float]],
    changes: Dict[str, Optional[float]],
    volumes: Dict[str, Optional[float]],
) -> None:
    mk = None
    try:
        import worker_api as w

        if w._kraken_ready():
            mk = w._markets()
    except Exception:
        pass
    for s in crypto_symbols:
        norm = normalize_fn(s)
        resolved = resolve_fn(norm)
        ck = (norm.upper(), "crypto")
        cached = _cache_get(ck)
        if cached is not None:
            out[norm] = cached.get("price")
            changes[norm] = cached.get("change")
            volumes[norm] = cached.get("volume")
            continue
        ticker = None
        if batch_map:
            ticker = batch_map.get(resolved) or batch_map.get(norm) or batch_map.get(s)
        if not ticker and mk and resolved in mk:
            try:
                import worker_api as w

                ticker = w._ticker_cached(resolved, ttl_sec=15) or {}
            except Exception:
                ticker = {}
        if ticker:
            price = float(ticker.get("last") or 0.0) if ticker.get("last") else None
            out[norm] = price if price and price > 0 else None
            pct = ticker.get("percentage")
            changes[norm] = float(pct) if pct is not None else None
            qv = ticker.get("quoteVolume")
            volumes[norm] = float(qv) if qv is not None else None
            _cache_put(norm, "crypto", out[norm], changes[norm], volumes[norm])
        else:
            out[norm] = None


def apply_stocks_snapshots_to_bucket(
    stock_symbols: List[str],
    snap_payload: Dict[str, Any],
    out: Dict[str, Optional[float]],
    changes: Dict[str, Optional[float]],
    volumes: Dict[str, Optional[float]],
) -> None:
    snapshots = snap_payload.get("snapshots", {}) if isinstance(snap_payload, dict) else {}
    for sym in stock_symbols:
        ck = (sym.upper(), "stocks")
        cached = _cache_get(ck)
        if cached is not None:
            out[sym] = cached.get("price")
            changes[sym] = cached.get("change")
            volumes[sym] = cached.get("volume")
            continue
        snap = snapshots.get(sym) or snapshots.get(sym.upper()) or {}
        price = None
        if snap:
            latest_trade = snap.get("latestTrade", {})
            daily_bar = snap.get("dailyBar", {}) or {}
            prev_bar = snap.get("prevDailyBar", {}) or {}
            if latest_trade and latest_trade.get("p"):
                price = float(latest_trade.get("p", 0))
            elif daily_bar and daily_bar.get("c"):
                price = float(daily_bar.get("c", 0))
            out[sym] = price if price and price > 0 else None
            if price and price > 0:
                prev_c = prev_bar.get("c") if prev_bar else None
                if prev_c is not None:
                    prev_close = float(prev_c)
                    if prev_close > 0:
                        changes[sym] = ((price - prev_close) / prev_close) * 100.0
                elif daily_bar and daily_bar.get("o") is not None:
                    o = float(daily_bar.get("o", 0))
                    if o > 0:
                        changes[sym] = ((price - o) / o) * 100.0
            if daily_bar and daily_bar.get("v") is not None:
                volumes[sym] = float(daily_bar.get("v", 0))
        else:
            out[sym] = None
        _cache_put(sym, "stocks", out.get(sym), changes.get(sym), volumes.get(sym))


def _worker_crypto(crypto_symbols: List[str], target: Dict[str, Dict[str, Any]]) -> None:
    batch: Dict[str, Dict[str, Any]] = {}
    if crypto_symbols:
        needs_fetch = False
        for s in crypto_symbols:
            norm = normalize_fn(s)
            if _cache_get((norm.upper(), "crypto")) is None:
                needs_fetch = True
                break
        if needs_fetch:
            batch = kraken_batch_fn() or {}
    out: Dict[str, Optional[float]] = {}
    ch: Dict[str, Optional[float]] = {}
    vol: Dict[str, Optional[float]] = {}
    apply_kraken_to_bucket(crypto_symbols, batch, out, ch, vol)
    target["prices"] = out
    target["changes"] = ch
    target["volumes"] = vol


def _worker_stocks(stock_symbols: List[str], target: Dict[str, Dict[str, Any]]) -> None:
    payload: Dict[str, Any] = {"snapshots": {}}
    if stock_symbols:
        missing_chunks = []
        for sym in stock_symbols:
            if _cache_get((sym.upper(), "stocks")) is None:
                missing_chunks.append(sym)
        if missing_chunks:
            for i in range(0, len(missing_chunks), 100):
                chunk = missing_chunks[i : i + 100]
                part = stocks_snapshots_fn(chunk)
                if isinstance(part, dict) and isinstance(part.get("snapshots"), dict):
                    payload["snapshots"].update(part["snapshots"])
    out: Dict[str, Optional[float]] = {}
    ch: Dict[str, Optional[float]] = {}
    vol: Dict[str, Optional[float]] = {}
    apply_stocks_snapshots_to_bucket(stock_symbols, payload, out, ch, vol)
    target["prices"] = out
    target["changes"] = ch
    target["volumes"] = vol


async def fetch_prices_async(
    symbols: str,
    market_type: str,
    timeout_sec: float = 3.0,
) -> Dict[str, Any]:
    req = [s.strip() for s in (symbols or "").split(",") if s.strip()]
    out: Dict[str, Optional[float]] = {}
    changes: Dict[str, Optional[float]] = {}
    volumes: Dict[str, Optional[float]] = {}
    if not req:
        return {
            "ok": True,
            "prices": out,
            "changes": changes,
            "volumes": volumes,
            "partial": False,
        }

    crypto_syms, stock_syms = split_buckets_fn(req, market_type or "all")
    errors: Dict[str, str] = {}
    partial = False
    loop = asyncio.get_event_loop()

    async def guarded(name: str, fn: Callable[[], None]) -> None:
        try:
            await asyncio.wait_for(loop.run_in_executor(None, fn), timeout=timeout_sec)
        except asyncio.TimeoutError as e:
            errors[name] = "timeout"
            raise e
        except Exception as e:
            errors[name] = f"{type(e).__name__}: {e}"
            raise

    tasks = []
    crypto_box: Dict[str, Any] = {}
    stock_box: Dict[str, Any] = {}

    if crypto_syms:
        tasks.append(
            (
                "crypto",
                guarded("crypto", lambda: _worker_crypto(crypto_syms, crypto_box)),
            )
        )
    if stock_syms:
        tasks.append(
            (
                "stocks",
                guarded("stocks", lambda: _worker_stocks(stock_syms, stock_box)),
            )
        )

    results = await asyncio.gather(
        *[t[1] for t in tasks],
        return_exceptions=True,
    )

    for i, (name, _) in enumerate(tasks):
        r = results[i]
        if isinstance(r, Exception):
            partial = True
            if name not in errors:
                errors[name] = (
                    "timeout"
                    if isinstance(r, asyncio.TimeoutError)
                    else f"{type(r).__name__}: {r}"
                )

    # Merge successful buckets
    ti = 0
    for name, _ in tasks:
        r = results[ti]
        ti += 1
        if isinstance(r, Exception):
            continue
        if name == "crypto":
            out.update(crypto_box.get("prices") or {})
            changes.update(crypto_box.get("changes") or {})
            volumes.update(crypto_box.get("volumes") or {})
        else:
            out.update(stock_box.get("prices") or {})
            changes.update(stock_box.get("changes") or {})
            volumes.update(stock_box.get("volumes") or {})

    payload: Dict[str, Any] = {
        "ok": True,
        "prices": out,
        "changes": changes,
        "volumes": volumes,
        "partial": partial,
    }
    if errors:
        payload["errors"] = errors
    return payload
