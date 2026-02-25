"""
MarketDataRouter — Single entry point for all OHLCV and price fetches.
Routes by symbol type (stock/crypto) to Alpaca or Kraken.
Ensures consistent candle schema, symbol validation, retry, caching, and stale detection.
"""
import os
import time
import logging
from typing import Any, Dict, List, Optional, Tuple

from symbol_classifier import classify_symbol, is_stock_symbol, is_crypto_symbol

logger = logging.getLogger(__name__)

# OHLCV schema: [ts_ms, open, high, low, close, volume]
OHLCV_TS = 0
OHLCV_O, OHLCV_H, OHLCV_L, OHLCV_C, OHLCV_V = 1, 2, 3, 4, 5

# Timeframe mapping: our keys -> provider-specific
ALPACA_TF_MAP = {
    "1m": "1Min", "5m": "5Min", "15m": "15Min",
    "1h": "1Hour", "4h": "4Hour", "1d": "1Day", "1w": "1Week",
}
# Alpaca stocks may not support 4Hour directly; we aggregate 1h
ALPACA_STOCKS_SUPPORTED_TF = {"1Min", "5Min", "15Min", "1Hour", "1Day"}

# Stale threshold: last candle must be within N * timeframe_seconds of now
STALE_MULTIPLIER = 3  # e.g. 1h tf -> last candle within 3h
CACHE_TTL_SEC = int(os.getenv("MARKET_DATA_CACHE_TTL", "60"))
MAX_RETRIES = int(os.getenv("MARKET_DATA_MAX_RETRIES", "3"))
RETRY_BASE_SLEEP = float(os.getenv("MARKET_DATA_RETRY_SLEEP", "0.5"))

# Last fetch error per symbol (so intelligence layer can show real cause)
_DATA_FETCH_ERRORS: Dict[str, str] = {}
_DATA_FETCH_ERRORS_LOCK = __import__("threading").Lock()


def get_last_data_error(symbol: str) -> Optional[str]:
    """Return last data fetch error for symbol, if any. Used by intelligence layer."""
    with _DATA_FETCH_ERRORS_LOCK:
        return _DATA_FETCH_ERRORS.get((symbol or "").upper())


def _normalize_crypto_symbol(symbol: str) -> str:
    """Normalize crypto to Kraken format. BTC -> XBT for Kraken."""
    s = (symbol or "").strip().upper()
    if not s:
        return ""
    if "/" in s:
        base, quote = s.split("/", 1)
        base = base.strip()
        quote = (quote or "USD").strip()
        if base == "BTC":
            base = "XBT"  # Kraken uses XBT
        return f"{base}/{quote}"
    if len(s) >= 6 and "USD" in s:
        base = s.replace("USD", "").strip()
        if base == "BTC":
            base = "XBT"
        return f"{base}/USD"
    return f"{s}/USD" if s else ""


def _normalize_stock_symbol(symbol: str) -> str:
    """Normalize stock to upper ticker, no slash."""
    s = (symbol or "").strip().upper()
    if "/" in s:
        s = s.split("/", 1)[0].strip()
    return s


class MarketDataRouter:
    """
    Single router for market data. Routes by symbol type to Kraken (crypto) or Alpaca (stocks).
    Provides: get_candles, get_last_price, validate_symbol.
    """

    def __init__(
        self,
        kraken_client: Optional[Any] = None,
        alpaca_paper: Optional[Any] = None,
        alpaca_live: Optional[Any] = None,
    ):
        self._kc = kraken_client
        self._alpaca_paper = alpaca_paper
        self._alpaca_live = alpaca_live
        self._cache: Dict[str, Dict[str, Any]] = {}
        self._cache_lock = __import__("threading").Lock()
        self._alpaca_adapter_cls = None  # Lazy import to avoid circular deps

    def _get_alpaca_adapter(self, use_live: bool = False):
        if self._alpaca_adapter_cls is None:
            from alpaca_adapter import AlpacaAdapter
            self._alpaca_adapter_cls = AlpacaAdapter
        client = self._alpaca_live if use_live else (self._alpaca_paper or self._alpaca_live)
        if not client:
            return None
        return self._alpaca_adapter_cls(client)

    def validate_symbol(
        self,
        symbol: str,
        market_type: Optional[str] = None,
    ) -> Tuple[bool, Optional[str], str, str]:
        """
        Validate symbol and return (ok, reason, normalized_symbol, provider).
        provider is "kraken" or "alpaca".
        """
        if not symbol or not str(symbol).strip():
            return False, "Empty symbol", "", ""

        detected = classify_symbol(symbol)
        use_type = (market_type or "").strip().lower() or detected
        if use_type in ("stock", "stocks"):
            use_type = "stock"
        elif use_type in ("crypto", "cryptocurrency"):
            use_type = "crypto"
        else:
            use_type = detected

        if use_type == "stock":
            norm = _normalize_stock_symbol(symbol)
            if not norm:
                return False, "Invalid stock symbol format", "", ""
            if not self._alpaca_paper and not self._alpaca_live:
                return False, "Alpaca client not configured", norm, "alpaca"
            try:
                adapter = self._get_alpaca_adapter(use_live=False)
                if not adapter:
                    return False, "Alpaca client not available", norm, "alpaca"
                asset = adapter.client.get_asset(norm) or {}
                # Allow when asset empty (API failed) or tradable - we use Yahoo for data; orders fail at execution if unsupported
                if asset and not asset.get("tradable"):
                    skip = os.getenv("SKIP_ALPACA_TRADABLE_CHECK", "1").strip().lower() in ("1", "true", "yes")
                    if not skip:
                        return False, "Invalid/unsupported stock symbol or not tradable on Alpaca", norm, "alpaca"
                status = str(asset.get("status") or "active").lower()
                skip_status = os.getenv("SKIP_ALPACA_STATUS_CHECK", "0").strip().lower() in ("1", "true", "yes")
                if not skip_status and status == "inactive":
                    return False, "Stock symbol not active on Alpaca", norm, "alpaca"
                return True, None, norm, "alpaca"
            except Exception as e:
                err = str(e).lower()
                # Allow on 404/not found - we use Yahoo for data; symbol may not be in Alpaca's list
                if "404" in err or "not found" in err:
                    return True, None, norm, "alpaca"
                return False, f"Alpaca validation failed: {e}", norm, "alpaca"

        else:
            norm = _normalize_crypto_symbol(symbol)
            if not norm:
                return False, "Invalid crypto symbol format", "", ""
            if not self._kc:
                return False, "Kraken client not configured", norm, "kraken"
            try:
                self._kc.load_markets()
                if norm not in (self._kc.ex.markets or {}):
                    return False, "Invalid Kraken market", norm, "kraken"
                return True, None, norm, "kraken"
            except Exception as e:
                return False, f"Kraken validation failed: {e}", norm, "kraken"

    def _normalize_ohlcv(self, raw: List, provider: str) -> List[List[float]]:
        """Ensure [ts_ms, o, h, l, c, v] format."""
        out = []
        for row in (raw or []):
            if not row or len(row) < 5:
                continue
            ts = int(row[0])
            if ts < 1e12:  # seconds
                ts *= 1000
            o = float(row[1]) if len(row) > 1 else 0.0
            h = float(row[2]) if len(row) > 2 else o
            l = float(row[3]) if len(row) > 3 else o
            c = float(row[4]) if len(row) > 4 else o
            v = float(row[5]) if len(row) > 5 else 0.0
            out.append([ts, o, h, l, c, v])
        return out

    def _fetch_candles_kraken(
        self,
        symbol: str,
        timeframe: str,
        limit: int,
    ) -> Tuple[List[List[float]], Optional[str]]:
        """Fetch from Kraken. Returns (candles, error)."""
        if not self._kc:
            return [], "Kraken client not configured"
        ex = getattr(self._kc, "ex", self._kc)
        for attempt in range(MAX_RETRIES):
            try:
                raw = ex.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)
                return self._normalize_ohlcv(raw, "kraken"), None
            except Exception as e:
                if attempt < MAX_RETRIES - 1:
                    time.sleep(RETRY_BASE_SLEEP * (2 ** attempt))
                return [], str(e)
        return [], "Kraken fetch failed after retries"

    def _fetch_candles_alpaca(
        self,
        symbol: str,
        timeframe: str,
        limit: int,
    ) -> Tuple[List[List[float]], Optional[str]]:
        """Fetch from Alpaca with buffer, retries, 4h/1d aggregation fallback, Yahoo fallback."""
        global _DATA_FETCH_ERRORS
        adapter = self._get_alpaca_adapter(use_live=False)
        if not adapter:
            return [], "Alpaca client not configured"
        tf_lower = (timeframe or "1h").lower()
        is_stock = "/" not in symbol
        min_acceptable = 20
        req_limit = max(limit + 30, min_acceptable + 10)

        # One-time log of data feed
        feed = getattr(adapter.client, "data_feed", "") or ""
        if not hasattr(self, "_alpaca_feed_logged"):
            logger.info("Alpaca data feed=%s", feed or "(default)")
            self._alpaca_feed_logged = True  # type: ignore

        for attempt in range(MAX_RETRIES):
            try:
                raw = adapter.client.get_ohlcv(symbol, tf_lower, req_limit)
                count = len(raw) if raw else 0
                if count == 0:
                    logger.warning(
                        "_fetch_candles_alpaca empty: symbol=%r tf=%s limit=%d requested=%d attempt=%d",
                        symbol, tf_lower, limit, req_limit, attempt + 1,
                    )
                else:
                    logger.info(
                        "_fetch_candles_alpaca %s %s attempt=%d: received=%d requested=%d",
                        symbol, tf_lower, attempt + 1, count, req_limit,
                    )
                # If Alpaca returns 0/small bars for stocks, validate symbol
                if is_stock and (not raw or len(raw) < min_acceptable) and attempt == 0:
                    try:
                        asset = adapter.client.get_asset(symbol)
                        tradable = asset.get("tradable", False)
                        status = (asset.get("status") or "").strip().lower()
                        if not tradable or status != "active":
                            err = f"Alpaca asset invalid/untradable: {symbol} (tradable={tradable}, status={status})"
                            logger.warning(err)
                            with _DATA_FETCH_ERRORS_LOCK:
                                _DATA_FETCH_ERRORS[(symbol or "").upper()] = err
                            return [], err
                    except ValueError as ve:
                        if "404" in str(ve):
                            err = f"Alpaca asset not found (404): {symbol}"
                            logger.warning(err)
                            with _DATA_FETCH_ERRORS_LOCK:
                                _DATA_FETCH_ERRORS[(symbol or "").upper()] = err
                            return [], err
                    except Exception as asset_e:
                        logger.debug("get_asset %s: %s", symbol, asset_e)
                if raw and len(raw) >= min_acceptable:
                    with _DATA_FETCH_ERRORS_LOCK:
                        _DATA_FETCH_ERRORS.pop((symbol or "").upper(), None)
                    out = self._normalize_ohlcv(raw[-limit:], "alpaca")
                    return out, None
                if raw:
                    normalized = self._normalize_ohlcv(raw, "alpaca")
                    if len(normalized) >= min_acceptable:
                        return normalized[-limit:], None
                if tf_lower in ("4h", "4hour") and is_stock:
                    one_h = adapter.client.get_ohlcv(symbol, "1h", limit=min(limit * 6, 500))
                    if one_h and len(one_h) >= 50:
                        agg = getattr(adapter.client, "_aggregate_ohlcv", None)
                        if agg:
                            agg_rows = agg(one_h, 4)
                            if agg_rows and len(agg_rows) >= min_acceptable:
                                return self._normalize_ohlcv(agg_rows[-limit:], "alpaca"), None
                if tf_lower in ("1d", "1day") and is_stock:
                    one_h = adapter.client.get_ohlcv(symbol, "1h", limit=min(limit * 24, 500))
                    if one_h and len(one_h) >= 24:
                        agg = getattr(adapter.client, "_aggregate_ohlcv", None)
                        if agg:
                            agg_rows = agg(one_h, 24)
                            if agg_rows and len(agg_rows) >= min_acceptable:
                                return self._normalize_ohlcv(agg_rows[-limit:], "alpaca"), None
                if is_stock and (not raw or len(raw) < min_acceptable):
                    yahoo = self._fetch_candles_yahoo_fallback(symbol, tf_lower, max(limit, 100))
                    if yahoo and len(yahoo) >= min_acceptable:
                        return yahoo[-limit:], None
                if raw:
                    return self._normalize_ohlcv(raw[-limit:], "alpaca"), None
            except Exception as e:
                logger.warning("_fetch_candles_alpaca %s %s attempt %d: %s", symbol, tf_lower, attempt + 1, e)
                if attempt < MAX_RETRIES - 1:
                    time.sleep(RETRY_BASE_SLEEP * (2 ** attempt))
                elif is_stock:
                    yahoo = self._fetch_candles_yahoo_fallback(symbol, tf_lower, max(limit, 100))
                    if yahoo and len(yahoo) >= min_acceptable:
                        return yahoo[-limit:], None
                err_str = str(e)
                logger.warning("_fetch_candles_alpaca failed: symbol=%r tf=%s limit=%d err=%s", symbol, tf_lower, limit, e)
                with _DATA_FETCH_ERRORS_LOCK:
                    _DATA_FETCH_ERRORS[(symbol or "").upper()] = err_str
                return [], err_str
        err_final = "Alpaca fetch failed after retries"
        logger.warning("_fetch_candles_alpaca returning empty: symbol=%r tf=%s limit=%d reason=%s", symbol, tf_lower, limit, err_final)
        with _DATA_FETCH_ERRORS_LOCK:
            _DATA_FETCH_ERRORS[(symbol or "").upper()] = err_final
        return [], err_final

    def _expand_daily_to_intraday(self, daily: List[List[float]], target_tf: str, limit: int = 500) -> List[List[float]]:
        """Expand daily bars into synthetic 1h or 4h bars for symbols with no intraday data."""
        if not daily or len(daily) < 20:
            return []
        tf_lower = (target_tf or "1h").lower()
        bars_per_day = 24 if tf_lower in ("1h", "1hour") else 6 if tf_lower in ("4h", "4hour") else 1
        if bars_per_day == 1:
            return daily[-limit:] if len(daily) > limit else daily
        out = []
        for row in daily:
            if len(row) < 6:
                continue
            ts_ms, o, h, l, c, v = row[0], row[1], row[2], row[3], row[4], row[5]
            v_per_bar = v / bars_per_day if v else 0
            for i in range(bars_per_day):
                offset_ms = int(i * (86400000 / bars_per_day))
                out.append([ts_ms - 86400000 + offset_ms, o, h, l, c, v_per_bar])
        return out[-limit:] if len(out) > limit else out

    def _fetch_candles_yahoo_fallback(self, symbol: str, timeframe: str, limit: int) -> Optional[List[List[float]]]:
        """Yahoo Finance fallback for stocks when Alpaca returns no/insufficient data."""
        import time as _time
        try:
            from phase2_data_fetcher import _fetch_candles_yahoo
            for try_limit in [max(limit, 50), 100, 80]:
                yahoo = _fetch_candles_yahoo(symbol, timeframe, try_limit)
                if yahoo and len(yahoo) >= 20:
                    logger.info("Yahoo fallback: %s %s -> %d candles", symbol, timeframe, len(yahoo))
                    return self._normalize_ohlcv(yahoo, "alpaca")
                if yahoo and len(yahoo) > 0:
                    _time.sleep(0.3)
            for _ in range(2):
                yahoo = _fetch_candles_yahoo(symbol, timeframe, 100)
                if yahoo and len(yahoo) >= 20:
                    return self._normalize_ohlcv(yahoo, "alpaca")
                _time.sleep(0.5)
        except Exception as e:
            logger.debug("Yahoo fallback %s %s: %s", symbol, timeframe, e)
        return None

    def get_candles(
        self,
        symbol: str,
        timeframe: str,
        limit: int = 100,
        market_type: Optional[str] = None,
        use_cache: bool = True,
    ) -> List[List[float]]:
        """
        Fetch OHLCV candles. Routes to Kraken (crypto) or Alpaca (stocks).
        Returns list of [ts_ms, o, h, l, c, v].
        """
        ok, reason, norm_sym, provider = self.validate_symbol(symbol, market_type)
        if not ok:
            logger.warning("get_candles validate failed: symbol=%s provider=%s reason=%s", symbol, provider, reason)
            return []

        cache_key = f"{norm_sym}|{timeframe}|{limit}|{provider}"
        now = time.time()
        if use_cache:
            with self._cache_lock:
                entry = self._cache.get(cache_key)
                if entry and (now - entry.get("ts", 0)) <= CACHE_TTL_SEC:
                    return entry.get("data") or []

        if provider == "kraken":
            candles, err = self._fetch_candles_kraken(norm_sym, timeframe, limit)
        else:
            candles, err = [], None
            # Stocks: Yahoo FIRST (Alpaca free tier often lacks data for UFPI, etc.)
            if "/" not in norm_sym:
                yahoo = self._fetch_candles_yahoo_fallback(norm_sym, (timeframe or "1h").lower(), max(limit, 100))
                if yahoo and len(yahoo) >= 20:
                    candles = yahoo[-limit:] if limit < len(yahoo) else yahoo
                    err = None
                    logger.info("get_candles: Yahoo %s %s -> %d candles", norm_sym, timeframe, len(candles))
            if not candles or len(candles) < 20:
                candles, err = self._fetch_candles_alpaca(norm_sym, timeframe, limit)
                if (not candles or len(candles) < 20) and "/" not in norm_sym:
                    yahoo = self._fetch_candles_yahoo_fallback(norm_sym, (timeframe or "1h").lower(), max(limit, 100))
                    if yahoo and len(yahoo) >= 20:
                        candles = yahoo[-limit:] if limit < len(yahoo) else yahoo
                        err = None
            # Final fallback for stocks: expand 1d into synthetic 1h/4h when intraday fails (UFPI, etc.)
            if (not candles or len(candles) < 20) and "/" not in norm_sym:
                tf_lower = (timeframe or "1h").lower()
                if tf_lower in ("1h", "1hour", "4h", "4hour"):
                    daily, _ = self._fetch_candles_alpaca(norm_sym, "1d", max(limit, 100))
                    if not daily or len(daily) < 20:
                        daily_yahoo = self._fetch_candles_yahoo_fallback(norm_sym, "1d", max(limit, 100))
                        if daily_yahoo:
                            daily = daily_yahoo
                    if daily and len(daily) >= 20:
                        candles = self._expand_daily_to_intraday(daily, tf_lower, limit)
                        if candles and len(candles) >= 20:
                            err = None
                            logger.info("get_candles: %s %s synthetic from 1d (%d bars)", norm_sym, tf_lower, len(candles))

        if err:
            logger.warning(
                "get_candles fetch failed: symbol=%s tf=%s limit=%s provider=%s err=%s",
                norm_sym, timeframe, limit, provider, err
            )

        if use_cache and candles:
            with self._cache_lock:
                self._cache[cache_key] = {"ts": now, "data": candles}

        return candles or []

    def get_last_price(self, symbol: str, market_type: Optional[str] = None) -> Optional[float]:
        """Fetch last price. Routes by symbol type."""
        ok, _, norm_sym, provider = self.validate_symbol(symbol, market_type)
        if not ok:
            return None
        try:
            if provider == "kraken":
                if not self._kc:
                    return None
                t = self._kc.ex.fetch_ticker(norm_sym)
                return float(t.get("last") or t.get("close") or 0) or None
            else:
                adapter = self._get_alpaca_adapter(use_live=False)
                if not adapter:
                    return None
                return adapter.fetch_ticker_last(norm_sym) or None
        except Exception as e:
            logger.debug("get_last_price failed: symbol=%s provider=%s err=%s", norm_sym, provider, e)
            return None

    def get_data_health(
        self,
        symbol: str,
        market_type: Optional[str] = None,
        required_tfs: Optional[List[str]] = None,
        min_candles: int = 20,
    ) -> Dict[str, Any]:
        """
        Return data health summary for UI: counts, last candle age, provider, symbol.
        """
        required_tfs = required_tfs or ["1h", "4h", "1d"]
        ok, reason, norm_sym, provider = self.validate_symbol(symbol, market_type)
        result = {
            "ok": ok,
            "reason": reason,
            "normalized_symbol": norm_sym,
            "provider": provider,
            "timeframes": {},
            "blocked": not ok,
        }
        if not ok:
            return result

        for tf in required_tfs:
            candles = self.get_candles(symbol, tf, limit=max(min_candles, 50), market_type=market_type, use_cache=True)
            count = len(candles) if candles else 0
            last_ts = candles[-1][OHLCV_TS] / 1000 if candles and len(candles[-1]) > OHLCV_TS else 0
            age_sec = (time.time() - last_ts) if last_ts else None
            tf_sec = {"1m": 60, "5m": 300, "15m": 900, "1h": 3600, "4h": 14400, "1d": 86400, "1w": 604800}.get(tf, 3600)
            stale = age_sec is not None and age_sec > (STALE_MULTIPLIER * tf_sec) if tf_sec else False
            result["timeframes"][tf] = {
                "count": count,
                "last_candle_age_sec": age_sec,
                "stale": stale,
                "sufficient": count >= min_candles,
            }
        return result


# Singleton instance, set by worker_api / bot_manager after clients are ready
_router: Optional[MarketDataRouter] = None


def get_router() -> Optional[MarketDataRouter]:
    return _router


def set_router(router: MarketDataRouter) -> None:
    global _router
    _router = router


def _get_candles_with_fallback(
    symbol: str,
    timeframe: str,
    limit: int = 100,
    market_type: Optional[str] = None,
    min_acceptable: int = 20,
) -> List[List[float]]:
    """
    Get candles with Yahoo fallback for stocks when primary provider returns no/insufficient data.
    Returns [[ts_ms, o, h, l, c, v], ...].
    """
    r = get_router()
    candles = r.get_candles(symbol, timeframe, limit, market_type) if r else []
    if candles and len(candles) >= min_acceptable:
        return candles
    is_stock = "/" not in (symbol or "")
    if is_stock and len(candles or []) < min_acceptable:
        try:
            from phase2_data_fetcher import _fetch_candles_yahoo
            yahoo = _fetch_candles_yahoo(symbol, timeframe, limit)
            if yahoo and len(yahoo) >= min_acceptable:
                logger.info("_get_candles_with_fallback Yahoo: %s %s → %s candles", symbol, timeframe, len(yahoo))
                # Normalize to [ts_ms, o, h, l, c, v]
                out = []
                for row in (yahoo or []):
                    if not row or len(row) < 5:
                        continue
                    ts = int(row[0])
                    if ts < 1e12:
                        ts *= 1000
                    out.append([
                        ts,
                        float(row[1]) if len(row) > 1 else 0.0,
                        float(row[2]) if len(row) > 2 else 0.0,
                        float(row[3]) if len(row) > 3 else 0.0,
                        float(row[4]) if len(row) > 4 else 0.0,
                        float(row[5]) if len(row) > 5 else 0.0,
                    ])
                return out
        except Exception as e:
            logger.debug("_get_candles_with_fallback Yahoo %s %s: %s", symbol, timeframe, e)
    return candles or []


def get_candles(symbol: str, timeframe: str, limit: int = 100, market_type: Optional[str] = None) -> List[List[float]]:
    r = get_router()
    if not r:
        return []
    return r.get_candles(symbol, timeframe, limit, market_type)


def get_last_price(symbol: str, market_type: Optional[str] = None) -> Optional[float]:
    r = get_router()
    if not r:
        return None
    return r.get_last_price(symbol, market_type)


def validate_symbol(symbol: str, market_type: Optional[str] = None) -> Tuple[bool, Optional[str], str, str]:
    r = get_router()
    if not r:
        return False, "MarketDataRouter not initialized", "", ""
    return r.validate_symbol(symbol, market_type)
