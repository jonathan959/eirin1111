# kraken_client.py
import os
import time
import threading
import zlib
from typing import Any, Dict, List, Optional, Tuple

import ccxt

from symbol_classifier import is_stock_symbol


def _userref_from_client_order_id(cid: str) -> int:
    """Kraken userref is 32-bit int. Derive from idempotency key."""
    return abs(zlib.adler32((cid or "").encode("utf-8"))) & 0x7FFFFFFF


class KrakenClient:
    """
    Stability-focused Kraken wrapper for ccxt.

    Core safeguards:
    - One lock for ALL private calls (prevents InvalidNonce / race conditions).
    - Cached markets with TTL.
    - Retry/backoff on transient exchange/network errors.
    - Market buy by quote uses ASK price + slippage buffer to avoid overspending.
    - Precision + min amount/cost enforcement based on ccxt market metadata.
    """

    def __init__(self):
        api_key = os.getenv("KRAKEN_API_KEY", "").strip()
        api_secret = os.getenv("KRAKEN_API_SECRET", "").strip()
        if not api_key or not api_secret:
            raise RuntimeError("Missing KRAKEN_API_KEY / KRAKEN_API_SECRET environment variables.")

        self.ex = ccxt.kraken(
            {
                "apiKey": api_key,
                "secret": api_secret,
                "enableRateLimit": True,
                "timeout": 30000,
            }
        )

        # Some exchanges need a price for market buys; Kraken behavior varies by ccxt version.
        # We do not rely on cost-based market buy; we convert quote->base ourselves.
        self.ex.options = dict(getattr(self.ex, "options", {}) or {})
        self.ex.options["createMarketBuyOrderRequiresPrice"] = True

        # Single lock for every private endpoint call.
        self._private_lock = threading.Lock()

        # Markets cache
        self._markets: Optional[Dict[str, Any]] = None
        self._markets_ts: float = 0.0
        self._markets_ttl_sec: int = 300  # 5 minutes

        # Slippage buffer used for quote->base conversion for market buys.
        # Example: 30 bps = 0.30% buffer (spend slightly less base to avoid overspending).
        self._market_buy_slippage_bps: float = float(os.getenv("MARKET_BUY_SLIPPAGE_BPS", "30"))

    # -----------------------
    # Internal: retry helpers
    # -----------------------
    def _is_transient(self, e: Exception) -> bool:
        transient_types = (
            ccxt.NetworkError,
            ccxt.ExchangeNotAvailable,
            ccxt.DDoSProtection,
            ccxt.RequestTimeout,
        )
        # Sometimes Kraken returns these as generic ExchangeError with a transient message.
        msg = str(e).lower()
        if isinstance(e, transient_types):
            return True
        if isinstance(e, ccxt.ExchangeError) and any(
            k in msg for k in ("rate limit", "eagain", "temporarily unavailable", "service unavailable", "timeout")
        ):
            return True
        return False

    def _retry(self, fn, *args, **kwargs):
        attempts = int(kwargs.pop("_attempts", 5))
        base_sleep = float(kwargs.pop("_base_sleep", 1.0))
        max_sleep = float(kwargs.pop("_max_sleep", 30.0))

        last_err = None
        for i in range(attempts):
            try:
                return fn(*args, **kwargs)
            except Exception as e:
                last_err = e
                if not self._is_transient(e):
                    raise
                is_ddos = isinstance(e, ccxt.DDoSProtection) or "rate limit" in str(e).lower()
                sleep_s = min(max_sleep, base_sleep * (2 ** i))
                if is_ddos:
                    sleep_s = min(max_sleep, sleep_s * 2)
                time.sleep(sleep_s)
        raise last_err

    # -----------------------
    # Markets / symbols
    # -----------------------
    def load_markets(self, force: bool = False) -> Dict[str, Any]:
        now = time.time()
        if force or self._markets is None or (now - self._markets_ts) > self._markets_ttl_sec:
            self._markets = self._retry(self.ex.load_markets)
            self._markets_ts = now
        return self._markets or {}

    def ensure_symbol(self, symbol: str) -> None:
        mk = self.load_markets()
        if symbol not in mk:
            raise ValueError(f"Symbol not found/active on Kraken markets: {symbol}")

    def list_spot_symbols(self, quote: str = "USD") -> List[str]:
        q = (quote or "USD").upper().strip()
        mk = self.load_markets()
        out = []
        for sym, m in mk.items():
            if m.get("spot") and m.get("active") and m.get("quote") == q:
                out.append(sym)
        return sorted(out)

    # -----------------------
    # Public endpoints
    # -----------------------
    def fetch_ticker(self, symbol: str) -> Dict[str, Any]:
        self.ensure_symbol(symbol)
        t = self._retry(self.ex.fetch_ticker, symbol)
        if not isinstance(t, dict):
             return {"symbol": symbol, "last": 0.0, "bid": 0.0, "ask": 0.0}
        return t

    def fetch_ticker_last(self, symbol: str) -> float:
        t = self.fetch_ticker(symbol)
        return float(t.get("last") or 0.0)

    def fetch_ticker_ask(self, symbol: str) -> float:
        t = self.fetch_ticker(symbol)
        ask = t.get("ask")
        if ask is None:
            # Fallback to last if ask not provided
            ask = t.get("last")
        return float(ask or 0.0)

    def fetch_ohlcv(self, symbol: str, timeframe: str = "5m", limit: int = 200):
        self.ensure_symbol(symbol)
        return self._retry(self.ex.fetch_ohlcv, symbol, timeframe=timeframe, limit=limit)

    def fetch_ohlcv_range(self, symbol: str, timeframe: str, since_ms: int, until_ms: int, limit: int = 720) -> List[List[float]]:
        """Fetch OHLCV for a date range (ms) with pagination."""
        self.ensure_symbol(symbol)
        out: List[List[float]] = []
        since = int(since_ms)
        tf_ms = int(self.ex.parse_timeframe(timeframe) * 1000)
        while since < until_ms:
            batch = self._retry(self.ex.fetch_ohlcv, symbol, timeframe=timeframe, since=since, limit=limit)
            if not batch:
                break
            out.extend(batch)
            last_ts = int(batch[-1][0])
            if last_ts <= since:
                break
            since = last_ts + tf_ms
        return [c for c in out if int(c[0]) <= int(until_ms)]

    # -----------------------
    # Private endpoints (LOCKED)
    # -----------------------
    def fetch_balance(self) -> Dict[str, Any]:
        with self._private_lock:
            return self._retry(self.ex.fetch_balance)

    def fetch_order_book(self, symbol: str, limit: Optional[int] = None) -> Dict[str, Any]:
        self.ensure_symbol(symbol)
        return self._retry(self.ex.fetch_order_book, symbol, limit or 50)

    def fetch_open_orders(self, symbol: str) -> List[Dict[str, Any]]:
        self.ensure_symbol(symbol)
        with self._private_lock:
            return self._retry(self.ex.fetch_open_orders, symbol)

    def fetch_my_trades(self, symbol: str, limit: int = 200) -> List[Dict[str, Any]]:
        self.ensure_symbol(symbol)
        with self._private_lock:
            return self._retry(self.ex.fetch_my_trades, symbol, limit=limit)

    def cancel_order(self, order_id: str, symbol: str):
        self.ensure_symbol(symbol)
        with self._private_lock:
            return self._retry(self.ex.cancel_order, order_id, symbol)

    def cancel_all_open_orders(self, symbol: str) -> None:
        self.ensure_symbol(symbol)
        with self._private_lock:
            opens = self._retry(self.ex.fetch_open_orders, symbol)
            for o in opens or []:
                oid = o.get("id")
                if not oid:
                    continue
                try:
                    self._retry(self.ex.cancel_order, oid, symbol)
                except Exception:
                    # best-effort cancellation; do not crash
                    pass

    # -----------------------
    # Precision / limits helpers
    # -----------------------
    def _limits_for(self, symbol: str) -> Dict[str, Any]:
        mk = self.load_markets()
        return (mk.get(symbol) or {}).get("limits", {}) or {}

    def _precision_amount(self, symbol: str, amount: float) -> float:
        try:
            return float(self.ex.amount_to_precision(symbol, amount))
        except Exception:
            return float(amount)

    def _precision_price(self, symbol: str, price: float) -> float:
        try:
            return float(self.ex.price_to_precision(symbol, price))
        except Exception:
            return float(price)

    def _enforce_minimums(self, symbol: str, amount: float, price: float) -> Tuple[float, float]:
        """
        Ensure amount/cost meet min limits if provided by ccxt market metadata.
        Returns (amount, cost).
        """
        limits = self._limits_for(symbol)
        amt = float(amount)
        px = float(price)
        if amt <= 0 or px <= 0:
            raise ValueError("amount and price must be > 0")

        cost = amt * px

        min_amt = (((limits.get("amount") or {}) or {}).get("min"))
        if min_amt is not None:
            try:
                min_amt_f = float(min_amt)
                if amt < min_amt_f:
                    amt = min_amt_f
                    cost = amt * px
            except Exception:
                pass

        min_cost = (((limits.get("cost") or {}) or {}).get("min"))
        if min_cost is not None:
            try:
                min_cost_f = float(min_cost)
                if cost < min_cost_f and px > 0:
                    amt = min_cost_f / px
                    cost = amt * px
            except Exception:
                pass

        amt = self._precision_amount(symbol, amt)
        cost = float(amt) * px
        return float(amt), float(cost)

    # -----------------------
    # Trading helpers
    # -----------------------
    def create_market_buy_quote(
        self, symbol: str, quote_amount: float, client_order_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Market BUY using a quote currency amount (e.g., spend 25 USD).
        Optional client_order_id for idempotency (passed as Kraken userref).
        """
        if is_stock_symbol(symbol):
            raise ValueError("Kraken spot cannot place stock orders (analysis-only)")
        self.ensure_symbol(symbol)
        q = float(quote_amount)
        if q <= 0:
            raise ValueError("quote_amount must be > 0")
        ask = float(self.fetch_ticker_ask(symbol))
        if ask <= 0:
            raise RuntimeError("Invalid ask/last price from ticker")
        slip = max(0.0, float(self._market_buy_slippage_bps)) / 10000.0
        effective_px = ask * (1.0 + slip)
        raw_amt = q / effective_px
        amt, _ = self._enforce_minimums(symbol, raw_amt, effective_px)
        params: Dict[str, Any] = {"price": effective_px}
        if client_order_id is not None:
            params["userref"] = _userref_from_client_order_id(client_order_id)
        with self._private_lock:
            return self._retry(self.ex.create_order, symbol, "market", "buy", float(amt), None, params)

    def create_market_sell_base(
        self, symbol: str, base_amount: float, client_order_id: Optional[str] = None
    ) -> Dict[str, Any]:
        if is_stock_symbol(symbol):
            raise ValueError("Kraken spot cannot place stock orders (analysis-only)")
        self.ensure_symbol(symbol)
        amt = float(base_amount)
        if amt <= 0:
            raise ValueError("base_amount must be > 0")
        amt = self._precision_amount(symbol, amt)
        params: Dict[str, Any] = {}
        if client_order_id is not None:
            params["userref"] = _userref_from_client_order_id(client_order_id)
        with self._private_lock:
            return self._retry(
                self.ex.create_order, symbol, "market", "sell", float(amt), None, params
            )

    def create_limit_sell_base(
        self, symbol: str, base_amount: float, price: float, client_order_id: Optional[str] = None
    ) -> Dict[str, Any]:
        if is_stock_symbol(symbol):
            raise ValueError("Kraken spot cannot place stock orders (analysis-only)")
        self.ensure_symbol(symbol)
        amt = float(base_amount)
        if amt <= 0:
            raise ValueError("base_amount must be > 0")
        p = float(price)
        if p <= 0:
            raise ValueError("price must be > 0")
        amt = self._precision_amount(symbol, amt)
        p = self._precision_price(symbol, p)
        params: Dict[str, Any] = {}
        if client_order_id is not None:
            params["userref"] = _userref_from_client_order_id(client_order_id)
        with self._private_lock:
            return self._retry(
                self.ex.create_order, symbol, "limit", "sell", float(amt), float(p), params
            )

    def create_iceberg_limit(
        self,
        symbol: str,
        side: str,
        base_amount: float,
        price: float,
        display_amount: Optional[float] = None,
        client_order_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Iceberg (hidden) limit order. Kraken supports this; only display_amount visible in book.
        display_amount must be >= 1/15 of total. If None, uses 1/15.
        """
        if is_stock_symbol(symbol):
            raise ValueError("Kraken spot cannot place stock orders (analysis-only)")
        self.ensure_symbol(symbol)
        amt = float(base_amount)
        p = float(price)
        if amt <= 0 or p <= 0:
            raise ValueError("base_amount and price must be > 0")
        display = display_amount if display_amount is not None and display_amount >= amt / 15 else max(amt / 15, amt * 0.1)
        amt = self._precision_amount(symbol, amt)
        p = self._precision_price(symbol, p)
        display = self._precision_amount(symbol, display)
        params: Dict[str, Any] = {"ordertype": "iceberg", "displayvol": str(display)}
        if client_order_id is not None:
            params["userref"] = _userref_from_client_order_id(client_order_id)
        with self._private_lock:
            return self._retry(
                self.ex.create_order, symbol, "limit", side.lower(), float(amt), float(p), params
            )

    def create_limit_buy_base(
        self, symbol: str, base_amount: float, price: float, client_order_id: Optional[str] = None
    ) -> Dict[str, Any]:
        if is_stock_symbol(symbol):
            raise ValueError("Kraken spot cannot place stock orders (analysis-only)")
        self.ensure_symbol(symbol)
        amt = float(base_amount)
        if amt <= 0:
            raise ValueError("base_amount must be > 0")
        p = float(price)
        if p <= 0:
            raise ValueError("price must be > 0")
        amt, _ = self._enforce_minimums(symbol, amt, p)
        amt = self._precision_amount(symbol, amt)
        p = self._precision_price(symbol, p)
        params: Dict[str, Any] = {}
        if client_order_id is not None:
            params["userref"] = _userref_from_client_order_id(client_order_id)
        with self._private_lock:
            return self._retry(
                self.ex.create_order, symbol, "limit", "buy", float(amt), float(p), params
            )

    def cancel_orders_by_tag(self, symbol: str, tag: str) -> int:
        self.ensure_symbol(symbol)
        if not tag:
            return 0
        cancelled = 0
        with self._private_lock:
            opens = self._retry(self.ex.fetch_open_orders, symbol)
            for o in opens or []:
                oid = o.get("id")
                client_id = (o.get("clientOrderId") or o.get("client_id") or "")
                if tag in str(client_id):
                    try:
                        self._retry(self.ex.cancel_order, oid, symbol)
                        cancelled += 1
                    except Exception:
                        continue
        return cancelled