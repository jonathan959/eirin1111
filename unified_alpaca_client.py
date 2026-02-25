"""
Unified Alpaca Client - WebSocket + Cache + Rate Limiting
Replaces AlpacaClient for rate limit reduction.
Supports ALPACA_API_KEY_PAPER/LIVE env vars for paper/live mode.
Provides AlpacaClient-compatible interface (get_ticker dict, get_account dict, etc).
"""
import os
import time
import logging
from typing import Optional, List, Dict, Any, Tuple, Union
from datetime import datetime, timezone, timedelta

logger = logging.getLogger(__name__)

OHLCV_MIN_BUFFER = int(os.getenv("OHLCV_MIN_BUFFER", "30"))
OHLCV_RETRIES = int(os.getenv("OHLCV_RETRIES", "5"))
OHLCV_BASE_SLEEP = float(os.getenv("OHLCV_BASE_SLEEP", "0.5"))

USE_UNIFIED_ALPACA = os.getenv("USE_UNIFIED_ALPACA", "1").strip().lower() in ("1", "true", "yes", "y", "on")

try:
    from alpaca.trading.client import TradingClient
    from alpaca.data.historical import StockHistoricalDataClient
    from alpaca.trading.requests import MarketOrderRequest, LimitOrderRequest
    from alpaca.trading.enums import OrderSide, TimeInForce
    from alpaca.data.requests import StockBarsRequest, StockLatestTradeRequest
    from alpaca.data.timeframe import TimeFrame, TimeFrameUnit
    ALPACA_PY_AVAILABLE = True
except ImportError as e:
    ALPACA_PY_AVAILABLE = False
    TradingClient = StockHistoricalDataClient = None

from websocket_manager import WebSocketManager
from data_cache import DataCache, CachedAlpacaClient
from enhanced_rate_limiter import SmartRateLimiter


class UnifiedAlpacaClient:
    """
    Unified Alpaca client: WebSocket prices + cached account/positions + rate-limited trading.
    Implements AlpacaClient-compatible interface for drop-in replacement.
    """
    
    def __init__(
        self,
        mode: str = "paper",
        api_key: Optional[str] = None,
        secret_key: Optional[str] = None,
        auto_start_websocket: bool = True
    ):
        """
        Args:
            mode: "paper" or "live"
            api_key: Override (else from ALPACA_API_KEY_PAPER/LIVE)
            secret_key: Override
        """
        self.mode = mode.lower()
        paper = self.mode != "live"
        
        if self.mode == "paper":
            self.api_key = api_key or os.getenv("ALPACA_API_KEY_PAPER", os.getenv("ALPACA_API_KEY", ""))
            self.secret_key = secret_key or os.getenv("ALPACA_API_SECRET_PAPER", os.getenv("ALPACA_SECRET_KEY", ""))
            self.base_url = "https://paper-api.alpaca.markets"
        else:
            self.api_key = api_key or os.getenv("ALPACA_API_KEY_LIVE", os.getenv("ALPACA_API_KEY", ""))
            self.secret_key = secret_key or os.getenv("ALPACA_API_SECRET_LIVE", os.getenv("ALPACA_SECRET_KEY", ""))
            self.base_url = "https://api.alpaca.markets"
        
        self.data_url = "https://data.alpaca.markets"
        self.paper = paper
        
        if not self.api_key or not self.secret_key:
            raise ValueError(f"Alpaca {mode} API keys not found")
        
        if not ALPACA_PY_AVAILABLE:
            raise ImportError("alpaca-py required. Install: pip install alpaca-py")
        
        self.trading_client = TradingClient(api_key=self.api_key, secret_key=self.secret_key, paper=paper)
        self.data_client = StockHistoricalDataClient(api_key=self.api_key, secret_key=self.secret_key)
        self.cache = DataCache()
        self.cached_client = CachedAlpacaClient(self.trading_client, self.cache)
        self.rate_limiter = SmartRateLimiter(base_limit=180)
        self.websocket = WebSocketManager(api_key=self.api_key, secret_key=self.secret_key, paper=paper)
        if auto_start_websocket:
            self.websocket.start()
        logger.info("UnifiedAlpacaClient initialized (mode=%s, websocket=%s)", mode, "started" if auto_start_websocket else "deferred")
    
    # ---------- AlpacaClient-compatible: Price/Ticker ----------
    
    def get_latest_price(self, symbol: str) -> Optional[float]:
        """WebSocket first, then REST fallback."""
        price = self.websocket.get_latest_price(symbol)
        if price is not None:
            return price
        return self._get_latest_price_rest(symbol)
    
    def _get_latest_price_rest(self, symbol: str) -> Optional[float]:
        cache_key = f"price_{symbol}"
        cached = self.cache.get(cache_key)
        if cached is not None:
            return cached
        if not self.rate_limiter.acquire("get_latest_trade", priority=5):
            return None
        try:
            req = StockLatestTradeRequest(symbol_or_symbols=symbol)
            trade = self.data_client.get_stock_latest_trade(req)
            if trade and symbol in trade:
                price = float(trade[symbol].price)
                self.cache.set(cache_key, price, ttl_seconds=30)
                return price
        except Exception as e:
            logger.debug("REST price %s: %s", symbol, e)
        return None
    
    def get_ticker(self, symbol: str) -> Dict[str, Any]:
        """AlpacaClient-compat: returns dict with last, bid, ask, timestamp."""
        price = self.get_latest_price(symbol)
        quote = self.websocket.get_latest_quote(symbol)
        bid = ask = price
        if quote and hasattr(quote, 'bid_price') and hasattr(quote, 'ask_price'):
            bid = float(quote.bid_price)
            ask = float(quote.ask_price)
            if price is None:
                price = (bid + ask) / 2
        if price is None:
            price = bid or ask
        return {
            "symbol": symbol,
            "last": float(price) if price else 0.0,
            "bid": float(bid) if bid else 0.0,
            "ask": float(ask) if ask else 0.0,
            "volume": 0,
            "timestamp": None,
        }
    
    # ---------- WebSocket subscriptions ----------
    
    def subscribe_to_symbols(self, symbols: List[str], callback=None):
        self.websocket.subscribe_trades(symbols, callback=callback)
        self.websocket.subscribe_quotes(symbols)
        self.websocket.subscribe_bars(symbols)
        logger.info("Subscribed to %d symbols: %s", len(symbols), ", ".join(sorted(symbols)[:10]) + ("..." if len(symbols) > 10 else ""))
    
    # ---------- AlpacaClient-compatible: Account (dict) ----------
    
    def get_account(self) -> Dict[str, Any]:
        """Returns dict for AlpacaAdapter compatibility."""
        acc = self.cached_client.get_account()
        return self._account_to_dict(acc)
    
    def _account_to_dict(self, acc) -> Dict[str, Any]:
        if isinstance(acc, dict):
            return acc
        return {
            "cash": str(getattr(acc, "cash", 0)),
            "buying_power": str(getattr(acc, "buying_power", 0)),
            "portfolio_value": str(getattr(acc, "portfolio_value", 0)),
            "equity": str(getattr(acc, "equity", 0)),
        }
    
    def get_balance(self) -> float:
        return float(self.get_account().get("cash", 0))
    
    def get_buying_power(self) -> float:
        return float(self.get_account().get("buying_power", 0))
    
    def get_portfolio_value(self) -> float:
        return float(self.get_account().get("portfolio_value", 0))
    
    # ---------- AlpacaClient-compatible: Clock ----------
    
    def get_clock(self) -> Dict[str, Any]:
        if not self.rate_limiter.acquire("get_clock", priority=3):
            return {"is_open": True}
        try:
            clock = self.trading_client.get_clock()
            return {
                "timestamp": str(getattr(clock, "timestamp", "")),
                "is_open": getattr(clock, "is_open", True),
                "next_open": str(getattr(clock, "next_open", "")),
                "next_close": str(getattr(clock, "next_close", "")),
            }
        except Exception:
            return {"is_open": True}
    
    def is_market_open(self) -> bool:
        return bool(self.get_clock().get("is_open", True))
    
    # ---------- AlpacaClient-compatible: Positions (dicts) ----------
    
    def get_positions(self) -> List[Dict[str, Any]]:
        pos_list = self.cached_client.get_all_positions()
        return [self._position_to_dict(p) for p in pos_list]
    
    def get_position(self, symbol: str) -> Optional[Dict[str, Any]]:
        pos = self.cached_client.get_open_position(symbol)
        return self._position_to_dict(pos) if pos else None
    
    def _position_to_dict(self, pos) -> Dict[str, Any]:
        if isinstance(pos, dict):
            return pos
        return {
            "symbol": getattr(pos, "symbol", ""),
            "qty": str(getattr(pos, "qty", 0)),
            "side": getattr(pos, "side", "long"),
            "market_value": str(getattr(pos, "market_value", 0)),
            "unrealized_pl": str(getattr(pos, "unrealized_pl", 0)),
            "current_price": str(getattr(pos, "current_price", 0)),
        }
    
    # ---------- AlpacaClient-compatible: Asset / Validation ----------

    def get_asset(self, symbol: str) -> Dict[str, Any]:
        """Get asset details for validation. Returns dict with tradable, status."""
        if not self.rate_limiter.acquire("get_asset", priority=3):
            return {}
        try:
            asset = self.trading_client.get_asset(symbol)
            if not asset:
                return {}
            st = getattr(asset, "status", "active")
            status_str = getattr(st, "value", st) if st is not None else "active"
            status_str = str(status_str or "active").lower()
            return {
                "symbol": getattr(asset, "symbol", symbol),
                "tradable": getattr(asset, "tradable", True),
                "status": status_str,
            }
        except Exception as e:
            logger.debug("get_asset %s: %s", symbol, e)
            return {}

    def _check_symbol_tradable(self, symbol: str) -> bool:
        """Verify symbol exists and is tradable on Alpaca."""
        try:
            asset = self.trading_client.get_asset(symbol)
            if not asset:
                return False
            if not getattr(asset, "tradable", True):
                logger.warning("Symbol %s not tradable on Alpaca", symbol)
                return False
            if str(getattr(asset, "status", "active")).lower() != "active":
                logger.warning("Symbol %s not active on Alpaca: %s", symbol, getattr(asset, "status", "?"))
                return False
            return True
        except Exception as e:
            logger.debug("Symbol check %s: %s", symbol, e)
            return True

    def _get_ohlcv_rest_fallback(
        self, symbol: str, timeframe: str, limit: int, days_back: int
    ) -> List[List]:
        """Fallback via direct REST stock bars endpoint when alpaca-py returns empty."""
        import requests
        tf_map = {
            "1m": "1Min", "5m": "5Min", "15m": "15Min",
            "1h": "1Hour", "1hour": "1Hour",
            "4h": "4Hour", "4hour": "4Hour",
            "1d": "1Day", "1day": "1Day",
            "1w": "1Week",
        }
        tf_api = tf_map.get((timeframe or "1h").lower(), "1Hour")
        end = datetime.now(timezone.utc)
        req_limit = min(max(int(limit) + OHLCV_MIN_BUFFER, 100), 10000)
        for attempt in range(OHLCV_RETRIES):
            try:
                # Extend date range on retry
                mult = 1 + (attempt * 2)
                days = max(14, int(days_back or 30) * mult)
                start = end - timedelta(days=min(days, 730))
                url = f"{self.data_url}/v2/stocks/{symbol}/bars"
                headers = {"APCA-API-KEY-ID": self.api_key, "APCA-API-SECRET-KEY": self.secret_key}
                feed = os.getenv("ALPACA_STOCK_FEED", "iex").strip() or "iex"
                params = {
                    "timeframe": tf_api,
                    "start": start.isoformat(),
                    "end": end.isoformat(),
                    "limit": req_limit,
                    "adjustment": "split",
                    "feed": feed if attempt < 2 else None,
                }
                params = {k: v for k, v in params.items() if v is not None}
                logger.info(
                    "REST OHLCV %s %s: attempt=%d range=%s..%s limit=%d",
                    symbol, timeframe, attempt + 1, start.date(), end.date(), req_limit,
                )
                r = requests.get(url, params=params, headers=headers, timeout=15)
                payload = r.json() if r.content else {}
                bars = (payload or {}).get("bars") or []
                logger.info("REST OHLCV %s %s: status=%d bars=%d", symbol, timeframe, r.status_code, len(bars))
                if r.status_code >= 400:
                    logger.warning("REST OHLCV %s %s: HTTP %d %s", symbol, timeframe, r.status_code, payload)
                    if attempt < OHLCV_RETRIES - 1:
                        time.sleep(OHLCV_BASE_SLEEP * (2 ** attempt))
                    continue
                if bars:
                    rows = []
                    for bar in bars[-limit:]:
                        t_val = bar.get("t", "")
                        if not t_val:
                            continue
                        ts = int(datetime.fromisoformat(str(t_val).replace("Z", "+00:00")).timestamp() * 1000)
                        rows.append([
                            ts,
                            float(bar.get("o", 0) or 0),
                            float(bar.get("h", 0) or 0),
                            float(bar.get("l", 0) or 0),
                            float(bar.get("c", 0) or 0),
                            int(bar.get("v", 0) or 0),
                        ])
                    return rows
                if attempt < OHLCV_RETRIES - 1:
                    time.sleep(OHLCV_BASE_SLEEP * (2 ** attempt))
            except Exception as e:
                logger.warning("REST OHLCV fallback %s %s attempt %d: %s", symbol, timeframe, attempt + 1, e)
                if attempt < OHLCV_RETRIES - 1:
                    time.sleep(OHLCV_BASE_SLEEP * (2 ** attempt))
        return []
    
    def get_ohlcv(self, symbol: str, timeframe: str = "1Hour", limit: int = 100) -> List[List]:
        """Returns [[ts_ms, o, h, l, c, v], ...]. Retries with exponential backoff, REST/Yahoo fallback for stocks."""
        tf_lower = (timeframe or "1h").lower()
        tf_map = {
            "1m": TimeFrame(1, TimeFrameUnit.Minute), "5m": TimeFrame(5, TimeFrameUnit.Minute),
            "15m": TimeFrame(15, TimeFrameUnit.Minute), "1h": TimeFrame(1, TimeFrameUnit.Hour),
            "1hour": TimeFrame(1, TimeFrameUnit.Hour),
            "4h": TimeFrame(4, TimeFrameUnit.Hour), "4hour": TimeFrame(4, TimeFrameUnit.Hour),
            "1d": TimeFrame(1, TimeFrameUnit.Day), "1day": TimeFrame(1, TimeFrameUnit.Day),
            "1w": TimeFrame(1, TimeFrameUnit.Week),
        }
        days_back_map = {"1m": 14, "5m": 21, "15m": 45, "1h": 60, "1hour": 60, "4h": 120, "4hour": 120, "1d": 500, "1day": 500, "1w": 104}
        tf = tf_map.get(tf_lower) or TimeFrame(1, TimeFrameUnit.Hour)
        days_back = days_back_map.get(tf_lower, 60)
        min_acceptable = 20
        req_limit = max(limit + OHLCV_MIN_BUFFER, min_acceptable + 10)
        cache_key = f"ohlcv_{symbol}_{tf_lower}_{limit}"
        cached = self.cache.get(cache_key)
        if cached and len(cached) >= min_acceptable:
            return cached

        is_stock = "/" not in symbol and len(symbol) < 10
        if is_stock:
            self._check_symbol_tradable(symbol)

        for attempt in range(OHLCV_RETRIES):
            if not self.rate_limiter.acquire("get_bars", priority=4):
                break
            try:
                mult = 1 + (attempt * 2)
                days = min(max(days_back * mult, 30), 730)
                end = datetime.now(timezone.utc)
                start = end - timedelta(days=days)
                req = StockBarsRequest(symbol_or_symbols=symbol, timeframe=tf, start=start, end=end, limit=req_limit)
                logger.info(
                    "get_ohlcv %s %s attempt=%d range=%s..%s limit=%d",
                    symbol, tf_lower, attempt + 1, start.date(), end.date(), req_limit,
                )
                bars = self.data_client.get_stock_bars(req)
                bar_list = []
                if bars:
                    data = getattr(bars, "data", None) or getattr(bars, "_data", None)
                    if data and symbol in data:
                        bar_list = list(data[symbol])
                    elif hasattr(bars, "__getitem__"):
                        try:
                            bar_list = list(bars[symbol]) if bars[symbol] else []
                        except (KeyError, TypeError):
                            pass
                logger.info("get_ohlcv %s %s: alpaca returned %d bars", symbol, tf_lower, len(bar_list))
                if bar_list:
                    result = []
                    for b in bar_list[-limit:]:
                        ts = int(b.timestamp.timestamp() * 1000) if hasattr(b.timestamp, "timestamp") else 0
                        result.append([ts, float(b.open), float(b.high), float(b.low), float(b.close), int(getattr(b, "volume", 0))])
                    if len(result) >= min_acceptable:
                        self.cache.set(cache_key, result, ttl_seconds=60)
                        logger.info("get_ohlcv %s %s: returning %d candles", symbol, tf_lower, len(result))
                        return result
                if attempt < OHLCV_RETRIES - 1:
                    time.sleep(OHLCV_BASE_SLEEP * (2 ** attempt))
            except Exception as e:
                logger.warning("get_ohlcv %s %s attempt %d: %s", symbol, tf_lower, attempt + 1, e)
                if attempt < OHLCV_RETRIES - 1:
                    time.sleep(OHLCV_BASE_SLEEP * (2 ** attempt))

        if is_stock:
            rest_rows = self._get_ohlcv_rest_fallback(symbol, tf_lower, limit, days_back)
            if rest_rows and len(rest_rows) >= min_acceptable:
                logger.info("REST fallback: %s %s -> %d candles", symbol, tf_lower, len(rest_rows))
                self.cache.set(cache_key, rest_rows, ttl_seconds=60)
                return rest_rows
            for yahoo_attempt in range(2):
                try:
                    from phase2_data_fetcher import _fetch_candles_yahoo
                    yahoo = _fetch_candles_yahoo(symbol, tf_lower, max(limit, 100))
                    if yahoo and len(yahoo) >= min_acceptable:
                        logger.info("Yahoo fallback: %s %s -> %d candles", symbol, tf_lower, len(yahoo))
                        self.cache.set(cache_key, yahoo, ttl_seconds=60)
                        return yahoo
                    if yahoo_attempt < 1:
                        time.sleep(OHLCV_BASE_SLEEP)
                except Exception as e:
                    logger.debug("Yahoo fallback %s %s: %s", symbol, tf_lower, e)
                    if yahoo_attempt < 1:
                        time.sleep(OHLCV_BASE_SLEEP)
        logger.warning("get_ohlcv failed: %s %s after %d attempts", symbol, tf_lower, OHLCV_RETRIES)
        return []

    def _aggregate_ohlcv(self, bars: List[List], n: int) -> List[List]:
        """Aggregate n bars into 1. bars = [[ts, o, h, l, c, v], ...]. AlpacaClient-compatible."""
        if not bars or n <= 1:
            return bars
        out = []
        for i in range(0, len(bars), n):
            chunk = bars[i:i + n]
            if not chunk:
                break
            ts = int(chunk[0][0])
            o = float(chunk[0][1])
            h = max(float(b[2]) for b in chunk)
            l = min(float(b[3]) for b in chunk)
            c = float(chunk[-1][4])
            v = sum(int(b[5]) if len(b) > 5 else 0 for b in chunk)
            out.append([ts, o, h, l, c, v])
        return out
    
    # ---------- AlpacaClient-compatible: Snapshots / Top movers ----------
    
    def get_snapshots(self, symbols: List[str]) -> Dict[str, Any]:
        """Returns {snapshots: {SYMBOL: {...}}}. Uses REST fallback for compatibility."""
        if not symbols or not self.rate_limiter.acquire("get_snapshots", priority=4):
            return {"snapshots": {}}
        try:
            import requests
            sym_str = ",".join(symbols[:100])
            url = f"{self.data_url}/v2/stocks/snapshots"
            headers = {"APCA-API-KEY-ID": self.api_key, "APCA-API-SECRET-KEY": self.secret_key}
            r = requests.get(url, params={"symbols": sym_str}, headers=headers, timeout=10)
            r.raise_for_status()
            data = r.json()
            if isinstance(data, dict) and "snapshots" in data:
                return data
            return {"snapshots": data if isinstance(data, dict) else {}}
        except Exception as e:
            logger.debug("get_snapshots: %s", e)
            return {"snapshots": {}}
    
    def get_top_movers(self) -> Dict[str, List[Dict]]:
        """Compatible with AlpacaClient.get_top_movers()."""
        watchlist = ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA", "NVDA", "META", "NFLX", "SPY", "QQQ", "IWM", "AMD", "INTC", "COIN", "MSTR", "MARA"]
        snaps = self.get_snapshots(watchlist)
        snap_dict = snaps.get("snapshots", {})
        parsed = []
        for sym, data in (snap_dict.items() if isinstance(snap_dict, dict) else []):
            try:
                if not isinstance(data, dict):
                    continue
                db = data.get("dailyBar") or {}
                pt = data.get("prevDailyBar") or {}
                lt = data.get("latestTrade") or {}
                current = float(db.get("c", 0) or lt.get("p", 0))
                prev = float(pt.get("c", 0) or current)
                pct = (current - prev) / prev * 100.0 if prev > 0 else 0.0
                vol = float(db.get("v", 0))
                parsed.append({"symbol": sym, "last": current, "percentage": pct, "quoteVolume": current * vol})
            except Exception:
                continue
        parsed.sort(key=lambda x: x["percentage"], reverse=True)
        return {"gainers": parsed[:6], "losers": sorted(parsed, key=lambda x: x["percentage"])[:6], "hot": sorted(parsed, key=lambda x: -x["quoteVolume"])[:6]}
    
    def search_assets(self, query: str, asset_class: str = "us_equity") -> List[Dict[str, Any]]:
        if not self.rate_limiter.acquire("search_assets", priority=3):
            return []
        try:
            import requests
            url = f"{self.base_url}/v2/assets"
            headers = {"APCA-API-KEY-ID": self.api_key, "APCA-API-SECRET-KEY": self.secret_key}
            r = requests.get(url, params={"status": "active", "asset_class": asset_class}, headers=headers, timeout=15)
            r.raise_for_status()
            assets = r.json() or []
            query_upper = (query or "").upper().strip()
            out = [a for a in assets if a.get("tradable") and a.get("status") == "active" and (not query_upper or query_upper in (a.get("symbol") or "").upper())][:50]
            return out
        except Exception as e:
            logger.debug("search_assets: %s", e)
            return []
    
    # ---------- AlpacaClient-compatible: Orders ----------
    
    def get_orders(self, status: str = "open", limit: int = 50) -> List[Dict[str, Any]]:
        orders = self.cached_client.get_orders(status=status)
        return [self._order_to_dict(o) for o in orders[:limit]]
    
    def _order_to_dict(self, o) -> Dict[str, Any]:
        if isinstance(o, dict):
            return o
        return {
            "id": str(getattr(o, "id", "")),
            "symbol": getattr(o, "symbol", ""),
            "side": getattr(o, "side", ""),
            "type": getattr(o, "type", ""),
            "qty": str(getattr(o, "qty", 0)),
            "filled_qty": str(getattr(o, "filled_qty", 0)),
            "limit_price": str(getattr(o, "limit_price", 0)) if getattr(o, "limit_price", None) else None,
            "status": str(getattr(o, "status", "")),
        }
    
    def place_market_order(self, symbol: str, qty: float, side: str, time_in_force: str = "day") -> Dict[str, Any]:
        if not self.rate_limiter.acquire("submit_order", priority=10):
            raise Exception("Rate limit: cannot place order")
        try:
            req = MarketOrderRequest(symbol=symbol, qty=qty, side=OrderSide.BUY if side.lower() == "buy" else OrderSide.SELL, time_in_force=TimeInForce.DAY if time_in_force.lower() == "day" else TimeInForce.GTC)
            order = self.trading_client.submit_order(order_data=req)
            self.cache.invalidate_pattern("orders_")
            return self._order_to_dict(order)
        except Exception as e:
            if "429" in str(e):
                self.rate_limiter.enter_backoff(429)
            raise
    
    def place_limit_order(self, symbol: str, qty: float, limit_price: float, side: str, time_in_force: str = "day") -> Dict[str, Any]:
        if not self.rate_limiter.acquire("submit_order", priority=10):
            raise Exception("Rate limit: cannot place order")
        try:
            req = LimitOrderRequest(symbol=symbol, qty=qty, limit_price=limit_price, side=OrderSide.BUY if side.lower() == "buy" else OrderSide.SELL, time_in_force=TimeInForce.DAY if time_in_force.lower() == "day" else TimeInForce.GTC)
            order = self.trading_client.submit_order(order_data=req)
            self.cache.invalidate_pattern("orders_")
            return self._order_to_dict(order)
        except Exception as e:
            if "429" in str(e):
                self.rate_limiter.enter_backoff(429)
            raise
    
    def cancel_order(self, order_id: str) -> Dict[str, Any]:
        if not self.rate_limiter.acquire("cancel_order", priority=9):
            raise Exception("Rate limit: cannot cancel order")
        self.trading_client.cancel_order_by_id(order_id)
        return {"id": order_id, "status": "canceled"}
    
    def cancel_all_orders(self) -> List[Dict]:
        self.trading_client.cancel_orders()
        return []
    
    def close_position(self, symbol: str) -> Dict[str, Any]:
        if not self.rate_limiter.acquire("close_position", priority=10):
            raise Exception("Rate limit: cannot close position")
        self.trading_client.close_position(symbol)
        self.cache.invalidate(f"position_{symbol}")
        self.cache.invalidate("positions_all")
        return {"symbol": symbol, "closed": True}
    
    # ---------- Unified client methods (order API) ----------
    
    def place_market_order_unified(self, symbol: str, qty: float, side: str, time_in_force: str = "day"):
        return self.place_market_order(symbol, qty, side, time_in_force)
    
    def place_limit_order_unified(self, symbol: str, qty: float, side: str, limit_price: float, time_in_force: str = "day"):
        return self.place_limit_order(symbol, qty, limit_price, side, time_in_force)
    
    # ---------- Stats & Shutdown ----------
    
    def get_stats(self) -> Dict:
        return {"websocket": self.websocket.get_stats(), "cache": self.cache.get_stats(), "rate_limiter": self.rate_limiter.get_all_stats(), "subscribed_symbols": self.websocket.get_subscribed_symbols()}
    
    def print_stats(self):
        stats = self.get_stats()
        cache = stats.get("cache", {})
        rl = stats.get("rate_limiter", {})
        global_rl = rl.get("global", {}) if isinstance(rl, dict) else {}
        tokens = global_rl.get("available_tokens", "?")
        logger.info(
            "WebSocket stats: cache_hit_rate=%s hits=%s api_saved=%s rate_limit_tokens=%s/180",
            cache.get("hit_rate", "0%"), cache.get("hits", 0), cache.get("api_calls_saved", 0), tokens
        )

    def check_websocket_health(self) -> Tuple[bool, List[str]]:
        """Verify WebSocket, cache, and rate limiter. Returns (ok, issues)."""
        issues = []
        try:
            ws_stats = self.websocket.get_stats()
            if not ws_stats.get("running", False):
                issues.append("WebSocket not running")
            cache_stats = self.cache.get_stats()
            hit_rate_str = cache_stats.get("hit_rate", "0%")
            hit_rate_val = float(str(hit_rate_str).replace("%", "")) if isinstance(hit_rate_str, str) else 0
            if hit_rate_val < 70 and cache_stats.get("hits", 0) + cache_stats.get("misses", 0) > 10:
                issues.append(f"Low cache hit rate: {hit_rate_str}")
            rl = self.rate_limiter.get_all_stats()
            tokens = rl.get("global", {}).get("available_tokens", 0)
            if tokens < 50:
                issues.append(f"Low rate limit tokens: {tokens}/180")
            return len(issues) == 0, issues
        except Exception as e:
            return False, [f"WebSocket health check error: {e}"]
    
    def shutdown(self):
        logger.info("Shutting down UnifiedAlpacaClient...")
        self.websocket.stop()
        self.cache.clear()
