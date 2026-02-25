"""
Alpaca Trading Client
Wrapper for Alpaca Markets API - handles stock trading, market data, and account management.
Similar to kraken_client.py but for stocks/ETFs.
"""

import os
import time
import requests
from typing import Optional, Dict, List, Any, Tuple
from datetime import datetime, timezone


class AlpacaClient:
    """
    Alpaca API wrapper for stock trading.
    Supports both paper trading and live trading.
    
    Documentation: https://docs.alpaca.markets/docs
    """
    
    def __init__(self, mode: str = "paper"):
        """
        Initialize Alpaca client.
        
        Args:
            mode: "paper" for paper trading, "live" for real money
        """
        self.mode = mode
        
        # Get API keys from environment
        if mode == "paper":
            self.api_key = os.getenv("ALPACA_API_KEY_PAPER", "")
            self.secret_key = os.getenv("ALPACA_API_SECRET_PAPER", "")
            self.base_url = "https://paper-api.alpaca.markets"
        else:
            self.api_key = os.getenv("ALPACA_API_KEY_LIVE", "")
            self.secret_key = os.getenv("ALPACA_API_SECRET_LIVE", "")
            self.base_url = "https://api.alpaca.markets"
        
        self.data_url = "https://data.alpaca.markets"
        
        # Validate credentials
        if not self.api_key or not self.secret_key:
            raise ValueError(f"Alpaca {mode} API keys not found in environment variables")
        
        # Default headers for all requests
        self.headers = {
            "APCA-API-KEY-ID": self.api_key,
            "APCA-API-SECRET-KEY": self.secret_key,
            "Content-Type": "application/json"
        }
        
        # Rate limiting
        self.last_request_time = 0
        self.min_request_interval = 0.2  # 200ms between requests (max 200 req/min)

        # Data feed for stocks (iex=free, sip/otc etc require entitlements)
        self.data_feed = os.getenv("ALPACA_DATA_FEED", "").strip().lower()
        
        # Connection pooling
        self.session = requests.Session()
        self.session.headers.update(self.headers)
    
    def _rate_limit(self):
        """Enforce rate limiting between requests."""
        now = time.time()
        elapsed = now - self.last_request_time
        if elapsed < self.min_request_interval:
            time.sleep(self.min_request_interval - elapsed)
        self.last_request_time = time.time()
    
    def _request(self, method: str, endpoint: str, params: Optional[Dict] = None, 
                 data: Optional[Dict] = None, base_url: Optional[str] = None) -> Dict:
        """
        Make authenticated request to Alpaca API.
        Uses alpaca_rate_limiter to prevent 429 (123.md Fix 1).
        """
        self._rate_limit()
        try:
            from alpaca_rate_limiter import alpaca_rate_limiter
            alpaca_rate_limiter.acquire(request_name=endpoint.split("/")[-1] or "request")
        except ImportError:
            pass

        url = f"{base_url or self.base_url}{endpoint}"
        timeout_sec = 15
        max_retries = 3
        last_exc = None
        for attempt in range(max_retries):
            try:
                resp = self.session.request(method, url, params=params, json=data, timeout=timeout_sec)
                last_exc = None
                break
            except (requests.exceptions.ConnectionError, requests.exceptions.Timeout, OSError) as e:
                last_exc = e
                if attempt < max_retries - 1:
                    delay = (2 ** attempt) * 1.0
                    time.sleep(min(delay, 8))
                else:
                    msg = str(e)
                    if "getNameResolver" in msg or "Name or service not known" in msg or "getaddrinfo" in msg.lower():
                        raise Exception(f"DNS/network error: {msg}") from e
                    raise Exception(f"Connection failed after {max_retries} attempts: {msg}") from e
        
        try:
            if resp.status_code == 404:
                try:
                    err_json = resp.json()
                    msg = err_json.get("message", "")
                except Exception:
                    msg = resp.text
                raise ValueError(f"Resource not found (404): {url} - {msg}")
            resp.raise_for_status()
            return resp.json()
        except requests.exceptions.HTTPError as e:
            status = e.response.status_code
            error_msg = f"Alpaca API error: HTTP {status}"
            try:
                error_data = e.response.json()
                api_msg = error_data.get('message', '') or error_data.get('error', '')
                if api_msg:
                    error_msg += f" - {api_msg}"
            except Exception:
                pass
            if status == 429:
                try:
                    from alpaca_rate_limiter import alpaca_rate_limiter
                    alpaca_rate_limiter.handle_429_error()
                    alpaca_rate_limiter.acquire(request_name="retry")
                    resp = self.session.request(method, url, params=params, json=data, timeout=timeout_sec)
                    resp.raise_for_status()
                    alpaca_rate_limiter.reset_backoff()
                    return resp.json()
                except ImportError:
                    pass
            exc = Exception(error_msg)
            exc.status_code = status
            raise exc
        except requests.exceptions.Timeout:
            raise Exception("Alpaca API request timed out")
        except ValueError:
            raise
        except Exception as e:
            raise Exception(f"Alpaca API request failed: {str(e)}") from e
    
    # =========================================================================
    # ACCOUNT & BALANCE
    # =========================================================================
    
    def get_account(self) -> Dict[str, Any]:
        """
        Get account information including balance and buying power.
        
        Returns:
            {
                'cash': '10000.00',
                'buying_power': '10000.00',
                'portfolio_value': '10000.00',
                'equity': '10000.00',
                'last_equity': '10000.00',
                'daytrade_count': 0,
                'pattern_day_trader': False,
                ...
            }
        """
        return self._request("GET", "/v2/account")
    
    def get_balance(self) -> float:
        """Get available cash balance in USD."""
        account = self.get_account()
        return float(account.get("cash", 0))
    
    def get_buying_power(self) -> float:
        """Get buying power (cash + margin if applicable)."""
        account = self.get_account()
        return float(account.get("buying_power", 0))
    
    def get_portfolio_value(self) -> float:
        """Get total portfolio value (cash + positions)."""
        account = self.get_account()
        return float(account.get("portfolio_value", 0))
    
    # =========================================================================
    # MARKET HOURS & STATUS
    # =========================================================================
    
    def get_clock(self) -> Dict[str, Any]:
        """
        Get market clock status.
        
        Returns:
            {
                'timestamp': '2024-01-24T12:00:00-05:00',
                'is_open': True,
                'next_open': '2024-01-25T09:30:00-05:00',
                'next_close': '2024-01-24T16:00:00-05:00'
            }
        """
        return self._request("GET", "/v2/clock")
    
    def is_market_open(self) -> bool:
        """Check if market is currently open for trading."""
        import logging
        logger = logging.getLogger(__name__)
        try:
            clock = self.get_clock()
            is_open = clock.get("is_open", False)
            logger.debug(f"Alpaca clock check: is_open={is_open}, timestamp={clock.get('timestamp')}")
            return is_open
        except Exception as e:
            logger.error(f"Failed to check market status from Alpaca: {e}", exc_info=True)
            # If API fails, don't assume closed - let caller decide or retry
            raise
    
    def get_market_hours(self) -> Tuple[Optional[str], Optional[str]]:
        """
        Get today's market open and close times.
        
        Returns:
            (open_time, close_time) as ISO strings, or (None, None) if closed
        """
        clock = self.get_clock()
        if not clock.get("is_open"):
            return None, None
        return clock.get("next_open"), clock.get("next_close")
    
    # =========================================================================
    # MARKET DATA
    # =========================================================================
    
    def get_ticker(self, symbol: str) -> Dict[str, Any]:
        """
        Get latest quote for a symbol with robust fallbacks.
        Validates all prices (must be > 0, finite).
        
        Args:
            symbol: Stock symbol (e.g., "AAPL", "TSLA")
        
        Returns:
            {
                'symbol': 'AAPL',
                'last': 175.50,
                'bid': 175.48,
                'ask': 175.52,
                'volume': 50000000,
                'timestamp': '2024-01-24T12:00:00Z'
            }
        """
        import math
        last_errors = []
        
        # Method 1: Latest quotes (most accurate for live market)
        try:
            endpoint = f"/v2/stocks/{symbol}/quotes/latest"
            data = self._request("GET", endpoint, base_url=self.data_url)
            quote = data.get("quote", {})
            ap = float(quote.get("ap", 0))  # ask price
            bp = float(quote.get("bp", 0))  # bid price
            
            # Use mid-price if both available, otherwise use ask or bid
            if ap > 0 and bp > 0:
                last = (ap + bp) / 2
            elif ap > 0:
                last = ap
            elif bp > 0:
                last = bp
            else:
                last = 0.0
            
            # Validate: must be positive and finite
            if last > 0 and math.isfinite(last):
                return {
                    "symbol": symbol,
                    "last": last,
                    "bid": bp if bp > 0 and math.isfinite(bp) else last,
                    "ask": ap if ap > 0 and math.isfinite(ap) else last,
                    "volume": 0, # Volume not available in quotes/latest
                    "timestamp": quote.get("t", "")
                }
        except Exception as e:
            last_errors.append(f"quotes/latest: {type(e).__name__}")

        # Method 2: Latest trade (good fallback)
        try:
            endpoint = f"/v2/stocks/{symbol}/trades/latest"
            data = self._request("GET", endpoint, base_url=self.data_url)
            trade = data.get("trade", {})
            last = float(trade.get("p", 0))  # price
            
            if last > 0 and math.isfinite(last):
                return {
                    "symbol": symbol,
                    "last": last,
                    "bid": last,  # Approximate
                    "ask": last,  # Approximate
                    "volume": float(trade.get("s", 0)) if trade.get("s") else 0,
                    "timestamp": trade.get("t", "")
                }
        except Exception as e:
            last_errors.append(f"trades/latest: {type(e).__name__}")

        # Method 3: Snapshot (most reliable, works even when market closed)
        try:
            endpoint = f"/v2/stocks/snapshots"
            data = self._request("GET", endpoint, params={"symbols": symbol}, base_url=self.data_url)
            snapshots = data.get("snapshots", {})
            snap = snapshots.get(symbol, {})
            
            # Try latest trade from snapshot first
            latest_trade = snap.get("latestTrade", {})
            if latest_trade:
                last = float(latest_trade.get("p", 0))
                if last > 0 and math.isfinite(last):
                    quote = snap.get("latestQuote", {})
                    bp = float(quote.get("bp", last)) if quote.get("bp") else last
                    ap = float(quote.get("ap", last)) if quote.get("ap") else last
                    return {
                        "symbol": symbol,
                        "last": last,
                        "bid": bp if bp > 0 and math.isfinite(bp) else last,
                        "ask": ap if ap > 0 and math.isfinite(ap) else last,
                        "volume": float(latest_trade.get("s", 0)) if latest_trade.get("s") else 0,
                        "timestamp": latest_trade.get("t", "")
                    }
            
            # Fallback to daily bar close (last known price when market closed)
            daily_bar = snap.get("dailyBar", {})
            if daily_bar:
                last = float(daily_bar.get("c", 0))  # close
                if last > 0 and math.isfinite(last):
                    return {
                        "symbol": symbol,
                        "last": last,
                        "bid": last,
                        "ask": last,
                        "volume": float(daily_bar.get("v", 0)) if daily_bar.get("v") else 0,
                        "timestamp": daily_bar.get("t", "")
                    }
        except Exception as e:
            last_errors.append(f"snapshots: {type(e).__name__}")

        # Method 4: Crypto API (for crypto symbols if stock API fails)
        try:
            # crypto symbol needs /USD usually, or might be just symbol
            c_sym = symbol if "/" in symbol else f"{symbol}/USD"
            endpoint = "/v1beta3/crypto/us/quotes/latest"
            data = self._request("GET", endpoint, params={"symbols": c_sym}, base_url=self.data_url)
            quotes = data.get("quotes", {})
            q = quotes.get(c_sym, {})
            
            if q:
                last_c = (float(q.get("ap", 0)) + float(q.get("bp", 0))) / 2
                if last_c > 0:
                    return {
                        "symbol": symbol,
                        "last": last_c,
                        "bid": float(q.get("bp", 0)),
                        "ask": float(q.get("ap", 0)),
                        "volume": 0, # Crypto snapshot volume often separate
                        "timestamp": q.get("t", "")
                    }
        except Exception:
            pass

        # Return empty/zero if all methods fail - this will cause bot to retry
        # Log the failure for debugging (but don't raise exception to allow retry logic)
        import logging
        logging.warning(f"AlpacaClient.get_ticker: All methods failed for {symbol}. Check API keys, market hours, and symbol validity.")
        return {
            "symbol": symbol,
            "last": 0.0,
            "bid": 0.0,
            "ask": 0.0,
            "volume": 0,
            "timestamp": None
        }
    
    def _aggregate_ohlcv(self, bars: List[List], n: int) -> List[List]:
        """Aggregate n bars into 1. bars = [[ts, o, h, l, c, v], ...]. Returns same format."""
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
            v = sum(int(b[5]) for b in chunk)
            out.append([ts, o, h, l, c, v])
        return out

    def get_ohlcv(self, symbol: str, timeframe: str = "1Hour", limit: int = 100) -> List[List]:
        """
        Get historical OHLCV bars for a symbol.
        
        Args:
            symbol: Stock symbol
            timeframe: Bar timeframe ("1Min", "5Min", "15Min", "1Hour", "1Day")
            limit: Number of bars to fetch (max 10000)
        
        Returns:
            List of [timestamp_ms, open, high, low, close, volume]
        """
        import logging
        log = logging.getLogger(__name__)

        # Map timeframes. For stocks, 4h: aggregate from 1h (Alpaca stocks may not support 4Hour).
        tf_map_stock = {
            "1m": "1Min", "5m": "5Min", "15m": "15Min",
            "1h": "1Hour", "1d": "1Day", "1w": "1Week"
        }
        tf_map_crypto = dict(tf_map_stock, **{"4h": "4Hour"})
        tf_lower = (timeframe or "1h").lower()
        is_stock = "/" not in symbol

        # For stocks 4h: always aggregate 1h->4h (do not send 4Hour to Alpaca)
        if is_stock and tf_lower in ("4h", "4hour"):
            try:
                one_h = self.get_ohlcv(symbol, "1h", limit=min(limit * 4, 500))
                if one_h and len(one_h) >= 20:
                    agg = self._aggregate_ohlcv(one_h, 4)
                    if agg:
                        return agg[-limit:] if len(agg) > limit else agg
            except Exception as e:
                log.warning("get_ohlcv returning []: symbol=%r timeframe=4h limit=%s mode=%s exc=%s",
                            symbol, limit, self.mode, e)
                return []
            log.warning("get_ohlcv returning []: symbol=%r timeframe=4h limit=%s mode=%s reason=empty_or_insufficient_1h",
                        symbol, limit, self.mode)
            return []

        # Try Stock API
        alpaca_tf = tf_map_stock.get(tf_lower, timeframe)
        try:
            endpoint = f"/v2/stocks/{symbol}/bars"
            params = {
                "timeframe": alpaca_tf,
                "limit": 10000,
                "start": (datetime.now(timezone.utc).replace(year=datetime.now(timezone.utc).year - 5)).isoformat(),
                "adjustment": "split",
            }
            if self.data_feed and self.data_feed in ("iex", "sip", "delayed_sip", "otc", "boats", "overnight"):
                params["feed"] = self.data_feed

            data = self._request("GET", endpoint, params=params, base_url=self.data_url)

            # Check for API error in response dict
            if isinstance(data, dict):
                err_msg = data.get("message") or data.get("error")
                if err_msg:
                    log.warning(
                        "get_ohlcv returning []: symbol=%r timeframe=%s limit=%s mode=%s api_error=%s",
                        symbol,
                        tf_lower,
                        limit,
                        self.mode,
                        err_msg,
                    )
                    bars = []
                else:
                    bars = data.get("bars") or []
            else:
                bars = []

            if not bars:
                log.warning(
                    "get_ohlcv returning []: symbol=%r timeframe=%s limit=%s mode=%s reason=empty_bars_from_api",
                    symbol,
                    tf_lower,
                    limit,
                    self.mode,
                )
            else:
                if len(bars) > limit:
                    bars = bars[-limit:]

                result = [
                    [
                        int(datetime.fromisoformat(bar["t"].replace("Z", "+00:00")).timestamp() * 1000),
                        float(bar["o"]),
                        float(bar["h"]),
                        float(bar["l"]),
                        float(bar["c"]),
                        int(bar["v"]),
                    ]
                    for bar in bars
                ]
                return result[-limit:] if len(result) > limit else result
        except Exception as e:
            log.warning(
                "get_ohlcv stock call failed: symbol=%r timeframe=%s limit=%s mode=%s exc=%s",
                symbol,
                tf_lower,
                limit,
                self.mode,
                e,
            )

        # Fallback to Crypto API (for symbols like ETH/USD)
        if "/" in symbol:
            try:
                c_sym = symbol
                alpaca_tf = tf_map_crypto.get(tf_lower, alpaca_tf)
                endpoint = "/v1beta3/crypto/us/bars"
                params = {
                    "symbols": c_sym,
                    "timeframe": alpaca_tf,
                    "limit": min(limit, 1000)
                }
                data = self._request("GET", endpoint, params=params, base_url=self.data_url)
                c_bars = (data.get("bars") or {}).get(c_sym, [])

                if len(c_bars) > 0:
                    return [
                        [
                            int(datetime.fromisoformat(bar["t"].replace("Z", "+00:00")).timestamp() * 1000),
                            float(bar["o"]),
                            float(bar["h"]),
                            float(bar["l"]),
                            float(bar["c"]),
                            int(bar["v"])
                        ]
                        for bar in c_bars
                    ]
            except Exception as e:
                log.warning("get_ohlcv returning []: symbol=%r timeframe=%s limit=%s mode=%s (crypto fallback) exc=%s",
                            symbol, tf_lower, limit, self.mode, e)

        log.warning("get_ohlcv returning []: symbol=%r timeframe=%s limit=%s mode=%s reason=no_bars_after_fallback",
                    symbol, tf_lower, limit, self.mode)
        return []
    
    # =========================================================================
    # POSITIONS
    # =========================================================================
    
    def get_positions(self) -> List[Dict[str, Any]]:
        """
        Get all open positions.
        
        Returns:
            List of position dicts
        """
        return self._request("GET", "/v2/positions")
    
    def get_position(self, symbol: str) -> Optional[Dict[str, Any]]:
        """Get position for specific symbol, or None if no position."""
        try:
            return self._request("GET", f"/v2/positions/{symbol}")
        except Exception as e:
            import logging
            logging.getLogger(__name__).debug("get_position %s failed: %s", symbol, e)
            return None
    
    def close_position(self, symbol: str) -> Dict[str, Any]:
        """Close entire position for a symbol (market order)."""
        return self._request("DELETE", f"/v2/positions/{symbol}")
    
    # =========================================================================
    # ORDERS
    # =========================================================================
    
    def place_market_order(self, symbol: str, qty: float, side: str, 
                          time_in_force: str = "day") -> Dict[str, Any]:
        """
        Place market order.
        
        Args:
            symbol: Stock symbol
            qty: Quantity (can be fractional)
            side: "buy" or "sell"
            time_in_force: "day", "gtc", "ioc", "fok"
        
        Returns:
            Order dict with order_id, status, etc.
        """
        data = {
            "symbol": symbol,
            "qty": str(qty),
            "side": side.lower(),
            "type": "market",
            "time_in_force": time_in_force
        }
        return self._request("POST", "/v2/orders", data=data)
    
    def place_limit_order(self, symbol: str, qty: float, limit_price: float, 
                         side: str, time_in_force: str = "day") -> Dict[str, Any]:
        """
        Place limit order.
        
        Args:
            symbol: Stock symbol
            qty: Quantity (can be fractional)
            limit_price: Limit price
            side: "buy" or "sell"
            time_in_force: "day", "gtc", "ioc", "fok"
        
        Returns:
            Order dict
        """
        data = {
            "symbol": symbol,
            "qty": str(qty),
            "side": side.lower(),
            "type": "limit",
            "time_in_force": time_in_force,
            "limit_price": str(limit_price)
        }
        return self._request("POST", "/v2/orders", data=data)
    
    def place_stop_order(self, symbol: str, qty: float, stop_price: float, 
                        side: str, time_in_force: str = "day") -> Dict[str, Any]:
        """Place stop (stop-loss) order."""
        data = {
            "symbol": symbol,
            "qty": str(qty),
            "side": side.lower(),
            "type": "stop",
            "time_in_force": time_in_force,
            "stop_price": str(stop_price)
        }
        return self._request("POST", "/v2/orders", data=data)
    
    def get_orders(self, status: str = "open", limit: int = 50) -> List[Dict[str, Any]]:
        """
        Get orders.
        
        Args:
            status: "open", "closed", "all"
            limit: Max orders to return
        
        Returns:
            List of order dicts
        """
        params = {"status": status, "limit": limit}
        return self._request("GET", "/v2/orders", params=params)
    
    def get_order(self, order_id: str) -> Dict[str, Any]:
        """Get specific order by ID."""
        return self._request("GET", f"/v2/orders/{order_id}")
    
    def cancel_order(self, order_id: str) -> Dict[str, Any]:
        """Cancel an order."""
        return self._request("DELETE", f"/v2/orders/{order_id}")
    
    def cancel_all_orders(self) -> List[Dict[str, Any]]:
        """Cancel all open orders."""
        return self._request("DELETE", "/v2/orders")
    
    # =========================================================================
    # ASSET SEARCH
    # =========================================================================
    
    # =========================================================================
    # ASSET SEARCH & DISCOVERY
    # =========================================================================
    
    def search_assets(self, query: str, asset_class: str = "us_equity") -> List[Dict[str, Any]]:
        """
        Search for tradeable assets. Filtering is STRICT to avoid OTC/unavailable assets.
        
        Args:
            query: Search query (symbol or name)
            asset_class: "us_equity", "crypto"
        
        Returns:
            List of asset dicts
        """
        params = {
            "status": "active",
            "asset_class": asset_class
        }
        try:
            assets = self._request("GET", "/v2/assets", params=params)
        except Exception:
            return []
        
        query_upper = query.upper().strip()
        query_lower = query.lower().strip()
        
        results = []
        for asset in assets:
            # Strict filtering: Must be tradeable and active
            if not asset.get("tradable") or asset.get("status") != "active":
                continue
                
            # Filter by query
            sym = asset.get("symbol", "")
            name = asset.get("name", "")
            
            # Simple match
            if query_upper in sym.upper() or query_lower in name.lower():
                results.append(asset)
                
        return results[:50] # Limit results
    
    def get_asset(self, symbol: str) -> Dict[str, Any]:
        """Get asset details for a symbol."""
        return self._request("GET", f"/v2/assets/{symbol}")

    def get_active_assets(self, asset_class: str = "us_equity") -> List[Dict[str, Any]]:
        """
        Get ALL active, tradable assets.
        Used for broader market scanning.
        """
        params = {
            "status": "active",
            "asset_class": asset_class
        }
        try:
            assets = self._request("GET", "/v2/assets", params=params)
            # Filter for tradable AND active explicitly
            return [a for a in assets if a.get("tradable") and a.get("status") == "active"]
        except Exception:
            return []

    def get_snapshots(self, symbols: List[str]) -> Dict[str, Any]:
        """
        Get snapshots (price, change, volume) for multiple symbols.
        Used for the dashboard/explore tab.
        Returns: {"snapshots": {"SYMBOL": {...}}}
        """
        if not symbols:
            return {"snapshots": {}}
            
        # Paper API limit is often 100 symbols per request
        # We will just fetch details for the requested list
        sym_str = ",".join(symbols[:100])
        endpoint = "/v2/stocks/snapshots"
        
        try:
            result = self._request("GET", endpoint, params={"symbols": sym_str}, base_url=self.data_url)
            # Ensure result has "snapshots" key
            if isinstance(result, dict) and "snapshots" in result:
                return result
            elif isinstance(result, dict):
                # If response is flat dict, wrap it
                return {"snapshots": result}
            else:
                return {"snapshots": {}}
        except Exception as e:
            import logging
            logging.warning(f"get_snapshots failed: {e}")
            return {"snapshots": {}}
            
    def get_top_movers(self) -> Dict[str, List[Dict]]:
        """
        Simulate 'Top Movers' by fetching a predefined list of popular stocks/crypto
        and sorting them, since Alpaca Data API 'movers' endpoint requires premium subscription usually.
        We will return a structure compatible with the frontend.
        """
        # List of popular assets to track for the dashboard
        # Mix of Tech, ETFs, Crypto-stocks
        watchlist = [
            "AAPL", "MSFT", "GOOGL", "AMZN", "TSLA", "NVDA", "META", "NFLX",
            "SPY", "QQQ", "IWM", "AMD", "INTC", "COIN", "MSTR", "MARA",
            "RIOT", "PLTR", "SOFI", "UBER", "HOOD", "ROKU", "PYPL", "SQ"
        ]
        
        snaps = self.get_snapshots(watchlist)
        
        parsed = []
        for sym, data in (snaps.get("snapshots") or {}).items():
            try:
                # Alpaca snapshot structure: 
                # { "dailyBar": { "c": ..., "o": ... }, "prevDailyBar": { "c": ... } }
                # percent change = (current - prev_close) / prev_close
                
                # Use dailyBar close if available, else latestTrade
                current = 0.0
                if "dailyBar" in data and data["dailyBar"]:
                    current = data["dailyBar"].get("c", 0)
                elif "latestTrade" in data and data["latestTrade"]:
                    current = data["latestTrade"].get("p", 0)
                    
                prev = 0.0
                if "prevDailyBar" in data and data["prevDailyBar"]:
                    prev = data["prevDailyBar"].get("c", 0)
                    
                pct = 0.0
                if prev > 0 and current > 0:
                    pct = (current - prev) / prev * 100.0
                    
                vol = 0
                if "dailyBar" in data and data["dailyBar"]:
                    vol = data["dailyBar"].get("v", 0)
                    
                parsed.append({
                    "symbol": sym,
                    "last": current,
                    "percentage": pct,
                    "quoteVolume": float(current * vol) # approx dollar volume
                })
            except Exception as e:
                import logging
                logging.getLogger(__name__).debug("parse market mover %s: %s", sym, e)
                continue
                
        # Sort for categories
        parsed.sort(key=lambda x: x["percentage"], reverse=True)
        gainers = parsed[:6]
        losers = sorted(parsed, key=lambda x: x["percentage"])[:6]
        
        parsed.sort(key=lambda x: x["quoteVolume"], reverse=True)
        hot = parsed[:6]
        
        return {
            "gainers": gainers,
            "losers": losers,
            "hot": hot
        }
    
    # =========================================================================
    # CONVENIENCE METHODS
    # =========================================================================
    
    def format_order_result(self, order: Dict) -> str:
        """Format order result for display."""
        symbol = order.get("symbol", "?")
        side = order.get("side", "?").upper()
        qty = order.get("qty", "?")
        order_type = order.get("type", "?")
        status = order.get("status", "?")
        
        if order_type == "market":
            return f"{side} {qty} {symbol} (market) - {status}"
        elif order_type == "limit":
            price = order.get("limit_price", "?")
            return f"{side} {qty} {symbol} @ ${price} (limit) - {status}"
        else:
            return f"{side} {qty} {symbol} ({order_type}) - {status}"
    
    def is_tradeable(self, symbol: str) -> bool:
        """Check if symbol is tradeable."""
        try:
            asset = self.get_asset(symbol)
            return asset.get("tradable", False) and asset.get("status") == "active"
        except Exception as e:
            import logging
            logging.getLogger(__name__).debug("is_tradeable %s: %s", symbol, e)
            return False
    
    def __repr__(self):
        return f"<AlpacaClient mode={self.mode} base_url={self.base_url}>"


# Quick test function
if __name__ == "__main__":
    # Test with paper trading
    try:
        client = AlpacaClient(mode="paper")
        print(f"✓ Alpaca client initialized: {client}")
        
        # Test account
        account = client.get_account()
        print(f"✓ Account balance: ${float(account.get('cash', 0)):,.2f}")
        
        # Test market status
        is_open = client.is_market_open()
        print(f"✓ Market is {'OPEN' if is_open else 'CLOSED'}")
        
        # Test ticker
        ticker = client.get_ticker("AAPL")
        print(f"✓ AAPL last price: ${ticker.get('last', 0):.2f}")
        
        print("\n✅ All tests passed! Alpaca integration ready.")
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        print("\nMake sure you have set ALPACA_PAPER_API_KEY and ALPACA_PAPER_SECRET_KEY in your .env file")
