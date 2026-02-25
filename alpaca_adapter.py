
from typing import Dict, List, Any, Optional
import time
import logging
from alpaca_client import AlpacaClient

logger = logging.getLogger(__name__)

class AlpacaAdapter:
    """
    Wraps AlpacaClient to provide a KrakenClient-like interface
    so it can be used by BotRunner without major refactoring.
    """
    def __init__(self, client: AlpacaClient):
        # CRITICAL: Verify we received an AlpacaClient, not KrakenClient
        try:
            from unified_alpaca_client import UnifiedAlpacaClient
            ok = isinstance(client, (AlpacaClient, UnifiedAlpacaClient))
        except ImportError:
            ok = isinstance(client, AlpacaClient)
        if not ok:
            raise TypeError(f"AlpacaAdapter requires AlpacaClient or UnifiedAlpacaClient, got {type(client).__name__}.")
        self.client = client
        self._markets_cache = {}
        self._market_open_cache: Optional[bool] = None
        self._market_open_cache_ts: float = 0.0
        self._market_open_cache_ttl: float = 60.0  # Cache for 60 seconds

    def load_markets(self) -> Dict[str, Any]:
        # BotRunner calls this to resolve symbols.
        # We can't load all Alpaca assets efficiently every loop.
        # Instead, we return a "Magic" dict or just the cached ones?
        # BotRunner logic:
        #   markets = self.kc.load_markets()
        #   symbol = _try_resolve_symbol(markets, raw_symbol)
        # where _try_resolve_symbol checks if key exists.
        #
        # We can implement a "smart" dict or just rely on the fact that
        # valid inputs are passed. But BotRunner checks:
        # if not symbol: error.
        
        # We will populate it on demand or return a dict that claims to have everything?
        # A simple approach: return an empty dict but ensure valid symbols pass locally?
        # No, BotRunner uses `mk = markets[symbol]`.
        
        # So we must return a dict containing at least the symbols we care about.
        # But we don't know the symbol here yet (BotRunner knows it).
        
        # Hack: The adapter doesn't know the symbol until fetch_ticker is called.
        # But `load_markets` is called first.
        
        # Better approach: The BotRunner logic for `load_markets` is crypto specific.
        # We can mock it to return a dict that acts like it has the symbol?
        return self._markets_cache

    def ensure_market(self, symbol: str):
        """Populate the fake market entry so validation passes."""
        if symbol not in self._markets_cache:
            self._markets_cache[symbol] = {
                "id": symbol,
                "symbol": symbol,
                "base": symbol,
                "quote": "USD",
                "active": True,
                "spot": True,
                "precision": {"amount": 4, "price": 2}, # Approximate for stocks
                "limits": {"amount": {"min": 0.0001}, "cost": {"min": 1.0}}
            }

    def get_ticker(self, symbol: str) -> Dict[str, Any]:
        """Alias for the underlying AlpacaClient's get_ticker method."""
        return self.client.get_ticker(symbol)

    def fetch_ticker(self, symbol: str) -> Dict[str, Any]:
        """Fetch ticker data for a symbol. Always returns a dict, even on error."""
        try:
            t = self.client.get_ticker(symbol)
            
            # If t is not a dict (e.g. string error), return partial
            if not isinstance(t, dict):
                return {
                    "symbol": symbol,
                    "bid": 0.0,
                    "ask": 0.0,
                    "last": 0.0,
                    "price": 0.0,
                    "close": 0.0,
                    "timestamp": None
                }
            
            # Extract price from multiple possible fields
            last = t.get("last") or t.get("price") or t.get("close") or 0.0
            
            res = {
                "symbol": symbol,
                "bid": t.get("bid") or 0.0,
                "ask": t.get("ask") or 0.0,
                "last": float(last),
                "price": float(last),
                "close": float(last),
                "timestamp": t.get("timestamp")
            }
            return res
        except Exception as e:
            # Handle 404 or other connection errors gracefully
            # Return dict with 0.0 so retry logic can work
            return {
                "symbol": symbol,
                "bid": 0.0,
                "ask": 0.0,
                "last": 0.0,
                "price": 0.0,
                "close": 0.0,
                "timestamp": None
            }

    def is_market_open(self) -> bool:
        """Check if market is open. Caches result for 30 seconds."""
        import time
        import logging
        logger = logging.getLogger(__name__)
        now = time.time()
        # Reduce cache TTL to 30 seconds to catch market open faster
        cache_ttl = 30.0
        # Only use cache if it's fresh and was a successful check (not an error)
        if self._market_open_cache is not None and (now - self._market_open_cache_ts) <= cache_ttl:
            return self._market_open_cache
        
        # Cache expired or doesn't exist - fetch fresh status
        try:
            is_open = self.client.is_market_open()
            # Only cache successful API responses
            self._market_open_cache = is_open
            self._market_open_cache_ts = now
            logger.info(f"Market status check: is_open={is_open}")
            return is_open
        except Exception as e:
            # Log the error but don't cache the result - will retry next time
            logger.warning(f"Market status check failed: {e}. Will retry on next check.")
            # If we have a recent cached value (even if expired), use it as fallback
            if self._market_open_cache is not None and (now - self._market_open_cache_ts) <= 300:  # Use cache up to 5 min old
                logger.info(f"Using cached market status: {self._market_open_cache}")
                return self._market_open_cache
            # If no cache or very old, assume open to avoid blocking trades unnecessarily
            # This is safer than assuming closed when API fails
            logger.warning("No market status available, assuming open to avoid blocking trades")
            return True
    
    def fetch_ticker_last(self, symbol: str) -> float:
        """
        Fetch last price for a symbol with robust fallbacks.
        When market is closed, still returns last known price (from daily bar close).
        Returns 0.0 only if truly unavailable.
        """
        import math
        import logging
        logger = logging.getLogger(__name__)
        
        market_open = self.is_market_open()
        
        # Try main ticker fetch first
        try:
            t = self.fetch_ticker(symbol)
            last = float(t.get("last") or 0.0)
            # If last is 0, try alternative fields
            if last <= 0:
                last = float(t.get("price") or t.get("close") or 0.0)
            
            # Strict validation: must be positive and finite
            if last > 0 and math.isfinite(last):
                return last
        except Exception as e:
            logger.debug(f"Primary ticker fetch failed for {symbol}: {e}")
        
        # ALWAYS try snapshot fallback (not just when market closed)
        try:
            if hasattr(self.client, "get_snapshots"):
                snaps = self.client.get_snapshots([symbol])
                sym_snap = (snaps.get("snapshots") or {}).get(symbol, {})
            elif hasattr(self.client, "_request"):
                snap = self.client._request("GET", "/v2/stocks/snapshots",
                                            params={"symbols": symbol},
                                            base_url=self.client.data_url)
                sym_snap = (snap.get("snapshots") or snap or {}).get(symbol, {}) or (snap if isinstance(snap, dict) and symbol in snap else {})
            else:
                sym_snap = {}
            
            # Try latestTrade price first
            latest_trade = sym_snap.get("latestTrade", {})
            trade_price = float(latest_trade.get("p", 0))
            if trade_price > 0 and math.isfinite(trade_price):
                logger.debug(f"Using latestTrade for {symbol}: {trade_price}")
                return trade_price
            
            # Try daily bar close
            daily_bar = sym_snap.get("dailyBar", {})
            close_price = float(daily_bar.get("c", 0))
            if close_price > 0 and math.isfinite(close_price):
                logger.debug(f"Using daily bar close for {symbol}: {close_price}")
                return close_price
            
            # Try previous daily bar
            prev_daily = sym_snap.get("prevDailyBar", {})
            prev_close = float(prev_daily.get("c", 0))
            if prev_close > 0 and math.isfinite(prev_close):
                logger.debug(f"Using prev daily bar for {symbol}: {prev_close}")
                return prev_close
                
        except Exception as snap_err:
            logger.debug(f"Snapshot fallback failed for {symbol}: {snap_err}")

        # Fallback: use latest OHLCV bar close when ticker/snapshot return 0
        try:
            bars = self.client.get_ohlcv(symbol, "1d", limit=2)
            if bars and len(bars) >= 1 and len(bars[-1]) >= 5:
                close = float(bars[-1][4])
                if close > 0 and math.isfinite(close):
                    logger.debug(f"Using OHLCV close for {symbol}: {close}")
                    return close
        except Exception as bar_err:
            logger.debug(f"OHLCV fallback for {symbol}: {bar_err}")

        # If we get here, price is truly unavailable
        if market_open:
            logger.warning(f"Price fetch returned 0.0 for {symbol} (market status: open) - all endpoints returned 0")
        else:
            logger.info(f"Price fetch returned 0.0 for {symbol} (market closed) - will use last known when available")
        return 0.0

    def fetch_ohlcv(self, symbol: str, timeframe: str = "1m", limit: int = 100) -> List[List[float]]:
        # Kraken: [[ts, o, h, l, c, v], ...] (ts in seconds?)
        # AlpacaClient.get_ohlcv returns [ts_ms, o, h, l, c, v].
        # We need to check what BotRunner expects.
        # Kraken usually returns seconds for `ts`.
        # `bot_manager.py` line 2353 in `api_bot_ohlc`: `ts_ms = int(row[0])`. 
        # Wait, if Kraken returns seconds, naming it `ts_ms` is confusing?
        # Or does Kraken return ms?
        # CCXT default is ms. `KrakenClient` likely wraps ccxt or is custom.
        # Let's assume ms to be safe, or check `KrakenClient`.
        # But based on `ts_ms // 1000` usage in `worker_api.py`, it expects MS.
        return self.client.get_ohlcv(symbol, timeframe, limit)

    def fetch_balance(self) -> Dict[str, Any]:
        """
        Kraken format:
        {'total': {'USD': 100.0, 'AAPL': 10}, 
         'free': {'USD': 50.0, 'AAPL': 10}, 
         'used': {'USD': 50.0, 'AAPL': 0}}
        """
        acct = self.client.get_account()
        cash = float(acct.get("cash", 0))
        buying_power = float(acct.get("buying_power", 0))
        # buying_power approx cash for non-margin?
        # Alpaca separates cash and positions.
        
        positions = self.client.get_positions()
        
        total = {"USD": cash} # Actually cash is free? No, cash + position value = equity.
        # But Kraken `total['USD']` is usually checking cash balance?
        # `balance_free_total` uses `total['USD']` as total cash available?
        
        free = {"USD": cash} # simplified
        used = {"USD": 0.0} # simplified
        
        for p in positions:
            sym = p.get("symbol")
            qty = float(p.get("qty", 0))
            total[sym] = qty
            free[sym] = qty # stocks are fully "free" unless sell order locked?
            # Alpaca manages locking differently.
            used[sym] = 0.0
            
        return {"total": total, "free": free, "used": used}

    def fetch_order_book(self, symbol: str, limit: Optional[int] = None) -> Dict[str, Any]:
        """
        Build order-book-like structure from Alpaca latest quote for slippage gating.
        Stocks use NBBO; we approximate depth from snapshot/quote.
        Returns {"bids": [[price, qty], ...], "asks": [[price, qty], ...]}.
        """
        try:
            ticker = self.client.get_ticker(symbol)
            if not ticker:
                return {"bids": [], "asks": []}
            bid = float(ticker.get("bid") or 0)
            ask = float(ticker.get("ask") or 0)
            last = float(ticker.get("last") or 0)
            if bid <= 0 and ask <= 0:
                bid = ask = last
            # Approximate size: $50k depth per level for liquid stocks
            size = 50000.0 / (last if last > 0 else 1.0)
            bids = [[bid, size]] if bid > 0 else []
            asks = [[ask, size]] if ask > 0 else []
            return {"bids": bids, "asks": asks}
        except Exception as e:
            logger.debug("fetch_order_book %s failed: %s", symbol, e)
            return {"bids": [], "asks": []}

    def fetch_open_orders(self, symbol: Optional[str] = None) -> List[Dict[str, Any]]:
        # Kraken format list of dicts.
        orders = self.client.get_orders(status="open")
        out = []
        for o in orders:
            if symbol and o.get("symbol") != symbol:
                continue
            out.append({
                "id": o.get("id"),
                "symbol": o.get("symbol"),
                "side": o.get("side"),
                "type": o.get("type"),
                "price": float(o.get("limit_price") or 0.0),
                "amount": float(o.get("qty") or 0.0),
                "filled": float(o.get("filled_qty") or 0.0),
                "remaining": float(o.get("qty") or 0.0) - float(o.get("filled_qty") or 0.0),
                "status": o.get("status"),
                "client_order_id": o.get("client_order_id"),
                "info": o,
            })
        return out

    def cancel_order(self, order_id: str, symbol: Optional[str] = None) -> Any:
        return self.client.cancel_order(order_id)

    def cancel_all_open_orders(self, symbol: Optional[str] = None) -> Any:
        # Alpaca cancel_all cancels EVERYTHING.
        # Optimally we filter by symbol if needed, but Alpaca SDK might not support symbol filter on cancel_all?
        # self.client.cancel_all_orders() cancels all.
        # If we need symbol specific, we fetch and cancel individually.
        if symbol:
            orders = self.fetch_open_orders(symbol)
            for o in orders:
                self.cancel_order(o["id"])
        else:
            return self.client.cancel_all_orders()

    def create_limit_buy_base(
        self, symbol: str, amount: float, price: float, client_order_id: Optional[str] = None
    ) -> Dict[str, Any]:
        res = self.client.place_limit_order(symbol, amount, price, "buy")
        return {"id": res.get("id"), "info": res}

    def create_limit_sell_base(
        self, symbol: str, amount: float, price: float, client_order_id: Optional[str] = None
    ) -> Dict[str, Any]:
        res = self.client.place_limit_order(symbol, amount, price, "sell")
        return {"id": res.get("id"), "info": res}

    def create_market_buy_quote(
        self, symbol: str, quote_amount: float, client_order_id: Optional[str] = None
    ) -> Dict[str, Any]:
        # Market buy by QUOTE amount (spend $100).
        # Alpaca supports `notional` for market orders.
        # My AlpacaClient wrapper `place_market_order` takes `qty` (shares).
        # Does it support notional?
        # Looking at `alpaca_client.py`:
        # `data = {"symbol": symbol, "qty": str(qty), ...}` 
        # It doesn't seem to support `notional` in the wrapper yet.
        # I might need to approximate shares: qty = quote_amount / last_price.
        
        last = self.fetch_ticker_last(symbol)
        if last <= 0:
            raise ValueError(f"Cannot calculate qty for market buy, no price for {symbol}")
        
        # Alpaca supports fractional shares? Client says "qty can be fractional".
        qty = quote_amount / last
        # Truncate to safe precision?
        # Alpaca handles high precision.
        
        res = self.client.place_market_order(symbol, qty, "buy")
        return {"id": res.get("id"), "info": res}

    def create_market_sell_base(
        self, symbol: str, amount: float, client_order_id: Optional[str] = None
    ) -> Dict[str, Any]:
        res = self.client.place_market_order(symbol, amount, "sell")
        return {"id": res.get("id"), "info": res}

    def fetch_my_trades(self, symbol: Optional[str] = None, limit: int = 50) -> List[Dict[str, Any]]:
        # This is used for deal metrics (avg entry calculation).
        # Alpaca doesn't have a direct "my trades" simple endpoint in the wrapper?
        # We can use `get_orders(status="all")` and filter for filled?
        # Or `get_account_activities(activity_type="FILL")` (not in wrapper).
        
        # Fallback: Use `get_orders` closed/all and look for fills.
        # This is inefficient but functional for now.
        orders = self.client.get_orders(status="closed", limit=limit) # limit might need to be higher
        trades = []
        for o in orders:
            if symbol and o.get("symbol") != symbol:
                continue
            if o.get("status") in ("filled", "partially_filled"):
                # Construct trade-like object
                # checking filled_at for timestamp (ISO string)
                filled_at = o.get("filled_at")
                ts = 0
                if filled_at:
                    try:
                        # Parse ISO to ms timestamp
                        # e.g. 2023-10-25T10:00:00.123456Z
                        import datetime
                        # Python 3.7+ fromisoformat handles some ISO, but safely:
                        dt = datetime.datetime.fromisoformat(filled_at.replace("Z", "+00:00"))
                        ts = int(dt.timestamp() * 1000)
                    except Exception as e:
                        logger.debug("Timestamp parse failed: %s", e)
                
                trades.append({
                    "id": o.get("id"),
                    "order_id": o.get("id"),
                    "symbol": o.get("symbol"),
                    "side": o.get("side"),
                    "amount": float(o.get("filled_qty") or 0),
                    "price": float(o.get("filled_avg_price") or o.get("limit_price") or 0), # avg fill price is best
                    "cost": float(o.get("filled_qty") or 0) * float(o.get("filled_avg_price") or 0),
                    "timestamp": ts,
                    "datetime": filled_at
                })
        return trades
