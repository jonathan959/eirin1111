"""
WebSocket Manager for Alpaca Streaming Data
Replaces REST API polling with real-time WebSocket streaming.
"""
import asyncio
import logging
from typing import Any, Dict, List, Callable, Optional, Set
from collections import defaultdict
import threading

logger = logging.getLogger(__name__)

# Optional alpaca-py imports
try:
    from alpaca.data.live import StockDataStream
    from alpaca.data.models import Trade, Quote, Bar
    ALPACA_WS_AVAILABLE = True
except ImportError:
    ALPACA_WS_AVAILABLE = False
    StockDataStream = None
    Trade = Quote = Bar = None


class WebSocketManager:
    """Manages WebSocket connections for real-time market data."""
    
    def __init__(self, api_key: str, secret_key: str, paper: bool = True):
        self.api_key = api_key
        self.secret_key = secret_key
        self.paper = paper
        self.stock_stream = None
        self._lock = threading.RLock()
        self._latest_trades: Dict[str, Any] = {}
        self._latest_quotes: Dict[str, Any] = {}
        self._latest_bars: Dict[str, Any] = {}
        self._subscribed_symbols: Set[str] = set()
        self._callbacks: Dict[str, List[Callable]] = defaultdict(list)
        self._running = False
        self._thread: Optional[threading.Thread] = None
        
        if ALPACA_WS_AVAILABLE:
            self.stock_stream = StockDataStream(api_key, secret_key, raw_data=False)
    
    def start(self):
        if not ALPACA_WS_AVAILABLE or not self.stock_stream:
            return
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(target=self._run_streams, daemon=True)
        self._thread.start()
    
    def stop(self):
        self._running = False
        if self._thread:
            self._thread.join(timeout=5)
    
    def subscribe_trades(self, symbols: List[str], callback: Optional[Callable] = None):
        if not self.stock_stream:
            return
        for symbol in symbols:
            self._subscribed_symbols.add(symbol)
            if callback:
                self._callbacks[f"trade_{symbol}"].append(callback)
            async def trade_handler(trade, s=symbol):
                with self._lock:
                    self._latest_trades[s] = trade
                for cb in self._callbacks.get(f"trade_{s}", []):
                    try:
                        cb(trade)
                    except Exception as e:
                        logger.debug("Trade callback error: %s", e)
            self.stock_stream.subscribe_trades(trade_handler, symbol)
    
    def subscribe_quotes(self, symbols: List[str], callback: Optional[Callable] = None):
        if not self.stock_stream:
            return
        for symbol in symbols:
            self._subscribed_symbols.add(symbol)
            if callback:
                self._callbacks[f"quote_{symbol}"].append(callback)
            async def quote_handler(quote, s=symbol):
                with self._lock:
                    self._latest_quotes[s] = quote
                for cb in self._callbacks.get(f"quote_{s}", []):
                    try:
                        cb(quote)
                    except Exception as e:
                        logger.debug("Quote callback error: %s", e)
            self.stock_stream.subscribe_quotes(quote_handler, symbol)
    
    def subscribe_bars(self, symbols: List[str], callback: Optional[Callable] = None):
        if not self.stock_stream:
            return
        for symbol in symbols:
            self._subscribed_symbols.add(symbol)
            if callback:
                self._callbacks[f"bar_{symbol}"].append(callback)
            async def bar_handler(bar, s=symbol):
                with self._lock:
                    self._latest_bars[s] = bar
                for cb in self._callbacks.get(f"bar_{s}", []):
                    try:
                        cb(bar)
                    except Exception as e:
                        logger.debug("Bar callback error: %s", e)
            self.stock_stream.subscribe_bars(bar_handler, symbol)
    
    def _run_streams(self):
        if not self.stock_stream:
            return
        loop = None
        try:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            loop.run_until_complete(self.stock_stream._run_forever())
        except Exception as e:
            logger.error("WebSocket stream error: %s", e)
        finally:
            if loop is not None:
                try:
                    loop.close()
                except Exception:
                    pass
    
    def get_latest_price(self, symbol: str) -> Optional[float]:
        with self._lock:
            trade = self._latest_trades.get(symbol)
            if trade:
                return float(trade.price)
            quote = self._latest_quotes.get(symbol)
            if quote:
                return float(quote.bid_price) if hasattr(quote, 'bid_price') else None
            bar = self._latest_bars.get(symbol)
            if bar:
                return float(bar.close) if hasattr(bar, 'close') else None
        return None
    
    def get_latest_quote(self, symbol: str):
        with self._lock:
            return self._latest_quotes.get(symbol)
    
    def get_subscribed_symbols(self) -> List[str]:
        return list(self._subscribed_symbols)
    
    def unsubscribe(self, symbol: str):
        self._subscribed_symbols.discard(symbol)
        for k in list(self._callbacks.keys()):
            if symbol in k:
                del self._callbacks[k]
    
    def get_stats(self) -> Dict:
        with self._lock:
            return {
                "running": self._running,
                "subscribed_symbols": len(self._subscribed_symbols),
                "cached_trades": len(self._latest_trades),
                "cached_quotes": len(self._latest_quotes),
                "cached_bars": len(self._latest_bars),
            }
