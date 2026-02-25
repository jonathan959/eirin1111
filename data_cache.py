"""
Data Cache Manager - Reduces Alpaca REST API calls
"""
import logging
import threading
import time
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class CacheEntry:
    data: Any
    timestamp: datetime
    ttl_seconds: int
    
    def is_expired(self) -> bool:
        return datetime.now() > (self.timestamp + timedelta(seconds=self.ttl_seconds))


class DataCache:
    """Thread-safe cache for Alpaca data. Max 500 entries to prevent memory growth."""
    MAX_ENTRIES = 500

    def __init__(self):
        self._cache: Dict[str, CacheEntry] = {}
        self._lock = threading.RLock()
        self._stats = {"hits": 0, "misses": 0, "api_calls_saved": 0}

    def _evict_if_needed(self):
        """Evict oldest expired entries, or newest if over limit (prevents unbounded growth)."""
        if len(self._cache) <= self.MAX_ENTRIES:
            return
        now = datetime.now()
        to_del = []
        for k, entry in self._cache.items():
            if entry.is_expired():
                to_del.append(k)
        for k in to_del:
            del self._cache[k]
        while len(self._cache) > self.MAX_ENTRIES:
            # Evict oldest by timestamp
            oldest = min(self._cache.items(), key=lambda x: x[1].timestamp)
            del self._cache[oldest[0]]

    def get(self, key: str) -> Optional[Any]:
        with self._lock:
            entry = self._cache.get(key)
            if entry is None:
                self._stats["misses"] += 1
                return None
            if entry.is_expired():
                del self._cache[key]
                self._stats["misses"] += 1
                return None
            self._stats["hits"] += 1
            self._stats["api_calls_saved"] += 1
            return entry.data
    
    def set(self, key: str, data: Any, ttl_seconds: int = 300):
        with self._lock:
            self._cache[key] = CacheEntry(data=data, timestamp=datetime.now(), ttl_seconds=ttl_seconds)
            self._evict_if_needed()
    
    def invalidate(self, key: str):
        with self._lock:
            self._cache.pop(key, None)
    
    def invalidate_pattern(self, pattern: str):
        with self._lock:
            for k in list(self._cache.keys()):
                if pattern in k:
                    del self._cache[k]
    
    def clear(self):
        with self._lock:
            self._cache.clear()
    
    def get_stats(self) -> Dict:
        with self._lock:
            total = self._stats["hits"] + self._stats["misses"]
            hit_rate = (self._stats["hits"] / total * 100) if total > 0 else 0
            return {
                "hits": self._stats["hits"],
                "misses": self._stats["misses"],
                "hit_rate": f"{hit_rate:.2f}%",
                "api_calls_saved": self._stats["api_calls_saved"],
                "cached_items": len(self._cache)
            }


class CachedAlpacaClient:
    """Wrapper around alpaca-py TradingClient with caching."""
    
    def __init__(self, alpaca_client, cache: Optional[DataCache] = None):
        self.client = alpaca_client
        self.cache = cache or DataCache()
    
    def _retry_api(self, fn, *args, max_attempts: int = 3, **kwargs):
        """Retry API call with exponential backoff (1s, 2s, 4s)."""
        import time
        last_err = None
        for attempt in range(max_attempts):
            try:
                return fn(*args, **kwargs)
            except Exception as e:
                last_err = e
                if attempt < max_attempts - 1:
                    sleep_sec = 2 ** attempt
                    time.sleep(sleep_sec)
        raise last_err

    def get_account(self, force_refresh: bool = False) -> Any:
        cache_key = "account"
        if not force_refresh:
            cached = self.cache.get(cache_key)
            if cached is not None:
                return cached
        account = self._retry_api(self.client.get_account)
        self.cache.set(cache_key, account, ttl_seconds=300)
        return account

    def get_all_positions(self, force_refresh: bool = False) -> List[Any]:
        cache_key = "positions_all"
        if not force_refresh:
            cached = self.cache.get(cache_key)
            if cached is not None:
                return cached
        positions = list(self._retry_api(self.client.get_all_positions))
        self.cache.set(cache_key, positions, ttl_seconds=120)
        for pos in positions:
            self.cache.set(f"position_{pos.symbol}", pos, ttl_seconds=120)
        return positions
    
    def get_open_position(self, symbol: str, force_refresh: bool = False) -> Optional[Any]:
        cache_key = f"position_{symbol}"
        if not force_refresh:
            cached = self.cache.get(cache_key)
            if cached is not None:
                return cached
        try:
            position = self.client.get_open_position(symbol)
            self.cache.set(cache_key, position, ttl_seconds=120)
            return position
        except Exception:
            return None
    
    def get_orders(self, status: str = "open", force_refresh: bool = False) -> List[Any]:
        cache_key = f"orders_{status}"
        if not force_refresh:
            cached = self.cache.get(cache_key)
            if cached is not None:
                return cached
        try:
            from alpaca.trading.requests import GetOrdersRequest
            filt = GetOrdersRequest(status=status)
            orders = list(self.client.get_orders(filter=filt))
        except Exception:
            orders = list(self.client.get_orders())
        self.cache.set(cache_key, orders, ttl_seconds=10)
        return orders
    
    def submit_order(self, *args, **kwargs):
        result = self.client.submit_order(*args, **kwargs)
        self.cache.invalidate_pattern("orders_")
        self.cache.invalidate_pattern("position_")
        self.cache.invalidate("positions_all")
        return result
    
    def cancel_order(self, order_id: str):
        result = self.client.cancel_order_by_id(order_id)
        self.cache.invalidate_pattern("orders_")
        return result
    
    def close_position(self, symbol: str):
        result = self.client.close_position(symbol)
        self.cache.invalidate(f"position_{symbol}")
        self.cache.invalidate("positions_all")
        return result
