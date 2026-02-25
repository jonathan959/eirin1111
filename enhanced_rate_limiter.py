"""
Enhanced Rate Limiter for Alpaca API
Implements token bucket algorithm with exponential backoff and request queuing
"""
import logging
import threading
import time
from typing import Optional, Callable, Any
from collections import deque
from datetime import datetime, timedelta
import functools

logger = logging.getLogger(__name__)


class RateLimiter:
    """Token bucket rate limiter with exponential backoff."""
    
    def __init__(self, max_requests: int = 180, time_window: int = 60):
        self.max_requests = max_requests
        self.time_window = time_window
        self.tokens = max_requests
        self.last_refill = time.time()
        self.request_queue = deque()
        self.queue_lock = threading.Lock()
        self.total_requests = 0
        self.rejected_requests = 0
        self.queued_requests = 0
        self.consecutive_failures = 0
        self.backoff_until = None
    
    def _refill_tokens(self):
        now = time.time()
        elapsed = now - self.last_refill
        if elapsed >= self.time_window:
            self.tokens = self.max_requests
            self.last_refill = now
        else:
            tokens_to_add = (elapsed / self.time_window) * self.max_requests
            self.tokens = min(self.max_requests, self.tokens + tokens_to_add)
            self.last_refill = now
    
    def acquire(self, blocking: bool = True, timeout: Optional[float] = None) -> bool:
        start_time = time.time()
        while True:
            if self.backoff_until and time.time() < self.backoff_until:
                if not blocking:
                    return False
                wait_time = self.backoff_until - time.time()
                time.sleep(min(wait_time, 1.0))
                continue
            with self.queue_lock:
                self._refill_tokens()
                if self.tokens >= 1:
                    self.tokens -= 1
                    self.total_requests += 1
                    self.consecutive_failures = 0
                    return True
                if not blocking:
                    self.rejected_requests += 1
                    return False
                if timeout and (time.time() - start_time) >= timeout:
                    self.rejected_requests += 1
                    return False
            time.sleep(0.1)
    
    def enter_backoff(self, error_code: Optional[int] = None):
        self.consecutive_failures += 1
        backoff_seconds = min(2 ** self.consecutive_failures, 60)
        self.backoff_until = time.time() + backoff_seconds
        logger.warning("Rate limit hit (failure #%s), backing off %ss", self.consecutive_failures, backoff_seconds)
    
    def reset_backoff(self):
        self.consecutive_failures = 0
        self.backoff_until = None
    
    def get_stats(self) -> dict:
        with self.queue_lock:
            self._refill_tokens()
            return {
                "total_requests": self.total_requests,
                "rejected_requests": self.rejected_requests,
                "available_tokens": int(self.tokens),
                "max_tokens": self.max_requests,
                "in_backoff": self.backoff_until is not None,
                "consecutive_failures": self.consecutive_failures
            }


class SmartRateLimiter:
    """Intelligent rate limiter with per-endpoint limits."""
    
    def __init__(self, base_limit: int = 180):
        self.base_limit = base_limit
        self.current_limit = base_limit
        self.endpoint_limiters: dict = {}
        self.global_limiter = RateLimiter(max_requests=base_limit, time_window=60)
    
    def get_limiter(self, endpoint: str) -> RateLimiter:
        if endpoint not in self.endpoint_limiters:
            endpoint_limit = max(10, self.current_limit // 10)
            self.endpoint_limiters[endpoint] = RateLimiter(max_requests=endpoint_limit, time_window=60)
        return self.endpoint_limiters[endpoint]
    
    def acquire(self, endpoint: str, priority: int = 5) -> bool:
        if not self.global_limiter.acquire(blocking=False):
            if priority >= 8:
                return self.global_limiter.acquire(blocking=True, timeout=10)
            return False
        endpoint_limiter = self.get_limiter(endpoint)
        return endpoint_limiter.acquire(blocking=(priority >= 7), timeout=5)
    
    def adjust_limits(self, decrease: bool = False):
        if decrease:
            self.current_limit = max(50, int(self.current_limit * 0.9))
        else:
            self.current_limit = min(self.base_limit, int(self.current_limit * 1.05))
        self.global_limiter.max_requests = self.current_limit
    
    def enter_backoff(self, error_code: int = 429):
        self.global_limiter.enter_backoff(error_code)
    
    def get_all_stats(self) -> dict:
        return {
            "global": self.global_limiter.get_stats(),
            "current_limit": self.current_limit,
            "endpoints": {k: v.get_stats() for k, v in self.endpoint_limiters.items()}
        }
