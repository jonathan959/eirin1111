"""
Alpaca API rate limiter to prevent 429 errors. (123.md Fix 1)
Token bucket + request queue. Exponential backoff on 429.
"""
import logging
import os
import time
import threading
from collections import deque
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

# Use 90% of Alpaca free tier limits
_REQUESTS_PER_MINUTE = int(os.getenv("ALPACA_RPM_LIMIT", "180"))
_REQUESTS_PER_DAY = int(os.getenv("ALPACA_DAILY_LIMIT", "9000"))


class AlpacaRateLimiter:
    """Token bucket rate limiter for Alpaca API."""

    def __init__(self, requests_per_minute=None, requests_per_day=None):
        self.rpm_limit = requests_per_minute or _REQUESTS_PER_MINUTE
        self.daily_limit = requests_per_day or _REQUESTS_PER_DAY
        self.tokens = float(self.rpm_limit)
        self.max_tokens = float(self.rpm_limit)
        self.last_refill = time.time()
        self.daily_requests = 0
        self.daily_reset_time = datetime.now() + timedelta(days=1)
        self.lock = threading.Lock()
        self.consecutive_429s = 0
        self.backoff_until = None

    def acquire(self, request_name="unknown"):
        """Acquire permission to make API request. Blocks if limit would be exceeded."""
        with self.lock:
            if self.backoff_until and time.time() < self.backoff_until:
                wait_time = self.backoff_until - time.time()
                logger.warning("Alpaca rate limit: in backoff, waiting %.1fs", wait_time)
                time.sleep(wait_time)
                self.backoff_until = None

            if datetime.now() >= self.daily_reset_time:
                self.daily_requests = 0
                self.daily_reset_time = datetime.now() + timedelta(days=1)
                logger.info("Alpaca daily request counter reset")

            if self.daily_requests >= self.daily_limit:
                raise Exception(
                    f"Alpaca daily request limit reached ({self.daily_limit}). "
                    f"Resets at {self.daily_reset_time}"
                )

            now = time.time()
            time_passed = now - self.last_refill
            new_tokens = time_passed * (self.max_tokens / 60.0)
            self.tokens = min(self.max_tokens, self.tokens + new_tokens)
            self.last_refill = now

            if self.tokens < 1:
                wait_time = (1 - self.tokens) / (self.max_tokens / 60.0)
                logger.debug("Alpaca rate limit: waiting %.2fs for token", wait_time)
                time.sleep(wait_time)
                self.tokens = 1

            self.tokens -= 1
            self.daily_requests += 1
            logger.debug(
                "Alpaca request %s: tokens=%.1f daily=%s/%s",
                request_name, self.tokens, self.daily_requests, self.daily_limit,
            )

    def handle_429_error(self):
        """Exponential backoff on 429."""
        self.consecutive_429s += 1
        backoff_seconds = min(2 ** self.consecutive_429s, 300)
        self.backoff_until = time.time() + backoff_seconds
        logger.error(
            "Alpaca 429 #%s: backing off %ss",
            self.consecutive_429s, backoff_seconds,
        )

    def reset_backoff(self):
        """Reset backoff on successful request."""
        self.consecutive_429s = 0


alpaca_rate_limiter = AlpacaRateLimiter()


def rate_limited_request(func):
    """Decorator to wrap Alpaca API calls with rate limiting and 429 retry."""
    def wrapper(*args, **kwargs):
        alpaca_rate_limiter.acquire(request_name=func.__name__)
        try:
            result = func(*args, **kwargs)
            alpaca_rate_limiter.reset_backoff()
            return result
        except Exception as e:
            err_str = str(e)
            status = getattr(e, "status_code", None)
            if (status == 429 or "429" in err_str or "rate limit" in err_str.lower()):
                logger.error("Alpaca 429 in %s: %s", func.__name__, err_str)
                alpaca_rate_limiter.handle_429_error()
                alpaca_rate_limiter.acquire(request_name=f"{func.__name__}_retry")
                try:
                    return func(*args, **kwargs)
                except Exception as retry_err:
                    alpaca_rate_limiter.reset_backoff()
                    raise
            raise

    return wrapper


# Alias for production-ready fix spec
alpaca_rate_limit = rate_limited_request
