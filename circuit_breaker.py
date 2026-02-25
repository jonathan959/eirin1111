"""
Circuit Breaker: API failure handling and auto-recovery.

- If 3 API calls fail in a row: pause that bot for 5 minutes
- If exchange errors persist >15 min: emergency stop all bots
- Auto-retry with exponential backoff
- Log all errors to error_log table
"""
import logging
import os
import threading
import time
from typing import Any, Callable, Dict, Optional

logger = logging.getLogger(__name__)

CIRCUIT_BREAKER_THRESHOLD = int(os.getenv("CIRCUIT_BREAKER_THRESHOLD", "3"))
BOT_PAUSE_MINUTES = int(os.getenv("CIRCUIT_BREAKER_BOT_PAUSE_MIN", "5"))
EMERGENCY_STOP_MINUTES = int(os.getenv("CIRCUIT_BREAKER_EMERGENCY_MIN", "15"))
RETRY_BASE_SLEEP = float(os.getenv("CIRCUIT_BREAKER_RETRY_BASE", "0.5"))
RETRY_MAX_SLEEP = float(os.getenv("CIRCUIT_BREAKER_RETRY_MAX", "32.0"))
RETRY_ATTEMPTS = int(os.getenv("CIRCUIT_BREAKER_RETRY_ATTEMPTS", "4"))

# Per-bot failure counts (bot_id -> list of failure timestamps)
_bot_failures: Dict[int, list] = {}
_global_failures: list = []
_lock = threading.RLock()
_emergency_stop_until: float = 0.0


def _prune_old(failures: list, max_age_sec: int) -> list:
    cutoff = time.time() - max_age_sec
    return [t for t in failures if t >= cutoff]


def record_api_failure(bot_id: Optional[int] = None, source: str = "api") -> None:
    """Record an API failure. May trigger circuit breaker."""
    global _emergency_stop_until
    now = time.time()
    with _lock:
        if bot_id is not None:
            _bot_failures.setdefault(bot_id, []).append(now)
            _bot_failures[bot_id] = _prune_old(_bot_failures[bot_id], EMERGENCY_STOP_MINUTES * 60)
        _global_failures.append(now)
        _global_failures[:] = _prune_old(_global_failures, EMERGENCY_STOP_MINUTES * 60)

    try:
        from db import log_error
        log_error(source, "api_failure", f"Bot {bot_id}" if bot_id else "Global", bot_id=bot_id)
    except Exception:
        pass


def is_bot_circuit_open(bot_id: int) -> bool:
    """True if bot should be paused due to circuit breaker."""
    with _lock:
        failures = _bot_failures.get(bot_id, [])
        failures = _prune_old(failures, BOT_PAUSE_MINUTES * 60)
        return len(failures) >= CIRCUIT_BREAKER_THRESHOLD


def is_emergency_stop_active() -> bool:
    """True if global emergency stop is active (all bots should stop)."""
    return time.time() < _emergency_stop_until


def get_bot_pause_until(bot_id: int) -> Optional[float]:
    """Unix timestamp until when bot is paused. None if not paused."""
    with _lock:
        failures = _bot_failures.get(bot_id, [])
        failures = _prune_old(failures, BOT_PAUSE_MINUTES * 60)
        if len(failures) >= CIRCUIT_BREAKER_THRESHOLD and failures:
            return max(failures) + BOT_PAUSE_MINUTES * 60
    return None


def check_and_trigger_emergency() -> bool:
    """
    If global failures >= threshold in last EMERGENCY_STOP_MINUTES, set emergency stop.
    Returns True if emergency was triggered.
    """
    global _emergency_stop_until
    with _lock:
        recent = _prune_old(_global_failures, EMERGENCY_STOP_MINUTES * 60)
        if len(recent) >= CIRCUIT_BREAKER_THRESHOLD * 3:  # e.g. 9+ failures
            _emergency_stop_until = time.time() + EMERGENCY_STOP_MINUTES * 60
            try:
                from db import log_error
                log_error("circuit_breaker", "emergency_stop", f"{len(recent)} failures in {EMERGENCY_STOP_MINUTES}min")
            except Exception:
                pass
            return True
    return False


def record_api_success(bot_id: Optional[int] = None) -> None:
    """Call on successful API operation. Resets bot circuit for faster recovery."""
    with _lock:
        if bot_id is not None:
            _bot_failures.pop(bot_id, None)


def reset_bot_circuit(bot_id: int) -> None:
    """Reset circuit for a bot after successful call."""
    record_api_success(bot_id)


def reset_global_circuit() -> None:
    """Reset global failure count (e.g. after recovery)."""
    global _emergency_stop_until
    with _lock:
        _global_failures.clear()
        _emergency_stop_until = 0.0


def retry_with_backoff(
    fn: Callable,
    *args,
    attempts: int = RETRY_ATTEMPTS,
    base_sleep: float = RETRY_BASE_SLEEP,
    max_sleep: float = RETRY_MAX_SLEEP,
    bot_id: Optional[int] = None,
    source: str = "api",
    **kwargs,
) -> Any:
    """
    Execute fn with exponential backoff. On final failure, records to circuit breaker.
    """
    last_err = None
    for i in range(attempts):
        try:
            return fn(*args, **kwargs)
        except Exception as e:
            last_err = e
            if i < attempts - 1:
                sleep_s = min(max_sleep, base_sleep * (2 ** i))
                time.sleep(sleep_s)
    record_api_failure(bot_id=bot_id, source=source)
    raise last_err


def get_status() -> Dict[str, Any]:
    """Status for health endpoint."""
    with _lock:
        bot_paused = {bid: get_bot_pause_until(bid) for bid in list(_bot_failures)}
        bot_paused = {k: v for k, v in bot_paused.items() if v is not None}
        recent = _prune_old(list(_global_failures), EMERGENCY_STOP_MINUTES * 60)
    return {
        "emergency_stop": is_emergency_stop_active(),
        "emergency_until": _emergency_stop_until if _emergency_stop_until > 0 else None,
        "bots_paused": bot_paused,
        "global_failures_recent": len(recent),
    }
