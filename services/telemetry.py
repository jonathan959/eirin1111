"""Simple in-memory telemetry ring buffer for recent backend errors.

FastAPI / Uvicorn log lines are ephemeral; this module captures a
small rolling buffer of WARN+ERROR records for display on the
dashboard's "All Systems" widget and via /api/telemetry/recent_errors.

Thread-safe: backed by a collections.deque under a lock.
"""
from __future__ import annotations

import logging
import threading
import time
import traceback
from collections import deque
from typing import Any, Dict, Iterable, List, Optional

MAX_BUFFER = 100

_lock = threading.Lock()
_buffer: "deque[Dict[str, Any]]" = deque(maxlen=MAX_BUFFER)


class _TelemetryLogHandler(logging.Handler):
    """Logging handler that captures WARN+ records into the ring."""

    def emit(self, record: logging.LogRecord) -> None:  # noqa: D401
        try:
            if record.levelno < logging.WARNING:
                return
            exc_info = None
            if record.exc_info:
                try:
                    exc_info = "".join(traceback.format_exception(*record.exc_info))
                except Exception:
                    exc_info = None
            record_dict: Dict[str, Any] = {
                "ts": int(time.time()),
                "level": record.levelname,
                "logger": record.name,
                "message": record.getMessage(),
                "module": record.module,
                "func": record.funcName,
                "line": record.lineno,
            }
            if exc_info:
                record_dict["exc_info"] = exc_info[:2000]
            with _lock:
                _buffer.append(record_dict)
        except Exception:
            # Never raise from a log handler.
            pass


_handler_installed = False


def install(level: int = logging.WARNING) -> None:
    """Install the telemetry handler on the root logger (idempotent)."""
    global _handler_installed
    if _handler_installed:
        return
    handler = _TelemetryLogHandler(level=level)
    handler.setFormatter(logging.Formatter("%(message)s"))
    logging.getLogger().addHandler(handler)
    _handler_installed = True


def record_error(message: str, level: str = "ERROR", **context: Any) -> None:
    """Manually record an error that didn't flow through `logging`."""
    with _lock:
        _buffer.append({
            "ts": int(time.time()),
            "level": level,
            "logger": "manual",
            "message": str(message)[:500],
            **{k: v for k, v in context.items() if k not in ("ts", "level", "message")},
        })


def recent_errors(limit: int = 50, since_ts: Optional[int] = None) -> List[Dict[str, Any]]:
    """Return a list of the most recent errors (newest first)."""
    with _lock:
        items = list(_buffer)
    items.reverse()
    if since_ts is not None:
        items = [r for r in items if r.get("ts", 0) >= int(since_ts)]
    return items[: max(0, int(limit))]


def clear() -> None:
    with _lock:
        _buffer.clear()


def stats() -> Dict[str, Any]:
    with _lock:
        items = list(_buffer)
    counts: Dict[str, int] = {}
    for r in items:
        counts[r.get("level", "UNKNOWN")] = counts.get(r.get("level", "UNKNOWN"), 0) + 1
    return {
        "total": len(items),
        "by_level": counts,
        "latest_ts": items[-1]["ts"] if items else None,
    }
