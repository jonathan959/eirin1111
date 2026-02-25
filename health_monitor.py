"""
Health Monitor: Expanded health checks for /api/health.

- Bot status (running, paused, error)
- Data freshness (last candle update)
- API connectivity (Kraken, Alpaca)
- DB performance metrics
- Circuit breaker status
- Data quality status
- Optional Prometheus /metrics endpoint
"""
import logging
import os
import time
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

ENABLE_PROMETHEUS = os.getenv("ENABLE_PROMETHEUS_METRICS", "0").strip().lower() in ("1", "true", "yes")


def get_bot_status_summary(list_bots_fn=None) -> Dict[str, Any]:
    """Count bots by state: running, paused, error."""
    try:
        if list_bots_fn is None:
            from db import list_bots
            list_bots_fn = list_bots
        bots = list_bots_fn() or []
        running = 0
        paused = 0
        enabled_not_running = 0
        for b in bots:
            enabled = int(b.get("enabled", 0)) == 1
            # We don't have direct "running" in db; last_running could indicate
            last_running = int(b.get("last_running", 0))
            if enabled:
                if last_running:
                    running += 1
                else:
                    enabled_not_running += 1
            else:
                paused += 1
        return {
            "total": len(bots),
            "running": running,
            "paused": paused,
            "enabled_not_running": enabled_not_running,
        }
    except Exception as e:
        logger.debug("get_bot_status_summary: %s", e)
        return {"total": 0, "running": 0, "paused": 0, "enabled_not_running": 0}


def get_data_freshness(market_data_router=None) -> Dict[str, Any]:
    """Last candle update time per provider. Placeholder - integrate with market_data."""
    result = {"kraken_last_candle_ts": None, "alpaca_last_candle_ts": None}
    # Could query PORT_HISTORY or cache timestamps from worker_api
    return result


def get_db_metrics() -> Dict[str, Any]:
    """DB performance: size, page count, integrity check (light)."""
    try:
        from db import DB_NAME, _conn
        import os as osmod
        path = DB_NAME
        size_bytes = osmod.path.getsize(path) if osmod.path.exists(path) else 0
        con = _conn()
        try:
            page_count = con.execute("PRAGMA page_count").fetchone()[0]
            page_size = con.execute("PRAGMA page_size").fetchone()[0]
            return {
                "size_mb": round(size_bytes / (1024 * 1024), 2),
                "pages": page_count,
                "page_size": page_size,
            }
        finally:
            con.close()
    except Exception as e:
        logger.debug("get_db_metrics: %s", e)
        return {"size_mb": 0, "pages": 0, "page_size": 0}


def get_circuit_breaker_status() -> Dict[str, Any]:
    """Circuit breaker state."""
    try:
        import circuit_breaker
        return circuit_breaker.get_status()
    except Exception as e:
        logger.debug("circuit_breaker status: %s", e)
        return {"emergency_stop": False, "bots_paused": {}}


def get_data_quality_status(minutes: int = 15) -> Dict[str, Any]:
    """Data quality: recent issue count, degraded flag."""
    try:
        from db import get_recent_data_quality_count
        count = get_recent_data_quality_count(minutes=minutes, min_severity="warning")
        return {
            "recent_issues": count,
            "degraded": count >= 5,
        }
    except Exception as e:
        logger.debug("data_quality status: %s", e)
        return {"recent_issues": 0, "degraded": False}


def get_memory_cpu_usage() -> Dict[str, Any]:
    """Best-effort memory/CPU. Cross-platform."""
    try:
        import psutil
        proc = psutil.Process()
        mem = proc.memory_info()
        return {
            "memory_rss_mb": round(mem.rss / (1024 * 1024), 1),
            "cpu_percent": proc.cpu_percent(interval=0.1) or 0,
        }
    except ImportError:
        return {"memory_rss_mb": None, "cpu_percent": None}
    except Exception as e:
        logger.debug("memory/cpu: %s", e)
        return {"memory_rss_mb": None, "cpu_percent": None}


def build_expanded_health(
    kraken_ready: bool,
    kraken_error: str,
    alpaca_paper_ready: bool,
    alpaca_live_ready: bool,
    alpaca_error: str,
    bot_manager_ready: bool,
    db_ok: bool,
    list_bots_fn=None,
    last_portfolio_ts: float = 0,
    last_reco_short_ts: float = 0,
    last_reco_long_ts: float = 0,
) -> Dict[str, Any]:
    """Build full health payload for /api/health."""
    bots = get_bot_status_summary(list_bots_fn)
    db_metrics = get_db_metrics()
    circuit = get_circuit_breaker_status()
    data_quality = get_data_quality_status()
    mem_cpu = get_memory_cpu_usage()
    data_freshness = get_data_freshness()
    data_freshness["last_portfolio_ts"] = last_portfolio_ts
    data_freshness["last_reco_short_ts"] = last_reco_short_ts
    data_freshness["last_reco_long_ts"] = last_reco_long_ts

    emergency = circuit.get("emergency_stop")
    dq_degraded = data_quality.get("degraded", False)
    healthy = db_ok and not emergency and not dq_degraded

    return {
        "ok": db_ok,
        "status": "healthy" if healthy else "degraded",
        "ts": int(time.time()),
        "kraken_ready": kraken_ready,
        "kraken_error": kraken_error or "",
        "alpaca_paper_ready": alpaca_paper_ready,
        "alpaca_live_ready": alpaca_live_ready,
        "alpaca_error": alpaca_error or "",
        "bot_manager_ready": bot_manager_ready,
        "db_ok": db_ok,
        "bots": bots,
        "db_metrics": db_metrics,
        "circuit_breaker": circuit,
        "data_quality": data_quality,
        "memory_cpu": mem_cpu,
        "data_freshness": data_freshness,
    }


def prometheus_metrics() -> Optional[str]:
    """Generate Prometheus text format if ENABLE_PROMETHEUS=1."""
    if not ENABLE_PROMETHEUS:
        return None
    lines = []
    try:
        bots = get_bot_status_summary()
        lines.append("# HELP eirin_bots_total Total bots")
        lines.append("# TYPE eirin_bots_total gauge")
        lines.append(f"eirin_bots_total {bots.get('total', 0)}")
        lines.append("# HELP eirin_bots_running Running bots")
        lines.append("# TYPE eirin_bots_running gauge")
        lines.append(f"eirin_bots_running {bots.get('running', 0)}")
        cb = get_circuit_breaker_status()
        lines.append("# HELP eirin_emergency_stop Emergency stop active")
        lines.append("# TYPE eirin_emergency_stop gauge")
        lines.append(f"eirin_emergency_stop {1 if cb.get('emergency_stop') else 0}")
    except Exception as e:
        logger.debug("prometheus_metrics: %s", e)
    return "\n".join(lines) if lines else None
