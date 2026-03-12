"""
Regression tests: DB persistence, Explore portfolio-aware filtering, Autopilot create_bots.
Run with: pytest tests/test_persistence_explore_autopilot.py -v

BotManager is initialized synchronously at startup. If Kraken (and optionally Alpaca) keys
are missing or ENABLE_ALPACA=0, bm may not start; tests that need bm skip with explicit reason.
db_info test always runs (no bm required). Set BM_READY_TIMEOUT=15 in env to override wait.
"""
import os
import time
import pytest


BM_READY_TIMEOUT = int(os.getenv("BM_READY_TIMEOUT", "15"))


@pytest.fixture(scope="module")
def app_and_db():
    from worker_api import app
    from db import init_db
    init_db()
    return app


@pytest.fixture
def client(app_and_db):
    from fastapi.testclient import TestClient
    return TestClient(app_and_db)


def _wait_for_bm(client, timeout_sec=None):
    """Poll /api/debug/bm_ready until true or timeout. Skip only with explicit reason."""
    timeout_sec = timeout_sec or BM_READY_TIMEOUT
    deadline = time.monotonic() + timeout_sec
    while time.monotonic() < deadline:
        r = client.get("/api/debug/bm_ready")
        if r.status_code == 200 and r.json().get("bm_ready"):
            return True
        time.sleep(0.5)
    return False


def test_db_info_returns_path_and_bot_count(client):
    """GET /api/debug/db_info returns ok, db_path, bot_count, cwd. Always runs (no bm required)."""
    r = client.get("/api/debug/db_info")
    assert r.status_code == 200
    data = r.json()
    assert data.get("ok") is True
    assert "db_path" in data
    assert "bot_count" in data
    assert "cwd" in data


def test_db_persistence_bots_not_disappear(client):
    """Create a bot via API, then list bots via API: bot is still there (same DB path). Skips if BotManager not ready (503)."""
    if not _wait_for_bm(client):
        r = client.get("/api/debug/startup_status")
        reason = "unknown"
        if r.status_code == 200:
            s = r.json().get("startup_status") or {}
            reason = s.get("last_startup_error") or ("kraken_ready=%s bm_ready=%s" % (s.get("kraken_ready"), s.get("bm_ready")))
        pytest.skip("BotManager not ready within %ss: %s" % (BM_READY_TIMEOUT, reason))
    r = client.post(
        "/api/bots",
        json={
            "name": "Persistence Test",
            "symbol": "XBT/USD",
            "enabled": 0,
            "dry_run": 1,
            "base_quote": 10.0,
            "safety_quote": 5.0,
            "max_safety": 2,
            "strategy_mode": "classic",
            "max_spend_quote": 25.0,
            "market_type": "crypto",
        },
        timeout=10,
    )
    if r.status_code == 503:
        data = r.json() if r.headers.get("content-type", "").startswith("application/json") else {}
        pytest.skip("BotManager not initialized: %s" % (data.get("reason") or data.get("error") or "503"))
    assert r.status_code == 200
    data = r.json()
    assert data.get("ok") is True
    bot_id = data.get("bot", {}).get("id")
    assert bot_id
    r2 = client.get("/api/bots")
    assert r2.status_code == 200
    bots = r2.json().get("bots") or []
    ids = [int(b.get("id")) for b in bots if b.get("id")]
    assert bot_id in ids, "Bot should appear in list (DB persistence)"
    r3 = client.get("/api/debug/db_info")
    assert r3.json().get("bot_count", 0) >= 1
    client.delete(f"/api/bots/{bot_id}")


def test_explore_filtering_already_active(client):
    """With an enabled bot for a symbol, Explore returns already_active and active_reason for it."""
    if not _wait_for_bm(client):
        pytest.skip("BotManager not ready within %ss (needed to create bot for Explore test)" % BM_READY_TIMEOUT)
    r = client.post(
        "/api/bots",
        json={
            "name": "Explore Filter Test",
            "symbol": "ETH/USD",
            "enabled": 1,
            "dry_run": 1,
            "base_quote": 10.0,
            "safety_quote": 5.0,
            "max_safety": 2,
            "strategy_mode": "classic",
            "max_spend_quote": 25.0,
            "market_type": "crypto",
        },
        timeout=10,
    )
    if r.status_code != 200:
        pytest.skip("Create bot failed (e.g. Kraken validation)")
    bot_id = r.json().get("bot", {}).get("id")
    assert bot_id
    try:
        r = client.get("/api/recommendations?horizon=short&market_type=crypto&limit=50&show_already_active=1")
        assert r.status_code in (200, 503)
        if r.status_code == 503:
            pytest.skip("Kraken not ready")
        data = r.json()
        items = data.get("items") or []
        for i in items:
            assert "already_active" in i
            if i.get("already_active"):
                assert "active_reason" in i
        # If ETH/USD appears in recommendations, it must be marked already_active (we have enabled bot)
        eth_usd_items = [i for i in items if (i.get("symbol") or "").upper() in ("ETH/USD", "ETH/USDT")]
        for i in eth_usd_items:
            assert i.get("already_active") is True, "ETH/USD has enabled bot so must be already_active"
            assert "enabled_bot" in (i.get("active_reason") or []), "active_reason should include enabled_bot"
    finally:
        client.delete(f"/api/bots/{bot_id}")


def test_autopilot_create_bots_dry_run(client):
    """POST /api/autopilot/create_bots with count=1 dry_run=1 returns ok and created list."""
    if not _wait_for_bm(client):
        pytest.skip("BotManager not ready within %ss (required for create_bots)" % BM_READY_TIMEOUT)
    r = client.post(
        "/api/autopilot/create_bots",
        json={"count": 1, "dry_run": 1, "horizon": "long"},
        timeout=15,
    )
    assert r.status_code == 200
    data = r.json()
    assert "created" in data
    assert "errors" in data
    if not data.get("ok"):
        pytest.skip(data.get("error", "create_bots returned ok=false"))
    created = data.get("created") or []
    for one in created:
        assert "symbol" in one
        if one.get("ok") and one.get("bot_id"):
            client.delete(f"/api/bots/{one['bot_id']}")
