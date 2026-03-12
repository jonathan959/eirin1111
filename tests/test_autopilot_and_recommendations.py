"""
Tests: Autopilot creation, duplicate prevention, recommendations portfolio-aware filtering.
Run: pytest tests/test_autopilot_and_recommendations.py -v

These tests validate:
1. Autopilot run returns detailed diagnostics (never silently fails)
2. Duplicate prevention: re-running autopilot does not create duplicate bots
3. Recommendations exclude active symbols by default
4. Debug endpoints return expected fields
5. Score breakdown is present in recommendations
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
    timeout_sec = timeout_sec or BM_READY_TIMEOUT
    deadline = time.monotonic() + timeout_sec
    while time.monotonic() < deadline:
        r = client.get("/api/debug/bm_ready")
        if r.status_code == 200 and r.json().get("bm_ready"):
            return True
        time.sleep(0.5)
    return False


class TestAutopilotRunDiagnostics:
    """Autopilot /api/autopilot/run must return detailed diagnostics, never silently succeed."""

    def test_run_returns_bm_ready_field(self, client):
        r = client.post("/api/autopilot/run")
        data = r.json()
        assert "bm_ready" in data, "Response must include bm_ready field"
        assert "kraken_ready" in data, "Response must include kraken_ready field"
        assert "paused" in data, "Response must include paused field"
        assert "kill_switch" in data, "Response must include kill_switch field"

    def test_run_bm_not_ready_returns_error(self, client):
        r = client.post("/api/autopilot/run")
        data = r.json()
        if not data.get("bm_ready"):
            assert data.get("ok") is False, "Must return ok=false when bm not ready"
            assert data.get("error"), "Must include error message when bm not ready"
            assert "created_bots" in data
            assert data["created"] == 0

    def test_run_returns_created_bots_list(self, client):
        if not _wait_for_bm(client):
            pytest.skip("BotManager not ready")
        r = client.post("/api/autopilot/run")
        data = r.json()
        assert "created_bots" in data, "Response must include created_bots list"
        assert "skipped" in data, "Response must include skipped list"
        assert "candidates_considered" in data, "Response must include candidates_considered"
        assert isinstance(data["created_bots"], list)
        assert isinstance(data["skipped"], list)

    def test_run_zero_created_has_error_message(self, client):
        if not _wait_for_bm(client):
            pytest.skip("BotManager not ready")
        r = client.post("/api/autopilot/run")
        data = r.json()
        if data.get("created", 0) == 0:
            assert data.get("error") is not None or data.get("candidates_considered", -1) == 0, \
                "When created=0, must explain why (error message or candidates_considered=0)"


class TestAutopilotDuplicatePrevention:
    """Re-running autopilot must not create duplicate bots for the same symbol."""

    def test_no_duplicate_bots_on_rerun(self, client):
        if not _wait_for_bm(client):
            pytest.skip("BotManager not ready")
        r1 = client.post("/api/autopilot/run")
        d1 = r1.json()
        created_first = d1.get("created_bots", [])
        if not created_first:
            pytest.skip("No bots created on first run (no recommendations available)")
        first_symbols = {b["symbol"] for b in created_first if b.get("symbol")}
        r2 = client.post("/api/autopilot/run")
        d2 = r2.json()
        created_second = d2.get("created_bots", [])
        second_symbols = {b["symbol"] for b in created_second if b.get("symbol")}
        overlap = first_symbols & second_symbols
        assert len(overlap) == 0, f"Duplicate bots created for: {overlap}"
        for bot in created_first:
            if bot.get("id"):
                client.delete(f"/api/bots/{bot['id']}")
        for bot in created_second:
            if bot.get("id"):
                client.delete(f"/api/bots/{bot['id']}")


class TestRecommendationsFiltering:
    """Recommendations must be portfolio-aware: exclude active symbols by default."""

    def test_active_bot_excluded_by_default(self, client):
        if not _wait_for_bm(client):
            pytest.skip("BotManager not ready")
        r = client.post(
            "/api/bots",
            json={
                "name": "Reco Filter Test",
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
            pytest.skip("Create bot failed")
        bot_id = r.json().get("bot", {}).get("id")
        assert bot_id
        try:
            r_default = client.get("/api/recommendations?horizon=short&market_type=crypto&limit=50&show_already_active=0")
            if r_default.status_code == 503:
                pytest.skip("Kraken not ready")
            items = r_default.json().get("items", [])
            eth_items = [i for i in items if "ETH" in (i.get("symbol") or "").upper()]
            for i in eth_items:
                assert i.get("already_active") is not True, \
                    "ETH/USD should be excluded by default (show_already_active=0)"

            r_active = client.get("/api/recommendations?horizon=short&market_type=crypto&limit=50&show_already_active=1")
            if r_active.status_code == 200:
                items_active = r_active.json().get("items", [])
                eth_active = [i for i in items_active if "ETH" in (i.get("symbol") or "").upper()]
                for i in eth_active:
                    if i.get("already_active"):
                        assert "active_reason" in i
                        assert isinstance(i["active_reason"], list)
        finally:
            client.delete(f"/api/bots/{bot_id}")

    def test_recommendations_include_score_breakdown(self, client):
        r = client.get("/api/recommendations?horizon=short&market_type=crypto&limit=5&show_already_active=0")
        if r.status_code == 503:
            pytest.skip("Kraken not ready")
        items = r.json().get("items", [])
        for item in items:
            assert "score_breakdown" in item, f"Item {item.get('symbol')} missing score_breakdown"
            assert isinstance(item["score_breakdown"], list)


class TestDebugEndpoints:
    """Debug endpoints must return all required diagnostic fields."""

    def test_startup_status(self, client):
        r = client.get("/api/debug/startup_status")
        assert r.status_code == 200
        data = r.json()
        assert data.get("ok") is True
        s = data.get("startup_status", {})
        for field in ["db_path", "kraken_ready", "alpaca_ready", "bm_ready", "paused", "kill_switch", "cwd"]:
            assert field in s, f"startup_status missing field: {field}"

    def test_bm_ready(self, client):
        r = client.get("/api/debug/bm_ready")
        assert r.status_code == 200
        data = r.json()
        assert "bm_ready" in data
        assert "kraken_ready" in data
        assert "kraken_error" in data
        if not data["bm_ready"]:
            assert data.get("reason") is not None, "Must include reason when bm not ready"

    def test_db_info(self, client):
        r = client.get("/api/debug/db_info")
        assert r.status_code == 200
        data = r.json()
        assert data.get("ok") is True
        assert "db_path" in data
        assert "bot_count" in data
        assert "autopilot_bot_count" in data
        assert "recommendation_count" in data
        assert "cwd" in data

    def test_db_info_same_path_as_startup(self, client):
        r1 = client.get("/api/debug/db_info")
        r2 = client.get("/api/debug/startup_status")
        db_path = r1.json().get("db_path")
        startup_path = r2.json().get("startup_status", {}).get("db_path")
        assert db_path == startup_path, f"DB path mismatch: db_info={db_path}, startup={startup_path}"


class TestExploreV2Scoring:
    """explore_v2 scoring enhancements: crash penalties, TF alignment, diversity."""

    def test_enhance_score_crash_penalty(self):
        from explore_v2 import enhance_score
        score, reasons = enhance_score(
            75.0,
            {"return_30d": -0.20},
            "BEAR",
        )
        assert score < 75.0, "Score should be penalized for 30d crash"
        assert any("crash" in r.lower() or "30d" in r.lower() for r in reasons)

    def test_enhance_score_tf_alignment_bonus(self):
        from explore_v2 import enhance_score
        score, reasons = enhance_score(
            70.0,
            {"regime_1d": "BULL", "regime_4h": "BREAKOUT"},
            "BULL",
        )
        assert score > 70.0, "Score should get bonus for multi-TF alignment"
        assert any("alignment" in r.lower() for r in reasons)

    def test_enhance_score_tf_divergence_penalty(self):
        from explore_v2 import enhance_score
        score, reasons = enhance_score(
            70.0,
            {"regime_1d": "BULL", "regime_4h": "BEAR"},
            "BULL",
        )
        assert score < 70.0, "Score should be penalized for TF divergence"

    def test_diversify_picks_limits_low_liquidity(self):
        from explore_v2 import diversify_picks
        items = [
            {"symbol": f"TEST{i}/USD", "score": 80 - i, "volume": 1000 if i < 5 else 100000}
            for i in range(10)
        ]
        result = diversify_picks(items, top_k=10)
        low_liq = [r for r in result if (r.get("volume") or 0) < 50000]
        assert len(low_liq) <= 3, "Should cap low-liquidity items"

    def test_compute_score_breakdown(self):
        from explore_v2 import compute_score_breakdown
        breakdown = compute_score_breakdown(
            80.0,
            {"weekly_trend": "Uptrend", "atr_pct": 0.01, "winrate": 0.75},
            "BULL",
        )
        assert isinstance(breakdown, list)
        assert len(breakdown) >= 1
        assert len(breakdown) <= 3


class TestAutopilotCycleUnit:
    """Unit tests for autopilot.run_autopilot_cycle return format."""

    def test_cycle_disabled_returns_status(self):
        from autopilot import run_autopilot_cycle
        result = run_autopilot_cycle(
            create_bot_fn=lambda p: 1,
            delete_bot_fn=lambda i: None,
            start_bot_fn=lambda i: None,
            stop_bot_fn=lambda i: None,
            get_portfolio_total_fn=lambda: 10000.0,
            force_run=False,
        )
        assert result.get("status") == "disabled" or result.get("created") == 0

    def test_cycle_force_run_returns_detailed(self):
        from autopilot import run_autopilot_cycle
        result = run_autopilot_cycle(
            create_bot_fn=lambda p: 1,
            delete_bot_fn=lambda i: None,
            start_bot_fn=lambda i: None,
            stop_bot_fn=lambda i: None,
            get_portfolio_total_fn=lambda: 10000.0,
            force_run=True,
        )
        assert "created_bots" in result
        assert "skipped" in result
        assert "candidates_considered" in result
        assert isinstance(result["created_bots"], list)
        assert isinstance(result["skipped"], list)
