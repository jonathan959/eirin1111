"""Exposure cap 422 on bot save + once-per-day autotune."""

from __future__ import annotations

import asyncio
import json
from unittest.mock import MagicMock, patch

import pytest

import db as dbm
from worker_api import api_create_bot, api_update_bot


class MockRequest:
    def __init__(self, json_data):
        self._json = json_data

    async def json(self):
        return self._json


@pytest.fixture()
def iso_db(tmp_path, monkeypatch):
    db_path = tmp_path / "exp.sqlite3"
    dbm._tl.__dict__.clear()
    monkeypatch.setattr(dbm, "DB_NAME", str(db_path), raising=True)
    dbm.init_db()
    yield
    dbm._tl.__dict__.clear()


def _minimal_bot_payload(**kwargs):
    base = {
        "name": "T",
        "symbol": "BTC/USD",
        "base_quote": 10.0,
        "safety_quote": 10.0,
        "max_safety": 3,
        "max_spend_quote": 100.0,
        "first_dev": 0.015,
        "step_mult": 1.2,
        "tp": 0.015,
    }
    base.update(kwargs)
    return base


def test_autotune_increases_pct(iso_db):
    from unittest.mock import MagicMock

    bid = dbm.create_bot(_minimal_bot_payload(per_symbol_exposure_pct=0.15, base_quote=25))
    ok = dbm.try_per_symbol_autotune(
        bid,
        equity=106.59,
        base_quote=25.0,
        position_value=0.0,
        notify=MagicMock(),
        bot_label="B",
    )
    assert ok is True
    b = dbm.get_bot(bid)
    assert float(b["per_symbol_exposure_pct"]) >= 0.24
    rows = dbm.list_journal_entries(limit=5)
    assert any((r.get("entry_reason") == "autotune") for r in rows)


def test_create_bot_exposure_422(iso_db):
    async def _run():
        mock_bm = MagicMock()
        mock_bm.get_portfolio_total.return_value = 106.59
        with patch("worker_api.bm", mock_bm), patch("worker_api.create_bot") as cr:
            req = MockRequest(_minimal_bot_payload(base_quote=25, per_symbol_exposure_pct=0.15))
            resp = await api_create_bot(req)
        assert resp.status_code == 422
        body = json.loads(resp.body)
        assert body["error"] == "exposure_cap_conflict"
        assert "suggestions" in body
        assert body["suggestions"]["per_symbol_pct"] >= 0.24
        cr.assert_not_called()

    asyncio.run(_run())


def test_update_bot_ok_under_cap(iso_db):
    bid = dbm.create_bot(_minimal_bot_payload())

    async def _run():
        mock_bm = MagicMock()
        mock_bm.get_portfolio_total.return_value = 10_000.0
        bot = dbm.get_bot(bid)
        with patch("worker_api.bm", mock_bm), patch("worker_api.get_bot", return_value=bot):
            req = MockRequest({"base_quote": 25, "per_symbol_exposure_pct": 0.15})
            resp = await api_update_bot(bid, req)
        assert resp.status_code == 200

    asyncio.run(_run())


def test_update_bot_exposure_422(iso_db):
    bid = dbm.create_bot(_minimal_bot_payload())

    async def _run():
        mock_bm = MagicMock()
        mock_bm.get_portfolio_total.return_value = 106.59
        bot = dbm.get_bot(bid)
        with patch("worker_api.bm", mock_bm), patch("worker_api.get_bot", return_value=bot):
            req = MockRequest({"base_order_quote": 25, "per_symbol_pct": 0.15})
            resp = await api_update_bot(bid, req)
        assert resp.status_code == 422
        body = json.loads(resp.body)
        assert body["error"] == "exposure_cap_conflict"
        eff = body["current"]["portfolio_value"] * body["current"]["per_symbol_pct"]
        assert body["current"]["base_order_quote"] > eff + 0.01

    asyncio.run(_run())


def test_autotune_once_per_day(iso_db):
    from datetime import timezone, datetime

    from unittest.mock import MagicMock

    bid = dbm.create_bot(_minimal_bot_payload(per_symbol_exposure_pct=0.15, base_quote=25))
    today = datetime.now(timezone.utc).date().isoformat()
    dbm.update_bot_fields(bid, {"last_autotune_date": today})
    ok = dbm.try_per_symbol_autotune(
        bid,
        equity=106.59,
        base_quote=25.0,
        position_value=20.0,
        notify=MagicMock(),
        bot_label="B",
    )
    assert ok is False
