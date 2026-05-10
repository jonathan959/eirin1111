"""Journal write-on-close via db.close_deal."""

from __future__ import annotations

import pytest

import db as dbm


@pytest.fixture()
def jdb(tmp_path, monkeypatch):
    db_path = tmp_path / "j.sqlite3"
    dbm._tl.__dict__.clear()
    monkeypatch.setattr(dbm, "DB_NAME", str(db_path), raising=True)
    dbm.init_db()
    yield
    dbm._tl.__dict__.clear()


def test_close_deal_inserts_journal(jdb):
    params = (
        "J", "BTC/USD", 1, 1, 10.0, 10.0, 3, 0.015, 1.2, 0.015, 0, 200, 100.0, 10, "auto", "", 6,
        1.0, 1.0, 0.003, 0.06, 2, 2, 0.6, 0.5, 0.15, 0.1, 6, 0.003, 45, 0.06, 6, 1, 0, "crypto", "paper", "",
        dbm.now_ts(),
    )

    def ins_bot(con):
        con.execute(
            """
            INSERT INTO bots(
                name, symbol, enabled, dry_run,
                base_quote, safety_quote, max_safety, first_dev, step_mult, tp,
                trend_filter, trend_sma,
                max_spend_quote, poll_seconds, strategy_mode, forced_strategy, max_open_orders,
                vol_gap_mult, tp_vol_mult, min_gap_pct, max_gap_pct,
                regime_hold_candles, regime_switch_ticks, regime_switch_threshold,
                max_total_exposure_pct, per_symbol_exposure_pct, min_free_cash_pct, max_concurrent_deals,
                spread_guard_pct, limit_timeout_sec, daily_loss_limit_pct, pause_hours,
                auto_restart, last_running, market_type, alpaca_mode, bot_type, created_at
            ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
            """,
            params,
        )
        return int(con.execute("SELECT last_insert_rowid()").fetchone()[0])

    bid = dbm.write_txn(None, ins_bot, name="test_ins_bot")
    did = dbm.open_deal(bid, "BTC/USD", state="OPEN", opened_at=1000)
    dbm.close_deal(
        did,
        entry_avg=100.0,
        exit_avg=110.0,
        base_amount=0.1,
        realized_pnl_quote=1.0,
        entry_strategy="classic_dca",
        exit_strategy="take_profit_hit",
    )
    rows = dbm.list_journal_entries(limit=5)
    assert len(rows) == 1
    assert rows[0]["bot_id"] == bid
    assert rows[0]["exit_reason"] == "tp_hit"


def test_exit_reason_explicit_tp():
    from services.journal import normalize_exit_reason

    assert normalize_exit_reason("noise", explicit="tp_hit") == "tp_hit"


def test_exit_reason_unknown_string_maps_manual():
    from services.journal import normalize_exit_reason

    assert normalize_exit_reason("some_vendor_reason_xyz") == "manual_close"
