"""Idempotent journal backfill from closed deals."""

from __future__ import annotations

import pytest

import db as dbm


@pytest.fixture()
def bdb(tmp_path, monkeypatch):
    db_path = tmp_path / "bf.sqlite3"
    dbm._tl.__dict__.clear()
    monkeypatch.setattr(dbm, "DB_NAME", str(db_path), raising=True)
    dbm.init_db()
    yield
    dbm._tl.__dict__.clear()


def test_backfill_idempotent(bdb):
    params = (
        "J", "ETH/USD", 1, 1, 10.0, 10.0, 3, 0.015, 1.2, 0.015, 0, 200, 100.0, 10, "auto", "", 6,
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

    bid = dbm.write_txn(None, ins_bot, name="test_ins_bot2")

    def ins_deal(con):
        con.execute(
            """
            INSERT INTO deals (
              bot_id, state, opened_at, closed_at, symbol,
              entry_avg, exit_avg, base_amount, realized_pnl_quote,
              entry_strategy, exit_strategy
            ) VALUES (?,?,?,?,?,?,?,?,?,?,?)
            """,
            (bid, "CLOSED", 2000, 3000, "ETH/USD", 2000.0, 2100.0, 0.05, 3.0, "grid", "stop_loss_hit"),
        )

    dbm.write_txn(bid, ins_deal, name="ins_deal")

    n1 = dbm.backfill_journal_from_closed_deals()
    n2 = dbm.backfill_journal_from_closed_deals()
    assert n1 == 1
    assert n2 == 0
    rows = dbm.list_journal_entries(limit=5)
    assert len(rows) == 1
    assert rows[0]["entry_reason"] == "backfilled"
    assert rows[0]["exit_reason"] == "sl_hit"
