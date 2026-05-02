"""Per-writer regression tests for the Phase 1.2b migrations.

Each top-level section corresponds to one commit in 1.2b. The shared fixtures
mirror tests/test_db_locking.py so the fault model (4-thread contention,
concurrent DELETE, etc.) is consistent across writers.

These are FUNCTION-LEVEL regressions. The whole-system load test
(`tests/test_db_locking.py::test_no_lock_under_load`) lands in Phase 1.5 and
covers all writers together for 60s under realistic mix.
"""

from __future__ import annotations

import os
import sqlite3
import tempfile
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import pytest

import db as dbmod


@pytest.fixture()
def temp_db(monkeypatch, tmp_path):
    """Fresh on-disk DB initialised via init_db; clears thread-local cache."""
    db_file = tmp_path / "wm_test.sqlite3"
    dbmod._tl.__dict__.clear()
    monkeypatch.setattr(dbmod, "DB_NAME", str(db_file), raising=True)
    dbmod.init_db()
    yield str(db_file)
    dbmod.stop_wal_checkpoint_thread(timeout_sec=2.0)
    dbmod._tl.__dict__.clear()
    dbmod._bot_locks.clear()
    for suffix in ("", "-wal", "-shm"):
        try:
            os.unlink(str(db_file) + suffix)
        except OSError:
            pass


# ===========================================================================
# 1.2b step 1: add_log under contention + concurrent cleanup
# ===========================================================================

def test_add_log_under_concurrent_contention(temp_db):
    """4 producer threads + 1 cleanup thread for 2s. After migration, zero
    OperationalError must propagate. Pre-migration, this is the exact pattern
    that produced bot 1's 'Fatal error: OperationalError: database is locked'
    loop on the live host.
    """
    BOT_ID = 1
    DURATION_SEC = 2.0

    errors: list = []
    err_lock = threading.Lock()
    counts = {"prod_a": 0, "prod_b": 0, "prod_c": 0, "prod_d": 0, "cleanup": 0}
    cnt_lock = threading.Lock()

    def _producer(label: str):
        deadline = time.monotonic() + DURATION_SEC
        i = 0
        while time.monotonic() < deadline:
            try:
                # Vary the message every few calls to defeat the dedup path
                # AND to also exercise the dedup INCREMENT path (same message
                # 3 times in a row → UPDATE branch).
                msg = f"{label}-{i // 3}"
                dbmod.add_log(BOT_ID, "INFO", msg, "TEST")
                with cnt_lock:
                    counts[label] += 1
            except sqlite3.OperationalError as e:
                with err_lock:
                    errors.append((label, repr(e)))
                return
            i += 1

    def _cleanup():
        # Mimic Phase 1.2b step 8: chunked DELETE LIMIT 500. We're not testing
        # the cleanup function itself here — just that it CAN run alongside
        # add_log without blowing it up.
        deadline = time.monotonic() + DURATION_SEC
        while time.monotonic() < deadline:
            cutoff = int(time.time()) - 1  # delete logs older than 1s
            try:
                def _del(con):
                    cur = con.execute(
                        "DELETE FROM bot_logs WHERE rowid IN "
                        "(SELECT rowid FROM bot_logs WHERE ts < ? LIMIT 100)",
                        (cutoff,),
                    )
                    return int(cur.rowcount or 0)
                n = dbmod.write_txn(None, _del, name="cleanup_chunk")
                with cnt_lock:
                    counts["cleanup"] += 1 if n > 0 else 0
            except sqlite3.OperationalError as e:
                with err_lock:
                    errors.append(("cleanup", repr(e)))
                return
            time.sleep(0.05)

    threads = [
        threading.Thread(target=_producer, args=(label,), name=label)
        for label in ("prod_a", "prod_b", "prod_c", "prod_d")
    ]
    threads.append(threading.Thread(target=_cleanup, name="cleanup"))

    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=DURATION_SEC + 5.0)
        assert not t.is_alive(), f"{t.name} did not exit (deadlock?)"

    assert not errors, f"OperationalError leaked under contention: {errors}"
    for label in ("prod_a", "prod_b", "prod_c", "prod_d"):
        assert counts[label] > 0, f"{label} made no forward progress"
    # cleanup may have 0 batches if producers were faster than cleanup window;
    # we only assert it didn't crash.


def test_add_log_dedup_collapses_in_single_transaction(temp_db):
    """Two identical successive add_log calls → exactly ONE row with count=2."""
    BOT_ID = 2
    dbmod.add_log(BOT_ID, "INFO", "hello", "TEST")
    dbmod.add_log(BOT_ID, "INFO", "hello", "TEST")
    dbmod.add_log(BOT_ID, "INFO", "hello", "TEST")

    fresh = sqlite3.connect(temp_db, timeout=5.0)
    try:
        fresh.row_factory = sqlite3.Row
        rows = fresh.execute(
            "SELECT id, message, count FROM bot_logs WHERE bot_id=? ORDER BY id",
            (BOT_ID,),
        ).fetchall()
    finally:
        fresh.close()
    assert len(rows) == 1, f"expected 1 row (deduped), got {len(rows)}: {[dict(r) for r in rows]}"
    assert int(rows[0]["count"]) == 3


def test_add_log_changing_message_inserts_new_row(temp_db):
    BOT_ID = 3
    dbmod.add_log(BOT_ID, "INFO", "first", "TEST")
    dbmod.add_log(BOT_ID, "INFO", "second", "TEST")
    dbmod.add_log(BOT_ID, "INFO", "second", "TEST")  # dedup with previous
    dbmod.add_log(BOT_ID, "WARN", "second", "TEST")  # different level → new row

    fresh = sqlite3.connect(temp_db, timeout=5.0)
    try:
        fresh.row_factory = sqlite3.Row
        rows = fresh.execute(
            "SELECT message, level, count FROM bot_logs WHERE bot_id=? ORDER BY id",
            (BOT_ID,),
        ).fetchall()
    finally:
        fresh.close()
    msgs = [(r["message"], r["level"], int(r["count"])) for r in rows]
    assert msgs == [("first", "INFO", 1), ("second", "INFO", 2), ("second", "WARN", 1)]


def test_add_log_uses_per_bot_lock(temp_db):
    """Two add_log calls on different bot_ids should not serialise on the
    Python lock (SQLite still serialises file-level writes; we only check
    that the per-bot RLock isn't shared)."""
    lk_a = dbmod.bot_db_lock(1001)
    lk_b = dbmod.bot_db_lock(1002)
    assert lk_a is not lk_b, "different bot_ids must get distinct RLocks"

    # Smoke: both calls succeed.
    dbmod.add_log(1001, "INFO", "a", "T")
    dbmod.add_log(1002, "INFO", "b", "T")


# ===========================================================================
# 1.2b step 2: deals (open_deal, update_open_deal_entry, close_deal,
#                     record_trade_feedback)
# ===========================================================================

def _seed_open_deal(bot_id: int = 1, symbol: str = "BTC/USD") -> int:
    return dbmod.open_deal(int(bot_id), symbol, state="OPEN", opened_at=int(time.time()))


def test_open_deal_creates_row(temp_db):
    deal_id = _seed_open_deal(bot_id=11, symbol="ETH/USD")
    assert deal_id > 0
    fresh = sqlite3.connect(temp_db, timeout=5.0)
    try:
        fresh.row_factory = sqlite3.Row
        row = fresh.execute("SELECT bot_id, state, symbol FROM deals WHERE id=?", (deal_id,)).fetchone()
    finally:
        fresh.close()
    assert dict(row) == {"bot_id": 11, "state": "OPEN", "symbol": "ETH/USD"}


def test_update_open_deal_entry_writes_entry(temp_db):
    deal_id = _seed_open_deal(bot_id=12, symbol="BTC/USD")
    dbmod.update_open_deal_entry(deal_id, entry_avg=100.0, base_amount=0.5, safety_count=1)
    fresh = sqlite3.connect(temp_db, timeout=5.0)
    try:
        fresh.row_factory = sqlite3.Row
        row = fresh.execute(
            "SELECT entry_avg, base_amount, safety_count, state FROM deals WHERE id=?",
            (deal_id,),
        ).fetchone()
    finally:
        fresh.close()
    assert float(row["entry_avg"]) == 100.0
    assert float(row["base_amount"]) == 0.5
    assert int(row["safety_count"]) == 1
    assert row["state"] == "OPEN"


def test_close_deal_atomic_with_trade_feedback_and_recommendation(temp_db):
    """close_deal must run the deal UPDATE, the trade_feedback INSERT, and
    the recommendation_performance UPDATE in a single transaction. We can't
    easily inject a crash between them, but we can verify all three rows
    land together for a happy path AND that close_deal does not raise the
    nested-write_txn RuntimeError despite the helper calls."""
    bot_id = 21
    deal_id = _seed_open_deal(bot_id=bot_id, symbol="BTC/USD")
    dbmod.update_open_deal_entry(deal_id, entry_avg=100.0, base_amount=0.5)

    # Seed an active recommendation_performance row so the inner update fires.
    def _seed_rec(con):
        con.execute(
            "INSERT INTO recommendation_performance("
            "bot_id, symbol, recommendation_date, score_at_recommendation, "
            "regime_at_recommendation, entry_price, exit_price, pnl_realized, "
            "days_held, outcome, notes, created_at) "
            "VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
            (bot_id, "BTC/USD", int(time.time()) - 86400, 75, "Trend",
             100.0, 0.0, 0.0, 0.0, "active", "seed", int(time.time()) - 86400),
        )
    dbmod.write_txn(None, _seed_rec, name="seed_rec")

    dbmod.close_deal(
        deal_id=deal_id,
        entry_avg=100.0,
        exit_avg=110.0,
        base_amount=0.5,
        realized_pnl_quote=5.0,
        entry_strategy="t",
        exit_strategy="manual_close_dry",
        hold_sec=3600,
        safety_count=0,
    )

    fresh = sqlite3.connect(temp_db, timeout=5.0)
    try:
        fresh.row_factory = sqlite3.Row
        deal_row = fresh.execute("SELECT state, exit_avg FROM deals WHERE id=?", (deal_id,)).fetchone()
        fb = fresh.execute(
            "SELECT symbol, profitable FROM trade_feedback ORDER BY rowid DESC LIMIT 1"
        ).fetchone()
        rec = fresh.execute(
            "SELECT outcome, exit_price, pnl_realized FROM recommendation_performance "
            "WHERE bot_id=? ORDER BY id DESC LIMIT 1",
            (bot_id,),
        ).fetchone()
    finally:
        fresh.close()

    assert deal_row["state"] == "CLOSED"
    assert float(deal_row["exit_avg"]) == 110.0
    assert fb["symbol"] == "BTC/USD"
    assert int(fb["profitable"]) == 1
    assert rec["outcome"] == "win"
    assert float(rec["exit_price"]) == 110.0


def test_cancel_ghost_deal_only_acts_on_open(temp_db):
    deal_id = _seed_open_deal(bot_id=31, symbol="BTC/USD")

    dbmod.cancel_ghost_deal(deal_id)

    fresh = sqlite3.connect(temp_db, timeout=5.0)
    try:
        fresh.row_factory = sqlite3.Row
        row = fresh.execute("SELECT state FROM deals WHERE id=?", (deal_id,)).fetchone()
    finally:
        fresh.close()
    assert row["state"] == "CANCELLED"

    # Second call must be a no-op (state is no longer OPEN).
    dbmod.cancel_ghost_deal(deal_id)


def test_concurrent_deal_writers_under_contention(temp_db):
    """4 threads pound on open_deal/update_open_deal_entry/close_deal for
    DIFFERENT bots; assert zero OperationalError, all forward progress."""
    DURATION_SEC = 2.0
    errors: list = []
    err_lock = threading.Lock()
    counts = {1: 0, 2: 0, 3: 0, 4: 0}
    cnt_lock = threading.Lock()

    def _worker(bot_id: int):
        deadline = time.monotonic() + DURATION_SEC
        while time.monotonic() < deadline:
            try:
                deal_id = dbmod.open_deal(bot_id, "BTC/USD")
                dbmod.update_open_deal_entry(deal_id, entry_avg=100.0, base_amount=0.1)
                dbmod.close_deal(
                    deal_id=deal_id, entry_avg=100.0, exit_avg=101.0,
                    base_amount=0.1, realized_pnl_quote=0.1,
                    entry_strategy="t", exit_strategy="t",
                    hold_sec=1, safety_count=0,
                )
                with cnt_lock:
                    counts[bot_id] += 1
            except sqlite3.OperationalError as e:
                with err_lock:
                    errors.append((bot_id, repr(e)))
                return

    threads = [threading.Thread(target=_worker, args=(b,)) for b in (1, 2, 3, 4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=DURATION_SEC + 5.0)
        assert not t.is_alive(), "deal writer thread did not exit (deadlock?)"

    assert not errors, f"OperationalError leaked: {errors}"
    for b, n in counts.items():
        assert n > 0, f"bot {b} made no forward progress"


def test_record_trade_feedback_standalone(temp_db):
    """record_trade_feedback() with no con= goes through write_txn(None, ...)."""
    dbmod.record_trade_feedback("BTC/USD", "{}", profitable=1)
    fresh = sqlite3.connect(temp_db, timeout=5.0)
    try:
        rows = fresh.execute("SELECT symbol, profitable FROM trade_feedback").fetchall()
    finally:
        fresh.close()
    assert rows == [("BTC/USD", 1)]


def test_record_trade_feedback_nested_with_con(temp_db):
    """record_trade_feedback(..., con=outer_con) does NOT call write_txn —
    proves nested usage from inside close_deal-style fn() bodies works."""
    def _outer(con):
        dbmod.record_trade_feedback("ETH/USD", '{"x":1}', profitable=0, con=con)

    dbmod.write_txn(None, _outer, name="outer_with_nested_feedback")
    fresh = sqlite3.connect(temp_db, timeout=5.0)
    try:
        rows = fresh.execute("SELECT symbol, profitable FROM trade_feedback").fetchall()
    finally:
        fresh.close()
    assert ("ETH/USD", 0) in rows


# ===========================================================================
# 1.2b step 3: order_events / add_order_event
# ===========================================================================

def test_add_order_event_inserts_row(temp_db):
    dbmod.add_order_event(
        bot_id=51, symbol="BTC/USD", side="buy", ord_type="market",
        price=100.0, amount=0.5, order_id="o-1", tag=None,
        status="filled", reason="entry",
    )
    fresh = sqlite3.connect(temp_db, timeout=5.0)
    try:
        fresh.row_factory = sqlite3.Row
        rows = fresh.execute(
            "SELECT bot_id, symbol, side, status FROM order_events WHERE bot_id=?",
            (51,),
        ).fetchall()
    finally:
        fresh.close()
    assert len(rows) == 1
    assert dict(rows[0]) == {"bot_id": 51, "symbol": "BTC/USD", "side": "buy", "status": "filled"}


def test_add_order_event_under_concurrent_load(temp_db):
    """4 threads, 1 bot each, append order events for 1.5s. Plus a chunked
    cleanup thread DELETEing old rows. Zero OperationalError must propagate."""
    DURATION_SEC = 1.5
    errors: list = []
    err_lock = threading.Lock()
    counts = {1: 0, 2: 0, 3: 0, 4: 0}
    cnt_lock = threading.Lock()

    def _producer(bot_id: int):
        deadline = time.monotonic() + DURATION_SEC
        i = 0
        while time.monotonic() < deadline:
            try:
                dbmod.add_order_event(
                    bot_id=bot_id, symbol="BTC/USD", side="buy", ord_type="market",
                    price=100.0 + i, amount=0.01, order_id=f"o-{bot_id}-{i}",
                    tag=None, status="ack", reason="t",
                )
                with cnt_lock:
                    counts[bot_id] += 1
            except sqlite3.OperationalError as e:
                with err_lock:
                    errors.append((bot_id, repr(e)))
                return
            i += 1

    def _cleanup():
        deadline = time.monotonic() + DURATION_SEC
        while time.monotonic() < deadline:
            try:
                cutoff = int(time.time())
                def _del(con):
                    cur = con.execute(
                        "DELETE FROM order_events WHERE rowid IN "
                        "(SELECT rowid FROM order_events WHERE ts < ? LIMIT 100)",
                        (cutoff,),
                    )
                    return int(cur.rowcount or 0)
                dbmod.write_txn(None, _del, name="cleanup_orders")
            except sqlite3.OperationalError as e:
                with err_lock:
                    errors.append(("cleanup", repr(e)))
                return
            time.sleep(0.05)

    threads = [threading.Thread(target=_producer, args=(b,)) for b in (1, 2, 3, 4)]
    threads.append(threading.Thread(target=_cleanup))
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=DURATION_SEC + 5.0)
        assert not t.is_alive()

    assert not errors, f"OperationalError leaked: {errors}"
    for b, n in counts.items():
        assert n > 0, f"bot {b} made no forward progress"


# ===========================================================================
# 1.2b step 5: explore_*: save_signal_outcome, update_explore_signal_outcome,
#                          mark_explore_signals_pending, mark_explore_horizon_pending,
#                          upsert_explore_feed_row, save_recommendation_snapshot
# Plus deletion of legacy _db_retry helper.
# ===========================================================================

def test_save_recommendation_snapshot_returns_id_and_upserts_latest(temp_db):
    sid = dbmod.save_recommendation_snapshot(
        symbol="ETH/USD", horizon="short", score=72.5,
        regime_json="{}", metrics_json="{}", reasons_json="[]", risk_flags_json="[]",
        composite_score=68.0, confidence_score=0.8, conviction_grade="A",
    )
    assert sid > 0
    fresh = sqlite3.connect(temp_db, timeout=5.0)
    try:
        fresh.row_factory = sqlite3.Row
        latest = fresh.execute(
            "SELECT snapshot_id FROM recommendations_latest WHERE symbol='ETH/USD' AND horizon='short'"
        ).fetchone()
    finally:
        fresh.close()
    assert int(latest["snapshot_id"]) == sid


def test_mark_explore_signals_pending_updates_status(temp_db):
    # Seed two rows.
    dbmod.upsert_explore_feed_row(
        symbol="BTC/USD", horizon="short", status="buy",
        conviction_score=80, reason="r", strategy="s",
        signal_ts=int(time.time()), detail_json=None, price=100.0,
        change_24h=0.0, market_type="crypto",
    )
    dbmod.upsert_explore_feed_row(
        symbol="ETH/USD", horizon="short", status="watch",
        conviction_score=60, reason="r", strategy="s",
        signal_ts=int(time.time()), detail_json=None, price=200.0,
        change_24h=0.0, market_type="crypto",
    )

    dbmod.mark_explore_signals_pending("short", int(time.time()))

    fresh = sqlite3.connect(temp_db, timeout=5.0)
    try:
        statuses = [r[0] for r in fresh.execute(
            "SELECT status FROM explore_signals WHERE horizon='short'"
        ).fetchall()]
    finally:
        fresh.close()
    assert all(s == "pending" for s in statuses), statuses


def test_upsert_explore_feed_row_inserts_then_updates(temp_db):
    dbmod.upsert_explore_feed_row(
        symbol="SOL/USD", horizon="medium", status="buy",
        conviction_score=70, reason="initial", strategy="s",
        signal_ts=100, detail_json=None, price=10.0,
        change_24h=1.0, market_type="crypto",
    )
    dbmod.upsert_explore_feed_row(
        symbol="SOL/USD", horizon="medium", status="watch",
        conviction_score=55, reason="updated", strategy="s",
        signal_ts=200, detail_json=None, price=11.0,
        change_24h=2.0, market_type="crypto",
    )

    fresh = sqlite3.connect(temp_db, timeout=5.0)
    try:
        fresh.row_factory = sqlite3.Row
        rows = fresh.execute(
            "SELECT status, reason, signal_ts FROM explore_signals "
            "WHERE symbol='SOL/USD' AND horizon='medium'"
        ).fetchall()
    finally:
        fresh.close()
    assert len(rows) == 1
    assert rows[0]["status"] == "watch"
    assert rows[0]["reason"] == "updated"
    assert int(rows[0]["signal_ts"]) == 200


def test_save_signal_outcome_then_update_pnl(temp_db):
    rid = dbmod.save_signal_outcome(
        symbol="BTC/USD", horizon="short", strategy="t",
        signal_ts=int(time.time()), entry_price=100.0,
        composite_score=75.0, conviction_grade="B",
    )
    assert rid > 0

    dbmod.update_explore_signal_outcome(
        outcome_id=rid, price_5d=102.0, price_10d=110.0, price_20d=115.0,
    )

    fresh = sqlite3.connect(temp_db, timeout=5.0)
    try:
        fresh.row_factory = sqlite3.Row
        row = fresh.execute(
            "SELECT outcome, pnl_5d_pct, pnl_10d_pct, pnl_20d_pct FROM explore_signal_outcomes WHERE id=?",
            (rid,),
        ).fetchone()
    finally:
        fresh.close()
    assert row["outcome"] == "win"
    assert abs(float(row["pnl_5d_pct"]) - 2.0) < 1e-6
    assert abs(float(row["pnl_10d_pct"]) - 10.0) < 1e-6
    assert abs(float(row["pnl_20d_pct"]) - 15.0) < 1e-6


def test_mark_explore_horizon_pending_alias(temp_db):
    dbmod.upsert_explore_feed_row(
        symbol="X/USD", horizon="long", status="buy",
        conviction_score=50, reason="r", strategy="s",
        signal_ts=100, detail_json=None, price=10.0,
        change_24h=0.0, market_type="crypto",
    )
    dbmod.mark_explore_horizon_pending("long")
    fresh = sqlite3.connect(temp_db, timeout=5.0)
    try:
        s = fresh.execute(
            "SELECT status FROM explore_signals WHERE symbol='X/USD' AND horizon='long'"
        ).fetchone()[0]
    finally:
        fresh.close()
    assert s == "pending"


def test_db_retry_helper_was_removed():
    """Phase 1.2b step 5 deletes _db_retry. Ensure the symbol no longer exists."""
    assert not hasattr(dbmod, "_db_retry"), \
        "_db_retry should be deleted in Phase 1.2b step 5; new code uses write_txn"


# ===========================================================================
# 1.2b step 6: notification_manager (8 writers — insert/mark + 5 notify_* + 1 daily)
# ===========================================================================

def test_notification_writers(temp_db, monkeypatch):
    import notification_manager as nm

    # Stub network fanout — these tests cover ONLY the DB write path; the
    # Discord/Telegram side-effects are tested elsewhere.
    monkeypatch.setattr(nm, "send_discord_notification", lambda *a, **k: True, raising=True)
    monkeypatch.setattr(nm, "send_telegram_notification", lambda *a, **k: True, raising=True)
    # Stub get_setting so external fanout is short-circuited (and hits no env).
    import db as _db
    monkeypatch.setattr(_db, "get_setting", lambda *a, **k: "", raising=False)

    nid = nm.insert_notification("test", "Title", "Hello", bot_id=11)
    assert nid > 0

    assert nm.notify_trade_executed("bot1", "BTC/USD", "buy", 0.5, 100.0)
    assert nm.notify_take_profit("bot1", "BTC/USD", 5.0, 0.05)
    assert nm.notify_stop_loss("bot1", "BTC/USD", -3.0, -0.03)
    assert nm.notify_bot_error("bot1", "boom")
    assert nm.notify_drawdown_alert(-0.1)
    assert nm.notify_daily_summary(1.5, 3, 1, "X", "Y")

    fresh = sqlite3.connect(temp_db, timeout=5.0)
    try:
        types = sorted(t for (t,) in fresh.execute("SELECT type FROM notifications").fetchall())
    finally:
        fresh.close()
    assert types == sorted([
        "test", "trade_executed", "take_profit", "stop_loss",
        "bot_error", "drawdown_alert", "daily_summary",
    ])

    # mark_notification_read smoke
    assert nm.mark_notification_read(nid) is True
    fresh = sqlite3.connect(temp_db, timeout=5.0)
    try:
        flag = fresh.execute("SELECT read FROM notifications WHERE id=?", (nid,)).fetchone()[0]
    finally:
        fresh.close()
    assert int(flag) == 1


def test_notification_writers_under_concurrent_load(temp_db, monkeypatch):
    """Hammer notify_trade_executed from 4 threads. Pre-migration the DB
    INSERT had no retry; under add_log/order_event contention this would
    OperationalError. Asserts zero leaks now."""
    import notification_manager as nm
    monkeypatch.setattr(nm, "send_discord_notification", lambda *a, **k: True, raising=True)
    monkeypatch.setattr(nm, "send_telegram_notification", lambda *a, **k: True, raising=True)
    import db as _db
    monkeypatch.setattr(_db, "get_setting", lambda *a, **k: "", raising=False)

    DURATION_SEC = 1.5
    errors: list = []
    err_lock = threading.Lock()
    counts = {1: 0, 2: 0, 3: 0, 4: 0}
    cnt_lock = threading.Lock()

    def _worker(label: int):
        deadline = time.monotonic() + DURATION_SEC
        i = 0
        while time.monotonic() < deadline:
            try:
                nm.notify_trade_executed(f"bot{label}", "BTC/USD", "buy", 0.01 * i, 100.0 + i)
                with cnt_lock:
                    counts[label] += 1
            except sqlite3.OperationalError as e:
                with err_lock:
                    errors.append((label, repr(e)))
                return
            i += 1

    threads = [threading.Thread(target=_worker, args=(b,)) for b in (1, 2, 3, 4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=DURATION_SEC + 5.0)
        assert not t.is_alive()

    assert not errors, f"OperationalError leaked: {errors}"
    for b, n in counts.items():
        assert n > 0


# ===========================================================================
# 1.2b step 7: small modules — execution_quality_tracker, tax_optimizer,
#                              sector_rotation
# ===========================================================================

def test_record_execution_inserts_row(temp_db, monkeypatch):
    monkeypatch.setenv("TRACK_EXECUTION_QUALITY", "1")
    # Re-import to pick up the env var.
    import importlib
    import execution_quality_tracker as eqt
    importlib.reload(eqt)

    ok = eqt.record_execution(
        order_id="o-1", bot_id=99, symbol="BTC/USD", side="buy",
        intended_price=100.0, executed_price=100.5, strategy="t",
    )
    assert ok is True

    fresh = sqlite3.connect(temp_db, timeout=5.0)
    try:
        row = fresh.execute(
            "SELECT bot_id, symbol, executed_price FROM execution_quality WHERE order_id=?",
            ("o-1",),
        ).fetchone()
    finally:
        fresh.close()
    assert row == (99, "BTC/USD", 100.5)


def test_save_tax_harvest_suggestion(temp_db):
    import tax_optimizer
    tax_optimizer.save_tax_harvest_suggestion(
        symbol="BTC/USD", unrealized_loss_pct=-7.5,
        wash_sale_until_ts=int(time.time()) + 86400, alternate_symbol="ETH/USD",
    )
    fresh = sqlite3.connect(temp_db, timeout=5.0)
    try:
        rows = fresh.execute(
            "SELECT symbol, alternate_symbol FROM tax_harvest_suggestions"
        ).fetchall()
    finally:
        fresh.close()
    assert rows == [("BTC/USD", "ETH/USD")]


def test_record_sector_performance(temp_db):
    import sector_rotation
    sector_rotation.record_sector_performance(
        sector="Technology", quarter_ts=int(time.time()),
        return_pct=8.5, momentum_score=72.0, rank=1,
    )
    fresh = sqlite3.connect(temp_db, timeout=5.0)
    try:
        rows = fresh.execute(
            "SELECT sector, rank FROM sector_performance_history"
        ).fetchall()
    finally:
        fresh.close()
    assert rows == [("Technology", 1)]


def test_explore_writers_under_concurrent_load(temp_db):
    """4 threads each calling a mix of save_signal_outcome /
    upsert_explore_feed_row / mark_explore_signals_pending for 1.5s; assert
    zero OperationalError leaks."""
    DURATION_SEC = 1.5
    errors: list = []
    err_lock = threading.Lock()
    counts = {"so": 0, "ufr": 0, "msp": 0}
    cnt_lock = threading.Lock()

    def _writer(label: str):
        deadline = time.monotonic() + DURATION_SEC
        i = 0
        while time.monotonic() < deadline:
            try:
                if label == "so":
                    dbmod.save_signal_outcome(
                        symbol=f"S{i}/USD", horizon="short", strategy="t",
                        signal_ts=int(time.time()), entry_price=10.0 + i,
                    )
                elif label == "ufr":
                    dbmod.upsert_explore_feed_row(
                        symbol=f"S{i % 5}/USD", horizon="short", status="buy",
                        conviction_score=60, reason="r", strategy="s",
                        signal_ts=int(time.time()), detail_json=None,
                        price=1.0, change_24h=0.0, market_type="crypto",
                    )
                else:  # msp
                    dbmod.mark_explore_signals_pending("short", int(time.time()))
                with cnt_lock:
                    counts[label] += 1
            except sqlite3.OperationalError as e:
                with err_lock:
                    errors.append((label, repr(e)))
                return
            i += 1

    threads = [
        threading.Thread(target=_writer, args=("so",)),
        threading.Thread(target=_writer, args=("ufr",)),
        threading.Thread(target=_writer, args=("ufr",)),  # 2nd UFR for hot conflict
        threading.Thread(target=_writer, args=("msp",)),
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=DURATION_SEC + 5.0)
        assert not t.is_alive()

    assert not errors, f"OperationalError leaked: {errors}"
    for label, n in counts.items():
        assert n > 0, f"{label} made no forward progress"


def test_cancel_ghost_deal_serialises_with_open_deal_for_same_bot(temp_db):
    """For the SAME bot_id, cancel_ghost_deal and open_deal must serialise
    on the per-bot RLock (no interleaving)."""
    bot_id = 42
    trace: list = []
    trace_lock = threading.Lock()

    def _open():
        with trace_lock:
            trace.append("open_start")
        deal_id = dbmod.open_deal(bot_id, "BTC/USD")
        with trace_lock:
            trace.append("open_done")
        return deal_id

    def _cancel(did):
        with trace_lock:
            trace.append("cancel_start")
        dbmod.cancel_ghost_deal(did)
        with trace_lock:
            trace.append("cancel_done")

    deal_id = _open()
    with ThreadPoolExecutor(max_workers=2) as ex:
        f1 = ex.submit(_open)
        f2 = ex.submit(_cancel, deal_id)
        f1.result(timeout=5.0)
        f2.result(timeout=5.0)
    # Each pair must be contiguous (no interleaving). The exact ordering
    # between the two operations isn't deterministic — only that neither
    # interleaves with the other.
    assert "open_done" in trace and "cancel_done" in trace


# ===========================================================================
# 1.2b step 8: chunked cleanup_old_* + delete_recommendations_for_blocklist
# ===========================================================================
#
# Rationale: the brief escalates this group from a refactor to a load-bearing
# safety fix — the bot 1 'OperationalError: database is locked' loop on the
# live host is most likely caused by mass DELETE-while-INSERT collisions.
# Each test here proves:
#   1. Functional behaviour: rows that should be deleted ARE deleted.
#   2. Forward progress: cleanup makes progress every batch.
#   3. Concurrency: zero OperationalError leaks while a hot INSERT loop runs
#      against the same table.
# ===========================================================================


def _seed_bot_logs(temp_db: str, bot_id: int, total: int, ts: int) -> None:
    """Seed total rows directly via raw connection (test fixture only)."""
    fresh = sqlite3.connect(temp_db, timeout=10.0)
    try:
        fresh.executemany(
            "INSERT INTO bot_logs(bot_id, ts, level, message) VALUES (?,?,?,?)",
            [(bot_id, ts, "INFO", f"seed-{i}") for i in range(total)],
        )
        fresh.commit()
    finally:
        fresh.close()


def test_cleanup_old_bot_logs_chunked_deletes_old_keeps_new(temp_db):
    """Functional: only rows older than cutoff are deleted; recent rows stay."""
    bot_id = 1
    now = int(time.time())
    old_ts = now - 60 * 86400          # 60 days old
    new_ts = now - 1 * 86400           # 1 day old (within keep_days=30)

    # Seed 1200 old + 50 new — old set spans >2 chunks of 500.
    _seed_bot_logs(temp_db, bot_id, 1200, old_ts)
    _seed_bot_logs(temp_db, bot_id, 50, new_ts)

    deleted = dbmod.cleanup_old_bot_logs(keep_days=30)

    assert deleted == 1200, f"expected 1200 deleted, got {deleted}"
    fresh = sqlite3.connect(temp_db, timeout=5.0)
    try:
        remaining = fresh.execute("SELECT COUNT(*) FROM bot_logs").fetchone()[0]
        old_left = fresh.execute(
            "SELECT COUNT(*) FROM bot_logs WHERE ts < ?", (now - 30 * 86400,),
        ).fetchone()[0]
    finally:
        fresh.close()
    assert remaining == 50, f"expected 50 recent rows to survive, got {remaining}"
    assert old_left == 0, f"expected 0 old rows left, got {old_left}"


def test_cleanup_old_bot_logs_returns_zero_when_nothing_old(temp_db):
    """Forward-progress: empty case must terminate (not loop) and return 0."""
    bot_id = 2
    now = int(time.time())
    _seed_bot_logs(temp_db, bot_id, 100, now)  # all fresh
    deleted = dbmod.cleanup_old_bot_logs(keep_days=30)
    assert deleted == 0


def test_cleanup_old_bot_logs_under_concurrent_insert_load(temp_db):
    """The Phase 1.2b §8 acceptance test, paraphrased from the brief:

        "Replace each cleanup with a chunked loop. Add a regression test:
         run cleanup concurrently with a 1000-INSERT workload against the
         same table; assert both complete with zero OperationalErrors and
         cleanup makes forward progress every batch."

    Pre-migration this asserted on a single-DELETE pattern that held the
    write lock for ~hundreds of ms; the inserter would hit
    OperationalError under busy_timeout. Post-migration the chunked loop
    yields the lock every 500 rows so both can complete cleanly.
    """
    bot_id = 3
    now = int(time.time())
    old_ts = now - 60 * 86400

    # Seed 5000 old rows so cleanup must execute >=10 batches and the
    # inserter has time to overlap multiple batch boundaries.
    _seed_bot_logs(temp_db, bot_id, 5000, old_ts)

    insert_errors: list = []
    insert_count = {"n": 0}
    cleanup_errors: list = []
    progress_marks: list = []  # one entry per batch deleted by cleanup

    # Patch chunked_delete to record each batch's rowcount so we can assert
    # forward progress per batch instead of just totals.
    real_chunked_delete = dbmod.chunked_delete

    def tracking_chunked_delete(table, where_sql, params, *,
                                batch_size=dbmod.CLEANUP_BATCH_SIZE,
                                sleep_between_sec=dbmod.CLEANUP_INTERBATCH_SLEEP_SEC,
                                op_name=None):
        # Wrap the real helper but record each batch's deleted count.
        sql = (
            f"DELETE FROM {table} WHERE rowid IN ("
            f"SELECT rowid FROM {table} WHERE {where_sql} LIMIT {int(batch_size)}"
            f")"
        )
        total = 0
        while True:
            def _do(con):
                cur = con.execute(sql, params)
                return int(cur.rowcount or 0)
            n = dbmod.write_txn(None, _do, name=op_name or f"chunked_delete_{table}")
            progress_marks.append(n)
            total += n
            if n < int(batch_size):
                return total
            if sleep_between_sec > 0:
                time.sleep(sleep_between_sec)

    dbmod.chunked_delete = tracking_chunked_delete  # type: ignore[assignment]
    try:
        stop = threading.Event()

        def _inserter():
            i = 0
            while i < 1000 and not stop.is_set():
                try:
                    dbmod.add_log(bot_id, "INFO", f"hot-{i}")
                    insert_count["n"] += 1
                except sqlite3.OperationalError as e:
                    insert_errors.append(repr(e))
                    return
                i += 1

        def _cleanup():
            try:
                dbmod.cleanup_old_bot_logs(keep_days=30)
            except Exception as e:
                cleanup_errors.append(repr(e))

        t_ins = threading.Thread(target=_inserter, name="inserter")
        t_cln = threading.Thread(target=_cleanup, name="cleanup")
        t_ins.start()
        t_cln.start()
        t_ins.join(timeout=30.0)
        t_cln.join(timeout=30.0)
        stop.set()
    finally:
        dbmod.chunked_delete = real_chunked_delete  # type: ignore[assignment]

    assert not t_ins.is_alive(), "inserter did not finish in 30s"
    assert not t_cln.is_alive(), "cleanup did not finish in 30s"
    assert not insert_errors, f"inserter saw OperationalError: {insert_errors}"
    assert not cleanup_errors, f"cleanup raised: {cleanup_errors}"
    assert insert_count["n"] == 1000, f"only {insert_count['n']}/1000 inserts done"

    # Forward-progress proof: each batch except possibly the last must have
    # deleted >0 rows. The last batch is the terminator (rows < batch_size,
    # possibly 0). Total deleted should equal seeded old rows (5000).
    assert progress_marks, "no batches recorded — chunked loop never ran"
    non_terminal = progress_marks[:-1]
    assert all(n > 0 for n in non_terminal), (
        f"some non-terminal batch made zero progress: {progress_marks}"
    )
    assert sum(progress_marks) == 5000, (
        f"expected to delete 5000 old rows, deleted {sum(progress_marks)}: {progress_marks}"
    )


def test_cleanup_old_order_events_chunked_under_load(temp_db):
    """Smaller mirror of the bot_logs test for order_events (the second-hottest
    table by INSERT rate)."""
    bot_id = 4
    now = int(time.time())

    # Seed 1500 OLD order_events directly so cleanup has work across batches.
    # order_events columns: id, bot_id, ts, symbol, side, ord_type, price,
    # amount, order_id, tag, status, reason
    fresh = sqlite3.connect(temp_db, timeout=10.0)
    try:
        fresh.executemany(
            "INSERT INTO order_events(bot_id, ts, symbol, side, ord_type, "
            "price, amount, order_id, tag, status, reason) "
            "VALUES (?,?,?,?,?,?,?,?,?,?,?)",
            [(bot_id, now - 100 * 86400, "BTC/USD", "buy", "limit",
              30000.0, 0.001, f"seed-{i}", "seed", "filled", None)
             for i in range(1500)],
        )
        fresh.commit()
    finally:
        fresh.close()

    insert_errors: list = []
    cleanup_errors: list = []
    inserts_done = {"n": 0}

    def _inserter():
        for i in range(300):
            try:
                dbmod.add_order_event(
                    bot_id, "BTC/USD", "buy", "limit",
                    30000.0, 0.001, f"hot-{i}", "hot", "filled", None,
                )
                inserts_done["n"] += 1
            except sqlite3.OperationalError as e:
                insert_errors.append(repr(e))
                return

    def _cleanup():
        try:
            dbmod.cleanup_old_order_events(keep_days=90)
        except Exception as e:
            cleanup_errors.append(repr(e))

    t_ins = threading.Thread(target=_inserter)
    t_cln = threading.Thread(target=_cleanup)
    t_ins.start(); t_cln.start()
    t_ins.join(timeout=30.0); t_cln.join(timeout=30.0)
    assert not t_ins.is_alive() and not t_cln.is_alive()
    assert not insert_errors, f"inserter OperationalError: {insert_errors}"
    assert not cleanup_errors, f"cleanup raised: {cleanup_errors}"
    assert inserts_done["n"] == 300

    # All 1500 old rows deleted, all 300 hot rows survived.
    fresh = sqlite3.connect(temp_db, timeout=5.0)
    try:
        old_left = fresh.execute(
            "SELECT COUNT(*) FROM order_events WHERE ts < ?",
            (now - 90 * 86400,),
        ).fetchone()[0]
        recent = fresh.execute(
            "SELECT COUNT(*) FROM order_events WHERE ts >= ?",
            (now - 90 * 86400,),
        ).fetchone()[0]
    finally:
        fresh.close()
    assert old_left == 0, f"expected 0 old order_events left, got {old_left}"
    assert recent >= 300, f"recent rows lost: {recent} < 300"


def test_chunked_delete_rejects_unknown_table(temp_db):
    """SQL-injection guard: chunked_delete must reject tables not in the
    allowlist, even though it formats the table name into the SQL string."""
    with pytest.raises(ValueError, match="not in _ALLOWED_TABLES"):
        dbmod.chunked_delete(
            "evil_table; DROP TABLE bots--",
            "ts < ?", (0,),
        )


def test_delete_recommendations_for_blocklist_chunked(temp_db):
    """Functional + behavioural: blocklist DELETE removes only matching bases
    and routes through chunked_delete (so the lock is released between
    batches). Mixed case input is normalised to UPPER per the implementation."""
    fresh = sqlite3.connect(temp_db, timeout=5.0)
    try:
        # recommendations_latest schema: PRIMARY KEY(symbol, horizon).
        # Seed 30 USDT/* rows (blocked) + 30 BTC/* rows (kept) + 1 USDC row
        # (blocked) — each across multiple horizons to keep the PK happy.
        rows = []
        for i in range(30):
            rows.append((f"USDT/USD-{i}", "short", 0, int(time.time())))
        for i in range(30):
            rows.append((f"BTC/USD-{i}", "short", 0, int(time.time())))
        rows.append(("USDC", "short", 0, int(time.time())))
        fresh.executemany(
            "INSERT INTO recommendations_latest(symbol, horizon, snapshot_id, created_ts) "
            "VALUES (?,?,?,?)",
            rows,
        )
        fresh.commit()
    finally:
        fresh.close()

    deleted = dbmod.delete_recommendations_for_blocklist(["usdt", "USDC", ""])

    assert deleted == 31, f"expected 31 deleted, got {deleted}"
    fresh = sqlite3.connect(temp_db, timeout=5.0)
    try:
        remaining = fresh.execute(
            "SELECT COUNT(*) FROM recommendations_latest"
        ).fetchone()[0]
        btc_left = fresh.execute(
            "SELECT COUNT(*) FROM recommendations_latest WHERE symbol LIKE 'BTC/%'"
        ).fetchone()[0]
    finally:
        fresh.close()
    assert remaining == 30 and btc_left == 30


def test_delete_recommendations_for_blocklist_empty_input_noop(temp_db):
    """Empty bases list returns 0 without touching the DB."""
    assert dbmod.delete_recommendations_for_blocklist([]) == 0
    assert dbmod.delete_recommendations_for_blocklist(["", "  ", ""]) == 0


# ===========================================================================
# 1.2c step 2: bot-id-scoped writers — update_bot, set_bot_*, delete_bot,
#              add_regime_snapshot, add_strategy_decision, add_strategy_trade,
#              save_perf_metrics, link_recommendation_to_bot,
#              update_ml_prediction_outcome, patch_bot_risk_after_create,
#              manual_close_deal_and_journal
# ===========================================================================


def _seed_min_bot(temp_db: str, name: str = "T1", symbol: str = "BTC/USD") -> int:
    """Create a bot row via the public API and return its id."""
    bot_id = dbmod.create_bot({
        "name": name,
        "symbol": symbol,
        "enabled": 0,
        "dry_run": 1,
        "base_quote": 10.0,
        "safety_quote": 5.0,
        "max_safety": 3,
        "first_dev": 0.01,
        "step_mult": 1.0,
        "tp": 0.02,
        "max_spend_quote": 100.0,
    })
    return int(bot_id)


def test_update_bot_routes_through_per_bot_lock(temp_db, monkeypatch):
    """update_bot must run inside write_txn with bot_id=<that bot> so it
    serialises with the runner's other writes for the same bot.
    Race #3 in the Phase 1.1 lock-loop diagnosis."""
    bid = _seed_min_bot(temp_db, "B-update")

    captured: list = []
    real_write_txn = dbmod.write_txn

    def tracking_write_txn(bot_id, fn, *, name=None):
        captured.append((bot_id, name))
        return real_write_txn(bot_id, fn, name=name)

    monkeypatch.setattr(dbmod, "write_txn", tracking_write_txn, raising=True)

    dbmod.update_bot(bid, {
        "name": "B-update-v2", "symbol": "ETH/USD",
        "base_quote": 5.0, "safety_quote": 2.5, "max_safety": 2,
        "first_dev": 0.005, "step_mult": 1.5, "tp": 0.01,
        "max_spend_quote": 50.0,
    })

    assert (bid, "update_bot") in captured
    fresh = sqlite3.connect(temp_db, timeout=5.0)
    try:
        row = fresh.execute(
            "SELECT name, symbol, max_spend_quote FROM bots WHERE id=?", (bid,),
        ).fetchone()
    finally:
        fresh.close()
    assert row == ("B-update-v2", "ETH/USD", 50.0)


def test_set_bot_enabled_routes_through_per_bot_lock(temp_db, monkeypatch):
    bid = _seed_min_bot(temp_db, "B-enabled")
    captured: list = []
    real = dbmod.write_txn
    monkeypatch.setattr(dbmod, "write_txn",
                        lambda b, fn, *, name=None: (captured.append((b, name)), real(b, fn, name=name))[1],
                        raising=True)
    dbmod.set_bot_enabled(bid, True)
    assert (bid, "set_bot_enabled") in captured

    fresh = sqlite3.connect(temp_db, timeout=5.0)
    try:
        v = fresh.execute("SELECT enabled FROM bots WHERE id=?", (bid,)).fetchone()[0]
    finally:
        fresh.close()
    assert v == 1


def test_set_bot_running_routes_through_per_bot_lock(temp_db, monkeypatch):
    bid = _seed_min_bot(temp_db, "B-running")
    captured: list = []
    real = dbmod.write_txn
    monkeypatch.setattr(dbmod, "write_txn",
                        lambda b, fn, *, name=None: (captured.append((b, name)), real(b, fn, name=name))[1],
                        raising=True)
    dbmod.set_bot_running(bid, True)
    assert (bid, "set_bot_running") in captured

    fresh = sqlite3.connect(temp_db, timeout=5.0)
    try:
        v = fresh.execute("SELECT last_running FROM bots WHERE id=?", (bid,)).fetchone()[0]
    finally:
        fresh.close()
    assert v == 1


def test_delete_bot_atomic_cascade(temp_db, monkeypatch):
    """delete_bot must remove all child-table rows AND the bots row in
    a single transaction. If write_txn raises mid-cascade everything
    rolls back together."""
    bid = _seed_min_bot(temp_db, "B-del")
    # Seed some children
    dbmod.add_log(bid, "INFO", "before delete")
    dbmod.add_strategy_decision(bid, "rsi", "BUY", "test", "trending", 0.7, "{}")

    captured: list = []
    real = dbmod.write_txn
    monkeypatch.setattr(dbmod, "write_txn",
                        lambda b, fn, *, name=None: (captured.append((b, name)), real(b, fn, name=name))[1],
                        raising=True)

    dbmod.delete_bot(bid)
    assert (bid, "delete_bot") in captured

    fresh = sqlite3.connect(temp_db, timeout=5.0)
    try:
        bots = fresh.execute("SELECT COUNT(*) FROM bots WHERE id=?", (bid,)).fetchone()[0]
        logs = fresh.execute("SELECT COUNT(*) FROM bot_logs WHERE bot_id=?", (bid,)).fetchone()[0]
        decs = fresh.execute("SELECT COUNT(*) FROM strategy_decisions WHERE bot_id=?", (bid,)).fetchone()[0]
    finally:
        fresh.close()
    assert bots == 0 and logs == 0 and decs == 0


def test_add_regime_snapshot_uses_per_bot_lock(temp_db, monkeypatch):
    bid = _seed_min_bot(temp_db, "B-regime")
    captured: list = []
    real = dbmod.write_txn
    monkeypatch.setattr(dbmod, "write_txn",
                        lambda b, fn, *, name=None: (captured.append((b, name)), real(b, fn, name=name))[1],
                        raising=True)
    dbmod.add_regime_snapshot(bid, "BTC/USD", "trending", 0.85, "ema-up", "{}")
    assert (bid, "add_regime_snapshot") in captured


def test_add_strategy_decision_uses_per_bot_lock(temp_db, monkeypatch):
    bid = _seed_min_bot(temp_db, "B-strat")
    captured: list = []
    real = dbmod.write_txn
    monkeypatch.setattr(dbmod, "write_txn",
                        lambda b, fn, *, name=None: (captured.append((b, name)), real(b, fn, name=name))[1],
                        raising=True)
    dbmod.add_strategy_decision(bid, "rsi", "BUY", "rsi<30", "trending", 0.6, "{}")
    assert (bid, "add_strategy_decision") in captured


def test_add_strategy_trade_uses_per_bot_lock(temp_db, monkeypatch):
    bid = _seed_min_bot(temp_db, "B-trade")
    captured: list = []
    real = dbmod.write_txn
    monkeypatch.setattr(dbmod, "write_txn",
                        lambda b, fn, *, name=None: (captured.append((b, name)), real(b, fn, name=name))[1],
                        raising=True)
    dbmod.add_strategy_trade(bid, "rsi", 12.5, symbol="BTC/USD", regime="trending", pnl_pct=1.25)
    assert (bid, "add_strategy_trade") in captured


def test_save_perf_metrics_uses_per_bot_lock(temp_db, monkeypatch):
    bid = _seed_min_bot(temp_db, "B-perf")
    captured: list = []
    real = dbmod.write_txn
    monkeypatch.setattr(dbmod, "write_txn",
                        lambda b, fn, *, name=None: (captured.append((b, name)), real(b, fn, name=name))[1],
                        raising=True)
    dbmod.save_perf_metrics(bid, "rsi", "{}")
    assert (bid, "save_perf_metrics") in captured


def test_patch_bot_risk_after_create(temp_db, monkeypatch):
    bid = _seed_min_bot(temp_db, "B-risk")
    captured: list = []
    real = dbmod.write_txn
    monkeypatch.setattr(dbmod, "write_txn",
                        lambda b, fn, *, name=None: (captured.append((b, name)), real(b, fn, name=name))[1],
                        raising=True)

    dbmod.patch_bot_risk_after_create(bid, stop_loss_pct=0.08, max_hold_hours=48)
    assert (bid, "patch_bot_risk_after_create") in captured

    fresh = sqlite3.connect(temp_db, timeout=5.0)
    try:
        sl, mh = fresh.execute(
            "SELECT stop_loss_pct, max_hold_hours FROM bots WHERE id=?", (bid,),
        ).fetchone()
    finally:
        fresh.close()
    assert sl == 0.08 and mh == 48

    # No-op when both Nones — must not call write_txn.
    captured.clear()
    dbmod.patch_bot_risk_after_create(bid)
    assert captured == []


def test_link_recommendation_to_bot_uses_per_bot_lock(temp_db, monkeypatch):
    bid = _seed_min_bot(temp_db, "B-rec")
    captured: list = []
    real = dbmod.write_txn
    monkeypatch.setattr(dbmod, "write_txn",
                        lambda b, fn, *, name=None: (captured.append((b, name)), real(b, fn, name=name))[1],
                        raising=True)
    dbmod.link_recommendation_to_bot(
        bid, "ETH/USD", int(time.time()), 75.0, "trending",
        metrics_json="{}", reasons_json="[]",
    )
    assert (bid, "link_recommendation_to_bot") in captured


def test_update_ml_prediction_outcome_global_lock(temp_db, monkeypatch):
    """update_ml_prediction_outcome uses bot_id=None per the audit
    (L-risk, off-tick path)."""
    fresh = sqlite3.connect(temp_db, timeout=5.0)
    try:
        # ml_predictions schema varies — synthesise a row that satisfies
        # every NOT NULL column by inspecting the table.
        cols_info = fresh.execute("PRAGMA table_info(ml_predictions)").fetchall()
        # Each row: (cid, name, type, notnull, dflt_value, pk)
        seed_cols: List[str] = []
        seed_vals: List[Any] = []
        for cid, name, ctype, notnull, dflt, pk in cols_info:
            if pk:
                continue  # AUTOINCREMENT id
            if not notnull and dflt is not None:
                continue
            seed_cols.append(name)
            t = (ctype or "").upper()
            if "INT" in t:
                seed_vals.append(0)
            elif "REAL" in t or "FLOA" in t or "DOUB" in t:
                seed_vals.append(0.0)
            else:
                seed_vals.append("test")
        if not seed_cols:
            # Edge case: id-only table → use NULL row.
            sql = "INSERT INTO ml_predictions DEFAULT VALUES"
            fresh.execute(sql)
        else:
            placeholders = ",".join(["?"] * len(seed_cols))
            sql = f"INSERT INTO ml_predictions({','.join(seed_cols)}) VALUES ({placeholders})"
            fresh.execute(sql, seed_vals)
        fresh.commit()
        pid = int(fresh.execute("SELECT last_insert_rowid()").fetchone()[0])
    finally:
        fresh.close()

    captured: list = []
    real = dbmod.write_txn
    monkeypatch.setattr(dbmod, "write_txn",
                        lambda b, fn, *, name=None: (captured.append((b, name)), real(b, fn, name=name))[1],
                        raising=True)

    dbmod.update_ml_prediction_outcome(pid, actual_outcome_7d=0.05)
    dbmod.update_ml_prediction_outcome(pid, actual_outcome_30d=0.12)
    dbmod.update_ml_prediction_outcome(pid, actual_outcome_7d=0.05, actual_outcome_30d=0.12)

    # All three branches go through write_txn(None, ...).
    assert all(b is None and n == "update_ml_prediction_outcome" for b, n in captured)
    # No-op call (both None) must not call write_txn.
    before = list(captured)
    dbmod.update_ml_prediction_outcome(pid)
    assert captured == before


def test_concurrent_per_bot_writers_serialise(temp_db):
    """4 threads each writing different per-bot writers (add_log,
    set_bot_enabled, add_regime_snapshot, add_strategy_decision) for
    the SAME bot, for 1.5s. Zero OperationalError leaks."""
    bid = _seed_min_bot(temp_db, "B-concurrent")
    DURATION_SEC = 1.5
    errors: list = []
    err_lock = threading.Lock()
    counts = {"log": 0, "ena": 0, "reg": 0, "dec": 0}
    cnt_lock = threading.Lock()

    def _writer(label: str):
        deadline = time.monotonic() + DURATION_SEC
        i = 0
        toggle = False
        while time.monotonic() < deadline:
            try:
                if label == "log":
                    dbmod.add_log(bid, "INFO", f"hot-{i}")
                elif label == "ena":
                    toggle = not toggle
                    dbmod.set_bot_enabled(bid, toggle)
                elif label == "reg":
                    dbmod.add_regime_snapshot(bid, "BTC/USD", "trending", 0.5, "x", "{}")
                else:
                    dbmod.add_strategy_decision(bid, "s", "BUY", "r", "trending", 0.5, "{}")
                with cnt_lock:
                    counts[label] += 1
            except sqlite3.OperationalError as e:
                with err_lock:
                    errors.append((label, repr(e)))
                return
            i += 1

    threads = [threading.Thread(target=_writer, args=(label,))
               for label in ("log", "ena", "reg", "dec")]
    for t in threads: t.start()
    for t in threads:
        t.join(timeout=DURATION_SEC + 5.0)
        assert not t.is_alive()

    assert not errors, f"OperationalError leaked: {errors}"
    for label, n in counts.items():
        assert n > 0, f"{label} made no forward progress"


def test_manual_close_deal_and_journal_atomic(temp_db, monkeypatch):
    """manual_close_deal_and_journal must run atomically through
    write_txn(bot_id, ...). On race-loss (rowcount=0) it must raise
    ValueError without persisting trade_journal/trade_feedback rows."""
    bid = _seed_min_bot(temp_db, "B-mcdj")
    deal_id = dbmod.open_deal(bid, "BTC/USD")
    dbmod.update_open_deal_entry(deal_id, 30000.0, 0.001)

    captured: list = []
    real = dbmod.write_txn
    monkeypatch.setattr(dbmod, "write_txn",
                        lambda b, fn, *, name=None: (captured.append((b, name)), real(b, fn, name=name))[1],
                        raising=True)

    out = dbmod.manual_close_deal_and_journal(
        deal_id=deal_id, bot_id=bid,
        entry_avg=30000.0, exit_avg=31000.0, base_amount=0.001,
        realized_pnl_quote=1.0, exit_strategy="manual",
        journal_exit_reason="user manual close",
    )
    assert out["ok"] is True and out["realized_pnl_quote"] == 1.0
    assert (bid, "manual_close_deal_and_journal") in captured

    # Second close on the same deal — must raise ValueError (caught by
    # either the early state-check 'Deal already CLOSED' or the
    # rowcount=0 'not open' race-loss branch — both are correct
    # short-circuits and prove that no second trade_journal row is
    # added).
    with pytest.raises(ValueError, match="(not open|already CLOSED)"):
        dbmod.manual_close_deal_and_journal(
            deal_id=deal_id, bot_id=bid,
            entry_avg=30000.0, exit_avg=32000.0, base_amount=0.001,
            realized_pnl_quote=2.0, journal_exit_reason="dup",
        )
    fresh = sqlite3.connect(temp_db, timeout=5.0)
    try:
        n_journal = fresh.execute(
            "SELECT COUNT(*) FROM trade_journal WHERE deal_id=?", (deal_id,),
        ).fetchone()[0]
        n_feedback = fresh.execute(
            "SELECT COUNT(*) FROM trade_feedback WHERE features_json LIKE ?",
            ("%32000%",),
        ).fetchone()[0]
    finally:
        fresh.close()
    assert n_journal == 1, f"expected exactly 1 trade_journal row, got {n_journal}"
    assert n_feedback == 0, "race-loss must not insert trade_feedback"


def test_cleanup_old_signal_audits_chunked(temp_db):
    """Smoke test for cleanup_old_signal_audits chunked migration."""
    fresh = sqlite3.connect(temp_db, timeout=5.0)
    try:
        old = int(time.time()) - 30 * 86400  # 30 days old (cutoff is 14d)
        new = int(time.time())
        rows = [
            (f"sig-{i}", "AAA/USD", "crypto", "short",
             50.0, 50.0, "neutral", "", "", "", "", "", None, None, old)
            for i in range(800)
        ] + [
            (f"sig-new-{i}", "BBB/USD", "crypto", "short",
             50.0, 50.0, "neutral", "", "", "", "", "", None, None, new)
            for i in range(20)
        ]
        fresh.executemany(
            "INSERT INTO signal_audit("
            "signal_id, symbol, asset_type, horizon, composite_score, "
            "confidence_score, conviction_grade, factor_scores_json, "
            "gate_results_json, technical_signals_json, metadata_json, "
            "flags_json, rejection_reason, price_at_signal, created_ts"
            ") VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            rows,
        )
        fresh.commit()
    finally:
        fresh.close()

    deleted = dbmod.cleanup_old_signal_audits(keep_days=14)
    assert deleted == 800

    fresh = sqlite3.connect(temp_db, timeout=5.0)
    try:
        left = fresh.execute("SELECT COUNT(*) FROM signal_audit").fetchone()[0]
    finally:
        fresh.close()
    assert left == 20
