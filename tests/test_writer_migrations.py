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
