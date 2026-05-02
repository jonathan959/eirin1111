"""Isolation tests for the Phase 1.2a chokepoint (db.write_txn).

These tests run BEFORE any caller migrates to write_txn (Phase 1.2b). Their
job is to lock the chokepoint's contract:
  * commits / rollbacks
  * retry-then-succeed
  * give-up → DBLockedError
  * lock model (per-bot vs global, parallelism, deadlock-free)
  * nested-call ban
  * SQL capture for diagnostics
  * open_migration_conn pragmas
  * WAL checkpoint thread lifecycle + chunked DELETE forward-progress under load

All tests use a tmp_path SQLite file (not :memory:) because WAL mode requires
a real file; per-thread connection caches are cleared between tests.

See audit/write_txn_design.md for the full design.
"""

from __future__ import annotations

import logging
import os
import sqlite3
import sys
import threading
import time
import types
from concurrent.futures import ThreadPoolExecutor

import pytest

import db as dbmod


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def temp_db(monkeypatch, tmp_path):
    """Fresh on-disk DB with a single test table; clears thread-local cache."""
    db_file = tmp_path / "wt_test.sqlite3"
    # Clear any thread-local cached conns pointing at the previous DB.
    dbmod._tl.__dict__.clear()
    monkeypatch.setattr(dbmod, "DB_NAME", str(db_file), raising=True)

    # Bootstrap a tiny schema (we don't need init_db for chokepoint tests).
    con = dbmod._make_real_conn()
    try:
        con.execute(
            "CREATE TABLE IF NOT EXISTS t (id INTEGER PRIMARY KEY AUTOINCREMENT, v INTEGER)"
        )
        con.execute(
            "CREATE TABLE IF NOT EXISTS bot_logs ("
            " id INTEGER PRIMARY KEY AUTOINCREMENT,"
            " bot_id INTEGER, ts INTEGER, level TEXT, category TEXT, message TEXT, count INTEGER)"
        )
        con.commit()
    finally:
        con.close()
    yield str(db_file)

    # Stop checkpoint thread if any test started it without stopping.
    dbmod.stop_wal_checkpoint_thread(timeout_sec=2.0)
    dbmod._tl.__dict__.clear()
    # Drop registries so tests don't bleed locks/state between runs.
    dbmod._bot_locks.clear()
    for suffix in ("", "-wal", "-shm"):
        try:
            os.unlink(str(db_file) + suffix)
        except OSError:
            pass


def _ensure_table_exists(db_file: str) -> None:
    """Sanity check that the on-disk DB has our test table."""
    con = sqlite3.connect(db_file, timeout=5.0)
    try:
        rows = con.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='t'"
        ).fetchall()
        assert rows, "test schema missing"
    finally:
        con.close()


# ---------------------------------------------------------------------------
# 1. write_txn returns the fn result
# ---------------------------------------------------------------------------

def test_write_txn_returns_fn_result(temp_db):
    def _fn(con):
        con.execute("INSERT INTO t(v) VALUES (?)", (42,))
        return 42

    out = dbmod.write_txn(None, _fn, name="ins42")
    assert out == 42


# ---------------------------------------------------------------------------
# 2. write_txn commits on success — verified via a fresh connection
# ---------------------------------------------------------------------------

def test_write_txn_commits_on_success(temp_db):
    def _fn(con):
        con.execute("INSERT INTO t(v) VALUES (?)", (7,))

    dbmod.write_txn(None, _fn, name="ins7")

    fresh = sqlite3.connect(temp_db, timeout=5.0)
    try:
        rows = fresh.execute("SELECT v FROM t").fetchall()
    finally:
        fresh.close()
    assert [r[0] for r in rows] == [7]


# ---------------------------------------------------------------------------
# 3. write_txn rolls back on exception, conn stays usable
# ---------------------------------------------------------------------------

def test_write_txn_rolls_back_on_exception(temp_db):
    def _bad(con):
        con.execute("INSERT INTO t(v) VALUES (?)", (1,))
        raise ValueError("boom")

    with pytest.raises(ValueError):
        dbmod.write_txn(None, _bad, name="rollback_test")

    # Row must NOT exist.
    fresh = sqlite3.connect(temp_db, timeout=5.0)
    try:
        rows = fresh.execute("SELECT v FROM t").fetchall()
    finally:
        fresh.close()
    assert rows == []

    # Subsequent write_txn on the same thread must still succeed (proves the
    # cached conn was rolled back to a clean state, not poisoned).
    def _good(con):
        con.execute("INSERT INTO t(v) VALUES (?)", (99,))

    dbmod.write_txn(None, _good, name="post_rollback_ok")

    fresh = sqlite3.connect(temp_db, timeout=5.0)
    try:
        rows = fresh.execute("SELECT v FROM t").fetchall()
    finally:
        fresh.close()
    assert [r[0] for r in rows] == [99]


# ---------------------------------------------------------------------------
# 4. fn calling commit() itself is harmless (commit is idempotent on no pending)
# ---------------------------------------------------------------------------

def test_write_txn_no_implicit_commit_in_fn(temp_db):
    """If fn commits inside, write_txn's outer commit is a no-op. Document this."""
    def _fn(con):
        con.execute("INSERT INTO t(v) VALUES (?)", (1,))
        con.commit()  # discouraged but harmless
        con.execute("INSERT INTO t(v) VALUES (?)", (2,))

    dbmod.write_txn(None, _fn, name="double_commit")

    fresh = sqlite3.connect(temp_db, timeout=5.0)
    try:
        rows = fresh.execute("SELECT v FROM t ORDER BY id").fetchall()
    finally:
        fresh.close()
    assert [r[0] for r in rows] == [1, 2]


# ---------------------------------------------------------------------------
# 5. retries on "database is locked" then succeeds
# ---------------------------------------------------------------------------

def test_write_txn_retries_on_database_locked_then_succeeds(temp_db, caplog, monkeypatch):
    """Inject a transient OperationalError("database is locked") for first 2
    attempts; the 3rd succeeds. Assert exactly 2 retry log lines."""
    caplog.set_level(logging.WARNING, logger="db")
    # Speed up the retry sleeps to keep the test fast.
    monkeypatch.setattr(dbmod, "_RETRY_SCHEDULE_MS", (1, 1, 1, 1, 1), raising=True)

    state = {"calls": 0}

    def _fn(con):
        state["calls"] += 1
        if state["calls"] < 3:
            raise sqlite3.OperationalError("database is locked")
        con.execute("INSERT INTO t(v) VALUES (?)", (state["calls"],))

    dbmod.write_txn(None, _fn, name="flaky_writer")

    assert state["calls"] == 3
    retry_logs = [r for r in caplog.records if "write_txn retry" in r.getMessage()]
    assert len(retry_logs) == 2, [r.getMessage() for r in retry_logs]


# ---------------------------------------------------------------------------
# 6. exhausts retries → DBLockedError with full context
# ---------------------------------------------------------------------------

def test_write_txn_raises_DBLockedError_after_retries(temp_db, monkeypatch):
    monkeypatch.setattr(dbmod, "_RETRY_SCHEDULE_MS", (1, 1, 1, 1, 1), raising=True)
    state = {"calls": 0}

    def _fn(con):
        state["calls"] += 1
        # Exercise the SQL-tracking proxy before raising.
        try:
            con.execute("UPDATE t SET v = v + 1 WHERE id = 999999")
        except Exception:
            pass
        raise sqlite3.OperationalError("database is locked")

    with pytest.raises(dbmod.DBLockedError) as exc_info:
        dbmod.write_txn(7, _fn, name="always_locked")

    err = exc_info.value
    assert state["calls"] == len(dbmod._RETRY_SCHEDULE_MS) + 1  # 6 attempts total
    assert err.attempts == state["calls"]
    assert err.bot_id == 7
    assert err.op_name == "always_locked"
    assert err.elapsed_ms >= 0
    assert "UPDATE t SET v = v + 1" in (err.last_sql or "")


def test_DBLockedError_is_OperationalError():
    err = dbmod.DBLockedError(
        bot_id=1, op_name="x", attempts=6, elapsed_ms=10,
        last_sql="UPDATE t SET v=1", last_exc=sqlite3.OperationalError("database is locked"),
    )
    assert isinstance(err, sqlite3.OperationalError)


def test_DBLockedError_str_format():
    err = dbmod.DBLockedError(
        bot_id=42, op_name="add_log", attempts=6, elapsed_ms=2300,
        last_sql="INSERT INTO bot_logs(...) VALUES(?)",
        last_exc=sqlite3.OperationalError("database is locked"),
    )
    s = str(err)
    assert "DBLockedError" in s
    assert "op='add_log'" in s
    assert "bot_id=42" in s
    assert "attempts=6" in s
    assert "elapsed_ms=2300" in s
    assert "INSERT INTO bot_logs" in s


def test_DBLockedError_truncates_long_sql():
    big_sql = "X" * 10_000
    err = dbmod.DBLockedError(
        bot_id=None, op_name="x", attempts=1, elapsed_ms=0,
        last_sql=big_sql, last_exc=sqlite3.OperationalError("locked"),
    )
    assert len(err.last_sql) == 512


# ---------------------------------------------------------------------------
# 7. non-locked OperationalError is NOT retried
# ---------------------------------------------------------------------------

def test_write_txn_does_not_retry_on_other_OperationalError(temp_db, monkeypatch):
    monkeypatch.setattr(dbmod, "_RETRY_SCHEDULE_MS", (1, 1, 1, 1, 1), raising=True)
    state = {"calls": 0}

    def _fn(con):
        state["calls"] += 1
        raise sqlite3.OperationalError("no such table: missing_table")

    with pytest.raises(sqlite3.OperationalError) as exc_info:
        dbmod.write_txn(None, _fn, name="bad_sql")

    assert state["calls"] == 1
    assert "no such table" in str(exc_info.value).lower()
    assert not isinstance(exc_info.value, dbmod.DBLockedError)


# ---------------------------------------------------------------------------
# 8. per-bot lock serialises writers for the same bot_id
# ---------------------------------------------------------------------------

def test_write_txn_per_bot_lock_serialises(temp_db):
    trace: list = []
    trace_lock = threading.Lock()

    def _fn(con):
        with trace_lock:
            trace.append(("enter", threading.get_ident(), time.monotonic()))
        # Hold inside the txn long enough that any concurrent acquirer would
        # interleave if locks were broken.
        time.sleep(0.05)
        with trace_lock:
            trace.append(("exit", threading.get_ident(), time.monotonic()))
        con.execute("INSERT INTO t(v) VALUES (?)", (1,))

    def _worker(_):
        dbmod.write_txn(123, _fn, name="ser_test")

    with ThreadPoolExecutor(max_workers=10) as ex:
        list(ex.map(_worker, range(10)))

    # Strict serialisation: every (enter, exit) pair must be contiguous.
    for i in range(0, len(trace), 2):
        assert trace[i][0] == "enter"
        assert trace[i + 1][0] == "exit"
        assert trace[i][1] == trace[i + 1][1], "interleaved threads in trace"


# ---------------------------------------------------------------------------
# 9. independent bots do not block each other (parallelism preserved)
# ---------------------------------------------------------------------------

def test_write_txn_independent_bots_do_not_block_each_other(temp_db):
    barrier = threading.Barrier(4)
    started_at: list = []
    started_lock = threading.Lock()

    def _fn(con):
        # Record entry time and hold for 100ms.
        with started_lock:
            started_at.append(time.monotonic())
        time.sleep(0.10)
        con.execute("INSERT INTO t(v) VALUES (?)", (1,))

    def _worker(bot_id):
        barrier.wait()
        dbmod.write_txn(bot_id, _fn, name="parallel_test")

    t0 = time.monotonic()
    with ThreadPoolExecutor(max_workers=4) as ex:
        list(ex.map(_worker, [1, 2, 3, 4]))
    elapsed = time.monotonic() - t0

    # If serialised, elapsed >= 4*0.10 = 0.40s. If parallel, ~0.10s.
    # SQLite's writer is still serialised at the file level, so we expect
    # somewhere between 0.10s (true parallel) and 0.40s (serial). The Python
    # locks should not add cross-bot serialisation — meaning enter times
    # should be tightly clustered.
    assert elapsed < 0.40, f"writers serialised across bots ({elapsed:.3f}s)"
    spread = max(started_at) - min(started_at)
    assert spread < 0.05, f"per-bot locks blocked entry: spread={spread:.3f}s"


# ---------------------------------------------------------------------------
# 10. global lock serialises bot_id=None writers
# ---------------------------------------------------------------------------

def test_write_txn_global_lock_serialises(temp_db):
    trace: list = []
    trace_lock = threading.Lock()

    def _fn(con):
        with trace_lock:
            trace.append(("enter", threading.get_ident()))
        time.sleep(0.02)
        with trace_lock:
            trace.append(("exit", threading.get_ident()))
        con.execute("INSERT INTO t(v) VALUES (?)", (1,))

    with ThreadPoolExecutor(max_workers=8) as ex:
        list(ex.map(lambda _: dbmod.write_txn(None, _fn, name="g"), range(8)))

    for i in range(0, len(trace), 2):
        assert trace[i][0] == "enter"
        assert trace[i + 1][0] == "exit"
        assert trace[i][1] == trace[i + 1][1]


# ---------------------------------------------------------------------------
# 11. per-bot and global writers can both make progress without deadlock
# ---------------------------------------------------------------------------

def test_write_txn_per_bot_and_global_independent(temp_db):
    """Soak: 4 per-bot writers + 4 global writers all run for a fixed period.
    Assert both sets make >0 forward progress and no deadlock occurs."""
    stop = threading.Event()
    counts = {"bot1": 0, "bot2": 0, "global_a": 0, "global_b": 0}
    cnt_lock = threading.Lock()

    def _writer(label, bot_id):
        while not stop.is_set():
            def _fn(con):
                con.execute("INSERT INTO t(v) VALUES (?)", (1,))
            try:
                dbmod.write_txn(bot_id, _fn, name=f"writer_{label}")
                with cnt_lock:
                    counts[label] += 1
            except Exception:  # noqa: BLE001 - test-side broad catch is OK
                pass

    threads = [
        threading.Thread(target=_writer, args=("bot1", 1)),
        threading.Thread(target=_writer, args=("bot2", 2)),
        threading.Thread(target=_writer, args=("global_a", None)),
        threading.Thread(target=_writer, args=("global_b", None)),
    ]
    for t in threads:
        t.start()
    time.sleep(1.0)
    stop.set()
    for t in threads:
        t.join(timeout=2.0)
        assert not t.is_alive(), "writer thread did not exit (deadlock?)"

    for label, n in counts.items():
        assert n > 0, f"{label} made no progress (deadlock?)"


# ---------------------------------------------------------------------------
# 12. nested write_txn raises RuntimeError
# ---------------------------------------------------------------------------

def test_write_txn_nested_call_raises_RuntimeError(temp_db):
    def _outer(con):
        def _inner(con2):
            con2.execute("INSERT INTO t(v) VALUES (?)", (1,))
        # This must raise — nested write_txn is a programming bug.
        dbmod.write_txn(None, _inner, name="inner")

    with pytest.raises(RuntimeError, match="nested write_txn"):
        dbmod.write_txn(None, _outer, name="outer")


# ---------------------------------------------------------------------------
# 13. open_migration_conn has the canonical PRAGMAs
# ---------------------------------------------------------------------------

def test_open_migration_conn_has_pragmas(temp_db):
    con = dbmod.open_migration_conn()
    try:
        mode = con.execute("PRAGMA journal_mode").fetchone()[0]
        timeout = con.execute("PRAGMA busy_timeout").fetchone()[0]
        sync = con.execute("PRAGMA synchronous").fetchone()[0]
        fk = con.execute("PRAGMA foreign_keys").fetchone()[0]
    finally:
        con.close()
    assert str(mode).lower() == "wal"
    assert int(timeout) >= 30000
    assert int(sync) == 1  # NORMAL
    assert int(fk) == 1


# ---------------------------------------------------------------------------
# 14. WAL checkpoint thread starts and stops cleanly
# ---------------------------------------------------------------------------

def test_wal_checkpoint_thread_starts_and_stops(temp_db):
    dbmod.start_wal_checkpoint_thread(interval_sec=1)
    # Idempotent: second call is a no-op.
    dbmod.start_wal_checkpoint_thread(interval_sec=1)
    time.sleep(0.5)
    dbmod.stop_wal_checkpoint_thread(timeout_sec=2.0)
    # After stop, the module-level handle is None.
    assert dbmod._wal_checkpoint_thread is None


def test_wal_checkpoint_runs_at_least_once(temp_db, caplog):
    """Generate enough WAL traffic to be visible, then verify the checkpoint
    log line was emitted by the background thread."""
    caplog.set_level(logging.DEBUG, logger="db")

    # Drive WAL growth.
    def _ins(con):
        for _ in range(100):
            con.execute("INSERT INTO t(v) VALUES (?)", (1,))

    dbmod.write_txn(None, _ins, name="seed_wal")

    dbmod.start_wal_checkpoint_thread(interval_sec=1)
    time.sleep(1.5)
    dbmod.stop_wal_checkpoint_thread(timeout_sec=2.0)

    cp_logs = [r for r in caplog.records if "WAL checkpoint:" in r.getMessage()]
    assert cp_logs, "no WAL checkpoint log line emitted"


# ---------------------------------------------------------------------------
# 15. chunked DELETE in write_txn does not starve concurrent INSERT writers
# ---------------------------------------------------------------------------

def test_chunked_cleanup_yields_writer_slot(temp_db):
    """Regression test for the bot-1 lock loop.

    Run a producer that INSERTs into bot_logs as fast as it can; concurrently
    run a chunked cleanup (DELETE LIMIT 500 per write_txn). Assert:
      * zero OperationalError propagates from either side
      * cleanup makes forward progress every batch
      * producer's INSERT count grows during the cleanup window
    """
    # Seed a chunk of "old" rows the cleanup will target.
    cutoff = int(time.time())
    seed = [(1, cutoff - 86400, "INFO", "S", "old", 1) for _ in range(2000)]

    def _seed(con):
        con.executemany(
            "INSERT INTO bot_logs(bot_id, ts, level, category, message, count) VALUES (?,?,?,?,?,?)",
            seed,
        )

    dbmod.write_txn(None, _seed, name="seed_old_rows")

    producer_ok = threading.Event()
    producer_inserts = {"n": 0}
    producer_errors: list = []
    cleanup_batches: list = []
    cleanup_errors: list = []

    def _producer():
        producer_ok.wait()
        deadline = time.monotonic() + 2.0
        while time.monotonic() < deadline:
            def _ins(con):
                con.execute(
                    "INSERT INTO bot_logs(bot_id, ts, level, category, message, count) "
                    "VALUES (?,?,?,?,?,?)",
                    (1, int(time.time()), "INFO", "S", "fresh", 1),
                )
            try:
                dbmod.write_txn(1, _ins, name="prod_insert")
                producer_inserts["n"] += 1
            except sqlite3.OperationalError as e:
                producer_errors.append(repr(e))
                break

    def _chunked_cleanup():
        producer_ok.wait()
        BATCH = 500
        while True:
            def _del(con):
                cur = con.execute(
                    "DELETE FROM bot_logs WHERE ts < ? AND rowid IN "
                    "(SELECT rowid FROM bot_logs WHERE ts < ? LIMIT ?)",
                    (cutoff, cutoff, BATCH),
                )
                return int(cur.rowcount or 0)
            try:
                n = dbmod.write_txn(None, _del, name="cleanup_chunk")
            except sqlite3.OperationalError as e:
                cleanup_errors.append(repr(e))
                break
            cleanup_batches.append(n)
            if n == 0:
                break
            time.sleep(0.05)

    t1 = threading.Thread(target=_producer)
    t2 = threading.Thread(target=_chunked_cleanup)
    t1.start()
    t2.start()
    producer_ok.set()
    t1.join(timeout=10.0)
    t2.join(timeout=10.0)

    assert not producer_errors, f"producer hit OperationalError: {producer_errors}"
    assert not cleanup_errors, f"cleanup hit OperationalError: {cleanup_errors}"
    assert sum(cleanup_batches) > 0, "cleanup made no forward progress"
    assert producer_inserts["n"] > 0, "producer made no forward progress during cleanup"


# ---------------------------------------------------------------------------
# 16. SQL capture works for both .execute and cursor.execute
# ---------------------------------------------------------------------------

def test_sql_capture_via_execute_and_cursor(temp_db, monkeypatch):
    monkeypatch.setattr(dbmod, "_RETRY_SCHEDULE_MS", (1, 1, 1, 1, 1), raising=True)

    seen: list = []

    def _fn_execute(_con):
        # Hit the SQL capture once via con.execute, then raise locked so the
        # captured SQL flows into DBLockedError.last_sql.
        _con.execute("UPDATE t SET v = ?", (123,))
        raise sqlite3.OperationalError("database is locked")

    with pytest.raises(dbmod.DBLockedError) as exc1:
        dbmod.write_txn(None, _fn_execute, name="capture_via_execute")
    assert "UPDATE t SET v = ?" in (exc1.value.last_sql or "")

    def _fn_cursor(con):
        cur = con.cursor()
        cur.execute("SELECT 1")
        cur.execute("INSERT INTO t(v) VALUES (?)", (5,))
        raise sqlite3.OperationalError("database is locked")

    with pytest.raises(dbmod.DBLockedError) as exc2:
        dbmod.write_txn(None, _fn_cursor, name="capture_via_cursor")
    assert "INSERT INTO t" in (exc2.value.last_sql or "")


# ---------------------------------------------------------------------------
# 17. RLock allows same-thread re-acquire (defense-in-depth)
# ---------------------------------------------------------------------------

def test_bot_db_lock_is_reentrant(temp_db):
    lk = dbmod.bot_db_lock(99)
    with lk:
        # Re-acquire from the same thread — must NOT deadlock.
        with lk:
            pass


# ---------------------------------------------------------------------------
# 18. BotManager.bot_db_lock delegates to db.bot_db_lock (single registry)
# ---------------------------------------------------------------------------

def test_botmanager_bot_db_lock_delegates_to_db_module():
    """Without spinning up a real BotManager (heavy), assert the delegate
    contract by importing the manager class and verifying its bot_db_lock
    returns the same RLock instance as db.bot_db_lock."""
    from bot_manager import BotManager

    # Skeleton instance: avoid full __init__ side effects.
    bm = BotManager.__new__(BotManager)

    direct = dbmod.bot_db_lock(101)
    delegated = bm.bot_db_lock(101)
    assert delegated is direct, "BotManager.bot_db_lock must return the canonical RLock"

    # Different bot ids → different locks.
    other = dbmod.bot_db_lock(102)
    assert other is not direct
