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
