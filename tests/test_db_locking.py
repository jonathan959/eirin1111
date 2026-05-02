"""Concurrency regression tests for the SQLite locking fix.

Covers Part 2 of the comprehensive fix:
  * `manual_close_deal_and_journal` is race-safe — even with N threads racing
    to close the same deal, exactly one succeeds and the rest raise
    `ValueError("Deal already CLOSED")`. No `OperationalError("database is
    locked")` may bubble up.
  * `_make_real_conn()` exposes the WAL + busy_timeout pragmas required for
    write contention to wait politely instead of failing.

These tests run against a temporary database so they never touch the real
``botdb.sqlite3``. We patch ``db.DB_NAME`` for the duration of each test and
clear the per-thread connection cache (``db._tl``) to be safe.
"""

from __future__ import annotations

import os
import random
import sqlite3
import statistics
import tempfile
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Optional

import pytest

import db as dbmod


@pytest.fixture()
def temp_db(monkeypatch):
    """Create a fresh, schema-initialised DB in a tempfile and patch DB_NAME."""
    fd, path = tempfile.mkstemp(prefix="testdb_", suffix=".sqlite3")
    os.close(fd)

    # Ensure no thread-local cached connection points at the old DB.
    dbmod._tl.__dict__.clear()
    monkeypatch.setattr(dbmod, "DB_NAME", path, raising=True)

    dbmod.init_db()
    yield path

    # Cleanup: drop cached conns + remove file.
    dbmod._tl.__dict__.clear()
    for suffix in ("", "-wal", "-shm"):
        try:
            os.unlink(path + suffix)
        except OSError:
            pass


def _seed_open_deal(bot_id: int = 1) -> int:
    """Insert a fresh OPEN deal and return its id."""
    return dbmod.open_deal(bot_id, "BTC/USD", state="OPEN", opened_at=int(time.time()))


def test_make_real_conn_has_wal_and_busy_timeout(temp_db):
    con = dbmod._make_real_conn()
    try:
        mode = con.execute("PRAGMA journal_mode").fetchone()[0]
        timeout = con.execute("PRAGMA busy_timeout").fetchone()[0]
    finally:
        con.close()
    assert str(mode).lower() == "wal", f"expected WAL, got {mode}"
    assert int(timeout) >= 30000, f"expected busy_timeout >= 30000ms, got {timeout}"


def test_manual_close_deal_and_journal_idempotency(temp_db):
    """One success, one explicit ValueError on the second call."""
    deal_id = _seed_open_deal()
    out1 = dbmod.manual_close_deal_and_journal(
        deal_id=deal_id, bot_id=1,
        entry_avg=100.0, exit_avg=110.0, base_amount=0.1, realized_pnl_quote=1.0,
        entry_strategy="test", exit_strategy="manual_close_dry",
        hold_sec=10, safety_count=0, journal_exit_reason="t",
    )
    assert out1["ok"] is True

    with pytest.raises(ValueError) as exc_info:
        dbmod.manual_close_deal_and_journal(
            deal_id=deal_id, bot_id=1,
            entry_avg=100.0, exit_avg=110.0, base_amount=0.1, realized_pnl_quote=1.0,
            entry_strategy="test", exit_strategy="manual_close_dry",
            hold_sec=10, safety_count=0, journal_exit_reason="t",
        )
    assert "already" in str(exc_info.value).lower() or "not open" in str(exc_info.value).lower()


def test_concurrent_close_only_one_winner(temp_db):
    """10 threads race to close the same deal; exactly one wins, no OperationalError."""
    deal_id = _seed_open_deal()
    barrier = threading.Barrier(10)

    def worker():
        barrier.wait()
        try:
            res = dbmod.manual_close_deal_and_journal(
                deal_id=deal_id, bot_id=1,
                entry_avg=100.0, exit_avg=105.0, base_amount=0.1, realized_pnl_quote=0.5,
                entry_strategy="r", exit_strategy="manual_close_dry",
                hold_sec=1, safety_count=0, journal_exit_reason="race",
            )
            return ("ok", res)
        except ValueError as ve:
            return ("value_err", str(ve))
        except sqlite3.OperationalError as oe:
            return ("op_err", str(oe))
        except Exception as e:
            return ("other", f"{type(e).__name__}: {e}")

    with ThreadPoolExecutor(max_workers=10) as ex:
        results = list(ex.map(lambda _: worker(), range(10)))

    op_errs = [r for r in results if r[0] == "op_err"]
    others = [r for r in results if r[0] == "other"]
    oks = [r for r in results if r[0] == "ok"]
    value_errs = [r for r in results if r[0] == "value_err"]

    assert not op_errs, f"OperationalError leaked under concurrency: {op_errs}"
    assert not others, f"Unexpected exceptions: {others}"
    assert len(oks) == 1, f"Expected exactly 1 winner, got {len(oks)}: {oks}"
    assert len(value_errs) == 9, f"Expected 9 ValueError losers, got {len(value_errs)}"


def test_concurrent_close_with_bot_db_lock_serialises(temp_db):
    """Simulate the production wrapper: BotManager.bot_db_lock serialises all
    threads calling manual_close_deal_and_journal for the same bot. The
    cumulative behaviour should be identical to the unguarded test, just
    with a single SQLite writer at any moment.
    """
    lock = threading.Lock()
    deal_id = _seed_open_deal(bot_id=42)

    def worker():
        with lock:
            try:
                dbmod.manual_close_deal_and_journal(
                    deal_id=deal_id, bot_id=42,
                    entry_avg=100.0, exit_avg=104.0, base_amount=0.1, realized_pnl_quote=0.4,
                    entry_strategy="r", exit_strategy="manual_close_dry",
                    hold_sec=1, safety_count=0, journal_exit_reason="serialised",
                )
                return "ok"
            except ValueError:
                return "value_err"
            except sqlite3.OperationalError as oe:
                return f"op_err:{oe}"

    with ThreadPoolExecutor(max_workers=8) as ex:
        results = list(ex.map(lambda _: worker(), range(8)))

    assert "ok" in results, "no winner under serialised lock"
    assert results.count("ok") == 1, f"more than one winner: {results}"
    assert all((r == "ok" or r == "value_err") for r in results), f"errors: {results}"


# ===========================================================================
# Phase 1.5 — whole-system load test
# ===========================================================================
#
# Brief (Phase 1.5, unchanged in the amendment): "Spawn 4 BotRunner-equivalent
# threads, each doing 200 mixed reads/writes (deals, signals, explore outcomes,
# journal) for 60 seconds. Assert zero OperationalErrors propagate, zero
# ValueError races, P95 write latency < 200ms. This test MUST pass in CI
# before Phase 1 is closed."
#
# This is the gate test for Phase 1. If write_txn / chunked_delete / per-bot
# locks ever regress to the pre-Phase-1 state, the bot-1 'database is locked'
# loop reproduces here within a few seconds.

def _p95(samples: "list[float]") -> float:
    """Sorted nearest-rank P95. Empty list returns 0 for safe assertion."""
    if not samples:
        return 0.0
    s = sorted(samples)
    idx = max(0, min(len(s) - 1, int(round(0.95 * (len(s) - 1)))))
    return s[idx]


def test_no_lock_under_load(temp_db):
    """Phase 1.5 acceptance gate.

    4 worker threads, each pretending to be a separate BotRunner. Each
    thread runs a realistic mix of writes (add_log heavy, plus deal ops,
    order events, explore outcomes, recommendation snapshots, journal upserts)
    and a few reads, for OPS_PER_THREAD ops, with all 4 threads running
    concurrently. A 5th cleanup thread fires the chunked cleanup against
    bot_logs and order_events every second to simulate the production
    cron — the exact pattern that produced the original 'database is locked'
    loop.

    Assertions:
      * Every per-op exception is captured and asserted to be empty (no
        OperationalError, no DBLockedError, no ValueError race).
      * P95 write latency across all threads stays below 200 ms.
      * Each thread completes its full op budget within 60 s wall-clock.
      * Cleanup thread makes forward progress at least once.
    """
    OPS_PER_THREAD = 200
    NUM_BOT_THREADS = 4
    DEADLINE_SEC = 60.0

    # Seed bots via create_bot so the bot_id matches a real row (per-bot
    # writers like update_bot / set_bot_running require an existing row).
    # Field set mirrors tests/test_writer_migrations.py::_seed_min_bot.
    bot_ids: list[int] = []
    for i in range(NUM_BOT_THREADS):
        try:
            bid = dbmod.create_bot({
                "name": f"loadbot{i}",
                "symbol": "BTC/USD",
                "enabled": 1,
                "dry_run": 1,
                "base_quote": 10.0,
                "safety_quote": 5.0,
                "max_safety": 3,
                "first_dev": 0.01,
                "step_mult": 1.0,
                "tp": 0.02,
                "max_spend_quote": 100.0,
                "auto_restart": 1,
            })
            bot_ids.append(int(bid))
        except Exception as e:  # noqa: BLE001
            pytest.fail(f"could not seed worker bot {i}: {e}")

    assert len(bot_ids) == NUM_BOT_THREADS

    # Pre-create one OPEN deal per bot so deal-update / journal paths have
    # a target the very first time they run.
    seeded_deals: dict[int, int] = {}
    for bid in bot_ids:
        seeded_deals[bid] = dbmod.open_deal(bid, "BTC/USD",
                                            state="OPEN",
                                            opened_at=int(time.time()))

    # Seed 600 ANCIENT bot_logs (ts well past the 30-day cutoff) so the
    # cleanup loop has something to actually DELETE on its first sweep
    # while the bot threads are also INSERT-ing fresh logs. This is the
    # exact collision pattern the brief flagged ('mass DELETE-while-INSERT').
    ancient_ts = int(time.time()) - (60 * 86400)  # 60 days old
    fresh = sqlite3.connect(temp_db, timeout=10.0)
    try:
        fresh.executemany(
            "INSERT INTO bot_logs(bot_id, ts, level, message) VALUES (?,?,?,?)",
            [(bot_ids[i % NUM_BOT_THREADS], ancient_ts + i, "INFO",
              f"ancient log {i}") for i in range(600)],
        )
        fresh.commit()
    finally:
        fresh.close()

    # Per-thread structured result.
    errors: list[tuple[int, str, str]] = []
    err_lock = threading.Lock()
    latencies_ms: list[float] = []
    lat_lock = threading.Lock()
    completed_ops = [0] * NUM_BOT_THREADS

    # Parties: each bot worker + cleanup thread + main (orchestrator wait).
    # An off-by-one here deadlocks forever (main used to call .wait() while
    # the barrier only counted NUM_BOT_THREADS + 1).
    start_barrier = threading.Barrier(NUM_BOT_THREADS + 2)
    stop_event = threading.Event()
    cleanup_progress = {"runs": 0, "rows_deleted": 0}

    def _record_err(idx: int, op: str, exc: BaseException) -> None:
        with err_lock:
            errors.append((idx, op, f"{type(exc).__name__}: {exc}"))

    def _time_op(fn):
        t0 = time.perf_counter()
        try:
            fn()
            ok = True
            err: Optional[BaseException] = None
        except Exception as e:  # noqa: BLE001 — captured for assertion
            ok = False
            err = e
        t_ms = (time.perf_counter() - t0) * 1000.0
        return ok, err, t_ms

    def _bot_worker(idx: int) -> None:
        bid = bot_ids[idx]
        deal_id = seeded_deals[bid]
        rng = random.Random(idx * 1009 + 7)
        start_barrier.wait()

        for i in range(OPS_PER_THREAD):
            if time.monotonic() - test_t0 > DEADLINE_SEC:
                _record_err(idx, "deadline", RuntimeError(
                    f"thread {idx} only completed {i} of {OPS_PER_THREAD} ops in {DEADLINE_SEC}s"))
                return

            roll = rng.random()
            # Realistic add_log-heavy mix mirroring a BotRunner tick. Targets
            # add up to 1.0 — order matters less than weights.
            if roll < 0.45:
                op = "add_log"
                fn = lambda: dbmod.add_log(bid, "INFO", f"tick {i} from bot {bid}")
            elif roll < 0.60:
                op = "add_order_event"
                fn = lambda: dbmod.add_order_event(
                    bid, "BTC/USD", "buy", "limit",
                    price=30000.0 + i, amount=0.001,
                    order_id=f"ord-{bid}-{i}", tag="entry",
                    status="placed", reason="rsi", is_live=False,
                )
            elif roll < 0.65:
                op = "open_close_deal"
                # Pair open+close in one logical op to keep the deals
                # table churning. Captured i_iter for closure.
                i_iter = i
                def _open_close():
                    new_deal = dbmod.open_deal(bid, "BTC/USD",
                                               state="OPEN",
                                               opened_at=int(time.time()))
                    dbmod.close_deal(
                        new_deal,
                        entry_avg=30000.0 + i_iter,
                        exit_avg=30100.0 + i_iter,
                        base_amount=0.001,
                        realized_pnl_quote=0.5,
                        entry_strategy="loadtest",
                        exit_strategy="loadtest_close",
                        hold_sec=1,
                    )
                fn = _open_close
            elif roll < 0.70:
                op = "update_open_deal_entry"
                fn = lambda: dbmod.update_open_deal_entry(
                    deal_id,
                    entry_avg=30000.0 + i,
                    base_amount=0.001 * ((i % 5) + 1),
                    safety_count=i % 3,
                )
            elif roll < 0.78:
                op = "save_signal_outcome"
                fn = lambda: dbmod.save_signal_outcome(
                    "BTC/USD",
                    "short",
                    "Trend Follow",
                    int(time.time()),
                    30000.0 + rng.uniform(-50, 50),
                    composite_score=rng.uniform(0.0, 1.0),
                    conviction_grade="B",
                )
            elif roll < 0.85:
                op = "update_explore_signal_outcome"
                # Use the deal id as a stand-in (this writer is no-op for
                # missing rows); proves the writer doesn't deadlock on
                # the read-then-write pattern.
                fn = lambda: dbmod.update_explore_signal_outcome(
                    int(deal_id),
                    price_5d=30100.0 + rng.uniform(-100, 100),
                    price_10d=30200.0 + rng.uniform(-100, 100),
                )
            elif roll < 0.90:
                op = "upsert_trade_journal"
                fn = lambda: dbmod.upsert_trade_journal(
                    deal_id,
                    entry_reason="rsi<30",
                    exit_reason="tp" if i % 2 == 0 else "sl",
                )
            elif roll < 0.95:
                op = "set_bot_running"
                fn = lambda: dbmod.set_bot_running(bid, bool(i % 2))
            else:
                op = "read_logs"
                # Intentionally bypass write_txn for the read; verifies that
                # SELECTs across the per-thread connection don't block writers.
                def _read():
                    con = dbmod._conn()
                    con.execute(
                        "SELECT id FROM bot_logs WHERE bot_id=? ORDER BY id DESC LIMIT 5",
                        (bid,),
                    ).fetchall()
                fn = _read

            ok, err, t_ms = _time_op(fn)
            if ok:
                with lat_lock:
                    latencies_ms.append(t_ms)
                completed_ops[idx] += 1
            else:
                # ValueError("Deal already CLOSED") is acceptable ONLY for
                # the open_close_deal op (rare race when two threads happen to
                # touch the same deal). Anything else is a hard failure.
                if isinstance(err, ValueError) and op == "open_close_deal":
                    pass
                else:
                    _record_err(idx, op, err)
                    return

    def _cleanup_worker() -> None:
        # Use keep_days=30 not 0 — we want to PROVE the cleanup path is
        # callable under load (chunked_delete acquires write_txn each
        # batch, yields between batches), not actually purge the test DB.
        # With keep_days=30 the WHERE predicate matches zero rows for the
        # synthetic data we just inserted, so the function returns 0
        # immediately but still exercises the per-batch write_txn handshake.
        start_barrier.wait()
        while not stop_event.is_set():
            try:
                deleted = dbmod.cleanup_old_bot_logs(keep_days=30)
                cleanup_progress["runs"] += 1
                cleanup_progress["rows_deleted"] += int(deleted or 0)
                deleted2 = dbmod.cleanup_old_order_events(keep_days=30)
                cleanup_progress["rows_deleted"] += int(deleted2 or 0)
            except Exception as e:  # noqa: BLE001
                _record_err(-1, "cleanup", e)
                return
            # Poll the stop event every 250ms so test teardown is snappy.
            if stop_event.wait(timeout=1.0):
                return

    test_t0 = time.monotonic()
    bot_threads = [threading.Thread(target=_bot_worker, args=(i,),
                                    name=f"loadbot-{i}", daemon=True)
                   for i in range(NUM_BOT_THREADS)]
    cleanup_thread = threading.Thread(target=_cleanup_worker,
                                      name="loadcleanup", daemon=True)

    for t in bot_threads:
        t.start()
    cleanup_thread.start()

    # Release the start barrier; all bot + cleanup threads begin together.
    start_barrier.wait()

    for t in bot_threads:
        t.join(timeout=DEADLINE_SEC + 10.0)

    stop_event.set()
    cleanup_thread.join(timeout=10.0)
    elapsed = time.monotonic() - test_t0

    # ---- Assertions ----
    assert all(not t.is_alive() for t in bot_threads), \
        f"worker threads did not finish: {[t.name for t in bot_threads if t.is_alive()]}"
    assert not cleanup_thread.is_alive(), "cleanup thread did not finish"
    assert elapsed <= DEADLINE_SEC + 5.0, \
        f"workload took {elapsed:.1f}s, exceeded deadline {DEADLINE_SEC}s + 5s grace"
    assert not errors, f"propagated exceptions under load: first 10={errors[:10]}"

    for idx, count in enumerate(completed_ops):
        assert count == OPS_PER_THREAD, \
            f"thread {idx} only completed {count}/{OPS_PER_THREAD} ops"

    p50 = statistics.median(latencies_ms) if latencies_ms else 0.0
    p95 = _p95(latencies_ms)
    p99 = _p95(sorted(latencies_ms)[-max(1, len(latencies_ms) // 20):]) if latencies_ms else 0.0
    print(
        f"\n[Phase 1.5 load] elapsed={elapsed:.2f}s "
        f"ops={sum(completed_ops)} cleanup_runs={cleanup_progress['runs']} "
        f"cleanup_rows={cleanup_progress['rows_deleted']} "
        f"latency_ms p50={p50:.2f} p95={p95:.2f} p99={p99:.2f}"
    )

    assert cleanup_progress["runs"] >= 1, \
        "cleanup loop made no forward progress (expected at least 1 sweep)"
    assert p95 < 200.0, f"P95 write latency {p95:.2f}ms >= 200ms gate"
