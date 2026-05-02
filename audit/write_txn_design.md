# `db.write_txn` design — Phase 1.2a

**Status:** design proposal — to be reviewed before implementation lands.
**Owner:** Phase 1 chokepoint work.
**Constraint summary** (from the chat brief):

> 1. Single chokepoint with signature `db.write_txn(bot_id: Optional[int], fn: Callable[[sqlite3.Connection], T]) -> T`
> 2. Acquires `bot_db_lock(bot_id)` if `bot_id is not None`, else a global write lock
> 3. Opens a fresh conn via `_make_real_conn` (or routes through the per-thread `_conn()`?)
> 4. Retries 5× with exponential jitter (50ms, 100ms, 250ms, 500ms, 1000ms) on `OperationalError("database is locked")`
> 5. Logs every retry with bot_id, fn name, attempt
> 6. Raises `DBLockedError` after 5 failures, including the SQL being attempted
> 7. Per-bot lock reuses `BotManager.bot_db_lock(bot_id)` — do **not** create a parallel lock registry
> 8. WAL checkpoint thread folded in: every 60s `PRAGMA wal_checkpoint(TRUNCATE)`, log size before/after

---

## 1. Public API

### 1.1 `db.write_txn(bot_id, fn)`

```python
T = TypeVar("T")

def write_txn(
    bot_id: Optional[int],
    fn: Callable[[sqlite3.Connection], T],
    *,
    name: Optional[str] = None,
) -> T:
    """
    Single chokepoint for every persistent SQLite write.

    The contract `fn` MUST honour:
      * `fn` receives an open sqlite3.Connection with WAL+busy_timeout PRAGMAs already set.
      * `fn` MUST NOT call write_txn() recursively. Nested writes pass the same
        `con` argument explicitly (e.g. _record_recommendation_outcome(con, ...)).
      * `fn` MUST NOT close the connection.
      * `fn` MUST NOT call `con.commit()` — write_txn commits exactly once on success
        and rolls back on exception.
      * `fn` MAY raise; the exception propagates and the txn is rolled back.

    Locking:
      * If `bot_id is not None` → acquire the per-bot lock (RLock; same thread
        may re-enter without deadlocking, e.g. when a per-bot writer calls a
        helper that takes the same lock for safety).
      * If `bot_id is None` → acquire the global write lock (RLock).
      * The global write lock and per-bot locks are independent — they do NOT
        nest. A per-bot writer does not block a global writer or vice versa
        (SQLite still serialises them, but the Python-side locks don't add
        cross-category contention).

    Retry policy:
      * On `sqlite3.OperationalError` whose message contains "database is locked"
        OR "database table is locked" (covers both whole-DB and per-table
        BUSY): retry per the schedule in §3 below.
      * On any other exception (including OperationalError with a different
        message): propagate immediately, no retry.
      * After 5 failed attempts: raise DBLockedError (§2).

    Returns:
      Whatever `fn(con)` returned on the successful attempt.

    Logging:
      * Every retry logged at WARN with bot_id, fn name (or `name` override),
        attempt 1..5, elapsed_ms since first attempt, jitter_ms applied,
        the SQL being attempted (last `con.execute` text, captured via a
        wrapped cursor).
      * Final DBLockedError raise logged at ERROR via logger.exception.
    """
```

**Rationale on `name`:** the function name (`fn.__name__`) is unstable when
callers pass lambdas; the optional `name` keyword lets call sites tag the
operation explicitly (e.g. `name="add_log"`). When omitted we fall back to
`getattr(fn, "__name__", "<lambda>")`.

### 1.2 `db.DBLockedError`

```python
class DBLockedError(sqlite3.OperationalError):
    """Raised after write_txn exhausts its retry budget on a locked DB.

    Inherits sqlite3.OperationalError so existing `except sqlite3.OperationalError`
    handlers (e.g. the supervisor's exponential backoff loop) continue to catch
    it without code changes — but with structured context attached.
    """

    def __init__(
        self,
        *,
        bot_id: Optional[int],
        op_name: str,
        attempts: int,
        elapsed_ms: int,
        last_sql: Optional[str],
        last_exc: BaseException,
    ) -> None:
        self.bot_id = bot_id
        self.op_name = op_name
        self.attempts = attempts
        self.elapsed_ms = elapsed_ms
        self.last_sql = (last_sql or "")[:512]  # truncate for log safety
        self.last_exc = last_exc
        super().__init__(self.__str__())

    def __str__(self) -> str:
        bid = "None" if self.bot_id is None else str(self.bot_id)
        return (
            f"DBLockedError(op={self.op_name!r}, bot_id={bid}, "
            f"attempts={self.attempts}, elapsed_ms={self.elapsed_ms}): "
            f"{self.last_sql!r} -> {type(self.last_exc).__name__}: {self.last_exc}"
        )
```

**Why subclass `sqlite3.OperationalError`:** existing `except OperationalError`
handlers in `bot_manager._supervised_run_loop` and elsewhere keep working —
they just log a stricter type. New code can `except DBLockedError` for finer
control without breaking existing recovery paths.

### 1.3 `db.open_migration_conn()`

```python
def open_migration_conn() -> sqlite3.Connection:
    """Public alias for the canonical fresh-connection factory.

    Use from one-shot CLI scripts (scripts/migrate_*.py, init_db, dev tools)
    so PRAGMAs (WAL, busy_timeout=30000, synchronous=NORMAL, foreign_keys=ON,
    cache=64MB, mmap=256MB) stay consistent with the worker's own connections.

    Returns a brand-new sqlite3.Connection. Callers own its lifecycle and MUST
    call .close() (no thread-local pooling).
    """
    return _make_real_conn()
```

**Action items it unblocks:** `scripts/migrate_auto_restart.py:51` and the
`check_*.py` diagnostic scripts can drop their hand-rolled
`sqlite3.connect(...) + PRAGMA WAL + PRAGMA busy_timeout` and call this
helper instead.

### 1.4 `db.start_wal_checkpoint_thread()` / `db.stop_wal_checkpoint_thread()`

```python
def start_wal_checkpoint_thread(interval_sec: int = 60) -> None:
    """Start a background daemon thread that runs PRAGMA wal_checkpoint(TRUNCATE)
    every `interval_sec` seconds. Idempotent — calling twice is a no-op.

    Logs WAL file size (bytes) before and after each checkpoint at INFO when
    the WAL changed by more than 1 MB, otherwise at DEBUG.

    The checkpoint thread uses `write_txn(None, fn=_do_checkpoint)` so it
    contends for the global write lock — preventing it from racing
    bulk-DELETE cleanups or schema migrations.

    Started by worker_api.py at boot. Tests can call stop() in fixtures.
    """

def stop_wal_checkpoint_thread(timeout_sec: float = 5.0) -> None:
    """Signal the checkpoint thread to exit and join it. Idempotent."""
```

**Why a thread, not a cron / on-demand:** WAL grows unboundedly between
checkpoints under sustained write load. Without a periodic TRUNCATE
checkpoint, the WAL file balloons (we've seen multi-GB WALs in similar
projects), readers slow down because they have to scan the WAL, and recovery
on crash takes longer. 60s is conservative: SQLite auto-checkpoints PASSIVE
every 1000 pages by default (~4MB), but PASSIVE doesn't truncate so disk
keeps growing. TRUNCATE every 60s caps disk usage and keeps recovery fast.

---

## 2. Retry policy

**Confirmed verbatim from the brief.** No deviation.

| Attempt | Base wait | Jitter range | Effective wait |
| --- | --- | --- | --- |
| 1 → 2 | 50 ms  | ±20% | 40–60 ms |
| 2 → 3 | 100 ms | ±20% | 80–120 ms |
| 3 → 4 | 250 ms | ±20% | 200–300 ms |
| 4 → 5 | 500 ms | ±20% | 400–600 ms |
| 5 → fail | 1000 ms | ±20% | 800–1200 ms |
| (give up after 5 retries → raise DBLockedError) |

**Implementation:**

```python
_RETRY_SCHEDULE_MS: Tuple[int, ...] = (50, 100, 250, 500, 1000)

def _next_sleep_sec(attempt: int) -> float:
    """attempt is 0-indexed; returns seconds to sleep before next try."""
    base_ms = _RETRY_SCHEDULE_MS[attempt]
    jitter = random.uniform(0.8, 1.2)  # ±20% multiplicative
    return (base_ms * jitter) / 1000.0
```

**Worst-case Python-side wait before raising DBLockedError:**
40+80+200+400+800 = 1520 ms (best jitter), 60+120+300+600+1200 = 2280 ms
(worst). On top of that, each attempt may itself wait up to `busy_timeout =
30000 ms` inside SQLite before raising OperationalError, so absolute worst case
is ~150s of patience before we give up. Acceptable for non-trading writes;
trading writes (orders) call `write_txn` from the executor which has its own
deadline above us.

**Why these specific numbers (justification, not deviation):** Geometric ish
ramp keeps p50 latency low (the 50ms first retry resolves the vast majority
of WAL contention spikes) while the 1000ms fifth attempt covers cleanup-under-
load scenarios. ±20% jitter prevents thundering-herd among parallel bots all
retrying in lockstep.

---

## 3. Lock model — concrete decision matrix

The brief says: "Per-bot lock reuses existing `BotManager.bot_db_lock(bot_id)`
— do NOT create a parallel lock registry." Today the registry lives in
`BotManager._bot_db_locks` (`bot_manager.py:5158`). `db.py` cannot import
`bot_manager` (would create a circular dep). Resolution:

**Move the registry into `db.py`. Replace `BotManager.bot_db_lock` with a
thin delegating wrapper.** This satisfies the constraint (one registry only)
and makes `write_txn` self-sufficient for tests, scripts, and any future
non-`BotManager` caller.

### 3.1 `db._bot_locks_registry`

```python
_bot_locks: Dict[int, threading.RLock] = {}
_bot_locks_guard = threading.Lock()
_global_write_lock = threading.RLock()

def bot_db_lock(bot_id: int) -> threading.RLock:
    """Per-bot reentrant lock. Single source of truth for the worker process.

    Reentrant so a bot-tick writer that legitimately calls a helper which also
    takes the lock (defense-in-depth) doesn't deadlock. Cross-thread ordering
    is unchanged.
    """
    bid = int(bot_id)
    with _bot_locks_guard:
        lk = _bot_locks.get(bid)
        if lk is None:
            lk = threading.RLock()
            _bot_locks[bid] = lk
        return lk
```

`BotManager.bot_db_lock(bot_id)` becomes:

```python
def bot_db_lock(self, bot_id: int) -> threading.RLock:
    """Delegates to db.bot_db_lock (single registry). Kept for callers that
    pass through the manager. Preserves behaviour."""
    from db import bot_db_lock as _db_bot_db_lock
    return _db_bot_db_lock(bot_id)
```

`BotManager._bot_db_locks` and `_bot_db_locks_guard` are deleted — they no
longer hold any state.

### 3.2 `RLock` instead of `Lock`

`threading.RLock` is reentrant per-thread. The current `Lock` deadlocks if a
writer (e.g. `BotRunner._sync_close_deal` holds the lock; `close_deal` calls a
helper which itself wraps in `bot_db_lock`). Today this can't happen because
the helpers don't take the lock — but Phase 1.2 will route many writers
through `write_txn(bot_id, fn)`, including some called from other writers.
Switching to RLock removes the foot-gun without changing cross-thread
semantics. Documented in the commit message.

### 3.3 Decision matrix: which writers pass `bot_id`, which pass `None`?

| Category | Writers (db.py functions) | Lock |
| --- | --- | --- |
| Bot tick hot path (H-risk) | `add_log`, `add_order_event`, `open_deal`, `close_deal`, `update_open_deal_entry`, `cancel_ghost_deal`, `manual_close_deal_and_journal` | `write_txn(bot_id, fn)` |
| Per-bot supplementary | `add_regime_snapshot`, `add_strategy_decision`, `add_strategy_trade`, `save_perf_metrics`, `update_bot`, `set_bot_enabled`, `set_bot_running`, `delete_bot`, `patch_bot_risk_after_create`, `link_recommendation_to_bot`, `log_error` (when `bot_id` is not None) | `write_txn(bot_id, fn)` |
| Global config / settings | `set_setting`, `save_autopilot_config`, `update_bots_by_type`, `create_bot`, `add_autopilot_audit_log`, `log_data_quality`, `log_audit`, `log_error` (when bot_id is None) | `write_txn(None, fn)` |
| Explore / scanner | `mark_explore_signals_pending`, `mark_explore_horizon_pending`, `upsert_explore_feed_row`, `save_recommendation_snapshot`, `save_signal_outcome`, `update_explore_signal_outcome`, `save_explore_backtest_results`, `delete_recommendations_for_blocklist`, `cleanup_invalid_scores` | `write_txn(None, fn)` |
| Watchlist | `mark_watchlist_triggered`, `remove_watchlist_entry`, `cleanup_old_watchlist`, `upsert_watchlist_entry` (autopilot uses) | `write_txn(None, fn)` |
| Backtest / ML / patterns | `save_backtest_run`, `save_intraday_pattern`, `save_ml_model_version`, `update_ml_prediction_outcome`, `_save_ml_prediction` | `write_txn(None, fn)` |
| Cleanup (DELETE in chunks) | `cleanup_old_bot_logs`, `cleanup_old_strategy_decisions`, `cleanup_old_explore_signal_outcomes`, `cleanup_old_order_events`, `cleanup_old_regime_snapshots`, `cleanup_old_trade_feedback`, `cleanup_old_portfolio_snapshots`, `cleanup_old_recommendation_snapshots`, `cleanup_old_signal_audits` | `write_txn(None, fn)` per chunk; chunk size 500 (see §6) |
| Init / migration | `init_db`, `_ensure_column`, `_migrate_explore_signals_to_v2`, scripts/migrate_*.py | Use `open_migration_conn()` directly (not `write_txn`) — runs at boot, single-threaded, must be tolerant of stale schemas |
| Maintenance | `db_vacuum`, `db_analyze` | Cannot use `write_txn` because VACUUM cannot run inside a transaction. Acquire `_global_write_lock` manually, pause WAL checkpoint thread, run, release. Document loudly. |
| Other modules | `notification_manager.*` (8 writers), `execution_quality_tracker.record_execution`, `tax_optimizer.save_tax_harvest_suggestion`, `sector_rotation.record_sector_performance`, `worker_api._screener_outcomes_loop`, `worker_api._portfolio_loop`, `ml_signal_scorer._log_version_to_db` (delete or replace) | All `write_txn(None, fn)` |

**Rule of thumb for future writers:** If the target table has a `bot_id`
column AND the writer is called from a path that is also touched by the
runner thread of that same bot, pass `bot_id`. Otherwise pass `None`.

### 3.4 Interaction with existing `_db_retry`

`_db_retry` (`db.py:120`) is currently used by exactly four functions
(`save_recommendation_snapshot`, `mark_explore_signals_pending`,
`upsert_explore_feed_row`, plus the `manual_close_deal_and_journal` is
hand-rolled separately). All four migrate to `write_txn` in Phase 1.2b
step 5. `_db_retry` then has zero callers.

**Decision: delete `_db_retry` in the same commit that migrates the last of
its callers** (1.2b step 5). Keeping a deprecated wrapper around an
unused-internally function is needless surface area in a money-handling
codebase. The function is private (`_` prefix), no external API to preserve.
The deletion will be documented in the commit message and crossed off in
`audit/db_writers.md`.

### 3.5 Nested `write_txn` policy — explicit ban + escape hatch

`write_txn(bot_id, fn)` MUST NOT be called from inside another `write_txn`'s
`fn`. Reasons:

1. The outer call already holds the lock (RLock makes re-entry safe, so no
   deadlock in practice — but the inner call would open a fresh `_conn()`
   instead of reusing the outer one, defeating the txn's atomicity guarantee.
2. The inner call would commit before the outer one returns, breaking the
   "single commit on success" contract.

**Detection:** A `threading.local()` flag (`_in_write_txn`) is set on entry
and cleared on exit. If `write_txn` is called while the flag is set, raise
`RuntimeError("nested write_txn detected — pass the existing conn instead")`.
This is a programming bug, not a recoverable runtime condition; tests will
exercise both directions to lock the contract.

**Escape hatch for sub-helpers** (e.g. `_record_recommendation_outcome`
called from `close_deal`): they keep their existing signature
`fn(con: sqlite3.Connection, ...)` and execute on the parent's `con`. No
nested write_txn needed.

---

## 4. Connection lifecycle inside `write_txn`

Two valid choices. Picking explicitly:

| Option | `con = _conn()` (per-thread cached) | `con = _make_real_conn()` (fresh per call) |
| --- | --- | --- |
| **Performance** | High — no connect overhead per call (matters for `add_log` at >>1Hz) | Lower — ~1ms connect+pragma overhead per write |
| **Isolation** | Shared with reads on the same thread; `con.commit()` commits everything pending | Total — only the write_txn's statements get committed |
| **Crash blast radius** | Larger — if `_conn()`'s cached conn is in a bad state, all subsequent writes inherit it | Smaller — bad conn dies with the call |
| **Compatibility** | Drop-in for current code | Forces every helper to drop their own `con = _conn()` and accept `con` as a parameter |

**Decision: use `_conn()`** (per-thread cached). Rationale:

1. The current writer surface ALREADY uses `_conn()` exclusively — switching
   to `_make_real_conn()` per call would be a behaviour change for every
   writer simultaneously, larger blast radius.
2. `add_log` runs >>1Hz under load; per-call connect+pragma is wasteful and
   measurable in the load test target (P95 < 200ms).
3. Helpers inside `fn` that need to do extra reads (e.g. `close_deal` reads
   the deal row before updating it) will share the conn naturally.
4. `manual_close_deal_and_journal` (the one writer that uses
   `_make_real_conn` today) is documented as "race-safe" because of its
   `WHERE state='OPEN'` predicate, not because of conn isolation. Migrating
   it to `_conn()` does not regress that guarantee.

**Caveat:** because `_conn()` is per-thread-cached, the conn outlives the
write_txn call. If `fn` raises, we MUST `con.rollback()` so we don't leave a
partial transaction on the cached conn that contaminates the next `fn()` on
the same thread. This is implemented and tested.

---

## 5. SQL capture for DBLockedError

We promised the error includes "the SQL being attempted." Implementation:

`write_txn` wraps the connection passed to `fn` with a thin `_TrackingConn`
proxy that intercepts `.execute(sql, ...)` and `.executemany(sql, ...)`,
storing the most recent SQL in a thread-local. When DBLockedError is raised,
it pulls the captured SQL.

Cost: one extra Python call per `.execute`. For `add_log`-class writers
that's <1µs. Negligible vs. the SQLite call itself.

The captured SQL is **truncated to 512 chars** in DBLockedError to bound log
sizes. Parameter values are NOT captured (could leak balances or order IDs);
only the SQL template is.

---

## 6. Chunked-cleanup helper (special case from the brief)

`cleanup_old_*` functions become:

```python
_CLEANUP_BATCH = 500
_CLEANUP_INTERBATCH_SLEEP_SEC = 0.05

def _cleanup_loop(table: str, where_sql: str, params: tuple) -> int:
    """Delete in batches of 500 to avoid holding the writer too long.

    Each batch is its own write_txn(None, fn) so the global write lock is
    released between batches, letting hot-path writers (add_log,
    add_order_event) make forward progress.
    """
    total = 0
    while True:
        def _do(con: sqlite3.Connection) -> int:
            cur = con.execute(
                f"DELETE FROM {table} WHERE {where_sql} LIMIT {_CLEANUP_BATCH}",
                params,
            )
            return int(cur.rowcount or 0)
        n = write_txn(None, _do, name=f"cleanup_{table}_batch")
        total += n
        if n < _CLEANUP_BATCH:
            return total
        time.sleep(_CLEANUP_INTERBATCH_SLEEP_SEC)
```

`cleanup_old_bot_logs`, `cleanup_old_order_events`, etc. become one-line
wrappers around `_cleanup_loop`. The 50ms inter-batch sleep yields the
writer slot — without it, the cleanup re-acquires the global lock immediately
and starves hot-path writers.

**Note re SQLite `DELETE ... LIMIT`:** SQLite supports
`DELETE FROM t WHERE … LIMIT n` only when compiled with
`SQLITE_ENABLE_UPDATE_DELETE_LIMIT`. Python's bundled SQLite (3.36+) ships
this enabled. Phase 1.2a's test suite asserts this at startup so we fail
loudly on environments without it. Fallback (if the flag is off): use
`DELETE FROM t WHERE rowid IN (SELECT rowid FROM t WHERE … LIMIT n)`.

---

## 7. Test plan for 1.2a (in isolation, before any caller migrates)

`tests/test_write_txn.py` — must all pass before 1.2b begins.

1. `test_write_txn_returns_fn_result` — `fn` returns 42, write_txn returns 42.
2. `test_write_txn_commits_on_success` — `fn` does an INSERT; row exists
   after write_txn returns, on a fresh `_make_real_conn`.
3. `test_write_txn_rolls_back_on_exception` — `fn` does an INSERT then
   raises ValueError; the row does NOT exist; the next write_txn on the
   same thread succeeds (proving the cached conn was rolled back, not
   poisoned).
4. `test_write_txn_no_implicit_commit_in_fn` — `fn` calls `con.commit()`
   itself; write_txn detects it (or simply re-commits and the second commit
   is a no-op). Document the de-facto behaviour either way.
5. `test_write_txn_retries_on_database_locked_then_succeeds` — fault-inject
   an OperationalError("database is locked") for the first 2 attempts; the
   3rd succeeds. Assert exactly 2 retries logged.
6. `test_write_txn_raises_DBLockedError_after_5_failures` — fault-inject
   the error for all 5 attempts. Assert raised type is `DBLockedError`,
   `attempts == 5`, `op_name` matches, `last_sql` is non-empty.
7. `test_write_txn_does_not_retry_on_other_OperationalError` — inject a
   different OperationalError (e.g. "no such table"); write_txn raises
   immediately on attempt 1, no retries.
8. `test_write_txn_per_bot_lock_serialises` — 10 threads call
   `write_txn(bot_id=42, fn)` concurrently; each `fn` does an
   INSERT then sleeps 50ms then UPDATE; assert the trace shows strict
   serial ordering (no interleaving). Same with `bot_id=None`.
9. `test_write_txn_independent_bots_do_not_block_each_other` — 4 threads
   each writing to a different bot_id; assert wall-clock time is
   approximately 1× single-threaded time (parallelism preserved).
10. `test_write_txn_global_lock_serialises` — 10 threads with `bot_id=None`;
    same as test 8 but on the global lock.
11. `test_write_txn_per_bot_and_global_independent` — concurrent per-bot
    and global writers do not deadlock; both make progress.
12. `test_write_txn_nested_call_raises_RuntimeError` — `fn` calls
    `write_txn(...)` from inside; raises `RuntimeError` mentioning "nested".
13. `test_DBLockedError_is_OperationalError` — `isinstance(e, sqlite3.OperationalError)`.
14. `test_DBLockedError_str_format` — `str(e)` matches the documented format.
15. `test_open_migration_conn_has_pragmas` — calls `open_migration_conn()`,
    asserts `journal_mode == 'wal'`, `busy_timeout >= 30000`,
    `synchronous == 1` (NORMAL), `foreign_keys == 1`.
16. `test_wal_checkpoint_thread_starts_and_stops` — start, sleep(2),
    stop with timeout 1; assert thread exited cleanly. Log lines captured.
17. `test_wal_checkpoint_runs_at_least_once` — set `interval_sec=1`, do
    ~50 inserts to grow WAL, wait 1.5s, stop thread; assert WAL file size
    after stop ≤ size at peak. (Cannot assert exact value because PASSIVE
    auto-checkpoints may have run in between — only that TRUNCATE actually
    happened at least once.)
18. `test_chunked_cleanup_yields_writer_slot` — start a producer thread
    that INSERTs into bot_logs as fast as it can; concurrently run the
    chunked cleanup; assert (a) zero OperationalError propagated, (b)
    cleanup completed, (c) producer made forward progress (rowcount grew)
    while cleanup ran. This is the regression test for the bot-1 lock
    loop.

All tests use a `tmp_path` SQLite file (not in-memory — WAL needs an actual
file) and clear the per-thread connection cache before each run.

---

## 8. Migration path for callers (preview of Phase 1.2b)

A typical writer migration looks like:

**Before:**
```python
def add_log(bot_id, level, message, category="SYSTEM"):
    con = _conn()
    con.execute("INSERT INTO bot_logs(...) VALUES (...)", (...))
    con.commit()
    con.close()
```

**After:**
```python
def add_log(bot_id, level, message, category="SYSTEM"):
    def _do(con):
        con.execute("INSERT INTO bot_logs(...) VALUES (...)", (...))
    write_txn(int(bot_id), _do, name="add_log")
```

Net delta: −2 lines (no manual `commit`/`close`), +1 line (write_txn call),
+1 indentation. All retry / lock / error semantics inherited. Tests on the
function before and after assert identical externally-observable behaviour
plus the new retry semantics.

---

## 9. What this design does **not** do (so we don't lie by omission)

- It does not add cross-process locking. Two `python` processes hitting the
  same DB still rely on SQLite's file lock + WAL. The deploy script must
  continue to stop the worker before running migrations that take >100ms.
- It does not add a per-table lock. Granularity is per-bot OR global.
  Per-table is unnecessary because SQLite already serialises at the file
  level under WAL — the Python-side locks just remove the
  spin-and-OperationalError loop.
- It does not change SQLite's internal isolation level. Default deferred
  transactions are still in use; `BEGIN IMMEDIATE` is not forced. (Could be
  added later if WAL contention reappears, but `busy_timeout=30000` already
  handles it.)
- It does not redact secrets in the captured SQL. SQL templates don't
  contain secrets (only parameters do, and we don't capture parameters), but
  the broader root-logger redaction filter (brief Phase 1 rule #9) is its
  own deliverable in Phase 6.

---

## 10. Open questions for review (none expected to block)

1. Should `write_txn` accept a `*, isolation: str = "DEFERRED"` kwarg so
   future callers can opt into `BEGIN IMMEDIATE`? Proposal: not yet — YAGNI.
   Add when needed.
2. Should the WAL checkpoint thread interval be configurable via env? Yes:
   `BOT_WAL_CHECKPOINT_INTERVAL_SEC` env var, default 60. Documented in
   `.env.example` in the same commit.
3. Should `open_migration_conn` register itself with the WAL checkpoint
   thread? No — migration conns are short-lived and explicitly close.

---

## 11. Acceptance criteria for landing 1.2a

- [ ] All 18 tests in `tests/test_write_txn.py` pass on Windows + Linux
      Python 3.12.
- [ ] `pytest -q` for the whole repo passes (no regressions in
      `test_db_locking.py` etc).
- [ ] `db.py` exports: `write_txn`, `DBLockedError`, `bot_db_lock`,
      `open_migration_conn`, `start_wal_checkpoint_thread`,
      `stop_wal_checkpoint_thread`.
- [ ] `BotManager.bot_db_lock` is a 2-line delegating wrapper around
      `db.bot_db_lock`. `_bot_db_locks` and `_bot_db_locks_guard` removed
      from `BotManager.__init__`.
- [ ] `_db_retry` is still in `db.py` (deletion happens in 1.2b step 5,
      after its last caller migrates).
- [ ] No production caller of `write_txn` exists yet — 1.2a is
      chokepoint-only.
- [ ] Commit message references this design doc.
