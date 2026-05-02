# db.py  (REPLACE ENTIRE FILE)
import json
import logging
import math
import os
import random
import sqlite3
import threading
import time
from typing import Any, Callable, Dict, List, Optional, Tuple, TypeVar

logger = logging.getLogger(__name__)

# Generic return type for write_txn.
T = TypeVar("T")

# Allow override via env; default keeps compatibility with your existing file
DB_NAME = os.getenv("BOT_DB_PATH", "botdb.sqlite3")

# Per-thread connection cache: each thread (scan_short, scan_medium, scan_long,
# db_cleanup, ml_retrain, etc.) gets its own SQLite connection.  WAL mode allows
# concurrent readers; the per-thread approach eliminates most write contention
# because each thread can pipeline its own reads/writes without blocking others.
_tl = threading.local()

# Whitelist for dynamic SQL (prevents SQL injection)
_ALLOWED_TABLES = frozenset({
    "bots", "bot_logs", "deals", "settings", "strategy_decisions", "regime_snapshots",
    "autopilot_config",
    "order_events", "perf_metrics", "backtest_runs", "recommendations_snapshots",
    "recommendations_latest", "strategy_perf_trades", "intelligence_decisions",
    "recommendation_performance", "scoring_calibration_log", "intraday_patterns",
    "sector_performance_history", "dividend_events", "tax_harvest_suggestions", "market_events",
    "insider_transactions", "ml_predictions", "ml_model_versions", "execution_quality",
    "data_quality_log", "error_log", "trade_journal", "portfolio_snapshots",
    "autopilot_audit_log",  # LIVE-HARDENED: autopilot decision traceability
    "scanner_watchlist",  # Purposed-entry watchlist: symbols awaiting trigger
    "notifications",  # User-facing notifications: trades, alerts, summaries
    "audit_log",  # Security and compliance tracking
    "trade_feedback",  # ML learning from closed trades
    "signal_audit",  # Hybrid screener signal audit trail
    "explore_signals",  # Per (symbol, horizon) Explore feed: pending/buy/watch/rejected (UPSERT only)
    "explore_backtest_results",  # Cached Explore strategy backtest aggregates
    "explore_signal_outcomes",  # Tracked buy-signal forward outcomes for strategy win rates
    "signal_accuracy_baseline",  # Pre-computed strategy win-rate baselines for Explore badges
})
_ALLOWED_COLUMNS = frozenset({"bot_id", "id"})


class _NoCloseConn:
    """
    Wraps a sqlite3.Connection so that .close() is a no-op.
    The underlying connection stays open for the lifetime of its thread.
    All other attributes/methods are passed through transparently.
    """
    __slots__ = ("_real",)

    def __init__(self, real: sqlite3.Connection) -> None:
        object.__setattr__(self, "_real", real)

    def close(self) -> None:
        pass  # intentional no-op — thread-local connection is reused

    def __getattr__(self, name: str):
        return getattr(object.__getattribute__(self, "_real"), name)

    def __setattr__(self, name: str, value) -> None:
        if name == "_real":
            object.__setattr__(self, name, value)
        else:
            setattr(object.__getattribute__(self, "_real"), name, value)

    def __enter__(self):
        return object.__getattribute__(self, "_real").__enter__()

    def __exit__(self, *args):
        return object.__getattribute__(self, "_real").__exit__(*args)


def _make_real_conn() -> sqlite3.Connection:
    """Create a brand-new raw SQLite connection with all performance pragmas set."""
    con = sqlite3.connect(DB_NAME, check_same_thread=True, timeout=30.0)
    con.row_factory = sqlite3.Row
    for pragma in (
        "PRAGMA journal_mode=WAL;",        # WAL: concurrent readers, serialised writers
        "PRAGMA synchronous=NORMAL;",      # Safe & fast (vs. FULL)
        "PRAGMA busy_timeout=30000;",      # Wait up to 30 s before OperationalError
        "PRAGMA foreign_keys=ON;",
        "PRAGMA cache_size=-65536;",       # 64 MB page cache per connection
        "PRAGMA temp_store=MEMORY;",
        "PRAGMA mmap_size=268435456;",     # 256 MB memory-mapped I/O
    ):
        try:
            con.execute(pragma)
        except Exception:
            pass
    try:
        con.execute("PRAGMA wal_checkpoint(PASSIVE);")
    except Exception:
        pass
    return con


def _conn() -> _NoCloseConn:
    """
    Return the per-thread cached SQLite connection (creating it on first call).
    Each OS thread (scan_short, scan_medium, scan_long, ml_retrain, db_cleanup, …)
    gets its own connection, so they never block each other.
    WAL mode allows unlimited concurrent readers; writers queue inside SQLite.
    The returned object silently ignores .close() so callers need not be changed.
    """
    real = getattr(_tl, "conn", None)
    if real is not None:
        try:
            real.execute("SELECT 1")        # quick liveness check
            return _NoCloseConn(real)
        except Exception:
            pass                            # connection was closed externally — recreate
    real = _make_real_conn()
    _tl.conn = real
    return _NoCloseConn(real)


def _db_retry(fn, *args, _retries: int = 5, **kwargs):
    """
    Call fn(*args, **kwargs) with exponential-backoff retry on
    OperationalError('database is locked').  All other exceptions propagate immediately.

    DEPRECATED: new code MUST use ``write_txn``. This helper exists for the
    transitional window in Phase 1.2b while the four legacy callers
    (save_recommendation_snapshot, mark_explore_signals_pending,
    upsert_explore_feed_row, mark_explore_horizon_pending) are migrated. It is
    deleted in the final commit of Phase 1.2b step 5.
    """
    delay = 0.1
    for attempt in range(_retries + 1):
        try:
            return fn(*args, **kwargs)
        except sqlite3.OperationalError as exc:
            if "database is locked" in str(exc).lower() and attempt < _retries:
                logger.debug("DB locked — retry %d/%d in %.1fs", attempt + 1, _retries, delay)
                time.sleep(delay)
                delay = min(delay * 2, 2.0)
            else:
                raise


# ============================================================================
# Phase 1.2a — write_txn chokepoint
# See audit/write_txn_design.md for the full contract and rationale.
# ============================================================================

# Module-level lock state. Single source of truth for the worker process.
# BotManager.bot_db_lock delegates to db.bot_db_lock (no parallel registry).
_bot_locks: Dict[int, "threading.RLock"] = {}
_bot_locks_guard: "threading.Lock" = threading.Lock()
_global_write_lock: "threading.RLock" = threading.RLock()

# Per-thread re-entry guard. Nested write_txn calls are a programming bug
# (they would open a fresh per-thread conn under the same lock and commit
# before the outer call returns, breaking atomicity). They MUST raise.
_write_txn_state: "threading.local" = threading.local()


def bot_db_lock(bot_id: int) -> "threading.RLock":
    """Return the canonical per-bot reentrant lock.

    Reentrant (RLock) so a writer can call into a defense-in-depth helper that
    re-acquires the lock without deadlocking. Cross-thread serialisation is
    unchanged.

    BotManager.bot_db_lock delegates here; do not maintain a parallel registry.
    """
    bid = int(bot_id)
    with _bot_locks_guard:
        lk = _bot_locks.get(bid)
        if lk is None:
            lk = threading.RLock()
            _bot_locks[bid] = lk
        return lk


class DBLockedError(sqlite3.OperationalError):
    """Raised when ``write_txn`` exhausts its retry budget on a locked DB.

    Inherits ``sqlite3.OperationalError`` so existing handlers (e.g.
    BotRunner._supervised_run_loop) continue to catch it without code changes.
    Carries structured context for diagnostics: bot_id, op_name, attempts,
    elapsed_ms, last_sql, last_exc.
    """

    def __init__(
        self,
        *,
        bot_id: Optional[int],
        op_name: str,
        attempts: int,
        elapsed_ms: int,
        last_sql: Optional[str],
        last_exc: Optional[BaseException],
    ) -> None:
        self.bot_id = bot_id
        self.op_name = op_name
        self.attempts = attempts
        self.elapsed_ms = elapsed_ms
        self.last_sql = (last_sql or "")[:512]  # truncate to bound log size
        self.last_exc = last_exc
        super().__init__(self.__str__())

    def __str__(self) -> str:
        bid = "None" if self.bot_id is None else str(self.bot_id)
        last_type = type(self.last_exc).__name__ if self.last_exc else "?"
        return (
            f"DBLockedError(op={self.op_name!r}, bot_id={bid}, "
            f"attempts={self.attempts}, elapsed_ms={self.elapsed_ms}): "
            f"{self.last_sql!r} -> {last_type}: {self.last_exc}"
        )


# Sleep schedule for the 5 retries (in milliseconds). The Nth entry is the
# sleep BEFORE the (N+1)th retry, i.e. between attempt (N+1) and attempt (N+2)
# in 1-indexed terms. Total max attempts: 6 (initial + 5 retries).
_RETRY_SCHEDULE_MS: Tuple[int, ...] = (50, 100, 250, 500, 1000)


def _next_sleep_sec(retry_idx: int) -> float:
    """Sleep duration before retry #(retry_idx+1). Applies +/-20% jitter."""
    base_ms = _RETRY_SCHEDULE_MS[retry_idx]
    return (base_ms * random.uniform(0.8, 1.2)) / 1000.0


def _is_database_locked(exc: BaseException) -> bool:
    """True iff exc is an OperationalError indicating SQLite-level BUSY/LOCKED."""
    if not isinstance(exc, sqlite3.OperationalError):
        return False
    msg = str(exc).lower()
    return "database is locked" in msg or "database table is locked" in msg


class _TrackingCursor:
    """Cursor proxy that records the most recent SQL on the parent state dict."""

    __slots__ = ("_real", "_state")

    def __init__(self, real: sqlite3.Cursor, state: Dict[str, Any]) -> None:
        object.__setattr__(self, "_real", real)
        object.__setattr__(self, "_state", state)

    def execute(self, sql, *args, **kwargs):
        object.__getattribute__(self, "_state")["last_sql"] = sql
        return object.__getattribute__(self, "_real").execute(sql, *args, **kwargs)

    def executemany(self, sql, *args, **kwargs):
        object.__getattribute__(self, "_state")["last_sql"] = sql
        return object.__getattribute__(self, "_real").executemany(sql, *args, **kwargs)

    def executescript(self, sql_script, *args, **kwargs):
        object.__getattribute__(self, "_state")["last_sql"] = (sql_script or "")[:200]
        return object.__getattribute__(self, "_real").executescript(sql_script, *args, **kwargs)

    def __iter__(self):
        return iter(object.__getattribute__(self, "_real"))

    def __getattr__(self, name):
        return getattr(object.__getattribute__(self, "_real"), name)

    def __setattr__(self, name, value):
        if name in ("_real", "_state"):
            object.__setattr__(self, name, value)
        else:
            setattr(object.__getattribute__(self, "_real"), name, value)


class _TrackingConn:
    """Connection proxy that records the most recent SQL for DBLockedError context."""

    __slots__ = ("_real", "_state")

    def __init__(self, real, state: Dict[str, Any]) -> None:
        object.__setattr__(self, "_real", real)
        object.__setattr__(self, "_state", state)

    def execute(self, sql, *args, **kwargs):
        object.__getattribute__(self, "_state")["last_sql"] = sql
        return object.__getattribute__(self, "_real").execute(sql, *args, **kwargs)

    def executemany(self, sql, *args, **kwargs):
        object.__getattribute__(self, "_state")["last_sql"] = sql
        return object.__getattribute__(self, "_real").executemany(sql, *args, **kwargs)

    def executescript(self, sql_script, *args, **kwargs):
        object.__getattribute__(self, "_state")["last_sql"] = (sql_script or "")[:200]
        return object.__getattribute__(self, "_real").executescript(sql_script, *args, **kwargs)

    def cursor(self, *args, **kwargs):
        return _TrackingCursor(
            object.__getattribute__(self, "_real").cursor(*args, **kwargs),
            object.__getattribute__(self, "_state"),
        )

    def __enter__(self):
        return object.__getattribute__(self, "_real").__enter__()

    def __exit__(self, *a):
        return object.__getattribute__(self, "_real").__exit__(*a)

    def __getattr__(self, name):
        return getattr(object.__getattribute__(self, "_real"), name)

    def __setattr__(self, name, value):
        if name in ("_real", "_state"):
            object.__setattr__(self, name, value)
        else:
            setattr(object.__getattribute__(self, "_real"), name, value)


def write_txn(
    bot_id: Optional[int],
    fn: Callable[[Any], T],
    *,
    name: Optional[str] = None,
) -> T:
    """Single chokepoint for every persistent SQLite write.

    Contract (see audit/write_txn_design.md §1.1):
      * ``fn`` receives a connection-like object with WAL+busy_timeout PRAGMAs.
      * ``fn`` MUST NOT call ``write_txn`` recursively (raises RuntimeError).
      * ``fn`` MUST NOT close the conn or call ``commit()`` itself; this function
        commits exactly once on success and rolls back on exception.
      * Per-bot lock when ``bot_id`` is set; global write lock otherwise.
      * Up to 6 attempts (initial + 5 retries) on "database is locked"; sleeps
        between retries follow ``_RETRY_SCHEDULE_MS`` with +/-20% jitter.
      * On final failure raises ``DBLockedError`` (subclass of OperationalError)
        carrying bot_id, op_name, attempt count, elapsed_ms, last SQL, last exc.
    """
    if getattr(_write_txn_state, "active", False):
        raise RuntimeError(
            "nested write_txn detected — pass the existing conn to the inner helper instead"
        )

    op_name = name or getattr(fn, "__name__", "<lambda>")
    bid: Optional[int] = None if bot_id is None else int(bot_id)
    lock = bot_db_lock(bid) if bid is not None else _global_write_lock

    state: Dict[str, Any] = {"last_sql": None}
    started_at = time.monotonic()
    last_exc: Optional[BaseException] = None
    attempts_made = 0

    _write_txn_state.active = True
    try:
        with lock:
            for attempt in range(len(_RETRY_SCHEDULE_MS) + 1):  # 0..5 → up to 6 tries
                if attempt > 0:
                    sleep_sec = _next_sleep_sec(attempt - 1)
                    elapsed_ms = int((time.monotonic() - started_at) * 1000)
                    logger.warning(
                        "write_txn retry %d/%d op=%s bot_id=%s sleep=%.3fs elapsed_ms=%d",
                        attempt,
                        len(_RETRY_SCHEDULE_MS),
                        op_name,
                        "None" if bid is None else bid,
                        sleep_sec,
                        elapsed_ms,
                    )
                    time.sleep(sleep_sec)

                attempts_made = attempt + 1
                noclose = _conn()  # per-thread cached _NoCloseConn over a real conn
                tracking = _TrackingConn(noclose, state)
                try:
                    ret = fn(tracking)
                except sqlite3.OperationalError as exc:
                    last_exc = exc
                    try:
                        noclose.rollback()
                    except Exception:
                        logger.exception(
                            "write_txn: rollback after OperationalError failed (op=%s bot_id=%s)",
                            op_name, bid,
                        )
                    if _is_database_locked(exc):
                        continue  # retry
                    raise
                except BaseException:
                    try:
                        noclose.rollback()
                    except Exception:
                        logger.exception(
                            "write_txn: rollback after non-Operational exception failed (op=%s bot_id=%s)",
                            op_name, bid,
                        )
                    raise

                # Success path: commit, return.
                try:
                    noclose.commit()
                except sqlite3.OperationalError as exc:
                    # COMMIT itself can fail with "database is locked" under heavy
                    # WAL contention. Treat the same as fn failure: rollback & retry.
                    last_exc = exc
                    try:
                        noclose.rollback()
                    except Exception:
                        logger.exception(
                            "write_txn: rollback after commit-failure failed (op=%s bot_id=%s)",
                            op_name, bid,
                        )
                    if _is_database_locked(exc):
                        continue
                    raise
                return ret

            # Exhausted all attempts.
            elapsed_ms = int((time.monotonic() - started_at) * 1000)
            err = DBLockedError(
                bot_id=bid,
                op_name=op_name,
                attempts=attempts_made,
                elapsed_ms=elapsed_ms,
                last_sql=state.get("last_sql"),
                last_exc=last_exc,
            )
            logger.error(
                "write_txn exhausted retries: %s",
                err,
            )
            raise err from last_exc
    finally:
        _write_txn_state.active = False


def open_migration_conn() -> sqlite3.Connection:
    """Public alias for the canonical fresh-connection factory.

    Use from one-shot CLI scripts (scripts/migrate_*.py, dev tools) so PRAGMAs
    (WAL, busy_timeout=30000, synchronous=NORMAL, foreign_keys=ON, cache=64MB,
    mmap=256MB) stay consistent with worker-side connections.

    Returns a brand-new ``sqlite3.Connection`` that the caller owns and MUST
    close. Does NOT participate in the per-thread pool used by ``_conn()``.
    """
    return _make_real_conn()


# ----------------------------------------------------------------------------
# WAL checkpoint background thread (folded down from Phase 1.4)
# ----------------------------------------------------------------------------

_wal_checkpoint_thread: Optional["threading.Thread"] = None
_wal_checkpoint_stop_event: "threading.Event" = threading.Event()
_wal_checkpoint_lifecycle_lock: "threading.Lock" = threading.Lock()


def _wal_size_bytes() -> int:
    """Best-effort current size of the WAL sidecar file. 0 if absent."""
    try:
        wal_path = DB_NAME + "-wal"
        return os.path.getsize(wal_path) if os.path.exists(wal_path) else 0
    except OSError:
        return 0


def _wal_checkpoint_loop(interval_sec: int) -> None:
    """Daemon-thread body: TRUNCATE the WAL every interval_sec seconds.

    Uses ``write_txn(None, ...)`` so the checkpoint contends for the global
    write lock and cannot race bulk-DELETE cleanups or schema migrations.
    """
    # Initial checkpoint runs immediately on thread start, then we sleep.
    while True:
        try:
            size_before = _wal_size_bytes()

            def _do_checkpoint(con):
                cur = con.execute("PRAGMA wal_checkpoint(TRUNCATE)")
                row = cur.fetchone()
                return tuple(row) if row else None

            try:
                result = write_txn(None, _do_checkpoint, name="wal_checkpoint")
            except DBLockedError as e:
                logger.warning("WAL checkpoint deferred (DB busy): %s", e)
                result = None

            size_after = _wal_size_bytes()
            delta = size_before - size_after
            level = logging.INFO if abs(delta) >= (1 << 20) else logging.DEBUG
            logger.log(
                level,
                "WAL checkpoint: before=%d after=%d delta=%d result=%s",
                size_before,
                size_after,
                delta,
                result,
            )
        except Exception:
            logger.exception("WAL checkpoint loop iteration failed")

        if _wal_checkpoint_stop_event.wait(timeout=interval_sec):
            return


def start_wal_checkpoint_thread(interval_sec: Optional[int] = None) -> None:
    """Start the periodic WAL checkpoint daemon. Idempotent.

    Default interval is 60s, overridable via ``BOT_WAL_CHECKPOINT_INTERVAL_SEC``
    env var. Worker startup (worker_api.py) calls this at boot. Tests should
    pair this with ``stop_wal_checkpoint_thread`` in their fixtures.
    """
    global _wal_checkpoint_thread
    if interval_sec is None:
        try:
            interval_sec = int(os.getenv("BOT_WAL_CHECKPOINT_INTERVAL_SEC", "60"))
        except ValueError:
            interval_sec = 60
    interval_sec = max(1, int(interval_sec))

    with _wal_checkpoint_lifecycle_lock:
        existing = _wal_checkpoint_thread
        if existing is not None and existing.is_alive():
            return
        _wal_checkpoint_stop_event.clear()
        t = threading.Thread(
            target=_wal_checkpoint_loop,
            args=(interval_sec,),
            name="db-wal-checkpoint",
            daemon=True,
        )
        _wal_checkpoint_thread = t
        t.start()
    logger.info("WAL checkpoint thread started (interval=%ds)", interval_sec)


def stop_wal_checkpoint_thread(timeout_sec: float = 5.0) -> None:
    """Signal the checkpoint thread to exit and join. Idempotent."""
    global _wal_checkpoint_thread
    with _wal_checkpoint_lifecycle_lock:
        t = _wal_checkpoint_thread
        if t is None or not t.is_alive():
            _wal_checkpoint_thread = None
            return
        _wal_checkpoint_stop_event.set()
    t.join(timeout=timeout_sec)
    with _wal_checkpoint_lifecycle_lock:
        _wal_checkpoint_thread = None
    if t.is_alive():
        logger.warning(
            "WAL checkpoint thread did not exit within %.1fs",
            timeout_sec,
        )
    else:
        logger.info("WAL checkpoint thread stopped")


# ============================================================================
# End Phase 1.2a chokepoint
# ============================================================================


def now_ts() -> int:
    return int(time.time())


def _table_columns(con: sqlite3.Connection, table: str) -> List[str]:
    if table not in _ALLOWED_TABLES:
        raise ValueError(f"Invalid table name: {table}")
    rows = con.execute(f"PRAGMA table_info({table})").fetchall()
    return [r["name"] for r in rows]


def _ensure_column(con: sqlite3.Connection, table: str, col: str, col_def_sql: str) -> None:
    if table not in _ALLOWED_TABLES:
        raise ValueError(f"Invalid table name: {table}")
    cols = _table_columns(con, table)
    if col not in cols:
        try:
            con.execute(f"ALTER TABLE {table} ADD COLUMN {col} {col_def_sql}")
            con.commit()
        except sqlite3.OperationalError as e:
            # Column may have been added by concurrent init_db call — check again
            if "duplicate column" not in str(e).lower():
                logger.error("_ensure_column failed for %s.%s: %s", table, col, e)
                raise


def _migrate_explore_signals_to_v2(cur: sqlite3.Cursor) -> None:
    """Rebuild explore_signals when pre-v2 schema (snapshot_id / rejection_reason) is present."""
    cur.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='explore_signals'")
    if not cur.fetchone():
        cur.execute(
            """
            CREATE TABLE explore_signals (
                symbol TEXT NOT NULL,
                horizon TEXT NOT NULL,
                status TEXT NOT NULL,
                conviction_score REAL NOT NULL DEFAULT 0,
                reason TEXT,
                strategy TEXT,
                signal_ts INTEGER NOT NULL DEFAULT 0,
                updated_ts INTEGER NOT NULL,
                market_type TEXT,
                price REAL,
                change_24h REAL,
                detail_json TEXT,
                PRIMARY KEY (symbol, horizon)
            );
            """
        )
        cur.execute(
            "CREATE INDEX IF NOT EXISTS idx_explore_signals_horizon_status ON explore_signals(horizon, status);"
        )
        return
    cur.execute("PRAGMA table_info(explore_signals)")
    cols = {r[1] for r in cur.fetchall()}
    if "conviction_score" in cols:
        return
    cur.execute("DROP TABLE explore_signals")
    cur.execute(
        """
        CREATE TABLE explore_signals (
            symbol TEXT NOT NULL,
            horizon TEXT NOT NULL,
            status TEXT NOT NULL,
            conviction_score REAL NOT NULL DEFAULT 0,
            reason TEXT,
            strategy TEXT,
            signal_ts INTEGER NOT NULL DEFAULT 0,
            updated_ts INTEGER NOT NULL,
            market_type TEXT,
            price REAL,
            change_24h REAL,
            detail_json TEXT,
            PRIMARY KEY (symbol, horizon)
        );
        """
    )
    cur.execute(
        "CREATE INDEX IF NOT EXISTS idx_explore_signals_horizon_status ON explore_signals(horizon, status);"
    )


def init_db() -> None:
    """
    Creates tables if missing + performs lightweight migrations to keep older DBs compatible.
    Safe to call multiple times.
    """
    con = _conn()
    cur = con.cursor()

    # --- bots
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS bots (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            symbol TEXT NOT NULL,
            enabled INTEGER NOT NULL DEFAULT 0,
            dry_run INTEGER NOT NULL DEFAULT 1,

            base_quote REAL NOT NULL DEFAULT 0,
            safety_quote REAL NOT NULL DEFAULT 0,
            max_safety INTEGER NOT NULL DEFAULT 0,
            first_dev REAL NOT NULL DEFAULT 0.01,
            step_mult REAL NOT NULL DEFAULT 1.2,
            tp REAL NOT NULL DEFAULT 0.01,
            hard_sl_pct REAL NOT NULL DEFAULT 0.0,

            trend_filter INTEGER NOT NULL DEFAULT 0,
            trend_sma INTEGER NOT NULL DEFAULT 200,

            max_spend_quote REAL NOT NULL DEFAULT 0,
            poll_seconds INTEGER NOT NULL DEFAULT 10,
            strategy_mode TEXT NOT NULL DEFAULT 'classic',
            forced_strategy TEXT NOT NULL DEFAULT '',
            max_open_orders INTEGER NOT NULL DEFAULT 6,
            vol_gap_mult REAL NOT NULL DEFAULT 1.0,
            tp_vol_mult REAL NOT NULL DEFAULT 1.0,
            min_gap_pct REAL NOT NULL DEFAULT 0.003,
            max_gap_pct REAL NOT NULL DEFAULT 0.06,
            regime_hold_candles INTEGER NOT NULL DEFAULT 2,
            regime_switch_ticks INTEGER NOT NULL DEFAULT 2,
            regime_switch_threshold REAL NOT NULL DEFAULT 0.6,
            max_total_exposure_pct REAL NOT NULL DEFAULT 0.50,
            per_symbol_exposure_pct REAL NOT NULL DEFAULT 0.15,
            min_free_cash_pct REAL NOT NULL DEFAULT 0.1,
            max_concurrent_deals INTEGER NOT NULL DEFAULT 6,
            spread_guard_pct REAL NOT NULL DEFAULT 0.003,
            limit_timeout_sec INTEGER NOT NULL DEFAULT 8,
            daily_loss_limit_pct REAL NOT NULL DEFAULT 0.06,
            pause_hours INTEGER NOT NULL DEFAULT 6,
            auto_restart INTEGER NOT NULL DEFAULT 1,
            last_running INTEGER NOT NULL DEFAULT 0,
            
            market_type TEXT NOT NULL DEFAULT 'crypto',
            alpaca_mode TEXT NOT NULL DEFAULT 'paper',

            created_at INTEGER NOT NULL
        );
        """
    )

    # --- logs
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS bot_logs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            bot_id INTEGER NOT NULL,
            ts INTEGER NOT NULL,
            level TEXT NOT NULL,
            category TEXT NOT NULL DEFAULT 'SYSTEM',
            message TEXT NOT NULL,
            count INTEGER NOT NULL DEFAULT 1
        );
        """
    )

    # --- deals
    # --- strategy decisions
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS strategy_decisions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            bot_id INTEGER NOT NULL,
            ts INTEGER NOT NULL,
            strategy TEXT NOT NULL,
            action TEXT NOT NULL,
            reason TEXT NOT NULL,
            regime TEXT,
            confidence REAL,
            payload TEXT
        );
        """
    )

    # --- regime snapshots
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS regime_snapshots (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            bot_id INTEGER NOT NULL,
            ts INTEGER NOT NULL,
            symbol TEXT NOT NULL,
            regime TEXT NOT NULL,
            confidence REAL,
            why TEXT,
            snapshot TEXT
        );
        """
    )

    # --- order events
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS order_events (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            bot_id INTEGER NOT NULL,
            ts INTEGER NOT NULL,
            symbol TEXT NOT NULL,
            side TEXT NOT NULL,
            ord_type TEXT NOT NULL,
            price REAL,
            amount REAL,
            order_id TEXT,
            tag TEXT,
            status TEXT,
            reason TEXT
        );
        """
    )

    # --- performance metrics (strategy-level)
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS perf_metrics (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            bot_id INTEGER NOT NULL,
            ts INTEGER NOT NULL,
            strategy TEXT NOT NULL,
            payload TEXT
        );
        """
    )

    # --- backtest runs
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS backtest_runs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ts INTEGER NOT NULL,
            symbol TEXT NOT NULL,
            strategy TEXT NOT NULL,
            params TEXT,
            metrics TEXT,
            equity TEXT
        );
        """
    )
    # --- recommendations
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS recommendations_snapshots (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            symbol TEXT NOT NULL,
            horizon TEXT NOT NULL,
            score REAL NOT NULL,
            regime_json TEXT,
            metrics_json TEXT,
            reasons_json TEXT,
            risk_flags_json TEXT,
            created_ts INTEGER NOT NULL
        );
        """
    )
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS recommendations_latest (
            symbol TEXT NOT NULL,
            horizon TEXT NOT NULL,
            snapshot_id INTEGER NOT NULL,
            created_ts INTEGER NOT NULL,
            PRIMARY KEY(symbol, horizon)
        );
        """
    )
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS signal_audit (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            signal_id TEXT NOT NULL,
            symbol TEXT NOT NULL,
            asset_type TEXT NOT NULL,
            horizon TEXT NOT NULL DEFAULT 'short',
            composite_score REAL,
            confidence_score REAL,
            conviction_grade TEXT,
            factor_scores_json TEXT,
            gate_results_json TEXT,
            technical_signals_json TEXT,
            metadata_json TEXT,
            flags_json TEXT,
            rejection_reason TEXT,
            price_at_signal REAL,
            created_ts INTEGER NOT NULL
        );
        """
    )
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS explore_signals (
            symbol TEXT NOT NULL,
            horizon TEXT NOT NULL,
            status TEXT NOT NULL,
            conviction_score REAL NOT NULL DEFAULT 0,
            reason TEXT,
            strategy TEXT,
            signal_ts INTEGER NOT NULL DEFAULT 0,
            updated_ts INTEGER NOT NULL,
            market_type TEXT,
            price REAL,
            change_24h REAL,
            detail_json TEXT,
            PRIMARY KEY (symbol, horizon)
        );
        """
    )
    cur.execute(
        "CREATE INDEX IF NOT EXISTS idx_explore_signals_horizon_status ON explore_signals(horizon, status);"
    )
    cur.execute(
        "CREATE INDEX IF NOT EXISTS idx_explore_signals_status ON explore_signals(status, updated_ts DESC);"
    )
    cur.execute(
        "CREATE INDEX IF NOT EXISTS idx_explore_signals_symbol ON explore_signals(symbol, horizon);"
    )
    cur.execute(
        "CREATE INDEX IF NOT EXISTS idx_explore_signals_conviction ON explore_signals(conviction_score DESC, updated_ts DESC);"
    )
    _migrate_explore_signals_to_v2(cur)
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS explore_backtest_results (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            horizon TEXT NOT NULL,
            computed_ts INTEGER NOT NULL,
            results_json TEXT NOT NULL
        );
        """
    )
    cur.execute(
        "CREATE INDEX IF NOT EXISTS idx_explore_bt_horizon_ts ON explore_backtest_results(horizon, computed_ts DESC);"
    )
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS explore_signal_outcomes (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            symbol TEXT NOT NULL,
            horizon TEXT NOT NULL,
            strategy TEXT NOT NULL,
            signal_ts INTEGER NOT NULL,
            entry_price REAL NOT NULL,
            price_5d REAL,
            price_10d REAL,
            price_20d REAL,
            pnl_5d_pct REAL,
            pnl_10d_pct REAL,
            pnl_20d_pct REAL,
            outcome TEXT,
            composite_score REAL,
            conviction_grade TEXT,
            checked_ts INTEGER
        );
        """
    )
    cur.execute(
        "CREATE INDEX IF NOT EXISTS idx_ex_sig_out_horizon_ts ON explore_signal_outcomes(horizon, signal_ts);"
    )
    cur.execute(
        "CREATE INDEX IF NOT EXISTS idx_ex_sig_out_pending ON explore_signal_outcomes(outcome, signal_ts);"
    )
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS deals (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            bot_id INTEGER NOT NULL,
            state TEXT NOT NULL,
            opened_at INTEGER NOT NULL,
            closed_at INTEGER,
            symbol TEXT NOT NULL,

            entry_avg REAL,
            exit_avg REAL,
            base_amount REAL,
            realized_pnl_quote REAL,
            entry_regime TEXT,
            exit_regime TEXT,
            entry_strategy TEXT,
            exit_strategy TEXT,
            mae REAL,
            mfe REAL,
            hold_sec INTEGER,
            safety_count INTEGER
        );
        """
    )

    # --- strategy performance trades (rolling stats)
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS strategy_perf_trades (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            bot_id INTEGER NOT NULL,
            symbol TEXT,
            regime TEXT,
            strategy TEXT NOT NULL,
            pnl REAL NOT NULL,
            pnl_pct REAL,
            ts INTEGER NOT NULL
        );
        """
    )

    # --- settings (global app flags)
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS settings (
            key TEXT PRIMARY KEY,
            value TEXT NOT NULL
        );
        """
    )
    
    # --- intelligence_decisions (Intelligence Layer decision log)
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS intelligence_decisions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            bot_id INTEGER NOT NULL,
            ts INTEGER NOT NULL,
            symbol TEXT NOT NULL,
            allowed_actions TEXT NOT NULL,
            final_action TEXT NOT NULL,
            final_reason TEXT,
            data_ok INTEGER NOT NULL,
            data_reasons TEXT,
            safety_allowed TEXT,
            safety_reasons TEXT,
            regime TEXT,
            regime_confidence REAL,
            strategy_mode TEXT,
            entry_style TEXT,
            exit_style TEXT,
            base_size REAL,
            order_type TEXT,
            manage_actions TEXT,
            proposed_orders TEXT,
            debug_json TEXT,
            execution_result TEXT,
            realized_slippage REAL,
            fill_quality TEXT
        );
        """
    )

    # --- recommendation_performance (track recommendation accuracy vs actual trades)
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS recommendation_performance (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            symbol TEXT NOT NULL,
            recommendation_date INTEGER NOT NULL,
            score_at_recommendation REAL NOT NULL,
            regime_at_recommendation TEXT,
            bot_id INTEGER,
            deal_id INTEGER,
            entry_price REAL,
            exit_price REAL,
            pnl_realized REAL,
            days_held REAL,
            outcome TEXT NOT NULL DEFAULT 'active',
            notes TEXT,
            technical_patterns_json TEXT,
            snapshot_id INTEGER,
            created_at INTEGER NOT NULL
        );
        """
    )

    # --- scoring_calibration_log (audit trail for adaptive scoring)
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS scoring_calibration_log (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ts INTEGER NOT NULL,
            scoring_version TEXT NOT NULL,
            changes_json TEXT,
            analysis_window_days INTEGER,
            notes TEXT
        );
        """
    )

    # --- intraday_patterns (day-trading: opening range breaks, VWAP crosses, volume spikes)
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS intraday_patterns (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            bot_id INTEGER,
            symbol TEXT NOT NULL,
            pattern_type TEXT NOT NULL,
            ts INTEGER NOT NULL,
            price REAL,
            vwap REAL,
            or_high REAL,
            or_low REAL,
            volume_spike_ratio REAL,
            payload_json TEXT
        );
        """
    )

    # --- sector_performance_history (for sector rotation strategy)
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS sector_performance_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            sector TEXT NOT NULL,
            quarter_ts INTEGER NOT NULL,
            return_pct REAL,
            momentum_score REAL,
            rank INTEGER,
            payload_json TEXT
        );
        """
    )

    # --- dividend_events (dividend tracking)
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS dividend_events (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            symbol TEXT NOT NULL,
            ex_date INTEGER NOT NULL,
            payment_date INTEGER,
            amount REAL NOT NULL,
            dividend_yield_pct REAL,
            recorded_at INTEGER NOT NULL
        );
        """
    )

    # --- tax_harvest_suggestions (tax-loss harvesting suggestions)
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS tax_harvest_suggestions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            symbol TEXT NOT NULL,
            unrealized_loss_pct REAL NOT NULL,
            suggest_sell_ts INTEGER NOT NULL,
            wash_sale_until_ts INTEGER,
            alternate_symbol TEXT,
            recorded_at INTEGER NOT NULL
        );
        """
    )

    # --- autopilot_config (Master Upgrade Part 4 - full autopilot)
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS autopilot_config (
            id INTEGER PRIMARY KEY CHECK (id = 1),
            enabled INTEGER DEFAULT 0,
            total_capital_allocated REAL DEFAULT 100.0,
            max_positions INTEGER DEFAULT 3,
            position_size_mode TEXT DEFAULT 'conviction_based',
            asset_types TEXT DEFAULT 'both',
            min_score_threshold INTEGER DEFAULT 75,
            min_conviction_threshold INTEGER DEFAULT 5,
            excluded_sectors TEXT,
            max_loss_per_day_pct REAL DEFAULT 5.0,
            max_loss_per_week_pct REAL DEFAULT 10.0,
            max_correlated_exposure_pct REAL DEFAULT 50.0,
            max_sector_exposure_pct REAL DEFAULT 50.0,
            scan_frequency_hours INTEGER DEFAULT 4,
            auto_create_bots INTEGER DEFAULT 1,
            auto_start_bots INTEGER DEFAULT 1,
            auto_close_underperformers INTEGER DEFAULT 1,
            underperformer_threshold_score INTEGER DEFAULT 60,
            rebalance_enabled INTEGER DEFAULT 1,
            rebalance_frequency_days INTEGER DEFAULT 7,
            notify_on_new_bot INTEGER DEFAULT 1,
            notify_on_close INTEGER DEFAULT 1,
            notify_daily_summary INTEGER DEFAULT 1,
            last_scan INTEGER,
            last_rebalance INTEGER
        );
        """
    )
    cur.execute("INSERT OR IGNORE INTO autopilot_config (id, enabled) VALUES (1, 0)")
    # Migration: bump exposure limits and reduce defaults (11.md critical fixes)
    _ensure_column(con, "autopilot_config", "capital_per_bot", "REAL DEFAULT 500.0")
    cur.execute("""
        UPDATE autopilot_config SET
            max_correlated_exposure_pct = 50.0,
            max_sector_exposure_pct = 50.0
        WHERE id = 1 AND (max_correlated_exposure_pct < 50 OR max_sector_exposure_pct < 50)
    """)
    # Migrate: correct autopilot capital if still at factory default high value
    try:
        cur.execute("""
            UPDATE autopilot_config
            SET total_capital_allocated = 100.0,
                max_positions = 3,
                capital_per_bot = 10.0
            WHERE id = 1
            AND total_capital_allocated >= 1000.0
        """)
    except Exception:
        pass
    con.commit()

    # --- portfolio_snapshots (for charts - 11.md)
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS portfolio_snapshots (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            total_value REAL,
            total_pnl REAL,
            total_pnl_pct REAL,
            active_positions INTEGER,
            realized_pnl REAL,
            unrealized_pnl REAL
        )
        """
    )
    cur.execute(
        "CREATE INDEX IF NOT EXISTS idx_portfolio_snapshots_ts ON portfolio_snapshots(timestamp)"
    )
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS signal_accuracy_baseline (
            strategy_id TEXT NOT NULL,
            horizon TEXT NOT NULL,
            asset_type TEXT NOT NULL,
            sample_size INTEGER DEFAULT 0,
            win_rate REAL DEFAULT 0,
            avg_return_pct REAL DEFAULT 0,
            avg_hold_hours REAL DEFAULT 0,
            sharpe_ratio REAL DEFAULT 0,
            last_updated INTEGER DEFAULT 0,
            PRIMARY KEY (strategy_id, horizon, asset_type)
        );
        """
    )
    try:
        _bcnt = cur.execute("SELECT COUNT(*) FROM signal_accuracy_baseline").fetchone()
        if _bcnt and int(_bcnt[0]) == 0:
            _tsb = now_ts()
            _seed_rows = [
                ("momentum_breakout", "short", "crypto", 150, 0.44, 2.4, 36.0, 0.9, _tsb),
                ("pullback_support", "medium", "crypto", 120, 0.41, 3.1, 72.0, 0.75, _tsb),
                ("oversold_reversal", "medium", "stock", 200, 0.39, 1.8, 120.0, 0.55, _tsb),
                ("trend_continuation", "long", "crypto", 90, 0.38, 4.5, 240.0, 0.62, _tsb),
                ("crypto_momentum", "short", "crypto", 180, 0.46, 2.9, 24.0, 1.0, _tsb),
                ("volume_capitulation", "medium", "crypto", 60, 0.52, 3.8, 96.0, 0.7, _tsb),
                ("oversold_bounce", "medium", "crypto", 100, 0.40, 2.2, 80.0, 0.58, _tsb),
            ]
            cur.executemany(
                """
                INSERT INTO signal_accuracy_baseline(
                    strategy_id, horizon, asset_type, sample_size, win_rate,
                    avg_return_pct, avg_hold_hours, sharpe_ratio, last_updated
                ) VALUES (?,?,?,?,?,?,?,?,?)
                """,
                _seed_rows,
            )
    except Exception as _seed_bl_err:
        logger.debug("signal_accuracy_baseline seed: %s", _seed_bl_err)

    cur.execute("CREATE INDEX IF NOT EXISTS idx_recommendations_latest_horizon ON recommendations_latest(horizon)")

    # --- market_events (earnings, Fed, etc. - avoid entries day before)
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS market_events (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            event_date INTEGER NOT NULL,
            event_type TEXT NOT NULL,
            symbol TEXT,
            impact_level INTEGER NOT NULL DEFAULT 2,
            description TEXT,
            recorded_at INTEGER NOT NULL
        );
        """
    )

    # --- insider_transactions (SEC Form 4 - CEO/CFO buys/sells)
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS insider_transactions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            symbol TEXT NOT NULL,
            transaction_date INTEGER NOT NULL,
            transaction_type TEXT NOT NULL,
            shares REAL NOT NULL DEFAULT 0,
            value_usd REAL,
            insider_title TEXT,
            filing_url TEXT,
            recorded_at INTEGER NOT NULL
        );
        """
    )

    # --- ml_predictions (log every ML prediction, track outcomes 7d/30d)
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS ml_predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            symbol TEXT NOT NULL,
            prediction_date INTEGER NOT NULL,
            predicted_direction TEXT NOT NULL,
            predicted_price REAL,
            confidence REAL NOT NULL DEFAULT 0,
            price_at_prediction REAL,
            actual_outcome_7d REAL,
            actual_outcome_30d REAL,
            model_version TEXT,
            model_used TEXT,
            regime_at_prediction TEXT,
            recorded_at INTEGER NOT NULL
        );
        """
    )

    # --- ml_model_versions (track deployed model versions, validation accuracy)
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS ml_model_versions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            model_type TEXT NOT NULL,
            version TEXT NOT NULL,
            validation_accuracy REAL,
            trained_at INTEGER NOT NULL,
            deployed INTEGER NOT NULL DEFAULT 0
        );
        """
    )

    # --- execution_quality (slippage tracking, post-trade analysis)
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS execution_quality (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            order_id TEXT,
            bot_id INTEGER,
            symbol TEXT NOT NULL,
            side TEXT NOT NULL,
            strategy TEXT,
            intended_price REAL NOT NULL,
            executed_price REAL,
            slippage_pct REAL,
            slippage_dollars REAL,
            vwap_at_execution REAL,
            twap_at_execution REAL,
            execution_quality_score INTEGER,
            created_at INTEGER NOT NULL
        );
        """
    )

    # --- data_quality_log (missing candles, stale prices, spreads, volume anomalies)
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS data_quality_log (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ts INTEGER NOT NULL,
            source TEXT NOT NULL,
            issue_type TEXT NOT NULL,
            severity TEXT NOT NULL DEFAULT 'info',
            details_json TEXT
        );
        """
    )
    cur.execute("CREATE INDEX IF NOT EXISTS idx_data_quality_ts ON data_quality_log(ts);")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_data_quality_severity ON data_quality_log(severity, ts);")

    # --- error_log (API errors, circuit breaker events, recovery logs)
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS error_log (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ts INTEGER NOT NULL,
            source TEXT NOT NULL,
            error_type TEXT NOT NULL,
            message TEXT,
            bot_id INTEGER,
            details_json TEXT
        );
        """
    )
    cur.execute("CREATE INDEX IF NOT EXISTS idx_error_log_ts ON error_log(ts);")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_error_log_source ON error_log(source, ts);")

    # --- autopilot_audit_log (LIVE-HARDENED: every autopilot decision with reason)
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS autopilot_audit_log (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ts INTEGER NOT NULL,
            action TEXT NOT NULL,
            symbol TEXT,
            reason TEXT,
            details_json TEXT
        );
        """
    )
    cur.execute("CREATE INDEX IF NOT EXISTS idx_autopilot_audit_ts ON autopilot_audit_log(ts);")

    # --- trade_journal (entry/exit reasons, lessons, screenshots for closed deals)
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS trade_journal (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            deal_id INTEGER NOT NULL UNIQUE,
            entry_reason TEXT,
            exit_reason TEXT,
            lessons_learned TEXT,
            screenshot_data TEXT,
            updated_at INTEGER NOT NULL
        );
        """
    )
    cur.execute("CREATE INDEX IF NOT EXISTS idx_trade_journal_deal ON trade_journal(deal_id);")

    # --- scanner_watchlist: symbols identified by market_scanner but not yet ready
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS scanner_watchlist (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            symbol TEXT NOT NULL,
            market_type TEXT NOT NULL DEFAULT 'crypto',
            setup_json TEXT NOT NULL DEFAULT '{}',
            trigger_conditions TEXT NOT NULL DEFAULT '',
            regime TEXT NOT NULL DEFAULT '',
            entry_type TEXT NOT NULL DEFAULT '',
            confidence REAL NOT NULL DEFAULT 0.0,
            edge_score REAL NOT NULL DEFAULT 0.0,
            status TEXT NOT NULL DEFAULT 'watching',
            created_at INTEGER NOT NULL,
            updated_at INTEGER NOT NULL,
            triggered_at INTEGER,
            bot_id INTEGER
        );
        """
    )
    cur.execute("CREATE INDEX IF NOT EXISTS idx_watchlist_symbol ON scanner_watchlist(symbol, status);")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_watchlist_status ON scanner_watchlist(status);")

    # --- notifications: in-app alerts for trades, profits, losses, errors
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS notifications (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp INTEGER NOT NULL,
            type TEXT NOT NULL,
            title TEXT NOT NULL,
            message TEXT NOT NULL,
            bot_id INTEGER,
            read INTEGER DEFAULT 0
        );
        """
    )
    cur.execute("CREATE INDEX IF NOT EXISTS idx_notifications_ts ON notifications(timestamp);")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_notifications_read ON notifications(read);")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_notifications_type ON notifications(type);")

    # --- audit log: security and compliance tracking
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS audit_log (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp REAL NOT NULL,
            action TEXT NOT NULL,
            details TEXT,
            ip TEXT
        );
        """
    )
    cur.execute("CREATE INDEX IF NOT EXISTS idx_audit_timestamp ON audit_log(timestamp);")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_audit_action ON audit_log(action);")

    # --- trade_feedback: ML learning from closed trades
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS trade_feedback (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            symbol TEXT NOT NULL,
            timestamp REAL NOT NULL,
            features_json TEXT,
            profitable INTEGER NOT NULL
        );
        """
    )
    cur.execute("CREATE INDEX IF NOT EXISTS idx_trade_feedback_symbol ON trade_feedback(symbol);")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_trade_feedback_profitable ON trade_feedback(profitable);")

    # Helpful indexes
    cur.execute("CREATE INDEX IF NOT EXISTS idx_logs_bot_ts ON bot_logs(bot_id, ts);")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_logs_bot_id ON bot_logs(bot_id);")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_logs_ts ON bot_logs(ts);")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_bots_symbol ON bots(symbol);")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_bots_enabled ON bots(enabled);")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_deals_bot_id ON deals(bot_id);")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_deals_state ON deals(state);")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_deals_symbol ON deals(symbol);")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_deals_bot_state ON deals(bot_id, state);")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_deals_closed_at ON deals(closed_at);")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_intel_bot_ts ON intelligence_decisions(bot_id, ts);")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_reco_perf_bot_outcome ON recommendation_performance(bot_id, outcome);")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_reco_perf_symbol ON recommendation_performance(symbol, outcome);")
    # Signal outcome tracking columns on recommendations_snapshots
    for col_def in [
        ("entry_price", "REAL"),
        ("price_24h", "REAL"),
        ("price_72h", "REAL"),
        ("outcome_24h", "TEXT"),
        ("outcome_72h", "TEXT"),
        ("outcome_checked", "INTEGER DEFAULT 0"),
    ]:
        try:
            cur.execute(f"ALTER TABLE recommendations_snapshots ADD COLUMN {col_def[0]} {col_def[1]}")
        except Exception:
            pass
    cur.execute("CREATE INDEX IF NOT EXISTS idx_reco_snap_outcome ON recommendations_snapshots(outcome_checked, created_ts);")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_intraday_symbol_ts ON intraday_patterns(symbol, ts);")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_sector_perf_sector_ts ON sector_performance_history(sector, quarter_ts);")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_dividend_symbol_ex ON dividend_events(symbol, ex_date);")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_market_events_date ON market_events(event_date, event_type);")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_insider_symbol_date ON insider_transactions(symbol, transaction_date);")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_ml_predictions_symbol_date ON ml_predictions(symbol, prediction_date);")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_ml_predictions_recorded ON ml_predictions(recorded_at);")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_execution_quality_symbol ON execution_quality(symbol, created_at);")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_execution_quality_strategy ON execution_quality(strategy, created_at);")

    # Lightweight migrations (in case older DB exists with missing columns)
    try:
        _ensure_column(con, "bots", "enabled", "INTEGER NOT NULL DEFAULT 0")
        _ensure_column(con, "bots", "dry_run", "INTEGER NOT NULL DEFAULT 1")

        _ensure_column(con, "bots", "trend_filter", "INTEGER NOT NULL DEFAULT 0")
        _ensure_column(con, "bots", "trend_sma", "INTEGER NOT NULL DEFAULT 200")
        _ensure_column(con, "bots", "max_spend_quote", "REAL NOT NULL DEFAULT 0")
        _ensure_column(con, "bots", "poll_seconds", "INTEGER NOT NULL DEFAULT 10")
        _ensure_column(con, "bots", "strategy_mode", "TEXT NOT NULL DEFAULT 'classic'")
        _ensure_column(con, "bots", "forced_strategy", "TEXT NOT NULL DEFAULT ''")
        _ensure_column(con, "bots", "max_open_orders", "INTEGER NOT NULL DEFAULT 6")
        _ensure_column(con, "bots", "vol_gap_mult", "REAL NOT NULL DEFAULT 1.0")
        _ensure_column(con, "bots", "tp_vol_mult", "REAL NOT NULL DEFAULT 1.0")
        _ensure_column(con, "bots", "min_gap_pct", "REAL NOT NULL DEFAULT 0.003")
        _ensure_column(con, "bots", "max_gap_pct", "REAL NOT NULL DEFAULT 0.06")
        _ensure_column(con, "bots", "regime_hold_candles", "INTEGER NOT NULL DEFAULT 2")
        _ensure_column(con, "bots", "regime_switch_ticks", "INTEGER NOT NULL DEFAULT 2")
        _ensure_column(con, "bots", "regime_switch_threshold", "REAL NOT NULL DEFAULT 0.6")
        _ensure_column(con, "bots", "max_total_exposure_pct", "REAL NOT NULL DEFAULT 0.50")
        _ensure_column(con, "bots", "per_symbol_exposure_pct", "REAL NOT NULL DEFAULT 0.15")
        _ensure_column(con, "bots", "min_free_cash_pct", "REAL NOT NULL DEFAULT 0.1")
        _ensure_column(con, "bots", "max_concurrent_deals", "INTEGER NOT NULL DEFAULT 6")
        _ensure_column(con, "bots", "spread_guard_pct", "REAL NOT NULL DEFAULT 0.003")
        _ensure_column(con, "bots", "limit_timeout_sec", "INTEGER NOT NULL DEFAULT 8")
        _ensure_column(con, "bots", "daily_loss_limit_pct", "REAL NOT NULL DEFAULT 0.06")
        _ensure_column(con, "bots", "pause_hours", "INTEGER NOT NULL DEFAULT 6")
        _ensure_column(con, "bots", "auto_restart", "INTEGER NOT NULL DEFAULT 1")
        _ensure_column(con, "bots", "last_running", "INTEGER NOT NULL DEFAULT 0")
        _ensure_column(con, "bots", "market_type", "TEXT NOT NULL DEFAULT 'crypto'")
        _ensure_column(con, "bots", "alpaca_mode", "TEXT NOT NULL DEFAULT 'paper'")

        _ensure_column(con, "order_events", "is_live", "INTEGER DEFAULT 0")

        # Phase 1: Quick Wins - Trailing Stop Loss
        _ensure_column(con, "bots", "trailing_stop_enabled", "INTEGER NOT NULL DEFAULT 1")
        _ensure_column(con, "bots", "trailing_activation_pct", "REAL NOT NULL DEFAULT 0.02")
        _ensure_column(con, "bots", "trailing_distance_pct", "REAL NOT NULL DEFAULT 0.01")
        
        # Phase 1: Cooldown After Stop Loss
        _ensure_column(con, "bots", "stop_loss_cooldown_sec", "INTEGER NOT NULL DEFAULT 172800")  # 48h after SL
        _ensure_column(con, "bots", "last_stop_loss_at", "INTEGER")
        
        # Phase 1: Volatility-Based TP Scaling
        _ensure_column(con, "bots", "adaptive_tp_enabled", "INTEGER NOT NULL DEFAULT 1")
        _ensure_column(con, "bots", "tp_volatility_mult", "REAL NOT NULL DEFAULT 1.5")

        # Phase 2: Hard Stop Loss (Emergency Exit)
        _ensure_column(con, "bots", "hard_sl_pct", "REAL NOT NULL DEFAULT 0.0")

        # Phase 2: Grid Trading Parameters
        _ensure_column(con, "bots", "grid_lower", "REAL NOT NULL DEFAULT 0.0")
        _ensure_column(con, "bots", "grid_upper", "REAL NOT NULL DEFAULT 0.0")
        _ensure_column(con, "bots", "grid_levels", "INTEGER NOT NULL DEFAULT 10")

        # Phase 1: BTC Correlation Guard
        _ensure_column(con, "bots", "btc_correlation_guard", "INTEGER NOT NULL DEFAULT 1")
        _ensure_column(con, "bots", "btc_dump_threshold_pct", "REAL NOT NULL DEFAULT 0.05")
        
        # Phase 1: Time-Based Filters
        _ensure_column(con, "bots", "time_filter_enabled", "INTEGER NOT NULL DEFAULT 1")
        _ensure_column(con, "bots", "skip_first_30min", "INTEGER NOT NULL DEFAULT 1")
        _ensure_column(con, "bots", "skip_last_30min", "INTEGER NOT NULL DEFAULT 1")
        
        # Phase 1: Adaptive Volume & Spread
        _ensure_column(con, "bots", "min_volume_ratio", "REAL NOT NULL DEFAULT 1.5")
        _ensure_column(con, "bots", "adaptive_spread_enabled", "INTEGER NOT NULL DEFAULT 1")
        _ensure_column(con, "bots", "max_drawdown_pct", "REAL NOT NULL DEFAULT 0.15")
        
        # Phase A: Kelly Criterion position sizing
        _ensure_column(con, "bots", "use_kelly_sizing", "INTEGER NOT NULL DEFAULT 1")
        _ensure_column(con, "bots", "kelly_fraction", "REAL NOT NULL DEFAULT 0.25")
        _ensure_column(con, "bots", "max_position_pct", "REAL NOT NULL DEFAULT 0.10")
        # Live trading gate: require explicit confirmation for real orders
        _ensure_column(con, "bots", "live_confirmed", "INTEGER NOT NULL DEFAULT 1")
        # Day-trading / scalping
        _ensure_column(con, "bots", "day_trading_mode", "INTEGER NOT NULL DEFAULT 0")
        _ensure_column(con, "bots", "scalp_trailing_activation_pct", "REAL NOT NULL DEFAULT 0.05")
        _ensure_column(con, "bots", "scalp_trailing_distance_pct", "REAL NOT NULL DEFAULT 0.002")
        _ensure_column(con, "bots", "auto_close_eod", "INTEGER NOT NULL DEFAULT 0")
        _ensure_column(con, "bots", "long_term_mode", "INTEGER NOT NULL DEFAULT 0")
        _ensure_column(con, "bots", "long_term_trailing_pct", "REAL NOT NULL DEFAULT 0.15")

        # Master Upgrade: Trading mode system (Part 1)
        _ensure_column(con, "bots", "trading_mode", "TEXT DEFAULT 'swing_trade'")
        _ensure_column(con, "bots", "intended_hold_days", "INTEGER DEFAULT 14")
        _ensure_column(con, "bots", "conviction_level", "INTEGER DEFAULT 5")
        _ensure_column(con, "bots", "auto_dip_buy", "INTEGER DEFAULT 0")
        _ensure_column(con, "bots", "fundamental_exit_only", "INTEGER DEFAULT 0")
        _ensure_column(con, "bots", "rebalance_enabled", "INTEGER DEFAULT 0")
        _ensure_column(con, "bots", "bot_type", "TEXT DEFAULT ''")

        # Universal stop-loss and time-based exit
        _ensure_column(con, "bots", "stop_loss_pct", "REAL NOT NULL DEFAULT 0.08")
        _ensure_column(con, "bots", "max_hold_hours", "INTEGER NOT NULL DEFAULT 0")
        _ensure_column(con, "bots", "risk_profile", "TEXT NOT NULL DEFAULT 'balanced'")

        # Deals table - tracking for trailing stops
        _ensure_column(con, "deals", "highest_price", "REAL")
        _ensure_column(con, "deals", "trailing_stop_price", "REAL")
        _ensure_column(con, "deals", "trailing_stop_active", "INTEGER NOT NULL DEFAULT 0")
        
        _ensure_column(con, "bot_logs", "category", "TEXT NOT NULL DEFAULT 'SYSTEM'")
        _ensure_column(con, "bot_logs", "count", "INTEGER NOT NULL DEFAULT 1")
        _ensure_column(con, "deals", "entry_regime", "TEXT")
        _ensure_column(con, "deals", "exit_regime", "TEXT")
        _ensure_column(con, "deals", "entry_strategy", "TEXT")
        _ensure_column(con, "deals", "exit_strategy", "TEXT")
        _ensure_column(con, "deals", "mae", "REAL")
        _ensure_column(con, "deals", "mfe", "REAL")
        _ensure_column(con, "deals", "hold_sec", "INTEGER")
        _ensure_column(con, "deals", "safety_count", "INTEGER")
        _ensure_column(con, "deals", "realized_pnl_pct", "REAL")
        _ensure_column(con, "deals", "entry_avg_estimated", "INTEGER NOT NULL DEFAULT 0")
        _ensure_column(con, "deals", "data_source", "TEXT")
        _ensure_column(con, "explore_signals", "rejection_reason", "TEXT")
        _ensure_column(con, "strategy_perf_trades", "symbol", "TEXT")
        _ensure_column(con, "strategy_perf_trades", "regime", "TEXT")
        _ensure_column(con, "strategy_perf_trades", "pnl_pct", "REAL")
        _ensure_column(con, "recommendations_snapshots", "scoring_version", "TEXT NOT NULL DEFAULT 'v1'")
        _ensure_column(con, "recommendations_snapshots", "score_breakdown_json", "TEXT")
        _ensure_column(con, "recommendations_snapshots", "composite_score", "REAL")
        _ensure_column(con, "recommendations_snapshots", "confidence_score", "REAL")
        _ensure_column(con, "recommendations_snapshots", "conviction_grade", "TEXT")
        _ensure_column(con, "recommendations_snapshots", "factor_scores_json", "TEXT")
        _ensure_column(con, "recommendations_snapshots", "signal_flags_json", "TEXT")
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS explore_signals (
                symbol TEXT NOT NULL,
                horizon TEXT NOT NULL,
                status TEXT NOT NULL,
                conviction_score REAL NOT NULL DEFAULT 0,
                reason TEXT,
                strategy TEXT,
                signal_ts INTEGER NOT NULL DEFAULT 0,
                updated_ts INTEGER NOT NULL,
                market_type TEXT,
                price REAL,
                change_24h REAL,
                detail_json TEXT,
                PRIMARY KEY (symbol, horizon)
            );
            """
        )
        cur.execute(
            "CREATE INDEX IF NOT EXISTS idx_explore_signals_horizon_status ON explore_signals(horizon, status);"
        )
        cur.execute(
            "CREATE INDEX IF NOT EXISTS idx_explore_signals_status ON explore_signals(status, updated_ts DESC);"
        )
        cur.execute(
            "CREATE INDEX IF NOT EXISTS idx_explore_signals_symbol ON explore_signals(symbol, horizon);"
        )
        cur.execute(
            "CREATE INDEX IF NOT EXISTS idx_explore_signals_conviction ON explore_signals(conviction_score DESC, updated_ts DESC);"
        )
        _migrate_explore_signals_to_v2(cur)
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS explore_backtest_results (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                horizon TEXT NOT NULL,
                computed_ts INTEGER NOT NULL,
                results_json TEXT NOT NULL
            );
            """
        )
        cur.execute(
            "CREATE INDEX IF NOT EXISTS idx_explore_bt_horizon_ts ON explore_backtest_results(horizon, computed_ts DESC);"
        )
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS explore_signal_outcomes (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                symbol TEXT NOT NULL,
                horizon TEXT NOT NULL,
                strategy TEXT NOT NULL,
                signal_ts INTEGER NOT NULL,
                entry_price REAL NOT NULL,
                price_5d REAL,
                price_10d REAL,
                price_20d REAL,
                pnl_5d_pct REAL,
                pnl_10d_pct REAL,
                pnl_20d_pct REAL,
                outcome TEXT,
                composite_score REAL,
                conviction_grade TEXT,
                checked_ts INTEGER
            );
            """
        )
        cur.execute(
            "CREATE INDEX IF NOT EXISTS idx_ex_sig_out_horizon_ts ON explore_signal_outcomes(horizon, signal_ts);"
        )
        cur.execute(
            "CREATE INDEX IF NOT EXISTS idx_ex_sig_out_pending ON explore_signal_outcomes(outcome, signal_ts);"
        )
        # Indexes that depend on migrated columns
        cur.execute("CREATE INDEX IF NOT EXISTS idx_strategy_perf_bot_ts ON strategy_perf_trades(bot_id, ts);")
        cur.execute("CREATE INDEX IF NOT EXISTS idx_strategy_perf_sym ON strategy_perf_trades(symbol, ts);")
        cur.execute("CREATE INDEX IF NOT EXISTS idx_signal_audit_symbol ON signal_audit(symbol, created_ts);")
        cur.execute("CREATE INDEX IF NOT EXISTS idx_signal_audit_grade ON signal_audit(conviction_grade, created_ts);")
    except Exception:
        # If migration fails, do not crash app; tables still usable.
        pass

    try:
        con.execute("""
            UPDATE bots
            SET max_hold_hours = 72
            WHERE id IN (59, 42)
            AND (max_hold_hours IS NULL OR max_hold_hours = 0)
        """)
        con.commit()
    except Exception:
        pass

    con.commit()
    con.close()


# =========================================================
# Logs
# =========================================================
def _normalize_message(msg: str) -> str:
    s = str(msg or "").strip()
    if s.startswith("[") and "]" in s[:32]:
        s = s.split("]", 1)[-1].strip()
    for sep in ("AM ", "PM "):
        if sep in s[:32]:
            s = s.split(sep, 1)[-1].strip()
    return s


def add_log(bot_id: int, level: str, message: str, category: str = "SYSTEM") -> None:
    """Append a bot log row, dedup-collapsing consecutive duplicates.

    Migrated to write_txn(bot_id, ...) in Phase 1.2b: per-bot RLock + 5-retry
    backoff means concurrent ticks no longer blow up with
    OperationalError('database is locked'). The dedup SELECT and the
    INSERT/UPDATE now run inside a single transaction so two parallel
    ticks can't both INSERT a row that should have collapsed.
    """
    norm = _normalize_message(message)

    def _do(con) -> None:
        row = con.execute(
            "SELECT id, level, category, message, count FROM bot_logs "
            "WHERE bot_id=? ORDER BY id DESC LIMIT 1",
            (int(bot_id),),
        ).fetchone()
        if row:
            last_norm = _normalize_message(row["message"])
            if (
                str(row["level"]) == str(level)
                and str(row["category"]) == str(category)
                and last_norm == norm
            ):
                con.execute(
                    "UPDATE bot_logs SET ts=?, count=? WHERE id=?",
                    (now_ts(), int(row["count"] or 1) + 1, int(row["id"])),
                )
                return
        con.execute(
            "INSERT INTO bot_logs(bot_id, ts, level, category, message, count) "
            "VALUES (?,?,?,?,?,?)",
            (int(bot_id), now_ts(), str(level), str(category), str(message), 1),
        )

    write_txn(int(bot_id), _do, name="add_log")


def list_logs(bot_id: int, limit: int = 200) -> List[Dict[str, Any]]:
    con = _conn()
    rows = con.execute(
        "SELECT ts, level, category, message, count FROM bot_logs WHERE bot_id=? ORDER BY ts DESC LIMIT ?",
        (int(bot_id), int(limit)),
    ).fetchall()
    con.close()
    return [dict(r) for r in rows]


def list_logs_window(bot_id: int, start_ts: int, end_ts: int, limit: int = 200) -> List[Dict[str, Any]]:
    con = _conn()
    rows = con.execute(
        """
        SELECT ts, level, category, message, count
        FROM bot_logs
        WHERE bot_id=? AND ts>=? AND ts<=?
        ORDER BY ts ASC
        LIMIT ?
        """,
        (int(bot_id), int(start_ts), int(end_ts), int(limit)),
    ).fetchall()
    con.close()
    return [dict(r) for r in rows]


def log_data_quality(source: str, issue_type: str, severity: str, details: Optional[Dict] = None) -> None:
    """Log data quality issue to data_quality_log."""
    con = _conn()
    try:
        con.execute(
            "INSERT INTO data_quality_log(ts, source, issue_type, severity, details_json) VALUES (?,?,?,?,?)",
            (now_ts(), str(source)[:64], str(issue_type)[:64], str(severity)[:32],
             str(__import__("json").dumps(details)) if details else None),
        )
        con.commit()
    finally:
        con.close()


def get_recent_data_quality_count(minutes: int = 15, min_severity: str = "warning") -> int:
    """Count data quality issues in last N minutes. Severity order: critical > error > warning > info."""
    sev_order = {"critical": 0, "error": 1, "warning": 2, "info": 3}
    min_level = sev_order.get(min_severity.lower(), 2)
    con = _conn()
    try:
        since = now_ts() - minutes * 60
        rows = con.execute(
            "SELECT severity FROM data_quality_log WHERE ts >= ?",
            (since,),
        ).fetchall()
        count = 0
        for r in rows:
            if sev_order.get(str(r["severity"] or "").lower(), 99) <= min_level:
                count += 1
        return count
    finally:
        con.close()


def log_error(source: str, error_type: str, message: Optional[str] = None, bot_id: Optional[int] = None, details: Optional[Dict] = None) -> None:
    """Log error to error_log table."""
    con = _conn()
    try:
        con.execute(
            "INSERT INTO error_log(ts, source, error_type, message, bot_id, details_json) VALUES (?,?,?,?,?,?)",
            (now_ts(), str(source)[:64], str(error_type)[:64], (message or "")[:1024],
             int(bot_id) if bot_id else None,
             str(__import__("json").dumps(details)) if details else None),
        )
        con.commit()
    finally:
        con.close()


def add_autopilot_audit_log(action: str, symbol: Optional[str] = None, reason: Optional[str] = None, details: Optional[Dict] = None) -> None:
    """LIVE-HARDENED: Log every autopilot decision for traceability."""
    con = _conn()
    try:
        con.execute(
            "INSERT INTO autopilot_audit_log(ts, action, symbol, reason, details_json) VALUES (?,?,?,?,?)",
            (now_ts(), str(action)[:64], (symbol or "")[:32], (reason or "")[:512],
             str(__import__("json").dumps(details)) if details else None),
        )
        con.commit()
    finally:
        con.close()


def list_autopilot_audit_log(limit: int = 50) -> List[Dict[str, Any]]:
    """Return latest autopilot audit entries (newest first)."""
    con = _conn()
    try:
        rows = con.execute(
            "SELECT id, ts, action, symbol, reason, details_json FROM autopilot_audit_log ORDER BY ts DESC LIMIT ?",
            (int(limit),),
        ).fetchall()
        return [dict(r) for r in rows]
    finally:
        con.close()


# =========================================================
# Bots CRUD
# =========================================================

def list_logs_since(bot_id: int, last_id: int, limit: int = 200) -> List[Dict[str, Any]]:
    """Return logs newer than last_id (ascending order)."""
    con = _conn()
    rows = con.execute(
        "SELECT id, ts, level, category, message, count FROM bot_logs WHERE bot_id=? AND id>? ORDER BY id ASC LIMIT ?",
        (int(bot_id), int(last_id), int(limit)),
    ).fetchall()
    con.close()
    return [dict(r) for r in rows]

def create_bot(data: Dict[str, Any]) -> int:
    con = _conn()
    cur = con.cursor()
    cur.execute(
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
        (
            str(data["name"]),
            str(data["symbol"]),
            int(data.get("enabled", 0)),
            int(data.get("dry_run", 1)),
            float(data["base_quote"]),
            float(data["safety_quote"]),
            int(data["max_safety"]),
            float(data["first_dev"]),
            float(data["step_mult"]),
            float(data["tp"]),
            int(data.get("trend_filter", 0)),
            int(data.get("trend_sma", 200)),
            float(data["max_spend_quote"]),
            int(data.get("poll_seconds", 10)),
            str(data.get("strategy_mode", "classic")),
            str(data.get("forced_strategy", "")),
            int(data.get("max_open_orders", 6)),
            float(data.get("vol_gap_mult", 1.0)),
            float(data.get("tp_vol_mult", 1.0)),
            float(data.get("min_gap_pct", 0.003)),
            float(data.get("max_gap_pct", 0.06)),
            int(data.get("regime_hold_candles", 2)),
            int(data.get("regime_switch_ticks", 2)),
            float(data.get("regime_switch_threshold", 0.6)),
            float(data.get("max_total_exposure_pct", 0.50)),
            float(data.get("per_symbol_exposure_pct", 0.15)),
            float(data.get("min_free_cash_pct", 0.1)),
            int(data.get("max_concurrent_deals", 6)),
            float(data.get("spread_guard_pct", 0.003)),
            int(data.get("limit_timeout_sec", 8)),
            float(data.get("daily_loss_limit_pct", 0.06)),
            int(data.get("pause_hours", 6)),
            int(data.get("auto_restart", 1)),
            int(data.get("last_running", 0)),
            str(data.get("market_type", "crypto")),
            str(data.get("alpaca_mode", "paper")),
            str(data.get("bot_type", "")),
            now_ts(),
        ),
    )
    con.commit()
    bot_id = int(cur.lastrowid)
    con.close()
    return bot_id


def patch_bot_risk_after_create(
    bot_id: int,
    *,
    stop_loss_pct: Optional[float] = None,
    max_hold_hours: Optional[int] = None,
) -> None:
    """Update risk columns not included in INSERT INTO bots (create_bot)."""
    if stop_loss_pct is None and max_hold_hours is None:
        return
    bid = int(bot_id)
    con = _conn()
    try:
        if stop_loss_pct is not None and max_hold_hours is not None:
            con.execute(
                "UPDATE bots SET stop_loss_pct=?, max_hold_hours=? WHERE id=?",
                (float(stop_loss_pct), int(max_hold_hours), bid),
            )
        elif stop_loss_pct is not None:
            con.execute(
                "UPDATE bots SET stop_loss_pct=? WHERE id=?",
                (float(stop_loss_pct), bid),
            )
        else:
            con.execute(
                "UPDATE bots SET max_hold_hours=? WHERE id=?",
                (int(max_hold_hours or 0), bid),
            )
        con.commit()
    finally:
        con.close()


def update_bot(bot_id: int, data: Dict[str, Any]) -> None:
    con = _conn()
    con.execute(
        """
        UPDATE bots SET
            name=?,
            symbol=?,
            enabled=?,
            dry_run=?,
            base_quote=?,
            safety_quote=?,
            max_safety=?,
            first_dev=?,
            step_mult=?,
            tp=?,
            hard_sl_pct=?,
            trend_filter=?,
            trend_sma=?,
            max_spend_quote=?,
            poll_seconds=?,
            strategy_mode=?,
            forced_strategy=?,
            max_open_orders=?,
            vol_gap_mult=?,
            tp_vol_mult=?,
            min_gap_pct=?,
            max_gap_pct=?,
            regime_hold_candles=?,
            regime_switch_ticks=?,
            regime_switch_threshold=?,
            max_total_exposure_pct=?,
            per_symbol_exposure_pct=?,
            min_free_cash_pct=?,
            max_concurrent_deals=?,
            spread_guard_pct=?,
            limit_timeout_sec=?,
            daily_loss_limit_pct=?,
            pause_hours=?,
            auto_restart=?,
            market_type=?,
            alpaca_mode=?,
            max_drawdown_pct=?,
            trading_mode=?,
            intended_hold_days=?,
            conviction_level=?,
            auto_dip_buy=?,
            fundamental_exit_only=?,
            rebalance_enabled=?,
            grid_lower=?,
            grid_upper=?,
            grid_levels=?
        WHERE id=?
        """,
        (
            str(data["name"]),
            str(data["symbol"]),
            int(data.get("enabled", 0)),
            int(data.get("dry_run", 1)),
            float(data["base_quote"]),
            float(data["safety_quote"]),
            int(data["max_safety"]),
            float(data["first_dev"]),
            float(data["step_mult"]),
            float(data["tp"]),
            float(data.get("hard_sl_pct", 0.0)),
            int(data.get("trend_filter", 0)),
            int(data.get("trend_sma", 200)),
            float(data["max_spend_quote"]),
            int(data.get("poll_seconds", 10)),
            str(data.get("strategy_mode", "classic")),
            str(data.get("forced_strategy", "")),
            int(data.get("max_open_orders", 6)),
            float(data.get("vol_gap_mult", 1.0)),
            float(data.get("tp_vol_mult", 1.0)),
            float(data.get("min_gap_pct", 0.003)),
            float(data.get("max_gap_pct", 0.06)),
            int(data.get("regime_hold_candles", 2)),
            int(data.get("regime_switch_ticks", 2)),
            float(data.get("regime_switch_threshold", 0.6)),
            float(data.get("max_total_exposure_pct", 0.50)),
            float(data.get("per_symbol_exposure_pct", 0.15)),
            float(data.get("min_free_cash_pct", 0.1)),
            int(data.get("max_concurrent_deals", 6)),
            float(data.get("spread_guard_pct", 0.003)),
            int(data.get("limit_timeout_sec", 8)),
            float(data.get("daily_loss_limit_pct", 0.06)),
            int(data.get("pause_hours", 6)),
            int(data.get("auto_restart", 1)),
            str(data.get("market_type", "crypto")),
            str(data.get("alpaca_mode", "paper")),
            float(data.get("max_drawdown_pct", 0.0)),
            str(data.get("trading_mode", "swing_trade")),
            int(data.get("intended_hold_days", 14)),
            int(data.get("conviction_level", 5)),
            int(data.get("auto_dip_buy", 0)),
            int(data.get("fundamental_exit_only", 0)),
            int(data.get("rebalance_enabled", 0)),
            float(data.get("grid_lower", 0.0)),
            float(data.get("grid_upper", 0.0)),
            int(data.get("grid_levels", 10)),
            int(bot_id),
        ),
    )
    con.commit()
    con.close()


def update_bots_by_type(bot_type: str, enabled: int) -> int:
    """Update enabled status for all bots with given bot_type. Returns count updated."""
    con = _conn()
    try:
        cur = con.execute(
            "UPDATE bots SET enabled=? WHERE LOWER(TRIM(COALESCE(bot_type,''))) = LOWER(TRIM(?))",
            (int(enabled), str(bot_type or "").strip()),
        )
        con.commit()
        return cur.rowcount
    finally:
        con.close()


def delete_bot(bot_id: int) -> None:
    """Delete bot and all related rows. Child tables first, then bots."""
    import logging
    logger = logging.getLogger(__name__)
    bid = int(bot_id)
    con = _conn()
    try:
        allowed_pairs = [
            ("order_events", "bot_id"),
            ("strategy_decisions", "bot_id"),
            ("regime_snapshots", "bot_id"),
            ("perf_metrics", "bot_id"),
            ("intelligence_decisions", "bot_id"),
            ("bot_logs", "bot_id"),
            ("deals", "bot_id"),
        ]
        for table, col in allowed_pairs:
            if table in _ALLOWED_TABLES and col in _ALLOWED_COLUMNS:
                try:
                    con.execute(f"DELETE FROM {table} WHERE {col}=?", (bid,))
                except sqlite3.OperationalError as e:
                    if "no such table" in str(e).lower():
                        logger.debug("delete_bot: skip %s (no such table)", table)
                    else:
                        raise
        con.execute("DELETE FROM bots WHERE id=?", (bid,))
        con.commit()
    finally:
        con.close()


def set_bot_enabled(bot_id: int, enabled: bool) -> None:
    con = _conn()
    con.execute(
        "UPDATE bots SET enabled=? WHERE id=?",
        (1 if enabled else 0, int(bot_id)),
    )
    con.commit()
    con.close()


def set_bot_running(bot_id: int, running: bool) -> None:
    con = _conn()
    con.execute(
        "UPDATE bots SET last_running=? WHERE id=?",
        (1 if running else 0, int(bot_id)),
    )
    con.commit()
    con.close()


def get_bot(bot_id: int) -> Optional[Dict[str, Any]]:
    con = _conn()
    row = con.execute("SELECT * FROM bots WHERE id=?", (int(bot_id),)).fetchone()
    con.close()
    return dict(row) if row else None


def list_bots() -> List[Dict[str, Any]]:
    con = _conn()
    rows = con.execute("SELECT * FROM bots ORDER BY id DESC").fetchall()
    con.close()
    return [dict(r) for r in rows]


def set_setting(key: str, value: Any) -> None:
    con = _conn()
    con.execute(
        "INSERT INTO settings(key, value) VALUES(?, ?) ON CONFLICT(key) DO UPDATE SET value=excluded.value",
        (str(key), str(value)),
    )
    con.commit()
    con.close()


def get_setting(key: str, default: Optional[str] = None) -> Optional[str]:
    con = _conn()
    row = con.execute("SELECT value FROM settings WHERE key=?", (str(key),)).fetchone()
    con.close()
    if not row:
        return default
    try:
        return str(row["value"])
    except Exception:
        return default


# =========================================================
# Autopilot config (Master Upgrade Part 4)
# =========================================================
def get_autopilot_config_row() -> Optional[Dict[str, Any]]:
    """Get autopilot_config row (id=1). Returns dict or None."""
    con = _conn()
    try:
        row = con.execute("SELECT * FROM autopilot_config WHERE id = 1").fetchone()
        return dict(row) if row else None
    except Exception:
        return None
    finally:
        con.close()


def save_autopilot_config(data: Dict[str, Any]) -> None:
    """Upsert autopilot_config row (id=1)."""
    con = _conn()
    try:
        con.execute(
            """
            INSERT INTO autopilot_config (id, enabled, total_capital_allocated, max_positions,
                asset_types, min_score_threshold, min_conviction_threshold,
                max_loss_per_day_pct, max_loss_per_week_pct,
                scan_frequency_hours, auto_create_bots, auto_start_bots,
                auto_close_underperformers, underperformer_threshold_score,
                rebalance_enabled, rebalance_frequency_days,
                notify_on_new_bot, notify_on_close, notify_daily_summary, last_scan, last_rebalance)
            VALUES (1, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(id) DO UPDATE SET
                enabled=excluded.enabled, total_capital_allocated=excluded.total_capital_allocated,
                max_positions=excluded.max_positions, asset_types=excluded.asset_types,
                min_score_threshold=excluded.min_score_threshold,
                min_conviction_threshold=excluded.min_conviction_threshold,
                max_loss_per_day_pct=excluded.max_loss_per_day_pct,
                max_loss_per_week_pct=excluded.max_loss_per_week_pct,
                scan_frequency_hours=excluded.scan_frequency_hours,
                auto_create_bots=excluded.auto_create_bots, auto_start_bots=excluded.auto_start_bots,
                auto_close_underperformers=excluded.auto_close_underperformers,
                underperformer_threshold_score=excluded.underperformer_threshold_score,
                rebalance_enabled=excluded.rebalance_enabled,
                rebalance_frequency_days=excluded.rebalance_frequency_days,
                notify_on_new_bot=excluded.notify_on_new_bot,
                notify_on_close=excluded.notify_on_close,
                notify_daily_summary=excluded.notify_daily_summary,
                last_scan=excluded.last_scan, last_rebalance=excluded.last_rebalance
            """,
            (
                int(data.get("enabled", 0)),
                float(data.get("total_capital_allocated", 10000)),
                int(data.get("max_positions", 10)),
                str(data.get("asset_types", "both")),
                int(data.get("min_score_threshold", 75)),
                int(data.get("min_conviction_threshold", 5)),
                float(data.get("max_loss_per_day_pct", 5.0)),
                float(data.get("max_loss_per_week_pct", 10.0)),
                int(data.get("scan_frequency_hours", 4)),
                int(data.get("auto_create_bots", 1)),
                int(data.get("auto_start_bots", 1)),
                int(data.get("auto_close_underperformers", 1)),
                int(data.get("underperformer_threshold_score", 60)),
                int(data.get("rebalance_enabled", 1)),
                int(data.get("rebalance_frequency_days", 7)),
                int(data.get("notify_on_new_bot", 1)),
                int(data.get("notify_on_close", 1)),
                int(data.get("notify_daily_summary", 1)),
                data.get("last_scan"),
                data.get("last_rebalance"),
            ),
        )
        con.commit()
    finally:
        con.close()


# =========================================================
# Deals
# =========================================================
def open_deal(bot_id: int, symbol: str, state: str = "OPEN", opened_at: Optional[int] = None) -> int:
    con = _conn()
    cur = con.cursor()
    _opened = int(opened_at) if opened_at is not None else now_ts()
    cur.execute(
        "INSERT INTO deals(bot_id, state, opened_at, symbol) VALUES (?,?,?,?)",
        (int(bot_id), str(state), _opened, str(symbol)),
    )
    con.commit()
    deal_id = int(cur.lastrowid)
    con.close()
    return deal_id


def update_open_deal_entry(
    deal_id: int,
    entry_avg: float,
    base_amount: float,
    safety_count: int = 0,
) -> None:
    """
    Update an OPEN deal with real entry price and size.
    Called after a buy executes so the DB reflects the
    actual position. Without this, entry_avg stays NULL
    on open deals even when a real position exists.
    """
    if not deal_id or not entry_avg or entry_avg <= 0:
        return
    con = _conn()
    try:
        con.execute(
            """UPDATE deals SET
                entry_avg = ?,
                base_amount = ?,
                safety_count = ?
            WHERE id = ? AND state = 'OPEN'""",
            (
                float(entry_avg),
                float(base_amount) if base_amount else None,
                int(safety_count),
                int(deal_id),
            ),
        )
        con.commit()
    finally:
        con.close()


def close_deal(
    deal_id: int,
    entry_avg: float,
    exit_avg: float,
    base_amount: float,
    realized_pnl_quote: float,
    entry_regime: Optional[str] = None,
    exit_regime: Optional[str] = None,
    entry_strategy: Optional[str] = None,
    exit_strategy: Optional[str] = None,
    mae: Optional[float] = None,
    mfe: Optional[float] = None,
    hold_sec: Optional[int] = None,
    safety_count: Optional[int] = None,
    entry_avg_estimated: bool = False,
) -> None:
    con = _conn()
    # Fetch bot_id and opened_at before update (for recommendation_performance)
    row = con.execute(
        "SELECT bot_id, opened_at FROM deals WHERE id=?",
        (int(deal_id),),
    ).fetchone()
    closed_ts = now_ts()
    opened_ts = int(row["opened_at"] or 0) if row else 0
    bot_id_val = int(row["bot_id"]) if row else None

    realized_pnl_pct_val: Optional[float] = None
    try:
        ea = float(entry_avg) if entry_avg is not None else None
        ex = float(exit_avg) if exit_avg is not None else None
        ba = float(base_amount) if base_amount is not None else None
        rq = float(realized_pnl_quote) if realized_pnl_quote is not None else None
        if (
            ea is not None and ba is not None and rq is not None
            and ea > 0 and ba > 0 and math.isfinite(ea) and math.isfinite(ba) and math.isfinite(rq)
        ):
            denom = ea * ba
            if denom != 0:
                realized_pnl_pct_val = (rq / denom) * 100.0
        if realized_pnl_pct_val is None and ea is not None and ex is not None and ea != 0 and math.isfinite(ea) and math.isfinite(ex):
            realized_pnl_pct_val = (ex - ea) / ea * 100.0
    except (TypeError, ValueError):
        realized_pnl_pct_val = None

    con.execute(
        """
        UPDATE deals SET
            state=?,
            closed_at=?,
            entry_avg=?,
            exit_avg=?,
            base_amount=?,
            realized_pnl_quote=?,
            realized_pnl_pct=?,
            entry_avg_estimated=?,
            entry_regime=?,
            exit_regime=?,
            entry_strategy=?,
            exit_strategy=?,
            mae=?,
            mfe=?,
            hold_sec=?,
            safety_count=?
        WHERE id=?
        """,
        (
            "CLOSED",
            closed_ts,
            float(entry_avg) if entry_avg is not None else None,
            float(exit_avg) if exit_avg is not None else None,
            float(base_amount) if base_amount is not None else None,
            float(realized_pnl_quote) if realized_pnl_quote is not None else None,
            float(realized_pnl_pct_val) if realized_pnl_pct_val is not None else None,
            1 if entry_avg_estimated else 0,
            str(entry_regime) if entry_regime is not None else None,
            str(exit_regime) if exit_regime is not None else None,
            str(entry_strategy) if entry_strategy is not None else None,
            str(exit_strategy) if exit_strategy is not None else None,
            float(mae) if mae is not None else None,
            float(mfe) if mfe is not None else None,
            int(hold_sec) if hold_sec is not None else None,
            int(safety_count) if safety_count is not None else None,
            int(deal_id),
        ),
    )
    con.commit()

    # Record trade feedback for ML learning
    try:
        profitable_flag = 1 if (realized_pnl_quote or 0) > 0 else 0
        # Fetch symbol from deal
        deal_row = con.execute("SELECT symbol FROM deals WHERE id=?", (int(deal_id),)).fetchone()
        deal_symbol = deal_row["symbol"] if deal_row else ""
        import json as _json
        features = _json.dumps({
            "entry_avg": float(entry_avg) if entry_avg else 0,
            "exit_avg": float(exit_avg) if exit_avg else 0,
            "pnl": float(realized_pnl_quote) if realized_pnl_quote else 0,
            "exit_strategy": str(exit_strategy or ""),
            "entry_regime": str(entry_regime or ""),
            "hold_sec": int(hold_sec) if hold_sec else 0,
        })
        record_trade_feedback(deal_symbol, features, profitable_flag)
    except Exception as fb_err:
        logger.warning("Failed to record trade feedback on deal close: %s", fb_err)

    # Record outcome for recommendation performance tracking (if bot was created from recommendation)
    if bot_id_val and entry_avg is not None and exit_avg is not None and realized_pnl_quote is not None:
        try:
            _record_recommendation_outcome(
                con, bot_id_val, deal_id,
                float(entry_avg), float(exit_avg), float(realized_pnl_quote),
                closed_ts, opened_ts,
            )
            con.commit()
        except Exception:
            pass  # Do not fail deal close if performance tracking fails
    con.close()


def manual_close_deal_and_journal(
    deal_id: int,
    bot_id: int,
    entry_avg: float,
    exit_avg: float,
    base_amount: float,
    realized_pnl_quote: float,
    entry_strategy: Optional[str] = None,
    exit_strategy: Optional[str] = None,
    hold_sec: Optional[int] = None,
    safety_count: Optional[int] = None,
    journal_exit_reason: str = "",
    entry_regime: Optional[str] = None,
    exit_regime: Optional[str] = None,
    mae: Optional[float] = None,
    mfe: Optional[float] = None,
    entry_avg_estimated: bool = False,
) -> Dict[str, Any]:
    """
    Manual close from API: one dedicated SQLite connection + single commit so we do not
    race the per-thread _conn() cache used elsewhere. Caller must serialize with BotManager.bot_db_lock.
    """
    con = _make_real_conn()
    try:
        row = con.execute("SELECT * FROM deals WHERE id=?", (int(deal_id),)).fetchone()
        if not row:
            raise ValueError("Deal not found")
        drow = dict(row)
        if int(drow.get("bot_id") or 0) != int(bot_id):
            raise ValueError("Deal does not belong to this bot")
        st = str(drow.get("state") or "").upper()
        if st in ("CLOSED", "CANCELLED"):
            raise ValueError(f"Deal already {st}")

        closed_ts = now_ts()
        opened_ts = int(drow.get("opened_at") or 0)

        realized_pnl_pct_val: Optional[float] = None
        try:
            ea = float(entry_avg) if entry_avg is not None else None
            ex = float(exit_avg) if exit_avg is not None else None
            ba = float(base_amount) if base_amount is not None else None
            rq = float(realized_pnl_quote) if realized_pnl_quote is not None else None
            if (
                ea is not None and ba is not None and rq is not None
                and ea > 0 and ba > 0 and math.isfinite(ea) and math.isfinite(ba) and math.isfinite(rq)
            ):
                denom = ea * ba
                if denom != 0:
                    realized_pnl_pct_val = (rq / denom) * 100.0
            if realized_pnl_pct_val is None and ea is not None and ex is not None and ea != 0 and math.isfinite(ea) and math.isfinite(ex):
                realized_pnl_pct_val = (ex - ea) / ea * 100.0
        except (TypeError, ValueError):
            realized_pnl_pct_val = None

        # Race-safe close: filter on state='OPEN' so concurrent callers see
        # rowcount=0 and raise ValueError below. Without this predicate
        # 10 threads can all read the row as OPEN, then all overwrite it.
        cur = con.execute(
            """
            UPDATE deals SET
                state=?,
                closed_at=?,
                entry_avg=?,
                exit_avg=?,
                base_amount=?,
                realized_pnl_quote=?,
                realized_pnl_pct=?,
                entry_avg_estimated=?,
                entry_regime=?,
                exit_regime=?,
                entry_strategy=?,
                exit_strategy=?,
                mae=?,
                mfe=?,
                hold_sec=?,
                safety_count=?
            WHERE id=? AND bot_id=? AND state='OPEN'
            """,
            (
                "CLOSED",
                closed_ts,
                float(entry_avg) if entry_avg is not None else None,
                float(exit_avg) if exit_avg is not None else None,
                float(base_amount) if base_amount is not None else None,
                float(realized_pnl_quote) if realized_pnl_quote is not None else None,
                float(realized_pnl_pct_val) if realized_pnl_pct_val is not None else None,
                1 if entry_avg_estimated else 0,
                str(entry_regime) if entry_regime is not None else None,
                str(exit_regime) if exit_regime is not None else None,
                str(entry_strategy) if entry_strategy is not None else None,
                str(exit_strategy) if exit_strategy is not None else None,
                float(mae) if mae is not None else None,
                float(mfe) if mfe is not None else None,
                int(hold_sec) if hold_sec is not None else None,
                int(safety_count) if safety_count is not None else None,
                int(deal_id),
                int(bot_id),
            ),
        )
        if int(cur.rowcount or 0) == 0:
            # Lost the race — another writer closed it first. Roll back so
            # we do not double-write the trade_journal / trade_feedback below.
            try:
                con.rollback()
            except Exception:
                pass
            raise ValueError(f"Deal {deal_id} not open or not found")

        jr = con.execute("SELECT * FROM trade_journal WHERE deal_id=?", (int(deal_id),)).fetchone()
        if jr:
            jdict = dict(jr)
            er = jdict.get("entry_reason") or ""
            xr = journal_exit_reason if journal_exit_reason else (jdict.get("exit_reason") or "")
            ll = jdict.get("lessons_learned") or ""
            sc = jdict.get("screenshot_data") or ""
            con.execute(
                """UPDATE trade_journal SET entry_reason=?, exit_reason=?, lessons_learned=?, screenshot_data=?, updated_at=? WHERE deal_id=?""",
                (er, xr, ll, sc, closed_ts, int(deal_id)),
            )
        else:
            con.execute(
                """
                INSERT INTO trade_journal(deal_id, entry_reason, exit_reason, lessons_learned, screenshot_data, updated_at)
                VALUES (?,?,?,?,?,?)
                """,
                (int(deal_id), "", journal_exit_reason or "", "", "", closed_ts),
            )

        try:
            profitable_flag = 1 if (realized_pnl_quote or 0) > 0 else 0
            sym_row = con.execute("SELECT symbol FROM deals WHERE id=?", (int(deal_id),)).fetchone()
            deal_symbol = sym_row["symbol"] if sym_row else ""
            features = json.dumps({
                "entry_avg": float(entry_avg) if entry_avg else 0,
                "exit_avg": float(exit_avg) if exit_avg else 0,
                "pnl": float(realized_pnl_quote) if realized_pnl_quote else 0,
                "exit_strategy": str(exit_strategy or ""),
                "entry_regime": str(entry_regime or ""),
                "hold_sec": int(hold_sec) if hold_sec else 0,
            })
            con.execute(
                "INSERT INTO trade_feedback(symbol, timestamp, features_json, profitable) VALUES (?, ?, ?, ?)",
                (str(deal_symbol), time.time(), features, int(profitable_flag)),
            )
        except Exception as fb_err:
            logger.warning("manual_close_deal_and_journal: trade_feedback failed: %s", fb_err)

        if entry_avg is not None and exit_avg is not None and realized_pnl_quote is not None:
            try:
                _record_recommendation_outcome(
                    con, int(bot_id), int(deal_id),
                    float(entry_avg), float(exit_avg), float(realized_pnl_quote),
                    closed_ts, opened_ts,
                )
            except Exception:
                pass

        con.commit()
        rp = float(realized_pnl_quote)
        rp_pct = ((float(exit_avg) - float(entry_avg)) / float(entry_avg)) * 100.0 if entry_avg and float(entry_avg) > 0 else 0.0
        return {
            "ok": True,
            "deal_id": int(deal_id),
            "realized_pnl": rp,
            "realized_pnl_quote": rp,
            "realized_pnl_pct": float(rp_pct),
            "closed_at": closed_ts,
        }
    except Exception:
        try:
            con.rollback()
        except Exception:
            pass
        raise
    finally:
        try:
            con.close()
        except Exception:
            pass


def _safe_float_db(val: Any, default: float = 0.0) -> float:
    try:
        v = float(val)
        return v if math.isfinite(v) else default
    except (TypeError, ValueError):
        return default


def _repair_entry_from_trades_window(
    kc: Any,
    symbol: str,
    opened_sec: int,
    closed_sec: int,
    extend_back_sec: int,
    trade_limit: int = 1000,
) -> Optional[float]:
    """Average buy price from exchange trades between deal open/close (with optional extended fetch window)."""
    try:
        trades = kc.fetch_my_trades(symbol, limit=int(trade_limit))
    except Exception:
        return None
    opened_ms = int(opened_sec) * 1000
    closed_ms = int(closed_sec) * 1000
    since_ms = max(0, int(opened_sec) - int(extend_back_sec)) * 1000
    buy_amt = 0.0
    buy_cost = 0.0
    for t in trades or []:
        ts = t.get("timestamp")
        if ts is None:
            continue
        try:
            ts_i = int(ts)
        except Exception:
            continue
        if ts_i < since_ms or ts_i < opened_ms or ts_i > closed_ms:
            continue
        side = (t.get("side") or "").lower()
        if side != "buy":
            continue
        amt = _safe_float_db(t.get("amount"), 0.0)
        price = _safe_float_db(t.get("price"), 0.0)
        if amt <= 0 or price <= 0:
            continue
        buy_amt += amt
        buy_cost += amt * price
    if buy_amt <= 0:
        return None
    return buy_cost / buy_amt


def _repair_entry_from_ohlcv(kc: Any, symbol: str, opened_sec: int) -> Optional[float]:
    """Closest 1h candle close to deal open time."""
    try:
        opened_ms = int(opened_sec) * 1000
        since_ms = opened_ms - 48 * 3600 * 1000
        until_ms = opened_ms + 3600 * 1000
        candles: List[Any] = []
        if hasattr(kc, "fetch_ohlcv_range"):
            candles = kc.fetch_ohlcv_range(symbol, "1h", since_ms, until_ms, limit=120) or []
        if not candles:
            candles = kc.fetch_ohlcv(symbol, "1h", limit=120) or []
        if not candles:
            return None
        best: Optional[float] = None
        best_d: Optional[int] = None
        for c in candles:
            if not c or len(c) < 5:
                continue
            cm = int(c[0])
            d = abs(cm - opened_ms)
            if best_d is None or d < best_d:
                best_d = d
                best = _safe_float_db(c[4], 0.0)
        if best is not None and best > 0:
            return float(best)
    except Exception as ex:
        logger.debug("repair_entry_from_ohlcv: %s", ex)
    return None


def get_deal_buy_avg_from_order_events(deal_id: int) -> Optional[Tuple[float, float]]:
    """
    Recover average buy price from local order_events for a deal window.
    Returns (entry_avg, total_buy_qty) or None.
    """
    con = _conn()
    try:
        deal = con.execute(
            "SELECT bot_id, symbol, opened_at, closed_at FROM deals WHERE id=?",
            (int(deal_id),),
        ).fetchone()
        if not deal:
            return None
        bot_id = int(deal["bot_id"] or 0)
        sym = str(deal["symbol"] or "").strip()
        opened = int(deal["opened_at"] or 0)
        closed = int(deal["closed_at"] or 0) or now_ts()
        if bot_id <= 0 or not sym or opened <= 0:
            return None
        row = con.execute(
            """
            SELECT COALESCE(SUM(amount), 0) AS amt, COALESCE(SUM(amount * price), 0) AS cost
            FROM order_events
            WHERE bot_id = ? AND symbol = ? AND LOWER(side) = 'buy'
              AND price IS NOT NULL AND amount IS NOT NULL
              AND ts >= ? AND ts <= ?
            """,
            (bot_id, sym, opened, closed),
        ).fetchone()
        amt = _safe_float_db(row["amt"] if row else 0, 0.0)
        cost = _safe_float_db(row["cost"] if row else 0, 0.0)
        if amt <= 0 or cost <= 0:
            return None
        return (cost / amt, amt)
    except Exception as ex:
        logger.debug("get_deal_buy_avg_from_order_events: %s", ex)
        return None
    finally:
        con.close()


def repair_closed_deals_missing_entry(kc: Any = None) -> int:
    """
    CLOSED deals only: entry_avg IS NULL OR entry_avg = 0.
    Order of recovery: order_events (local) → exchange trades → OHLCV → exit_avg fallback.
    """
    con = _conn()
    try:
        rows = con.execute(
            """
            SELECT id, bot_id, symbol, opened_at, closed_at, exit_avg, base_amount
            FROM deals
            WHERE state='CLOSED' AND (entry_avg IS NULL OR entry_avg = 0)
            """
        ).fetchall()
    finally:
        con.close()

    if not rows:
        logger.info("repair_closed_deals_missing_entry: repaired 0 deals (none needed)")
        return 0

    repaired = 0
    now_sec = int(time.time())
    for r in rows:
        did = int(r["id"])
        sym = str(r["symbol"] or "").strip()
        opened = int(r["opened_at"] or 0)
        closed = int(r["closed_at"] or 0)
        exit_avg = r["exit_avg"]
        base_amt = r["base_amount"]

        if not sym or opened <= 0:
            continue
        if closed <= 0:
            closed = now_sec

        entry_avg: Optional[float] = None
        ohlcv_estimated = False
        data_source: Optional[str] = None

        ob = get_deal_buy_avg_from_order_events(did)
        if ob and ob[0] > 0:
            entry_avg = float(ob[0])
            data_source = "repaired"

        if (entry_avg is None or entry_avg <= 0) and kc is not None:
            extend_back = min(max(0, closed - opened) * 2, 7 * 86400)
            entry_avg = _repair_entry_from_trades_window(
                kc, sym, opened, closed, extend_back_sec=extend_back, trade_limit=1000,
            )
            if entry_avg and entry_avg > 0:
                data_source = data_source or "repaired"

        if (entry_avg is None or entry_avg <= 0) and kc is not None:
            entry_avg = _repair_entry_from_ohlcv(kc, sym, opened)
            if entry_avg is not None and entry_avg > 0:
                ohlcv_estimated = True
                data_source = data_source or "repaired"

        if entry_avg is None or entry_avg <= 0:
            ex = _safe_float_db(exit_avg, 0.0)
            if ex > 0:
                entry_avg = ex
                ohlcv_estimated = True
                data_source = data_source or "repaired"
                logger.warning(
                    "repair_closed_deals_missing_entry: deal %d %s entry unrecoverable; using exit_avg as conservative fallback",
                    did, sym,
                )
            else:
                logger.warning(
                    "repair_closed_deals_missing_entry: deal %d %s skipped (no entry and no exit)",
                    did, sym,
                )
                continue

        ex = _safe_float_db(exit_avg, 0.0)
        ba = _safe_float_db(base_amt, 0.0)
        rpnl: Optional[float] = None
        rpct: Optional[float] = None
        if ex > 0 and ba > 0:
            rpnl = (ex - float(entry_avg)) * ba
            ea = float(entry_avg)
            if ea > 0 and ba > 0 and rpnl is not None:
                rpct = (float(rpnl) / (ea * ba)) * 100.0

        con = _conn()
        try:
            cur = con.cursor()
            cur.execute(
                """
                UPDATE deals SET
                    entry_avg = ?,
                    realized_pnl_quote = ?,
                    realized_pnl_pct = ?,
                    entry_avg_estimated = ?,
                    data_source = COALESCE(?, data_source)
                WHERE id = ? AND state = 'CLOSED'
                  AND (entry_avg IS NULL OR entry_avg = 0)
                """,
                (
                    float(entry_avg),
                    float(rpnl) if rpnl is not None else None,
                    float(rpct) if rpct is not None else None,
                    1 if ohlcv_estimated else 0,
                    data_source,
                    did,
                ),
            )
            if cur.rowcount and int(cur.rowcount) > 0:
                repaired += 1
            con.commit()
        finally:
            con.close()

    logger.info("repair_closed_deals_missing_entry: repaired %d deals", repaired)
    return repaired


def repair_null_entry_avg_deals(kc: Any = None) -> int:
    """Alias for repair_closed_deals_missing_entry (closed deals only; see order_events + exchange repair)."""
    return repair_closed_deals_missing_entry(kc)


def get_symbols_with_open_deals() -> List[str]:
    """Return distinct symbols from all open deals."""
    con = _conn()
    rows = con.execute(
        """
        SELECT DISTINCT symbol FROM deals
        WHERE state='OPEN' AND symbol IS NOT NULL AND symbol != ''
        """
    ).fetchall()
    con.close()
    return [str(r[0]) for r in rows]


def get_symbols_with_open_deals_excluding(bot_id: int) -> List[str]:
    """Return distinct symbols from open deals of other bots (exclude bot_id)."""
    con = _conn()
    rows = con.execute(
        """
        SELECT DISTINCT symbol FROM deals
        WHERE state='OPEN' AND bot_id != ? AND symbol IS NOT NULL AND symbol != ''
        """,
        (int(bot_id),),
    ).fetchall()
    con.close()
    return [str(r[0]) for r in rows]


def latest_open_deal(bot_id: int) -> Optional[Dict[str, Any]]:
    con = _conn()
    row = con.execute(
        """
        SELECT * FROM deals
        WHERE bot_id=? AND state='OPEN'
        ORDER BY opened_at DESC
        LIMIT 1
        """,
        (int(bot_id),),
    ).fetchone()
    con.close()
    return dict(row) if row else None


def count_closed_deals() -> int:
    """Count CLOSED deals (live track record — used for Explore honesty badges)."""
    con = _conn()
    try:
        row = con.execute("SELECT COUNT(*) AS c FROM deals WHERE state='CLOSED'").fetchone()
        return int(row["c"]) if row else 0
    finally:
        con.close()


def find_stale_ghost_deals(max_age_sec: int = 7200) -> List[Dict[str, Any]]:
    """Find deals that are OPEN with no entry (entry_avg IS NULL) older than max_age_sec."""
    con = _conn()
    cutoff = now_ts() - max_age_sec
    rows = con.execute(
        """
        SELECT id, bot_id, symbol, opened_at, state
        FROM deals
        WHERE state='OPEN' AND entry_avg IS NULL AND opened_at < ?
        ORDER BY opened_at ASC
        """,
        (cutoff,),
    ).fetchall()
    con.close()
    return [dict(r) for r in rows]


def cancel_ghost_deal(deal_id: int) -> None:
    """Cancel a ghost deal (OPEN with no entry). Sets state=CANCELLED."""
    con = _conn()
    con.execute(
        "UPDATE deals SET state='CANCELLED', closed_at=? WHERE id=? AND state='OPEN'",
        (now_ts(), int(deal_id)),
    )
    con.commit()
    con.close()


def list_deals(bot_id: int, limit: int = 50) -> List[Dict[str, Any]]:
    con = _conn()
    rows = con.execute(
        """
        SELECT
            id, state, opened_at, closed_at, symbol,
            entry_avg, exit_avg, base_amount, realized_pnl_quote, realized_pnl_pct,
            entry_avg_estimated
        FROM deals
        WHERE bot_id=?
        ORDER BY opened_at DESC
        LIMIT ?
        """,
        (int(bot_id), int(limit)),
    ).fetchall()
    con.close()
    return [dict(r) for r in rows]


def list_all_deals(state: Optional[str] = None, limit: int = 200) -> List[Dict[str, Any]]:
    con = _conn()
    if state:
        rows = con.execute(
            """
            SELECT
                id, bot_id, state, opened_at, closed_at, symbol,
                entry_avg, exit_avg, base_amount, realized_pnl_quote, realized_pnl_pct,
                data_source
            FROM deals
            WHERE state=?
            ORDER BY opened_at DESC
            LIMIT ?
            """,
            (str(state), int(limit)),
        ).fetchall()
    else:
        rows = con.execute(
            """
            SELECT
                id, bot_id, state, opened_at, closed_at, symbol,
                entry_avg, exit_avg, base_amount, realized_pnl_quote, realized_pnl_pct,
                data_source
            FROM deals
            ORDER BY opened_at DESC
            LIMIT ?
            """,
            (int(limit),),
        ).fetchall()
    con.close()
    return [dict(r) for r in rows]


def list_closed_deals_for_journal(since_ts: Optional[int] = None, limit: int = 200) -> List[Dict[str, Any]]:
    """List closed deals with full columns for journal. since_ts = only deals closed after."""
    con = _conn()
    if since_ts:
        rows = con.execute(
            """
            SELECT id, bot_id, state, opened_at, closed_at, symbol,
                   entry_avg, exit_avg, base_amount, realized_pnl_quote,
                   entry_regime, entry_strategy, exit_regime, exit_strategy
            FROM deals
            WHERE state='CLOSED' AND closed_at IS NOT NULL AND closed_at >= ?
            ORDER BY closed_at DESC
            LIMIT ?
            """,
            (int(since_ts), int(limit)),
        ).fetchall()
    else:
        rows = con.execute(
            """
            SELECT id, bot_id, state, opened_at, closed_at, symbol,
                   entry_avg, exit_avg, base_amount, realized_pnl_quote,
                   entry_regime, entry_strategy, exit_regime, exit_strategy
            FROM deals
            WHERE state='CLOSED' AND closed_at IS NOT NULL
            ORDER BY closed_at DESC
            LIMIT ?
            """,
            (int(limit),),
        ).fetchall()
    con.close()
    return [dict(r) for r in rows]


def get_deal(deal_id: int, full: bool = False) -> Optional[Dict[str, Any]]:
    """Get deal. full=True includes entry_regime, entry_strategy, exit_regime, exit_strategy."""
    con = _conn()
    cols = "id, bot_id, state, opened_at, closed_at, symbol, entry_avg, exit_avg, base_amount, realized_pnl_quote"
    if full:
        cols += ", entry_regime, entry_strategy, exit_regime, exit_strategy"
    row = con.execute(f"SELECT {cols} FROM deals WHERE id=?", (int(deal_id),)).fetchone()
    con.close()
    return dict(row) if row else None


def bot_pnl_series(bot_id: int, limit: int = 500) -> List[Dict[str, Any]]:
    con = _conn()
    rows = con.execute(
        """
        SELECT closed_at, realized_pnl_quote
        FROM deals
        WHERE bot_id=? AND state='CLOSED' AND closed_at IS NOT NULL
        ORDER BY closed_at ASC
        LIMIT ?
        """,
        (int(bot_id), int(limit)),
    ).fetchall()
    con.close()
    out: List[Dict[str, Any]] = []
    cum = 0.0
    for r in rows:
        try:
            pnl = float(r["realized_pnl_quote"] or 0.0)
            ts = int(r["closed_at"] or 0)
            cum += pnl
            out.append({"time": ts, "value": float(cum)})
        except Exception:
            continue
    return out


def bot_drawdown_series(bot_id: int, limit: int = 500) -> List[Dict[str, Any]]:
    con = _conn()
    rows = con.execute(
        """
        SELECT closed_at, realized_pnl_quote
        FROM deals
        WHERE bot_id=? AND state='CLOSED' AND closed_at IS NOT NULL
        ORDER BY closed_at ASC
        LIMIT ?
        """,
        (int(bot_id), int(limit)),
    ).fetchall()
    con.close()
    out: List[Dict[str, Any]] = []
    cum = 0.0
    peak = 0.0
    for r in rows:
        try:
            pnl = float(r["realized_pnl_quote"] or 0.0)
            ts = int(r["closed_at"] or 0)
            cum += pnl
            if cum > peak:
                peak = cum
            drawdown = max(0.0, peak - cum)
            # negative value for chart readability
            out.append({"time": ts, "value": float(-drawdown)})
        except Exception:
            continue
    return out


def bot_performance_stats(bot_id: int) -> Dict[str, Any]:
    con = _conn()
    rows = con.execute(
        """
        SELECT opened_at, closed_at, realized_pnl_quote, entry_avg, base_amount
        FROM deals
        WHERE bot_id=? AND state='CLOSED' AND closed_at IS NOT NULL
        ORDER BY closed_at ASC
        """,
        (int(bot_id),),
    ).fetchall()
    con.close()
    pnls: List[float] = []
    pnl_pcts: List[float] = []
    durations: List[int] = []
    for r in rows:
        try:
            pnl = float(r["realized_pnl_quote"] or 0.0)
            entry_avg = float(r["entry_avg"] or 0.0)
            base_amount = float(r["base_amount"] or 0.0)
            opened = int(r["opened_at"] or 0)
            closed = int(r["closed_at"] or 0)
            if closed > 0 and opened > 0:
                durations.append(max(0, closed - opened))
            pnls.append(pnl)
            cost_basis = entry_avg * base_amount if entry_avg > 0 and base_amount > 0 else 0.0
            if cost_basis > 0:
                pnl_pcts.append(pnl / cost_basis)
        except Exception:
            continue
    wins = sum(1 for x in pnls if x > 0)
    losses = sum(1 for x in pnls if x < 0)
    total = len(pnls)
    win_rate = (wins / total) if total else 0.0
    avg_duration = (sum(durations) / len(durations)) if durations else 0.0
    win_pcts = [p for p in pnl_pcts if p > 0]
    loss_pcts = [abs(p) for p in pnl_pcts if p < 0]
    avg_profit_pct = (sum(win_pcts) / len(win_pcts)) if win_pcts else 0.02
    avg_loss_pct = (sum(loss_pcts) / len(loss_pcts)) if loss_pcts else 0.01
    return {
        "wins": int(wins),
        "losses": int(losses),
        "total": int(total),
        "win_rate": float(win_rate),
        "avg_duration_sec": float(avg_duration),
        "avg_profit_pct": float(avg_profit_pct),
        "avg_loss_pct": float(avg_loss_pct),
        "total_trades": int(total),
        "winning_trades": int(wins),
    }


def get_global_consecutive_losses(n: int = 10) -> int:
    """
    Last N closed deals across ALL bots. Returns negative streak: -3 means 3 consecutive losses.
    Used for 3-loss circuit breaker (pause autopilot 24h).
    """
    con = _conn()
    rows = con.execute(
        """
        SELECT realized_pnl_quote
        FROM deals
        WHERE state='CLOSED' AND closed_at IS NOT NULL
        ORDER BY closed_at DESC
        LIMIT ?
        """,
        (int(n),),
    ).fetchall()
    con.close()
    streak = 0
    for r in rows:
        pnl = float(r["realized_pnl_quote"] or 0)
        o = 1 if pnl > 0 else -1
        if streak == 0:
            streak = o
        elif (streak > 0 and o > 0) or (streak < 0 and o < 0):
            streak += o
        else:
            break
    return streak


def get_rolling_trade_stats_last_n(n: int = 30) -> Dict[str, Any]:
    """
    Rolling stats from last N closed deals (all bots). For Kelly position sizing.
    Returns: total_trades, winning_trades, avg_profit_pct, avg_loss_pct, win_rate
    """
    con = _conn()
    rows = con.execute(
        """
        SELECT realized_pnl_quote, entry_avg, base_amount
        FROM deals
        WHERE state='CLOSED' AND closed_at IS NOT NULL
        ORDER BY closed_at DESC
        LIMIT ?
        """,
        (int(n),),
    ).fetchall()
    con.close()
    pnls: List[float] = []
    pnl_pcts: List[float] = []
    for r in rows:
        try:
            pnl = float(r["realized_pnl_quote"] or 0)
            entry = float(r["entry_avg"] or 0)
            base = float(r["base_amount"] or 0)
            pnls.append(pnl)
            cost = entry * base if entry > 0 and base > 0 else 0
            if cost > 0:
                pnl_pcts.append(pnl / cost)
        except Exception:
            continue
    wins = sum(1 for x in pnls if x > 0)
    win_pcts = [p for p in pnl_pcts if p > 0]
    loss_pcts = [abs(p) for p in pnl_pcts if p < 0]
    return {
        "total_trades": len(pnls),
        "winning_trades": wins,
        "win_rate": wins / len(pnls) if pnls else 0.5,
        "avg_profit_pct": sum(win_pcts) / len(win_pcts) if win_pcts else 0.05,
        "avg_loss_pct": sum(loss_pcts) / len(loss_pcts) if loss_pcts else 0.03,
    }


def get_bot_recent_streak(bot_id: int, n: int = 5) -> int:
    """
    Last N deal outcomes: positive = wins, negative = losses.
    Returns streak: 3 means 3 consecutive wins, -3 means 3 consecutive losses.
    """
    con = _conn()
    rows = con.execute(
        """
        SELECT realized_pnl_quote
        FROM deals
        WHERE bot_id=? AND state='CLOSED' AND closed_at IS NOT NULL
        ORDER BY closed_at DESC
        LIMIT ?
        """,
        (int(bot_id), int(n)),
    ).fetchall()
    con.close()
    outcomes = []
    for r in rows:
        pnl = float(r["realized_pnl_quote"] or 0)
        outcomes.append(1 if pnl > 0 else -1)
    streak = 0
    for o in outcomes:
        if streak == 0:
            streak = o
        elif (streak > 0 and o > 0) or (streak < 0 and o < 0):
            streak += o
        else:
            break
    return streak


def bot_deal_stats(bot_id: int) -> Dict[str, Any]:
    """
    Aggregate counts + realized PnL for a bot.
    """
    con = _conn()
    row = con.execute(
        """
        SELECT
            SUM(CASE WHEN state='OPEN' THEN 1 ELSE 0 END) AS open_count,
            SUM(CASE WHEN state='CLOSED' THEN 1 ELSE 0 END) AS closed_count,
            COALESCE(SUM(CASE WHEN state='CLOSED' THEN realized_pnl_quote ELSE 0 END), 0) AS realized_total
        FROM deals
        WHERE bot_id=?
        """,
        (int(bot_id),),
    ).fetchone()
    con.close()
    if not row:
        return {"open_count": 0, "closed_count": 0, "realized_total": 0.0}
    return {
        "open_count": int(row["open_count"] or 0),
        "closed_count": int(row["closed_count"] or 0),
        "realized_total": float(row["realized_total"] or 0.0),
    }


def all_deal_stats() -> Dict[str, Any]:
    """
    Aggregate counts + realized PnL across all bots.
    """
    con = _conn()
    row = con.execute(
        """
        SELECT
            SUM(CASE WHEN state='OPEN' THEN 1 ELSE 0 END) AS open_count,
            SUM(CASE WHEN state='CLOSED' THEN 1 ELSE 0 END) AS closed_count,
            COALESCE(SUM(CASE WHEN state='CLOSED' THEN realized_pnl_quote ELSE 0 END), 0) AS realized_total
        FROM deals
        """
    ).fetchone()
    con.close()
    if not row:
        return {"open_count": 0, "closed_count": 0, "realized_total": 0.0}
    return {
        "open_count": int(row["open_count"] or 0),
        "closed_count": int(row["closed_count"] or 0),
        "realized_total": float(row["realized_total"] or 0.0),
    }


# =========================================================
# PnL summary (stable schema used by app + bot logic)
# =========================================================
def pnl_summary(since_ts: int) -> Dict[str, Any]:
    """
    Summarize realized PnL for CLOSED deals closed_at >= since_ts.
    Returns:
      - realized (float)
      - deals_closed (int)
      - wins (int)
      - losses (int)
      - since_ts (int)
    """
    con = _conn()
    rows = con.execute(
        """
        SELECT realized_pnl_quote
        FROM deals
        WHERE state='CLOSED' AND closed_at IS NOT NULL AND closed_at >= ?
        """,
        (int(since_ts),),
    ).fetchall()
    con.close()

    pnls: List[float] = []
    for r in rows:
        try:
            v = r["realized_pnl_quote"]
            if v is None:
                continue
            pnls.append(float(v))
        except Exception:
            continue

    realized = float(sum(pnls)) if pnls else 0.0
    wins = sum(1 for x in pnls if x > 0)
    losses = sum(1 for x in pnls if x < 0)

    return {
        "since_ts": int(since_ts),
        "realized": float(realized),
        "deals_closed": int(len(pnls)),
        "wins": int(wins),
        "losses": int(losses),
    }


# =========================================================
# Strategy/regime persistence
# =========================================================
def add_regime_snapshot(bot_id: int, symbol: str, regime: str, confidence: float, why: str, snapshot: str) -> None:
    con = _conn()
    con.execute(
        "INSERT INTO regime_snapshots(bot_id, ts, symbol, regime, confidence, why, snapshot) VALUES (?,?,?,?,?,?,?)",
        (int(bot_id), now_ts(), str(symbol), str(regime), float(confidence), str(why), str(snapshot)),
    )
    con.commit()
    con.close()


def get_latest_regime_for_symbols(symbols: List[str]) -> Dict[str, Dict[str, Any]]:
    """Most recent regime_snapshots row per symbol (any bot)."""
    out: Dict[str, Dict[str, Any]] = {}
    if not symbols:
        return out
    try:
        con = _conn()
        for sym in symbols:
            s = str(sym or "").strip()
            if not s:
                continue
            row = con.execute(
                """
                SELECT regime, confidence, ts FROM regime_snapshots
                WHERE symbol = ? ORDER BY ts DESC LIMIT 1
                """,
                (s,),
            ).fetchone()
            if row:
                out[s] = {
                    "regime": str(row["regime"] or ""),
                    "regime_score": float(row["confidence"] or 0),
                    "updated_at": int(row["ts"] or 0),
                }
        con.close()
    except Exception as e:
        logger.warning("get_latest_regime_for_symbols: %s", e)
    return out


def add_strategy_decision(bot_id: int, strategy: str, action: str, reason: str, regime: str, confidence: float, payload: str) -> None:
    con = _conn()
    con.execute(
        """
        INSERT INTO strategy_decisions(bot_id, ts, strategy, action, reason, regime, confidence, payload)
        VALUES (?,?,?,?,?,?,?,?)
        """,
        (int(bot_id), now_ts(), str(strategy), str(action), str(reason), str(regime), float(confidence), str(payload)),
    )
    con.commit()
    con.close()


def add_strategy_trade(
    bot_id: int,
    strategy: str,
    pnl: float,
    symbol: Optional[str] = None,
    regime: Optional[str] = None,
    pnl_pct: Optional[float] = None,
    ts: Optional[int] = None,
) -> None:
    con = _conn()
    con.execute(
        "INSERT INTO strategy_perf_trades(bot_id, symbol, regime, strategy, pnl, pnl_pct, ts) VALUES (?,?,?,?,?,?,?)",
        (int(bot_id), str(symbol or ""), str(regime or ""), str(strategy), float(pnl), float(pnl_pct) if pnl_pct is not None else None, int(ts or now_ts())),
    )
    con.commit()
    con.close()


def get_strategy_perf(bot_id: int, strategy: str, window: int = 30) -> Dict[str, Any]:
    con = _conn()
    rows = con.execute(
        "SELECT pnl, pnl_pct FROM strategy_perf_trades WHERE bot_id=? AND strategy=? ORDER BY ts DESC LIMIT ?",
        (int(bot_id), str(strategy), int(window)),
    ).fetchall()
    con.close()
    pnls = [float(r["pnl"]) for r in rows] if rows else []
    pnl_pcts = [float(r["pnl_pct"]) for r in rows if r["pnl_pct"] is not None]
    if not pnls:
        return {"trades": 0, "win_rate": 0.0, "profit_factor": 0.0, "expectancy": 0.0, "max_drawdown": 0.0, "avg_win": 0.0, "avg_loss": 0.0}
    wins = [p for p in pnls if p > 0]
    losses = [p for p in pnls if p < 0]
    win_rate = len(wins) / len(pnls) if pnls else 0.0
    gross_profit = sum(wins)
    gross_loss = abs(sum(losses)) if losses else 0.0
    profit_factor = (gross_profit / gross_loss) if gross_loss > 0 else (2.0 if gross_profit > 0 else 0.0)
    expectancy = sum(pnls) / len(pnls) if pnls else 0.0
    avg_win = (sum([p for p in pnl_pcts if p > 0]) / max(1, len([p for p in pnl_pcts if p > 0]))) if pnl_pcts else 0.0
    avg_loss = abs(sum([p for p in pnl_pcts if p < 0]) / max(1, len([p for p in pnl_pcts if p < 0]))) if pnl_pcts else 0.0
    # Simple max drawdown on cumulative pnl
    peak = 0.0
    dd = 0.0
    cum = 0.0
    for p in reversed(pnls):
        cum += p
        peak = max(peak, cum)
        dd = max(dd, peak - cum)
    return {
        "trades": len(pnls),
        "win_rate": win_rate,
        "profit_factor": profit_factor,
        "expectancy": expectancy,
        "max_drawdown": dd,
        "avg_win": avg_win,
        "avg_loss": avg_loss,
    }


def get_symbol_profit_factor(symbol: str, window_days: int = 90) -> Dict[str, Any]:
    """
    Compute profit_factor, win_rate, and trade count for a symbol across ALL bots.
    Used by intelligence_layer generate_recommendation() to enrich metrics.
    Returns: {profit_factor, win_rate, trades, expectancy}
    """
    cutoff = int(now_ts()) - window_days * 86400
    con = _conn()
    rows = con.execute(
        """
        SELECT realized_pnl_quote FROM deals
        WHERE symbol=? AND state='CLOSED' AND closed_at IS NOT NULL AND closed_at >= ?
        """,
        (str(symbol), cutoff),
    ).fetchall()
    con.close()
    if not rows:
        return {"profit_factor": 0.0, "win_rate": 0.0, "trades": 0, "expectancy": 0.0}
    pnls = [float(r["realized_pnl_quote"] or 0.0) for r in rows]
    wins = [p for p in pnls if p > 0]
    losses = [p for p in pnls if p < 0]
    total = len(pnls)
    gross_profit = sum(wins)
    gross_loss = abs(sum(losses)) if losses else 0.0
    profit_factor = (gross_profit / gross_loss) if gross_loss > 0 else (2.0 if gross_profit > 0 else 0.0)
    win_rate = len(wins) / total if total else 0.0
    expectancy = sum(pnls) / total if total else 0.0
    return {
        "profit_factor": round(profit_factor, 3),
        "win_rate": round(win_rate, 3),
        "trades": total,
        "expectancy": round(expectancy, 4),
    }


def get_strategy_leaderboard(window_days: int = 90) -> List[Dict[str, Any]]:
    """
    Aggregate strategy performance across all bots.
    Returns list of {strategy, trades, win_rate, profit_factor, total_pnl, max_drawdown, sharpe_approx, score}.
    Uses strategy_perf_trades if available, else falls back to closed deals.
    """
    import math
    try:
        init_db()
    except Exception:
        pass
    since_ts = now_ts() - (window_days * 86400)
    con = _conn()

    # Try strategy_perf_trades first
    try:
        rows = con.execute(
            """
            SELECT strategy, pnl, pnl_pct, ts FROM strategy_perf_trades
            WHERE ts >= ?
            ORDER BY ts ASC
            """,
            (since_ts,),
        ).fetchall()
        raw_trades = [{"strategy": r["strategy"], "pnl": float(r["pnl"] or 0), "pnl_pct": r["pnl_pct"], "ts": r["ts"]} for r in rows]
    except Exception:
        raw_trades = []

    # Fallback: build from closed deals if no strategy_perf_trades data
    if not raw_trades:
        try:
            deal_rows = con.execute(
                """
                SELECT exit_strategy, entry_strategy, realized_pnl_quote, closed_at, entry_avg, base_amount
                FROM deals
                WHERE state='CLOSED' AND closed_at IS NOT NULL AND closed_at >= ?
                ORDER BY closed_at ASC
                """,
                (since_ts,),
            ).fetchall()
            for r in deal_rows:
                strat = str(r["exit_strategy"] or r["entry_strategy"] or "classic_dca").strip() or "classic_dca"
                pnl = float(r["realized_pnl_quote"] or 0)
                entry = float(r["entry_avg"] or 0)
                base_amt = float(r["base_amount"] or 0)
                notional = entry * base_amt if (entry > 0 and base_amt > 0) else 0.0
                pnl_pct = (pnl / notional) if notional and notional > 0 else None
                raw_trades.append({"strategy": strat, "pnl": pnl, "pnl_pct": pnl_pct, "ts": r["closed_at"]})
        except Exception:
            pass
    con.close()
    if not raw_trades:
        return []

    rows = raw_trades
    by_strategy: Dict[str, List[Dict[str, Any]]] = {}
    for r in rows:
        s = str(r["strategy"] or "unknown").strip() or "unknown"
        if s not in by_strategy:
            by_strategy[s] = []
        by_strategy[s].append({
            "pnl": float(r["pnl"] or 0),
            "pnl_pct": float(r["pnl_pct"]) if r["pnl_pct"] is not None else None,
            "ts": int(r["ts"] or 0),
        })
    result = []
    for strategy, trades in by_strategy.items():
        pnls = [t["pnl"] for t in trades]
        pnl_pcts = [t["pnl_pct"] for t in trades if t["pnl_pct"] is not None]
        n = len(pnls)
        if n == 0:
            continue
        wins = [p for p in pnls if p > 0]
        losses = [p for p in pnls if p < 0]
        win_rate = len(wins) / n
        gross_profit = sum(wins)
        gross_loss = abs(sum(losses)) if losses else 0.0
        profit_factor = (gross_profit / gross_loss) if gross_loss > 0 else (2.0 if gross_profit > 0 else 0.0)
        total_pnl = sum(pnls)
        peak = 0.0
        cum = 0.0
        max_dd = 0.0
        for p in pnls:
            cum += p
            peak = max(peak, cum)
            max_dd = max(max_dd, peak - cum)
        mean_pct = sum(pnl_pcts) / len(pnl_pcts) if pnl_pcts else 0.0
        std = 0.0
        if len(pnl_pcts) >= 5:
            variance = sum((x - mean_pct) ** 2 for x in pnl_pcts) / len(pnl_pcts)
            std = math.sqrt(variance) if variance > 0 else 0.0
        sharpe_approx = (mean_pct / std) if std > 0.0001 else 0.0
        score = (win_rate * 0.25) + (min(profit_factor, 3.0) / 3.0 * 0.25) + (max(0, 1.0 - max_dd / 50) * 0.25) + (max(0, min(sharpe_approx, 2.0)) / 2.0 * 0.25)
        result.append({
            "strategy": strategy,
            "trades": n,
            "win_rate": round(win_rate * 100, 1),
            "profit_factor": round(profit_factor, 2),
            "total_pnl": round(total_pnl, 2),
            "max_drawdown": round(max_dd, 2),
            "sharpe_approx": round(sharpe_approx, 2),
            "score": round(score * 100, 1),
        })
    result.sort(key=lambda x: x["score"], reverse=True)
    return result


def get_expected_edge(
    symbol: str,
    regime: str,
    strategy: str,
    window: int = 100,
    prior_weight: int = 50,
) -> Dict[str, Any]:
    """
    Bayesian-shrunk expected edge in pct terms using (symbol, regime, strategy).
    expected_edge = P_win * avg_win - P_loss * avg_loss
    """
    con = _conn()
    rows = con.execute(
        """
        SELECT pnl_pct FROM strategy_perf_trades
        WHERE symbol=? AND regime=? AND strategy=?
        ORDER BY ts DESC LIMIT ?
        """,
        (str(symbol), str(regime), str(strategy), int(window)),
    ).fetchall()
    global_rows = con.execute(
        "SELECT pnl_pct FROM strategy_perf_trades ORDER BY ts DESC LIMIT ?",
        (int(window),),
    ).fetchall()
    con.close()

    def _stats(rs):
        vals = [float(r["pnl_pct"]) for r in rs if r["pnl_pct"] is not None]
        wins = [v for v in vals if v > 0]
        losses = [v for v in vals if v < 0]
        p_win = len(wins) / len(vals) if vals else 0.0
        avg_win = (sum(wins) / len(wins)) if wins else 0.0
        avg_loss = abs(sum(losses) / len(losses)) if losses else 0.0
        return len(vals), p_win, avg_win, avg_loss

    n, p_win, avg_win, avg_loss = _stats(rows)
    g_n, g_p_win, g_avg_win, g_avg_loss = _stats(global_rows)

    w = max(1, int(prior_weight))
    if g_n <= 0:
        g_p_win, g_avg_win, g_avg_loss = 0.5, 0.01, 0.01

    p_win_b = (n * p_win + w * g_p_win) / (n + w)
    avg_win_b = (n * avg_win + w * g_avg_win) / (n + w)
    avg_loss_b = (n * avg_loss + w * g_avg_loss) / (n + w)
    p_loss_b = 1.0 - p_win_b
    edge = (p_win_b * avg_win_b) - (p_loss_b * avg_loss_b)

    return {
        "trades": n,
        "p_win": p_win_b,
        "p_loss": p_loss_b,
        "avg_win": avg_win_b,
        "avg_loss": avg_loss_b,
        "expected_edge": edge,
    }


def count_orders_today(bot_id: int, live_only: bool = True) -> int:
    """Count order events for bot in last 24h. Used by risk engine.
    
    live_only=True (default): only count real orders (is_live=1),
    so dry-run simulations don't inflate the counter and block trading.
    """
    try:
        start = int(time.time()) - (24 * 3600)
        con = _conn()
        if live_only:
            row = con.execute(
                "SELECT COUNT(*) as n FROM order_events WHERE bot_id=? AND ts>=? AND is_live=1",
                (int(bot_id), start),
            ).fetchone()
        else:
            row = con.execute(
                "SELECT COUNT(*) as n FROM order_events WHERE bot_id=? AND ts>=?",
                (int(bot_id), start),
            ).fetchone()
        con.close()
        return int(row["n"]) if row else 0
    except Exception:
        return 0


def add_order_event(
    bot_id: int,
    symbol: str,
    side: str,
    ord_type: str,
    price: Optional[float],
    amount: Optional[float],
    order_id: Optional[str],
    tag: Optional[str],
    status: str,
    reason: str,
    is_live: int = 0,
) -> None:
    con = _conn()
    con.execute(
        """
        INSERT INTO order_events(bot_id, ts, symbol, side, ord_type, price, amount, order_id, tag, status, reason, is_live)
        VALUES (?,?,?,?,?,?,?,?,?,?,?,?)
        """,
        (
            int(bot_id),
            now_ts(),
            str(symbol),
            str(side),
            str(ord_type),
            float(price) if price is not None else None,
            float(amount) if amount is not None else None,
            str(order_id) if order_id else None,
            str(tag) if tag else None,
            str(status),
            str(reason),
            int(1 if is_live else 0),
        ),
    )
    con.commit()
    con.close()


def save_perf_metrics(bot_id: int, strategy: str, payload: str) -> None:
    con = _conn()
    con.execute(
        "INSERT INTO perf_metrics(bot_id, ts, strategy, payload) VALUES (?,?,?,?)",
        (int(bot_id), now_ts(), str(strategy), str(payload)),
    )
    con.commit()
    con.close()


def save_backtest_run(symbol: str, strategy: str, params: str, metrics: str, equity: str) -> None:
    con = _conn()
    con.execute(
        "INSERT INTO backtest_runs(ts, symbol, strategy, params, metrics, equity) VALUES (?,?,?,?,?,?)",
        (now_ts(), str(symbol), str(strategy), str(params), str(metrics), str(equity)),
    )
    con.commit()
    con.close()


def latest_regime(bot_id: int) -> Optional[Dict[str, Any]]:
    con = _conn()
    row = con.execute(
        "SELECT ts, symbol, regime, confidence, why, snapshot FROM regime_snapshots WHERE bot_id=? ORDER BY ts DESC LIMIT 1",
        (int(bot_id),),
    ).fetchone()
    con.close()
    return dict(row) if row else None


def list_strategy_decisions(bot_id: int, limit: int = 100) -> List[Dict[str, Any]]:
    con = _conn()
    rows = con.execute(
        """
        SELECT ts, strategy, action, reason, regime, confidence, payload
        FROM strategy_decisions
        WHERE bot_id=?
        ORDER BY ts DESC
        LIMIT ?
        """,
        (int(bot_id), int(limit)),
    ).fetchall()
    con.close()
    return [dict(r) for r in rows]


def list_backtest_runs(limit: int = 50) -> List[Dict[str, Any]]:
    con = _conn()
    rows = con.execute(
        """
        SELECT ts, symbol, strategy, params, metrics, equity
        FROM backtest_runs
        ORDER BY ts DESC
        LIMIT ?
        """,
        (int(limit),),
    ).fetchall()
    con.close()
    return [dict(r) for r in rows]


# =========================================================
# Recommendations
# =========================================================
def save_recommendation_snapshot(
    symbol: str,
    horizon: str,
    score: float,
    regime_json: str,
    metrics_json: str,
    reasons_json: str,
    risk_flags_json: str,
    score_breakdown_json: str = "",
    composite_score: Optional[float] = None,
    confidence_score: Optional[float] = None,
    conviction_grade: Optional[str] = None,
    factor_scores_json: str = "",
    signal_flags_json: str = "",
) -> int:
    def _do_save():
        con = _conn()
        cur = con.cursor()
        cur.execute(
            """
            INSERT INTO recommendations_snapshots(
                symbol, horizon, score, regime_json, metrics_json, reasons_json, risk_flags_json,
                created_ts, score_breakdown_json, composite_score, confidence_score, conviction_grade,
                factor_scores_json, signal_flags_json
            ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)
            """,
            (
                str(symbol),
                str(horizon),
                float(score),
                str(regime_json or ""),
                str(metrics_json or ""),
                str(reasons_json or ""),
                str(risk_flags_json or ""),
                now_ts(),
                str(score_breakdown_json or ""),
                float(composite_score) if composite_score is not None else None,
                float(confidence_score) if confidence_score is not None else None,
                str(conviction_grade) if conviction_grade else None,
                str(factor_scores_json or ""),
                str(signal_flags_json or ""),
            ),
        )
        snapshot_id = int(cur.lastrowid)
        cur.execute(
            """
            INSERT INTO recommendations_latest(symbol, horizon, snapshot_id, created_ts)
            VALUES (?,?,?,?)
            ON CONFLICT(symbol, horizon) DO UPDATE SET snapshot_id=excluded.snapshot_id, created_ts=excluded.created_ts
            """,
            (str(symbol), str(horizon), snapshot_id, now_ts()),
        )
        con.commit()
        con.close()
        return snapshot_id
    return _db_retry(_do_save)


def mark_explore_signals_pending(horizon: str, scan_ts: int) -> None:
    """Start-of-scan: mark all explore_signals rows for this horizon pending (stale-safe)."""
    hor = str(horizon or "short").strip().lower()
    if hor not in ("short", "medium", "long"):
        hor = "short"
    def _do_mark():
        con = _conn()
        con.execute(
            "UPDATE explore_signals SET status='pending', updated_ts=? WHERE horizon=?",
            (int(scan_ts), hor),
        )
        con.commit()
        con.close()
    try:
        _db_retry(_do_mark)
    except Exception as e:
        logger.warning("mark_explore_signals_pending failed: %s", e)


def mark_explore_horizon_pending(horizon: str) -> None:
    """Backward-compatible alias using current timestamp as scan cycle id."""
    mark_explore_signals_pending(horizon, now_ts())


def upsert_explore_feed_row(
    symbol: str,
    horizon: str,
    status: str,
    conviction_score: float,
    reason: Optional[str],
    strategy: Optional[str],
    signal_ts: int,
    detail_json: Optional[str],
    price: Optional[float],
    change_24h: Optional[float],
    market_type: Optional[str],
    rejection_reason: Optional[str] = None,
) -> None:
    """
    Single source of truth for Explore: UPSERT only, PK (symbol, horizon).
    status: pending | buy | watch | rejected
    """
    sym = str(symbol or "").strip()
    hor = str(horizon or "short").strip().lower()
    if not sym:
        return
    if hor not in ("short", "medium", "long"):
        hor = "short"
    st = str(status or "rejected").strip().lower()
    if st not in ("pending", "buy", "watch", "rejected"):
        st = "rejected"
    ts = now_ts()
    def _do_upsert():
        con = _conn()
        con.execute(
            """
            INSERT INTO explore_signals(
                symbol, horizon, status, conviction_score, reason, strategy,
                signal_ts, updated_ts, market_type, price, change_24h, detail_json,
                rejection_reason
            ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)
            ON CONFLICT(symbol, horizon) DO UPDATE SET
                status=excluded.status,
                conviction_score=excluded.conviction_score,
                reason=excluded.reason,
                strategy=excluded.strategy,
                signal_ts=excluded.signal_ts,
                updated_ts=excluded.updated_ts,
                market_type=excluded.market_type,
                price=excluded.price,
                change_24h=excluded.change_24h,
                detail_json=excluded.detail_json,
                rejection_reason=excluded.rejection_reason
            """,
            (
                sym,
                hor,
                st,
                float(conviction_score or 0),
                (str(reason)[:2000] if reason else None),
                (str(strategy)[:128] if strategy else None),
                int(signal_ts or 0),
                ts,
                (str(market_type)[:16] if market_type else None),
                float(price) if price is not None else None,
                float(change_24h) if change_24h is not None else None,
                (str(detail_json)[:12000] if detail_json else None),
                (str(rejection_reason)[:128] if rejection_reason else None),
            ),
        )
        con.commit()
        con.close()
    try:
        _db_retry(_do_upsert)
    except Exception as e:
        logger.warning("upsert_explore_feed_row failed: %s", e)


def list_explore_feed(
    horizon: str,
    *,
    market_type: str = "all",
    statuses: Optional[List[str]] = None,
    limit: int = 100,
) -> List[Dict[str, Any]]:
    """Read-only feed for Explore UI — no API-side filtering beyond market_type / status."""
    hor = str(horizon or "short").strip().lower()
    if hor not in ("short", "medium", "long"):
        hor = "short"
    want_mt = str(market_type or "all").strip().lower()
    st_list = statuses if statuses else ["buy", "watch"]
    st_list = [str(s).lower() for s in st_list if s]
    if not st_list:
        st_list = ["buy", "watch"]
    lim = max(1, min(int(limit), 500))
    placeholders = ",".join("?" * len(st_list))
    try:
        con = _conn()
        q = f"""
            SELECT symbol, status, conviction_score, reason, strategy, signal_ts, updated_ts,
                   market_type, price, change_24h, detail_json, rejection_reason
            FROM explore_signals
            WHERE horizon=? AND status IN ({placeholders})
        """
        params: List[Any] = [hor] + st_list
        if want_mt == "crypto":
            q += " AND market_type='crypto'"
        elif want_mt == "stocks":
            q += " AND market_type='stocks'"
        q += " ORDER BY conviction_score DESC LIMIT ?"
        params.append(lim)
        rows = con.execute(q, tuple(params)).fetchall()
        con.close()
        return [
            {
                "symbol": str(r[0]),
                "status": str(r[1]),
                "conviction_score": float(r[2] or 0),
                "reason": str(r[3] or ""),
                "strategy": str(r[4] or ""),
                "signal_ts": int(r[5] or 0),
                "updated_ts": int(r[6] or 0),
                "market_type": str(r[7] or ""),
                "price": float(r[8]) if r[8] is not None else None,
                "change_24h": float(r[9]) if r[9] is not None else None,
                "detail_json": str(r[10] or ""),
                "rejection_reason": str(r[11] or "") if len(r) > 11 else "",
            }
            for r in rows
        ]
    except Exception as e:
        logger.warning("list_explore_feed failed: %s", e)
        return []


def explore_signals_max_updated_ts(horizon: str, statuses: Optional[List[str]] = None) -> int:
    """Latest updated_ts among explore_signals rows for freshness checks (autopilot)."""
    hor = str(horizon or "short").strip().lower()
    if hor not in ("short", "medium", "long"):
        hor = "short"
    st = statuses or ["buy", "watch"]
    st = [str(x).lower() for x in st if x]
    if not st:
        st = ["buy", "watch"]
    ph = ",".join("?" * len(st))
    try:
        con = _conn()
        row = con.execute(
            f"SELECT MAX(updated_ts) AS m FROM explore_signals WHERE horizon=? AND status IN ({ph})",
            tuple([hor] + st),
        ).fetchone()
        con.close()
        return int(row[0] or 0) if row else 0
    except Exception:
        return 0


def list_signal_accuracy_baselines() -> List[Dict[str, Any]]:
    try:
        con = _conn()
        rows = con.execute(
            """
            SELECT strategy_id, horizon, asset_type, sample_size, win_rate,
                   avg_return_pct, avg_hold_hours, sharpe_ratio, last_updated
            FROM signal_accuracy_baseline
            ORDER BY horizon, asset_type, strategy_id
            """
        ).fetchall()
        con.close()
        return [dict(r) for r in rows]
    except Exception as e:
        logger.warning("list_signal_accuracy_baselines: %s", e)
        return []


def list_portfolio_equity_curve(days: int = 30) -> List[Dict[str, Any]]:
    """Time series from portfolio_snapshots for equity curve charts."""
    d = max(1, min(int(days), 730))
    try:
        con = _conn()
        rows = con.execute(
            """
            SELECT timestamp, total_value, total_pnl, realized_pnl, unrealized_pnl
            FROM portfolio_snapshots
            WHERE datetime(timestamp) >= datetime('now', '-' || ? || ' days')
            ORDER BY datetime(timestamp) ASC
            """,
            (d,),
        ).fetchall()
        con.close()
        out: List[Dict[str, Any]] = []
        cum = 0.0
        for r in rows:
            drow = dict(r)
            rp = _safe_float_db(drow.get("realized_pnl"), 0.0)
            cum += rp
            out.append({
                "timestamp": str(drow.get("timestamp") or ""),
                "portfolio_value": _safe_float_db(drow.get("total_value"), 0.0),
                "total_pnl": _safe_float_db(drow.get("total_pnl"), 0.0),
                "realized_pnl": rp,
                "unrealized_pnl": _safe_float_db(drow.get("unrealized_pnl"), 0.0),
                "cumulative_realized_pnl": round(cum, 6),
            })
        return out
    except Exception as e:
        logger.warning("list_portfolio_equity_curve: %s", e)
        return []


def save_explore_backtest_results(horizon: str, results: Dict[str, Any]) -> int:
    """Persist full backtest JSON; returns row id."""
    hor = str(horizon or "short").strip().lower()
    if hor not in ("short", "medium", "long"):
        hor = "short"
    try:
        con = _conn()
        cur = con.cursor()
        cur.execute(
            "INSERT INTO explore_backtest_results(horizon, computed_ts, results_json) VALUES (?,?,?)",
            (hor, now_ts(), json.dumps(results)),
        )
        rid = int(cur.lastrowid)
        con.commit()
        con.close()
        return rid
    except Exception as e:
        logger.warning("save_explore_backtest_results failed: %s", e)
        return 0


def get_latest_explore_backtest(horizon: str) -> Optional[Dict[str, Any]]:
    hor = str(horizon or "short").strip().lower()
    try:
        con = _conn()
        row = con.execute(
            """
            SELECT id, computed_ts, results_json FROM explore_backtest_results
            WHERE horizon=? ORDER BY computed_ts DESC LIMIT 1
            """,
            (hor,),
        ).fetchone()
        con.close()
        if not row:
            return None
        return {
            "id": int(row[0]),
            "computed_ts": int(row[1] or 0),
            "results": json.loads(row[2] or "{}"),
        }
    except Exception as e:
        logger.warning("get_latest_explore_backtest failed: %s", e)
        return None


def get_explore_rejected_symbols(horizon: str) -> List[str]:
    """Symbols marked rejected for this horizon (rejection log / strict filter)."""
    hor = str(horizon or "short").strip().lower()
    try:
        con = _conn()
        rows = con.execute(
            "SELECT symbol FROM explore_signals WHERE horizon=? AND status='rejected'",
            (hor,),
        ).fetchall()
        con.close()
        return [str(r[0]) for r in rows if r and r[0]]
    except Exception as e:
        logger.warning("get_explore_rejected_symbols failed: %s", e)
        return []


def get_explore_api_excluded_symbols(horizon: str) -> List[str]:
    """Hide from /api/recommendations: rejected or not yet updated this scan cycle (pending)."""
    hor = str(horizon or "short").strip().lower()
    try:
        con = _conn()
        rows = con.execute(
            """
            SELECT symbol FROM explore_signals
            WHERE horizon=? AND status IN ('rejected','pending')
            """,
            (hor,),
        ).fetchall()
        con.close()
        return [str(r[0]) for r in rows if r and r[0]]
    except Exception as e:
        logger.warning("get_explore_api_excluded_symbols failed: %s", e)
        return []


def save_signal_outcome(
    symbol: str,
    horizon: str,
    strategy: str,
    signal_ts: int,
    entry_price: float,
    composite_score: Optional[float] = None,
    conviction_grade: Optional[str] = None,
) -> int:
    """Record a new buy-signal row for forward outcome tracking (pending)."""
    sym = str(symbol or "").strip()
    hor = str(horizon or "short").strip().lower()
    if not sym or hor not in ("short", "medium", "long"):
        return 0
    st = str(strategy or "Trend Follow")[:200]
    try:
        ep = float(entry_price)
        if ep <= 0:
            return 0
    except (TypeError, ValueError):
        return 0
    ts = int(signal_ts or now_ts())
    chk = now_ts()
    try:
        con = _conn()
        cur = con.cursor()
        cur.execute(
            """
            INSERT INTO explore_signal_outcomes(
                symbol, horizon, strategy, signal_ts, entry_price,
                outcome, composite_score, conviction_grade, checked_ts
            ) VALUES (?,?,?,?,?,?,?,?,?)
            """,
            (
                sym,
                hor,
                st,
                ts,
                ep,
                "pending",
                float(composite_score) if composite_score is not None else None,
                str(conviction_grade)[:8] if conviction_grade else None,
                chk,
            ),
        )
        rid = int(cur.lastrowid)
        con.commit()
        con.close()
        return rid
    except Exception as e:
        logger.warning("save_signal_outcome failed: %s", e)
        return 0


def update_explore_signal_outcome(
    outcome_id: int,
    price_5d: Optional[float] = None,
    price_10d: Optional[float] = None,
    price_20d: Optional[float] = None,
) -> None:
    """Fill forward prices for explore_signal_outcomes; derive PnL % and outcome from 10d bar when available."""
    if not outcome_id:
        return
    try:
        con = _conn()
        row = con.execute(
            "SELECT entry_price, outcome FROM explore_signal_outcomes WHERE id=?",
            (int(outcome_id),),
        ).fetchone()
        if not row:
            con.close()
            return
        entry = float(row[0] or 0)
        prev_out = str(row[1] or "pending")
        if entry <= 0:
            con.close()
            return
        p5 = float(price_5d) if price_5d is not None else None
        p10 = float(price_10d) if price_10d is not None else None
        p20 = float(price_20d) if price_20d is not None else None
        pnl5 = ((p5 - entry) / entry * 100.0) if p5 is not None else None
        pnl10 = ((p10 - entry) / entry * 100.0) if p10 is not None else None
        pnl20 = ((p20 - entry) / entry * 100.0) if p20 is not None else None
        outcome = prev_out
        if str(prev_out).lower() not in ("win", "loss"):
            if pnl10 is not None:
                if pnl10 > 0:
                    outcome = "win"
                elif pnl10 < -2.0:
                    outcome = "loss"
                elif pnl20 is not None:
                    outcome = "win" if pnl20 > 0 else "loss"
            elif pnl20 is not None:
                outcome = "win" if pnl20 > 0 else "loss"
        con.execute(
            """
            UPDATE explore_signal_outcomes SET
                price_5d=COALESCE(?, price_5d),
                price_10d=COALESCE(?, price_10d),
                price_20d=COALESCE(?, price_20d),
                pnl_5d_pct=COALESCE(?, pnl_5d_pct),
                pnl_10d_pct=COALESCE(?, pnl_10d_pct),
                pnl_20d_pct=COALESCE(?, pnl_20d_pct),
                outcome=?,
                checked_ts=?
            WHERE id=?
            """,
            (p5, p10, p20, pnl5, pnl10, pnl20, outcome, now_ts(), int(outcome_id)),
        )
        con.commit()
        con.close()
    except Exception as e:
        logger.warning("update_explore_signal_outcome failed: %s", e)


def list_explore_outcomes_pending_old(
    min_age_sec: int = 5 * 86400,
    limit: int = 500,
) -> List[Dict[str, Any]]:
    """Pending rows at least min_age_sec old (calendar proxy for trading days)."""
    lim = max(1, min(int(limit), 2000))
    cutoff = now_ts() - int(min_age_sec)
    try:
        con = _conn()
        rows = con.execute(
            """
            SELECT id, symbol, horizon, strategy, signal_ts, entry_price
            FROM explore_signal_outcomes
            WHERE (outcome='pending' OR outcome IS NULL OR outcome='')
              AND signal_ts <= ?
            ORDER BY signal_ts ASC
            LIMIT ?
            """,
            (cutoff, lim),
        ).fetchall()
        con.close()
        return [
            {
                "id": int(r[0]),
                "symbol": str(r[1]),
                "horizon": str(r[2]),
                "strategy": str(r[3]),
                "signal_ts": int(r[4] or 0),
                "entry_price": float(r[5] or 0),
            }
            for r in rows
        ]
    except Exception as e:
        logger.warning("list_explore_outcomes_pending_old failed: %s", e)
        return []


def get_strategy_win_rates(horizon: str, lookback_days: int = 90) -> Dict[str, Any]:
    """
    Per-strategy aggregates for completed outcomes in lookback window.
    Returns { strategy: {signals, wins, win_rate, avg_return_10d, low_accuracy} }
    """
    hor = str(horizon or "short").strip().lower()
    if hor not in ("short", "medium", "long"):
        hor = "short"
    cutoff = now_ts() - max(1, int(lookback_days)) * 86400
    out: Dict[str, Any] = {}
    try:
        con = _conn()
        rows = con.execute(
            """
            SELECT strategy,
                   COUNT(*) AS n,
                   SUM(CASE WHEN outcome='win' THEN 1 ELSE 0 END) AS wins,
                   AVG(pnl_10d_pct) AS avg10
            FROM explore_signal_outcomes
            WHERE horizon=? AND signal_ts>=? AND outcome IN ('win','loss')
            GROUP BY strategy
            """,
            (hor, cutoff),
        ).fetchall()
        con.close()
        for r in rows:
            strat = str(r[0] or "")
            n = int(r[1] or 0)
            w = int(r[2] or 0)
            avg10 = float(r[3]) if r[3] is not None else None
            wr = (w / n) if n else 0.0
            out[strat] = {
                "signals": n,
                "wins": w,
                "win_rate": wr,
                "avg_return_10d": avg10,
                "low_accuracy": bool(n >= 5 and wr < 0.55),
            }
        return out
    except Exception as e:
        logger.warning("get_strategy_win_rates failed: %s", e)
        return {}


def list_explore_rejected(horizon: str, limit: int = 50) -> List[Dict[str, Any]]:
    """Recent rejections for Explore UI / /api/explore/signals."""
    hor = str(horizon or "short").strip().lower()
    lim = max(1, min(int(limit), 200))
    try:
        con = _conn()
        rows = con.execute(
            """
            SELECT symbol, reason, strategy, updated_ts, rejection_reason
            FROM explore_signals
            WHERE horizon=? AND status='rejected'
            ORDER BY updated_ts DESC
            LIMIT ?
            """,
            (hor, lim),
        ).fetchall()
        con.close()
        return [
            {
                "symbol": str(r[0]),
                "reason": str(r[1] or ""),
                "strategy": str(r[2] or ""),
                "updated_ts": int(r[3] or 0),
                "rejection_reason": str(r[4] or "").strip() or str(r[1] or ""),
            }
            for r in rows
        ]
    except Exception as e:
        logger.warning("list_explore_rejected failed: %s", e)
        return []


def count_recommendations_by_horizon() -> Dict[str, int]:
    """Return {horizon: count} for short, medium, long. Used for API logging and UI."""
    out: Dict[str, int] = {"short": 0, "medium": 0, "long": 0}
    try:
        con = _conn()
        for h in ("short", "medium", "long"):
            n = con.execute(
                "SELECT COUNT(*) FROM recommendations_latest WHERE horizon=?",
                (str(h),),
            ).fetchone()[0]
            out[h] = int(n) if n is not None else 0
        con.close()
    except Exception as e:
        logger.error("count_recommendations_by_horizon error: %s", e)
    return out


def list_recommendations(
    horizon: str, limit: int = 200, exclude_bases: Optional[List[str]] = None
) -> List[Dict[str, Any]]:
    """Fast query for recommendations. Returns empty list on error.
    exclude_bases: list of crypto base symbols to never return (e.g. ['STABLE'])."""
    try:
        con = _conn()
        con.execute("PRAGMA busy_timeout = 10000")
        rows = con.execute(
            """
            SELECT s.*
            FROM recommendations_snapshots s
            JOIN recommendations_latest l ON l.snapshot_id = s.id
            WHERE l.horizon=?
            ORDER BY s.score DESC, s.created_ts DESC
            LIMIT ?
            """,
            (str(horizon), int(limit)),
        ).fetchall()
        result = [dict(r) for r in rows]
        con.close()
        # Filter blocklisted crypto symbols
        if exclude_bases:
            exclude_set = {str(b).strip().upper() for b in exclude_bases if b}
            filtered = []
            for r in result:
                sym = str(r.get("symbol") or "")
                if "/" in sym:
                    base = (sym.split("/")[0] or "").upper()
                    if base in exclude_set:
                        continue
                filtered.append(r)
            return filtered
        return result
    except Exception as e:
        # Log but don't raise - return empty list
        import logging
        logger = logging.getLogger(__name__)
        logger.error(f"list_recommendations error: {e}")
        return []


def delete_recommendations_for_blocklist(bases: List[str]) -> int:
    """Remove recommendations for crypto symbols whose base is in the blocklist.
    Call on startup to purge STABLE and other blocked tokens from Explore.
    Returns count of symbols removed."""
    if not bases:
        return 0
    try:
        con = _conn()
        cur = con.cursor()
        deleted = 0
        for base in bases:
            b = str(base).strip().upper()
            if not b:
                continue
            # Match STABLE/USD, stable/usd, etc. (case-insensitive via UPPER)
            cur.execute(
                "DELETE FROM recommendations_latest WHERE UPPER(symbol) LIKE ? OR UPPER(symbol) = ?",
                (b + "/%", b),
            )
            deleted += cur.rowcount
        con.commit()
        con.close()
        return deleted
    except Exception as e:
        import logging
        logging.getLogger(__name__).warning("delete_recommendations_for_blocklist: %s", e)
        return 0


def cleanup_invalid_scores() -> int:
    """Remove recommendations with score >= 98 and empty/null reasons (data artifacts).
    Also remove stock recommendations older than 24h with no valid metrics."""
    try:
        con = _conn()
        cur = con.cursor()
        # Remove perfect scores with no reasons (artifacts)
        # Note: recommendations_latest doesn't have a score column — join to snapshots
        cur.execute(
            """DELETE FROM recommendations_latest WHERE snapshot_id IN (
                SELECT id FROM recommendations_snapshots
                WHERE score >= 98 AND (reasons_json IS NULL OR reasons_json = '[]' OR reasons_json = '')
            )"""
        )
        deleted = cur.rowcount
        # Remove very old recommendations (>48h) to prevent stale data
        cur.execute(
            "DELETE FROM recommendations_latest WHERE created_ts < ? AND created_ts > 0",
            (int(time.time()) - 48 * 3600,),
        )
        deleted += cur.rowcount
        con.commit()
        con.close()
        return deleted
    except Exception as e:
        import logging
        logging.getLogger(__name__).warning("cleanup_invalid_scores: %s", e)
        return 0


def get_recommendation(symbol: str, horizon: str) -> Optional[Dict[str, Any]]:
    con = _conn()
    row = con.execute(
        """
        SELECT s.*
        FROM recommendations_snapshots s
        JOIN recommendations_latest l ON l.snapshot_id = s.id
        WHERE l.symbol=? AND l.horizon=?
        LIMIT 1
        """,
        (str(symbol), str(horizon)),
    ).fetchone()
    con.close()
    return dict(row) if row else None


# =========================================================
# Recommendation performance tracking
# =========================================================
def link_recommendation_to_bot(
    bot_id: int,
    symbol: str,
    recommendation_date: int,
    score_at_recommendation: float,
    regime_at_recommendation: str,
    metrics_json: str = "",
    reasons_json: str = "",
    snapshot_id: Optional[int] = None,
) -> None:
    """Record that a bot was created from a recommendation. Creates recommendation_performance row with outcome='active'."""
    con = _conn()
    try:
        con.execute(
            """
            INSERT INTO recommendation_performance(
                symbol, recommendation_date, score_at_recommendation, regime_at_recommendation,
                bot_id, outcome, notes, technical_patterns_json, snapshot_id, created_at
            ) VALUES (?,?,?,?,?,?,?,?,?,?)
            """,
            (
                str(symbol),
                int(recommendation_date),
                float(score_at_recommendation),
                str(regime_at_recommendation or ""),
                int(bot_id),
                "active",
                "",
                str(reasons_json or ""),
                int(snapshot_id) if snapshot_id else None,
                now_ts(),
            ),
        )
        con.commit()
    finally:
        con.close()


def _record_recommendation_outcome(
    con: sqlite3.Connection,
    bot_id: int,
    deal_id: int,
    entry_avg: float,
    exit_avg: float,
    realized_pnl: float,
    closed_ts: int,
    opened_ts: int,
) -> None:
    """Update recommendation_performance when a deal closes (first closed deal per bot from recommendation)."""
    row = con.execute(
        """
        SELECT id FROM recommendation_performance
        WHERE bot_id=? AND outcome='active'
        ORDER BY id ASC LIMIT 1
        """,
        (int(bot_id),),
    ).fetchone()
    if not row:
        return
    days_held = (closed_ts - opened_ts) / 86400.0 if closed_ts > opened_ts else 0.0
    outcome = "win" if realized_pnl > 0 else "loss"
    con.execute(
        """
        UPDATE recommendation_performance SET
            deal_id=?, entry_price=?, exit_price=?, pnl_realized=?, days_held=?, outcome=?
        WHERE id=?
        """,
        (int(deal_id), float(entry_avg), float(exit_avg), float(realized_pnl), float(days_held), outcome, int(row["id"])),
    )


def get_recommendation_performance_stats(days: int = 30) -> Dict[str, Any]:
    """Aggregate stats for closed recommendation outcomes. Used by /api/recommendations/performance."""
    since_ts = now_ts() - (int(days) * 86400)
    con = _conn()
    rows = con.execute(
        """
        SELECT score_at_recommendation, regime_at_recommendation, pnl_realized, outcome
        FROM recommendation_performance
        WHERE outcome IN ('win','loss') AND recommendation_date >= ?
        """,
        (since_ts,),
    ).fetchall()
    con.close()
    if not rows:
        return {
            "total_closed": 0,
            "wins": 0,
            "losses": 0,
            "win_rate": 0.0,
            "avg_profit_per_recommendation": 0.0,
            "by_score_range": [],
            "by_regime": [],
        }
    wins = [r for r in rows if r["outcome"] == "win"]
    losses = [r for r in rows if r["outcome"] == "loss"]
    total = len(rows)
    pnls = [float(r["pnl_realized"] or 0) for r in rows]
    avg_pnl = sum(pnls) / total if total else 0.0

    # By score range
    def _score_bucket(s: float) -> str:
        if s >= 80:
            return "80-100"
        if s >= 60:
            return "60-80"
        if s >= 40:
            return "40-60"
        return "0-40"

    by_range: Dict[str, List[str]] = {}
    for r in rows:
        b = _score_bucket(float(r["score_at_recommendation"] or 0))
        if b not in by_range:
            by_range[b] = []
        by_range[b].append(r["outcome"])
    by_score_range = [
        {"range": k, "total": len(v), "wins": sum(1 for o in v if o == "win"), "win_rate": (sum(1 for o in v if o == "win") / len(v) * 100) if v else 0}
        for k, v in sorted(by_range.items(), key=lambda x: x[0])
    ]

    # By regime
    by_reg: Dict[str, List[str]] = {}
    for r in rows:
        reg = str(r["regime_at_recommendation"] or "").strip() or "unknown"
        if reg not in by_reg:
            by_reg[reg] = []
        by_reg[reg].append(r["outcome"])
    by_regime = [
        {"regime": k, "total": len(v), "wins": sum(1 for o in v if o == "win"), "win_rate": (sum(1 for o in v if o == "win") / len(v) * 100) if v else 0}
        for k, v in sorted(by_reg.items(), key=lambda x: -len(x[1]))
    ]

    return {
        "total_closed": total,
        "wins": len(wins),
        "losses": len(losses),
        "win_rate": (len(wins) / total * 100) if total else 0.0,
        "avg_profit_per_recommendation": round(avg_pnl, 2),
        "by_score_range": by_score_range,
        "by_regime": by_regime,
    }


def get_per_symbol_accuracy(symbols: list, days: int = 90) -> Dict[str, Dict[str, Any]]:
    """Per-symbol win rate from recommendation_performance. Returns {symbol: {total, wins, win_rate}}."""
    if not symbols:
        return {}
    since_ts = now_ts() - (int(days) * 86400)
    con = _conn()
    placeholders = ",".join("?" for _ in symbols)
    rows = con.execute(
        f"""
        SELECT symbol, outcome
        FROM recommendation_performance
        WHERE outcome IN ('win','loss') AND recommendation_date >= ? AND symbol IN ({placeholders})
        """,
        [since_ts] + list(symbols),
    ).fetchall()
    con.close()
    result: Dict[str, Dict[str, Any]] = {}
    for r in rows:
        sym = r["symbol"]
        if sym not in result:
            result[sym] = {"total": 0, "wins": 0}
        result[sym]["total"] += 1
        if r["outcome"] == "win":
            result[sym]["wins"] += 1
    for sym in result:
        t = result[sym]["total"]
        result[sym]["win_rate"] = round(result[sym]["wins"] / t * 100, 1) if t else 0
    return result


def get_open_signal_outcomes(limit: int = 100):
    """Get BUY signals that need 24h/72h outcome checking."""
    con = _conn()
    try:
        rows = con.execute(
            """SELECT id, symbol, score, created_ts, entry_price, price_24h, price_72h, outcome_checked,
                      metrics_json
               FROM recommendations_snapshots
               WHERE outcome_checked < 2 AND score >= 60 AND created_ts > 0
               ORDER BY created_ts ASC LIMIT ?""",
            (limit,),
        ).fetchall()
        return [dict(r) for r in rows]
    finally:
        con.close()


def update_signal_outcome(snap_id: int, entry_price=None, price_24h=None, price_72h=None, outcome_24h=None, outcome_72h=None, outcome_checked=None):
    """Update outcome tracking fields on a recommendations_snapshots row."""
    con = _conn()
    try:
        sets = []
        vals = []
        if entry_price is not None:
            sets.append("entry_price=?"); vals.append(entry_price)
        if price_24h is not None:
            sets.append("price_24h=?"); vals.append(price_24h)
        if price_72h is not None:
            sets.append("price_72h=?"); vals.append(price_72h)
        if outcome_24h is not None:
            sets.append("outcome_24h=?"); vals.append(outcome_24h)
        if outcome_72h is not None:
            sets.append("outcome_72h=?"); vals.append(outcome_72h)
        if outcome_checked is not None:
            sets.append("outcome_checked=?"); vals.append(outcome_checked)
        if not sets:
            return
        vals.append(snap_id)
        con.execute(f"UPDATE recommendations_snapshots SET {','.join(sets)} WHERE id=?", vals)
        con.commit()
    finally:
        con.close()


def get_signal_accuracy_stats(days: int = 30) -> Dict[str, Any]:
    """Aggregate signal outcome stats for the accuracy bar."""
    since_ts = now_ts() - (int(days) * 86400)
    con = _conn()
    try:
        rows_24h = con.execute(
            "SELECT outcome_24h FROM recommendations_snapshots WHERE outcome_24h IS NOT NULL AND created_ts >= ?",
            (since_ts,),
        ).fetchall()
        rows_72h = con.execute(
            "SELECT outcome_72h FROM recommendations_snapshots WHERE outcome_72h IS NOT NULL AND created_ts >= ?",
            (since_ts,),
        ).fetchall()
        total_24h = len(rows_24h)
        wins_24h = sum(1 for r in rows_24h if r["outcome_24h"] == "WIN")
        total_72h = len(rows_72h)
        wins_72h = sum(1 for r in rows_72h if r["outcome_72h"] == "WIN")
        return {
            "total_24h": total_24h,
            "wins_24h": wins_24h,
            "win_rate_24h": round(wins_24h / total_24h * 100, 1) if total_24h else 0,
            "total_72h": total_72h,
            "wins_72h": wins_72h,
            "win_rate_72h": round(wins_72h / total_72h * 100, 1) if total_72h else 0,
            "total_tracked": max(total_24h, total_72h),
        }
    finally:
        con.close()


def save_scoring_calibration_log(
    scoring_version: str,
    changes_json: str,
    analysis_window_days: int,
    notes: str = "",
) -> None:
    """Log a calibration run for audit trail."""
    con = _conn()
    try:
        con.execute(
            """
            INSERT INTO scoring_calibration_log(ts, scoring_version, changes_json, analysis_window_days, notes)
            VALUES (?,?,?,?,?)
            """,
            (now_ts(), str(scoring_version), str(changes_json), int(analysis_window_days), str(notes)),
        )
        con.commit()
    finally:
        con.close()


def save_dividend_event(
    symbol: str,
    ex_date: int,
    amount: float,
    payment_date: Optional[int] = None,
    dividend_yield_pct: Optional[float] = None,
) -> None:
    """Record a dividend event for tracking."""
    con = _conn()
    try:
        con.execute(
            """
            INSERT INTO dividend_events(symbol, ex_date, payment_date, amount, dividend_yield_pct, recorded_at)
            VALUES (?,?,?,?,?,?)
            """,
            (str(symbol), int(ex_date), int(payment_date) if payment_date else None, float(amount), dividend_yield_pct, now_ts()),
        )
        con.commit()
    finally:
        con.close()


def list_dividend_events(symbol: Optional[str] = None, limit: int = 100) -> List[Dict[str, Any]]:
    """List dividend events, optionally filtered by symbol."""
    con = _conn()
    if symbol:
        rows = con.execute(
            "SELECT * FROM dividend_events WHERE symbol=? ORDER BY ex_date DESC LIMIT ?",
            (str(symbol), int(limit)),
        ).fetchall()
    else:
        rows = con.execute(
            "SELECT * FROM dividend_events ORDER BY ex_date DESC LIMIT ?",
            (int(limit),),
        ).fetchall()
    con.close()
    return [dict(r) for r in rows]


def get_trade_journal(deal_id: int) -> Optional[Dict[str, Any]]:
    """Get journal entry for deal."""
    con = _conn()
    row = con.execute("SELECT * FROM trade_journal WHERE deal_id=?", (int(deal_id),)).fetchone()
    con.close()
    return dict(row) if row else None


def upsert_trade_journal(
    deal_id: int,
    entry_reason: Optional[str] = None,
    exit_reason: Optional[str] = None,
    lessons_learned: Optional[str] = None,
    screenshot_data: Optional[str] = None,
) -> None:
    """Create or update trade journal entry. Omit fields to leave unchanged on update."""
    con = _conn()
    now = now_ts()
    cur = con.execute("SELECT * FROM trade_journal WHERE deal_id=?", (int(deal_id),))
    existing = cur.fetchone()
    if existing:
        row = dict(existing)
        er = entry_reason if entry_reason is not None else (row.get("entry_reason") or "")
        xr = exit_reason if exit_reason is not None else (row.get("exit_reason") or "")
        ll = lessons_learned if lessons_learned is not None else (row.get("lessons_learned") or "")
        sc = screenshot_data if screenshot_data is not None else (row.get("screenshot_data") or "")
        con.execute(
            """UPDATE trade_journal SET entry_reason=?, exit_reason=?, lessons_learned=?, screenshot_data=?, updated_at=? WHERE deal_id=?""",
            (er, xr, ll, sc, now, int(deal_id)),
        )
    else:
        con.execute(
            """
            INSERT INTO trade_journal(deal_id, entry_reason, exit_reason, lessons_learned, screenshot_data, updated_at)
            VALUES (?,?,?,?,?,?)
            """,
            (int(deal_id), entry_reason or "", exit_reason or "", lessons_learned or "", screenshot_data or "", now),
        )
    con.commit()
    con.close()


def list_trade_journals_for_deals(deal_ids: List[int]) -> Dict[int, Dict[str, Any]]:
    """Return journal entries keyed by deal_id for given deals."""
    if not deal_ids:
        return {}
    con = _conn()
    placeholders = ",".join("?" * len(deal_ids))
    rows = con.execute(
        f"SELECT * FROM trade_journal WHERE deal_id IN ({placeholders})",
        deal_ids,
    ).fetchall()
    con.close()
    return {int(r["deal_id"]): dict(r) for r in rows}


def save_market_event(
    event_date: int,
    event_type: str,
    symbol: Optional[str] = None,
    impact_level: int = 2,
    description: str = "",
) -> None:
    """Record market event (earnings, Fed, etc.). event_date = Unix date midnight."""
    con = _conn()
    try:
        con.execute(
            """
            INSERT INTO market_events(event_date, event_type, symbol, impact_level, description, recorded_at)
            VALUES (?,?,?,?,?,?)
            """,
            (int(event_date), str(event_type), str(symbol or ""), int(impact_level), str(description or ""), now_ts()),
        )
        con.commit()
    finally:
        con.close()


def get_events_for_dates(start_ts: int, end_ts: int, symbol: Optional[str] = None) -> List[Dict[str, Any]]:
    """Get market events in date range. start_ts/end_ts = date midnight Unix."""
    con = _conn()
    if symbol:
        rows = con.execute(
            "SELECT * FROM market_events WHERE event_date>=? AND event_date<=? AND (symbol=? OR symbol='' OR symbol IS NULL) ORDER BY event_date",
            (int(start_ts), int(end_ts), str(symbol)),
        ).fetchall()
    else:
        rows = con.execute(
            "SELECT * FROM market_events WHERE event_date>=? AND event_date<=? ORDER BY event_date",
            (int(start_ts), int(end_ts)),
        ).fetchall()
    con.close()
    return [dict(r) for r in rows]


def save_insider_transaction(
    symbol: str,
    transaction_date: int,
    transaction_type: str,
    shares: float,
    value_usd: Optional[float] = None,
    insider_title: Optional[str] = None,
    filing_url: Optional[str] = None,
) -> None:
    """Record SEC Form 4 insider transaction."""
    con = _conn()
    try:
        con.execute(
            """
            INSERT INTO insider_transactions(symbol, transaction_date, transaction_type, shares, value_usd, insider_title, filing_url, recorded_at)
            VALUES (?,?,?,?,?,?,?,?)
            """,
            (str(symbol), int(transaction_date), str(transaction_type), float(shares), value_usd, insider_title or "", filing_url or "", now_ts()),
        )
        con.commit()
    finally:
        con.close()


def get_insider_transactions(symbol: str, days_back: int = 90) -> List[Dict[str, Any]]:
    """Get recent insider transactions for symbol."""
    from datetime import datetime, timezone, timedelta
    cutoff = int((datetime.now(timezone.utc) - timedelta(days=days_back)).timestamp())
    con = _conn()
    rows = con.execute(
        "SELECT * FROM insider_transactions WHERE symbol=? AND transaction_date>=? ORDER BY transaction_date DESC LIMIT 100",
        (str(symbol), cutoff),
    ).fetchall()
    con.close()
    return [dict(r) for r in rows]


def save_ml_prediction(
    symbol: str,
    prediction_date: int,
    predicted_direction: str,
    predicted_price: Optional[float] = None,
    confidence: float = 0.0,
    price_at_prediction: Optional[float] = None,
    model_version: Optional[str] = None,
    model_used: Optional[str] = None,
    regime_at_prediction: Optional[str] = None,
) -> int:
    """Log ML prediction. Returns inserted row id."""
    con = _conn()
    try:
        cur = con.execute(
            """
            INSERT INTO ml_predictions(symbol, prediction_date, predicted_direction, predicted_price, confidence,
                price_at_prediction, model_version, model_used, regime_at_prediction, recorded_at)
            VALUES (?,?,?,?,?,?,?,?,?,?)
            """,
            (str(symbol), int(prediction_date), str(predicted_direction), predicted_price, float(confidence),
             price_at_prediction, model_version or "", model_used or "", regime_at_prediction or "", now_ts()),
        )
        con.commit()
        return cur.lastrowid or 0
    finally:
        con.close()


def update_ml_prediction_outcome(prediction_id: int, actual_outcome_7d: Optional[float] = None, actual_outcome_30d: Optional[float] = None) -> None:
    """Update prediction with actual outcome after 7/30 days."""
    con = _conn()
    try:
        if actual_outcome_7d is not None and actual_outcome_30d is not None:
            con.execute("UPDATE ml_predictions SET actual_outcome_7d=?, actual_outcome_30d=? WHERE id=?", (actual_outcome_7d, actual_outcome_30d, prediction_id))
        elif actual_outcome_7d is not None:
            con.execute("UPDATE ml_predictions SET actual_outcome_7d=? WHERE id=?", (actual_outcome_7d, prediction_id))
        elif actual_outcome_30d is not None:
            con.execute("UPDATE ml_predictions SET actual_outcome_30d=? WHERE id=?", (actual_outcome_30d, prediction_id))
        con.commit()
    finally:
        con.close()


def get_ml_predictions(symbol: Optional[str] = None, limit: int = 100, days_back: int = 0) -> List[Dict[str, Any]]:
    """Get ML predictions, optionally filtered by symbol."""
    con = _conn()
    cutoff = int(time.time()) - (days_back * 86400) if days_back > 0 else 0
    if symbol:
        rows = con.execute(
            "SELECT * FROM ml_predictions WHERE symbol=? AND recorded_at>=? ORDER BY recorded_at DESC LIMIT ?",
            (str(symbol), cutoff, limit),
        ).fetchall()
    else:
        rows = con.execute(
            "SELECT * FROM ml_predictions WHERE recorded_at>=? ORDER BY recorded_at DESC LIMIT ?",
            (cutoff, limit),
        ).fetchall()
    con.close()
    return [dict(r) for r in rows]


def get_ml_model_accuracy(days_back: int = 30, model_used: Optional[str] = None) -> Dict[str, Any]:
    """Calculate model accuracy over rolling window. Returns accuracy, precision, recall, F1."""
    con = _conn()
    cutoff = int(time.time()) - (days_back * 86400)
    where = "WHERE recorded_at>=? AND actual_outcome_7d IS NOT NULL"
    params = [cutoff]
    if model_used:
        where += " AND (model_used=? OR model_version=?)"
        params.extend([model_used, model_used])
    params = tuple(params)
    rows = con.execute(
        f"SELECT predicted_direction, actual_outcome_7d FROM ml_predictions {where}",
        params,
    ).fetchall()
    con.close()
    if not rows:
        return {"accuracy": 0.5, "precision": 0.5, "recall": 0.5, "f1": 0.5, "total": 0}
    correct = 0
    tp = fp = fn = tn = 0
    for r in rows:
        pred_up = str(r["predicted_direction"] or "").upper() == "UP"
        actual_up = float(r["actual_outcome_7d"] or 0) > 0
        if pred_up == actual_up:
            correct += 1
        if pred_up and actual_up:
            tp += 1
        elif pred_up and not actual_up:
            fp += 1
        elif not pred_up and actual_up:
            fn += 1
        else:
            tn += 1
    n = len(rows)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.5
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.5
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.5
    return {
        "accuracy": correct / n,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "total": n,
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
    }


def save_ml_model_version(model_type: str, version: str, validation_accuracy: float, deployed: bool = False) -> None:
    """Record new model version after training."""
    con = _conn()
    try:
        con.execute(
            "INSERT INTO ml_model_versions(model_type, version, validation_accuracy, trained_at, deployed) VALUES (?,?,?,?,?)",
            (str(model_type), str(version), float(validation_accuracy), now_ts(), 1 if deployed else 0),
        )
        con.commit()
    finally:
        con.close()


def save_intraday_pattern(
    symbol: str,
    pattern_type: str,
    ts: int,
    price: Optional[float] = None,
    vwap: Optional[float] = None,
    or_high: Optional[float] = None,
    or_low: Optional[float] = None,
    volume_spike_ratio: Optional[float] = None,
    bot_id: Optional[int] = None,
    payload_json: str = "",
) -> None:
    """Save intraday pattern (opening range break, VWAP cross, volume spike) for analysis."""
    con = _conn()
    try:
        con.execute(
            """
            INSERT INTO intraday_patterns(symbol, pattern_type, ts, price, vwap, or_high, or_low, volume_spike_ratio, bot_id, payload_json)
            VALUES (?,?,?,?,?,?,?,?,?,?)
            """,
            (str(symbol), str(pattern_type), int(ts), price, vwap, or_high, or_low, volume_spike_ratio, int(bot_id) if bot_id else None, str(payload_json or "")),
        )
        con.commit()
    finally:
        con.close()


def add_intelligence_decision(
    bot_id: int,
    symbol: str,
    allowed_actions: str,
    final_action: str,
    final_reason: str,
    data_ok: bool,
    data_reasons: str,
    safety_allowed: str,
    safety_reasons: str,
    regime: str,
    regime_confidence: float,
    strategy_mode: str,
    entry_style: str,
    exit_style: str,
    base_size: float,
    order_type: str,
    manage_actions: str,
    proposed_orders: str,
    debug_json: str,
    execution_result: Optional[str] = None,
    realized_slippage: Optional[float] = None,
    fill_quality: Optional[str] = None,
) -> int:
    """Log an intelligence decision to the database."""
    import json
    con = _conn()
    cur = con.cursor()
    cur.execute(
        """
        INSERT INTO intelligence_decisions(
            bot_id, ts, symbol, allowed_actions, final_action, final_reason,
            data_ok, data_reasons, safety_allowed, safety_reasons,
            regime, regime_confidence, strategy_mode, entry_style, exit_style,
            base_size, order_type, manage_actions, proposed_orders, debug_json,
            execution_result, realized_slippage, fill_quality
        )
        VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
        """,
        (
            int(bot_id),
            now_ts(),
            str(symbol),
            str(allowed_actions),
            str(final_action),
            str(final_reason),
            1 if data_ok else 0,
            str(data_reasons),
            str(safety_allowed),
            str(safety_reasons),
            str(regime),
            float(regime_confidence),
            str(strategy_mode),
            str(entry_style),
            str(exit_style),
            float(base_size),
            str(order_type),
            str(manage_actions),
            str(proposed_orders),
            str(debug_json),
            str(execution_result) if execution_result else None,
            float(realized_slippage) if realized_slippage is not None else None,
            str(fill_quality) if fill_quality else None,
        ),
    )
    decision_id = cur.lastrowid
    con.commit()
    con.close()
    return decision_id


def get_intelligence_decisions(bot_id: int, limit: int = 100) -> List[Dict[str, Any]]:
    """Get recent intelligence decisions for a bot."""
    con = _conn()
    rows = con.execute(
        """
        SELECT * FROM intelligence_decisions
        WHERE bot_id=?
        ORDER BY ts DESC
        LIMIT ?
        """,
        (int(bot_id), int(limit)),
    ).fetchall()
    con.close()
    return [dict(r) for r in rows]


def db_vacuum() -> None:
    """Run VACUUM to reclaim space and maintain performance. Safe but can take time on large DBs."""
    con = _conn()
    try:
        con.execute("VACUUM")
        con.commit()
    finally:
        con.close()


def db_analyze() -> None:
    """Run ANALYZE to update query planner statistics. Lightweight, safe to run regularly."""
    con = _conn()
    try:
        con.execute("ANALYZE")
        con.commit()
    finally:
        con.close()


def backup_db(dest_path: Optional[str] = None) -> str:
    """Copy DB to dest_path or defaults to {DB_NAME}.backup_{ts}. Returns path."""
    import shutil
    dest = dest_path or f"{DB_NAME}.backup_{now_ts()}"
    shutil.copy2(DB_NAME, dest)
    return dest


# =========================================================
# Scanner Watchlist
# =========================================================

def upsert_watchlist_entry(
    symbol: str,
    market_type: str,
    setup_json: str,
    trigger_conditions: str,
    regime: str = "",
    entry_type: str = "",
    confidence: float = 0.0,
    edge_score: float = 0.0,
) -> int:
    """Insert or update a watchlist entry for a symbol. Returns row id."""
    con = _conn()
    ts = now_ts()
    existing = con.execute(
        "SELECT id FROM scanner_watchlist WHERE symbol=? AND status='watching' LIMIT 1",
        (str(symbol),),
    ).fetchone()
    if existing:
        con.execute(
            """UPDATE scanner_watchlist
               SET setup_json=?, trigger_conditions=?, regime=?, entry_type=?,
                   confidence=?, edge_score=?, updated_at=?, market_type=?
               WHERE id=?""",
            (setup_json, trigger_conditions, regime, entry_type,
             confidence, edge_score, ts, market_type, int(existing["id"])),
        )
        row_id = int(existing["id"])
    else:
        cur = con.cursor()
        cur.execute(
            """INSERT INTO scanner_watchlist(
                   symbol, market_type, setup_json, trigger_conditions,
                   regime, entry_type, confidence, edge_score, status, created_at, updated_at
               ) VALUES (?,?,?,?,?,?,?,?,?,?,?)""",
            (str(symbol), market_type, setup_json, trigger_conditions,
             regime, entry_type, confidence, edge_score, "watching", ts, ts),
        )
        row_id = cur.lastrowid
    con.commit()
    con.close()
    return row_id


def list_watchlist(status: str = "watching", limit: int = 50) -> List[Dict[str, Any]]:
    """List watchlist entries by status."""
    con = _conn()
    rows = con.execute(
        "SELECT * FROM scanner_watchlist WHERE status=? ORDER BY edge_score DESC LIMIT ?",
        (str(status), int(limit)),
    ).fetchall()
    con.close()
    return [dict(r) for r in rows]


def get_watchlist_entry(symbol: str, status: str = "watching") -> Optional[Dict[str, Any]]:
    """Get a specific watchlist entry by symbol."""
    con = _conn()
    row = con.execute(
        "SELECT * FROM scanner_watchlist WHERE symbol=? AND status=? LIMIT 1",
        (str(symbol), str(status)),
    ).fetchone()
    con.close()
    return dict(row) if row else None


def mark_watchlist_triggered(symbol: str, bot_id: Optional[int] = None) -> None:
    """Mark a watchlist entry as triggered (converted to bot)."""
    con = _conn()
    con.execute(
        """UPDATE scanner_watchlist
           SET status='triggered', triggered_at=?, bot_id=?, updated_at=?
           WHERE symbol=? AND status='watching'""",
        (now_ts(), bot_id, now_ts(), str(symbol)),
    )
    con.commit()
    con.close()


def remove_watchlist_entry(symbol: str) -> None:
    """Remove a watchlist entry (expired or manually removed)."""
    con = _conn()
    con.execute(
        "UPDATE scanner_watchlist SET status='expired', updated_at=? WHERE symbol=? AND status='watching'",
        (now_ts(), str(symbol)),
    )
    con.commit()
    con.close()


def cleanup_old_watchlist(max_age_hours: int = 72) -> int:
    """Expire watchlist entries older than max_age_hours. Returns count expired."""
    con = _conn()
    cutoff = now_ts() - (max_age_hours * 3600)
    cur = con.cursor()
    cur.execute(
        "UPDATE scanner_watchlist SET status='expired', updated_at=? WHERE status='watching' AND created_at < ?",
        (now_ts(), cutoff),
    )
    count = cur.rowcount
    con.commit()
    con.close()
    return count


def cleanup_old_portfolio_snapshots(keep_days: int = 90) -> int:
    """Delete portfolio_snapshots older than keep_days. Returns count deleted."""
    if keep_days < 1 or keep_days > 3650:
        keep_days = 90
    con = _conn()
    try:
        cur = con.cursor()
        # Modifier must be literal; keep_days is code-controlled, not user input
        cur.execute(
            "DELETE FROM portfolio_snapshots WHERE datetime(timestamp) < datetime('now', '-' || ? || ' days')",
            (keep_days,),
        )
        count = cur.rowcount
        con.commit()
        return count
    except Exception as e:
        import logging
        logging.getLogger(__name__).warning("cleanup_old_portfolio_snapshots: %s", e)
        return 0
    finally:
        con.close()


def cleanup_old_recommendation_snapshots(keep_days: int = 7) -> int:
    """Delete recommendation snapshots older than keep_days, except those still referenced by recommendations_latest.
    Keeps at least 7 days of history for 24h/72h outcome tracking.
    Returns number of rows deleted."""
    try:
        con = _conn()
        cutoff = int(time.time()) - (keep_days * 86400)
        # Delete old snapshots NOT referenced by recommendations_latest
        cur = con.execute(
            """DELETE FROM recommendations_snapshots
               WHERE created_ts < ?
               AND id NOT IN (SELECT snapshot_id FROM recommendations_latest)""",
            (cutoff,)
        )
        deleted = cur.rowcount
        if deleted > 0:
            con.execute("PRAGMA wal_checkpoint(TRUNCATE)")
        con.commit()
        con.close()
        return deleted
    except Exception as e:
        import logging
        logging.getLogger(__name__).warning("cleanup_old_recommendation_snapshots: %s", e)
        return 0


def save_signal_audit(
    signal_id: str,
    symbol: str,
    asset_type: str,
    horizon: str,
    composite_score: float,
    confidence_score: float,
    conviction_grade: str,
    factor_scores_json: str = "",
    gate_results_json: str = "",
    technical_signals_json: str = "",
    metadata_json: str = "",
    flags_json: str = "",
    rejection_reason: Optional[str] = None,
    price_at_signal: Optional[float] = None,
) -> int:
    """Save a signal audit record for the hybrid screener."""
    try:
        con = _conn()
        cur = con.cursor()
        cur.execute(
            """INSERT INTO signal_audit(
                signal_id, symbol, asset_type, horizon, composite_score, confidence_score,
                conviction_grade, factor_scores_json, gate_results_json, technical_signals_json,
                metadata_json, flags_json, rejection_reason, price_at_signal, created_ts
            ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            (
                str(signal_id), str(symbol), str(asset_type), str(horizon),
                float(composite_score), float(confidence_score), str(conviction_grade),
                str(factor_scores_json or ""), str(gate_results_json or ""),
                str(technical_signals_json or ""), str(metadata_json or ""),
                str(flags_json or ""), str(rejection_reason) if rejection_reason else None,
                float(price_at_signal) if price_at_signal else None, now_ts(),
            ),
        )
        audit_id = int(cur.lastrowid)
        con.commit()
        con.close()
        return audit_id
    except Exception as e:
        logger.warning("save_signal_audit failed: %s", e)
        return -1


def list_signal_audits(
    symbol: str = "",
    conviction_grade: str = "",
    limit: int = 100,
    since_ts: int = 0,
) -> List[Dict[str, Any]]:
    """Query signal audit records."""
    try:
        con = _conn()
        where = ["created_ts > ?"]
        params: list = [since_ts]
        if symbol:
            where.append("symbol = ?")
            params.append(symbol)
        if conviction_grade:
            where.append("conviction_grade = ?")
            params.append(conviction_grade)
        where_str = " AND ".join(where)
        params.append(limit)
        rows = con.execute(
            f"SELECT * FROM signal_audit WHERE {where_str} ORDER BY created_ts DESC LIMIT ?",
            params,
        ).fetchall()
        con.close()
        return [dict(r) for r in rows]
    except Exception as e:
        logger.warning("list_signal_audits failed: %s", e)
        return []


def cleanup_old_signal_audits(keep_days: int = 14) -> int:
    """Delete signal audit records older than keep_days."""
    try:
        con = _conn()
        cutoff = int(time.time()) - (keep_days * 86400)
        cur = con.execute("DELETE FROM signal_audit WHERE created_ts < ?", (cutoff,))
        deleted = cur.rowcount
        con.commit()
        con.close()
        return deleted
    except Exception as e:
        logger.warning("cleanup_old_signal_audits: %s", e)
        return 0


def log_audit(action: str, details: str = "", ip: str = "") -> None:
    """Log an audit event for security and compliance tracking."""
    try:
        con = _conn()
        con.execute(
            "INSERT INTO audit_log(timestamp, action, details, ip) VALUES (?, ?, ?, ?)",
            (time.time(), str(action), str(details) if details else None, str(ip) if ip else None),
        )
        con.commit()
        con.close()
    except Exception as e:
        logger.warning(f"Failed to log audit event: {e}")


def record_trade_feedback(symbol: str, features_json: str = "", profitable: int = 0) -> None:
    """Record trade outcome for ML model feedback and learning."""
    try:
        con = _conn()
        con.execute(
            "INSERT INTO trade_feedback(symbol, timestamp, features_json, profitable) VALUES (?, ?, ?, ?)",
            (str(symbol), time.time(), str(features_json) if features_json else None, int(profitable)),
        )
        con.commit()
        con.close()
    except Exception as e:
        logger.warning(f"Failed to record trade feedback: {e}")


def get_trade_feedback(symbol: str = "", profitable: Optional[int] = None, limit: int = 1000) -> List[Dict[str, Any]]:
    """Get recorded trade feedback for ML training."""
    try:
        con = _conn()
        query = "SELECT * FROM trade_feedback WHERE 1=1"
        params = []
        if symbol:
            query += " AND symbol = ?"
            params.append(str(symbol))
        if profitable is not None:
            query += " AND profitable = ?"
            params.append(int(profitable))
        query += " ORDER BY timestamp DESC LIMIT ?"
        params.append(int(limit))
        rows = con.execute(query, params).fetchall()
        con.close()
        return [dict(r) for r in rows]
    except Exception as e:
        logger.warning(f"Failed to get trade feedback: {e}")
        return []


# ──────────────────────────────────────────────────────────────────────────────
# Scheduled DB maintenance — prune tables that grow unbounded
# ──────────────────────────────────────────────────────────────────────────────

def cleanup_old_bot_logs(keep_days: int = 30) -> int:
    """Delete bot_logs entries older than keep_days. Returns count deleted."""
    try:
        con = _conn()
        cutoff = int(time.time()) - (keep_days * 86400)
        cur = con.execute("DELETE FROM bot_logs WHERE ts < ?", (cutoff,))
        deleted = cur.rowcount
        if deleted > 0:
            con.execute("PRAGMA wal_checkpoint(TRUNCATE)")
        con.commit()
        con.close()
        return deleted
    except Exception as e:
        logger.warning("cleanup_old_bot_logs: %s", e)
        return 0


def cleanup_old_strategy_decisions(keep_days: int = 30) -> int:
    """Delete strategy_decisions entries older than keep_days. Returns count deleted."""
    try:
        con = _conn()
        cutoff = int(time.time()) - (keep_days * 86400)
        cur = con.execute("DELETE FROM strategy_decisions WHERE ts < ?", (cutoff,))
        deleted = cur.rowcount
        if deleted > 0:
            con.execute("PRAGMA wal_checkpoint(TRUNCATE)")
        con.commit()
        con.close()
        return deleted
    except Exception as e:
        logger.warning("cleanup_old_strategy_decisions: %s", e)
        return 0


def cleanup_old_explore_signal_outcomes(keep_days: int = 90) -> int:
    """Delete explore_signal_outcomes entries older than keep_days. Returns count deleted."""
    try:
        con = _conn()
        cutoff = int(time.time()) - (keep_days * 86400)
        cur = con.execute("DELETE FROM explore_signal_outcomes WHERE signal_ts < ?", (cutoff,))
        deleted = cur.rowcount
        if deleted > 0:
            con.execute("PRAGMA wal_checkpoint(TRUNCATE)")
        con.commit()
        con.close()
        return deleted
    except Exception as e:
        logger.warning("cleanup_old_explore_signal_outcomes: %s", e)
        return 0


def cleanup_old_order_events(keep_days: int = 90) -> int:
    """Delete order_events entries older than keep_days. Returns count deleted."""
    try:
        con = _conn()
        cutoff = int(time.time()) - (keep_days * 86400)
        cur = con.execute("DELETE FROM order_events WHERE ts < ?", (cutoff,))
        deleted = cur.rowcount
        if deleted > 0:
            con.execute("PRAGMA wal_checkpoint(TRUNCATE)")
        con.commit()
        con.close()
        return deleted
    except Exception as e:
        logger.warning("cleanup_old_order_events: %s", e)
        return 0


def cleanup_old_regime_snapshots(keep_days: int = 30) -> int:
    """Delete regime_snapshots entries older than keep_days. Returns count deleted."""
    try:
        con = _conn()
        cutoff = int(time.time()) - (keep_days * 86400)
        cur = con.execute("DELETE FROM regime_snapshots WHERE ts < ?", (cutoff,))
        deleted = cur.rowcount
        if deleted > 0:
            con.execute("PRAGMA wal_checkpoint(TRUNCATE)")
        con.commit()
        con.close()
        return deleted
    except Exception as e:
        logger.warning("cleanup_old_regime_snapshots: %s", e)
        return 0


def cleanup_old_trade_feedback(keep_days: int = 180) -> int:
    """Delete trade_feedback entries older than keep_days (keep for ML). Returns count deleted."""
    try:
        con = _conn()
        cutoff = time.time() - (keep_days * 86400)
        cur = con.execute("DELETE FROM trade_feedback WHERE timestamp < ?", (cutoff,))
        deleted = cur.rowcount
        if deleted > 0:
            con.execute("PRAGMA wal_checkpoint(TRUNCATE)")
        con.commit()
        con.close()
        return deleted
    except Exception as e:
        logger.warning("cleanup_old_trade_feedback: %s", e)
        return 0


def run_db_vacuum() -> bool:
    """Run VACUUM to reclaim disk space after bulk deletes. Returns True on success."""
    try:
        con = _conn()
        con.execute("VACUUM")
        con.close()
        logger.info("DB VACUUM completed successfully")
        return True
    except Exception as e:
        logger.warning("run_db_vacuum: %s", e)
        return False


def db_maintenance_cleanup() -> Dict[str, int]:
    """
    Run all periodic cleanup jobs. Returns dict of {table: rows_deleted}.
    Safe to call daily — each function is idempotent.
    """
    results: Dict[str, int] = {}
    jobs = [
        ("bot_logs_7d",               lambda: cleanup_old_bot_logs(7)),
        ("strategy_decisions_7d",     lambda: cleanup_old_strategy_decisions(7)),
        ("regime_snapshots_30d",      lambda: cleanup_old_regime_snapshots(30)),
        ("order_events_90d",          lambda: cleanup_old_order_events(90)),
        ("explore_signal_outcomes_30d", lambda: cleanup_old_explore_signal_outcomes(30)),
        ("recommendation_snapshots_3d", lambda: cleanup_old_recommendation_snapshots(3)),
        ("signal_audit_7d",           lambda: cleanup_old_signal_audits(7)),
        ("portfolio_snapshots_30d",   lambda: cleanup_old_portfolio_snapshots(30)),
        ("trade_feedback_180d",       lambda: cleanup_old_trade_feedback(180)),
    ]
    total_deleted = 0
    for label, fn in jobs:
        try:
            n = fn()
            results[label] = n
            total_deleted += n
        except Exception as e:
            logger.warning("db_maintenance_cleanup %s: %s", label, e)
            results[label] = -1
    try:
        run_db_vacuum()
        results["vacuum"] = 1
    except Exception as e:
        logger.warning("VACUUM failed: %s", e)
    logger.info("db_maintenance_cleanup: total_deleted=%d results=%s", total_deleted, results)
    return results
