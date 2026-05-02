# DB Writer Audit — Phase 1.1

**Date:** 2026-05-02
**Scope:** Every persistent SQLite write in the repo. Reads are out of scope unless they share a transaction with a write.
**Goal:** Identify every writer that currently bypasses one of the two required guards
(`_make_real_conn` / WAL+busy_timeout pragmas, and `bot_db_lock`) so Phase 1.2/1.3 can wrap them through a single chokepoint (`db.write_txn`).

---

## 1. Existing primitives in this repo

| Primitive | Defined at | Behaviour | Notes |
| --- | --- | --- | --- |
| `_make_real_conn()` | `db.py:76` | Fresh `sqlite3.connect(DB_NAME, timeout=30)` with `journal_mode=WAL`, `busy_timeout=30000`, `synchronous=NORMAL`, `foreign_keys=ON`, `cache_size=-65536`, `temp_store=MEMORY`, `mmap_size=256MB`. Issues `PRAGMA wal_checkpoint(PASSIVE)` once per connection. | Canonical real connection. |
| `_conn()` | `db.py:100` | Per-thread cached `_NoCloseConn` wrapper around a connection produced by `_make_real_conn()`. `.close()` is a no-op (thread-local pooling). Recreates if the cached conn fails a `SELECT 1` liveness ping. | Used by ~95% of `db.py` writers. PRAGMAs ARE present (because the underlying conn is `_make_real_conn`). |
| `_db_retry(fn, *args, _retries=5)` | `db.py:120` | Exponential backoff (0.1s → 2s cap) on `OperationalError` containing "database is locked". | Currently used by **only 4** db.py functions (see §3). |
| `BotManager.bot_db_lock(bot_id)` | `bot_manager.py:5172` | Per-bot `threading.Lock`. Acquired by `BotRunner._sync_open_deal / _sync_close_deal / _sync_update_open_deal_entry / _sync_cancel_ghost_deal` and by `BotManager.manual_close_open_deal`. | Only serialises within the worker process; does not span threads that bypass the wrappers. |
| `BotRunner._bot_db_sync()` | `bot_manager.py:546` | `contextmanager` returning `bot_db_lock(self.bot_id)`. Used by 4 `_sync_*` helpers. | The only path in BotRunner where a bot lock is taken before a DB write. |
| `manual_close_deal_and_journal` | `db.py:1990` | Uses `_make_real_conn()` directly (not pooled). Race-safe `UPDATE ... WHERE state='OPEN'`. Documents that the caller must hold `bot_db_lock`. | The single write site that today uses `_make_real_conn` outside of `_conn()`. |

**There is no `write_txn()` chokepoint yet, no `DBLockedError`, and no global write lock.** Phase 1.3 will introduce them.

---

## 2. Methodology

Searches run from repo root (PowerShell-safe, ripgrep through Cursor's Grep tool):

```text
rg --type py "sqlite3\.connect\("
rg --type py "^import sqlite3|^from sqlite3"
rg --type py -i "INSERT INTO|UPDATE\s+\w+\s+SET|DELETE FROM|REPLACE INTO|INSERT OR REPLACE|INSERT OR IGNORE"
rg --type py -i "CREATE TABLE|ALTER TABLE|DROP TABLE|CREATE INDEX|DROP INDEX"
rg --type py "\.commit\(\)"
rg --type py "\.executemany\("
rg --type py "_make_real_conn|bot_db_lock|_db_retry|write_txn|DBLockedError"
```

Repo-wide write inventory (count of `INSERT/UPDATE/DELETE/REPLACE` SQL keyword hits):

| File | Write SQL hits | Has `import sqlite3`? | Uses `_conn()` / `_make_real_conn`? | Uses `_db_retry`? | Uses `bot_db_lock`? |
| --- | --- | --- | --- | --- | --- |
| `db.py` | 86 | yes | both | 4 sites | no (db.py is unaware of the lock by design — caller's job) |
| `worker_api.py` | 11 | yes | `_conn()` | no | no |
| `notification_manager.py` | 8 | no (uses `from db import _conn`) | `_conn()` | no | no |
| `scripts/migrate_auto_restart.py` | 1 (+1 ALTER) | yes | raw `sqlite3.connect` | no | no |
| `ml_signal_scorer.py` | 1 | no (uses `from db import get_db`) | unknown — see §4.5 | no | no |
| `execution_quality_tracker.py` | 1 | no (uses `from db import _conn`) | `_conn()` | no | no |
| `tax_optimizer.py` | 1 | no (uses `from db import _conn`) | `_conn()` | no | no |
| `sector_rotation.py` | 1 | no (uses `from db import _conn`) | `_conn()` | no | no |
| `one_server.py` | 0 | no | n/a | n/a | n/a |
| `app.py` | 0 | no | n/a | n/a | n/a |
| `bot_manager.py` | 0 (consumer of db.py) | no | uses `_make_real_conn` for one read at line 587 | n/a | yes (defines the lock) |
| `one_server_v2.py` | 0 (read-only) | yes | raw `sqlite3.connect` | no | no |
| `check_db.py`, `check_db_schema.py`, `check_schema.py`, `check_bots.py`, `check_bots_count.py`, `enable_bot.py`, `audit_system.py` | 0 (read-only or local dev only) | yes | raw `sqlite3.connect` | no | no |

Total `.commit()` callers across the repo: **85** — 69 in db.py, the rest in the files above.

---

## 3. db.py — function-by-function inventory

`_db_retry` is used by exactly four functions:

| Function | Line | bot-id-scoped? | conn | retry | bot_db_lock by callers |
| --- | --- | --- | --- | --- | --- |
| `save_recommendation_snapshot` | 3343 | no (global table) | `_conn()` | yes | n/a |
| `mark_explore_signals_pending` | 3401 | no (global) | `_conn()` | yes | n/a |
| `mark_explore_horizon_pending` | 3420 | no (global) | delegates to above | yes | n/a |
| `upsert_explore_feed_row` | 3425 | no (global) | `_conn()` | yes | n/a |
| `manual_close_deal_and_journal` | 1990 | yes (`bot_id`) | `_make_real_conn()` (own conn) + race-safe predicate | no (single attempt) | **yes** — caller `BotManager.manual_close_open_deal` (`bot_manager.py:5202`) |

Every other writer in `db.py` uses raw `_conn()` with **no `_db_retry`** and relies on the
caller to hold `bot_db_lock` if applicable. The full inventory follows. Risk levels:

- **H (High):** bot-tick hot path, runs once or more per second per bot, contends with the runner's other writes for the same bot row.
- **M (Medium):** runs from background loops or scanners, contends with bot writers but on different rows.
- **L (Low):** init / migration / one-shot / cron cleanup; minimal contention.

### 3.1 Bot-id-scoped writers (require `bot_db_lock`)

| Function | Line | Risk | Conn | Retry | Lock at callers? | Action for Phase 1.2/1.3 |
| --- | --- | --- | --- | --- | --- | --- |
| `add_log` | 1299 | **H** | `_conn()` | no | **NO** — called all over (BotRunner ticks, autopilot, executor, strategies, …); never wrapped in `bot_db_lock`. Same bot's runner thread also writes `bot_logs` indirectly via other helpers. | Route through `write_txn(bot_id=bot_id, fn=...)`. The bug currently causes "database is locked" loops on bot 1 because UPDATE+INSERT inside one transaction races with concurrent `add_log` from other threads. |
| `add_order_event` | 3241 | **H** | `_conn()` | no | **NO** | Route through `write_txn(bot_id, fn)`. Called from executor on every order intent/place/fill. |
| `open_deal` | 1822 | **H** | `_conn()` | no | yes when called via `BotRunner._sync_open_deal` (560); **direct callers in api/scripts bypass the lock** | Route through `write_txn(bot_id, fn)`. Drop the `_sync_*` wrappers (or keep them; `write_txn` will be idempotent re-entrant via lock). |
| `update_open_deal_entry` | 1836 | **H** | `_conn()` | no | yes via `BotRunner._sync_update_open_deal_entry` (560); **direct callers bypass** | `write_txn(bot_id, fn)`. |
| `close_deal` | 1870 | **H** | `_conn()` | no | yes via `BotRunner._sync_close_deal` (552); **direct callers bypass** | `write_txn(bot_id, fn)`. Also: this function calls `record_trade_feedback` and `_record_recommendation_outcome` inside the same conn — keep them in the same txn. |
| `cancel_ghost_deal` | 2486 | **H** | `_conn()` | no | yes via `BotRunner._sync_cancel_ghost_deal` (570); **direct callers bypass** | `write_txn(bot_id, fn)`. |
| `manual_close_deal_and_journal` | 1990 | **H** | `_make_real_conn()` (own conn) | no | **YES** — `BotManager.manual_close_open_deal` (`bot_manager.py:5202`) | Already correct in pattern. Migrate to `write_txn(bot_id, fn)` for consistency; semantics unchanged. |
| `add_regime_snapshot` | 2906 | M | `_conn()` | no | no | `write_txn(bot_id, fn)`. Per-bot regime updates collide with the bot's own writes. |
| `add_strategy_decision` | 2946 | M | `_conn()` | no | no | `write_txn(bot_id, fn)`. |
| `add_strategy_trade` | 2959 | M | `_conn()` | no | no | `write_txn(bot_id, fn)`. |
| `save_perf_metrics` | 3279 | M | `_conn()` | no | no | `write_txn(bot_id, fn)`. |
| `update_bot` | 1536 | M | `_conn()` | no | no | `write_txn(bot_id, fn)`. The bot-edit POST in `worker_api.py:5466` calls this; if the runner thread is mid-write, this races. **Also implicated in the auto_restart=0 regression — see §6.** |
| `set_bot_enabled` | 1688 | M | `_conn()` | no | no | `write_txn(bot_id, fn)`. |
| `set_bot_running` | 1698 | M | `_conn()` | no | no | `write_txn(bot_id, fn)`. |
| `delete_bot` | 1657 | M | `_conn()` | no | no | `write_txn(bot_id, fn)`. Should also acquire the lock to keep the runner thread from writing into a partially-deleted row set. |
| `patch_bot_risk_after_create` | 1504 | L | `_conn()` | no | n/a (right after create) | `write_txn(bot_id, fn)`. |
| `update_ml_prediction_outcome` | 4545 | L | `_conn()` | no | no | `write_txn(bot_id=None, fn)` (ml_predictions has bot_id but is not on the tick path). |
| `link_recommendation_to_bot` | 4076 | L | `_conn()` | no | no | `write_txn(bot_id, fn)`. |

### 3.2 Global / cross-bot writers (require global write lock, not per-bot)

| Function | Line | Risk | Conn | Retry | Action |
| --- | --- | --- | --- | --- | --- |
| `set_setting` | 1722 | M | `_conn()` | no | `write_txn(bot_id=None, fn)`. |
| `save_autopilot_config` | 1759 | L | `_conn()` | no | `write_txn(None, fn)`. |
| `update_bots_by_type` | 1643 | M | `_conn()` | no | `write_txn(None, fn)`. Touches all bots → bypass per-bot locking, use global. |
| `create_bot` | 1440 | L | `_conn()` | no | `write_txn(None, fn)`. New bot has no runner yet. |
| `add_autopilot_audit_log` | 1399 | L | `_conn()` | no | `write_txn(None, fn)`. |
| `log_data_quality` | 1350 | L | `_conn()` | no | `write_txn(None, fn)`. |
| `log_error` | 1384 | L | `_conn()` | no | `write_txn(None, fn)`. Note: takes optional `bot_id` — when present, prefer `write_txn(bot_id, fn)`. |
| `log_audit` | 5025 | L | `_conn()` | no | `write_txn(None, fn)`. |
| `record_trade_feedback` | 5039 | L | `_conn()` | no | Currently called from `close_deal` and `manual_close_deal_and_journal`; **ensure it stays inside the same outer transaction** to avoid double-locking. Standalone callers go through `write_txn(None, fn)`. |
| `mark_explore_signals_pending` | 3401 | M | `_conn()` | **yes** | Promote to `write_txn(None, fn)`. Retry behaviour will be handled there. |
| `mark_explore_horizon_pending` | 3420 | M | delegates | yes | n/a (delegates). |
| `upsert_explore_feed_row` | 3425 | M | `_conn()` | **yes** | `write_txn(None, fn)`. |
| `save_recommendation_snapshot` | 3343 | M | `_conn()` | **yes** | `write_txn(None, fn)`. |
| `save_signal_outcome` | 3711 | M | `_conn()` | **NO** | `write_txn(None, fn)`. **Was missing retry — actively reproducible source of "database is locked" during scans.** |
| `update_explore_signal_outcome` | 3765 | M | `_conn()` | **NO** | `write_txn(None, fn)`. **Explicitly called out by user — was the "explore outcome writes lock the DB" suspect.** |
| `save_explore_backtest_results` | 3631 | L | `_conn()` | no | `write_txn(None, fn)`. |
| `delete_recommendations_for_blocklist` | 3998 | L | `_conn()` | no | `write_txn(None, fn)`. |
| `cleanup_invalid_scores` | 4027 | L | `_conn()` | no | `write_txn(None, fn)`. |
| `_record_recommendation_outcome` | 4114 | M | accepts conn | no | Keep "accepts conn" form — it must execute inside the caller's `write_txn`. Callers (`close_deal`, `manual_close_deal_and_journal`) must pass through their own conn. |
| `save_perf_metrics` | 3279 | M | `_conn()` | no | listed above (bot-id). |
| `save_backtest_run` | 3289 | L | `_conn()` | no | `write_txn(None, fn)`. |
| `save_intraday_pattern` | 4637 | L | `_conn()` | no | `write_txn(None, fn)`. |
| `save_ml_model_version` | 4624 | L | `_conn()` | no | `write_txn(None, fn)`. **Note duplicate of `ml_signal_scorer.py:584` (`_log_version_to_db`) — see §4.5.** |
| `_save_ml_prediction` (and helpers around 4500–4540) | ~4500 | L | `_conn()` | no | `write_txn(None, fn)`. |
| `mark_watchlist_triggered` | 4848 | L | `_conn()` | no | `write_txn(None, fn)`. |
| `remove_watchlist_entry` | 4861 | L | `_conn()` | no | `write_txn(None, fn)`. |
| `cleanup_old_watchlist` | 4872 | L | `_conn()` | no | `write_txn(None, fn)`. |
| `cleanup_old_portfolio_snapshots` | 4887 | L | `_conn()` | no | `write_txn(None, fn)`. |
| `cleanup_old_recommendation_snapshots` | 4910 | L | `_conn()` | no | `write_txn(None, fn)`. |
| `cleanup_old_signal_audits` | 5010 | L | `_conn()` | no | `write_txn(None, fn)`. |
| `cleanup_old_bot_logs` | 5079 | M | `_conn()` | no | `write_txn(None, fn)`. **Risk: this DELETEs from `bot_logs` — same table BotRunner writes via `add_log`. Must use global write lock.** |
| `cleanup_old_strategy_decisions` | 5096 | L | `_conn()` | no | `write_txn(None, fn)`. |
| `cleanup_old_explore_signal_outcomes` | 5113 | L | `_conn()` | no | `write_txn(None, fn)`. |
| `cleanup_old_order_events` | 5130 | M | `_conn()` | no | `write_txn(None, fn)`. **Risk: same as bot_logs — DELETEs from `order_events` while bots write to it.** |
| `cleanup_old_regime_snapshots` | 5147 | L | `_conn()` | no | `write_txn(None, fn)`. |
| `cleanup_old_trade_feedback` | 5164 | L | `_conn()` | no | `write_txn(None, fn)`. |
| `db_vacuum` | 4751 | L | `_conn()` | no | Special: `VACUUM` cannot run inside a transaction. Keep outside `write_txn` but acquire a global write lock and pause the WAL checkpoint thread (Phase 1.4) for its duration. |
| `db_analyze` | 4761 | L | `_conn()` | no | `write_txn(None, fn)`. |

### 3.3 Schema / migrations / init

| Function | Line | Risk | Notes |
| --- | --- | --- | --- |
| `init_db` | 220 | L | Runs at startup. Many `CREATE TABLE IF NOT EXISTS` + `ALTER TABLE` statements. Currently uses a `_conn()` cursor and commits in chunks. Single-threaded at boot, so safe — but should still be migrated to `write_txn(None, fn)` so it participates in the global lock and benefits from retry if init runs concurrently with anything (e.g., scripts). |
| `_ensure_column` | 149 | L | Helper; called from `init_db`. Same treatment. |
| `_migrate_explore_signals_to_v2` | 164 | L | Helper; runs inside init. |
| `db.py:786` (`cur.executemany(...)`) | 786 | L | Used during init to bulk-insert seed rows. |

---

## 4. Writers outside db.py

### 4.1 `worker_api.py` — 2 raw writers

| Site | Line | Conn | Retry | Lock | Risk | Action |
| --- | --- | --- | --- | --- | --- | --- |
| `_screener_outcomes_loop` (background thread) — `INSERT OR IGNORE INTO recommendation_performance` | 4077 | `_conn()` from inline `from db import _conn` | no | no | M | Replace with new `db.write_txn(None, fn)` — cleaner, retries on lock. Currently a silent suppression (`except Exception: pass`) that hides "database is locked" errors during scans. |
| `_portfolio_loop` (background thread) — `INSERT INTO portfolio_snapshots` | 4244 | `_conn()` from inline import | no | no | M | Replace with `write_txn(None, fn)`. Today wrapped in a bare `try/except: pass` — violates the brief's rule #4 (no silent fallbacks). |

The other 9 worker_api hits in `rg "INSERT INTO|UPDATE\s+\w+\s+SET|DELETE FROM"` are docstring/comment occurrences (e.g. line 5388 "Update bot settings.") — verified: **only 2 actual write statements** in worker_api.py.

### 4.2 `notification_manager.py` — 8 writers

All 8 sites follow the same anti-pattern:

```python
from db import _conn, now_ts
con = _conn()
con.execute("INSERT INTO notifications(...) VALUES (?, ?, ?, ?, ?, 0)", (...))
con.commit()
con.close()
```

| Function | Line | Risk | Action |
| --- | --- | --- | --- |
| `insert_notification` | 207 | M | `write_txn(None, fn)` |
| `mark_notification_read` | 250 | L | `write_txn(None, fn)` |
| `notify_trade_executed` | 291 | M | `write_txn(None, fn)` (bot_id is in `bot_name` only — DB row stores `bot_id=NULL`) |
| `notify_take_profit` | 338 | M | `write_txn(None, fn)` |
| `notify_stop_loss` | 382 | M | `write_txn(None, fn)` |
| `notify_bot_error` | 421 | M | `write_txn(None, fn)` |
| `notify_drawdown_alert` | 460 | L | `write_txn(None, fn)` |
| (one more between 500-510 — same shape) | 509 | L | `write_txn(None, fn)` |

Risk M because these run from arbitrary worker/strategy/runner threads concurrent with bot ticks; they all contend for the same SQLite write lock. They also violate brief rule #4 — every site has `except Exception: logger.debug(..., e); return False/-1`, which hides DB locks as silent failures.

### 4.3 `execution_quality_tracker.py` — 1 writer

| Function | Line | Risk | Action |
| --- | --- | --- | --- |
| `record_execution` | 41 | M | `write_txn(None, fn)`. Called from executor on every fill. Currently `except Exception: logger.debug(...); return False` — silent failure, must be tightened per rule #4 (log at WARN, propagate or set typed error). |

### 4.4 `tax_optimizer.py` — 1 writer

| Function | Line | Risk | Action |
| --- | --- | --- | --- |
| `save_tax_harvest_suggestion` | 37 | L | `write_txn(None, fn)`. Off by default (`ENABLE_TAX_HARVESTING=0`). |

### 4.5 `sector_rotation.py` — 1 writer

| Function | Line | Risk | Action |
| --- | --- | --- | --- |
| `record_sector_performance` | 43 | L | `write_txn(None, fn)`. |

### 4.6 `ml_signal_scorer.py` — 1 writer + a bug

| Function | Line | Risk | Action |
| --- | --- | --- | --- |
| `_log_version_to_db` | 589 | L | **BUG:** uses `from db import get_db` and calls `conn = get_db()`. `db.py` does **not** export `get_db()`. This call will raise `ImportError` and is caught by the outer `except Exception: logger.debug(...)`. Net effect: the function silently does nothing. |

Action: replace with `db.save_ml_model_version(...)` (already exists at `db.py:4624`), routed through `write_txn(None, fn)` after Phase 1.3. Drop the inline write entirely.

### 4.7 `autopilot.py`

The grep produced only mentions of the word `upsert` (`from db import upsert_watchlist_entry`, line 1078), not raw write statements. autopilot calls into `db.py` exclusively — **no direct DB writes**. Once `db.upsert_watchlist_entry` is migrated to `write_txn`, autopilot is automatically covered.

### 4.8 `scripts/migrate_auto_restart.py`

| Site | Line | Conn | Retry | Lock | Risk | Action |
| --- | --- | --- | --- | --- | --- | --- |
| Schema add: `ALTER TABLE bots ADD COLUMN auto_restart INTEGER NOT NULL DEFAULT 1` | 66 | raw `sqlite3.connect(db_path, timeout=30)`, sets `journal_mode=WAL` and `busy_timeout=30000` after open | no | no | L | This is a one-shot CLI; called by `deploy.ps1` BEFORE the service restart. If the service is up, this runs concurrently and may collide. **Mitigation:** keep it standalone but route the connection setup through the same helper that produces a `_make_real_conn`-equivalent. Phase 1.3 will export a small `db.open_migration_conn()` helper that mirrors `_make_real_conn` so migrations stop drifting from the canonical pragmas. |
| Data update: `UPDATE bots SET auto_restart=1 WHERE auto_restart=0 OR auto_restart IS NULL` | 78 | same conn | no | no | L | Same. |

**Open question (Phase 2.5 not Phase 1.1):** the user reports `/api/bots` still returns `auto_restart=0` after this script claimed success. The `_ov` helper in `worker_api.py:5411` uses `b.get(key, default)` when payload omits the key, so the previous DB value persists — but the bot edit form likely re-submits `auto_restart=0` explicitly (most form serialisers send unchecked checkboxes as `0` rather than omitting). Will be confirmed in Phase 2.5 with an end-to-end test that POSTs the form and asserts the row stays at 1.

### 4.9 `one_server_v2.py`

| Site | Line | Type | Risk |
| --- | --- | --- | --- |
| `_conn()` defined at | 94 | raw `sqlite3.connect(_db_path(), timeout=30.0)` — **no PRAGMAs** | M if invoked |

There are no `INSERT/UPDATE/DELETE/REPLACE` callers in this file — read-only. **However** the connection has no `journal_mode=WAL`, no `busy_timeout`. If a future refactor uses this `_conn()` for a write, or if reads from this conn block writers (long `SELECT` in unreleased mode), trouble starts. Per CLAUDE.md the live service runs `one_server.py` not `one_server_v2.py`, so this is dormant code. **Action for Phase 1.3:** delete this `_conn()` and import the canonical `db._conn` (or `db.write_txn` for any future write).

### 4.10 Diagnostic / one-shot scripts (low risk)

| File | Pattern | Disposition |
| --- | --- | --- |
| `check_db.py`, `check_db_schema.py`, `check_schema.py`, `check_bots.py`, `check_bots_count.py`, `enable_bot.py`, `audit_system.py` | raw `sqlite3.connect("botdb.sqlite3")` | Local CLI; user-invoked. Not run in production. Will re-route to `db.open_migration_conn()` in Phase 1.3 to keep PRAGMAs consistent, but they are not contributing to the production lock storms. |

---

## 5. Crashed-loop diagnosis (live bug "Fatal error: OperationalError: database is locked" on bot 1)

Most likely root causes for the **persistent** "database is locked" loop on bot 1, ranked by likelihood:

1. **`add_log` re-entrancy under contention.** The runner tick logs heavily; `cleanup_old_bot_logs` (Phase 1.3 cron) DELETEs from the same table; explore scans on other threads `_db_retry`-loop on `upsert_explore_feed_row` while holding the writer slot. Because `add_log` itself does not retry and does not hold `bot_db_lock`, even a brief contention window throws `OperationalError` straight to the supervisor → `BotRunner` exits → supervisor restarts → repeat. Fix: route `add_log` through `write_txn(bot_id, fn)` with retry.
2. **`cleanup_old_bot_logs` / `cleanup_old_order_events` mass DELETE under WAL.** A mass DELETE holds the writer for hundreds of ms; concurrent `add_log` / `add_order_event` see SQLITE_BUSY → `OperationalError`. Fix: same as above (retry + global lock).
3. **`update_bot` from API edit racing the runner.** A user editing the bot in the UI fires `UPDATE bots SET ...` while the runner tick is mid-write. With no shared lock, contention can throw. Fix: bot-id-scoped `write_txn` in both writers.
4. **`save_signal_outcome` / `update_explore_signal_outcome` lacking retry.** Explored to confirm — these are the explicit "neither" candidates the user named in Phase 1.2 ("update_explore_signal_outcome", "mark_explore_signals_pending"). `mark_explore_signals_pending` already has retry; the outcome writers do not. Fix: retry via `write_txn`.

Phase 1.5's load test will model exactly this scenario (4 writer threads × 200 mixed ops × 60s) and gate Phase 1 close.

---

## 6. Out-of-scope findings recorded for later phases

These were uncovered while auditing and will not be fixed in Phase 1.1, but are pinned here so they don't get lost:

- **`worker_api.py:5444` — `auto_restart` regression suspect.** `_ov("auto_restart", 1, ...)` falls back to `b.get(key, default)` when payload omits the key, but the bot-edit form most likely posts `auto_restart=0` for an unchecked checkbox (rather than omitting it). Net effect: the user's UI silently reverts the migration. **Phase 2.5** will add (a) a NOT NULL DEFAULT 1 constraint on the column at the schema level (idempotent migration), (b) a server-side test that POSTs the edit form with a missing `auto_restart` field and asserts the row remains at 1, (c) a test that POSTs `auto_restart=0` and confirms the form refuses (or coerces to 1) per the brief.
- **`worker_api.py:5459` — `hard_sl_pct` defaults to 0.0.** Same `_ov` pattern with default 0.0. Per the brief Phase 3.2, `hard_sl_pct == 0` must block live mode; the server-side default cannot be 0. **Phase 3.2** will add the validator.
- **Silent excepts everywhere.** `notification_manager.py`, `execution_quality_tracker.py`, `worker_api.py:_portfolio_loop`, `worker_api.py:_screener_outcomes_loop`, `ml_signal_scorer.py:_log_version_to_db`, `tax_optimizer.py`, `sector_rotation.py` all swallow exceptions with `logger.debug(...); return False/None`. Per brief rule #4, every `except` must `logger.exception` and propagate or set a typed flag. **Phase 1.3 will tighten the writers we touch; everything else moves to Phase 2.** Tracked here so we don't forget.
- **`ml_signal_scorer.py:_log_version_to_db` calls non-existent `db.get_db()`.** Currently a silent no-op. Replace with `db.save_ml_model_version` in Phase 1.2.
- **`one_server_v2.py:94` opens a connection without WAL/busy_timeout pragmas.** Dormant today (not the live entrypoint), but a landmine. Replace with `db._conn`/`db.write_txn` in Phase 1.3.

---

## 7. Closing checklist for Phase 1.1

- [x] Repo-wide grep complete (commands logged in §2).
- [x] Every writer enumerated with file:line, function, conn, retry, lock status (§3, §4).
- [x] Every site classified Pending / Action documented (column "Action").
- [x] Hot-path candidates flagged H so Phase 1.5 can target them in the load test (`add_log`, `add_order_event`, `open_deal`, `close_deal`, `update_open_deal_entry`, `cancel_ghost_deal`, `manual_close_deal_and_journal`).
- [x] Out-of-scope-for-Phase-1 findings recorded so we can pick them up in 2.5/3.2 (§6).

Phase 1.1 deliverable: this document. **No code changes**; no behaviour changes. Next:
- **1.2:** Wrap each writer to use `write_txn`. One commit per writer (or per logical group: deals, logs, explore, scanners, …) so reverts are surgical.
- **1.3:** Introduce `db.write_txn`, `db.DBLockedError`, `db.open_migration_conn`, plus optional global write lock for non-bot writers.
- **1.4:** Add WAL checkpoint thread (60s `wal_checkpoint(TRUNCATE)`).
- **1.5:** Write `tests/test_db_locking.py::test_no_lock_under_load` (4 threads × 200 ops × 60s) — must pass before closing Phase 1.
- **1.6:** Deploy + 10-min `journalctl` tail proving bot 1 is no longer in the locking loop.

---

## 8. Phase 1.2 migration ledger (added at end of Phase 1.3)

This section is the **source-of-truth checklist** for the Phase 1.2a–1.2e migrations.
Every writer enumerated in §3 and §4 is listed below with its MIGRATED status,
the commit/phase it landed in, and the test that exercises it. New writers
added after Phase 1 must follow the rules in
`audit/write_txn_design.md` and append a row here.

### 8.1 The chokepoint — `db.write_txn(bot_id, fn, *, name=None) -> T`

Defined in `db.py` (Phase 1.2a). See `audit/write_txn_design.md` for the full
contract. Summary:

- `bot_id=None` → acquires `db._global_write_lock` (RLock).
- `bot_id=int` → acquires `db.bot_db_lock(bot_id)` (RLock per bot).
- Opens a fresh `_make_real_conn()`-equivalent connection (pragma-canonical).
- Runs `fn(con)`; commits on success, rolls back on any exception.
- Retries `OperationalError("database is locked")` with the schedule
  `[50, 100, 250, 500, 1000] ms ±20% jitter`; **5 attempts total** before
  raising `db.DBLockedError(sql, bot_id, attempts, elapsed_ms)`.
- Nested calls are **forbidden**; entry guarded by a `threading.local` flag.
  A nested call raises `RuntimeError`. Use the inner-conn-passthrough pattern
  for cross-function transactions (e.g. `_record_recommendation_outcome(con, ...)`).
- WAL checkpoint thread runs `PRAGMA wal_checkpoint(TRUNCATE)` every 60 s
  (held for the duration of the global write lock).

### 8.2 Bot-id-scoped writers (§3.1)

| Function | Phase | Lock | Test | Status |
| --- | --- | --- | --- | --- |
| `add_log` | 1.2b/1 | per-bot | `test_add_log_under_concurrent_contention` | MIGRATED |
| `add_order_event` | 1.2b/3 | per-bot | `test_add_order_event_under_load` | MIGRATED |
| `open_deal` | 1.2b/2 | per-bot | `test_open_deal_routes_through_per_bot_lock` | MIGRATED |
| `update_open_deal_entry` | 1.2b/2 | per-bot | `test_update_open_deal_entry_routes_through_per_bot_lock` | MIGRATED |
| `close_deal` | 1.2b/2 | per-bot | `test_close_deal_atomic_with_recommendation_outcome` | MIGRATED |
| `cancel_ghost_deal` | 1.2b/4 | per-bot | `test_cancel_ghost_deal_routes_through_per_bot_lock` | MIGRATED |
| `manual_close_deal_and_journal` | 1.2c/2 | per-bot | `test_manual_close_deal_and_journal_atomic` | MIGRATED |
| `record_trade_feedback` | 1.2b/2 | per-bot (resolved from deal) | covered via `close_deal` test | MIGRATED |
| `add_regime_snapshot` | 1.2c/2 | per-bot | `test_add_regime_snapshot_uses_per_bot_lock` | MIGRATED |
| `add_strategy_decision` | 1.2c/2 | per-bot | `test_add_strategy_decision_uses_per_bot_lock` | MIGRATED |
| `add_strategy_trade` | 1.2c/2 | per-bot | `test_add_strategy_trade_uses_per_bot_lock` | MIGRATED |
| `save_perf_metrics` | 1.2c/2 | per-bot | `test_save_perf_metrics_uses_per_bot_lock` | MIGRATED |
| `update_bot` | 1.2c/2 | per-bot | `test_update_bot_routes_through_per_bot_lock` | MIGRATED |
| `set_bot_enabled` | 1.2c/2 | per-bot | `test_set_bot_enabled_routes_through_per_bot_lock` | MIGRATED |
| `set_bot_running` | 1.2c/2 | per-bot | `test_set_bot_running_routes_through_per_bot_lock` | MIGRATED |
| `delete_bot` | 1.2c/2 | per-bot | `test_delete_bot_atomic_cascade` | MIGRATED |
| `patch_bot_risk_after_create` | 1.2c/2 | per-bot | `test_patch_bot_risk_after_create` | MIGRATED |
| `update_ml_prediction_outcome` | 1.2c/2 | global | `test_update_ml_prediction_outcome_global_lock` | MIGRATED |
| `link_recommendation_to_bot` | 1.2c/2 | per-bot | `test_link_recommendation_to_bot_uses_per_bot_lock` | MIGRATED |

### 8.3 Global / cross-bot writers (§3.2)

| Function | Phase | Lock | Test | Status |
| --- | --- | --- | --- | --- |
| `set_setting` | 1.2c/3 | global | `test_set_setting_uses_write_txn_global` | MIGRATED |
| `save_autopilot_config` | 1.2c/3 | global | `test_save_autopilot_config_uses_write_txn_global` | MIGRATED |
| `update_bots_by_type` | 1.2c/3 | global | (covered by `test_global_writers_under_concurrent_load`) | MIGRATED |
| `create_bot` | 1.2c/3 | global | `test_create_bot_routes_through_write_txn_global` | MIGRATED |
| `add_autopilot_audit_log` | 1.2c/3 | global | (smoke via load test) | MIGRATED |
| `log_data_quality` | 1.2c/3 | global | `test_log_data_quality_uses_write_txn_global` | MIGRATED |
| `log_error` | 1.2c/3 | global or per-bot (`bot_id` if set) | `test_log_error_uses_per_bot_when_bot_id_set` | MIGRATED |
| `log_audit` | 1.2c/3 | global | `test_log_audit_uses_write_txn_global` | MIGRATED |
| `mark_explore_signals_pending` | 1.2b/5 | global | `test_explore_writers_route_through_write_txn` | MIGRATED |
| `mark_explore_horizon_pending` | n/a (delegates) | global | (delegates) | MIGRATED |
| `upsert_explore_feed_row` | 1.2b/5 | global | `test_explore_writers_route_through_write_txn` | MIGRATED |
| `save_recommendation_snapshot` | 1.2b/5 | global | `test_explore_writers_route_through_write_txn` | MIGRATED |
| `save_signal_outcome` | 1.2b/5 | global | `test_explore_writers_route_through_write_txn` | MIGRATED |
| `update_explore_signal_outcome` | 1.2b/5 | global | `test_explore_writers_route_through_write_txn` | MIGRATED |
| `save_explore_backtest_results` | 1.2c/3 | global | (smoke) | MIGRATED |
| `delete_recommendations_for_blocklist` | 1.2b/8 | global (chunked) | `test_delete_recommendations_for_blocklist_chunked` | MIGRATED |
| `cleanup_invalid_scores` | 1.2b/8 | global | (covered by load test) | MIGRATED |
| `_record_recommendation_outcome` | 1.2b/2 | inner-conn passthrough | covered via `close_deal` test | MIGRATED (callee pattern) |
| `save_backtest_run` | 1.2c/3 | global | (smoke) | MIGRATED |
| `save_intraday_pattern` | 1.2c/3 | global | `test_save_intraday_pattern_routes_through_write_txn` | MIGRATED |
| `save_ml_model_version` | 1.2c/3 | global | `test_save_ml_model_version_routes_through_write_txn` | MIGRATED |
| `save_ml_prediction` | 1.2c/3 | global | (smoke; per-bot variant covered by `update_ml_prediction_outcome` test) | MIGRATED |
| `add_intelligence_decision` | 1.2c/3 | per-bot | `test_add_intelligence_decision_uses_per_bot_lock` | MIGRATED |
| `upsert_watchlist_entry` | 1.2c/3 | global | `test_watchlist_writers_route_through_write_txn` | MIGRATED |
| `mark_watchlist_triggered` | 1.2c/3 | global | `test_watchlist_writers_route_through_write_txn` | MIGRATED |
| `remove_watchlist_entry` | 1.2c/3 | global | `test_watchlist_writers_route_through_write_txn` | MIGRATED |
| `cleanup_old_watchlist` | 1.2b/8 | global (chunked UPDATE) | covered by chunked-cleanup load tests | MIGRATED |
| `cleanup_old_portfolio_snapshots` | 1.2b/8 | global (chunked) | (load test) | MIGRATED |
| `cleanup_old_recommendation_snapshots` | 1.2b/8 | global (chunked) | (load test) | MIGRATED |
| `cleanup_old_signal_audits` | 1.2b/8 | global (chunked) | `test_cleanup_old_signal_audits_chunked` | MIGRATED |
| `cleanup_old_bot_logs` | 1.2b/8 | global (chunked) | `test_cleanup_old_bot_logs_under_concurrent_insert_load` | MIGRATED |
| `cleanup_old_strategy_decisions` | 1.2b/8 | global (chunked) | (load test) | MIGRATED |
| `cleanup_old_explore_signal_outcomes` | 1.2b/8 | global (chunked) | (load test) | MIGRATED |
| `cleanup_old_order_events` | 1.2b/8 | global (chunked) | `test_cleanup_old_order_events_chunked_under_load` | MIGRATED |
| `cleanup_old_regime_snapshots` | 1.2b/8 | global (chunked) | (load test) | MIGRATED |
| `cleanup_old_trade_feedback` | 1.2b/8 | global (chunked) | (load test) | MIGRATED |
| `db_vacuum` | 1.2c/3 | global (special: VACUUM cannot run inside a txn — manually acquires `_global_write_lock` and uses `open_migration_conn()`) | `test_db_vacuum_does_not_use_write_txn_but_holds_global_lock` | MIGRATED (special-case) |
| `db_analyze` | 1.2c/3 | global | `test_db_analyze_routes_through_write_txn_global` | MIGRATED |
| `save_dividend_event` | 1.2c/3 | global | `test_save_dividend_event_uses_write_txn_global` | MIGRATED |
| `save_market_event` | 1.2c/3 | global | `test_save_market_event_uses_write_txn_global` | MIGRATED |
| `upsert_trade_journal` | 1.2c/3 | per-bot (resolved from deal) | `test_upsert_trade_journal_routes_through_write_txn` | MIGRATED |
| `save_scoring_calibration_log` | 1.2c/3 | global | (smoke) | MIGRATED |
| `init_db` / `_ensure_column` / `_migrate_explore_signals_to_v2` | n/a | n/a | run at boot, single-threaded; not migrated to `write_txn` (would deadlock the bootstrap path) | INTENTIONALLY UNMIGRATED — single-threaded boot path |

### 8.4 Out-of-`db.py` writers (§4)

| Module | Function(s) | Phase | Routing | Status |
| --- | --- | --- | --- | --- |
| `worker_api.py` | `_screener_outcomes_loop` (background) | 1.2c/1 | `db.write_txn(None, ...)` for inserts; network I/O moved outside the txn; loop health surfaced via `_BACKGROUND_LOOP_HEALTH` to `/health/full` | MIGRATED |
| `worker_api.py` | `_portfolio_loop` (background) | 1.2c/1 | `db.write_txn(None, ...)` for inserts; failures surface as `degraded` to `/health/full` | MIGRATED |
| `notification_manager.py` | `insert_notification`, `mark_notification_read`, `notify_trade_executed`, `notify_take_profit`, `notify_stop_loss`, `notify_bot_error`, `notify_drawdown_alert`, `notify_daily_summary` | 1.2b/6 | `_insert_notification_row` helper through `write_txn(None, ...)`; `_send_external` runs **after** the txn commits so Discord/Telegram latency cannot hold the lock | MIGRATED |
| `execution_quality_tracker.py` | `record_execution` | 1.2b/7 | `write_txn(int(bot_id), ...)`; `logger.exception` on failure | MIGRATED |
| `tax_optimizer.py` | `save_tax_harvest_suggestion` | 1.2b/7 | `write_txn(None, ...)` | MIGRATED |
| `sector_rotation.py` | `record_sector_performance` | 1.2b/7 | `write_txn(None, ...)` | MIGRATED |
| `ml_signal_scorer.py` | `_log_version_to_db` | 1.2d | calls `db.save_ml_model_version` (which is itself migrated, 1.2c/3); `logger.exception` on failure | MIGRATED (silent no-op fixed) |
| `autopilot.py` | (no direct writes — calls `db.upsert_watchlist_entry` etc.) | n/a | covered transitively | n/a — no direct writes |
| `scripts/migrate_auto_restart.py` | one-shot ALTER + UPDATE | 1.2a + 1.3 | `db.open_migration_conn()` (drops bespoke pragma drift) | MIGRATED |
| `one_server_v2.py` | `_conn()` (read-only) | 1.2e | delegates to `db._conn()` (canonical pragmas) | MIGRATED |
| `check_db.py`, `check_db_schema.py`, `check_schema.py`, `check_bots.py`, `check_bots_count.py`, `enable_bot.py`, `audit_system.py` | local dev/diagnostic CLIs | n/a | not run in production; out of scope for Phase 1. Will route through `db.open_migration_conn()` opportunistically | NOT MIGRATED — local dev tools only |

### 8.5 Out-of-scope finds (§6) — status

- **`worker_api.py:5444` — `auto_restart` regression** → tracked in
  [`audit/issues/phase-2-5-auto-restart-regression.md`](issues/phase-2-5-auto-restart-regression.md).
  Lands in Phase 2.5. **NOT FIXED in Phase 1** — outside DB-locking blast radius.
- **`worker_api.py:5459` — `hard_sl_pct` defaults to 0.0** → tracked in
  [`audit/issues/phase-3-2-hard-sl-pct-default.md`](issues/phase-3-2-hard-sl-pct-default.md).
  Lands in Phase 3.2. **NOT FIXED in Phase 1**.
- **Silent `except: pass` everywhere** → fixed for the modules touched in
  Phase 1.2 (`worker_api._screener_outcomes_loop`, `worker_api._portfolio_loop`,
  `notification_manager.*`, `execution_quality_tracker.record_execution`,
  `tax_optimizer.save_tax_harvest_suggestion`, `sector_rotation.record_sector_performance`,
  `ml_signal_scorer._log_version_to_db`). Remaining silent excepts in non-DB
  paths are tracked for Phase 2.
- **`ml_signal_scorer.py:_log_version_to_db` calls non-existent `db.get_db()`** →
  fixed in Phase 1.2d. Source-level grep regression in
  `tests/test_writer_migrations.py::test_ml_signal_scorer_no_inline_db_get_db_import`.
- **`one_server_v2.py:94` raw connection** → fixed in Phase 1.2e. Source-level
  grep regression in `tests/test_writer_migrations.py::test_one_server_v2_no_raw_sqlite_connect`.

### 8.6 New rules for future writers

1. **MUST** call `db.write_txn(bot_id, fn, *, name=...)`. No raw `_conn()` for
   writes. No raw `sqlite3.connect()` anywhere.
2. **MUST NOT** nest `write_txn` calls. If you need a multi-step transaction
   that spans helpers, make the helper accept an injected `con` and call it
   from inside one outer `write_txn`. Pattern reference:
   `db._record_recommendation_outcome(con, ...)`.
3. **MUST** pass `bot_id` when the row's lifecycle is owned by a single bot
   runner. Pass `None` for global tables (settings, audits, watchlists, etc.).
4. **MUST NOT** swallow `OperationalError` or `DBLockedError` silently.
   `write_txn` already retries; if it raises after the retry budget, the
   caller logs with `logger.exception` and surfaces a typed failure
   (return False / set state flag / raise upstream).
5. **MUST** put network I/O (Discord, Telegram, exchange calls) **after**
   `write_txn` commits. Holding the global write lock during a remote call
   re-creates the original "database is locked" symptom under load.
6. **For mass DELETEs** — use `db.chunked_delete(table, where, params, ...)`.
   Single-shot DELETEs against hot tables (bot_logs, order_events, ...) are
   forbidden. Chunked variants run 500 rows per `write_txn` and `time.sleep(0.05)`
   between batches to yield the writer slot.
7. **For one-shot CLI / migration scripts** — use `db.open_migration_conn()`,
   never raw `sqlite3.connect`. See `scripts/migrations/README.md` for the
   full checklist.
8. **Update this ledger** when adding a writer. The grep on `write_txn(`
   in `db.py` plus the count of MIGRATED rows here must stay 1:1.
