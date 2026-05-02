# Database Migrations Index

This document is the authoritative ledger for every schema or data migration
that has ever been applied to `botdb.sqlite3` on the live host. Per
**ground rule #10** (CLAUDE.md): every migration is **idempotent**, **additive**,
and **has a rollback note**.

## Conventions

- **Idempotent** — re-running the migration on an already-migrated DB is a
  no-op and exits 0. New scripts MUST guard with `PRAGMA table_info(...)` /
  `SELECT 1 FROM sqlite_master WHERE type='table' AND name=?` before
  `ALTER TABLE` / `CREATE TABLE`.
- **Additive** — never `DROP COLUMN` (SQLite cannot do it cleanly anyway).
  Rename via "create new column → backfill → switch reads → leave the old
  column in place tagged as deprecated".
- **PRAGMAs in lock-step** — every migration script that opens its own
  SQLite connection MUST call `db.open_migration_conn()` (Phase 1.2a). This
  is the canonical fresh-connection factory: WAL, `busy_timeout=30000`,
  `synchronous=NORMAL`, `foreign_keys=ON`, 64 MB cache, 256 MB mmap. Raw
  `sqlite3.connect(...)` in `scripts/migrate_*.py` is a bug and must be
  caught in code review.
- **Best-effort exit code** — migration scripts run inside `deploy.ps1`
  before service restart. They MUST exit 0 on benign failures (e.g. table
  not present yet on a fresh DB) so a transient migration glitch never
  blocks a deploy. Hard ALTER errors should print to stderr but still
  exit 0; the caller checks the printed log for `ERROR:`.

## Pre-deploy / deploy hook

`deploy.ps1` runs every script under `scripts/migrate_*.py` against the
live DB **before** `tradingserver.service` is restarted. Each script must
take an optional `[DB_PATH]` argv so deploys can target a backup copy
without touching prod.

## Rollback strategy

1. **Backup** — `deploy.ps1` takes a fresh backup via the SQLite `.backup`
   API immediately before the migration loop runs (Phase 7.7 / 7.8). The
   backup is named `botdb.sqlite3.bak.<timestamp>` on the live host. Quick
   deploys (`deploy.ps1 -Quick`) skip the backup and disable rollback for
   that run — explicit operator opt-in only.
2. **Restore** — `deploy_restore.sh` swaps the most recent backup back into
   place atomically (move-then-replace) and restarts the service. See
   `deploy.ps1` for the exact ordering.
3. **Rollback note per migration** — every entry below explains exactly
   which DDL the rollback would undo and any data caveats.

## Migrations

### `scripts/migrate_auto_restart.py`

- **What it does:** Adds the `auto_restart INTEGER NOT NULL DEFAULT 1`
  column to `bots` if missing, then `UPDATE bots SET auto_restart=1 WHERE
  auto_restart=0 OR auto_restart IS NULL`.
- **Why:** Live host had a population of bots with `auto_restart=0` from
  an old code path; the supervisor would not auto-recover them after a
  fatal error. The default-1 behaviour matches the new lifecycle policy
  (Phase 2). The bot-edit POST handler still has a regression that can
  silently revert this to 0 — tracked in
  [`audit/issues/phase-2-5-auto-restart-regression.md`](../../audit/issues/phase-2-5-auto-restart-regression.md);
  fix lands in Phase 2.5, not here.
- **Rollback:** Idempotent and backward-compatible. The added column has
  a default of 1; older code that does not know about `auto_restart`
  simply ignores it. To force a value-only rollback (revert all rows back
  to 0): `UPDATE bots SET auto_restart=0;` — but this would break the
  supervisor's recovery semantics, so it should only ever be done as part
  of an emergency revert.
- **Phase 1.2a note:** This script previously opened its own raw
  `sqlite3.connect(db_path, timeout=30.0)` with only `journal_mode=WAL`
  and `busy_timeout=30000` — a pragma drift risk flagged in the audit
  (`audit/db_writers.md` §4.7 and §5). Phase 1.2a switched it to
  `db.open_migration_conn()` so it now sets all canonical pragmas
  (`synchronous=NORMAL`, `foreign_keys=ON`, 64 MB cache, 256 MB mmap)
  identically to worker-side connections.

## Adding a new migration (checklist)

1. New file goes under `scripts/migrate_<topic>.py`, NOT here.
2. Use `db.open_migration_conn()`, never raw `sqlite3.connect()`.
3. Wrap every `ALTER TABLE` / `CREATE TABLE` in an existence check.
4. Take an optional `[DB_PATH]` argv (default to `BOT_DB_PATH` env, then
   `./botdb.sqlite3`).
5. Exit 0 on benign failures; print `[migrate_<topic>] ERROR: ...` on
   hard failures.
6. Add a rollback entry to **this file** in the same commit.
7. Hook the script into `deploy.ps1`'s pre-restart migration loop if it
   needs to run automatically on every deploy.
