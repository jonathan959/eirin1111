"""One-shot migration: ensure every bot has auto_restart=1.

Run as part of deploy.ps1 before restarting the service. Safe to re-run.

Usage:
    python scripts/migrate_auto_restart.py [DB_PATH]

Defaults DB_PATH to ./botdb.sqlite3 (the path used by the live service).
Exit code is always 0 unless the DB file is missing or unreadable; we never
want a benign migration to fail a deploy.

Phase 1.2a: this script now uses db.open_migration_conn() so the PRAGMA
configuration (WAL, busy_timeout=30000, synchronous=NORMAL, foreign_keys=ON,
cache=64MB, mmap=256MB) stays in lock-step with worker-side connections —
fixes the drift risk flagged in audit/db_writers.md sec 4.7 and sec 5.
"""

from __future__ import annotations

import os
import sqlite3
import sys
import time


def _resolve_db_path(argv: list[str]) -> str:
    if len(argv) > 1 and argv[1].strip():
        return argv[1].strip()
    env = os.getenv("BOT_DB_PATH", "").strip()
    if env:
        return env
    return "botdb.sqlite3"


def _table_exists(conn: sqlite3.Connection, name: str) -> bool:
    cur = conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name=?",
        (name,),
    )
    return cur.fetchone() is not None


def _column_exists(conn: sqlite3.Connection, table: str, column: str) -> bool:
    cur = conn.execute(f"PRAGMA table_info({table})")
    return any(row[1] == column for row in cur.fetchall())


def main() -> int:
    db_path = _resolve_db_path(sys.argv)
    if not os.path.isfile(db_path):
        print(f"[migrate_auto_restart] DB file not found: {db_path} (skipping)")
        return 0

    print(f"[migrate_auto_restart] db={db_path}")

    # Force db.DB_NAME to the resolved path BEFORE the import so
    # open_migration_conn() targets the right database. This matters when
    # the operator passes an explicit CLI argument that diverges from
    # BOT_DB_PATH.
    os.environ["BOT_DB_PATH"] = db_path

    # Make sure scripts/ subdir doesn't shadow project root on sys.path.
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)

    try:
        import db  # noqa: E402 — must follow BOT_DB_PATH override
    except Exception as e:
        print(f"[migrate_auto_restart] FATAL: could not import db module: {e}")
        return 0  # best-effort migration

    conn = db.open_migration_conn()
    try:
        if not _table_exists(conn, "bots"):
            print("[migrate_auto_restart] 'bots' table does not exist yet (fresh DB?). Nothing to do.")
            return 0

        if not _column_exists(conn, "bots", "auto_restart"):
            print("[migrate_auto_restart] adding missing auto_restart column ...")
            try:
                conn.execute("ALTER TABLE bots ADD COLUMN auto_restart INTEGER NOT NULL DEFAULT 1")
                conn.commit()
            except Exception as e:
                print(f"[migrate_auto_restart] could not add column (may already exist): {e}")

        before = conn.execute(
            "SELECT COUNT(*) FROM bots WHERE auto_restart=0 OR auto_restart IS NULL"
        ).fetchone()[0]
        print(f"[migrate_auto_restart] rows needing fix: {before}")

        t0 = time.time()
        conn.execute(
            "UPDATE bots SET auto_restart=1 WHERE auto_restart=0 OR auto_restart IS NULL"
        )
        conn.commit()
        changed = conn.total_changes
        elapsed_ms = int((time.time() - t0) * 1000)
        print(f"[migrate_auto_restart] updated rows: {changed} (in {elapsed_ms} ms)")

        sample = conn.execute(
            "SELECT id, name, auto_restart FROM bots ORDER BY id LIMIT 10"
        ).fetchall()
        for row in sample:
            print(f"  bot id={row[0]} name={row[1]!r} auto_restart={row[2]}")

        return 0
    except Exception as e:
        print(f"[migrate_auto_restart] ERROR: {e}")
        # Migration is best-effort: do not fail the deploy on transient SQLite issues.
        return 0
    finally:
        try:
            conn.close()
        except Exception:
            pass


if __name__ == "__main__":
    sys.exit(main())
