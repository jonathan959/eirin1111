#!/usr/bin/env python3
"""
DB Backup Script - for automated daily backups and disaster recovery.

Usage:
  python db_backup.py
  python db_backup.py --dest /path/to/backups/
  python db_backup.py --vacuum   # Run VACUUM before backup
  python db_backup.py --multi    # Copy to multiple locations (DB_BACKUP_DIRS)

Environment:
  DB_BACKUP_FREQUENCY - hours between backups (for cron/scheduler)
  BOT_DB_PATH - override DB path
  DB_BACKUP_DIRS - colon-separated paths for multi-location backups
"""
import argparse
import os
import sys

# Ensure project root
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dest", help="Backup directory (default: .)")
    parser.add_argument("--vacuum", action="store_true", help="Run VACUUM before backup")
    parser.add_argument("--multi", action="store_true", help="Copy to all DB_BACKUP_DIRS locations")
    args = parser.parse_args()

    from db import DB_NAME, backup_db, db_vacuum, db_analyze, now_ts

    if args.vacuum:
        print("Running VACUUM...")
        db_vacuum()
        print("VACUUM done.")
    try:
        db_analyze()
    except Exception:
        pass

    dest_dir = args.dest or "."
    base_name = f"{os.path.basename(DB_NAME)}.backup_{now_ts()}"
    dest_path = os.path.join(dest_dir, base_name)

    path = backup_db(dest_path)
    print(f"Backup saved: {path}")

    if args.multi:
        dirs_env = os.getenv("DB_BACKUP_DIRS", "").strip()
        if dirs_env:
            import shutil
            for d in dirs_env.split(":"):
                d = d.strip()
                if d and os.path.isdir(d):
                    p2 = os.path.join(d, base_name)
                    shutil.copy2(path, p2)
                    print(f"  -> {p2}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
