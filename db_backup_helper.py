"""
Database backup utility - creates daily backups of botdb.sqlite3
"""

import os
import shutil
import logging
import time
from pathlib import Path
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

DB_NAME = os.getenv("BOT_DB_PATH", "botdb.sqlite3")
BACKUP_DIR = os.getenv("DB_BACKUP_DIR", "backups")
BACKUP_RETENTION_DAYS = int(os.getenv("DB_BACKUP_RETENTION_DAYS", "30"))
BACKUP_ENABLED = os.getenv("DB_BACKUP_ENABLED", "1").strip().lower() in ("1", "true", "yes", "y", "on")


def ensure_backup_dir():
    """Create backup directory if it doesn't exist."""
    path = Path(BACKUP_DIR)
    path.mkdir(parents=True, exist_ok=True)
    return path


def backup_database():
    """Create a timestamped backup of the database file."""
    if not BACKUP_ENABLED:
        return False

    if not os.path.exists(DB_NAME):
        logger.warning(f"Database file not found: {DB_NAME}")
        return False

    try:
        backup_dir = ensure_backup_dir()
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_path = backup_dir / f"botdb_backup_{timestamp}.sqlite3"

        shutil.copy2(DB_NAME, backup_path)
        logger.info(f"Database backed up to: {backup_path}")

        # Cleanup old backups
        cleanup_old_backups(backup_dir)
        return True

    except Exception as e:
        logger.error(f"Database backup failed: {e}")
        return False


def cleanup_old_backups(backup_dir=None):
    """Remove backup files older than retention period."""
    if backup_dir is None:
        backup_dir = Path(BACKUP_DIR)

    if not backup_dir.exists():
        return 0

    try:
        cutoff_time = time.time() - (BACKUP_RETENTION_DAYS * 86400)
        deleted_count = 0

        for backup_file in backup_dir.glob("botdb_backup_*.sqlite3"):
            if os.path.getmtime(backup_file) < cutoff_time:
                os.remove(backup_file)
                deleted_count += 1
                logger.info(f"Deleted old backup: {backup_file}")

        return deleted_count

    except Exception as e:
        logger.warning(f"Cleanup old backups failed: {e}")
        return 0


def get_latest_backup():
    """Get path to the most recent backup file."""
    backup_dir = Path(BACKUP_DIR)
    if not backup_dir.exists():
        return None

    try:
        backups = sorted(backup_dir.glob("botdb_backup_*.sqlite3"), reverse=True)
        return str(backups[0]) if backups else None
    except Exception as e:
        logger.warning(f"Failed to find latest backup: {e}")
        return None


def restore_backup(backup_file):
    """Restore from a backup file."""
    if not os.path.exists(backup_file):
        logger.error(f"Backup file not found: {backup_file}")
        return False

    try:
        # Create safety backup of current DB first
        if os.path.exists(DB_NAME):
            safety_backup = f"{DB_NAME}.before_restore"
            shutil.copy2(DB_NAME, safety_backup)
            logger.info(f"Created safety backup at: {safety_backup}")

        # Restore from backup
        shutil.copy2(backup_file, DB_NAME)
        logger.info(f"Restored database from: {backup_file}")
        return True

    except Exception as e:
        logger.error(f"Database restore failed: {e}")
        return False


if __name__ == "__main__":
    # Test backup
    logging.basicConfig(level=logging.INFO)
    if backup_database():
        print("Backup successful")
    else:
        print("Backup failed")
