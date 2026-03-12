#!/bin/bash
# Restore database from .previous backup (e.g. to recover bots after empty DB).
# Run on server: bash restore_db_from_backup.sh
# Safe: only restores if .previous/botdb.sqlite3 exists and is readable.

set -e
cd /home/ubuntu/local_3comas_clone_v2 2>/dev/null || { echo "Run from project root on server"; exit 1; }

if [ ! -f .previous/botdb.sqlite3 ]; then
  echo "No backup found at .previous/botdb.sqlite3. Cannot restore."
  echo "From next deploy onward, DB is backed up before deploy; use deploy_restore.sh to restore."
  exit 1
fi

echo "Stopping app to avoid DB locks..."
sudo systemctl stop tradingserver 2>/dev/null || true
sudo systemctl stop ai-bot 2>/dev/null || true
sleep 2

echo "Restoring database from .previous/..."
cp -p .previous/botdb.sqlite3 .
[ -f .previous/botdb.sqlite3-wal ] && cp -p .previous/botdb.sqlite3-wal . 2>/dev/null || true
[ -f .previous/botdb.sqlite3-shm ] && cp -p .previous/botdb.sqlite3-shm . 2>/dev/null || true
chown ubuntu:ubuntu botdb.sqlite3* 2>/dev/null || true

echo "Starting tradingserver..."
sudo systemctl start tradingserver 2>/dev/null || true
echo "Done. Check /bots and /autopilot. If still empty, the backup had no bots (backup was taken after loss)."
exit 0
