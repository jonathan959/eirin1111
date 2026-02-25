#!/bin/bash
# Disk cleanup - prevents "No space left on device" on EC2.
# Installed by deploy.ps1: daily at 4am + weekly deep clean Sunday 3am.

DIR="${1:-/home/ubuntu/local_3comas_clone_v2}"
cd "$DIR" 2>/dev/null || exit 0

rm -rf ~/.cache/pip 2>/dev/null
sudo apt-get clean 2>/dev/null
sudo journalctl --vacuum-size=30M 2>/dev/null
sudo truncate -s 0 /var/log/syslog 2>/dev/null
sudo find /var/log -type f -name '*.gz' -delete 2>/dev/null
sudo rm -rf /tmp/* 2>/dev/null
sudo chmod 1777 /tmp 2>/dev/null
mkdir -p "$DIR/tmp" 2>/dev/null
chmod 700 "$DIR/tmp" 2>/dev/null
find "$DIR" -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
# Keep only last backup
if [ -d "$DIR/backups" ]; then
  ls -t "$DIR/backups/" 2>/dev/null | tail -n +2 | xargs -r -I {} rm -rf "$DIR/backups/{}" 2>/dev/null
fi

logger -t disk-cleanup "Cleanup done. $(df -h / | tail -1)"
