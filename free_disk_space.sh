#!/bin/bash
# Free disk space on EC2 - run when "No space left on device"
# Usage: ssh ubuntu@3.148.6.246 'bash -s' < free_disk_space.sh
# Or: scp free_disk_space.sh ubuntu@3.148.6.246:~/  && ssh ubuntu@3.148.6.246 'bash ~/free_disk_space.sh'

set -e
echo "=== Disk usage before ==="
df -h /
df -h /tmp 2>/dev/null || true
echo ""
du -sh /home/ubuntu/local_3comas_clone_v2/*/ 2>/dev/null | sort -hr | head -10
echo ""

echo "=== Freeing space ==="
# 1. Clear pip cache (often hundreds of MB)
rm -rf ~/.cache/pip 2>/dev/null
echo "  pip cache cleared"

# 2. Clear apt cache
sudo apt-get clean 2>/dev/null
echo "  apt cache cleared"

# 3. Truncate journal logs (keep last 20MB)
sudo journalctl --vacuum-size=20M 2>/dev/null
echo "  journal logs trimmed"

# 4. Remove old backups (keep last 1)
cd /home/ubuntu/local_3comas_clone_v2 2>/dev/null
if [ -d backups ]; then
  ls -t backups/ 2>/dev/null | tail -n +2 | xargs -r -I {} rm -rf backups/{} 2>/dev/null
  echo "  old backups removed"
fi

# 5. Clear /tmp
sudo rm -rf /tmp/* /tmp/.* 2>/dev/null
sudo chmod 1777 /tmp
echo "  /tmp cleared"

# 6. Python __pycache__
find /home/ubuntu/local_3comas_clone_v2 -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
echo "  __pycache__ cleared"

# 7. Large log files (syslog, journal)
sudo truncate -s 0 /var/log/syslog 2>/dev/null || true
sudo find /var/log -type f -name '*.gz' -delete 2>/dev/null || true
echo "  large logs trimmed"

echo ""
echo "=== Disk usage after ==="
df -h /
echo ""
echo "=== Done ==="
