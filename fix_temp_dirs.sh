#!/bin/bash
# Fix temp directories - needed when /tmp is full or has wrong permissions after reboot.
# Run on server before pip/uvicorn: sudo bash fix_temp_dirs.sh

echo "=== Fixing temp directories ==="
sudo mkdir -p /tmp /var/tmp /usr/tmp
sudo chmod 1777 /tmp
sudo chmod 1777 /var/tmp
sudo chmod 1777 /usr/tmp
# Fallback: project tmp for TMPDIR if system tmp is broken
mkdir -p /home/ubuntu/local_3comas_clone_v2/tmp
chmod 700 /home/ubuntu/local_3comas_clone_v2/tmp
echo "Temp dirs OK"
df -h /tmp 2>/dev/null || df -h / 2>/dev/null || true
