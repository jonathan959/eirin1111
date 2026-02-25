#!/bin/bash
# Monitor pip install (PID 1512) every 10 min for up to 6 checks
cd /mnt/c/Users/jonat/OneDrive/Desktop/local_3comas_clone_v2
for i in 1 2 3 4 5 6; do
  echo "=== Check $i @ $(date) ==="
  if ! ps -p 1512 -o pid,etime,%cpu,%mem,cmd 2>/dev/null; then
    echo "PID 1512 NOT FOUND - pip has exited"
    exit 0
  fi
  if [ $i -lt 6 ]; then
    sleep 600
  fi
done
echo "=== Monitoring complete ==="
