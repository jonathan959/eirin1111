#!/bin/bash
# SSH_DIAGNOSTIC.sh - Diagnose SSH connectivity to 3.148.6.246
# Run this if deploy.sh fails with connection errors

HOST_IP="3.148.6.246"
HOST_USER="ubuntu"
SSH_TARGET="${HOST_USER}@${HOST_IP}"

echo "=== SSH Diagnostic for 3.148.6.246 ==="
echo "Target: $SSH_TARGET"
echo ""

# 1. Test ping
echo "[1] Ping test..."
if ping -c 2 -W 5 "$HOST_IP" 2>/dev/null; then
    echo "PING: OK"
else
    echo "PING: FAILED - Host unreachable"
    echo "  -> Check: Is the EC2 instance running?"
    echo "  -> Check: Is your network/VPN correct?"
fi
echo ""

# 2. Test port 22
echo "[2] Port 22 test..."
if timeout 5 bash -c "echo >/dev/tcp/$HOST_IP/22" 2>/dev/null; then
    echo "PORT 22: OK"
else
    echo "PORT 22: FAILED - Cannot reach SSH port"
    echo ""
    echo "AWS SECURITY GROUP FIX:"
    echo "  1. AWS Console -> EC2 -> Security Groups"
    echo "  2. Select the SG attached to instance $HOST_IP"
    echo "  3. Inbound rules: ADD rule"
    echo "     - Type: SSH"
    echo "     - Port: 22"
    echo "     - Source: Your IP (e.g. 0.0.0.0/0 for anywhere, or your office IP)"
    echo "  4. Save rules"
fi
echo ""

# 3. Test SSH connection
echo "[3] SSH connection test..."
if ssh -v -o ConnectTimeout=10 -o BatchMode=yes "$SSH_TARGET" "echo 'SSH OK'" 2>&1 | tail -20; then
    echo ""
    echo "SSH: OK - Connection successful"
else
    echo ""
    echo "SSH: FAILED"
    echo ""
    echo "AWS SECURITY GROUP FIX (if port 22 was OK but SSH failed):"
    echo "  - Verify Inbound: SSH (22) from your IP"
    echo "  - Ensure key pair is correct (check ~/.ssh/)"
    echo ""
    echo "If port 22 test failed, fix that first (see step 2 above)."
fi
echo ""

echo "=== Diagnostic complete ==="
