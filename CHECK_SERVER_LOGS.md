# How to check server logs (find Internal Server Error cause)

## 1. SSH into the server

From **PowerShell** on your PC (use your key path):

```powershell
ssh -i "C:\Users\jonat\OneDrive\Desktop\server\eirn-bot-key.pem" ubuntu@3.148.6.246
```

## 2. Where to look for errors

### A) App output (if you started with the deploy script)

```bash
cd ~/local_3comas_clone_v2
tail -200 deploy.log
```

To follow live (refresh as you reproduce the error):

```bash
tail -f deploy.log
```

### B) Systemd service logs (if the app runs as a service)

```bash
# Try these service names (one may be used on your server)
sudo journalctl -u tradingserver -n 150 --no-pager
sudo journalctl -u ai-bot -n 150 --no-pager
```

### C) Search for Python tracebacks

```bash
cd ~/local_3comas_clone_v2
grep -A 20 "Traceback\|Error\|Exception" deploy.log | tail -80
```

Or in journalctl:

```bash
sudo journalctl -u tradingserver -n 300 --no-pager | grep -B 2 -A 15 "Traceback\|Error\|Exception"
```

## 3. One-liner from your PC (no need to type on server)

Run this in PowerShell – it SSHs in and shows the last 150 lines of the app log:

```powershell
ssh -i "C:\Users\jonat\OneDrive\Desktop\server\eirn-bot-key.pem" ubuntu@3.148.6.246 "cd ~/local_3comas_clone_v2 && tail -150 deploy.log"
```

For systemd service logs:

```powershell
ssh -i "C:\Users\jonat\OneDrive\Desktop\server\eirn-bot-key.pem" ubuntu@3.148.6.246 "sudo journalctl -u tradingserver -n 150 --no-pager"
```

---

**Tip:** Reproduce the Internal Server Error (open the page again), then immediately run one of the commands above so the traceback appears at the end of the log.
