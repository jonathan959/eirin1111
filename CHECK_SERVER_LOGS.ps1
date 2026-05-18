# Quick check: SSH to server and find app logs (to find Internal Server Error).
# Run from project folder: .\CHECK_SERVER_LOGS.ps1
# Requires: $env:AWS_DEPLOY_KEY set to your .pem path, or edit $KeyPath below.

$ScriptRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$KeyPath = $env:AWS_DEPLOY_KEY
if (-not $KeyPath -or -not (Test-Path -LiteralPath $KeyPath)) {
    $KeyPath = $env:EIRIN_DEPLOY_KEY
}
if (-not $KeyPath -or -not (Test-Path -LiteralPath $KeyPath)) {
    $KeyPath = Join-Path $ScriptRoot "eirin-bot-key.pem"
}
if (-not $KeyPath -or -not (Test-Path -LiteralPath $KeyPath)) {
    $KeyPath = "C:\Users\jonat\OneDrive\Desktop\server\eirin-bot-key.pem"
}
if (-not $KeyPath -or -not (Test-Path -LiteralPath $KeyPath)) {
    Write-Error "No SSH key found. Set `$env:EIRIN_DEPLOY_KEY or `$env:AWS_DEPLOY_KEY, or place eirin-bot-key.pem next to this script."
    exit 1
}
$Server = (Get-Content (Join-Path $ScriptRoot "deploy_host.txt") -Raw).Trim()
if (-not $Server) { Write-Error "deploy_host.txt is missing or empty."; exit 1 }
$User = "ubuntu"

Write-Host "=== Finding what is running and where logs are ===" -ForegroundColor Cyan
Write-Host ""

# Run diagnostic: ports, processes, log files, systemd
$remote = "echo '=== Port 8000/80 ===' && (sudo ss -tlnp 2>/dev/null | grep -E ':8000|:80 ' || true) && echo '=== Python/uvicorn ===' && (ps aux | grep -E '[p]ython|[u]vicorn' || true) && echo '=== deploy.log last 80 ===' && (tail -80 ~/local_3comas_clone_v2/deploy.log 2>/dev/null || echo 'No deploy.log') && echo '=== nohup.out last 80 ===' && (tail -80 ~/nohup.out 2>/dev/null || echo 'No nohup.out') && echo '=== Systemd services (trading|bot|ai) ===' && (systemctl list-unit-files --type=service --no-pager 2>/dev/null | grep -iE 'trading|bot|ai|uvicorn' || echo 'None') && echo '=== journalctl for tradingserver ===' && (sudo journalctl -u tradingserver -n 50 --no-pager 2>/dev/null || echo 'No entries') && echo '=== journalctl for ai-bot ===' && (sudo journalctl -u ai-bot -n 50 --no-pager 2>/dev/null || echo 'No entries')"
& ssh -i $KeyPath "${User}@${Server}" $remote

Write-Host ""
Write-Host "=== Next step: If you see a service name above (e.g. myapp.service), run ===" -ForegroundColor Yellow
Write-Host "   ssh -i `"$KeyPath`" ${User}@${Server} `"sudo journalctl -u SERVICENAME -n 150 --no-pager`"" -ForegroundColor Gray
