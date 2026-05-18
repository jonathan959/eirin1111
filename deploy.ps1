# Trading bot deploy - production host from deploy_host.txt (single source of truth). No DuckDNS.
# Usage: .\deploy.ps1 [-Quick] [-BringUpOnly] [-KeyPath <pem>]
#   -Quick       Skip backup (use when bots are running and backup hangs)
#   -BringUpOnly Just run quick_fix_502 on server to restore site, then exit (no deploy)
#   -KeyPath     SSH private key (.pem); overrides env and defaults — see Resolve-EirinDeployKeyPath below.

param(
  [switch]$Quick,
  [switch]$BringUpOnly,
  [switch]$RecoverEc2,
  [switch]$SkipRecover,
  [string]$KeyPath = ""
)

$ErrorActionPreference = "Stop"
$LocalRoot = Split-Path -Parent $MyInvocation.MyCommand.Path

function Get-EirinDeployHost {
  $hostFile = Join-Path $LocalRoot "deploy_host.txt"
  if (-not (Test-Path -LiteralPath $hostFile)) {
    throw "Missing deploy_host.txt in project root (canonical production IP)."
  }
  $h = (Get-Content -LiteralPath $hostFile -Raw).Trim()
  if (-not $h) { throw "deploy_host.txt is empty." }
  return $h
}

function Resolve-EirinDeployKeyPath {
  param([string]$Explicit)
  $candidates = [System.Collections.Generic.List[string]]::new()
  if ($Explicit -and $Explicit.Trim()) {
    [void]$candidates.Add($Explicit.Trim())
  }
  $envKey = $env:EIRIN_DEPLOY_KEY
  if ($envKey -and $envKey.Trim()) {
    [void]$candidates.Add($envKey.Trim())
  }
  [void]$candidates.Add((Join-Path $LocalRoot "eirin-bot-key.pem"))
  # Historical desktop stash (correct spelling: eirin-bot-key.pem)
  [void]$candidates.Add("C:\Users\jonat\OneDrive\Desktop\server\eirin-bot-key.pem")

  foreach ($p in $candidates) {
    if ([string]::IsNullOrWhiteSpace($p)) { continue }
    if (Test-Path -LiteralPath $p) {
      return (Resolve-Path -LiteralPath $p).Path
    }
  }

  $lines = ($candidates | ForEach-Object { "  - $_" }) -join "`n"
  Write-Error @"
No SSH private key found for deploy.

Checked (in order):
$lines

Fix one of:
  * Place your EC2 key next to deploy.ps1 as eirin-bot-key.pem
  * `$env:EIRIN_DEPLOY_KEY = 'C:\full\path\eirin-bot-key.pem'
  * .\deploy.ps1 -KeyPath 'C:\full\path\eirin-bot-key.pem'
"@
  exit 1
}

$KeyPath = Resolve-EirinDeployKeyPath -Explicit $KeyPath
Write-Host "Using SSH key: $KeyPath" -ForegroundColor DarkGray

function Test-DeployHostReachable {
  param([string]$TargetHost, [int]$TimeoutMs = 8000)
  try {
    $c = New-Object System.Net.Sockets.TcpClient
    $iar = $c.BeginConnect($TargetHost, 22, $null, $null)
    if (-not $iar.AsyncWaitHandle.WaitOne($TimeoutMs)) { $c.Close(); return $false }
    $c.EndConnect($iar)
    $c.Close()
    return $true
  } catch { return $false }
}

$HostName = Get-EirinDeployHost
$User = "ubuntu"
$RemoteDir = "/home/ubuntu/local_3comas_clone_v2"

if (-not $SkipRecover -and ($RecoverEc2 -or -not (Test-DeployHostReachable -TargetHost $HostName))) {
  if (-not (Test-DeployHostReachable -TargetHost $HostName)) {
    Write-Host "Server $HostName port 22 unreachable - attempting server recovery..." -ForegroundColor Yellow
  }
  $recoverScript = Join-Path $LocalRoot "scripts\recover_ec2_and_deploy.ps1"
  if (Test-Path $recoverScript) {
    $recoverArgs = @()
    if ($Quick) { $recoverArgs += "-Quick" }
    if ($RecoverEc2 -and (Test-DeployHostReachable -TargetHost $HostName)) {
      $recoverArgs += "-DeployOnly"
    }
    & $recoverScript @recoverArgs
    exit $LASTEXITCODE
  } else {
    Write-Host "Missing $recoverScript - ensure $HostName is up and SSH (port 22) is open, then re-run deploy." -ForegroundColor Red
    exit 1
  }
}

if ($BringUpOnly) {
  Write-Host "Bring-up only: running quick_fix_502 on server..." -ForegroundColor Cyan
  $SshArgs = @("-o","StrictHostKeyChecking=no","-o","ConnectTimeout=15","-o","ServerAliveInterval=10","-o","ServerAliveCountMax=3")
  & scp @SshArgs -i $KeyPath (Join-Path $LocalRoot "quick_fix_502.sh") "$User@${HostName}:$RemoteDir/"
  if ($LASTEXITCODE -ne 0) { Write-Host "SCP failed. Is the server reachable (SSH port 22)?" -ForegroundColor Red; exit 1 }
  & ssh @SshArgs -i $KeyPath "$User@$HostName" "sed -i 's/\r$//' $RemoteDir/quick_fix_502.sh 2>/dev/null; chmod +x $RemoteDir/quick_fix_502.sh; bash $RemoteDir/quick_fix_502.sh"
  if ($LASTEXITCODE -ne 0) { Write-Host "SSH/quick_fix_502 failed." -ForegroundColor Red; exit 1 }
  Write-Host "Done. Check http://${HostName}" -ForegroundColor Green
  exit 0
}

$SshArgs = @(
  "-o", "StrictHostKeyChecking=no",
  "-o", "ConnectTimeout=15",
  "-o", "ServerAliveInterval=10",
  "-o", "ServerAliveCountMax=3"
)

$Files = @(
  "one_server.py", "one_server_v2.py", "worker_api.py", "worker.py", "bot_manager.py", "executor.py",
  "strategies.py", "symbol_classifier.py", "alpaca_adapter.py", "alpaca_client.py",
  "stock_metadata.py", "stock_universe.py", "market_data.py", "risk_circuit_breaker.py",
  "price_predictor.py", "strategy_optimizer.py", "portfolio_optimizer.py",
  "anomaly_detector.py", "db.py", "kraken_client.py", "intelligence_layer.py", "phase1_intelligence.py",
  "data_validator.py", "circuit_breaker.py", "health_monitor.py", "autopilot.py",
  "portfolio_initializer.py", "notification_manager.py",
  "optimizer.py", "monte_carlo.py", "strategy_discovery.py", "backtest.py",
  "funding_rate_tracker.py", "crypto_cycle_detector.py", "meme_coin_detector.py",
  "crypto_correlation.py", "defi_yield_tracker.py",
  "portfolio_correlation.py", "env_utils.py", "multi_timeframe.py", "sentiment_analyzer.py",
  "order_book_analyzer.py", "ml_predictor.py", "kelly_criterion.py", "recommendation_validator.py",
  "intraday_regime.py", "scalping_strategy.py", "fundamental_valuation.py",
  "sector_rotation.py", "tax_optimizer.py", "long_term_strategies.py",
  "social_sentiment.py", "options_flow.py", "onchain_analyzer.py", "insider_tracker.py",
  "alternative_data.py", "event_calendar.py", "pattern_recognition.py",
  "ml_prediction_tracker.py", "ml_ensemble.py", "price_predictor.py",
  "phase2_data_fetcher.py", "alpaca_rate_limiter.py",
  "unified_alpaca_client.py", "websocket_manager.py", "data_cache.py", "enhanced_rate_limiter.py",
  "correlation_trading.py", "seasonality.py",
  "earnings_momentum.py", "zscore_trading.py", "momentum_ranking.py",
  "explore_composite_scorer.py", "explore_rank.py", "explore_scorer.py",
  "rl_agent.py", "high_frequency.py",
  "app.py", "requirements.txt", "db_backup.py"
)

$Scripts = @(
  "validate_before_restart.sh", "fix_service.sh", "quick_fix_502.sh",
  "deploy_restart.sh", "health_watchdog.sh",
  "check_nginx.sh", "setup_nginx.sh", "fix_port80.sh",
  "deploy_backup.sh", "deploy_restore.sh", "restore_db_from_backup.sh", "server_stability_diagnostic.sh",
  "disk_cleanup_cron.sh", "free_disk_space.sh", "smoke_test.sh", "phase0_discovery.sh", "install_ai_bot.sh"
)

function Invoke-Ssh {
  param([string]$Cmd)
  & ssh @SshArgs -i $KeyPath "$User@$HostName" $Cmd
}

function Invoke-SshWithTimeout {
  param([string]$Cmd, [int]$TimeoutSec = 60)
  $job = Start-Job -ScriptBlock {
    param($opts, $key, $u, $h, $c)
    & ssh @opts -i $key "$u@$h" $c 2>&1
  } -ArgumentList (,$SshArgs), $KeyPath, $User, $HostName, $Cmd
  $null = Wait-Job $job -Timeout $TimeoutSec
  if ($job.State -eq "Running") {
    Stop-Job $job; Remove-Job $job -Force
    return $null
  }
  $out = Receive-Job $job
  Remove-Job $job -Force
  return $out
}

# --- 0. Copy backup/restore scripts first (needed for backup step) ---
Write-Host "Uploading deploy helpers..." -ForegroundColor Cyan
$helpers = @("deploy_backup.sh", "deploy_restore.sh")
foreach ($h in $helpers) {
  $p = Join-Path $LocalRoot $h
  if (Test-Path $p) { & scp @SshArgs -i $KeyPath $p "$User@${HostName}:$RemoteDir/" }
}

# --- 1. Backup (timeout 30s) - skip with -Quick ---
$NoBackup = $false
if (-not $Quick) {
  Write-Host "Backing up current app (30s timeout)..." -ForegroundColor Cyan
  Invoke-Ssh "sed -i 's/\r$//' $RemoteDir/deploy_backup.sh 2>/dev/null; chmod +x $RemoteDir/deploy_backup.sh"
  $backupOut = Invoke-SshWithTimeout -Cmd "timeout 30 bash $RemoteDir/deploy_backup.sh 2>&1" -TimeoutSec 40
  if (-not $backupOut -or ($backupOut | Out-String) -notmatch "BACKUP_DONE") {
    Write-Host "Backup timed out or failed. Continuing without backup (rollback disabled)." -ForegroundColor Yellow
    $NoBackup = $true
  } else {
    Write-Host "Backup done." -ForegroundColor Green
  }
} else {
  Write-Host "Skipping backup (-Quick)." -ForegroundColor Yellow
  $NoBackup = $true
}

# --- 1.5. Free disk if full (run inline - no copy needed) ---
Write-Host "Ensuring disk space..." -ForegroundColor Cyan
Invoke-Ssh "rm -rf ~/.cache/pip 2>/dev/null; sudo apt-get clean 2>/dev/null; sudo journalctl --vacuum-size=30M 2>/dev/null; sudo truncate -s 0 /var/log/syslog 2>/dev/null; sudo find /var/log -type f -name '*.gz' -delete 2>/dev/null; sudo rm -rf /tmp/* 2>/dev/null; sudo chmod 1777 /tmp; mkdir -p $RemoteDir/tmp; chmod 700 $RemoteDir/tmp; find $RemoteDir -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true; if [ -d $RemoteDir/backups ]; then ls -t $RemoteDir/backups/ 2>/dev/null | tail -n +2 | xargs -r -I {} rm -rf $RemoteDir/backups/{} 2>/dev/null; fi; sudo chown -R ubuntu:ubuntu $RemoteDir/venv 2>/dev/null || true; df -h / | tail -1"

# --- 2. Copy all files ---
Write-Host "Copying files to server..." -ForegroundColor Cyan
$allPy = $Files | ForEach-Object { Join-Path $LocalRoot $_ }
& scp @SshArgs -i $KeyPath $allPy "$User@${HostName}:$RemoteDir/"
# Copy .env (API keys) - required for Kraken/Alpaca and BotManager init
$envPath = Join-Path $LocalRoot ".env"
if (Test-Path $envPath) {
  & scp @SshArgs -i $KeyPath $envPath "$User@${HostName}:$RemoteDir/.env"
  Write-Host "  .env copied (API keys)." -ForegroundColor Gray
} else {
  Write-Host "  WARNING: .env not found locally. Server may have stale/missing API keys." -ForegroundColor Yellow
}
# Ensure project dir and DB are writable (prevents "readonly database" after deploy)
& ssh @SshArgs -i $KeyPath "$User@$HostName" "sudo chown -R ubuntu:ubuntu $RemoteDir 2>/dev/null; sudo chmod u+w $RemoteDir 2>/dev/null"
# Ensure templates and static dirs exist and are writable
& ssh @SshArgs -i $KeyPath "$User@$HostName" "sudo chmod -R u+w $RemoteDir/templates $RemoteDir/static 2>/dev/null; mkdir -p $RemoteDir/templates $RemoteDir/static"
& scp @SshArgs -i $KeyPath -r (Join-Path $LocalRoot "templates") "$User@${HostName}:$RemoteDir/"
& scp @SshArgs -i $KeyPath -r (Join-Path $LocalRoot "static") "$User@${HostName}:$RemoteDir/"
if (Test-Path (Join-Path $LocalRoot "scripts")) {
  & scp @SshArgs -i $KeyPath -r (Join-Path $LocalRoot "scripts") "$User@${HostName}:$RemoteDir/"
}
if (Test-Path (Join-Path $LocalRoot "services")) {
  Remove-Item -Recurse -Force (Join-Path $LocalRoot "services/__pycache__") -ErrorAction SilentlyContinue
  & ssh @SshArgs -i $KeyPath "$User@$HostName" "rm -rf $RemoteDir/services/__pycache__ 2>/dev/null || true"
  Write-Host "Copying services/..." -ForegroundColor Cyan
  & scp @SshArgs -i $KeyPath -r (Join-Path $LocalRoot "services") "$User@${HostName}:$RemoteDir/"
}
foreach ($s in $Scripts) {
  $p = Join-Path $LocalRoot $s
  if (Test-Path $p) { & scp @SshArgs -i $KeyPath $p "$User@${HostName}:$RemoteDir/" }
}
& scp @SshArgs -i $KeyPath (Join-Path $LocalRoot "tradingserver.service") "$User@${HostName}:$RemoteDir/"
& scp @SshArgs -i $KeyPath (Join-Path $LocalRoot "ai-bot.service") "$User@${HostName}:$RemoteDir/"
& scp @SshArgs -i $KeyPath (Join-Path $LocalRoot "nginx-ai-bot.conf") "$User@${HostName}:$RemoteDir/"
if (Test-Path (Join-Path $LocalRoot "ops")) {
  & scp @SshArgs -i $KeyPath -r (Join-Path $LocalRoot "ops") "$User@${HostName}:$RemoteDir/"
}
$journalConf = Join-Path $LocalRoot "journald_size_limit.conf"
if (Test-Path $journalConf) {
  & scp @SshArgs -i $KeyPath $journalConf "$User@${HostName}:$RemoteDir/"
}

Write-Host "Installing service + Nginx..." -ForegroundColor Cyan
# Fix temp dirs (often broken/full after EC2 reboot) - run inline, no file copy needed
Invoke-Ssh "sudo chmod 1777 /tmp 2>/dev/null; mkdir -p $RemoteDir/tmp; chmod 700 $RemoteDir/tmp"
# Production: ai-bot (one_server_v2:app). Stop tradingserver so only one binds :8000.
Invoke-Ssh "sudo systemctl stop tradingserver 2>/dev/null; sudo systemctl disable tradingserver 2>/dev/null; sudo fuser -k 8000/tcp 2>/dev/null; sleep 2"
Invoke-Ssh "sudo cp $RemoteDir/ai-bot.service /etc/systemd/system/ai-bot.service; sudo systemctl daemon-reload; sudo systemctl enable --now ai-bot"
Invoke-Ssh "sudo mkdir -p /etc/systemd/journald.conf.d; sudo cp $RemoteDir/journald_size_limit.conf /etc/systemd/journald.conf.d/ 2>/dev/null; sudo systemctl restart systemd-journald 2>/dev/null || true"
Invoke-Ssh "sed -i 's/\r$//' $RemoteDir/check_nginx.sh $RemoteDir/setup_nginx.sh $RemoteDir/install_ai_bot.sh $RemoteDir/scripts/*.sh 2>/dev/null; chmod +x $RemoteDir/scripts/*.sh $RemoteDir/install_ai_bot.sh 2>/dev/null; chmod +x $RemoteDir/check_nginx.sh $RemoteDir/setup_nginx.sh; bash $RemoteDir/setup_nginx.sh"
# Install daily disk cleanup cron (4am UTC) - prevents disk fill from logs
Invoke-Ssh "sed -i 's/\r$//' $RemoteDir/disk_cleanup_cron.sh 2>/dev/null; chmod +x $RemoteDir/disk_cleanup_cron.sh; (crontab -l 2>/dev/null | grep -v disk_cleanup_cron; echo ""0 4 * * * $RemoteDir/disk_cleanup_cron.sh"") | crontab - 2>/dev/null || true"

# --- 3. Validate (90s timeout) then restart ---
Write-Host "Validating (180s timeout)..." -ForegroundColor Cyan
$validateCmd = "cd $RemoteDir; sed -i 's/\r$//' validate_before_restart.sh 2>/dev/null; chmod +x validate_before_restart.sh; source venv/bin/activate; timeout 120 bash validate_before_restart.sh 2>&1; if [ `$? -eq 0 ]; then echo VALIDATE_OK; else echo VALIDATE_FAIL; exit 1; fi"
$validateOut = Invoke-SshWithTimeout -Cmd $validateCmd -TimeoutSec 200
$validatePass = $validateOut -and ($validateOut | Out-String) -match "VALIDATE_OK"

if (-not $validatePass) {
  Write-Host "Validation failed. Restoring previous version (if backup exists) and restarting." -ForegroundColor Red
  if ($validateOut) { Write-Host ($validateOut | Out-String) }
  if (-not $NoBackup) {
    Invoke-Ssh "sed -i 's/\r$//' $RemoteDir/deploy_restore.sh 2>/dev/null; chmod +x $RemoteDir/deploy_restore.sh; bash $RemoteDir/deploy_restore.sh"
  }
  Invoke-Ssh "bash $RemoteDir/deploy_restart.sh 2>/dev/null || sudo systemctl restart ai-bot"
  Write-Host "Deployment aborted. Fix code and run deploy again." -ForegroundColor Yellow
  exit 1
}

Write-Host "Validation passed. Restarting application (deploy_restart.sh → ai-bot on :8000)..." -ForegroundColor Green
Invoke-Ssh "sed -i 's/\r$//' $RemoteDir/deploy_restart.sh 2>/dev/null; chmod +x $RemoteDir/deploy_restart.sh; bash $RemoteDir/deploy_restart.sh || true"

Write-Host "Waiting for service (up to 90s)..." -ForegroundColor Yellow
$baseUrl = "http://${HostName}"
$baseUrlDirect = "http://${HostName}:8000"
$urls = @(
  "$baseUrl/api/health", "$baseUrl/health",
  "$baseUrlDirect/api/health", "$baseUrlDirect/health"
)

function Test-Health {
  foreach ($url in $urls) {
    try {
      $r = Invoke-WebRequest -Uri $url -Method GET -TimeoutSec 8 -UseBasicParsing -ErrorAction Stop
      if ($r.StatusCode -ne 200) { continue }
      $j = $r.Content | ConvertFrom-Json -ErrorAction SilentlyContinue
      if (-not $j) { continue }
      if ($j.ok -eq $true -or $j.status -eq "healthy" -or $j.status -eq "degraded") {
        return @{ Ok = $true; Url = $url; Data = $j }
      }
    } catch { continue }
  }
  return @{ Ok = $false }
}

$maxAttempts = 45
$attempt = 0
$healthy = $false
$result = $null
while ($attempt -lt $maxAttempts) {
  Start-Sleep -Seconds 2
  $attempt++
  $result = Test-Health
  if ($result.Ok) {
    $healthy = $true
    Write-Host "`nConnected: $($result.Url)" -ForegroundColor Cyan
    break
  }
  if ($attempt % 5 -eq 0) { Write-Host "." -NoNewline -ForegroundColor Yellow }
}

# --- 4. Rollback on health failure ---
if (-not $healthy) {
  Write-Host "`nHealth failed. Rollback + restart..." -ForegroundColor Yellow
  if (-not $NoBackup) {
    Invoke-Ssh "bash $RemoteDir/deploy_restore.sh"
  }
  Invoke-Ssh "bash $RemoteDir/deploy_restart.sh 2>/dev/null || sudo systemctl restart ai-bot"
  Start-Sleep -Seconds 25
  $attempt = 0
  while ($attempt -lt 30) {
    Start-Sleep -Seconds 2
    $attempt++
    $result = Test-Health
    if ($result.Ok) {
      $healthy = $true
      Write-Host "`nConnected after rollback: $($result.Url)" -ForegroundColor Cyan
      break
    }
  }
}

if ($healthy) {
  Write-Host "`nService is ready." -ForegroundColor Green
  if ($result.Data) {
    Write-Host "  Status: $($result.Data.status)" -ForegroundColor Gray
    Write-Host "  Kraken: $($result.Data.kraken_ready)  Alpaca: $($result.Data.alpaca_ready)" -ForegroundColor Gray
  }
  Write-Host "`nDeploy completed." -ForegroundColor Green
  exit 0
}

# --- 5. Last resort: quick_fix_502 ---
Write-Host "`nHealth still failing. Running quick_fix_502..." -ForegroundColor Yellow
Invoke-Ssh "sed -i 's/\r$//' $RemoteDir/quick_fix_502.sh 2>/dev/null; chmod +x $RemoteDir/quick_fix_502.sh; bash $RemoteDir/quick_fix_502.sh"
Start-Sleep -Seconds 15
$attempt = 0
while ($attempt -lt 15) {
  Start-Sleep -Seconds 2
  $attempt++
  $result = Test-Health
  if ($result.Ok) {
    Write-Host "`nOK after quick_fix_502." -ForegroundColor Green
    exit 0
  }
}

Write-Host "`nStill failing. Run: ssh $User@$HostName 'sudo journalctl -u ai-bot -n 80 --no-pager'" -ForegroundColor Red
exit 1
