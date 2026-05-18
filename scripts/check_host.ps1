# SSH port 22 reachability check for canonical production host (deploy_host.txt).
$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot
$hostFile = Join-Path $Root "deploy_host.txt"
if (-not (Test-Path -LiteralPath $hostFile)) {
  Write-Host "FAIL: deploy_host.txt not found at $hostFile"
  exit 1
}
$DeployHost = (Get-Content -LiteralPath $hostFile -Raw).Trim()
if (-not $DeployHost) {
  Write-Host "FAIL: deploy_host.txt is empty"
  exit 1
}
Write-Host "Testing SSH (port 22) on $DeployHost ..."
try {
  $r = Test-NetConnection -ComputerName $DeployHost -Port 22 -WarningAction SilentlyContinue
  if ($r.TcpTestSucceeded) {
    Write-Host "PASS: port 22 reachable on $DeployHost"
    exit 0
  }
  Write-Host "FAIL: port 22 not reachable on $DeployHost"
  exit 1
} catch {
  Write-Host "FAIL: $($_.Exception.Message)"
  exit 1
}
