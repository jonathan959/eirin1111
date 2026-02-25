param(
  [string]$ProjectRoot = "C:\Users\jonat\OneDrive\Desktop\local_3comas_clone_v2",
  [string]$LogPath = "C:\Users\jonat\OneDrive\Desktop\local_3comas_clone_v2\auto_deploy.log",
  [int]$DebounceSeconds = 3
)

$ErrorActionPreference = "Continue"

function Write-Log([string]$msg) {
  $ts = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
  try {
    Add-Content -Path $LogPath -Value "[$ts] $msg"
  } catch {
    # Fallback to console if log write fails
    Write-Host "[$ts] $msg"
  }
}

if (-not (Test-Path $ProjectRoot)) {
  Write-Log "Project root not found: $ProjectRoot"
  exit 1
}

Write-Log "Auto-deploy watcher starting (root: $ProjectRoot)"
Write-Host "Auto-deploy watcher running. This window must stay open."

$watcher = New-Object System.IO.FileSystemWatcher
$watcher.Path = $ProjectRoot
$watcher.IncludeSubdirectories = $true
$watcher.EnableRaisingEvents = $true

$script:pending = $false
$script:lastEvent = Get-Date
$script:changed = New-Object System.Collections.Generic.List[string]

Register-ObjectEvent $watcher Created -Action {
  $script:pending = $true
  $script:lastEvent = Get-Date
  $script:changed.Add($Event.SourceEventArgs.FullPath) | Out-Null
} | Out-Null
Register-ObjectEvent $watcher Changed -Action {
  $script:pending = $true
  $script:lastEvent = Get-Date
  $script:changed.Add($Event.SourceEventArgs.FullPath) | Out-Null
} | Out-Null
Register-ObjectEvent $watcher Renamed -Action {
  $script:pending = $true
  $script:lastEvent = Get-Date
  $script:changed.Add($Event.SourceEventArgs.FullPath) | Out-Null
} | Out-Null
Register-ObjectEvent $watcher Deleted -Action {
  $script:pending = $true
  $script:lastEvent = Get-Date
  $script:changed.Add($Event.SourceEventArgs.FullPath) | Out-Null
} | Out-Null

while ($true) {
  Start-Sleep -Seconds 1
  if (-not $script:pending) { continue }
  $since = (Get-Date) - $script:lastEvent
  if ($since.TotalSeconds -lt $DebounceSeconds) { continue }
  $script:pending = $false

  # Only deploy if relevant files changed (based on event paths)
  $shouldDeploy = $false
  foreach ($p in $script:changed) {
    if ($p -match "\\templates\\|\\static\\") { $shouldDeploy = $true; break }
    if ($p -match "\.py$|\.sh$|requirements\.txt$|tradingserver\.service$") { $shouldDeploy = $true; break }
  }
  $script:changed.Clear()
  if (-not $shouldDeploy) { continue }

  Write-Log "Change detected. Running deploy (-Quick)..."
  try {
    & powershell -ExecutionPolicy Bypass -File "$ProjectRoot\deploy.ps1" -Quick | ForEach-Object { Write-Log $_ }
    Write-Log "Deploy finished."
  } catch {
    Write-Log "Deploy failed: $($_.Exception.Message)"
  }
}
