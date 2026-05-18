# Upload local project (this folder) to AWS server and deploy.
# Server: http://3.151.143.63/
# Run from this folder: .\UPLOAD_AND_DEPLOY_TO_AWS.ps1
#
# First-time setup (run once in PowerShell):
#   $env:AWS_DEPLOY_USER = "ubuntu"            # or "ec2-user" for Amazon Linux
#   $env:AWS_DEPLOY_KEY = "C:\path\to\your.pem"
#   $env:AWS_DEPLOY_REMOTE_DIR = "local_3comas_clone_v2"   # folder name on server (default)
#
# Then run: .\UPLOAD_AND_DEPLOY_TO_AWS.ps1

$ErrorActionPreference = "Stop"
$ProjectRoot = $PSScriptRoot
$Server = (Get-Content (Join-Path $ProjectRoot "deploy_host.txt") -Raw).Trim()
if (-not $Server) { throw "deploy_host.txt is missing or empty." }
$User = if ($env:AWS_DEPLOY_USER) { $env:AWS_DEPLOY_USER } else { "ubuntu" }
$KeyPath = $env:AWS_DEPLOY_KEY
$RemoteDir = if ($env:AWS_DEPLOY_REMOTE_DIR) { $env:AWS_DEPLOY_REMOTE_DIR } else { "local_3comas_clone_v2" }

Write-Host "=== Upload and deploy to AWS ($Server) ===" -ForegroundColor Cyan
Write-Host "Local project: $ProjectRoot"
Write-Host "Server user:   $User@$Server"
Write-Host "Remote dir:   ~/$RemoteDir"
Write-Host ""

# Check for scp/ssh (Windows OpenSSH or Git Bash)
$scp = Get-Command scp -ErrorAction SilentlyContinue
$ssh = Get-Command ssh -ErrorAction SilentlyContinue
if (-not $scp -or -not $ssh) {
    Write-Host "ERROR: scp and ssh are required. Install OpenSSH Client (Settings > Apps > Optional features) or use Git Bash." -ForegroundColor Red
    exit 1
}

# Build scp/ssh options
$sshOpts = @()
if ($KeyPath -and (Test-Path $KeyPath)) {
    $sshOpts = @("-i", $KeyPath)
    Write-Host "Using key: $KeyPath" -ForegroundColor Gray
} else {
    Write-Host "No key specified (AWS_DEPLOY_KEY). You may be prompted for password." -ForegroundColor Yellow
}
$sshArgs = $sshOpts + @("${User}@${Server}")
$scpArgs = $sshOpts + @("-r", "-o", "StrictHostKeyChecking=accept-new")

# Step 1: Copy project to temp dir excluding .git (avoids server permission errors) then upload
Write-Host "[1/3] Preparing upload (excluding .git, .venv, node_modules)..." -ForegroundColor Yellow
$stagingName = "local_3comas_clone_v2_staging"
$stagingDir = Join-Path $env:TEMP $stagingName
if (Test-Path $stagingDir) { Remove-Item -Recurse -Force $stagingDir }
New-Item -ItemType Directory -Path $stagingDir -Force | Out-Null
# Each excluded dir needs its own /XD (otherwise robocopy copies .git = thousands of files = looks stuck)
$robocopyArgs = @(
    $ProjectRoot,
    $stagingDir,
    "/E", "/R:1", "/W:2",
    "/XD", ".git",
    "/XD", ".cursor",
    "/XD", ".venv",
    "/XD", "venv",
    "/XD", "node_modules",
    "/XD", "__pycache__",
    "/NFL", "/NDL", "/NJH", "/NJS"
)
& robocopy @robocopyArgs
# robocopy exit 0=no copy, 1=ok, 2+ = extra; 8+ = failure
if ($LASTEXITCODE -ge 8) {
    Write-Host "ERROR: Failed to prepare staging directory." -ForegroundColor Red
    exit 1
}
Write-Host "Staging done." -ForegroundColor Green
Write-Host "[2/3] Uploading to ${User}@${Server} (/tmp)..." -ForegroundColor Yellow
# Upload to /tmp to avoid permission issues with ~/ on some servers
& scp @scpArgs "$stagingDir" "${User}@${Server}:/tmp/"
Remove-Item -Recurse -Force $stagingDir -ErrorAction SilentlyContinue
if ($LASTEXITCODE -ne 0) {
    Write-Host "Upload failed. Check SSH key and network." -ForegroundColor Red
    exit 1
}
Write-Host "Upload done." -ForegroundColor Green

# Step 2: Sync from /tmp into app dir (preserve server .env) and run deploy
Write-Host "[3/3] Syncing and running deploy on server..." -ForegroundColor Yellow
$remoteCmd = "STAGING=/tmp/$stagingName; if [ -d `"`$STAGING`" ]; then rsync -a --exclude='.env' --exclude='botdb.sqlite3' --exclude='botdb.sqlite3-wal' --exclude='botdb.sqlite3-shm' `"`$STAGING/`" ~/$RemoteDir/ 2>/dev/null || (cp -r `"`$STAGING`"/* ~/$RemoteDir/ 2>/dev/null; cp -r `"`$STAGING`"/.[!.]* ~/$RemoteDir/ 2>/dev/null); rm -rf `"`$STAGING`"; fi; cd ~/$RemoteDir && chmod +x deploy_aws.sh 2>/dev/null; nohup bash ./deploy_aws.sh > deploy.log 2>&1 &"
& ssh $sshArgs $remoteCmd
if ($LASTEXITCODE -eq 0) {
    Write-Host ""
    Write-Host "Deploy started on server. App should be at: http://${Server}:8000" -ForegroundColor Green
    Write-Host "To check status: ssh $User@$Server 'cd $RemoteDir && tail -20 deploy.log'" -ForegroundColor Gray
} else {
    Write-Host "Deploy command had issues. SSH to server and run: cd $RemoteDir && ./deploy_aws.sh" -ForegroundColor Yellow
}
