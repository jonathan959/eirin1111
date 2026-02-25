# Run AI Bot locally (no AWS needed)
# Uses one_server.py = full UI with templates, login, static files
# Database: botdb.sqlite3 (default). Replace with AWS copy to restore your bots.
# Usage: .\run_local.ps1
# For fresh DB: $env:BOT_DB_PATH = "botdb_local.sqlite3"; .\run_local.ps1

$projectRoot = $PSScriptRoot
Set-Location $projectRoot

# Create venv if missing
if (-not (Test-Path "$projectRoot\.venv\Scripts\python.exe")) {
    Write-Host "Creating virtual environment..." -ForegroundColor Cyan
    python -m venv .venv
    Write-Host "Installing dependencies (first run may take a few minutes)..." -ForegroundColor Cyan
    & "$projectRoot\.venv\Scripts\pip.exe" install -r requirements.txt
}

Write-Host "Starting AI Bot at http://127.0.0.1:8000" -ForegroundColor Green
Write-Host "Press Ctrl+C to stop" -ForegroundColor Gray

& "$projectRoot\.venv\Scripts\uvicorn.exe" one_server:app --reload --port 8000 --host 0.0.0.0
