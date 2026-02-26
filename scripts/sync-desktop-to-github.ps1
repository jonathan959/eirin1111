# Sync Desktop project with GitHub: make local the source of truth, merge cursor branch, push.
# Run from project root in a terminal where Git is available (Git Bash, or PowerShell after installing Git).
# Usage: cd C:\Users\jonat\OneDrive\Desktop\local_3comas_clone_v2; .\scripts\sync-desktop-to-github.ps1

param([switch]$DryRun)

$ErrorActionPreference = "Stop"
$RepoRoot = if (Test-Path "C:\Users\jonat\Desktop\local_3comas_clone_v2") { "C:\Users\jonat\Desktop\local_3comas_clone_v2" } else { (Split-Path $PSScriptRoot -Parent) }
$GitRepo = "https://github.com/jonathan959/eirin1111.git"
$CursorBranch = "cursor/development-environment-setup-3a90"

Set-Location $RepoRoot
Write-Host "Repo root: $RepoRoot" -ForegroundColor Cyan

# Find git
$git = $null
foreach ($p in @("git", "C:\Program Files\Git\cmd\git.exe", "C:\Program Files (x86)\Git\cmd\git.exe")) {
    try {
        if ($p -eq "git") { $null = Get-Command git -ErrorAction Stop; $git = "git"; break }
        elseif (Test-Path $p) { $git = $p; break }
    } catch { }
}
if (-not $git) { Write-Host "ERROR: Git not found. Install Git for Windows or run this in Git Bash." -ForegroundColor Red; exit 1 }

function Run-Git { & $git @args; if ($LASTEXITCODE -ne 0 -and $args[0] -ne "merge" -and $args[0] -ne "status") { throw "git $($args[0]) failed" } }

Write-Host "`n=== 2) GIT STATUS AND REMOTES ===" -ForegroundColor Cyan
Run-Git status
Run-Git remote -v
Run-Git branch -a
Write-Host "`nLast 10 commits on current branch:" -ForegroundColor Gray
Run-Git log --oneline -n 10

Write-Host "`n=== 3) FIX REMOTE IF NEEDED ===" -ForegroundColor Cyan
$remoteUrl = (Run-Git remote get-url origin 2>$null)
if ($remoteUrl -notmatch "jonathan959/eirin1111") {
    Write-Host "Removing origin and adding $GitRepo" -ForegroundColor Yellow
    Run-Git remote remove origin 2>$null
    Run-Git remote add origin $GitRepo
}
Run-Git fetch origin --prune

Write-Host "`n=== 4) CURSOR BRANCH ===" -ForegroundColor Cyan
$remoteCursor = "origin/$CursorBranch"
$hasRemote = (Run-Git branch -r) -match [regex]::Escape($CursorBranch)
if ($hasRemote) {
    Write-Host "Found $remoteCursor. Creating local tracking branch if missing." -ForegroundColor Green
    $hasLocal = (Run-Git branch) -match [regex]::Escape($CursorBranch)
    if (-not $hasLocal) { Run-Git checkout -b $CursorBranch $remoteCursor }
    else { Run-Git checkout $CursorBranch; Run-Git pull origin $CursorBranch }
} else {
    Write-Host "Remote branch $CursorBranch not found. Listing remote branches:" -ForegroundColor Yellow
    Run-Git branch -r
}

Write-Host "`n=== 5) MERGE INTO MAIN ===" -ForegroundColor Cyan
Run-Git checkout main
Run-Git pull origin main
if ($hasRemote -and (Run-Git log --oneline -1 main) -ne (Run-Git log --oneline -1 $CursorBranch)) {
    Write-Host "Merging $CursorBranch into main..." -ForegroundColor Cyan
    if ($DryRun) { Write-Host "[DRY RUN] would run: git merge $CursorBranch" -ForegroundColor Gray }
    else {
        try { Run-Git merge $CursorBranch -m "Merge cursor/development-environment-setup-3a90 into main" }
        catch {
            Write-Host "Merge had conflicts or failed. Resolve in repo, then: git add . ; git commit ; git push origin main" -ForegroundColor Yellow
            exit 1
        }
    }
} else {
    Write-Host "No merge needed (branch up to date or not found)." -ForegroundColor Gray
}

Write-Host "`n=== 6) ENSURE .env NOT TRACKED ===" -ForegroundColor Cyan
$tracked = Run-Git ls-files .env 2>$null
if ($tracked) {
    Write-Host "Untracking .env..." -ForegroundColor Yellow
    if (-not $DryRun) { Run-Git rm --cached .env 2>$null; Run-Git commit -m "Stop tracking .env" 2>$null }
}

Write-Host "`n=== 7) PUSH MAIN TO GITHUB ===" -ForegroundColor Cyan
if ($DryRun) { Write-Host "[DRY RUN] would run: git push -u origin main" -ForegroundColor Gray }
else { Run-Git push -u origin main }

Write-Host "`n=== DONE ===" -ForegroundColor Green
Run-Git log --oneline -n 3
