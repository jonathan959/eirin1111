# One-time: merge cursor branch into main, push, then delete the cursor branch.
# After this, only main exists. Run from project root.
# Usage: .\scripts\switch-to-one-branch.ps1

$ErrorActionPreference = "Stop"
$git = "C:\Program Files\Git\cmd\git.exe"
$repo = if (Test-Path "C:\Users\jonat\OneDrive\Desktop\local_3comas_clone_v2") { "C:\Users\jonat\OneDrive\Desktop\local_3comas_clone_v2" } else { (Split-Path $PSScriptRoot -Parent) }
$cursorBranch = "cursor/development-environment-setup-3a90"

Set-Location $repo
Write-Host "Fetching..." -ForegroundColor Cyan
& $git fetch origin --prune
& $git checkout main
& $git pull origin main

Write-Host "Merging $cursorBranch into main..." -ForegroundColor Cyan
$mergeOut = & $git merge "origin/$cursorBranch" -m "Merge cursor branch into main (single-branch setup)" 2>&1
if ($LASTEXITCODE -ne 0) {
    Write-Host "Merge had conflicts. Resolve in GitHub Desktop or: git status" -ForegroundColor Yellow
    exit 1
}
Write-Host "Pushing main..." -ForegroundColor Cyan
& $git push origin main
Write-Host "Deleting remote branch $cursorBranch..." -ForegroundColor Cyan
& $git push origin --delete $cursorBranch 2>&1
Write-Host "Done. Only main remains. Pull/Push main from now on." -ForegroundColor Green
