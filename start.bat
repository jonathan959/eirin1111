@echo off
REM Start AI Bot - Full website + all bots locally (Windows)
REM Usage: start.bat
cd /d "%~dp0"

echo === AI Bot - Local Mode ===
echo.

REM Create venv if missing
if not exist ".venv\Scripts\python.exe" (
    echo Creating virtual environment...
    python -m venv .venv
    echo Installing dependencies (first run: may take 5-10 minutes)...
    .venv\Scripts\python.exe -m pip install --upgrade pip
    .venv\Scripts\pip.exe install -r requirements.txt
)

REM Ensure database exists with correct schema (init_db is idempotent)
.venv\Scripts\python.exe init_db.py

echo.
echo Starting at http://127.0.0.1:8000
echo Press Ctrl+C to stop
echo.

.venv\Scripts\uvicorn.exe one_server:app --reload --port 8000 --host 0.0.0.0
