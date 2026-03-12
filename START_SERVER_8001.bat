@echo off
title AI Bot - Local Server (port 8001)
cd /d "%~dp0"
echo === Starting AI Bot (one_server_v2) ===
echo.
echo If port 8000 fails, use this. API + UI: http://127.0.0.1:8001
echo Close this window to stop the server.
echo.
python -m uvicorn one_server_v2:app --reload --port 8001 --host 127.0.0.1
pause
