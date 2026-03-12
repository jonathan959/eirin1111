@echo off
title AI Bot - Local Server
cd /d "%~dp0"
echo === Starting AI Bot (one_server_v2) ===
echo.
echo API + UI will be at: http://127.0.0.1:8000
echo Close this window to stop the server.
echo.
python -m uvicorn one_server_v2:app --reload --port 8000 --host 127.0.0.1
pause
