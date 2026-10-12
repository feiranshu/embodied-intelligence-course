@echo off
setlocal
cd /d "%~dp0"
".venv\Scripts\python.exe" -X utf8 advanced.py --execute voice --buttons
pause
