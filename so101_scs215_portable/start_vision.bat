@echo off
setlocal
cd /d "%~dp0"
".venv\Scripts\python.exe" -X utf8 vision_control.py --fixed-camera --point-at-ball --execute --wait-start --web
pause
