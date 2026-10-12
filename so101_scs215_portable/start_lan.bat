@echo off
setlocal
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" call setup_windows.bat
if not exist ".venv\Scripts\python.exe" exit /b 1
".venv\Scripts\python.exe" controller.py --lan
if errorlevel 1 echo Controller exited with an error. See HANDOFF.md troubleshooting.
pause
