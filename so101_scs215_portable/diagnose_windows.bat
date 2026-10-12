@echo off
setlocal
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" call setup_windows.bat
if not exist ".venv\Scripts\python.exe" exit /b 1
".venv\Scripts\python.exe" diagnose.py
if errorlevel 1 echo Diagnosis failed. See HANDOFF.md troubleshooting.
pause
