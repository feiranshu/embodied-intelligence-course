@echo off
setlocal
cd /d "%~dp0"
if exist ".venv\Scripts\python.exe" goto install
where py >nul 2>nul
if not errorlevel 1 (
  py -3 -m venv .venv
) else (
  python -m venv .venv
)
if errorlevel 1 goto fail
:install
".venv\Scripts\python.exe" -m pip install --no-cache-dir -r requirements_advanced.txt
if errorlevel 1 goto fail
echo Advanced dependencies installed. Read the advanced guide before using hardware.
pause
exit /b 0
:fail
echo Setup failed. Install standard Windows Python 3.10 or newer and try again.
pause
exit /b 1
