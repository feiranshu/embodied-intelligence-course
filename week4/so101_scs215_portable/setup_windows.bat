@echo off
setlocal
cd /d "%~dp0"
echo Creating a portable virtual environment...
if exist ".venv\Scripts\python.exe" goto install
if not "%~1"=="" (
  "%~1" -m venv .venv
  goto verify
)
where py >nul 2>nul
if not errorlevel 1 (
  py -3 -m venv .venv
  if not errorlevel 1 goto verify
)
where python >nul 2>nul
if errorlevel 1 goto fail
python -m venv .venv
:verify
if errorlevel 1 goto fail
if not exist ".venv\Scripts\python.exe" goto fail
:install
".venv\Scripts\python.exe" -m pip install --upgrade pip
if errorlevel 1 goto fail
".venv\Scripts\python.exe" -m pip install -r requirements.txt
if errorlevel 1 goto fail
echo.
echo Setup complete. Run diagnose_windows.bat next.
pause
exit /b 0
:fail
echo.
echo Setup failed. Install Python 3.10 or newer and try again.
pause
exit /b 1
