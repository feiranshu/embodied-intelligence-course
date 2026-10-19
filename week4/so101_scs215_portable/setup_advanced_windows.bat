@echo off
setlocal
cd /d "%~dp0"
if exist ".venv\Scripts\python.exe" goto check_env
if not "%~1"=="" (
  "%~1" -c "import sys; raise SystemExit(not (3,10) <= sys.version_info[:2] <= (3,13))"
  if errorlevel 1 goto bad_python
  "%~1" -m venv .venv
  goto check_env
)
where py >nul 2>nul
if not errorlevel 1 (
  py -3.12 -c "import sys" >nul 2>nul
  if not errorlevel 1 (
    py -3.12 -m venv .venv
    goto check_env
  )
  py -3.11 -c "import sys" >nul 2>nul
  if not errorlevel 1 (
    py -3.11 -m venv .venv
    goto check_env
  )
)
where python >nul 2>nul
if errorlevel 1 goto bad_python
python -c "import sys; raise SystemExit(not (3,10) <= sys.version_info[:2] <= (3,13))"
if errorlevel 1 goto bad_python
python -m venv .venv
:check_env
if errorlevel 1 goto fail
if not exist ".venv\Scripts\python.exe" goto fail
".venv\Scripts\python.exe" -c "import sys; raise SystemExit(not (3,10) <= sys.version_info[:2] <= (3,13))"
if errorlevel 1 goto bad_env
:install
".venv\Scripts\python.exe" -m pip install --no-cache-dir -r requirements_advanced.txt
if errorlevel 1 goto fail
echo Advanced dependencies installed. Read the advanced guide before using hardware.
pause
exit /b 0
:fail
echo Setup failed. Install standard Windows Python 3.11 or 3.12 and try again.
pause
exit /b 1
:bad_python
echo Use standard Windows Python 3.10-3.13, preferably 3.11 or 3.12.
echo You can pass its full path: setup_advanced_windows.bat "C:\path\to\python.exe"
pause
exit /b 1
:bad_env
echo Existing .venv uses an unsuitable Python version. Recreate it with Python 3.11 or 3.12.
pause
exit /b 1
