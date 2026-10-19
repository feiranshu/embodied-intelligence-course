@echo off
setlocal
cd /d "%~dp0"

if exist ".venv\Scripts\python.exe" goto install

if not "%~1"=="" (
  "%~1" -c "import sys; raise SystemExit(0 if sys.version_info >= (3,10) else 1)"
  if errorlevel 1 goto bad_python
  "%~1" -m venv .venv
  goto verify
)

where py >nul 2>nul
if not errorlevel 1 (
  py -3 -c "import sys; raise SystemExit(0 if sys.version_info >= (3,10) else 1)"
  if not errorlevel 1 (
    py -3 -m venv .venv
    goto verify
  )
)

where python >nul 2>nul
if errorlevel 1 goto no_python
python -c "import sys; raise SystemExit(0 if sys.version_info >= (3,10) else 1)"
if errorlevel 1 goto bad_python
python -m venv .venv

:verify
if errorlevel 1 goto create_failed
if not exist ".venv\Scripts\python.exe" goto wrong_layout

:install
echo Installing Python dependencies...
".venv\Scripts\python.exe" -m pip install --upgrade pip
if errorlevel 1 goto install_failed
".venv\Scripts\python.exe" -m pip install -r requirements.txt
if errorlevel 1 goto install_failed
echo.
echo Setup complete.
echo Activate with: .venv\Scripts\Activate.ps1
pause
exit /b 0

:no_python
echo.
echo No Python 3.10 or newer was found.
echo Install Windows Python, or pass its full path to this script.
goto fail

:bad_python
echo.
echo The selected Python is missing or older than Python 3.10.
goto fail

:wrong_layout
echo.
echo This Python did not create .venv\Scripts\python.exe.
echo It may be an MSYS Python. Remove this new .venv directory, then run:
echo setup_windows.bat "C:\path\to\windows\python.exe"
goto fail

:create_failed
echo.
echo Virtual environment creation failed.
goto fail

:install_failed
echo.
echo Dependency installation failed.

:fail
pause
exit /b 1

