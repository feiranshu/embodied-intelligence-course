@echo off
setlocal
cd /d "%~dp0"
call select_python.bat
if errorlevel 1 (
  call setup_advanced_windows.bat
  call select_python.bat
)
if errorlevel 1 exit /b 1
"%ARM_PYTHON%" -c "import numpy, serial, vosk, sounddevice"
if errorlevel 1 (
  echo Advanced dependencies are missing. Run setup_advanced_windows.bat first.
  pause
  exit /b 1
)
"%ARM_PYTHON%" -X utf8 advanced.py --execute voice --buttons
pause
