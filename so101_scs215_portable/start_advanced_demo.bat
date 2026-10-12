@echo off
setlocal
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
  echo Run setup_advanced_windows.bat first.
  pause
  exit /b 1
)
echo 1. Simulated keyboard control
echo 2. Chinese microphone recognition with simulated motion
echo 3. Simulated black-ball visual feedback
choice /c 123 /n /m "Choose 1, 2 or 3: "
if errorlevel 3 goto vision
if errorlevel 2 goto voice
".venv\Scripts\python.exe" -X utf8 advanced.py --simulate keyboard
goto done
:voice
".venv\Scripts\python.exe" -X utf8 advanced.py --simulate voice
goto done
:vision
".venv\Scripts\python.exe" -X utf8 vision_control.py --simulate
:done
pause
