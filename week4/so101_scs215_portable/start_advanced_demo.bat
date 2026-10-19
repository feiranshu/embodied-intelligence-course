@echo off
setlocal
cd /d "%~dp0"
call select_python.bat
if errorlevel 1 (
  echo Run setup_advanced_windows.bat first.
  pause
  exit /b 1
)
"%ARM_PYTHON%" -c "import numpy, serial, cv2, vosk, sounddevice"
if errorlevel 1 (
  echo Advanced dependencies are missing. Run setup_advanced_windows.bat first.
  pause
  exit /b 1
)
echo 1. Simulated keyboard control
echo 2. Chinese microphone recognition with simulated motion
echo 3. Simulated black-ball visual feedback
choice /c 123 /n /m "Choose 1, 2 or 3: "
if errorlevel 3 goto vision
if errorlevel 2 goto voice
"%ARM_PYTHON%" -X utf8 advanced.py --simulate keyboard
goto done
:voice
"%ARM_PYTHON%" -X utf8 advanced.py --simulate voice
goto done
:vision
"%ARM_PYTHON%" -X utf8 vision_control.py --simulate
:done
pause
