@echo off
setlocal
cd /d "%~dp0"
call select_python.bat
if errorlevel 1 (
  call setup_windows.bat
  call select_python.bat
)
if errorlevel 1 exit /b 1
"%ARM_PYTHON%" diagnose.py
if errorlevel 1 echo Diagnosis failed. See HANDOFF.md troubleshooting.
pause
