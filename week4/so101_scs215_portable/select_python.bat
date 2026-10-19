@echo off
rem Prefer an explicitly activated Conda environment, then the project venv.
set "ARM_PYTHON="
if defined CONDA_PREFIX if exist "%CONDA_PREFIX%\python.exe" set "ARM_PYTHON=%CONDA_PREFIX%\python.exe"
if not defined ARM_PYTHON if exist "%~dp0.venv\Scripts\python.exe" set "ARM_PYTHON=%~dp0.venv\Scripts\python.exe"
if defined ARM_PYTHON exit /b 0
exit /b 1
