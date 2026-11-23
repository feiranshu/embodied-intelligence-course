@echo off
chcp 65001 >nul
setlocal
cd /d "%~dp0"
set "AUDIT=最终标定结果"
set "PYTHON=D:\software\anaconda\envs\py312\python.exe"
if not exist "%PYTHON%" set "PYTHON=..\..\.venv\Scripts\python.exe"
if defined CONDA_PREFIX if exist "%CONDA_PREFIX%\python.exe" set "PYTHON=%CONDA_PREFIX%\python.exe"
set "OPENCV_OPENCL_RUNTIME=disabled"

powershell.exe -NoProfile -Command "try { if ((Invoke-RestMethod 'http://127.0.0.1:8765/api/state' -TimeoutSec 1).session_name -eq $env:AUDIT) { exit 0 }; exit 2 } catch { exit 1 }"
if "%ERRORLEVEL%"=="0" (
    start "" "http://127.0.0.1:8765/"
    exit /b 0
)
if "%ERRORLEVEL%"=="2" (
    echo Port 8765 is in use. Close the other calibration console and try again.
    goto failed
)

if not exist "%PYTHON%" (
    echo Calibration Python environment was not found.
    goto failed
)
if not exist "%AUDIT%\advanced.json" goto missing_config
if not exist "%AUDIT%\camera_intrinsics.json" goto missing_config
if not exist "%AUDIT%\so101_new_calib.urdf" goto missing_config

start "" /b powershell.exe -NoProfile -Command "for ($attempt=0; $attempt -lt 20; $attempt++) { try { if ((Invoke-RestMethod 'http://127.0.0.1:8765/api/state' -TimeoutSec 1).session_name -eq $env:AUDIT) { Start-Process 'http://127.0.0.1:8765/'; break } } catch {}; Start-Sleep -Milliseconds 500 }"
"%PYTHON%" -B -u -X utf8 -c "import comtypes.client, runpy; comtypes.client.gen_dir = None; runpy.run_path('web_app.py', run_name='__main__')" --record-with-keyboard --output "%AUDIT%" --advanced-config "%AUDIT%\advanced.json" --intrinsics "%AUDIT%\camera_intrinsics.json"
if errorlevel 1 goto failed
exit /b 0

:missing_config
echo The keyboard capture calibration files are missing from %AUDIT%.
:failed
pause
exit /b 1
