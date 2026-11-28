@echo off
cd /d "%~dp0"
python -X utf8 scripts\web_grasp_pouch.py --port 8765 --open-browser
pause
