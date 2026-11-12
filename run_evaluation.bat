@echo off
cd /d "%~dp0"
set PYTHONUTF8=1
if not exist .venv ( python -m venv .venv )
call .venv\Scripts\activate.bat
echo Installing packages (first run takes several minutes)...
python -m pip install -q -r requirements.txt
python -m scripts.demo_eval
echo.
echo Done. Results are in eval\results.md
pause
