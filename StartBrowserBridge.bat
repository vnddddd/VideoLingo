@echo off
cd /d "%~dp0"
set PYTHONUTF8=1
set PYTHONIOENCODING=utf-8
if exist ".venv\Scripts\python.exe" (
  ".venv\Scripts\python.exe" -u tools\browser_bridge.py
) else (
  python -u tools\browser_bridge.py
)
