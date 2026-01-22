@echo off
chcp 65001 >nul
echo Starting RN_AI...
set SCRIPT_DIR=%~dp0
pushd "%SCRIPT_DIR%"
if not exist ".venv\Scripts\python.exe" (
    echo ERROR: .venv not found. Run install.bat first.
    popd
    pause
    exit /b 1
)
".venv\Scripts\python.exe" main.py
popd
pause
