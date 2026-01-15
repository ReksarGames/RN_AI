@echo off
chcp 65001 >nul
echo ========================================
echo Python Profiler with py-spy (Offline)
echo ========================================

:: Check if py-spy is installed
python -m py_spy --version >nul 2>&1
if %errorlevel% neq 0 (
    echo py-spy not found. Install it manually before running this script.
    pause
    exit /b 1
)

echo.
echo Choose profiling mode:
echo [1] Real-time top (like htop for Python)
echo [2] Record flame graph (saves to profile.svg)
echo [3] Record flame graph speedscope format (profile.json)
echo.
set /p choice="Enter choice (1-3): "

if "%choice%"=="1" (
    echo Starting real-time profiler...
    echo Press Ctrl+C to stop
    python -m py_spy top -- python main.py
) else if "%choice%"=="2" (
    echo Recording flame graph...
    echo Press Ctrl+C to stop and save
    python -m py_spy record -o profile.svg -- python main.py
    echo Saved to profile.svg - open in browser
) else if "%choice%"=="3" (
    echo Recording speedscope format...
    echo Press Ctrl+C to stop and save
    python -m py_spy record -f speedscope -o profile.json -- python main.py
    echo Saved to profile.json - open at https://speedscope.app
) else (
    echo Invalid choice
)

pause
