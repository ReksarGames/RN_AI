@echo off
chcp 65001 >nul
echo ========================================
echo Python Profiler with py-spy
echo ========================================

set SCRIPT_DIR=%~dp0
pushd "%SCRIPT_DIR%"
set VENV_DIR=.venv
set PYTHON_EXE=%VENV_DIR%\Scripts\python.exe

if not exist "%PYTHON_EXE%" (
    echo ERROR: .venv not found. Run install.bat first.
    popd
    pause
    exit /b 1
)

set /p allow_install="Install py-spy if missing? (Y/N): "

:: Check if py-spy is installed
"%PYTHON_EXE%" -m py_spy --version >nul 2>&1
if %errorlevel% neq 0 (
    if /I "%allow_install%"=="Y" (
        echo py-spy not found, installing...
        "%PYTHON_EXE%" -m pip install py-spy
        if %errorlevel% neq 0 (
            echo Failed to install py-spy
            popd
            pause
            exit /b 1
        )
        echo py-spy installed successfully
    ) else (
        echo py-spy not found. Install it manually or rerun and choose Y.
        popd
        pause
        exit /b 1
    )
) else (
    echo py-spy is already installed
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
    "%PYTHON_EXE%" -m py_spy top -- "%PYTHON_EXE%" main.py
) else if "%choice%"=="2" (
    echo Recording flame graph...
    echo Press Ctrl+C to stop and save
    "%PYTHON_EXE%" -m py_spy record -o profile.svg -- "%PYTHON_EXE%" main.py
    echo Saved to profile.svg - open in browser
) else if "%choice%"=="3" (
    echo Recording speedscope format...
    echo Press Ctrl+C to stop and save
    "%PYTHON_EXE%" -m py_spy record -f speedscope -o profile.json -- "%PYTHON_EXE%" main.py
    echo Saved to profile.json - open at https://speedscope.app
) else (
    echo Invalid choice
)

popd
pause
