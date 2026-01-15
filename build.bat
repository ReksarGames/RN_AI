@echo off
chcp 65001 >nul
echo ========================================
echo    ZTXAI - Build Script
echo ========================================
echo.

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

echo [1/2] Installing dependencies...
"%PYTHON_EXE%" -m pip install -r requirements.txt
if errorlevel 1 (
    echo ERROR: Failed to install dependencies
    popd
    pause
    exit /b 1
)

echo.
echo [2/2] Building executable with PyInstaller...
"%PYTHON_EXE%" -m PyInstaller --onedir --noconsole --name main ^
    --add-data "*.ttf;." ^
    --add-data "*.ttc;." ^
    --add-data "config.json;." ^
    --add-data "core;core" ^
    --add-data "makcu;makcu" ^
    --hidden-import=dearpygui ^
    --hidden-import=dearpygui.dearpygui ^
    --hidden-import=cv2 ^
    --hidden-import=numpy ^
    --hidden-import=pynput ^
    --hidden-import=pynput.keyboard ^
    --hidden-import=pynput.mouse ^
    --hidden-import=PIL ^
    --hidden-import=PIL.Image ^
    --hidden-import=cryptography ^
    --hidden-import=Crypto ^
    --hidden-import=onnxruntime ^
    --hidden-import=scipy ^
    --hidden-import=filterpy ^
    --hidden-import=filterpy.kalman ^
    --hidden-import=websocket ^
    --hidden-import=serial ^
    --hidden-import=wmi ^
    --hidden-import=bettercam ^
    --hidden-import=arc4 ^
    --hidden-import=requests ^
    --hidden-import=win32api ^
    --hidden-import=win32con ^
    --hidden-import=win32gui ^
    --distpath output ^
    --workpath build ^
    --specpath build ^
    main.py

if errorlevel 1 (
    echo ERROR: Build failed
    popd
    pause
    exit /b 1
)

echo.
echo ========================================
echo    Build Complete!
echo    Output: output\main\main.exe
echo ========================================
popd
pause
