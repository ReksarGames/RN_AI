@echo off
chcp 65001 >nul
set SCRIPT_DIR=%~dp0
pushd "%SCRIPT_DIR%"
set VENV_DIR=.venv
set PYTHON_EXE=%VENV_DIR%\Scripts\python.exe

if not exist "%VENV_DIR%\Scripts\python.exe" (
    echo Creating virtual environment...
    py -3.12 -m venv "%VENV_DIR%"
    if errorlevel 1 (
        echo ERROR: Failed to create virtual environment.
        popd
        pause
        exit /b 1
    )
)

echo Upgrading pip...
"%PYTHON_EXE%" -m pip install --upgrade pip
if errorlevel 1 (
    echo ERROR: Failed to upgrade pip.
    popd
    pause
    exit /b 1
)

echo Installing requirements into %VENV_DIR%...
"%PYTHON_EXE%" -m pip install -r requirements.txt
if errorlevel 1 (
    echo ERROR: Failed to install requirements.
    popd
    pause
    exit /b 1
)

echo Installing PyInstaller into %VENV_DIR%...
"%PYTHON_EXE%" -m pip install pyinstaller
if errorlevel 1 (
    echo ERROR: Failed to install PyInstaller.
    popd
    pause
    exit /b 1
)

echo Installing PyTorch (CUDA 12.8) into %VENV_DIR%...
"%PYTHON_EXE%" -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
if errorlevel 1 (
    echo ERROR: Failed to install PyTorch (CUDA 12.8).
    popd
    pause
    exit /b 1
)

echo.
choice /M "Run installer.py for system checks and optional setup"
if errorlevel 2 goto :after_installer
if errorlevel 1 (
    echo Running installer.py...
    "%PYTHON_EXE%" src\installer.py
)
:after_installer

echo Done.
popd
pause
