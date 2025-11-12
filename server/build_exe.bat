@echo off
chcp 65001 >nul
echo ========================================
echo YouTube Live Filter Server - EXE Build
echo ========================================
echo.

REM Check if virtual environment exists
if not exist .venv\Scripts\python.exe (
    echo [1/5] Creating virtual environment...
    python -m venv .venv
    if errorlevel 1 (
        echo ERROR: Failed to create virtual environment
        echo Please make sure Python 3.8+ is installed
        pause
        exit /b 1
    )
    echo Virtual environment created successfully
    echo.
)

REM Activate virtual environment
echo [2/5] Activating virtual environment...
call .venv\Scripts\activate.bat
if errorlevel 1 (
    echo ERROR: Failed to activate virtual environment
    pause
    exit /b 1
)

REM Install/upgrade dependencies
echo [3/5] Installing dependencies...
pip install --upgrade pip >nul 2>&1
pip install -r requirements.txt
if errorlevel 1 (
    echo ERROR: Failed to install requirements.txt
    pause
    exit /b 1
)

REM Install PyInstaller
pip show pyinstaller >nul 2>&1
if errorlevel 1 (
    echo Installing PyInstaller...
    pip install pyinstaller>=5.13.0
    if errorlevel 1 (
        echo ERROR: Failed to install PyInstaller
        pause
        exit /b 1
    )
)

REM Clean previous build
if exist dist\youtube_live_filter_server (
    echo [4/5] Cleaning previous build...
    rmdir /s /q dist\youtube_live_filter_server
    rmdir /s /q build >nul 2>&1
)

REM Build with PyInstaller
echo [5/5] Building executable (this may take several minutes)...
echo.
pyinstaller build_exe.spec --clean

if errorlevel 1 (
    echo.
    echo ========================================
    echo BUILD FAILED!
    echo ========================================
    echo.
    echo Please check the error messages above
    pause
    exit /b 1
)

echo.
echo ========================================
echo BUILD COMPLETED SUCCESSFULLY!
echo ========================================
echo.
echo Executable location: dist\youtube_live_filter_server\youtube_live_filter_server.exe
echo.
echo Next steps:
echo 1. Copy the entire dist\youtube_live_filter_server folder
echo 2. Run youtube_live_filter_server.exe
echo 3. Server will start at http://127.0.0.1:8000
echo 4. User data will be saved in user_data folder
echo.
echo For testing: run test_exe.ps1 in PowerShell
echo.
pause
