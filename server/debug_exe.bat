@echo off
chcp 65001 >nul
echo ========================================
echo YouTube Live Filter Server - EXE 디버그 실행
echo ========================================
echo.

if not exist "dist\youtube_live_filter_server\youtube_live_filter_server.exe" (
    echo 실행 파일을 찾을 수 없습니다!
    echo.
    echo 먼저 build_exe.bat를 실행하여 빌드해주세요.
    echo.
    pause
    exit /b 1
)

echo 실행 파일을 실행합니다...
echo 에러 메시지가 표시됩니다.
echo.
echo 종료하려면 Ctrl+C를 누르세요.
echo.

cd dist\youtube_live_filter_server
youtube_live_filter_server.exe

if errorlevel 1 (
    echo.
    echo ========================================
    echo 프로그램이 에러로 종료되었습니다.
    echo ========================================
    echo.
    echo 위의 에러 메시지를 확인하세요.
    echo.
)

pause





