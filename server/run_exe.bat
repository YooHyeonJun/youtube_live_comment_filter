@echo off
REM YouTube Live Filter Server 실행 스크립트 (EXE 버전)
REM 
REM 이 스크립트는 빌드된 exe를 실행합니다.
REM dist/youtube_live_filter_server/youtube_live_filter_server.exe 파일이 있어야 합니다.

echo ========================================
echo YouTube Live Filter Server (EXE)
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

echo 서버를 시작합니다...
echo 서버 주소: http://127.0.0.1:8000
echo.
echo 종료하려면 Ctrl+C를 누르세요.
echo.

cd dist\youtube_live_filter_server
youtube_live_filter_server.exe


