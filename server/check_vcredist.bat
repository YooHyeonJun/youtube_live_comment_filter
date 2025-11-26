@echo off
chcp 65001 >nul
echo ========================================
echo Visual C++ Redistributable 확인
echo ========================================
echo.

echo Visual C++ Redistributable 설치 여부를 확인합니다...
echo.

REM Visual C++ 2015-2022 Redistributable 확인
reg query "HKEY_LOCAL_MACHINE\SOFTWARE\Microsoft\VisualStudio\14.0\VC\Runtimes\x64" >nul 2>&1
if %errorlevel% == 0 (
    echo [OK] Visual C++ 2015-2022 Redistributable (x64) 설치됨
) else (
    echo [X] Visual C++ 2015-2022 Redistributable (x64) 설치되지 않음
)

reg query "HKEY_LOCAL_MACHINE\SOFTWARE\WOW6432Node\Microsoft\VisualStudio\14.0\VC\Runtimes\x64" >nul 2>&1
if %errorlevel% == 0 (
    echo [OK] Visual C++ 2015-2022 Redistributable (x64) 설치됨 (WOW64)
) else (
    echo [X] Visual C++ 2015-2022 Redistributable (x64) 설치되지 않음 (WOW64)
)

REM Visual C++ 2013 Redistributable 확인
reg query "HKEY_LOCAL_MACHINE\SOFTWARE\Microsoft\VisualStudio\12.0\VC\Runtimes\x64" >nul 2>&1
if %errorlevel% == 0 (
    echo [OK] Visual C++ 2013 Redistributable (x64) 설치됨
) else (
    echo [X] Visual C++ 2013 Redistributable (x64) 설치되지 않음
)

echo.
echo ========================================
echo 확인 완료
echo ========================================
echo.
echo 만약 Visual C++ Redistributable이 설치되지 않았다면:
echo 다음 링크에서 다운로드하여 설치하세요:
echo https://aka.ms/vs/17/release/vc_redist.x64.exe
echo.
pause


