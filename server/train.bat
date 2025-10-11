@echo off
echo YouTube Live Chat Filter - 모델 추가 학습
echo ==========================================

cd /d "%~dp0"

echo 현재 디렉토리: %CD%
echo.

echo 학습 데이터 확인 중...
if not exist "..\training_data" (
    echo 오류: training_data 디렉토리가 없습니다.
    echo 먼저 확장 프로그램을 통해 학습 데이터를 수집해주세요.
    pause
    exit /b 1
)

echo 학습 데이터 파일들:
dir /b "..\training_data\training_data_*.jsonl" 2>nul
if errorlevel 1 (
    echo 오류: 학습 데이터 파일이 없습니다.
    echo 먼저 확장 프로그램을 통해 학습 데이터를 수집해주세요.
    pause
    exit /b 1
)

echo.
echo 모델 학습을 시작합니다...
echo.

python train.py --epochs 3 --batch-size 8 --learning-rate 2e-5

if errorlevel 1 (
    echo.
    echo 학습 중 오류가 발생했습니다.
    pause
    exit /b 1
)

echo.
echo 학습이 완료되었습니다!
echo 업데이트된 모델이 model_updated 디렉토리에 저장되었습니다.
echo.
echo 기존 모델을 백업하고 새 모델로 교체하시겠습니까? (Y/N)
set /p choice=

if /i "%choice%"=="Y" (
    echo.
    echo 모델 교체 중...
    
    if exist "..\model_backup" (
        rmdir /s /q "..\model_backup"
    )
    
    move "..\model" "..\model_backup"
    move "..\model_updated" "..\model"
    
    echo 모델 교체가 완료되었습니다!
    echo 백업된 기존 모델은 model_backup 디렉토리에 있습니다.
) else (
    echo 모델 교체를 건너뛰었습니다.
    echo 새 모델은 model_updated 디렉토리에 있습니다.
)

echo.
echo 서버를 재시작하여 새 모델을 적용하세요.
pause

