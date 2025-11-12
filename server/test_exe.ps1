# YouTube Live Filter Server - EXE 테스트 스크립트
# PowerShell에서 실행: .\test_exe.ps1

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "YouTube Live Filter Server - EXE 테스트" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

# exe 파일 존재 확인
$exePath = "dist\youtube_live_filter_server\youtube_live_filter_server.exe"
if (-not (Test-Path $exePath)) {
    Write-Host "❌ 실행 파일을 찾을 수 없습니다!" -ForegroundColor Red
    Write-Host "   경로: $exePath" -ForegroundColor Yellow
    Write-Host "   먼저 build_exe.bat를 실행하여 빌드해주세요." -ForegroundColor Yellow
    exit 1
}

Write-Host "✓ 실행 파일 확인됨: $exePath" -ForegroundColor Green
Write-Host ""

# 서버 실행 확인
Write-Host "서버 실행 여부를 확인합니다..." -ForegroundColor Yellow
Write-Host "(서버가 실행 중이어야 합니다. 실행되지 않았다면 별도 창에서 exe를 실행하세요.)" -ForegroundColor Gray
Start-Sleep -Seconds 2

$serverUrl = "http://127.0.0.1:8000"

# Health 체크
Write-Host ""
Write-Host "1. Health 체크..." -ForegroundColor Cyan
try {
    $response = Invoke-RestMethod -Uri "$serverUrl/health" -Method GET -TimeoutSec 5
    Write-Host "   ✓ 서버 응답 정상" -ForegroundColor Green
    Write-Host "   - Status: $($response.status)" -ForegroundColor Gray
    Write-Host "   - Device: $($response.device)" -ForegroundColor Gray
    Write-Host "   - Num Labels: $($response.num_labels)" -ForegroundColor Gray
} catch {
    Write-Host "   ❌ 서버 응답 실패!" -ForegroundColor Red
    Write-Host "   $($_.Exception.Message)" -ForegroundColor Red
    Write-Host ""
    Write-Host "서버를 시작하려면:" -ForegroundColor Yellow
    Write-Host "   cd dist\youtube_live_filter_server" -ForegroundColor Gray
    Write-Host "   .\youtube_live_filter_server.exe" -ForegroundColor Gray
    exit 1
}

# Labels 조회
Write-Host ""
Write-Host "2. Labels 조회..." -ForegroundColor Cyan
try {
    $labels = Invoke-RestMethod -Uri "$serverUrl/labels" -Method GET
    Write-Host "   ✓ 라벨 조회 성공" -ForegroundColor Green
    foreach ($key in $labels.PSObject.Properties) {
        Write-Host "   - $($key.Name): $($key.Value)" -ForegroundColor Gray
    }
} catch {
    Write-Host "   ❌ 라벨 조회 실패!" -ForegroundColor Red
    Write-Host "   $($_.Exception.Message)" -ForegroundColor Red
}

# 예측 테스트
Write-Host ""
Write-Host "3. 예측 테스트..." -ForegroundColor Cyan
$testTexts = @(
    "안녕하세요! 좋은 방송 감사합니다!",
    "이건 좀 별로네요",
    "욕설 테스트"
)

$predictBody = @{
    texts = $testTexts
} | ConvertTo-Json

try {
    $prediction = Invoke-RestMethod -Uri "$serverUrl/predict" -Method POST `
        -ContentType "application/json" -Body $predictBody
    
    Write-Host "   ✓ 예측 성공" -ForegroundColor Green
    for ($i = 0; $i -lt $testTexts.Length; $i++) {
        $text = $testTexts[$i]
        $label = $prediction.labels[$i]
        $labelName = $prediction.label_names."$label"
        $probs = $prediction.probs[$i]
        
        Write-Host "   [$i] Text: '$text'" -ForegroundColor Gray
        Write-Host "       Label: $label ($labelName)" -ForegroundColor Gray
        Write-Host "       Probs: [0]=$($probs[0].ToString('0.0000')) [1]=$($probs[1].ToString('0.0000')) [2]=$($probs[2].ToString('0.0000'))" -ForegroundColor Gray
    }
} catch {
    Write-Host "   ❌ 예측 실패!" -ForegroundColor Red
    Write-Host "   $($_.Exception.Message)" -ForegroundColor Red
}

# 학습 데이터 저장 테스트
Write-Host ""
Write-Host "4. 학습 데이터 저장 테스트..." -ForegroundColor Cyan
$trainingData = @{
    text = "테스트 댓글입니다 - PowerShell 스크립트에서 추가"
    label = 0
    user_id = "test_user_powershell"
} | ConvertTo-Json

try {
    $saveResult = Invoke-RestMethod -Uri "$serverUrl/training-data" -Method POST `
        -ContentType "application/json" -Body $trainingData
    
    if ($saveResult.success) {
        Write-Host "   ✓ 학습 데이터 저장 성공" -ForegroundColor Green
        Write-Host "   - Message: $($saveResult.message)" -ForegroundColor Gray
    } else {
        Write-Host "   ⚠ 학습 데이터 저장 실패" -ForegroundColor Yellow
        Write-Host "   - Message: $($saveResult.message)" -ForegroundColor Yellow
    }
} catch {
    Write-Host "   ❌ 학습 데이터 저장 실패!" -ForegroundColor Red
    Write-Host "   $($_.Exception.Message)" -ForegroundColor Red
}

# 학습 데이터 통계 조회
Write-Host ""
Write-Host "5. 학습 데이터 통계 조회..." -ForegroundColor Cyan
try {
    $stats = Invoke-RestMethod -Uri "$serverUrl/training-data/stats" -Method GET
    Write-Host "   ✓ 통계 조회 성공" -ForegroundColor Green
    Write-Host "   - Total Samples: $($stats.total_samples)" -ForegroundColor Gray
    Write-Host "   - Data Files: $($stats.data_files)" -ForegroundColor Gray
    Write-Host "   - Label Distribution:" -ForegroundColor Gray
    foreach ($key in $stats.label_distribution.PSObject.Properties) {
        Write-Host "     * $($key.Name): $($key.Value)" -ForegroundColor Gray
    }
} catch {
    Write-Host "   ❌ 통계 조회 실패!" -ForegroundColor Red
    Write-Host "   $($_.Exception.Message)" -ForegroundColor Red
}

# user_data 폴더 확인
Write-Host ""
Write-Host "6. user_data 폴더 확인..." -ForegroundColor Cyan
$userDataPath = "dist\youtube_live_filter_server\user_data"
if (Test-Path $userDataPath) {
    Write-Host "   ✓ user_data 폴더 존재함" -ForegroundColor Green
    
    $trainingDataPath = Join-Path $userDataPath "training_data"
    if (Test-Path $trainingDataPath) {
        $fileCount = (Get-ChildItem -Path $trainingDataPath -Filter "training_data_*.jsonl" -File).Count
        Write-Host "   - training_data 폴더: $fileCount 개 파일" -ForegroundColor Gray
    }
    
    $modelPath = Join-Path $userDataPath "model"
    if (Test-Path $modelPath) {
        Write-Host "   - 업데이트된 모델 폴더 존재함" -ForegroundColor Gray
    } else {
        Write-Host "   - 업데이트된 모델 폴더 없음 (기본 모델 사용 중)" -ForegroundColor Gray
    }
} else {
    Write-Host "   ⚠ user_data 폴더가 아직 생성되지 않았습니다" -ForegroundColor Yellow
    Write-Host "   (서버 실행 시 자동 생성됩니다)" -ForegroundColor Gray
}

# 완료
Write-Host ""
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "테스트 완료!" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "다음 단계:" -ForegroundColor Yellow
Write-Host "1. Chrome 확장 프로그램 설치" -ForegroundColor Gray
Write-Host "2. YouTube Live 채팅에서 실제 테스트" -ForegroundColor Gray
Write-Host "3. 학습 데이터 수집 및 모델 재학습" -ForegroundColor Gray
Write-Host ""

