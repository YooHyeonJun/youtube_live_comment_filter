# 🚀 빠른 시작 가이드

## EXE 빌드하기

### 1. 사전 준비
- ✅ Python 3.8 이상 설치
- ✅ 학습된 모델이 `../model/` 디렉토리에 있어야 함

### 2. 빌드 실행

```batch
cd server
build_exe.bat
```

빌드 스크립트가 자동으로:
- 가상환경 생성 (없을 경우)
- 의존성 설치
- PyInstaller로 exe 빌드

⏱️ **예상 시간**: 5-15분 (첫 빌드 시)

### 3. 빌드 결과 확인

빌드 완료 후:
```
server/dist/youtube_live_filter_server/
├── youtube_live_filter_server.exe  ← 실행 파일
└── _internal/                      ← 의존성 및 모델
```

## EXE 실행하기

### 방법 1: 직접 실행
```batch
cd dist\youtube_live_filter_server
youtube_live_filter_server.exe
```

### 방법 2: 테스트 스크립트 사용
```batch
cd server
run_exe.bat
```

서버가 시작되면:
- 주소: http://127.0.0.1:8000
- Health check: http://127.0.0.1:8000/health

## 테스트하기

### PowerShell 테스트 스크립트

```powershell
cd server
.\test_exe.ps1
```

테스트 항목:
- ✓ Health 체크
- ✓ 예측 API
- ✓ 학습 데이터 저장
- ✓ 통계 조회

## 배포하기

### 1. 배포 파일 압축

```batch
# dist\youtube_live_filter_server\ 폴더를 zip으로 압축
```

### 2. 사용자에게 전달

사용자는:
1. zip 압축 해제
2. `youtube_live_filter_server.exe` 실행
3. Chrome 확장 연결

**Python 설치 불필요!** 🎉

## 문제 해결

### ❌ "Python을 찾을 수 없습니다"
→ Python 3.8+ 설치: https://www.python.org/downloads/

### ❌ "model 디렉토리를 찾을 수 없습니다"
→ `../model/` 폴더에 학습된 모델 파일이 있는지 확인

### ❌ 빌드는 되는데 실행 시 오류
→ `test_exe.ps1`로 자세한 오류 확인

### ❌ Windows Defender 경고
→ "추가 정보" → "실행" 클릭

## 의존성 설명

### 🔧 자동 처리 (사용자 설치 불필요)
- Python 런타임
- PyTorch
- Transformers
- FastAPI/Uvicorn
- 모든 Python 패키지
- ML 모델 파일

### ⚠️ 필요할 수 있음 (대부분 이미 설치됨)
- Visual C++ Redistributable (자동 안내)
- CUDA (GPU 사용 시만, CPU는 자동)

## 추가 문서

- 📖 **상세 가이드**: `README_EXE.md`
- ✅ **배포 체크리스트**: `DEPLOYMENT_CHECKLIST.md`
- 📝 **변경사항**: `CHANGES_FOR_EXE.md`

---

문제가 있으면 `DEPLOYMENT_CHECKLIST.md`의 테스트 절차를 따라해보세요!


