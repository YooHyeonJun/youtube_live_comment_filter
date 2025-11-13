# EXE 배포를 위한 변경 사항 요약

## 문제점

기존 서버는 로컬 파일 시스템의 고정된 경로에 의존하여 재학습 시 문제가 발생했습니다:
- 모델 파일 경로가 하드코딩됨
- 학습 데이터 저장 위치가 고정됨
- subprocess로 별도 Python 프로세스 실행 (exe에서 불가능)

## 해결 방법

### 1. 동적 경로 처리

#### 추가된 헬퍼 함수 (app.py)

```python
def get_base_path():
    """실행 환경에 따른 기본 경로 반환 (exe vs 일반 실행)"""
    if getattr(sys, 'frozen', False):
        # PyInstaller로 패키징된 exe 실행 중
        return Path(sys.executable).parent
    else:
        # 일반 Python 스크립트 실행
        return Path(__file__).resolve().parents[1]

def get_model_path():
    """모델 경로 반환"""
    if getattr(sys, 'frozen', False):
        # exe 실행 시: exe와 함께 번들된 초기 모델
        return Path(sys._MEIPASS) / "model"
    else:
        # 일반 실행 시
        return Path(__file__).resolve().parents[1] / "model"

def get_user_data_path():
    """사용자 데이터 경로 반환 (학습 데이터, 업데이트된 모델 등)"""
    base = get_base_path()
    user_data_dir = base / "user_data"
    user_data_dir.mkdir(exist_ok=True)
    return user_data_dir
```

#### 경로 변수 변경

**Before:**
```python
MODEL_DIR = Path(__file__).resolve().parents[1] / "model"
TRAINING_DATA_DIR = Path(__file__).resolve().parents[1] / "training_data"
```

**After:**
```python
USER_DATA_PATH = get_user_data_path()

# 업데이트된 모델이 있으면 사용, 없으면 기본 모델 사용
UPDATED_MODEL_DIR = USER_DATA_PATH / "model"
DEFAULT_MODEL_DIR = get_model_path()
MODEL_DIR = UPDATED_MODEL_DIR if UPDATED_MODEL_DIR.exists() and any(UPDATED_MODEL_DIR.iterdir()) else DEFAULT_MODEL_DIR

TRAINING_DATA_DIR = USER_DATA_PATH / "training_data"
TEMP_TRAINING_DATA_DIR = USER_DATA_PATH / "training_temp"
```

### 2. 학습 방식 변경

#### subprocess → 직접 함수 호출

**Before (run_training_background):**
```python
cmd = [
    str(Path(__file__).parent / ".venv" / "Scripts" / "python.exe"),
    str(script_path),
    "--model-dir", str(MODEL_DIR),
    # ...
]
result = subprocess.run(cmd, capture_output=True, text=True, timeout=1800)
```

**After:**
```python
try:
    from train import train_model
    success = train_model(
        model_dir=DEFAULT_MODEL_DIR,
        training_data_dir=TRAINING_DATA_DIR,
        output_dir=output_dir,
        num_epochs=3,
        batch_size=8
    )
except Exception as e:
    LOGGER.error(f"Training module error: {e}")
```

### 3. 모델 업데이트 관리

**Before:**
- 기존 모델을 직접 덮어씀
- 백업 관리 복잡

**After:**
- 기본 모델: exe에 번들 (`_internal/model/`)
- 업데이트 모델: 사용자 데이터 폴더 (`user_data/model/`)
- 자동 우선순위: 업데이트 모델 → 기본 모델
- 백업 자동 관리

## 새로운 파일들

### 1. build_exe.spec
PyInstaller 빌드 설정 파일
- 모델 파일 번들링
- 필요한 라이브러리 포함
- Hidden imports 설정

### 2. build_exe.bat
Windows용 빌드 스크립트
- 가상환경 자동 활성화
- PyInstaller 설치 확인
- 빌드 실행 및 결과 확인

### 3. requirements_build.txt
빌드용 추가 의존성
- PyInstaller 포함

### 4. README_EXE.md
EXE 배포 가이드
- 빌드 방법
- 배포 방법
- 동작 원리 설명

### 5. DEPLOYMENT_CHECKLIST.md
배포 체크리스트
- 빌드 전 확인 사항
- 테스트 절차
- 배포 준비 단계

### 6. test_exe.ps1
PowerShell 테스트 스크립트
- Health 체크
- 예측 API 테스트
- 학습 데이터 저장 테스트

### 7. run_exe.bat
빌드된 exe 실행 스크립트

### 8. CHANGES_FOR_EXE.md (이 파일)
변경 사항 요약

## 디렉토리 구조 비교

### Before (Python 직접 실행)
```
project/
├── model/              # 모델 파일
├── training_data/      # 학습 데이터
├── server/
│   ├── app.py
│   ├── train.py
│   └── .venv/          # 가상환경
└── extension/
```

### After (EXE 배포)
```
youtube_live_filter_server.exe
├── _internal/          # 내부 (읽기 전용)
│   ├── model/          # 기본 모델 (번들)
│   └── (라이브러리들)
└── user_data/          # 외부 (읽기/쓰기)
    ├── training_data/  # 학습 데이터
    ├── training_temp/  # 임시 데이터
    ├── model/          # 업데이트된 모델
    └── model_backup/   # 백업
```

## 장점

1. **간편한 배포**
   - Python 설치 불필요
   - 의존성 자동 포함
   - 단일 폴더로 배포 가능

2. **데이터 영속성**
   - 사용자 데이터가 exe 외부에 저장
   - exe 업데이트 시에도 데이터 유지
   - 학습된 모델 보존

3. **재학습 내장**
   - 별도 스크립트 실행 불필요
   - Chrome 확장에서 바로 재학습 가능
   - 자동 모델 전환

4. **호환성**
   - Python 직접 실행 방식과 동일한 코드
   - 개발 시 Python, 배포 시 exe 선택 가능

## 제한 사항

1. **파일 크기**
   - 모델 포함으로 1-3GB
   - 배포 시 압축 필요

2. **첫 실행 시간**
   - 모델 로딩에 10-30초 소요

3. **GPU 사용**
   - CUDA 라이브러리가 큰 용량 차지
   - CPU 모드는 자동 지원

4. **보안 경고**
   - Windows Defender가 차단할 수 있음
   - 신뢰할 수 있는 앱으로 추가 필요

## 테스트 방법

1. **빌드**
   ```bash
   cd server
   build_exe.bat
   ```

2. **실행**
   ```bash
   cd dist\youtube_live_filter_server
   youtube_live_filter_server.exe
   ```

3. **테스트**
   ```powershell
   # 별도 PowerShell 창에서
   cd server
   .\test_exe.ps1
   ```

4. **Chrome 확장 연동**
   - 확장 설치
   - YouTube Live 채팅 테스트
   - 학습 데이터 수집
   - 재학습 실행

## 다음 단계

- [ ] GPU 지원 최적화 (선택적 CUDA 번들링)
- [ ] 자동 업데이트 기능
- [ ] 설정 UI 추가
- [ ] 로그 파일 관리
- [ ] 다국어 지원

## 참고 문서

- `README_EXE.md`: 상세 가이드
- `DEPLOYMENT_CHECKLIST.md`: 배포 체크리스트
- `build_exe.spec`: PyInstaller 설정
- PyInstaller 공식 문서: https://pyinstaller.org/


