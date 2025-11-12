# YouTube Live Filter Server - EXE 배포 가이드

## 개요

이 서버를 하나의 실행 파일(exe)로 패키징하여 배포할 수 있습니다. exe 파일은 Python이 설치되지 않은 환경에서도 실행 가능하며, 모든 필요한 의존성이 포함되어 있습니다.

## 빌드 방법

### 1. 필요 사항
- Python 3.8 이상
- 학습된 모델 파일 (`../model/` 디렉토리)

### 2. 빌드 옵션 선택

#### 옵션 A: 폴더 배포 방식 (권장 ⭐)

```batch
build_exe.bat
```

**결과:**
```
dist\youtube_live_filter_server\
├── youtube_live_filter_server.exe  (50-200MB)
└── _internal\                      (1-2GB)
```

**장점:**
- ✅ 빠른 실행 속도
- ✅ 작은 exe 파일 크기
- ✅ 표준 PyInstaller 방식
- ✅ 안정적

**단점:**
- ⚠️ 폴더 전체를 배포해야 함 (하지만 사용자는 exe만 클릭)

#### 옵션 B: 단일 EXE 파일 (비권장)

```batch
build_exe_onefile.bat
```

**결과:**
```
dist\youtube_live_filter_server.exe  (3GB+ 단일 파일)
```

**장점:**
- ✅ exe 파일 1개

**단점:**
- ❌ 거대한 파일 크기 (3GB+)
- ❌ 느린 실행 (첫 실행 30-60초)
- ❌ 실행할 때마다 임시 폴더에 압축 해제
- ❌ 디스크 공간 2배 사용

### 3. 권장 사항

**"폴더 배포 방식"을 권장**합니다. 이유:
1. 사용자 입장에서는 어차피 **exe 하나만 클릭**하면 됩니다
2. 실행 속도가 훨씬 빠릅니다
3. 업데이트가 쉽습니다
4. 대부분의 상용 소프트웨어도 이 방식을 사용합니다 (Chrome, VSCode 등)

## EXE 실행 시 구조

### 디렉토리 구조
```
youtube_live_filter_server.exe  (메인 실행 파일)
_internal/                      (내부 라이브러리 및 모델)
  └── model/                    (기본 학습 모델)
user_data/                      (사용자 데이터 - 자동 생성)
  ├── training_data/            (영구 학습 데이터)
  ├── training_temp/            (임시 학습 데이터)
  ├── model/                    (업데이트된 모델)
  └── model_backup/             (모델 백업)
```

### 동작 방식

1. **초기 실행**
   - exe에 번들된 기본 모델을 사용
   - `user_data` 폴더가 exe와 같은 위치에 자동 생성

2. **학습 데이터 저장**
   - Chrome 확장에서 라벨링한 데이터가 `user_data/training_data/`에 저장
   - 임시 데이터는 `user_data/training_temp/`에 저장

3. **모델 재학습**
   - 사용자가 재학습을 요청하면:
     - `user_data/training_data/`의 데이터로 학습
     - 기본 모델을 베이스로 추가 학습 (Fine-tuning)
     - 업데이트된 모델을 `user_data/model/`에 저장

4. **모델 사용 우선순위**
   - `user_data/model/` 존재 → 업데이트된 모델 사용 ✓
   - `user_data/model/` 없음 → 기본 모델 사용

## 의존성 관리

### ✅ 자동으로 포함되는 것 (설치 불필요)

exe 파일에 이미 번들되어 있어 **추가 설치 없이** 바로 실행됩니다:
- Python 인터프리터
- PyTorch
- Transformers
- FastAPI / Uvicorn
- 모든 Python 패키지들
- 기본 ML 모델 파일

### ⚠️ 사용자 시스템에 필요할 수 있는 것

#### Windows의 경우
대부분의 Windows 10/11 시스템에는 이미 설치되어 있지만, 없을 경우 필요:

1. **Visual C++ Redistributable** (필수)
   - 대부분 이미 설치되어 있음
   - 없을 경우: https://aka.ms/vs/17/release/vc_redist.x64.exe
   - 오류 시 자동으로 안내 메시지 표시됨

2. **CUDA Toolkit** (GPU 사용 시만 필요, 선택)
   - CPU만 사용할 경우: 불필요 (자동으로 CPU 모드로 실행)
   - GPU 사용 시: NVIDIA GPU + CUDA 11.8 이상
   - 대부분의 사용자는 CPU 모드로 충분히 빠름

### 📦 배포 패키지에 포함되는 것

```
youtube_live_filter_server/
├── youtube_live_filter_server.exe  (약 50-200MB)
└── _internal/                       (약 1-2GB)
    ├── model/                       # ML 모델 파일들
    ├── torch/                       # PyTorch 라이브러리
    ├── transformers/                # Transformers 라이브러리
    └── (기타 모든 의존성)
```

**총 크기: 약 1-3GB** (모델 포함)

## 배포 방법

### 1. 배포 패키지 생성

빌드 후 `dist/youtube_live_filter_server/` 폴더를 압축:

```
youtube_live_filter_server.zip  (약 500MB-1GB 압축됨)
└── youtube_live_filter_server/
    ├── youtube_live_filter_server.exe
    └── _internal/
        └── (모든 의존성 파일들)
```

### 2. 사용자 설치 방법

1. 압축 파일 해제
2. `youtube_live_filter_server.exe` 실행
   - **첫 실행 시 Windows Defender 경고가 나올 수 있음**
   - "추가 정보" → "실행" 클릭
3. 서버가 `http://127.0.0.1:8000`에서 시작됨
4. Chrome 확장과 연동

### 3. 시스템 요구사항

| 항목 | 최소 사양 | 권장 사양 |
|------|----------|----------|
| OS | Windows 10 (64bit) | Windows 11 |
| RAM | 4GB | 8GB 이상 |
| 저장 공간 | 3GB | 5GB 이상 |
| CPU | 2 코어 이상 | 4 코어 이상 |
| GPU | 불필요 (CPU 모드) | NVIDIA GPU (선택) |

### 3. 재학습 기능 사용

1. Chrome 확장에서 댓글 라벨링
2. 확장의 "모델 재학습" 버튼 클릭
3. 학습 완료 후 자동으로 새 모델 적용
4. 재시작 시에도 업데이트된 모델 계속 사용

## 주요 변경 사항

### 경로 처리

기존 코드의 절대 경로를 동적으로 처리하도록 변경:

- `get_base_path()`: exe 위치 기반 경로
- `get_model_path()`: 번들된 모델 경로
- `get_user_data_path()`: 사용자 데이터 저장 경로

### 학습 방식

subprocess로 별도 Python 프로세스 실행 → 직접 함수 호출로 변경
- exe 환경에서 subprocess 실행 문제 해결
- 모든 기능이 단일 exe에서 동작

### 데이터 영속성

- 학습 데이터와 업데이트된 모델이 exe 외부(`user_data/`)에 저장
- exe 업데이트 시에도 사용자 데이터 유지
- 백업 기능으로 모델 롤백 가능

## 문제 해결

### Q: exe 실행 시 "model not found" 오류

A: 빌드 시 모델 파일이 포함되지 않았을 수 있습니다.
- `../model/` 디렉토리에 모델 파일이 있는지 확인
- `build_exe.spec` 파일의 `model_datas` 설정 확인
- 다시 빌드

### Q: 재학습이 완료되지 않음

A: 
- `user_data/training_data/` 폴더에 최소 5개 이상의 학습 데이터 필요
- 로그 확인: 콘솔 창에 오류 메시지 표시
- GPU 메모리 부족 시 batch_size 줄이기

### Q: 업데이트된 모델이 적용되지 않음

A:
- exe 재시작
- `/model/reload` API 호출
- `user_data/model/` 폴더 확인

### Q: 포트 충돌 (8000번 포트 사용 중)

A:
- 환경변수 `PORT` 설정: `set PORT=8001`
- 또는 다른 프로그램의 포트 사용 중지

## 기술 스택

- **PyInstaller**: Python 앱을 실행 파일로 패키징
- **FastAPI**: 웹 서버 프레임워크
- **PyTorch**: 딥러닝 모델 실행
- **Transformers**: Hugging Face 모델 로드

## 라이선스

이 프로젝트의 라이선스에 따라 배포하세요.

