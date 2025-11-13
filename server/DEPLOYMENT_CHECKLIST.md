# 배포 체크리스트

## 빌드 전 확인 사항

- [ ] 모델 파일이 `../model/` 디렉토리에 있음
- [ ] `requirements.txt`의 모든 패키지가 설치됨
- [ ] `app.py`와 `train.py`가 정상 작동함
- [ ] Python 버전 3.8 이상

## 빌드 과정

1. [ ] `pip install -r requirements_build.txt` 실행
2. [ ] `build_exe.bat` 실행
3. [ ] 빌드 완료 확인: `dist/youtube_live_filter_server/` 폴더 생성됨
4. [ ] exe 파일 크기 확인 (정상: 1-3GB, 모델 포함)

## 빌드 후 테스트

### 기본 기능 테스트

1. [ ] exe 실행 확인
   ```
   cd dist\youtube_live_filter_server
   youtube_live_filter_server.exe
   ```

2. [ ] 서버 시작 확인
   - [ ] 콘솔에 "Uvicorn running on http://127.0.0.1:8000" 표시
   - [ ] 브라우저에서 http://127.0.0.1:8000/health 접속 → `{"status":"ok",...}` 응답

3. [ ] 모델 로드 확인
   - [ ] 콘솔에 오류 메시지 없음
   - [ ] `/health` 응답에 `num_labels` 포함

### 예측 기능 테스트

4. [ ] 예측 API 테스트
   ```powershell
   # PowerShell에서 실행
   $body = @{
       texts = @("테스트 댓글", "악플 예제")
   } | ConvertTo-Json

   Invoke-RestMethod -Uri "http://127.0.0.1:8000/predict" `
       -Method POST `
       -ContentType "application/json" `
       -Body $body
   ```
   - [ ] `labels`와 `probs` 반환됨

### 학습 데이터 저장 테스트

5. [ ] 학습 데이터 저장
   ```powershell
   $body = @{
       text = "테스트 댓글입니다"
       label = 0
       user_id = "test_user"
   } | ConvertTo-Json

   Invoke-RestMethod -Uri "http://127.0.0.1:8000/training-data" `
       -Method POST `
       -ContentType "application/json" `
       -Body $body
   ```
   - [ ] `success: true` 응답
   - [ ] `user_data/training_data/training_data_YYYY-MM-DD.jsonl` 파일 생성됨

6. [ ] 학습 데이터 조회
   ```powershell
   Invoke-RestMethod -Uri "http://127.0.0.1:8000/training-data/stats"
   ```
   - [ ] `total_samples`, `label_distribution` 반환

### 재학습 테스트 (선택)

7. [ ] 학습 데이터 5개 이상 추가

8. [ ] 재학습 시작
   ```powershell
   Invoke-RestMethod -Uri "http://127.0.0.1:8000/model/retrain" `
       -Method POST
   ```
   - [ ] `success: true` 응답

9. [ ] 학습 진행 상태 확인
   ```powershell
   Invoke-RestMethod -Uri "http://127.0.0.1:8000/model/training-status"
   ```
   - [ ] `is_training: true` → `is_training: false` (완료 시)
   - [ ] `progress` 증가
   - [ ] `error` 없음

10. [ ] 업데이트된 모델 확인
    - [ ] `user_data/model/` 폴더 생성됨
    - [ ] 서버 재시작 후에도 업데이트된 모델 사용

### Chrome 확장 연동 테스트

11. [ ] Chrome 확장에서 서버 연결
    - [ ] 확장 설치 및 YouTube Live 페이지 접속
    - [ ] 서버 연결 확인 (확장 아이콘 색상 변경)

12. [ ] 댓글 필터링 동작
    - [ ] 악성 댓글 자동 숨김/표시
    - [ ] 댓글 라벨링 기능

## 배포 패키지 준비

13. [ ] 폴더 구조 확인
    ```
    youtube_live_filter_server/
    ├── youtube_live_filter_server.exe
    └── _internal/
        └── (의존성 파일들)
    ```

14. [ ] 배포 파일 생성
    - [ ] `dist/youtube_live_filter_server/` 폴더를 zip으로 압축
    - [ ] 파일명: `youtube_live_filter_server_vX.X.X.zip`

15. [ ] 문서 포함
    - [ ] README_EXE.md 포함
    - [ ] 사용 가이드 작성

## 배포 후 확인

16. [ ] 다른 컴퓨터에서 테스트
    - [ ] Python 미설치 환경에서 실행
    - [ ] 기본 기능 동작 확인

17. [ ] 문제점 기록
    - [ ] 발견된 이슈 문서화
    - [ ] 해결 방법 정리

## 알려진 제한 사항

- [ ] exe 크기가 큼 (모델 포함으로 인해 1-3GB)
- [ ] 첫 실행 시 모델 로딩에 시간 소요 (10-30초)
- [ ] GPU 사용 시 CUDA 라이브러리 필요 (CPU 모드는 자동)
- [ ] Windows Defender가 차단할 수 있음 (신뢰할 수 있는 앱으로 추가 필요)

## 버전별 변경 사항 기록

### v1.0.0
- [ ] 초기 EXE 배포
- [ ] 기본 예측 기능
- [ ] 재학습 기능
- [ ] 학습 데이터 관리

---

✅ 모든 항목 완료 후 배포 진행


