import os
import json
import subprocess
import threading
import glob
from pathlib import Path
from typing import List, Dict, Any
import logging
from datetime import datetime
import atexit
import shutil
import re
import sys

# Ensure torch compile/dynamo is disabled before importing torch (safer for PyInstaller)
os.environ.setdefault("TORCH_COMPILE_DISABLE", "1")
os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_THREADING_LAYER", "SEQUENTIAL")
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "1")

from fastapi import FastAPI, BackgroundTasks, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# SAFE_MODE: 대상 PC에서 torch/peft 로딩 문제가 있을 때 모델 로딩을 건너뛰고 서버만 띄움
SAFE_MODE = os.environ.get("SAFE_MODE", "0") == "1"

if not SAFE_MODE:
	try:
		import torch  # type: ignore
		from transformers import AutoTokenizer, AutoModelForSequenceClassification  # type: ignore
		from peft import PeftModel  # type: ignore
	except Exception as _torch_err:
		# 안전하게 폴백
		SAFE_MODE = True
		torch = None  # type: ignore
		AutoTokenizer = None  # type: ignore
		AutoModelForSequenceClassification = None  # type: ignore
		PeftModel = None  # type: ignore
else:
	torch = None  # type: ignore
	AutoTokenizer = None  # type: ignore
	AutoModelForSequenceClassification = None  # type: ignore
	PeftModel = None  # type: ignore

# train 모듈 import (직접 함수 호출용)
try:
	from train import train as train_model_func
	TRAIN_MODULE_AVAILABLE = True
except ImportError:
	TRAIN_MODULE_AVAILABLE = False
	LOGGER_EARLY = logging.getLogger("yt_live_chat_filter")
	LOGGER_EARLY.warning("train module not available for direct import")

# Early logger reference for functions defined before logging setup below
LOGGER = logging.getLogger("yt_live_chat_filter")


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


class PredictRequest(BaseModel):
	texts: List[str]


class PredictResponse(BaseModel):
	labels: List[int]
	probs: List[List[float]]
	label_names: Dict[int, str]


class TrainingDataRequest(BaseModel):
	text: str
	label: int
	user_id: str = "anonymous"


class TrainingDataResponse(BaseModel):
	success: bool
	message: str


class LookupRequest(BaseModel):
	texts: List[str]


ADAPTER_STATUS = {"attached": False, "path": None}

def _load_model(model_dir: Path):
	if SAFE_MODE:
		# 토치 미사용 안전 모드: 토크나이저/모델 없이 동작
		return None, None, "cpu"
	
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # type: ignore[attr-defined]
	tokenizer = AutoTokenizer.from_pretrained(str(model_dir))  # type: ignore[operator]
	
	# Check for active adapter first
	adapter_dir = Path(__file__).resolve().parents[1] / "adapters" / "active"
	adapter_cfg = adapter_dir / "adapter_config.json"
	has_model_file = (adapter_dir / "model.safetensors").exists() or (adapter_dir / "pytorch_model.bin").exists()
	
	try:
		if adapter_cfg.exists():
			# PEFT adapter: load base model then attach adapter
			model = AutoModelForSequenceClassification.from_pretrained(str(model_dir))  # type: ignore[operator]
			model = PeftModel.from_pretrained(model, str(adapter_dir))  # type: ignore[operator]
			ADAPTER_STATUS["attached"] = True
			ADAPTER_STATUS["path"] = str(adapter_dir)
			LOGGER.info(f"PEFT adapter attached: {ADAPTER_STATUS['path']}")
		elif has_model_file:
			# Full fine-tuned model (BitFit/Linear): load directly
			model = AutoModelForSequenceClassification.from_pretrained(str(adapter_dir))  # type: ignore[operator]
			ADAPTER_STATUS["attached"] = True
			ADAPTER_STATUS["path"] = str(adapter_dir)
			LOGGER.info(f"Fine-tuned model loaded from: {ADAPTER_STATUS['path']}")
		else:
			# No adapter: use base model
			model = AutoModelForSequenceClassification.from_pretrained(str(model_dir))  # type: ignore[operator]
			ADAPTER_STATUS["attached"] = False
			ADAPTER_STATUS["path"] = None
			LOGGER.info("No adapter found. Using base model only.")
	except Exception as e:
		LOGGER.warning(f"Adapter load failed, using base model: {e}")
		model = AutoModelForSequenceClassification.from_pretrained(str(model_dir))  # type: ignore[operator]
		ADAPTER_STATUS["attached"] = False
		ADAPTER_STATUS["path"] = None
	
	model.to(device)  # type: ignore[union-attr]
	model.eval()  # type: ignore[union-attr]
	return tokenizer, model, device


def _softmax(logits):
	# SAFE_MODE에서는 호출되지 않음
	return torch.nn.functional.softmax(logits, dim=-1)  # type: ignore[name-defined]


def save_training_data(text: str, label: int, user_id: str, use_temp: bool = False) -> bool:
	"""학습 데이터를 JSON 파일에 저장"""
	try:
		timestamp = datetime.now().isoformat()
		data = {
			"text": text,
			"label": label,
			"user_id": user_id,
			"timestamp": timestamp
		}
		
		# 날짜별로 파일 분리
		date_str = datetime.now().strftime("%Y-%m-%d")
		base_dir = TRAINING_DATA_DIR
		data_file = base_dir / f"training_data_{date_str}.jsonl"
		
		# 동시 쓰기 안전: append + flush
		os.makedirs(base_dir, exist_ok=True)
		line = json.dumps(data, ensure_ascii=False) + "\n"
		with open(data_file, "a", encoding="utf-8") as f:
			f.write(line)
			f.flush()
		# temp 비활성화: 개수 제한 미적용
		
		if 'LOG_TRAINING_SAVES' in globals() and LOG_TRAINING_SAVES:
			LOGGER.info(f"Training data saved: label={label} ({LABEL_NAMES.get(label, '?')}) text=\"{text[:50]}...\"")
		return True
	except Exception as e:
		LOGGER.error(f"Failed to save training data: {e}")
		return False


def run_training_background():
	"""백그라운드에서 모델 재학습 실행"""
	global TRAINING_STATUS, TOKENIZER, MODEL, DEVICE
	
	try:
		TRAINING_STATUS["is_training"] = True
		TRAINING_STATUS["progress"] = 10
		TRAINING_STATUS["message"] = "학습 데이터 확인 중..."
		TRAINING_STATUS["error"] = None
		
		# 학습 데이터 개수 확인
		total_samples = 0
		for data_file in TRAINING_DATA_DIR.glob("training_data_*.jsonl"):
			with open(data_file, "r", encoding="utf-8") as f:
				for line in f:
					if line.strip():
						total_samples += 1
		
		if total_samples < 1:
			TRAINING_STATUS["error"] = f"학습 데이터가 부족합니다. 최소 1개 필요, 현재 {total_samples}개"
			TRAINING_STATUS["is_training"] = False
			return
		
		TRAINING_STATUS["progress"] = 20
		TRAINING_STATUS["message"] = f"학습 데이터 {total_samples}개 확인됨. 학습 시작..."
		
		output_dir = ADAPTER_UPDATED_DIR
		epochs = int(os.environ.get("TRAINING_EPOCHS", "1"))
		batch_size = int(os.environ.get("TRAINING_BATCH", "16"))
		lr = float(os.environ.get("TRAINING_LR", "5e-6"))
		warmup_ratio = float(os.environ.get("TRAINING_WARMUP_RATIO", "0.06"))
		augment_factor = 3000
		
		TRAINING_STATUS["progress"] = 30
		TRAINING_STATUS["message"] = "모델 학습 중... (시간이 걸릴 수 있습니다)"
		
		# EXE 환경이거나 train 모듈이 사용 가능하면 직접 호출
		is_frozen = getattr(sys, 'frozen', False)
		
		if is_frozen or TRAIN_MODULE_AVAILABLE:
			# 직접 함수 호출 (EXE 환경 또는 모듈 가능)
			LOGGER.info(f"Training directly (frozen={is_frozen})")
			LOGGER.info(f"model_dir={MODEL_DIR}, training_data_dir={TRAINING_DATA_DIR}, output_dir={output_dir}")
			LOGGER.info(f"epochs={epochs}, batch_size={batch_size}, lr={lr}, warmup_ratio={warmup_ratio}, augment_factor={augment_factor}")
			
			try:
				success = train_model_func(
					model_dir=Path(MODEL_DIR),
					training_data_dir=TRAINING_DATA_DIR,
					output_dir=output_dir,
					epochs=epochs,
					batch_size=batch_size,
					lr=lr,
					warmup_ratio=warmup_ratio,
					augment_factor=augment_factor
				)
				result_returncode = 0 if success else 1
			except Exception as train_err:
				LOGGER.error(f"Direct training failed: {train_err}", exc_info=True)
				result_returncode = 1
		else:
			# subprocess 실행 (일반 Python 환경)
			script_path = Path(__file__).parent / "train.py"
			python_exe = sys.executable
			
			cmd = [
				python_exe,
				str(script_path),
				"--model-dir", str(MODEL_DIR),
				"--training-data-dir", str(TRAINING_DATA_DIR),
				"--output-dir", str(output_dir),
				"--epochs", str(epochs),
				"--batch-size", str(batch_size),
				"--lr", str(lr),
				"--warmup-ratio", str(warmup_ratio),
				"--augment-factor", str(augment_factor)
			]
			
			LOGGER.info(f"Training via subprocess: {python_exe}")
			LOGGER.info(f"Script: {script_path}")
			
			# 환경 변수 설정
			env = os.environ.copy()
			env["TRAINING_MODE"] = "1"
			
			result = subprocess.run(
				cmd, 
				capture_output=True, 
				text=True, 
				timeout=None,
				cwd=str(Path(__file__).parent),
				env=env
			)
			result_returncode = result.returncode
			
			# 로그 출력
			LOGGER.info(f"Training exit code: {result_returncode}")
			if result.stdout:
				LOGGER.info(f"Training stdout:\n{result.stdout}")
			if result.stderr:
				LOGGER.error(f"Training stderr:\n{result.stderr}")
		
		if result_returncode == 0:
			TRAINING_STATUS["progress"] = 80
			TRAINING_STATUS["message"] = "학습 완료. 모델 교체 중..."
			
			# 어댑터 스왑: active -> backup, updated -> active
			if ADAPTER_BACKUP_DIR.exists():
				shutil.rmtree(ADAPTER_BACKUP_DIR)
			if ADAPTER_ACTIVE_DIR.exists():
				shutil.move(str(ADAPTER_ACTIVE_DIR), str(ADAPTER_BACKUP_DIR))
			if ADAPTER_UPDATED_DIR.exists():
				shutil.move(str(ADAPTER_UPDATED_DIR), str(ADAPTER_ACTIVE_DIR))
			
			TRAINING_STATUS["progress"] = 90
			TRAINING_STATUS["message"] = "모델 재로드 중..."
			
			# 모델 재로드(어댑터 포함)
			TOKENIZER, MODEL, DEVICE = _load_model(MODEL_DIR)
			
			TRAINING_STATUS["progress"] = 100
			TRAINING_STATUS["message"] = "재학습 완료!"
			TRAINING_STATUS["is_training"] = False
			
			LOGGER.info("Model retraining completed successfully!")
			
		else:
			TRAINING_STATUS["error"] = f"학습 실패 (코드 {result_returncode})"
			TRAINING_STATUS["is_training"] = False
			LOGGER.error(f"Training failed with code {result_returncode}")
		
	except Exception as e:
		TRAINING_STATUS["error"] = f"학습 중 오류: {str(e)}"
		TRAINING_STATUS["is_training"] = False
		LOGGER.error(f"Training error: {e}")


# 경로 설정
USER_DATA_PATH = get_user_data_path()

# 업데이트된 모델이 있으면 사용, 없으면 기본 모델 사용
UPDATED_MODEL_DIR = USER_DATA_PATH / "model"
DEFAULT_MODEL_DIR = get_model_path()
MODEL_DIR = UPDATED_MODEL_DIR if UPDATED_MODEL_DIR.exists() and any(UPDATED_MODEL_DIR.iterdir()) else DEFAULT_MODEL_DIR

TRAINING_DATA_DIR = USER_DATA_PATH / "training_data"
TRAINING_DATA_DIR.mkdir(exist_ok=True)

# 임시 학습 데이터 디렉터리
TEMP_TRAINING_DATA_DIR = USER_DATA_PATH / "training_temp"
TEMP_TRAINING_DATA_DIR.mkdir(exist_ok=True)

# PEFT 어댑터 디렉터리들
ADAPTERS_DIR = Path(__file__).resolve().parents[1] / "adapters"
ADAPTERS_DIR.mkdir(exist_ok=True)
ADAPTER_ACTIVE_DIR = ADAPTERS_DIR / "active"
ADAPTER_UPDATED_DIR = ADAPTERS_DIR / "updated"
ADAPTER_BACKUP_DIR = ADAPTERS_DIR / "backup"

def _cleanup_temp_dir():
	try:
		if TEMP_TRAINING_DATA_DIR.exists():
			shutil.rmtree(TEMP_TRAINING_DATA_DIR, ignore_errors=True)
	except Exception as e:
		LOGGER.warning(f"Failed to cleanup temp dir: {e}")

atexit.register(_cleanup_temp_dir)

def _enforce_temp_limit(max_items: int = 30) -> None:
	"""Ensure total number of temp cached items does not exceed max_items.
	Removes oldest lines first across temp files (oldest file first)."""
	try:
		files = sorted(TEMP_TRAINING_DATA_DIR.glob("training_data_*.jsonl"), key=lambda p: p.stat().st_mtime)
		if not files:
			return
		# Read counts
		total = 0
		file_lines: List[Dict[str, Any]] = []
		for fp in files:
			try:
				with open(fp, 'r', encoding='utf-8') as f:
					lines = f.readlines()
				valid_lines = [ln for ln in lines if ln.strip()]
				file_lines.append({"path": fp, "lines": valid_lines})
				total += len(valid_lines)
			except Exception:
				continue
		excess = max(0, total - max_items)
		idx = 0
		while excess > 0 and idx < len(file_lines):
			entry = file_lines[idx]
			lines = entry["lines"]
			if not lines:
				try:
					os.remove(entry["path"])  # type: ignore[arg-type]
				except Exception:
					pass
				idx += 1
				continue
			remove_n = min(excess, len(lines))
			new_lines = lines[remove_n:]
			try:
				if new_lines:
					with open(entry["path"], 'w', encoding='utf-8') as f:  # type: ignore[arg-type]
						f.writelines(new_lines)
				else:
					os.remove(entry["path"])  # type: ignore[arg-type]
			except Exception:
				pass
			excess -= remove_n
			idx += 1
	except Exception as e:
		LOGGER.warning(f"Failed to enforce temp limit: {e}")

TOKENIZER, MODEL, DEVICE = _load_model(MODEL_DIR)

# 0 정상, 1 약간 악성, 2 악성
LABEL_NAMES = {0: "정상", 1: "약간 악성", 2: "악성"}

# 재학습 상태 추적
TRAINING_STATUS = {
    "is_training": False,
    "progress": 0,
    "message": "",
    "error": None
}

# Logging
logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
LOGGER = logging.getLogger("yt_live_chat_filter")
LOG_PREDICTIONS = os.environ.get("LOG_PREDICTIONS", "1") == "1"
LOG_TRAINING_SAVES = os.environ.get("LOG_TRAINING_SAVES", "0") == "1"

app = FastAPI(title="YouTube Live Chat Moderation Model Server", version="1.0.0")

# Allow extension to call (default localhost)
ALLOWED_ORIGINS = [
	"http://localhost",
	"http://127.0.0.1",
	"http://localhost:3000",
	"http://127.0.0.1:3000",
	"chrome-extension://*",
]

app.add_middleware(
	CORSMiddleware,
	# 크롬 확장(Origin: chrome-extension://<id>)을 포함해 전부 허용
	allow_origins=["*"],
	allow_credentials=False,  # '*'와 함께 credentials 허용은 불가
	allow_methods=["*"],
	allow_headers=["*"],
)

# Preflight 안전 처리 (일부 환경에서 400 회피용)
@app.options("/predict")
def options_predict() -> Dict[str, Any]:
	return {"ok": True}


@app.get("/health")
def health() -> Dict[str, Any]:
	return {"status": "ok", "device": str(DEVICE), "num_labels": MODEL.config.num_labels}


@app.get("/labels")
def labels() -> Dict[int, str]:
	return LABEL_NAMES


@app.post("/training-data", response_model=TrainingDataResponse)
def add_training_data(req: TrainingDataRequest, temp: bool = False) -> TrainingDataResponse:
	"""사용자가 선택한 댓글과 라벨을 학습 데이터로 저장"""
	if req.label not in LABEL_NAMES:
		return TrainingDataResponse(success=False, message="Invalid label")
	
	if not req.text.strip():
		return TrainingDataResponse(success=False, message="Empty text")
	
	# 임시 저장 비활성화: temp 요청은 저장하지 않고 성공으로 무시
	if temp:
		return TrainingDataResponse(success=True, message="Temp training data is disabled; request ignored")
	
	# 명시적 사용자 클릭만 저장 허용: 과거 캐시 저장 등 비의도적 경로 차단
	allowed_user_ids = {"user"}  # 확장에서 클릭 저장 시 'user'로 전송
	if getattr(req, "user_id", None) not in allowed_user_ids:
		return TrainingDataResponse(success=False, message="Only explicit user-click data is allowed")
	
	success = save_training_data(req.text, req.label, req.user_id, use_temp=temp)
	if success:
		return TrainingDataResponse(success=True, message="Training data saved successfully")
	else:
		return TrainingDataResponse(success=False, message="Failed to save training data")


@app.delete("/training-data/temp")
def delete_temp_training_data() -> Dict[str, Any]:
	"""임시 학습 데이터 전체 삭제 (비활성화)"""
	return {"message": "Temp training data feature is disabled"}


@app.post("/training-data/lookup")
def lookup_cached_labels(req: LookupRequest) -> Dict[str, Any]:
	"""TEMP 캐시 비활성화: 항상 None으로 응답"""
	incoming = getattr(req, 'texts', []) or []
	return {"labels": [None] * len(incoming)}


@app.get("/training-data/stats")
def get_training_data_stats() -> Dict[str, Any]:
    """저장된 학습 데이터 통계 조회 (영구 데이터만)"""
    try:
        total_count = 0
        label_counts = {0: 0, 1: 0, 2: 0}
        for data_file in TRAINING_DATA_DIR.glob("training_data_*.jsonl"):
            try:
                with open(data_file, "r", encoding="utf-8", errors="ignore") as f:
                    for line in f:
                        if line.strip():
                            try:
                                data = json.loads(line)
                                total_count += 1
                                label_counts[int(data.get("label", 0))] += 1
                            except (json.JSONDecodeError, ValueError):
                                continue
            except Exception as e:
                LOGGER.warning(f"Failed to read file {data_file}: {e}")
                continue
        return {
            "total_samples": total_count,
            "label_distribution": { LABEL_NAMES[k]: v for k, v in label_counts.items() },
            "data_files": len(list(TRAINING_DATA_DIR.glob("training_data_*.jsonl")))
        }
    except Exception as e:
        LOGGER.error(f"Failed to get training data stats: {e}")
        return {"error": str(e)}


@app.get("/training-data/stats-temp")
def get_training_data_stats_temp() -> Dict[str, Any]:
	"""임시 학습 데이터 통계 비활성화"""
	return {
		"total_samples": 0,
		"label_distribution": { LABEL_NAMES[k]: 0 for k in LABEL_NAMES.keys() },
		"data_files": 0
	}

@app.get("/training-data/stats-all")
def get_training_data_stats_all() -> Dict[str, Any]:
    """저장된 학습 데이터 통계 조회 (영구만)"""
    try:
        def accumulate_from_dir(base: Path, total_label_counts: Dict[int, int]) -> int:
            total = 0
            for data_file in base.glob("training_data_*.jsonl"):
                try:
                    with open(data_file, "r", encoding="utf-8", errors="ignore") as f:
                        for line in f:
                            if line.strip():
                                try:
                                    data = json.loads(line)
                                    total += 1
                                    lbl = int(data.get("label", 0))
                                    if lbl in total_label_counts:
                                        total_label_counts[lbl] += 1
                                except (json.JSONDecodeError, ValueError):
                                    # 손상된 라인은 건너뛰기
                                    continue
                except Exception as e:
                    LOGGER.warning(f"Failed to read file {data_file}: {e}")
                    continue
            return total

        label_counts = {0: 0, 1: 0, 2: 0}
        total_count = 0
        total_count += accumulate_from_dir(TRAINING_DATA_DIR, label_counts)

        return {
            "total_samples": total_count,
            "label_distribution": { LABEL_NAMES[k]: v for k, v in label_counts.items() },
            "data_files": len(list(TRAINING_DATA_DIR.glob("training_data_*.jsonl")))
        }
    except Exception as e:
        LOGGER.error(f"Failed to get training data stats(all): {e}")
        return {"error": str(e)}


@app.post("/model/reload")
def reload_model() -> Dict[str, Any]:
	"""모델을 다시 로드 (새로 학습된 모델 적용)"""
	global TOKENIZER, MODEL, DEVICE, MODEL_DIR
	
	try:
		LOGGER.info("Reloading model...")
		# 업데이트된 모델이 있으면 사용
		if UPDATED_MODEL_DIR.exists() and any(UPDATED_MODEL_DIR.iterdir()):
			MODEL_DIR = UPDATED_MODEL_DIR
		else:
			MODEL_DIR = DEFAULT_MODEL_DIR
		TOKENIZER, MODEL, DEVICE = _load_model(MODEL_DIR)
		LOGGER.info(f"Model reloaded successfully from {MODEL_DIR}")
		return {"success": True, "message": "Model reloaded successfully"}
	except Exception as e:
		LOGGER.error(f"Failed to reload model: {e}")
		return {"success": False, "message": f"Failed to reload model: {e}"}


@app.post("/model/retrain")
def start_retraining(background_tasks: BackgroundTasks) -> Dict[str, Any]:
	"""모델 재학습 시작"""
	global TRAINING_STATUS
	
	if SAFE_MODE:
		return {"success": False, "message": "SAFE_MODE=1: 재학습이 비활성화되어 있습니다"}
	
	if TRAINING_STATUS["is_training"]:
		return {"success": False, "message": "이미 학습이 진행 중입니다"}
	
	# 백그라운드에서 학습 시작
	background_tasks.add_task(run_training_background)
	
	return {"success": True, "message": "재학습이 시작되었습니다"}


@app.get("/model/training-status")
def get_training_status() -> Dict[str, Any]:
	"""재학습 상태 조회"""
	return TRAINING_STATUS


@app.post("/model/reset-training-status")
def reset_training_status() -> Dict[str, Any]:
	"""재학습 상태 강제 리셋 (막힌 상태 해결용)"""
	global TRAINING_STATUS
	TRAINING_STATUS = {
		"is_training": False,
		"progress": 0,
		"message": "",
		"error": None
	}
	return {"success": True, "message": "학습 상태가 리셋되었습니다"}


@app.get("/model/info")
def model_info() -> Dict[str, Any]:
    """현재 모델/어댑터 상태 확인"""
    return {
        "device": str(DEVICE),
        "num_labels": MODEL.config.num_labels,
        "adapter_attached": bool(ADAPTER_STATUS.get("attached", False)),
        "adapter_path": ADAPTER_STATUS.get("path")
    }

@app.get("/training-data/files")
def get_training_data_files() -> Dict[str, Any]:
	"""학습 데이터 파일 목록 조회"""
	try:
		files = []
		pattern = str(TRAINING_DATA_DIR / "training_data_*.jsonl")
		for file_path in glob.glob(pattern):
			try:
				file_name = os.path.basename(file_path)
				file_size = os.path.getsize(file_path)
				file_date = datetime.fromtimestamp(os.path.getmtime(file_path))
				
				# 파일 내용 개수 계산 (손상/인코딩 오류 내구성)
				count = 0
				with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
					for line in f:
						if line.strip():
							count += 1
				
				files.append({
					"filename": file_name,
					"path": file_path,
					"size": file_size,
					"count": count,
					"date": file_date.isoformat()
				})
			except Exception as e:
				# 문제 있는 파일은 건너뛰고 계속
				LOGGER.warning(f"Skip file in listing due to error: {file_path} ({e})")
		
		# 날짜순으로 정렬 (최신순)
		files.sort(key=lambda x: x['date'], reverse=True)
		return {"files": files}
	except Exception as e:
		raise HTTPException(status_code=500, detail=f"파일 목록 조회 실패: {str(e)}")


@app.get("/training-data/files-temp/{filename}")
def get_training_data_file_temp(filename: str) -> Dict[str, Any]:
	"""특정 임시 학습 데이터 파일 내용 조회 (비활성화)"""
	raise HTTPException(status_code=410, detail="Temp training data feature is disabled")

@app.get("/training-data/files/{filename}")
def get_training_data_file(filename: str) -> Dict[str, Any]:
	"""특정 학습 데이터 파일 내용 조회"""
	try:
		file_path = TRAINING_DATA_DIR / filename
		if not file_path.exists():
			raise HTTPException(status_code=404, detail="파일을 찾을 수 없습니다")
		
		data = []
		with open(file_path, 'r', encoding='utf-8') as f:
			for line_num, line in enumerate(f, 1):
				if line.strip():
					try:
						item = json.loads(line.strip())
						item['line_number'] = line_num
						data.append(item)
					except json.JSONDecodeError:
						continue
		
		return {
			"filename": filename,
			"count": len(data),
			"data": data
		}
	except HTTPException:
		raise
	except Exception as e:
		raise HTTPException(status_code=500, detail=f"파일 조회 실패: {str(e)}")


@app.delete("/training-data/files/{filename}")
def delete_training_data_file(filename: str) -> Dict[str, Any]:
	"""특정 학습 데이터 파일 삭제"""
	try:
		file_path = TRAINING_DATA_DIR / filename
		if not file_path.exists():
			raise HTTPException(status_code=404, detail="파일을 찾을 수 없습니다")
		
		# 파일 삭제
		os.remove(file_path)
		
		return {"message": f"파일 '{filename}'이 삭제되었습니다"}
	except HTTPException:
		raise
	except Exception as e:
		raise HTTPException(status_code=500, detail=f"파일 삭제 실패: {str(e)}")


@app.delete("/training-data/files/{filename}/lines/{line_number}")
def delete_training_data_line(filename: str, line_number: int) -> Dict[str, Any]:
	"""특정 학습 데이터 파일의 특정 라인 삭제"""
	try:
		file_path = TRAINING_DATA_DIR / filename
		if not file_path.exists():
			raise HTTPException(status_code=404, detail="파일을 찾을 수 없습니다")
		
		# 파일 읽기
		lines = []
		with open(file_path, 'r', encoding='utf-8') as f:
			lines = f.readlines()
		
		# 라인 번호가 유효한지 확인
		if line_number < 1 or line_number > len(lines):
			raise HTTPException(status_code=400, detail="유효하지 않은 라인 번호입니다")
		
		# 해당 라인 삭제
		del lines[line_number - 1]
		
		# 파일 다시 쓰기
		with open(file_path, 'w', encoding='utf-8') as f:
			f.writelines(lines)
		
		return {"message": f"라인 {line_number}이 삭제되었습니다"}
	except HTTPException:
		raise
	except Exception as e:
		raise HTTPException(status_code=500, detail=f"라인 삭제 실패: {str(e)}")


@app.delete("/training-data/all")
def delete_all_training_data() -> Dict[str, Any]:
    """모든 학습 데이터 파일 삭제 (영구만)"""
    try:
        deleted_files = []
        for file_path in glob.glob(str(TRAINING_DATA_DIR / "training_data_*.jsonl")):
            file_name = os.path.basename(file_path)
            os.remove(file_path)
            deleted_files.append(file_name)
        return {"message": f"영구 데이터 {len(deleted_files)}개 삭제", "deleted_files": deleted_files}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"전체 삭제 실패: {str(e)}")


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest) -> PredictResponse:
	texts = [t if isinstance(t, str) else str(t) for t in req.texts]
	if len(texts) == 0:
		return PredictResponse(labels=[], probs=[], label_names=LABEL_NAMES)
	
	# SAFE_MODE: 토치/모델 없이 동작 (모두 정상으로 처리)
	if SAFE_MODE:
		return PredictResponse(labels=[0]*len(texts), probs=[[1.0, 0.0, 0.0] for _ in texts], label_names=LABEL_NAMES)

	encoded = TOKENIZER(
		texts,
		padding=True,
		truncation=True,
		max_length=256,
		return_tensors="pt",
	)
	encoded = {k: v.to(DEVICE) for k, v in encoded.items()}

	with torch.no_grad():
		outputs = MODEL(**encoded)
		logits = outputs.logits
		prob = _softmax(logits).cpu()
		pred = torch.argmax(prob, dim=-1).tolist()

	probs = prob.tolist()

	if LOG_PREDICTIONS:
		for i, (text, label, p) in enumerate(zip(texts, pred, probs)):
			preview = (text or "").replace("\n", " ")
			if len(preview) > 200:
				preview = preview[:200] + "…"
			LOGGER.info(
				f"PRED[{i}] label={label} ({LABEL_NAMES.get(label, '?')}) probs={ [round(x,4) for x in p] } text=\"{preview}\""
			)
	return PredictResponse(labels=pred, probs=probs, label_names=LABEL_NAMES)


if __name__ == "__main__":
	# exe 환경에서는 TRAINING_MODE 여부와 무관하게 항상 서버 실행
	is_frozen = getattr(sys, 'frozen', False)
	should_run_server = True if is_frozen else (os.environ.get("TRAINING_MODE") != "1")
	if should_run_server:
		import uvicorn
		port = int(os.environ.get("PORT", 8000))
		# exe 환경에서는 문자열 대신 직접 app 인스턴스를 전달해야 함
		uvicorn.run(app, host="127.0.0.1", port=port, reload=False)
