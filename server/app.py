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

import torch
from fastapi import FastAPI, BackgroundTasks, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from peft import PeftModel

# Early logger reference for functions defined before logging setup below
LOGGER = logging.getLogger("yt_live_chat_filter")


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
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	tokenizer = AutoTokenizer.from_pretrained(str(model_dir))
	
	# Check for active adapter first
	adapter_dir = Path(__file__).resolve().parents[1] / "adapters" / "active"
	adapter_cfg = adapter_dir / "adapter_config.json"
	has_model_file = (adapter_dir / "model.safetensors").exists() or (adapter_dir / "pytorch_model.bin").exists()
	
	try:
		if adapter_cfg.exists():
			# PEFT adapter: load base model then attach adapter
			model = AutoModelForSequenceClassification.from_pretrained(str(model_dir))
			model = PeftModel.from_pretrained(model, str(adapter_dir))
			ADAPTER_STATUS["attached"] = True
			ADAPTER_STATUS["path"] = str(adapter_dir)
			LOGGER.info(f"PEFT adapter attached: {ADAPTER_STATUS['path']}")
		elif has_model_file:
			# Full fine-tuned model (BitFit/Linear): load directly
			model = AutoModelForSequenceClassification.from_pretrained(str(adapter_dir))
			ADAPTER_STATUS["attached"] = True
			ADAPTER_STATUS["path"] = str(adapter_dir)
			LOGGER.info(f"Fine-tuned model loaded from: {ADAPTER_STATUS['path']}")
		else:
			# No adapter: use base model
			model = AutoModelForSequenceClassification.from_pretrained(str(model_dir))
			ADAPTER_STATUS["attached"] = False
			ADAPTER_STATUS["path"] = None
			LOGGER.info("No adapter found. Using base model only.")
	except Exception as e:
		LOGGER.warning(f"Adapter load failed, using base model: {e}")
		model = AutoModelForSequenceClassification.from_pretrained(str(model_dir))
		ADAPTER_STATUS["attached"] = False
		ADAPTER_STATUS["path"] = None
	
	model.to(device)
	model.eval()
	return tokenizer, model, device


def _softmax(logits: torch.Tensor) -> torch.Tensor:
	return torch.nn.functional.softmax(logits, dim=-1)


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
		base_dir = TEMP_TRAINING_DATA_DIR if use_temp else TRAINING_DATA_DIR
		data_file = base_dir / f"training_data_{date_str}.jsonl"
		
		# 동시 쓰기 안전: append + flush
		os.makedirs(base_dir, exist_ok=True)
		line = json.dumps(data, ensure_ascii=False) + "\n"
		with open(data_file, "a", encoding="utf-8") as f:
			f.write(line)
			f.flush()
		# temp 사용 시 개수 제한 적용 (30개 유지)
		if use_temp:
			_enforce_temp_limit(30)
		
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
		
		if total_samples < 5:
			TRAINING_STATUS["error"] = f"학습 데이터가 부족합니다. 최소 5개 필요, 현재 {total_samples}개"
			TRAINING_STATUS["is_training"] = False
			return
		
		TRAINING_STATUS["progress"] = 20
		TRAINING_STATUS["message"] = f"학습 데이터 {total_samples}개 확인됨. 학습 시작..."
		
		# 학습 스크립트 실행 (BitFit)
		script_path = Path(__file__).parent / "train.py"
		output_dir = ADAPTER_UPDATED_DIR
		
		cmd = [
			str(Path(__file__).parent / ".venv" / "Scripts" / "python.exe"),
			str(script_path),
			"--model-dir", str(MODEL_DIR),
			"--training-data-dir", str(TRAINING_DATA_DIR),
			"--output-dir", str(output_dir),
			"--epochs", os.environ.get("TRAINING_EPOCHS", "1"),
			"--batch-size", os.environ.get("TRAINING_BATCH", "16"),
			"--lr", os.environ.get("TRAINING_LR", "5e-6"),
			"--warmup-ratio", os.environ.get("TRAINING_WARMUP_RATIO", "0.06")
		]
		
		TRAINING_STATUS["progress"] = 30
		TRAINING_STATUS["message"] = "모델 학습 중... (시간이 걸릴 수 있습니다)"
		
		# 학습 실행 (타임아웃 없음)
		result = subprocess.run(cmd, capture_output=True, text=True, timeout=None)
		
		if result.returncode == 0:
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
			TRAINING_STATUS["error"] = f"학습 실패: {result.stderr}"
			TRAINING_STATUS["is_training"] = False
			LOGGER.error(f"Training failed: {result.stderr}")
			
	except subprocess.TimeoutExpired:
		TRAINING_STATUS["error"] = "학습 시간 초과 (30분)"
		TRAINING_STATUS["is_training"] = False
		LOGGER.error("Training timeout")
		
	except Exception as e:
		TRAINING_STATUS["error"] = f"학습 중 오류: {str(e)}"
		TRAINING_STATUS["is_training"] = False
		LOGGER.error(f"Training error: {e}")


MODEL_DIR = Path(__file__).resolve().parents[1] / "model"
TRAINING_DATA_DIR = Path(__file__).resolve().parents[1] / "training_data"
TRAINING_DATA_DIR.mkdir(exist_ok=True)

# 임시 학습 데이터 디렉터리 (프로젝트 폴더 내부)
TEMP_TRAINING_DATA_DIR = Path(__file__).resolve().parents[1] / "training_temp"
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
	
	success = save_training_data(req.text, req.label, req.user_id, use_temp=temp)
	if success:
		return TrainingDataResponse(success=True, message="Training data saved successfully")
	else:
		return TrainingDataResponse(success=False, message="Failed to save training data")


@app.delete("/training-data/temp")
def delete_temp_training_data() -> Dict[str, Any]:
	"""임시 학습 데이터 전체 삭제"""
	try:
		_cleanup_temp_dir()
		TEMP_TRAINING_DATA_DIR.mkdir(exist_ok=True)
		return {"message": "Temporary training data cleared"}
	except Exception as e:
		raise HTTPException(status_code=500, detail=f"임시 데이터 삭제 실패: {str(e)}")


@app.post("/training-data/lookup")
def lookup_cached_labels(req: LookupRequest) -> Dict[str, Any]:
    """TEMP에 저장된 텍스트 라벨 캐시 조회 (최근 날짜 우선)"""
    try:
        # 간단한 텍스트 단순 매칭(정규화 없이). 필요시 소문자/공백정리 확장 가능
        incoming = req.texts or []
        targets = [t.strip() for t in incoming if isinstance(t, str) and t.strip()]
        if not targets:
            return {"labels": [None] * len(incoming)}

        # 최신 파일부터 스캔 (동시 쓰기/락 오류는 건너뜀)
        files = sorted(TEMP_TRAINING_DATA_DIR.glob("training_data_*.jsonl"), key=lambda p: p.stat().st_mtime, reverse=True)
        cache: Dict[str, int] = {}
        for fp in files:
            try:
                with open(fp, 'r', encoding='utf-8') as f:
                    for line in f:
                        if not line.strip():
                            continue
                        try:
                            item = json.loads(line)
                            txt = str(item.get('text', '')).strip()
                            lbl = int(item.get('label', 0))
                            if txt and txt not in cache:
                                cache[txt] = lbl
                        except Exception:
                            continue
            except Exception:
                continue
            if len(cache) >= 20000:
                break

        # 원래 요청 순서를 유지하여 매핑 (빈/비문자열은 None)
        out = []
        for t in incoming:
            if isinstance(t, str):
                key = t.strip()
                out.append(cache.get(key, None) if key else None)
            else:
                out.append(None)
        return {"labels": out}
    except Exception as e:
        LOGGER.error(f"lookup 실패: {e}")
        return {"labels": [None] * len(getattr(req, 'texts', []) or [])}


@app.get("/training-data/stats")
def get_training_data_stats() -> Dict[str, Any]:
    """저장된 학습 데이터 통계 조회 (영구 데이터만)"""
    try:
        total_count = 0
        label_counts = {0: 0, 1: 0, 2: 0}
        for data_file in TRAINING_DATA_DIR.glob("training_data_*.jsonl"):
            with open(data_file, "r", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        data = json.loads(line)
                        total_count += 1
                        label_counts[int(data.get("label", 0))] += 1
        return {
            "total_samples": total_count,
            "label_distribution": { LABEL_NAMES[k]: v for k, v in label_counts.items() },
            "data_files": len(list(TRAINING_DATA_DIR.glob("training_data_*.jsonl")))
        }
    except Exception as e:
        LOGGER.error(f"Failed to get training data stats: {e}")
        return {"error": str(e)}


@app.get("/training-data/stats-all")
def get_training_data_stats_all() -> Dict[str, Any]:
    """저장된 학습 데이터 통계 조회 (영구 + 임시 모두)"""
    try:
        def accumulate_from_dir(base: Path, total_label_counts: Dict[int, int]) -> int:
            total = 0
            for data_file in base.glob("training_data_*.jsonl"):
                with open(data_file, "r", encoding="utf-8") as f:
                    for line in f:
                        if line.strip():
                            data = json.loads(line)
                            total += 1
                            lbl = int(data.get("label", 0))
                            if lbl in total_label_counts:
                                total_label_counts[lbl] += 1
            return total

        label_counts = {0: 0, 1: 0, 2: 0}
        total_count = 0
        total_count += accumulate_from_dir(TRAINING_DATA_DIR, label_counts)
        total_count += accumulate_from_dir(TEMP_TRAINING_DATA_DIR, label_counts)

        return {
            "total_samples": total_count,
            "label_distribution": { LABEL_NAMES[k]: v for k, v in label_counts.items() },
            "data_files": len(list(TRAINING_DATA_DIR.glob("training_data_*.jsonl"))) + len(list(TEMP_TRAINING_DATA_DIR.glob("training_data_*.jsonl")))
        }
    except Exception as e:
        LOGGER.error(f"Failed to get training data stats(all): {e}")
        return {"error": str(e)}


@app.post("/model/reload")
def reload_model() -> Dict[str, Any]:
	"""모델을 다시 로드 (새로 학습된 모델 적용)"""
	global TOKENIZER, MODEL, DEVICE
	
	try:
		LOGGER.info("Reloading model...")
		TOKENIZER, MODEL, DEVICE = _load_model(MODEL_DIR)
		LOGGER.info("Model reloaded successfully")
		return {"success": True, "message": "Model reloaded successfully"}
	except Exception as e:
		LOGGER.error(f"Failed to reload model: {e}")
		return {"success": False, "message": f"Failed to reload model: {e}"}


@app.post("/model/retrain")
def start_retraining(background_tasks: BackgroundTasks) -> Dict[str, Any]:
	"""모델 재학습 시작"""
	global TRAINING_STATUS
	
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
			file_name = os.path.basename(file_path)
			file_size = os.path.getsize(file_path)
			file_date = datetime.fromtimestamp(os.path.getmtime(file_path))
			
			# 파일 내용 개수 계산
			count = 0
			with open(file_path, 'r', encoding='utf-8') as f:
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
		
		# 날짜순으로 정렬 (최신순)
		files.sort(key=lambda x: x['date'], reverse=True)
		return {"files": files}
	except Exception as e:
		raise HTTPException(status_code=500, detail=f"파일 목록 조회 실패: {str(e)}")


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
	import uvicorn
	port = int(os.environ.get("PORT", 8000))
	uvicorn.run("app:app", host="127.0.0.1", port=port, reload=False)
