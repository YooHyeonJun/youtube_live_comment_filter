#!/usr/bin/env python3
"""
YouTube Live Chat Filter - 추가 학습 스크립트
수집된 학습 데이터를 사용하여 기존 모델을 추가 학습시킵니다.
"""

import os
import json
import logging
from pathlib import Path
from typing import List, Dict, Any, Tuple
from datetime import datetime

import torch
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoTokenizer, 
    AutoModelForSequenceClassification,
    TrainingArguments, 
    Trainer,
    EarlyStoppingCallback
)
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import numpy as np

# 로깅 설정
logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
logger = logging.getLogger("yt_live_chat_filter_trainer")

class ChatDataset(Dataset):
    """채팅 데이터셋 클래스"""
    
    def __init__(self, texts: List[str], labels: List[int], tokenizer, max_length: int = 256):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]
        
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }

def load_training_data(data_dir: Path) -> Tuple[List[str], List[int]]:
    """학습 데이터 로드"""
    texts = []
    labels = []
    
    data_files = list(data_dir.glob("training_data_*.jsonl"))
    if not data_files:
        logger.warning("No training data files found")
        return texts, labels
    
    logger.info(f"Found {len(data_files)} training data files")
    
    for data_file in data_files:
        logger.info(f"Loading {data_file}")
        with open(data_file, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    try:
                        data = json.loads(line)
                        texts.append(data['text'])
                        labels.append(data['label'])
                    except json.JSONDecodeError as e:
                        logger.warning(f"Failed to parse line: {e}")
    
    logger.info(f"Loaded {len(texts)} training samples")
    return texts, labels

def compute_metrics(eval_pred):
    """평가 메트릭 계산"""
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    
    precision, recall, f1, _ = precision_recall_fscore_support(labels, predictions, average='weighted')
    accuracy = accuracy_score(labels, predictions)
    
    return {
        'accuracy': accuracy,
        'f1': f1,
        'precision': precision,
        'recall': recall
    }

def train_model(
    model_dir: Path,
    training_data_dir: Path,
    output_dir: Path,
    num_epochs: int = 3,
    batch_size: int = 16,
    learning_rate: float = 2e-5,
    warmup_steps: int = 100,
    max_length: int = 256
):
    """모델 추가 학습"""
    
    # 디바이스 설정
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    # 모델과 토크나이저 로드
    logger.info(f"Loading model from {model_dir}")
    tokenizer = AutoTokenizer.from_pretrained(str(model_dir))
    model = AutoModelForSequenceClassification.from_pretrained(str(model_dir))
    model.to(device)
    
    # 학습 데이터 로드
    texts, labels = load_training_data(training_data_dir)
    if len(texts) == 0:
        logger.error("No training data available")
        return False
    
    # 라벨 분포 확인
    label_counts = {}
    for label in labels:
        label_counts[label] = label_counts.get(label, 0) + 1
    logger.info(f"Label distribution: {label_counts}")
    
    # 데이터셋 생성
    dataset = ChatDataset(texts, labels, tokenizer, max_length)
    
    # 학습/검증 분할 (80:20)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    
    logger.info(f"Training samples: {len(train_dataset)}")
    logger.info(f"Validation samples: {len(val_dataset)}")
    
    # 학습 인수 설정
    training_args = TrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=num_epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        warmup_steps=warmup_steps,
        weight_decay=0.01,
        logging_dir=str(output_dir / "logs"),
        logging_steps=10,
        eval_strategy="steps",
        eval_steps=50,
        save_strategy="steps",
        save_steps=100,
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        greater_is_better=True,
        learning_rate=learning_rate,
        report_to=None,  # wandb 비활성화
    )
    
    # 트레이너 생성
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)]
    )
    
    # 학습 시작
    logger.info("Starting training...")
    trainer.train()
    
    # 최종 평가
    logger.info("Evaluating model...")
    eval_results = trainer.evaluate()
    logger.info(f"Final evaluation results: {eval_results}")
    
    # 모델 저장
    logger.info(f"Saving model to {output_dir}")
    trainer.save_model()
    tokenizer.save_pretrained(str(output_dir))
    
    return True

def main():
    """메인 함수"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Train YouTube Live Chat Filter model")
    parser.add_argument("--model-dir", type=str, default="../model", help="Path to base model directory")
    parser.add_argument("--training-data-dir", type=str, default="../training_data", help="Path to training data directory")
    parser.add_argument("--output-dir", type=str, default="../model_updated", help="Path to save updated model")
    parser.add_argument("--epochs", type=int, default=3, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=16, help="Training batch size")
    parser.add_argument("--learning-rate", type=float, default=2e-5, help="Learning rate")
    parser.add_argument("--max-length", type=int, default=256, help="Maximum sequence length")
    
    args = parser.parse_args()
    
    # 경로 설정
    model_dir = Path(args.model_dir).resolve()
    training_data_dir = Path(args.training_data_dir).resolve()
    output_dir = Path(args.output_dir).resolve()
    
    # 디렉토리 생성
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 학습 실행
    success = train_model(
        model_dir=model_dir,
        training_data_dir=training_data_dir,
        output_dir=output_dir,
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        max_length=args.max_length
    )
    
    if success:
        logger.info("Training completed successfully!")
        logger.info(f"Updated model saved to: {output_dir}")
    else:
        logger.error("Training failed!")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())
