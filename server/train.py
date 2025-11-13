#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
KcELECTRA incremental finetuning with BitFit
- Keeps base weights stable
- Only trains bias and LayerNorm parameters
"""

import os
import json
import logging
from pathlib import Path
from typing import List, Tuple
from collections import Counter

# Disable torch compile/dynamo (for PyInstaller compatibility)
os.environ["TORCH_COMPILE_DISABLE"] = "1"
os.environ["TORCHDYNAMO_DISABLE"] = "1"
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

import numpy as np
import torch
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

from transformers import (
    AutoTokenizer,
    AutoConfig,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback,
)

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
log = logging.getLogger("kcelectra_bitfit")


class ChatDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length: int = 256):
        self.texts = texts
        self.labels = labels
        self.tok = tokenizer
        self.maxlen = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, i):
        enc = self.tok(
            str(self.texts[i]),
            truncation=True,
            padding="max_length",
            max_length=self.maxlen,
            return_tensors="pt",
        )
        return {
            "input_ids": enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "labels": torch.tensor(int(self.labels[i]), dtype=torch.long),
        }

def load_training_data(data_dir: Path, augment_factor: int = 3000) -> Tuple[List[str], List[int]]:
    texts: List[str] = []
    labels: List[int] = []
    files = sorted(data_dir.glob("training_data_*.jsonl"))
    if not files:
        log.error("no training_data_*.jsonl found")
        return texts, labels
    for p in files:
        with p.open("r", encoding="utf-8") as f:
            for line in f:
                s = line.strip()
                if not s:
                    continue
                try:
                    obj = json.loads(s)
                    texts.append(obj["text"])
                    labels.append(int(obj["label"]))
                except Exception as e:
                    log.warning(f"skip line in {p.name}: {e}")
    
    # 데이터 증강: 각 샘플을 augment_factor번 복제
    if augment_factor > 1 and texts:
        log.info(f"원본 데이터: {len(texts)}개 -> 증강 후: {len(texts) * augment_factor}개 (x{augment_factor})")
        texts = texts * augment_factor
        labels = labels * augment_factor
    
    return texts, labels


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=1)
    p, r, f1, _ = precision_recall_fscore_support(labels, preds, average="weighted", zero_division=0)
    acc = accuracy_score(labels, preds)
    return {"accuracy": acc, "f1": f1, "precision": p, "recall": r}


class WLossTrainer(Trainer):
    def __init__(self, class_weights: torch.Tensor | None = None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.class_weights = class_weights

    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits
        if self.class_weights is not None:
            loss_fct = torch.nn.CrossEntropyLoss(weight=self.class_weights.to(logits.device))
        else:
            loss_fct = torch.nn.CrossEntropyLoss()
        loss = loss_fct(logits, labels)
        return (loss, outputs) if return_outputs else loss


def apply_bitfit(model):
    """Enable training for classifier, bias, and LayerNorm parameters only"""
    for n, p in model.named_parameters():
        if "classifier" in n:
            p.requires_grad = True
        elif n.endswith(".bias") or ("LayerNorm" in n):
            p.requires_grad = True
        else:
            p.requires_grad = False
    return model


def train(
    model_dir: Path,
    training_data_dir: Path,
    output_dir: Path,
    max_length: int = 256,
    batch_size: int = 16,
    epochs: int = 1,
    lr: float = 5e-6,
    warmup_ratio: float = 0.06,
    use_class_weights: bool = True,
    seed: int = 42,
    augment_factor: int = 3000,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info(f"device: {device} | method: bitfit")

    config = AutoConfig.from_pretrained(str(model_dir))
    tokenizer = AutoTokenizer.from_pretrained(str(model_dir), use_fast=True)
    model = AutoModelForSequenceClassification.from_pretrained(str(model_dir), config=config)

    texts, labels = load_training_data(training_data_dir, augment_factor=augment_factor)
    if len(texts) == 0:
        log.error("no training data")
        return False

    uniq = sorted(set(labels))
    expect = (
        sorted(set(int(v) for v in config.label2id.values()))
        if isinstance(config.label2id, dict) and len(config.label2id) > 0
        else list(range(config.num_labels))
    )
    if set(uniq) - set(expect):
        log.error(f"label mismatch: data {uniq} vs model {expect}")
        return False

    X_tr, X_val, y_tr, y_val = train_test_split(texts, labels, test_size=0.2, random_state=seed, stratify=labels)

    ds_tr = ChatDataset(X_tr, y_tr, tokenizer, max_length)
    ds_val = ChatDataset(X_val, y_val, tokenizer, max_length)

    class_weights = None
    if use_class_weights:
        cnt = Counter(y_tr)
        num_labels = int(getattr(config, "num_labels", max(labels) + 1))
        total = max(1, len(y_tr))
        weights = []
        for c in range(num_labels):
            c_count = cnt.get(c, 0)
            if c_count <= 0:
                weights.append(0.0)
            else:
                weights.append(total / (num_labels * c_count))
        class_weights = torch.tensor(weights, dtype=torch.float)

    model = apply_bitfit(model)
    model.to(device)

    training_args = TrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        learning_rate=lr,
        warmup_ratio=warmup_ratio,
        weight_decay=0.01,
        logging_dir=str(output_dir / "logs"),
        logging_steps=10,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        greater_is_better=True,
        seed=seed,
        fp16=torch.cuda.is_available(),
        max_grad_norm=1.0,
        report_to=None,
        torch_compile=False,  # Disable torch compile for PyInstaller compatibility
    )

    trainer = WLossTrainer(
        class_weights=class_weights,
        model=model,
        args=training_args,
        train_dataset=ds_tr,
        eval_dataset=ds_val,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2)],
    )

    log.info("start training")
    trainer.train()
    log.info("evaluate")
    eval_res = trainer.evaluate()
    log.info(f"eval: {eval_res}")

    output_dir.mkdir(parents=True, exist_ok=True)
    # BitFit saves full model (not adapter)
    trainer.save_model()
    tokenizer.save_pretrained(str(output_dir))
    log.info(f"saved model to {output_dir}")

    return True


def main():
    import argparse

    ap = argparse.ArgumentParser(description="KcELECTRA BitFit finetuning")
    ap.add_argument("--model-dir", type=str, required=True)
    ap.add_argument("--training-data-dir", type=str, required=True)
    ap.add_argument("--output-dir", type=str, required=True)
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--lr", type=float, default=5e-6)
    ap.add_argument("--warmup-ratio", type=float, default=0.06)
    ap.add_argument("--max-length", type=int, default=256)
    ap.add_argument("--no-class-weights", action="store_true")
    ap.add_argument("--augment-factor", type=int, default=3000, help="데이터 증강 배수 (기본값: 3000)")
    args = ap.parse_args()

    ok = train(
        model_dir=Path(args.model_dir).resolve(),
        training_data_dir=Path(args.training_data_dir).resolve(),
        output_dir=Path(args.output_dir).resolve(),
        max_length=args.max_length,
        batch_size=args.batch_size,
        epochs=args.epochs,
        lr=args.lr,
        warmup_ratio=args.warmup_ratio,
        use_class_weights=(not args.no_class_weights),
        augment_factor=args.augment_factor,
    )
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
