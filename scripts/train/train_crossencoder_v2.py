"""Fine-tune cross-encoder/ms-marco-MiniLM-L-12-v2 on the v2 dataset.

Loads data/crossencoder_v2/{train,valid}.jsonl with rows
{anchor, candidate, label}. Trains a CE with BCEWithLogits via the
sentence-transformers CrossEncoder API.

Usage:
    python scripts/train/train_crossencoder_v2.py
"""
import argparse
import json
import os
from pathlib import Path

import torch
from sentence_transformers.cross_encoder import (
    CrossEncoder,
    CrossEncoderTrainer,
    CrossEncoderTrainingArguments,
)
from sentence_transformers.cross_encoder.losses import LambdaLoss, NDCGLoss2PPScheme
from datasets import Dataset

os.environ.setdefault("PYTORCH_MPS_HIGH_WATERMARK_RATIO", "0.0")

parser = argparse.ArgumentParser()
parser.add_argument("--data_dir", default="data/crossencoder_v2")
parser.add_argument("--out_dir", default="models/crossencoder_v2")
parser.add_argument("--base_model", default="cross-encoder/ms-marco-MiniLM-L-12-v2")
parser.add_argument("--epochs", type=int, default=1)
parser.add_argument("--batch_size", type=int, default=32)
parser.add_argument("--lr", type=float, default=2e-5)
parser.add_argument("--warmup_steps", type=int, default=500)
parser.add_argument("--max_length", type=int, default=256)
parser.add_argument("--eval_steps", type=int, default=2000)
parser.add_argument("--logging_steps", type=int, default=100)
args = parser.parse_args()


def load_jsonl(path: str) -> list[dict]:
    rows = []
    with open(path) as f:
        for line in f:
            d = json.loads(line)
            rows.append({
                "query":  d["query"],
                "docs":   d["docs"],
                "labels": [float(x) for x in d["labels"]],
            })
    return rows


print(f"Loading {args.data_dir}...")
train_rows = load_jsonl(f"{args.data_dir}/train.jsonl")
valid_rows = load_jsonl(f"{args.data_dir}/valid.jsonl")
print(f"  Train: {len(train_rows):,}  Valid: {len(valid_rows):,}")

train_ds = Dataset.from_list(train_rows)
valid_ds = Dataset.from_list(valid_rows) if valid_rows else None

print(f"Loading base CE: {args.base_model}")
model = CrossEncoder(args.base_model, num_labels=1, max_length=args.max_length)
loss  = LambdaLoss(model, weighting_scheme=NDCGLoss2PPScheme())

device = "mps" if torch.backends.mps.is_available() else (
    "cuda" if torch.cuda.is_available() else "cpu"
)
print(f"Device: {device}")

steps_per_epoch = max(len(train_rows) // args.batch_size, 1)
print(f"Steps per epoch: {steps_per_epoch:,}")

training_args = CrossEncoderTrainingArguments(
    output_dir=args.out_dir,
    num_train_epochs=args.epochs,
    per_device_train_batch_size=args.batch_size,
    per_device_eval_batch_size=args.batch_size,
    learning_rate=args.lr,
    warmup_steps=args.warmup_steps,
    lr_scheduler_type="cosine",
    eval_strategy="steps" if valid_ds is not None else "no",
    eval_steps=args.eval_steps,
    save_strategy="steps",
    save_steps=args.eval_steps,
    save_total_limit=2,
    load_best_model_at_end=False,  # v1 was killed by eval_loss-best reload that didn't track ranking quality
    metric_for_best_model="eval_loss",
    greater_is_better=False,
    logging_steps=args.logging_steps,
    fp16=False,
    bf16=False,
    dataloader_num_workers=0,
    report_to="none",
)

trainer = CrossEncoderTrainer(
    model=model,
    args=training_args,
    train_dataset=train_ds,
    eval_dataset=valid_ds,
    loss=loss,
)

print(f"Training: {steps_per_epoch * args.epochs:,} steps total")
trainer.train()

final_dir = Path(args.out_dir) / "final"
final_dir.mkdir(parents=True, exist_ok=True)
print(f"Saving to {final_dir}")
model.save(str(final_dir))
print("Done.")
