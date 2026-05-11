"""
Fine-tune a cross-encoder for music retrieval reranking.

Input format: (query_text, track_text) pairs with binary labels.
Positive: (query, gold_track_text) → label=1
Negative: (query, bm25_top_k_non_gold) → label=0

Usage:
    python scripts/train_crossencoder.py \
        --data_dir data/crossencoder_v1 \
        --out_dir models/crossencoder_v1 \
        --epochs 2 --batch_size 16
"""
import argparse
import json
import os

import torch
from datasets import Dataset
from sentence_transformers.cross_encoder import CrossEncoder
from sentence_transformers.cross_encoder.trainer import CrossEncoderTrainer
from sentence_transformers.cross_encoder.training_args import CrossEncoderTrainingArguments
from tqdm import tqdm

os.environ.setdefault("PYTORCH_MPS_HIGH_WATERMARK_RATIO", "0.0")

parser = argparse.ArgumentParser()
parser.add_argument("--data_dir", default="data/crossencoder_v1")
parser.add_argument("--out_dir", default="models/crossencoder_v1")
parser.add_argument("--base_model", default="cross-encoder/ms-marco-MiniLM-L-6-v2")
parser.add_argument("--epochs", type=int, default=2)
parser.add_argument("--batch_size", type=int, default=16)
parser.add_argument("--lr", type=float, default=2e-5)
parser.add_argument("--warmup_steps", type=int, default=200)
args = parser.parse_args()


def load_jsonl(path):
    with open(path) as f:
        return [json.loads(line) for line in f]


print(f"Loading data from {args.data_dir}...")
_train_small = f"{args.data_dir}/train_small.jsonl"
train_data = load_jsonl(_train_small if __import__('os').path.exists(_train_small) else f"{args.data_dir}/train.jsonl")
valid_data = load_jsonl(f"{args.data_dir}/valid.jsonl")
print(f"  Train: {len(train_data):,}  Valid: {len(valid_data):,}")

train_ds = Dataset.from_list(train_data)
valid_ds = Dataset.from_list(valid_data)

print(f"Loading cross-encoder: {args.base_model}")
model = CrossEncoder(args.base_model, num_labels=1, max_length=512)

device = "mps" if torch.backends.mps.is_available() else "cpu"
print(f"Device: {device}")

steps_per_epoch = len(train_data) // args.batch_size
eval_steps = max(steps_per_epoch // 4, 100)

training_args = CrossEncoderTrainingArguments(
    output_dir=args.out_dir,
    num_train_epochs=args.epochs,
    per_device_train_batch_size=args.batch_size,
    per_device_eval_batch_size=args.batch_size,
    learning_rate=args.lr,
    warmup_steps=args.warmup_steps,
    eval_strategy="steps",
    eval_steps=eval_steps,
    save_strategy="steps",
    save_steps=eval_steps,
    save_total_limit=2,
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    greater_is_better=False,
    logging_steps=100,
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
)

print(f"Training: {steps_per_epoch * args.epochs} steps total, eval every {eval_steps}")
trainer.train()

print(f"Saving to {args.out_dir}/final")
model.save(f"{args.out_dir}/final")
print("Done.")
