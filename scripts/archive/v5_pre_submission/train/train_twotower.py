"""
Fine-tune all-MiniLM-L6-v2 as a two-tower retrieval model.

Uses MultipleNegativesRankingLoss (in-batch negatives).
Strips hard-negative columns at load time to keep memory under MPS limit.

Usage:
    python scripts/train_twotower.py \
        --data_dir data/twotower \
        --out_dir models/twotower_v1 \
        --epochs 1 \
        --batch_size 16
"""
import argparse
import json
import os

import torch
from datasets import Dataset
from sentence_transformers import SentenceTransformer
from sentence_transformers.losses import MultipleNegativesRankingLoss, TripletLoss
from sentence_transformers.trainer import SentenceTransformerTrainer
from sentence_transformers.training_args import SentenceTransformerTrainingArguments

os.environ.setdefault("PYTORCH_MPS_HIGH_WATERMARK_RATIO", "0.0")

parser = argparse.ArgumentParser()
parser.add_argument("--data_dir", default="data/twotower")
parser.add_argument("--out_dir", default="models/twotower_v1")
parser.add_argument("--base_model", default="sentence-transformers/all-MiniLM-L6-v2")
parser.add_argument("--epochs", type=int, default=1)
parser.add_argument("--batch_size", type=int, default=16)
parser.add_argument("--lr", type=float, default=2e-5)
parser.add_argument("--warmup_steps", type=int, default=200)
parser.add_argument("--hard_neg", action="store_true", help="Include BM25 hard negative_1 in training")
parser.add_argument("--triplet", action="store_true", help="Use TripletLoss instead of MNRL (requires hard_neg)")
parser.add_argument("--triplet_margin", type=float, default=0.5, help="Margin for TripletLoss")
parser.add_argument("--trust_remote_code", action="store_true", help="Pass trust_remote_code=True to SentenceTransformer (e.g. nomic-embed)")
parser.add_argument("--max_seq_length", type=int, default=None, help="Override model max_seq_length after loading")
args = parser.parse_args()


def load_jsonl(path: str, use_hard_neg: bool = False) -> list[dict]:
    rows = []
    with open(path) as f:
        for line in f:
            d = json.loads(line)
            row = {"anchor": d["anchor"], "positive": d["positive"]}
            if use_hard_neg and d.get("negative_1", "").strip():
                row["negative"] = d["negative_1"]
            rows.append(row)
    return rows


print(f"Loading data from {args.data_dir} (hard_neg={args.hard_neg})...")
train_data = load_jsonl(f"{args.data_dir}/train.jsonl", use_hard_neg=args.hard_neg)
valid_data = load_jsonl(f"{args.data_dir}/valid.jsonl", use_hard_neg=args.hard_neg)
print(f"  Train: {len(train_data):,}  Valid: {len(valid_data):,}")
if args.hard_neg:
    has_neg = sum(1 for d in train_data if "negative" in d)
    print(f"  Examples with hard negative: {has_neg:,}/{len(train_data):,}")

train_ds = Dataset.from_list(train_data)
valid_ds = Dataset.from_list(valid_data)

print(f"Loading base model: {args.base_model}")
model = SentenceTransformer(args.base_model, trust_remote_code=args.trust_remote_code)
if args.max_seq_length:
    model.max_seq_length = args.max_seq_length
if args.triplet:
    if not args.hard_neg:
        raise ValueError("--triplet requires --hard_neg")
    loss = TripletLoss(model=model, triplet_margin=args.triplet_margin)
    # TripletLoss requires all examples have a negative; filter those without
    train_data = [d for d in train_data if "negative" in d]
    valid_data = [d for d in valid_data if "negative" in d]
    print(f"  After filtering for triplets: {len(train_data):,} train, {len(valid_data):,} valid")
    train_ds = Dataset.from_list(train_data)
    valid_ds = Dataset.from_list(valid_data)
    print(f"TripletLoss with margin={args.triplet_margin}")
else:
    loss = MultipleNegativesRankingLoss(model)

device = "mps" if torch.backends.mps.is_available() else "cpu"
print(f"Device: {device}")

steps_per_epoch = len(train_data) // args.batch_size
eval_steps = max(steps_per_epoch // 4, 100)

training_args = SentenceTransformerTrainingArguments(
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

trainer = SentenceTransformerTrainer(
    model=model,
    args=training_args,
    train_dataset=train_ds,
    eval_dataset=valid_ds,
    loss=loss,
)

print(f"Training: {steps_per_epoch * args.epochs} steps total, eval every {eval_steps}")
trainer.train()

print(f"Saving to {args.out_dir}/final")
model.save(f"{args.out_dir}/final")
print("Done.")
