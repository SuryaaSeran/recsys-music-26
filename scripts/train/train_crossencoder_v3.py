"""Fine-tune BAAI/bge-reranker-v2-m3 on the v3 listwise dataset.

Differences from v2:
  - Base model: bge-reranker-v2-m3 (XLM-RoBERTa, ~600M, 8K context). Loaded
    via sentence-transformers CrossEncoder (it auto-routes XLM-RoBERTa
    AutoModelForSequenceClassification with num_labels=1).
  - LoRA (PEFT) r=16 alpha=32 on attention projections (target_modules =
    ["query", "key", "value", "dense"] which are the XLM-RoBERTa attention
    layer names).
  - Loss: same LambdaLoss + NDCGLoss2PPScheme as v2 (loss wasn't the
    issue; the negatives were).
  - Single epoch, no eval_loss-best reload.

Usage:
    python scripts/train/train_crossencoder_v3.py
"""
import argparse
import json
import os
from pathlib import Path

import torch
from peft import LoraConfig, TaskType
from sentence_transformers.cross_encoder import (
    CrossEncoder,
    CrossEncoderTrainer,
    CrossEncoderTrainingArguments,
)
from sentence_transformers.cross_encoder.losses import LambdaLoss, NDCGLoss2PPScheme
from datasets import Dataset

os.environ.setdefault("PYTORCH_MPS_HIGH_WATERMARK_RATIO", "0.0")

parser = argparse.ArgumentParser()
parser.add_argument("--data_dir", default="data/crossencoder_v3")
parser.add_argument("--out_dir", default="models/crossencoder_v3")
parser.add_argument("--base_model", default="BAAI/bge-reranker-v2-m3")
parser.add_argument("--epochs", type=int, default=1)
parser.add_argument("--batch_size", type=int, default=4)
parser.add_argument("--grad_accum", type=int, default=4)
parser.add_argument("--lr", type=float, default=1e-4)
parser.add_argument("--warmup_steps", type=int, default=500)
parser.add_argument("--max_length", type=int, default=512)
parser.add_argument("--no_lora", action="store_true")
parser.add_argument("--lora_r", type=int, default=16)
parser.add_argument("--lora_alpha", type=int, default=32)
parser.add_argument("--lora_dropout", type=float, default=0.05)
parser.add_argument("--eval_steps", type=int, default=500)
parser.add_argument("--logging_steps", type=int, default=50)
parser.add_argument("--fp16", action="store_true")
parser.add_argument("--bf16", action="store_true")
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
print(f"  Train groups: {len(train_rows):,}  Valid: {len(valid_rows):,}")

train_ds = Dataset.from_list(train_rows)
valid_ds = Dataset.from_list(valid_rows) if valid_rows else None

print(f"Loading base CE: {args.base_model}")
model = CrossEncoder(args.base_model, num_labels=1, max_length=args.max_length,
                     trust_remote_code=True)

if not args.no_lora:
    lora_cfg = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type=TaskType.SEQ_CLS,
        target_modules=["query", "key", "value", "dense"],
    )
    model.model.add_adapter(lora_cfg)
    print(f"  LoRA enabled: r={args.lora_r}, alpha={args.lora_alpha}")

loss = LambdaLoss(model, weighting_scheme=NDCGLoss2PPScheme())

device = "mps" if torch.backends.mps.is_available() else (
    "cuda" if torch.cuda.is_available() else "cpu"
)
print(f"Device: {device}")
steps_per_epoch = max(len(train_rows) // (args.batch_size * args.grad_accum), 1)
print(f"Steps per epoch (after grad_accum): {steps_per_epoch:,}")

training_args = CrossEncoderTrainingArguments(
    output_dir=args.out_dir,
    num_train_epochs=args.epochs,
    per_device_train_batch_size=args.batch_size,
    per_device_eval_batch_size=args.batch_size,
    gradient_accumulation_steps=args.grad_accum,
    learning_rate=args.lr,
    warmup_steps=args.warmup_steps,
    lr_scheduler_type="cosine",
    eval_strategy="steps" if valid_ds is not None else "no",
    eval_steps=args.eval_steps,
    save_strategy="steps",
    save_steps=args.eval_steps,
    save_total_limit=2,
    load_best_model_at_end=False,
    logging_steps=args.logging_steps,
    fp16=args.fp16,
    bf16=args.bf16,
    dataloader_num_workers=0,
    report_to="none",
)

trainer = CrossEncoderTrainer(
    model=model, args=training_args,
    train_dataset=train_ds, eval_dataset=valid_ds, loss=loss,
)
print(f"Total optimiser updates: {steps_per_epoch * args.epochs:,}")
trainer.train()

final_dir = Path(args.out_dir) / "final"
final_dir.mkdir(parents=True, exist_ok=True)

# If LoRA was used, try to merge into base weights so the saved model can be
# loaded back via vanilla CrossEncoder(path) without PEFT.
if not args.no_lora:
    hf = trainer.model.model
    if hasattr(hf, "merge_and_unload"):
        print("Merging LoRA adapters into base weights...")
        merged = hf.merge_and_unload()
        trainer.model.model = merged
    else:
        print("Adapter saved via transformers' native PEFT path; auto-applies on reload.")

print(f"Saving to {final_dir}")
trainer.model.save_pretrained(str(final_dir))
print("Done.")
