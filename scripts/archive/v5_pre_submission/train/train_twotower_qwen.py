"""Fine-tune Qwen/Qwen3-Embedding-0.6B as a two-tower retrieval model with LoRA.

Loads data/twotower_v7/{train,valid,cold}.jsonl (anchor, positive [, negative_*]).
Default: MNRL with in-batch negatives only (matches v6 best-config baseline);
concatenates the session train stream with the cold-track stream so every
catalog track sees at least one positive pair.

Saves the merged model to <out_dir>/final so downstream code can load it
with a vanilla `SentenceTransformer(path)` call.

Usage:
    python scripts/train/train_twotower_qwen.py \
        --data_dir data/twotower_v7 \
        --out_dir models/twotower_v7 \
        --base_model Qwen/Qwen3-Embedding-0.6B \
        --epochs 1 --batch_size 4 --grad_accum 8 --lr 1e-4 --bf16
"""
import argparse
import json
import os

import torch
from datasets import Dataset
from peft import LoraConfig, TaskType
from sentence_transformers import SentenceTransformer
from sentence_transformers.losses import MultipleNegativesRankingLoss
from sentence_transformers.trainer import SentenceTransformerTrainer
from sentence_transformers.training_args import SentenceTransformerTrainingArguments

os.environ.setdefault("PYTORCH_MPS_HIGH_WATERMARK_RATIO", "0.0")

parser = argparse.ArgumentParser()
parser.add_argument("--data_dir", default="data/twotower_v7")
parser.add_argument("--out_dir", default="models/twotower_v7")
parser.add_argument("--base_model", default="Qwen/Qwen3-Embedding-0.6B")
parser.add_argument("--epochs", type=int, default=1)
parser.add_argument("--batch_size", type=int, default=4)
parser.add_argument("--grad_accum", type=int, default=8)
parser.add_argument("--lr", type=float, default=1e-4)
parser.add_argument("--warmup_steps", type=int, default=500)
parser.add_argument("--max_seq_length", type=int, default=512)
parser.add_argument("--no_lora", action="store_true",
                    help="Disable LoRA and do a full fine-tune (memory-hungry).")
parser.add_argument("--lora_r", type=int, default=16)
parser.add_argument("--lora_alpha", type=int, default=32)
parser.add_argument("--lora_dropout", type=float, default=0.05)
parser.add_argument("--fp16", action="store_true")
parser.add_argument("--bf16", action="store_true")
parser.add_argument("--use_hard_neg", action="store_true",
                    help="Include negative_1 column from session rows (drops cold stream).")
parser.add_argument("--max_train", type=int, default=0,
                    help="If >0, cap the training set size for a quick sanity run.")
parser.add_argument("--eval_steps", type=int, default=2000)
parser.add_argument("--logging_steps", type=int, default=100)
args = parser.parse_args()


def load_jsonl(path: str, use_hard_neg: bool) -> list[dict]:
    rows = []
    if not os.path.exists(path):
        return rows
    with open(path) as f:
        for line in f:
            d = json.loads(line)
            row = {"anchor": d["anchor"], "positive": d["positive"]}
            if use_hard_neg:
                if d.get("negative_1", "").strip():
                    row["negative"] = d["negative_1"]
                else:
                    continue
            rows.append(row)
    return rows


print(f"Loading data from {args.data_dir} (use_hard_neg={args.use_hard_neg})...")
session_train = load_jsonl(f"{args.data_dir}/train.jsonl", use_hard_neg=args.use_hard_neg)
session_valid = load_jsonl(f"{args.data_dir}/valid.jsonl", use_hard_neg=args.use_hard_neg)
cold_train = []
if not args.use_hard_neg:
    cold_train = load_jsonl(f"{args.data_dir}/cold.jsonl", use_hard_neg=False)
print(f"  Session train: {len(session_train):,}  valid: {len(session_valid):,}")
print(f"  Cold train:    {len(cold_train):,}")

train_data = session_train + cold_train
if args.max_train > 0 and len(train_data) > args.max_train:
    import random
    random.Random(0).shuffle(train_data)
    train_data = train_data[: args.max_train]
    print(f"  Truncated train to {len(train_data):,}")

train_ds = Dataset.from_list(train_data)
valid_ds = Dataset.from_list(session_valid) if session_valid else None

print(f"Loading base model: {args.base_model}")
model = SentenceTransformer(args.base_model)
model.max_seq_length = args.max_seq_length

if not args.no_lora:
    lora_cfg = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type=TaskType.FEATURE_EXTRACTION,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    )
    model.add_adapter(lora_cfg)
    print(f"  LoRA enabled: r={args.lora_r}, alpha={args.lora_alpha}, "
          f"dropout={args.lora_dropout}")

loss = MultipleNegativesRankingLoss(model)

device = "mps" if torch.backends.mps.is_available() else (
    "cuda" if torch.cuda.is_available() else "cpu"
)
print(f"Device: {device}")

steps_per_epoch = max(len(train_data) // (args.batch_size * args.grad_accum), 1)
print(f"Steps per epoch (after grad_accum): {steps_per_epoch:,}")

training_args = SentenceTransformerTrainingArguments(
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
    load_best_model_at_end=False,  # checkpoint reload strips PEFT wrapping
    metric_for_best_model="eval_loss",
    greater_is_better=False,
    logging_steps=args.logging_steps,
    fp16=args.fp16,
    bf16=args.bf16,
    dataloader_num_workers=0,
    report_to="none",
    remove_unused_columns=False,
)

trainer = SentenceTransformerTrainer(
    model=model,
    args=training_args,
    train_dataset=train_ds,
    eval_dataset=valid_ds,
    loss=loss,
)

print(f"Total optimiser updates: {steps_per_epoch * args.epochs:,}")
trainer.train()

final_dir = os.path.join(args.out_dir, "final")
os.makedirs(final_dir, exist_ok=True)

if not args.no_lora:
    hf_model = trainer.model[0].auto_model
    if hasattr(hf_model, "merge_and_unload"):
        print("Merging LoRA adapters into base weights...")
        merged = hf_model.merge_and_unload()
        trainer.model[0].auto_model = merged
    else:
        # transformers native PEFT integration: adapter is saved as
        # adapter_model.safetensors alongside the base and reapplied
        # automatically on SentenceTransformer(path) reload.
        print("Adapter saved via transformers' native PEFT path; will auto-apply on reload.")

print(f"Saving to {final_dir}")
trainer.model.save(final_dir)
print("Done.")
