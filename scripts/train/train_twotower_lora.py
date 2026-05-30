"""Fine-tune intfloat/multilingual-e5-base (or any XLM-R family model) as a two-tower
retrieval model using LoRA.

Why LoRA: full fine-tuning of 279M+ param models OOMs on Apple MPS (M4 16GB) because
Adam optimizer states alone require ~3x model weights. LoRA trains only ~885K params
(r=16 on q/k/v, 0.32% of total), dropping optimizer memory from ~3GB to ~7MB.
Gradient checkpointing (--gradient_checkpointing) is also required to keep activation
memory below the MPS limit.

Model notes:
  - multilingual-e5-base: 12L, 768-dim, 279M params, 512-token max. Requires
    "query: " prefix on anchors and "passage: " prefix on documents at both
    train and inference time. Use build_twotower_v8_data.py to build training data
    with these prefixes already baked in.
  - LoRA target modules for XLM-RoBERTa: "query,key,value" (default).
    For Qwen-family use "q_proj,k_proj,v_proj,o_proj".

After training, LoRA adapters are merged into the base weights and saved as a vanilla
SentenceTransformer so downstream code (build_twotower_index.py, inference) loads it
with SentenceTransformer(path) without needing peft installed.

Trained model: models/twotower_v8/final/
Index must be rebuilt after training: scripts/train/build_twotower_index.py
  --model models/twotower_v8/final --out_dir cache/twotower_v8
  --doc_prefix "passage: " --batch_size 32

Usage:
    python scripts/train/train_twotower_lora.py \\
        --data_dir data/twotower_v8 \\
        --out_dir models/twotower_v8 \\
        --epochs 2 --batch_size 16 --grad_accum 4 \\
        --lr 1e-4 --warmup_steps 200 --gradient_checkpointing
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
parser.add_argument("--data_dir", default="data/twotower_v8")
parser.add_argument("--out_dir", default="models/twotower_v8")
parser.add_argument("--base_model", default="intfloat/multilingual-e5-base")
parser.add_argument("--epochs", type=int, default=2)
parser.add_argument("--batch_size", type=int, default=8)
parser.add_argument("--grad_accum", type=int, default=4)
parser.add_argument("--lr", type=float, default=1e-4)
parser.add_argument("--warmup_steps", type=int, default=200)
parser.add_argument("--max_seq_length", type=int, default=512)
parser.add_argument("--no_lora", action="store_true",
                    help="Disable LoRA and do full fine-tune (very memory-hungry).")
parser.add_argument("--lora_r", type=int, default=16)
parser.add_argument("--lora_alpha", type=int, default=32)
parser.add_argument("--lora_dropout", type=float, default=0.05)
parser.add_argument("--lora_target_modules", default="query,key,value",
                    help="Comma-separated list of attention module name substrings to apply LoRA. "
                         "XLM-RoBERTa uses 'query,key,value'. Qwen uses 'q_proj,k_proj,v_proj,o_proj'.")
parser.add_argument("--fp16", action="store_true")
parser.add_argument("--gradient_checkpointing", action="store_true",
                    help="Enable gradient checkpointing to trade compute for memory (~20x less activation memory).")
parser.add_argument("--use_hard_neg", action="store_true",
                    help="Include negative_1 column as explicit negative.")
parser.add_argument("--max_train", type=int, default=0,
                    help="If >0, cap training set size for a quick sanity run.")
parser.add_argument("--eval_steps", type=int, default=500)
parser.add_argument("--logging_steps", type=int, default=100)
args = parser.parse_args()


parser.add_argument("--n_hard_negs", type=int, default=1,
                    help="Number of explicit hard negatives to load from JSONL columns "
                         "negative_1..negative_N. Only used when --use_hard_neg is set. "
                         "SentenceTransformer MNRL uses these as additional negatives "
                         "alongside in-batch negatives. More hard negs = harder training.")


def load_jsonl(path: str, use_hard_neg: bool) -> list[dict]:
    n_neg = args.n_hard_negs if use_hard_neg else 0
    rows = []
    if not os.path.exists(path):
        return rows
    with open(path) as f:
        for line in f:
            d = json.loads(line)
            row = {"anchor": d["anchor"], "positive": d["positive"]}
            if n_neg > 0:
                # Load up to n_neg explicit negatives
                has_any = False
                for ni in range(1, n_neg + 1):
                    key = f"negative_{ni}"
                    val = d.get(key, "").strip()
                    if val:
                        row[f"negative_{ni}"] = val
                        has_any = True
                    else:
                        break  # stop at first missing negative
                if not has_any:
                    continue  # skip rows without any hard negatives
            rows.append(row)
    return rows


print(f"Loading data from {args.data_dir} (use_hard_neg={args.use_hard_neg})...")
train_data = load_jsonl(f"{args.data_dir}/train.jsonl", use_hard_neg=args.use_hard_neg)
valid_data = load_jsonl(f"{args.data_dir}/valid.jsonl", use_hard_neg=args.use_hard_neg)
print(f"  Train: {len(train_data):,}  Valid: {len(valid_data):,}")

if args.max_train > 0 and len(train_data) > args.max_train:
    import random
    random.Random(0).shuffle(train_data)
    train_data = train_data[: args.max_train]
    print(f"  Truncated train to {len(train_data):,}")

train_ds = Dataset.from_list(train_data)
valid_ds = Dataset.from_list(valid_data) if valid_data else None

print(f"Loading base model: {args.base_model}")
model = SentenceTransformer(args.base_model)
model.max_seq_length = args.max_seq_length

target_modules = [m.strip() for m in args.lora_target_modules.split(",") if m.strip()]

if not args.no_lora:
    lora_cfg = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type=TaskType.FEATURE_EXTRACTION,
        target_modules=target_modules,
    )
    model.add_adapter(lora_cfg)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"  LoRA enabled: r={args.lora_r}, alpha={args.lora_alpha}, "
          f"target={target_modules}")
    print(f"  Trainable params: {trainable:,} / {total:,} "
          f"({100 * trainable / total:.2f}%)")

loss = MultipleNegativesRankingLoss(model)

device = "mps" if torch.backends.mps.is_available() else (
    "cuda" if torch.cuda.is_available() else "cpu"
)
print(f"Device: {device}")

steps_per_epoch = max(len(train_data) // (args.batch_size * args.grad_accum), 1)
print(f"Steps per epoch (after grad_accum): {steps_per_epoch:,}")
print(f"Total optimiser updates: {steps_per_epoch * args.epochs:,}")

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
    bf16=False,  # MPS does not support bfloat16
    gradient_checkpointing=args.gradient_checkpointing,
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
        print("Adapter saved via transformers native PEFT path; will auto-apply on reload.")

print(f"Saving to {final_dir}")
trainer.model.save(final_dir)
print("Done.")
