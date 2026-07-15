"""Fine-tune Qwen3 to predict L0 semantic bucket IDs from conversation context.

Two-stage:
  Stage 1 (fast, ~15 min): train only the classification head / lm_head
           on B/C examples (bucket↔description) to ground bucket tokens.
  Stage 2 (main, ~2-3h): LoRA fine-tune on all examples (A+B+C).

Output: models/sid_qwen3/final/

Usage:
    python scripts/train/finetune_sid_qwen3.py \
        --data_dir data/semantic_id_llm \
        --out_dir models/sid_qwen3 \
        --base_model Qwen/Qwen3-1.5B
"""
import argparse
import json
from pathlib import Path

import torch
from datasets import Dataset
from peft import LoraConfig, TaskType, get_peft_model
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    TrainingArguments,
    Trainer,
)

ap = argparse.ArgumentParser()
ap.add_argument("--data_dir",    default="data/semantic_id_llm")
ap.add_argument("--out_dir",     default="models/sid_qwen3")
ap.add_argument("--base_model",  default="Qwen/Qwen3-1.5B")
ap.add_argument("--epochs",      type=int, default=3)
ap.add_argument("--batch_size",  type=int, default=4)
ap.add_argument("--grad_accum",  type=int, default=8)
ap.add_argument("--lr",          type=float, default=2e-4)
ap.add_argument("--lora_r",      type=int, default=16)
ap.add_argument("--lora_alpha",  type=int, default=32)
ap.add_argument("--max_length",  type=int, default=1024)
ap.add_argument("--warmup_steps",type=int, default=100)
ap.add_argument("--eval_steps",  type=int, default=200)
ap.add_argument("--save_steps",  type=int, default=500)
args = ap.parse_args()

data_dir = Path(args.data_dir)
out_dir  = Path(args.out_dir)
out_dir.mkdir(parents=True, exist_ok=True)

device = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {device}")

# ── Load tokenizer + model ────────────────────────────────────────────────────
print(f"Loading {args.base_model}...")
tokenizer = AutoTokenizer.from_pretrained(args.base_model, padding_side="right")
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    args.base_model,
    torch_dtype=torch.float32 if device == "mps" else torch.bfloat16,
    device_map={"": device},
)

# ── LoRA ──────────────────────────────────────────────────────────────────────
lora_cfg = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    r=args.lora_r,
    lora_alpha=args.lora_alpha,
    lora_dropout=0.05,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                    "gate_proj", "up_proj", "down_proj"],
    bias="none",
)
model = get_peft_model(model, lora_cfg)
model.print_trainable_parameters()

# ── Tokenise data ─────────────────────────────────────────────────────────────
def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(l) for l in path.read_text().splitlines() if l.strip()]

def tokenise(examples: list[dict]) -> Dataset:
    input_ids_list, labels_list = [], []
    for ex in examples:
        messages = ex["messages"]
        # Format as chat
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False,
        )
        enc = tokenizer(
            text,
            max_length=args.max_length,
            truncation=True,
            padding=False,
            return_tensors=None,
        )
        ids = enc["input_ids"]

        # Mask everything before the last assistant turn
        # Find the assistant response start by encoding the prompt without it
        prompt_messages = messages[:-1]
        prompt_text = tokenizer.apply_chat_template(
            prompt_messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        prompt_len = len(tokenizer(prompt_text, add_special_tokens=False)["input_ids"])

        labels = [-100] * prompt_len + ids[prompt_len:]
        labels = labels[:args.max_length]
        ids    = ids[:args.max_length]

        input_ids_list.append(ids)
        labels_list.append(labels)

    return Dataset.from_dict({"input_ids": input_ids_list, "labels": labels_list})

print("Tokenising train...")
train_raw  = load_jsonl(data_dir / "train.jsonl")
valid_raw  = load_jsonl(data_dir / "valid.jsonl")
train_ds   = tokenise(train_raw)
valid_ds   = tokenise(valid_raw)
print(f"  Train: {len(train_ds):,}  Valid: {len(valid_ds):,}")

# ── Training ──────────────────────────────────────────────────────────────────
collator = DataCollatorForSeq2Seq(tokenizer, model=model, padding=True, pad_to_multiple_of=8)

training_args = TrainingArguments(
    output_dir=str(out_dir),
    num_train_epochs=args.epochs,
    per_device_train_batch_size=args.batch_size,
    per_device_eval_batch_size=args.batch_size,
    gradient_accumulation_steps=args.grad_accum,
    learning_rate=args.lr,
    warmup_steps=args.warmup_steps,
    lr_scheduler_type="cosine",
    weight_decay=0.01,
    evaluation_strategy="steps",
    eval_steps=args.eval_steps,
    save_strategy="steps",
    save_steps=args.save_steps,
    save_total_limit=2,
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    greater_is_better=False,
    logging_steps=50,
    report_to="none",
    use_mps_device=(device == "mps"),
    fp16=False,
    bf16=(device == "cuda"),
    gradient_checkpointing=(device != "mps"),
    dataloader_num_workers=0,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_ds,
    eval_dataset=valid_ds,
    data_collator=collator,
)

print("Training...")
trainer.train()

# ── Save merged model ─────────────────────────────────────────────────────────
print("Merging LoRA and saving...")
final_dir = out_dir / "final"
merged = model.merge_and_unload()
merged.save_pretrained(str(final_dir))
tokenizer.save_pretrained(str(final_dir))
print(f"Saved → {final_dir}")

# ── Quick eval: bucket accuracy on valid ──────────────────────────────────────
print("\nBucket prediction accuracy (valid set):")
merged.eval()
correct, total = 0, 0
for ex in valid_raw[:200]:
    if ex["messages"][0]["content"].startswith("You are a music recommendation"):
        gold = int(ex["messages"][-1]["content"].strip())
        prompt = ex["messages"][:-1]
        text = tokenizer.apply_chat_template(prompt, tokenize=False, add_generation_prompt=True)
        ids = tokenizer(text, return_tensors="pt").to(device)
        with torch.no_grad():
            out = merged.generate(**ids, max_new_tokens=4, do_sample=False)
        pred_text = tokenizer.decode(out[0][ids["input_ids"].shape[1]:], skip_special_tokens=True).strip()
        try:
            pred = int(pred_text.split()[0])
            if pred == gold:
                correct += 1
        except ValueError:
            pass
        total += 1

if total:
    print(f"  Top-1 accuracy: {correct}/{total} = {100*correct/total:.1f}%")
