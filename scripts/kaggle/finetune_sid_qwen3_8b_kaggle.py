"""Kaggle script: Fine-tune Qwen3-8B to predict L0 semantic bucket IDs.

Kaggle setup:
  - GPU: T4 x2 (recommended) or P100
  - Internet: ON
  - Secrets: HF_TOKEN, GITHUB_TOKEN
  - RAM: ~13GB model in 4-bit + training overhead

What it does:
  1. Clones repo (gets train.jsonl, valid.jsonl from data/semantic_id_llm/)
  2. Fine-tunes Qwen3-8B-Instruct with LoRA (4-bit QLoRA)
  3. Evaluates bucket prediction accuracy on valid set
  4. Pushes trained adapter to GitHub

Training task: given a TalkPlay conversation context → predict L0 bucket ID (0-63).
"""
import json
import os
import subprocess
import sys
from pathlib import Path

# ── Secrets ───────────────────────────────────────────────────────────────────
try:
    from kaggle_secrets import UserSecretsClient
    secrets = UserSecretsClient()
    os.environ["HF_TOKEN"]     = secrets.get_secret("HF_TOKEN")
    os.environ["GITHUB_TOKEN"] = secrets.get_secret("GITHUB_TOKEN")
    print("Secrets loaded")
except Exception:
    print("No Kaggle secrets — using env vars")

# ── Setup ─────────────────────────────────────────────────────────────────────
REPO_URL  = "https://github.com/SuryaaSeran/recsys-music-26.git"
REPO_DIR  = Path("/kaggle/working/repo")
DATA_DIR  = REPO_DIR / "data/semantic_id_llm"
OUT_DIR   = Path("/kaggle/working/sid_qwen3_8b")
MODEL_ID  = "Qwen/Qwen3-8B-Instruct"

import shutil
if REPO_DIR.exists():
    shutil.rmtree(REPO_DIR)
subprocess.run(["git", "clone", "--depth=1", REPO_URL, str(REPO_DIR)], check=True)
print(f"Cloned repo → {REPO_DIR}")

subprocess.run([sys.executable, "-m", "pip", "install", "-q",
                "transformers>=4.50", "peft>=0.14", "trl>=0.12",
                "bitsandbytes>=0.44", "accelerate>=1.0"], check=True)

import torch
from datasets import Dataset
from peft import LoraConfig, TaskType, get_peft_model
from transformers import (
    AutoModelForCausalLM, AutoTokenizer,
    BitsAndBytesConfig, DataCollatorForSeq2Seq,
    TrainingArguments, Trainer,
)

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {device}  GPUs: {torch.cuda.device_count()}")

# ── Load data ─────────────────────────────────────────────────────────────────
def load_jsonl(path):
    return [json.loads(l) for l in Path(path).read_text().splitlines() if l.strip()]

print("Loading data...")
train_raw = load_jsonl(DATA_DIR / "train.jsonl")
valid_raw = load_jsonl(DATA_DIR / "valid.jsonl")
print(f"  Train: {len(train_raw):,}  Valid: {len(valid_raw):,}")

# ── Load model (4-bit QLoRA) ──────────────────────────────────────────────────
print(f"Loading {MODEL_ID} in 4-bit...")
bnb_cfg = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
)
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, padding_side="right")
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    quantization_config=bnb_cfg,
    device_map="auto",
    torch_dtype=torch.bfloat16,
)
model.config.use_cache = False

lora_cfg = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                    "gate_proj", "up_proj", "down_proj"],
    bias="none",
)
model = get_peft_model(model, lora_cfg)
model.print_trainable_parameters()

# ── Tokenise ──────────────────────────────────────────────────────────────────
MAX_LEN = 1024

def tokenise(examples):
    input_ids_list, labels_list = [], []
    for ex in examples:
        messages = ex["messages"]
        full_text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False)
        prompt_text = tokenizer.apply_chat_template(
            messages[:-1], tokenize=False, add_generation_prompt=True)

        enc  = tokenizer(full_text,   max_length=MAX_LEN, truncation=True, padding=False)
        penc = tokenizer(prompt_text, max_length=MAX_LEN, truncation=True, padding=False,
                         add_special_tokens=False)

        ids       = enc["input_ids"]
        prompt_len = len(penc["input_ids"])
        labels    = [-100] * prompt_len + ids[prompt_len:]
        labels    = labels[:MAX_LEN]
        ids       = ids[:MAX_LEN]

        input_ids_list.append(ids)
        labels_list.append(labels)
    return Dataset.from_dict({"input_ids": input_ids_list, "labels": labels_list})

print("Tokenising...")
train_ds = tokenise(train_raw)
valid_ds = tokenise(valid_raw)
print(f"  Train: {len(train_ds):,}  Valid: {len(valid_ds):,}")

# ── Train ─────────────────────────────────────────────────────────────────────
collator = DataCollatorForSeq2Seq(tokenizer, model=model, padding=True, pad_to_multiple_of=8)

training_args = TrainingArguments(
    output_dir=str(OUT_DIR),
    num_train_epochs=3,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    gradient_accumulation_steps=8,       # effective batch = 32
    learning_rate=2e-4,
    warmup_steps=100,
    lr_scheduler_type="cosine",
    weight_decay=0.01,
    evaluation_strategy="steps",
    eval_steps=200,
    save_strategy="steps",
    save_steps=500,
    save_total_limit=2,
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    greater_is_better=False,
    logging_steps=50,
    report_to="none",
    fp16=False,
    bf16=True,
    gradient_checkpointing=True,
    dataloader_num_workers=2,
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

# ── Save adapter ──────────────────────────────────────────────────────────────
adapter_dir = OUT_DIR / "adapter"
model.save_pretrained(str(adapter_dir))
tokenizer.save_pretrained(str(adapter_dir))
print(f"Adapter saved → {adapter_dir}")

# ── Eval: bucket accuracy ─────────────────────────────────────────────────────
print("\nEvaluating bucket prediction accuracy...")
model.eval()

SYSTEM_A = "You are a music recommendation assistant. Given a conversation context, predict the semantic cluster ID (0-63) that best matches the next track to recommend. Output only the cluster number, nothing else."
a_valid = [ex for ex in valid_raw if ex["messages"][0]["content"] == SYSTEM_A][:300]

correct, total = 0, 0
top3_correct = 0
for ex in a_valid:
    gold = int(ex["messages"][-1]["content"].strip())
    prompt = tokenizer.apply_chat_template(
        ex["messages"][:-1], tokenize=False, add_generation_prompt=True)
    ids = tokenizer(prompt, return_tensors="pt", max_length=MAX_LEN, truncation=True).to(device)
    with torch.no_grad():
        out = model.generate(
            **ids, max_new_tokens=4, do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
    pred_text = tokenizer.decode(out[0][ids["input_ids"].shape[1]:], skip_special_tokens=True).strip()
    try:
        pred = int(pred_text.split()[0])
        if pred == gold: correct += 1
    except ValueError:
        pass
    total += 1

acc = 100 * correct / total if total else 0
print(f"  Top-1 bucket accuracy: {correct}/{total} = {acc:.1f}%")

# ── Push adapter to GitHub ────────────────────────────────────────────────────
token = os.environ.get("GITHUB_TOKEN", "")
if token:
    print("\nPushing adapter to GitHub...")
    adapter_git_path = "models/sid_qwen3_8b/adapter"
    dest = REPO_DIR / adapter_git_path
    dest.mkdir(parents=True, exist_ok=True)
    subprocess.run(["cp", "-r", str(adapter_dir) + "/.", str(dest)], check=True)
    subprocess.run(["git", "-C", str(REPO_DIR), "config", "user.email", "kaggle@reccysmusic"], check=True)
    subprocess.run(["git", "-C", str(REPO_DIR), "config", "user.name", "Kaggle Runner"], check=True)
    subprocess.run(["git", "-C", str(REPO_DIR), "add", adapter_git_path], check=True)
    subprocess.run(["git", "-C", str(REPO_DIR), "commit", "-m",
                    f"kaggle: Qwen3-8B LoRA adapter for L0 bucket prediction (acc={acc:.1f}%)"], check=True)
    subprocess.run(["git", "-C", str(REPO_DIR), "push",
                    f"https://{token}@github.com/SuryaaSeran/recsys-music-26.git", "main"], check=True)
    print("Pushed adapter to GitHub")
else:
    print("No GITHUB_TOKEN — download adapter from Kaggle output panel")
    print(f"Adapter at: {adapter_dir}")

print(f"\nDone. Top-1 accuracy: {acc:.1f}%")
