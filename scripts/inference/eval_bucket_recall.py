"""Evaluate L0 bucket recall on dev turns.

Measures two things:
  1. Bucket accuracy: fraction of dev turns where predicted L0 == gold L0
  2. Bucket recall @ k: fraction of dev turns where gold L0 is in top-k predicted buckets
     (target: >90% @ k=3 using the LLM predictor)

Baseline (SASRec):  uses SemanticIDRetriever to get predicted buckets per turn
LLM (fine-tuned):   uses sid_qwen3 model to predict L0 from conversation context

Usage:
  # Baseline (SASRec):
  python scripts/inference/eval_bucket_recall.py \
    --mode sasrec \
    --sids_dir cache/semantic_ids/runF_v8e_L2C64 \
    --sasrec_ckpt models/sasrec/sasrec_runF_v8e_L2C64/best_model.pth \
    --top_k 3

  # LLM:
  python scripts/inference/eval_bucket_recall.py \
    --mode llm \
    --sids_dir cache/semantic_ids/runF_v8e_L2C64 \
    --llm_model models/sid_qwen3/final \
    --eval_file data/semantic_id_llm/valid_query.jsonl \
    --top_k 3
"""
import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from datasets import load_dataset
from tqdm import tqdm

ap = argparse.ArgumentParser()
ap.add_argument("--mode",       choices=["sasrec", "llm"], required=True)
ap.add_argument("--sids_dir",   default="cache/semantic_ids/runF_v8e_L2C64")
ap.add_argument("--top_k",      type=int, default=3)
ap.add_argument("--n_sessions", type=int, default=1000,
                help="Dev sessions to evaluate (mode=sasrec only)")
# SASRec args
ap.add_argument("--sasrec_ckpt", default="models/sasrec/sasrec_runF_v8e_L2C64/best_model.pth")
# LLM args
ap.add_argument("--llm_model",  default="models/sid_qwen3/final")
ap.add_argument("--eval_file",  default="data/semantic_id_llm/valid_query.jsonl")
ap.add_argument("--batch_size", type=int, default=8)
args = ap.parse_args()

# ── Load codebook ─────────────────────────────────────────────────────────────
sids_dir = Path(args.sids_dir)
codes  = np.load(sids_dir / "semantic_ids.npy")
tids   = np.load(sids_dir / "track_ids.npy", allow_pickle=True).tolist()
tid_to_l0 = {t: int(c[0]) for t, c in zip(tids, codes)}
l0_to_tids = defaultdict(list)
for t, c in zip(tids, codes):
    l0_to_tids[int(c[0])].append(t)
print(f"Codebook: {len(tid_to_l0):,} tracks, {len(l0_to_tids)} L0 buckets")


# ─────────────────────────────────────────────────────────────────────────────
# Mode: SASRec
# ─────────────────────────────────────────────────────────────────────────────
if args.mode == "sasrec":
    from scripts.inference.semantic_id_retrieval import SemanticIDRetriever

    retriever = SemanticIDRetriever(
        sasrec_ckpt=args.sasrec_ckpt,
        sids_dir=str(sids_dir),
    )

    print(f"Loading {args.n_sessions} dev sessions...")
    ds = load_dataset("talkpl-ai/TalkPlayData-Challenge-Dataset", split="test")
    sessions = list(ds)[:args.n_sessions]

    REACTION_LABEL = {"MOVES_TOWARD_GOAL": "liked", "DOES_NOT_MOVE_TOWARD_GOAL": "rejected"}
    correct, recall_k, total = 0, 0, 0
    per_bucket_correct = defaultdict(int)
    per_bucket_total   = defaultdict(int)

    for item in tqdm(sessions, desc="sessions"):
        progress_by_turn = {
            a["turn_number"] - 1: a["goal_progress_assessment"]
            for a in (item.get("goal_progress_assessments") or [])
        }
        turn_data: dict[int, dict] = {}
        for t in item["conversations"]:
            tn   = t["turn_number"]
            slot = turn_data.setdefault(tn, {"user": "", "music": ""})
            if t["role"] == "user":  slot["user"]  = t["content"] or ""
            elif t["role"] == "music": slot["music"] = t["content"] or ""

        music_history, music_labels = [], []
        for tn in sorted(turn_data):
            td = turn_data[tn]
            gold_tid = td["music"]
            if not gold_tid or gold_tid not in tid_to_l0:
                if gold_tid:
                    music_history.append(gold_tid)
                    music_labels.append(progress_by_turn.get(tn, ""))
                continue

            gold_l0 = tid_to_l0[gold_tid]

            if music_history:
                # MOVES-only filter
                moves = [t for t, l in zip(music_history, music_labels)
                         if l == "MOVES_TOWARD_GOAL"] or music_history
                _, meta = retriever.expand(moves, top_k_l0=args.top_k, history_labels=music_labels)
                # Predicted top-k buckets are the unique l0 ranks in meta values
                pred_buckets_ranked = []
                seen = set()
                for tid2, (rank, _) in sorted(meta.items(), key=lambda x: x[1][0]):
                    b = tid_to_l0.get(tid2)
                    if b is not None and b not in seen:
                        pred_buckets_ranked.append(b)
                        seen.add(b)
                # Also just use SASRec's predict_l0_distribution directly
                dist = retriever.predict_l0_distribution(moves)
                top_k_buckets = [b for b, _ in dist[:args.top_k]]
            else:
                top_k_buckets = []

            if top_k_buckets:
                if top_k_buckets[0] == gold_l0:   correct   += 1
                if gold_l0 in top_k_buckets:       recall_k  += 1
            per_bucket_total[gold_l0]   += 1
            if top_k_buckets and gold_l0 in top_k_buckets:
                per_bucket_correct[gold_l0] += 1
            total += 1

            music_history.append(gold_tid)
            music_labels.append(progress_by_turn.get(tn, ""))

    print(f"\n=== SASRec Bucket Recall (top-{args.top_k}) ===")
    print(f"  Top-1 accuracy: {correct}/{total} = {100*correct/total:.1f}%")
    print(f"  Recall @ {args.top_k}:  {recall_k}/{total} = {100*recall_k/total:.1f}%")
    print(f"  (Target: >90% recall @ {args.top_k})")
    worst = sorted(per_bucket_total, key=lambda b: per_bucket_correct.get(b, 0)/per_bucket_total[b])[:5]
    print(f"\n  Worst-5 buckets by recall@{args.top_k}:")
    for b in worst:
        r = 100 * per_bucket_correct.get(b, 0) / per_bucket_total[b]
        print(f"    bucket {b:>2}: {r:.1f}%  ({per_bucket_total[b]} turns)")


# ─────────────────────────────────────────────────────────────────────────────
# Mode: LLM
# ─────────────────────────────────────────────────────────────────────────────
elif args.mode == "llm":
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print(f"Loading LLM: {args.llm_model}")
    device = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(args.llm_model, padding_side="left")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.llm_model,
        torch_dtype=torch.float32 if device == "mps" else torch.bfloat16,
        device_map={"": device},
    )
    model.eval()

    print(f"Loading eval file: {args.eval_file}")
    eval_data = [json.loads(l) for l in Path(args.eval_file).read_text().splitlines() if l.strip()]
    print(f"  {len(eval_data)} eval examples")

    correct, recall_k, total = 0, 0, 0
    per_bucket_correct = defaultdict(int)
    per_bucket_total   = defaultdict(int)

    def get_top_k_preds(prompt_text: str, k: int) -> list[int]:
        """Generate top-k bucket predictions using greedy + fallback."""
        ids = tokenizer(prompt_text, return_tensors="pt",
                        max_length=1024, truncation=True).to(device)
        with torch.no_grad():
            # Get logits for next token to find top-k bucket predictions
            out = model(**ids)
            logits = out.logits[0, -1, :]  # last token logits

        # Decode which token IDs correspond to digits 0-63
        preds = []
        for b in range(64):
            tok_ids = tokenizer.encode(str(b), add_special_tokens=False)
            if tok_ids:
                score = logits[tok_ids[0]].item()
                preds.append((b, score))
        preds.sort(key=lambda x: -x[1])
        return [b for b, _ in preds[:k]]

    for ex in tqdm(eval_data, desc="eval"):
        gold_l0 = ex["gold_l0"]
        messages = ex["messages"]
        prompt   = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True)
        top_k = get_top_k_preds(prompt, args.top_k)

        if top_k and top_k[0] == gold_l0: correct  += 1
        if gold_l0 in top_k:              recall_k += 1
        per_bucket_total[gold_l0]   += 1
        if gold_l0 in top_k:
            per_bucket_correct[gold_l0] += 1
        total += 1

    print(f"\n=== LLM Bucket Recall (top-{args.top_k}) ===")
    print(f"  Top-1 accuracy: {correct}/{total} = {100*correct/total:.1f}%")
    print(f"  Recall @ {args.top_k}:  {recall_k}/{total} = {100*recall_k/total:.1f}%")
    print(f"  (Target: >90% recall @ {args.top_k})")
    worst = sorted(per_bucket_total, key=lambda b: per_bucket_correct.get(b, 0)/per_bucket_total[b])[:5]
    print(f"\n  Worst-5 buckets by recall@{args.top_k}:")
    for b in worst:
        r = 100 * per_bucket_correct.get(b, 0) / per_bucket_total[b]
        print(f"    bucket {b:>2}: {r:.1f}%  ({per_bucket_total[b]} turns)")
