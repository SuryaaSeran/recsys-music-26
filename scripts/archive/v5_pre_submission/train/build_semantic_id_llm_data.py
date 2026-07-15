"""Build fine-tuning data for the semantic-ID LLM (L0 bucket predictor).

Four training types:
  A — Single-turn query → L0 bucket        (~73K, core task)
  B — Bucket ID → description              (64 × 2 = 128)
  C — Description / name → Bucket ID      (64 × 2 = 128)
  D — Multi-turn conversation → L0 bucket  (~20K, harder variant of A)

Type D uses up to 4 prior turns with listener thoughts, matching the actual
inference format exactly. Type A uses only the current query + goal.

Output: data/semantic_id_llm/train.jsonl + valid.jsonl (chat format)
Each line: {"messages": [{"role": ..., "content": ...}, ...]}

Eval metric target: >90% bucket recall @ top-3
  = fraction of dev turns where gold track's L0 bucket is in predicted top-3

Usage:
    python scripts/train/build_semantic_id_llm_data.py \
        --sids_dir cache/semantic_ids/runF_v8e_L2C64 \
        --descriptions cache/semantic_ids/runF_v8e_L2C64/bucket_descriptions.json \
        --out_dir data/semantic_id_llm
"""
import argparse
import json
import random
from collections import Counter
from pathlib import Path

import numpy as np
from datasets import load_dataset, concatenate_datasets
from tqdm import tqdm

ap = argparse.ArgumentParser()
ap.add_argument("--sids_dir",      default="cache/semantic_ids/runF_v8e_L2C64")
ap.add_argument("--descriptions",  default="cache/semantic_ids/runF_v8e_L2C64/bucket_descriptions.json")
ap.add_argument("--out_dir",       default="data/semantic_id_llm")
ap.add_argument("--max_chars",     type=int, default=3500,
                help="Approximate max chars per user message (~875 tokens at 4 chars/token)")
ap.add_argument("--valid_frac",    type=float, default=0.05)
ap.add_argument("--seed",          type=int, default=42)
ap.add_argument("--exclude_n",     type=int, default=6000)
ap.add_argument("--exclude_seed",  type=int, default=42)
ap.add_argument("--bc_repeats",    type=int, default=8,
                help="How many times to oversample B/C examples in train mix")
args = ap.parse_args()

random.seed(args.seed)
out_dir = Path(args.out_dir)
out_dir.mkdir(parents=True, exist_ok=True)

# ── Load codebook ─────────────────────────────────────────────────────────────
print("Loading semantic IDs...")
sids_dir = Path(args.sids_dir)
codes = np.load(sids_dir / "semantic_ids.npy")
tids  = np.load(sids_dir / "track_ids.npy", allow_pickle=True).tolist()
tid_to_l0: dict[str, int] = {t: int(c[0]) for t, c in zip(tids, codes)}
print(f"  {len(tid_to_l0):,} tracks mapped to {len(set(tid_to_l0.values()))} L0 buckets")

# ── Load descriptions ─────────────────────────────────────────────────────────
with open(args.descriptions) as f:
    desc_data: dict[str, dict] = json.load(f)
bucket_names = {int(k): v["name"]        for k, v in desc_data.items()}
bucket_descs = {int(k): v["description"] for k, v in desc_data.items()}

# ── Load metadata ─────────────────────────────────────────────────────────────
print("Loading track metadata...")
meta_ds    = load_dataset("talkpl-ai/TalkPlayData-Challenge-Track-Metadata")
all_tracks = concatenate_datasets([meta_ds["all_tracks"], meta_ds["test_tracks"]])
metadata: dict[str, dict] = {row["track_id"]: row for row in all_tracks}

def track_short(tid: str) -> str:
    row = metadata.get(tid, {})
    name   = (row.get("track_name")  or ["?"])[0]
    artist = (row.get("artist_name") or ["?"])[0]
    return f"{name} – {artist}" if name and artist else (name or artist or "?")

# ── System prompts ────────────────────────────────────────────────────────────
SYSTEM_QUERY = (
    "You are a music recommendation assistant. "
    "Given a conversation context, predict the L0 semantic cluster ID (0-63) "
    "that best matches the next recommended track. "
    "Output only the integer cluster ID, nothing else."
)

SYSTEM_BC = (
    "You are a music expert. Each cluster ID (0-63) represents a distinct "
    "musical style, genre, and mood in the TalkPlay catalog."
)

REACTION_LABEL = {"MOVES_TOWARD_GOAL": "liked", "DOES_NOT_MOVE_TOWARD_GOAL": "rejected"}

# ── Type B/C ─────────────────────────────────────────────────────────────────
print("Building Type B/C examples...")
bc_examples = []
for l0 in range(64):
    name = bucket_names.get(l0, f"Cluster {l0}")
    desc = bucket_descs.get(l0, "")
    if not desc:
        continue
    # B1: id → full description
    bc_examples.append({"type": "B", "messages": [
        {"role": "system",    "content": SYSTEM_BC},
        {"role": "user",      "content": f"What does cluster {l0} sound like?"},
        {"role": "assistant", "content": f"{name}. {desc}"},
    ]})
    # B2: id → name only
    bc_examples.append({"type": "B", "messages": [
        {"role": "system",    "content": SYSTEM_BC},
        {"role": "user",      "content": f"Give a short name for cluster {l0}."},
        {"role": "assistant", "content": name},
    ]})
    # C1: description → id
    bc_examples.append({"type": "C", "messages": [
        {"role": "system",    "content": SYSTEM_BC},
        {"role": "user",      "content": f"Which cluster ID best matches this description: {desc}"},
        {"role": "assistant", "content": str(l0)},
    ]})
    # C2: name → id
    bc_examples.append({"type": "C", "messages": [
        {"role": "system",    "content": SYSTEM_BC},
        {"role": "user",      "content": f"Which cluster ID is called '{name}'?"},
        {"role": "assistant", "content": str(l0)},
    ]})

print(f"  B/C: {len(bc_examples)} examples")

# ── Load sessions ─────────────────────────────────────────────────────────────
print("Loading TalkPlay train sessions...")
ds = load_dataset("talkpl-ai/TalkPlayData-Challenge-Dataset", split="train")
sessions = list(ds)
rng = random.Random(args.exclude_seed)
rng.shuffle(sessions)
sessions = sessions[args.exclude_n:]
print(f"  {len(sessions)} sessions (after {args.exclude_n} excluded)")

# ── Type A + D builder ────────────────────────────────────────────────────────
print("Building Type A (single-turn) + D (multi-turn) examples...")
a_examples, d_examples = [], []

for item in tqdm(sessions, desc="sessions"):
    profile_raw = item.get("user_profile", {}) or {}
    profile_parts = [
        profile_raw.get("age_group", "") or "",
        profile_raw.get("country_code", "") or "",
        profile_raw.get("gender", "") or "",
        profile_raw.get("preferred_musical_culture", "") or "",
        profile_raw.get("preferred_language", "") or "",
    ]
    profile_line = "[PROFILE] " + " · ".join(p for p in profile_parts if p)
    goal         = item.get("conversation_goal", {}) or {}
    goal_text    = goal.get("listener_goal", "") or ""
    spec         = goal.get("specificity", "") or ""
    goal_line    = f"[GOAL] {goal_text}" + (f"  ({spec})" if spec else "")

    progress_by_turn = {
        a["turn_number"] - 1: a["goal_progress_assessment"]
        for a in (item.get("goal_progress_assessments") or [])
    }

    turn_data: dict[int, dict] = {}
    for t in item["conversations"]:
        tn   = t["turn_number"]
        slot = turn_data.setdefault(tn, {"user": "", "music": "", "asst": "", "user_thought": ""})
        if t["role"] == "user":
            slot["user"]         = t["content"] or ""
            slot["user_thought"] = t.get("thought") or ""
        elif t["role"] == "music":
            slot["music"] = t["content"] or ""
        elif t["role"] == "assistant":
            slot["asst"] = t["content"] or ""

    history_blocks: list[str] = []

    for tn in sorted(turn_data):
        td      = turn_data[tn]
        gold_tid = td["music"]
        if not gold_tid or gold_tid not in tid_to_l0:
            # Still update history
            if gold_tid:
                rec  = track_short(gold_tid)
                rxn  = REACTION_LABEL.get(progress_by_turn.get(tn), "unknown")
                lt   = td["user_thought"] or ""
                blk  = f"[T{tn}] USER: {td['user']} | REC: {rec} | REACTION: {rxn}"
                if lt:
                    end = lt.find(". ")
                    lt  = lt[:end+1] if 0 < end < 200 else lt[:200]
                    blk += f" | LISTENER: {lt}"
                history_blocks.append(blk)
            continue

        l0 = tid_to_l0[gold_tid]

        # ── Type A: just profile + goal + current query ──────────────────────
        a_user = "\n".join([profile_line, goal_line,
                            f"[NOW] USER: {td['user']}",
                            "Predict the L0 cluster ID:"])
        a_examples.append({"type": "A", "messages": [
            {"role": "system",    "content": SYSTEM_QUERY},
            {"role": "user",      "content": a_user},
            {"role": "assistant", "content": str(l0)},
        ]})

        # ── Type D: full history + current query (last 4 turns) ──────────────
        if history_blocks:
            # Build history greedily within char budget
            budget = args.max_chars - len(profile_line) - len(goal_line) - len(td["user"]) - 50
            added  = []
            for hb in reversed(history_blocks[-6:]):
                if len("\n".join(added)) + len(hb) < budget:
                    added.append(hb)
            added.reverse()

            d_user = "\n".join(
                [profile_line, goal_line]
                + added
                + [f"[NOW] USER: {td['user']}", "Predict the L0 cluster ID:"]
            )
            d_examples.append({"type": "D", "messages": [
                {"role": "system",    "content": SYSTEM_QUERY},
                {"role": "user",      "content": d_user},
                {"role": "assistant", "content": str(l0)},
            ]})

        # Update history
        rec  = track_short(gold_tid)
        rxn  = REACTION_LABEL.get(progress_by_turn.get(tn), "unknown")
        lt   = td["user_thought"] or ""
        blk  = f"[T{tn}] USER: {td['user']} | REC: {rec} | REACTION: {rxn}"
        if lt:
            end = lt.find(". ")
            lt  = lt[:end+1] if 0 < end < 200 else lt[:200]
            blk += f" | LISTENER: {lt}"
        history_blocks.append(blk)

print(f"  Type A: {len(a_examples):,}  Type D: {len(d_examples):,}")

# ── Mix + split ───────────────────────────────────────────────────────────────
# Weight: A(1x) + D(1x) + B/C(8x to compensate for smaller count)
all_examples = a_examples + d_examples + bc_examples * args.bc_repeats
random.shuffle(all_examples)

n_valid = max(int(len(all_examples) * args.valid_frac), 200)
valid   = all_examples[:n_valid]
train   = all_examples[n_valid:]

# Type distribution
train_types = Counter(ex["type"] for ex in train)
valid_types = Counter(ex["type"] for ex in valid)
print(f"\nTrain: {len(train):,}  {dict(train_types)}")
print(f"Valid: {len(valid):,}  {dict(valid_types)}")

# Bucket distribution in A+D train
train_buckets = Counter(
    int(ex["messages"][-1]["content"])
    for ex in train if ex["type"] in ("A", "D")
)
print(f"Bucket coverage in A+D train: {len(train_buckets)}/64")
print(f"  Most common:  {train_buckets.most_common(3)}")
print(f"  Least common: {train_buckets.most_common()[-3:]}")

# ── Write ─────────────────────────────────────────────────────────────────────
# Strip 'type' key before writing (not needed by trainer)
def strip_type(ex):
    return {"messages": ex["messages"]}

with open(out_dir / "train.jsonl", "w") as f:
    for ex in train:
        f.write(json.dumps(strip_type(ex), ensure_ascii=False) + "\n")

with open(out_dir / "valid.jsonl", "w") as f:
    for ex in valid:
        f.write(json.dumps(strip_type(ex), ensure_ascii=False) + "\n")

# Also write a pure A+D valid set for bucket-recall eval
ad_valid = [ex for ex in valid if ex["type"] in ("A", "D")]
with open(out_dir / "valid_query.jsonl", "w") as f:
    for ex in ad_valid:
        # Keep gold label as separate field for eval
        gold = ex["messages"][-1]["content"]
        rec  = {"messages": ex["messages"][:-1], "gold_l0": int(gold)}
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")

with open(out_dir / "bucket_names.json", "w") as f:
    json.dump({str(k): v for k, v in bucket_names.items()}, f, indent=2)

print(f"\nSaved to {out_dir}/")
for fname in ["train.jsonl", "valid.jsonl", "valid_query.jsonl", "bucket_names.json"]:
    p = out_dir / fname
    print(f"  {fname}: {p.stat().st_size / 1e6:.1f} MB")
