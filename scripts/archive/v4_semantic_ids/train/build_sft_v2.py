"""
Build SFT data using codebook_v2.pkl (multimodal multilingual codebook).

Same prompt structure as build_sid_only_short_sft_data.py.
Output: data/sft_sid_only_short_v2/
"""

import json
import pickle
import random
from pathlib import Path

from datasets import load_dataset

random.seed(42)

OUT_DIR = Path("data/sft_sid_only_short_v2")
OUT_DIR.mkdir(parents=True, exist_ok=True)

with open("data/codebook_v2.pkl", "rb") as f:
    cb = pickle.load(f)

n_coarse = cb["n_coarse"]
track_to_codes = cb["track_to_codes"]

print(f"Codebook source: {cb['source']}")
print(f"Tracks in codebook: {len(track_to_codes):,}")


def sid_pair(track_id):
    if track_id not in track_to_codes:
        return None
    c, f = track_to_codes[track_id]
    return f"<SID_{c}> <SID_{n_coarse + f}>"


def prev_user_text(conversations, idx):
    for j in range(idx - 1, -1, -1):
        if conversations[j].get("role") == "user":
            return conversations[j].get("content", "")
    return ""


ds = load_dataset("talkpl-ai/TalkPlayData-Challenge-Dataset")["train"]

examples = []
skipped = 0

for row in ds:
    goal = row.get("conversation_goal", {}).get("listener_goal", "")
    culture = row.get("user_profile", {}).get("preferred_musical_culture", "")
    conversations = row["conversations"]

    for i, turn in enumerate(conversations):
        if turn.get("role") != "music":
            continue

        track_id = str(turn.get("content", "")).strip()
        sid = sid_pair(track_id)
        if not sid:
            skipped += 1
            continue

        user_msg = prev_user_text(conversations, i)

        prompt = f"""Goal: {goal}
User culture: {culture}
User request: {user_msg}

Return only one semantic ID pair."""

        examples.append({
            "messages": [
                {
                    "role": "system",
                    "content": "You are a music retrieval model. Return only one semantic ID pair like <SID_12> <SID_130>.",
                },
                {
                    "role": "user",
                    "content": prompt,
                },
                {
                    "role": "assistant",
                    "content": sid,
                },
            ]
        })

print(f"Built examples: {len(examples):,}  (skipped: {skipped:,} not in codebook)")

random.shuffle(examples)
n = len(examples)
train = examples[: int(0.9 * n)]
valid = examples[int(0.9 * n):]

for name, split in [("train", train), ("valid", valid)]:
    with open(OUT_DIR / f"{name}.jsonl", "w") as f:
        for ex in split:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")

print(f"Wrote {len(train):,} train and {len(valid):,} valid to {OUT_DIR}")
