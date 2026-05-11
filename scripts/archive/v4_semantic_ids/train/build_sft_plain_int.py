"""
Build SFT data using plain integer format instead of <SID_x> tokens.

Response format: "{c} {f}" where c in [0,127], f in [0,127].
Avoids all tokenizer patching / embedding init complexity.
Output: data/sft_plain_int_v1/
"""

import json
import pickle
import random
from pathlib import Path

from datasets import load_dataset

random.seed(42)

OUT_DIR = Path("data/sft_plain_int_v1")
OUT_DIR.mkdir(parents=True, exist_ok=True)

with open("data/codebook_v2.pkl", "rb") as f:
    cb = pickle.load(f)

track_to_codes = cb["track_to_codes"]

print(f"Codebook source: {cb['source']}")
print(f"Tracks in codebook: {len(track_to_codes):,}")


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
        if track_id not in track_to_codes:
            skipped += 1
            continue

        c, f = track_to_codes[track_id]
        user_msg = prev_user_text(conversations, i)

        prompt = f"""Goal: {goal}
User culture: {culture}
User request: {user_msg}

Return only one integer pair."""

        examples.append({
            "messages": [
                {
                    "role": "system",
                    "content": "You are a music retrieval model. Return only two integers separated by a space: coarse fine (both 0-127). Example: 62 45",
                },
                {
                    "role": "user",
                    "content": prompt,
                },
                {
                    "role": "assistant",
                    "content": f"{c} {f}",
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
