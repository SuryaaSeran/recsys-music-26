"""
Build balanced SFT data: cap examples per coarse code to reduce mode collapse.
Output: data/sft_balanced_v1/
"""
import json
import pickle
import random
from collections import defaultdict
from pathlib import Path

from datasets import load_dataset

random.seed(42)

OUT_DIR = Path("data/sft_balanced_v1")
OUT_DIR.mkdir(parents=True, exist_ok=True)

with open("data/codebook_v2.pkl", "rb") as f:
    cb = pickle.load(f)

track_to_codes = cb["track_to_codes"]
n_coarse = cb["n_coarse"]

CAP_PER_COARSE = 800  # 128 * 800 = 102K max, but we have 109K so some trimming


def prev_user_text(conversations, idx):
    for j in range(idx - 1, -1, -1):
        if conversations[j].get("role") == "user":
            return conversations[j].get("content", "")
    return ""


ds = load_dataset("talkpl-ai/TalkPlayData-Challenge-Dataset")["train"]

# Collect all examples grouped by coarse code
by_coarse = defaultdict(list)
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

        ex = {
            "messages": [
                {
                    "role": "system",
                    "content": "You are a music retrieval model. Return only two integers separated by a space: coarse fine (both 0-127). Example: 62 45",
                },
                {"role": "user", "content": prompt},
                {"role": "assistant", "content": f"{c} {f}"},
            ]
        }
        by_coarse[c].append(ex)

print(f"Before cap: {sum(len(v) for v in by_coarse.values()):,} examples, {skipped} skipped")
print(f"Coarse codes represented: {len(by_coarse)}")

# Cap per coarse code
examples = []
for c, exs in by_coarse.items():
    random.shuffle(exs)
    examples.extend(exs[:CAP_PER_COARSE])

print(f"After cap ({CAP_PER_COARSE}/coarse): {len(examples):,} examples")

random.shuffle(examples)
n = len(examples)
train = examples[: int(0.9 * n)]
valid = examples[int(0.9 * n):]

for name, split in [("train", train), ("valid", valid)]:
    with open(OUT_DIR / f"{name}.jsonl", "w") as f:
        for ex in split:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")

print(f"Wrote {len(train):,} train / {len(valid):,} valid to {OUT_DIR}")
