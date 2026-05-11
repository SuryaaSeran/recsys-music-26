"""
Build two-tower training dataset.

For each music turn in the train split:
  anchor   = conversation context (goal + culture + prev track metadata + text history + user query)
  positive = gold track text (name + artist + tags)
  negatives = BM25 top-K non-gold tracks (hard negatives)

Output: data/twotower/ with train.jsonl and valid.jsonl

Usage:
    python scripts/build_twotower_data.py --hard_negs 5
"""
import argparse
import json
import random
from pathlib import Path

import bm25s
from datasets import load_dataset, concatenate_datasets
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument("--hard_negs", type=int, default=5)
parser.add_argument("--bm25_pool", type=int, default=100, help="BM25 candidates to sample negatives from")
parser.add_argument("--valid_frac", type=float, default=0.05)
parser.add_argument("--out_dir", default="data/twotower")
parser.add_argument("--seed", type=int, default=42)
args = parser.parse_args()

random.seed(args.seed)
CACHE_PATH = "cache/bm25/track_metadata"

print("Loading track metadata...")
meta_ds = load_dataset("talkpl-ai/TalkPlayData-Challenge-Track-Metadata")
all_tracks = concatenate_datasets([meta_ds["all_tracks"], meta_ds["test_tracks"]])
metadata_dict = {row["track_id"]: row for row in all_tracks}


def get_track_text(tid: str) -> str:
    row = metadata_dict.get(tid, {})
    name = (row.get("track_name") or [""])[0]
    artist = (row.get("artist_name") or [""])[0]
    tags = " ".join(row.get("tag_list") or [])
    return f"{name} {artist} {tags}".strip()


print("Loading BM25 index...")
bm25_model = bm25s.BM25.load(CACHE_PATH, load_corpus=False)
with open(f"{CACHE_PATH}/track_ids.json") as f:
    track_ids_list = json.load(f)


def retrieve_bm25(query: str, topk: int) -> list[str]:
    tokens = bm25s.tokenize([query.lower()])
    results = bm25_model.retrieve(tokens, k=topk, return_as="tuple")
    return [track_ids_list[int(i)] for i in results.documents[0]]


print("Loading train conversations...")
ds = load_dataset("talkpl-ai/TalkPlayData-Challenge-Dataset")
sessions = list(ds["train"])
random.shuffle(sessions)

n_valid = int(len(sessions) * args.valid_frac)
valid_sessions = sessions[:n_valid]
train_sessions = sessions[n_valid:]

print(f"Train sessions: {len(train_sessions)}, Valid sessions: {len(valid_sessions)}")


def build_examples(sessions: list) -> list[dict]:
    examples = []
    for item in tqdm(sessions, desc="Building pairs"):
        goal = item.get("conversation_goal", {}).get("listener_goal", "")
        culture = item.get("user_profile", {}).get("preferred_musical_culture", "")
        conversations = item["conversations"]

        music_in_history = []
        text_in_history = []

        for turn in conversations:
            if turn["role"] == "music":
                gold_tid = turn["content"]
                gold_text = get_track_text(gold_tid)
                if not gold_text.strip():
                    music_in_history.append(gold_tid)
                    continue

                # Compact query: most important info first so it survives 256-token truncation.
                # latest user request + goal + culture + last 2 track name/artist (no tags).
                latest_user = text_in_history[-1] if text_in_history else ""
                parts = [latest_user, goal, culture]
                for tid in music_in_history[-2:]:
                    row = metadata_dict.get(tid, {})
                    name = (row.get("track_name") or [""])[0]
                    artist = (row.get("artist_name") or [""])[0]
                    if name or artist:
                        parts.append(f"{name} {artist}".strip())
                anchor = " ".join(p for p in parts if p).strip()
                if not anchor:
                    music_in_history.append(gold_tid)
                    continue

                # Hard negatives: BM25 top candidates excluding gold
                pool_k = args.bm25_pool + 1
                bm25_results = retrieve_bm25(anchor, topk=pool_k)
                seen = set(music_in_history) | {gold_tid}
                hard_neg_pool = [t for t in bm25_results if t not in seen]

                negatives = []
                for tid in hard_neg_pool[:args.hard_negs]:
                    neg_text = get_track_text(tid)
                    if neg_text.strip():
                        negatives.append(neg_text)

                ex = {"anchor": anchor, "positive": gold_text}
                for i, neg in enumerate(negatives):
                    ex[f"negative_{i+1}"] = neg
                examples.append(ex)

                music_in_history.append(gold_tid)

            elif turn["role"] in ("user", "assistant"):
                text_in_history.append(turn["content"])

    return examples


train_examples = build_examples(train_sessions)
valid_examples = build_examples(valid_sessions)

print(f"Train examples: {len(train_examples)}")
print(f"Valid examples: {len(valid_examples)}")

out_dir = Path(args.out_dir)
out_dir.mkdir(parents=True, exist_ok=True)

for name, examples in [("train", train_examples), ("valid", valid_examples)]:
    path = out_dir / f"{name}.jsonl"
    with open(path, "w") as f:
        for ex in examples:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")
    print(f"Saved {len(examples):,} examples to {path}")

# Print sample
print("\nSample train example:")
ex = train_examples[0]
print(f"  anchor[:120]: {ex['anchor'][:120]}")
print(f"  positive[:80]: {ex['positive'][:80]}")
if "negative_1" in ex:
    print(f"  negative_1[:80]: {ex['negative_1'][:80]}")
