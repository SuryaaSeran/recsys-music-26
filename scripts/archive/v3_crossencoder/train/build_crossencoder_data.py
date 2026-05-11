"""
Build (query, track_text, label) training pairs for cross-encoder.

For each turn:
  positive: (compact_query, gold_track_text, 1)
  negatives: (compact_query, bm25_top_k_non_gold_text, 0)

Output: data/crossencoder_v1/ with train.jsonl and valid.jsonl
Each line: {"query": str, "document": str, "label": int}

Usage:
    python scripts/build_crossencoder_data.py --neg_per_pos 5
"""
import argparse
import json
import random
from pathlib import Path

import bm25s
from datasets import load_dataset, concatenate_datasets
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument("--neg_per_pos", type=int, default=5, help="Negatives per positive example")
parser.add_argument("--bm25_pool", type=int, default=50, help="BM25 candidates to sample negatives from")
parser.add_argument("--valid_frac", type=float, default=0.05)
parser.add_argument("--out_dir", default="data/crossencoder_v1")
parser.add_argument("--seed", type=int, default=42)
args = parser.parse_args()

random.seed(args.seed)
CACHE_PATH = "cache/bm25/track_metadata"

print("Loading track metadata...")
meta_ds = load_dataset("talkpl-ai/TalkPlayData-Challenge-Track-Metadata")
all_tracks = concatenate_datasets([meta_ds["all_tracks"], meta_ds["test_tracks"]])
metadata_dict = {row["track_id"]: row for row in all_tracks}


def get_track_text(tid):
    row = metadata_dict.get(tid, {})
    name = (row.get("track_name") or [""])[0]
    artist = (row.get("artist_name") or [""])[0]
    tags = " ".join(row.get("tag_list") or [])
    return f"{name} {artist} {tags}".strip()


def get_track_name_artist(tid):
    row = metadata_dict.get(tid, {})
    name = (row.get("track_name") or [""])[0]
    artist = (row.get("artist_name") or [""])[0]
    return f"{name} {artist}".strip()


print("Loading BM25 index...")
bm25_model = bm25s.BM25.load(CACHE_PATH, load_corpus=False)
with open(f"{CACHE_PATH}/track_ids.json") as f:
    track_ids_list = json.load(f)


def retrieve_bm25(query, topk):
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


def build_examples(sessions):
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

                latest_user = text_in_history[-1] if text_in_history else ""
                parts = [latest_user, goal, culture]
                for tid in music_in_history[-2:]:
                    na = get_track_name_artist(tid)
                    if na:
                        parts.append(na)
                query = " ".join(p for p in parts if p).strip()
                if not query:
                    music_in_history.append(gold_tid)
                    continue

                # Positive example
                examples.append({"query": query, "document": gold_text, "label": 1})

                # Hard negatives: BM25 top-K excluding gold
                pool_k = args.bm25_pool + 1
                bm25_results = retrieve_bm25(query, topk=pool_k)
                seen = set(music_in_history) | {gold_tid}
                neg_pool = [t for t in bm25_results if t not in seen]

                for tid in neg_pool[:args.neg_per_pos]:
                    neg_text = get_track_text(tid)
                    if neg_text.strip():
                        examples.append({"query": query, "document": neg_text, "label": 0})

                music_in_history.append(gold_tid)

            elif turn["role"] in ("user", "assistant"):
                text_in_history.append(turn["content"])

    return examples


train_examples = build_examples(train_sessions)
valid_examples = build_examples(valid_sessions)

print(f"Train examples: {len(train_examples):,} (positives + negatives)")
print(f"Valid examples: {len(valid_examples):,}")

out_dir = Path(args.out_dir)
out_dir.mkdir(parents=True, exist_ok=True)

for name, examples in [("train", train_examples), ("valid", valid_examples)]:
    path = out_dir / f"{name}.jsonl"
    with open(path, "w") as f:
        for ex in examples:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")
    print(f"Saved {len(examples):,} examples to {path}")

# Stats
n_pos = sum(1 for e in train_examples if e["label"] == 1)
n_neg = sum(1 for e in train_examples if e["label"] == 0)
print(f"\nTrain: {n_pos:,} positives, {n_neg:,} negatives ({n_neg/n_pos:.1f}:1 ratio)")
