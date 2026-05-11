"""
Build two-tower v6 training dataset.

Improvements over v3:
  - Richer anchor: latest user turn + goal category/specificity + culture + age_group +
    country + last 2 track name/artist
  - Richer track text: name | artist | album | top-12 tags | release year
  - Progress-weighted positives: MOVES_TOWARD_GOAL=1.0, None=1.0,
    DOES_NOT_MOVE_TOWARD_GOAL=0.4
  - Mixed negatives: 2 random + 2 BM25 top-100 + 1 rejected track from history
    (a track rated DOES_NOT_MOVE_TOWARD_GOAL in a prior turn)

Output: data/twotower_v6/ with train.jsonl and valid.jsonl

Each row has: anchor, positive, weight, negative_1..5

Usage:
    python scripts/train/build_twotower_v6_data.py
    python scripts/train/build_twotower_v6_data.py --out_dir data/twotower_v6 --hard_negs 5
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
parser.add_argument("--bm25_pool", type=int, default=100)
parser.add_argument("--valid_frac", type=float, default=0.05)
parser.add_argument("--out_dir", default="data/twotower_v6")
parser.add_argument("--seed", type=int, default=42)
args = parser.parse_args()

random.seed(args.seed)
CACHE_PATH = "cache/bm25/track_metadata"

PROGRESS_WEIGHT = {
    "MOVES_TOWARD_GOAL": 1.0,
    "DOES_NOT_MOVE_TOWARD_GOAL": 0.4,
    None: 1.0,
}

print("Loading track metadata...")
meta_ds = load_dataset("talkpl-ai/TalkPlayData-Challenge-Track-Metadata")
all_tracks = concatenate_datasets([meta_ds["all_tracks"], meta_ds["test_tracks"]])
metadata_dict = {row["track_id"]: row for row in all_tracks}
all_track_ids = list(metadata_dict.keys())


def get_track_text(tid: str) -> str:
    row = metadata_dict.get(tid, {})
    name = (row.get("track_name") or [""])[0]
    artist = (row.get("artist_name") or [""])[0]
    album = (row.get("album_name") or [""])[0]
    tags = " ".join((row.get("tag_list") or [])[:12])
    release_date = row.get("release_date") or ""
    year = str(release_date)[:4] if release_date else ""
    parts = [name]
    if artist:
        parts.append(f"by {artist}")
    if album:
        parts.append(f"| Album: {album}")
    if tags:
        parts.append(f"| Tags: {tags}")
    if year:
        parts.append(f"| {year}")
    return " ".join(parts).strip()


def get_track_name_artist(tid: str) -> str:
    row = metadata_dict.get(tid, {})
    name = (row.get("track_name") or [""])[0]
    artist = (row.get("artist_name") or [""])[0]
    return f"{name} {artist}".strip()


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
        goal = item.get("conversation_goal", {}) or {}
        listener_goal = goal.get("listener_goal", "") or ""
        category = goal.get("category", "") or ""
        specificity = goal.get("specificity", "") or ""

        profile = item.get("user_profile", {}) or {}
        culture = profile.get("preferred_musical_culture", "") or ""
        age_group = profile.get("age_group", "") or ""
        country = profile.get("country_name", "") or ""

        # Build turn-number -> progress label lookup
        progress_by_turn = {
            a["turn_number"]: a["goal_progress_assessment"]
            for a in (item.get("goal_progress_assessments") or [])
        }

        conversations = item["conversations"]

        music_in_history = []   # list of (tid, turn_number)
        text_in_history = []    # list of user/assistant message strings
        rejected_tids = set()   # tracks rated DOES_NOT_MOVE at a prior turn

        for turn in conversations:
            role = turn["role"]
            turn_num = turn["turn_number"]

            if role == "music":
                gold_tid = turn["content"]
                gold_text = get_track_text(gold_tid)
                if not gold_text.strip():
                    music_in_history.append((gold_tid, turn_num))
                    continue

                progress = progress_by_turn.get(turn_num)
                weight = PROGRESS_WEIGHT[progress]

                # Compact anchor: latest user request + goal info + profile + last 2 tracks
                latest_user = text_in_history[-1] if text_in_history else ""
                parts = [latest_user]
                if listener_goal:
                    parts.append(f"Goal: {listener_goal}")
                if category or specificity:
                    parts.append(f"Type: {category} {specificity}".strip())
                if culture:
                    parts.append(culture)
                if age_group or country:
                    parts.append(f"{age_group} {country}".strip())
                for tid, _ in music_in_history[-2:]:
                    na = get_track_name_artist(tid)
                    if na:
                        parts.append(na)
                anchor = " ".join(p for p in parts if p).strip()
                if not anchor:
                    music_in_history.append((gold_tid, turn_num))
                    continue

                # Mixed negatives
                seen = set(t for t, _ in music_in_history) | {gold_tid}
                negatives = []

                # 1. BM25 top-100 negatives (up to 2)
                bm25_results = retrieve_bm25(anchor, topk=args.bm25_pool + 1)
                bm25_negs = [t for t in bm25_results if t not in seen]
                for tid in bm25_negs[:2]:
                    txt = get_track_text(tid)
                    if txt.strip():
                        negatives.append(txt)

                # 2. Rejected track from history (up to 1)
                rejected_pool = [t for t in rejected_tids if t not in seen]
                if rejected_pool:
                    tid = random.choice(rejected_pool)
                    txt = get_track_text(tid)
                    if txt.strip():
                        negatives.append(txt)

                # 3. Random negatives (fill remaining slots up to hard_negs)
                all_neg_pool = [t for t in all_track_ids if t not in seen]
                random.shuffle(all_neg_pool)
                for tid in all_neg_pool:
                    if len(negatives) >= args.hard_negs:
                        break
                    txt = get_track_text(tid)
                    if txt.strip():
                        negatives.append(txt)

                ex = {"anchor": anchor, "positive": gold_text, "weight": weight}
                for i, neg in enumerate(negatives[:args.hard_negs]):
                    ex[f"negative_{i+1}"] = neg
                examples.append(ex)

                music_in_history.append((gold_tid, turn_num))
                if progress == "DOES_NOT_MOVE_TOWARD_GOAL":
                    rejected_tids.add(gold_tid)

            elif role in ("user", "assistant"):
                text_in_history.append(turn["content"])

    return examples


train_examples = build_examples(train_sessions)
valid_examples = build_examples(valid_sessions)

print(f"Train examples: {len(train_examples):,}")
print(f"Valid examples: {len(valid_examples):,}")

# Stats
weights = [e["weight"] for e in train_examples]
w_counts = {1.0: weights.count(1.0), 0.4: weights.count(0.4)}
print(f"Weight distribution: {w_counts}")

def apply_weights(examples: list[dict]) -> list[dict]:
    """
    MNRL doesn't support per-sample weights, so we approximate by:
    - keeping weight=1.0 examples as-is
    - randomly dropping ~60% of weight=0.4 examples
    """
    out = []
    for ex in examples:
        w = ex.get("weight", 1.0)
        if w >= 1.0 or random.random() < w:
            out.append(ex)
    return out


train_examples = apply_weights(train_examples)
print(f"After weight filtering: {len(train_examples):,} train examples")

out_dir = Path(args.out_dir)
out_dir.mkdir(parents=True, exist_ok=True)

for name, exs in [("train", train_examples), ("valid", valid_examples)]:
    path = out_dir / f"{name}.jsonl"
    with open(path, "w") as f:
        for ex in exs:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")
    print(f"Saved {len(exs):,} examples to {path}")

print("\nSample:")
ex = train_examples[0]
print(f"  anchor: {ex['anchor'][:150]}")
print(f"  positive: {ex['positive'][:100]}")
print(f"  weight: {ex['weight']}")
if "negative_1" in ex:
    print(f"  negative_1: {ex['negative_1'][:100]}")
