"""Build two-tower v7 training dataset.

Differences from v6:
  - Loops over ALL 15,199 TRAIN sessions (v6 used 1000).
  - BM25 hard-neg pool depth 200 (v6 used 100).
  - Adds a cold-track metadata stream: one self-supervised pair per catalog
    track (47K), so the model sees every track at least once as a positive.
  - Bakes the Qwen3 query instruction prefix into the anchor field; the
    track side stays prefix-free. Cold-track pairs use no prefix on either
    side (both sides are metadata text).
  - Optional --exclude_seed/--exclude_n to hold out LTR-feature-dump
    sessions so a later LTR booster stays leak-free.

Output: data/twotower_v7/{train,valid,cold}.jsonl
Each row: anchor, positive, weight, negative_1..N (negs only for session stream).

Usage:
    python scripts/train/build_twotower_v7_data.py
    python scripts/train/build_twotower_v7_data.py --max_turns_per_session 4
"""
import argparse
import json
import random
from pathlib import Path

import bm25s
from datasets import load_dataset, concatenate_datasets
from tqdm import tqdm

QWEN_QUERY_PREFIX = (
    "Instruct: Given a music conversation, retrieve the track that best fits.\n"
    "Query: "
)

parser = argparse.ArgumentParser()
parser.add_argument("--hard_negs", type=int, default=5)
parser.add_argument("--bm25_pool", type=int, default=200)
parser.add_argument("--valid_frac", type=float, default=0.01)
parser.add_argument("--out_dir", default="data/twotower_v7")
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--max_turns_per_session", type=int, default=0,
                    help="If >0, sample up to this many music turns per session.")
parser.add_argument("--exclude_seed", type=int, default=-1,
                    help="If >=0, shuffle TRAIN with this seed and skip the first --exclude_n sessions.")
parser.add_argument("--exclude_n", type=int, default=0)
parser.add_argument("--no_cold_stream", action="store_true",
                    help="Disable the cold-track metadata stream.")
parser.add_argument("--no_prefix", action="store_true",
                    help="Skip the Qwen3 query prefix in anchors (use for non-Qwen training).")
args = parser.parse_args()

random.seed(args.seed)
CACHE_PATH = "cache/bm25/track_metadata"
PREFIX = "" if args.no_prefix else QWEN_QUERY_PREFIX

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


def build_cold_pair(tid: str):
    """Return (anchor, positive) for the cold-track stream, or None if the
    metadata is too sparse to split into two informative halves."""
    row = metadata_dict.get(tid, {})
    name = (row.get("track_name") or [""])[0]
    artist = (row.get("artist_name") or [""])[0]
    album = (row.get("album_name") or [""])[0]
    tags = (row.get("tag_list") or [])
    release_date = row.get("release_date") or ""
    year = str(release_date)[:4] if release_date else ""
    if not name:
        return None
    side_a = name
    if artist:
        side_a += f" by {artist}"
    if album:
        side_a += f" | Album: {album}"
    if tags:
        side_b = " ".join(tags[:12])
        if year:
            side_b += f" | {year}"
        return side_a.strip(), side_b.strip()
    if year and artist:
        return name, f"{artist} {year}".strip()
    if artist:
        return name, artist
    return None


print("Loading BM25 index...")
bm25_model = bm25s.BM25.load(CACHE_PATH, load_corpus=False)
with open(f"{CACHE_PATH}/track_ids.json") as f:
    bm25_track_ids = json.load(f)


def retrieve_bm25(query: str, topk: int) -> list[str]:
    tokens = bm25s.tokenize([query.lower()])
    results = bm25_model.retrieve(tokens, k=topk, return_as="tuple")
    return [bm25_track_ids[int(i)] for i in results.documents[0]]


print("Loading TRAIN conversations...")
ds = load_dataset("talkpl-ai/TalkPlayData-Challenge-Dataset", split="train")
sessions = list(ds)
if args.exclude_seed >= 0 and args.exclude_n > 0:
    rng = random.Random(args.exclude_seed)
    rng.shuffle(sessions)
    held_out = sessions[: args.exclude_n]
    sessions = sessions[args.exclude_n :]
    print(f"Holding out {len(held_out)} sessions (seed {args.exclude_seed}); "
          f"using {len(sessions)} for v7 training.")

random.shuffle(sessions)
n_valid = int(len(sessions) * args.valid_frac)
valid_sessions = sessions[:n_valid]
train_sessions = sessions[n_valid:]
print(f"Train sessions: {len(train_sessions)}, Valid sessions: {len(valid_sessions)}")


def build_examples(sessions: list, desc: str) -> list[dict]:
    examples = []
    for item in tqdm(sessions, desc=desc):
        goal = item.get("conversation_goal", {}) or {}
        listener_goal = goal.get("listener_goal", "") or ""
        category = goal.get("category", "") or ""
        specificity = goal.get("specificity", "") or ""

        profile = item.get("user_profile", {}) or {}
        culture = profile.get("preferred_musical_culture", "") or ""
        age_group = profile.get("age_group", "") or ""
        country = profile.get("country_name", "") or ""

        progress_by_turn = {
            a["turn_number"]: a["goal_progress_assessment"]
            for a in (item.get("goal_progress_assessments") or [])
        }

        conversations = item["conversations"]
        music_in_history = []
        text_in_history = []
        rejected_tids = set()
        per_session_pairs = []

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
                anchor_body = " ".join(p for p in parts if p).strip()
                if not anchor_body:
                    music_in_history.append((gold_tid, turn_num))
                    continue

                seen = set(t for t, _ in music_in_history) | {gold_tid}
                negatives = []

                bm25_results = retrieve_bm25(anchor_body, topk=args.bm25_pool + 1)
                bm25_negs = [t for t in bm25_results if t not in seen]
                for tid in bm25_negs[:2]:
                    txt = get_track_text(tid)
                    if txt.strip():
                        negatives.append(txt)

                rejected_pool = [t for t in rejected_tids if t not in seen]
                if rejected_pool:
                    tid = random.choice(rejected_pool)
                    txt = get_track_text(tid)
                    if txt.strip():
                        negatives.append(txt)

                all_neg_pool = [t for t in all_track_ids if t not in seen]
                random.shuffle(all_neg_pool)
                for tid in all_neg_pool:
                    if len(negatives) >= args.hard_negs:
                        break
                    txt = get_track_text(tid)
                    if txt.strip():
                        negatives.append(txt)

                ex = {
                    "anchor":   PREFIX + anchor_body,
                    "positive": gold_text,
                    "weight":   weight,
                }
                for i, neg in enumerate(negatives[: args.hard_negs]):
                    ex[f"negative_{i+1}"] = neg
                per_session_pairs.append(ex)

                music_in_history.append((gold_tid, turn_num))
                if progress == "DOES_NOT_MOVE_TOWARD_GOAL":
                    rejected_tids.add(gold_tid)

            elif role in ("user", "assistant"):
                text_in_history.append(turn["content"])

        if args.max_turns_per_session > 0 and len(per_session_pairs) > args.max_turns_per_session:
            per_session_pairs = random.sample(per_session_pairs, args.max_turns_per_session)
        examples.extend(per_session_pairs)
    return examples


print("Building session pairs (TRAIN)...")
train_examples = build_examples(train_sessions, "train")
print("Building session pairs (VALID)...")
valid_examples = build_examples(valid_sessions, "valid")

print(f"Train session pairs: {len(train_examples):,}")
print(f"Valid session pairs: {len(valid_examples):,}")

weights = [e["weight"] for e in train_examples]
w_counts = {1.0: weights.count(1.0), 0.4: weights.count(0.4)}
print(f"Weight distribution: {w_counts}")


def apply_weights(examples: list[dict]) -> list[dict]:
    out = []
    for ex in examples:
        w = ex.get("weight", 1.0)
        if w >= 1.0 or random.random() < w:
            out.append(ex)
    return out


train_examples = apply_weights(train_examples)
print(f"After weight filter: {len(train_examples):,} train")


cold_examples = []
if not args.no_cold_stream:
    print("Building cold-track metadata stream...")
    for tid in tqdm(all_track_ids, desc="cold"):
        pair = build_cold_pair(tid)
        if pair is None:
            continue
        a, p = pair
        cold_examples.append({"anchor": a, "positive": p, "weight": 1.0})
    print(f"Cold-track pairs: {len(cold_examples):,}")


out_dir = Path(args.out_dir)
out_dir.mkdir(parents=True, exist_ok=True)

for name, exs in [("train", train_examples), ("valid", valid_examples), ("cold", cold_examples)]:
    if not exs:
        continue
    path = out_dir / f"{name}.jsonl"
    with open(path, "w") as f:
        for ex in exs:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")
    print(f"Saved {len(exs):,} examples to {path}")

print("\nSample (train):")
ex = train_examples[0]
print(f"  anchor:   {ex['anchor'][:200]}")
print(f"  positive: {ex['positive'][:120]}")
print(f"  weight:   {ex['weight']}")
if "negative_1" in ex:
    print(f"  negative_1: {ex['negative_1'][:120]}")

if cold_examples:
    print("\nSample (cold):")
    ex = cold_examples[0]
    print(f"  anchor:   {ex['anchor'][:120]}")
    print(f"  positive: {ex['positive'][:120]}")

unique_positives = len({e["positive"] for e in train_examples} | {e["positive"] for e in cold_examples})
print(f"\nUnique positive texts across train+cold: {unique_positives:,} (catalog size {len(all_track_ids):,})")
