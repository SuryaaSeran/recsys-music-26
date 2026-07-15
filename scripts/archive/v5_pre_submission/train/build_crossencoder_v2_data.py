"""Build cross-encoder v2 training data.

For every music turn in TRAIN, emit one (anchor, candidate, label) triple
per candidate. Each turn produces:
  - 1 positive (gold track)
  - N hard negatives (mix of BM25 top-200 / rejected-history / random)

Anchor format mirrors the inference TT query so the CE sees the same
text distribution as the rescorer at eval time:
  "<latest_user> Goal:<goal> <culture> <name1> by <artist1> <name2> by <artist2>"

Candidate format mirrors the v6/v7 track text (name, artist, album,
top-12 tags, year).

Output: data/crossencoder_v2/{train,valid}.jsonl
Each row: {"anchor": str, "candidate": str, "label": 0|1}

Usage:
    python scripts/train/build_crossencoder_v2_data.py
"""
import argparse
import json
import random
from pathlib import Path

import bm25s
from datasets import load_dataset, concatenate_datasets
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument("--hard_negs", type=int, default=5,
                    help="Negatives per turn (mix of BM25 + rejected + random).")
parser.add_argument("--bm25_pool", type=int, default=200)
parser.add_argument("--valid_frac", type=float, default=0.01)
parser.add_argument("--out_dir", default="data/crossencoder_v2")
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--max_turns_per_session", type=int, default=2,
                    help="Sample up to this many music turns per session (matches v7 budget).")
args = parser.parse_args()

random.seed(args.seed)
CACHE_PATH = "cache/bm25/track_metadata"

print("Loading track metadata...")
meta_ds = load_dataset("talkpl-ai/TalkPlayData-Challenge-Track-Metadata")
all_tracks = concatenate_datasets([meta_ds["all_tracks"], meta_ds["test_tracks"]])
metadata_dict = {row["track_id"]: row for row in all_tracks}
all_track_ids = list(metadata_dict.keys())


def get_candidate_text(tid: str) -> str:
    row = metadata_dict.get(tid, {})
    name   = (row.get("track_name")  or [""])[0]
    artist = (row.get("artist_name") or [""])[0]
    album  = (row.get("album_name")  or [""])[0]
    tags   = " ".join((row.get("tag_list") or [])[:12])
    release = row.get("release_date") or ""
    year = str(release)[:4] if release else ""
    parts = [name]
    if artist: parts.append(f"by {artist}")
    if album:  parts.append(f"| Album: {album}")
    if tags:   parts.append(f"| Tags: {tags}")
    if year:   parts.append(f"| {year}")
    return " ".join(parts).strip()


def get_track_name_artist(tid: str) -> str:
    row = metadata_dict.get(tid, {})
    name   = (row.get("track_name")  or [""])[0]
    artist = (row.get("artist_name") or [""])[0]
    return f"{name} by {artist}".strip()


print("Loading BM25 index...")
bm25_model = bm25s.BM25.load(CACHE_PATH, load_corpus=False)
with open(f"{CACHE_PATH}/track_ids.json") as f:
    bm25_track_ids = json.load(f)


def retrieve_bm25(query: str, topk: int) -> list[str]:
    tokens = bm25s.tokenize([query.lower()])
    results = bm25_model.retrieve(tokens, k=topk, return_as="tuple")
    return [bm25_track_ids[int(i)] for i in results.documents[0]]


print("Loading TRAIN sessions...")
ds = load_dataset("talkpl-ai/TalkPlayData-Challenge-Dataset", split="train")
sessions = list(ds)
random.shuffle(sessions)
n_valid = int(len(sessions) * args.valid_frac)
valid_sessions = sessions[:n_valid]
train_sessions = sessions[n_valid:]
print(f"Train sessions: {len(train_sessions)}, Valid sessions: {len(valid_sessions)}")


def build_anchor(item: dict, latest_user: str, music_history: list[str]) -> str:
    goal = (item.get("conversation_goal") or {}).get("listener_goal", "") or ""
    culture = (item.get("user_profile") or {}).get("preferred_musical_culture", "") or ""
    parts = [latest_user]
    if goal:    parts.append(f"Goal: {goal}")
    if culture: parts.append(culture)
    for tid in music_history[-2:]:
        na = get_track_name_artist(tid)
        if na: parts.append(na)
    return " ".join(p for p in parts if p).strip()


def build_rows(sessions: list, desc: str) -> list[dict]:
    rows = []
    for item in tqdm(sessions, desc=desc):
        progress_by_turn = {
            a["turn_number"]: a["goal_progress_assessment"]
            for a in (item.get("goal_progress_assessments") or [])
        }
        music_history: list[str] = []
        text_history:  list[str] = []
        rejected_tids: set       = set()
        per_session: list[dict]  = []

        for turn in item["conversations"]:
            role = turn["role"]
            if role == "music":
                gold = turn["content"]
                gold_text = get_candidate_text(gold)
                if not gold_text:
                    music_history.append(gold); continue

                latest_user = text_history[-1] if text_history else ""
                anchor = build_anchor(item, latest_user, music_history)
                if not anchor:
                    music_history.append(gold); continue

                seen = set(music_history) | {gold}
                negs: list[str] = []

                bm25_results = retrieve_bm25(anchor, topk=args.bm25_pool + 1)
                bm25_negs = [t for t in bm25_results if t not in seen]
                for tid in bm25_negs[:2]:
                    txt = get_candidate_text(tid)
                    if txt: negs.append(txt)

                rej_pool = [t for t in rejected_tids if t not in seen]
                if rej_pool:
                    tid = random.choice(rej_pool)
                    txt = get_candidate_text(tid)
                    if txt: negs.append(txt)

                rand_pool = [t for t in all_track_ids if t not in seen]
                random.shuffle(rand_pool)
                for tid in rand_pool:
                    if len(negs) >= args.hard_negs: break
                    txt = get_candidate_text(tid)
                    if txt: negs.append(txt)

                per_session.append({"anchor": anchor, "positive": gold_text,
                                    "negatives": negs[: args.hard_negs]})

                music_history.append(gold)
                if progress_by_turn.get(turn["turn_number"]) == "DOES_NOT_MOVE_TOWARD_GOAL":
                    rejected_tids.add(gold)
            elif role in ("user", "assistant"):
                text_history.append(turn["content"])

        if args.max_turns_per_session > 0 and len(per_session) > args.max_turns_per_session:
            per_session = random.sample(per_session, args.max_turns_per_session)

        # listwise groups: one row per turn
        for ex in per_session:
            if len(ex["negatives"]) < args.hard_negs:
                continue  # require fixed group size so HF Dataset has a uniform column shape
            rows.append({
                "query":  ex["anchor"],
                "docs":   [ex["positive"]] + ex["negatives"],
                "labels": [1.0] + [0.0] * len(ex["negatives"]),
            })
    return rows


train_rows = build_rows(train_sessions, "train")
valid_rows = build_rows(valid_sessions, "valid")

print(f"Train groups: {len(train_rows):,}  (group_size={args.hard_negs + 1})")
print(f"Valid groups: {len(valid_rows):,}")

out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
for name, rows in [("train", train_rows), ("valid", valid_rows)]:
    p = out / f"{name}.jsonl"
    with open(p, "w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"Saved {len(rows):,} rows to {p}")

print("\nSample group:")
print("  query    :", train_rows[0]["query"][:200])
print("  positive :", train_rows[0]["docs"][0][:120])
print("  negative :", train_rows[0]["docs"][1][:120])
print("  labels   :", train_rows[0]["labels"])
