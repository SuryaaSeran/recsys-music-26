"""
Blind set inference: BM25 + tag expansion + exclude-seen.

Blind set format: conversations[-1] is the user query to predict.
All previous turns (including music turns) are historical context.

Usage:
    python scripts/run_inference_blind.py --dataset talkpl-ai/TalkPlayData-Challenge-Blind-A --tid blind_a_v2
"""
import argparse
import json
from pathlib import Path
from tqdm import tqdm

import bm25s
from datasets import load_dataset, concatenate_datasets

parser = argparse.ArgumentParser()
parser.add_argument("--dataset", default="talkpl-ai/TalkPlayData-Challenge-Blind-A")
parser.add_argument("--split", default="test")
parser.add_argument("--tid", default="blind_a_v2")
parser.add_argument("--out_dir", default="exp/inference/blind_a")
parser.add_argument("--topk", type=int, default=20)
parser.add_argument("--hist_turns", type=int, default=4, help="Music history turns (0=all)")
parser.add_argument("--text_turns", type=int, default=4)
args = parser.parse_args()

CACHE_PATH = "cache/bm25/track_metadata"

print("Loading track metadata...")
meta_ds = load_dataset("talkpl-ai/TalkPlayData-Challenge-Track-Metadata")
all_tracks = concatenate_datasets([meta_ds["all_tracks"], meta_ds["test_tracks"]])
metadata_dict = {row["track_id"]: row for row in all_tracks}

print("Loading BM25 index...")
bm25_model = bm25s.BM25.load(CACHE_PATH, load_corpus=False)
with open(f"{CACHE_PATH}/track_ids.json") as f:
    track_ids = json.load(f)


def get_track_text(tid: str) -> str:
    row = metadata_dict.get(tid, {})
    name = (row.get("track_name") or [""])[0]
    artist = (row.get("artist_name") or [""])[0]
    tags = " ".join(row.get("tag_list") or [])
    return f"{name} {artist} {tags}".strip()


def retrieve_bm25(query: str, topk: int) -> list[str]:
    tokens = bm25s.tokenize([query.lower()])
    results = bm25_model.retrieve(tokens, k=topk, return_as="tuple")
    return [track_ids[int(i)] for i in results.documents[0]]


print(f"Loading {args.dataset} split={args.split}...")
ds = load_dataset(args.dataset)[args.split]
sessions = list(ds)

print(f"Running inference on {len(sessions)} sessions...")
inference_results = []

for item in tqdm(sessions, desc="Sessions"):
    session_id = item["session_id"]
    user_id = item["user_id"]
    goal = item.get("conversation_goal", {}).get("listener_goal", "")
    culture = item.get("user_profile", {}).get("preferred_musical_culture", "")
    conversations = item["conversations"]

    # Last conversation entry is the target user query
    user_query = conversations[-1]["content"]
    turn_number = conversations[-1]["turn_number"]
    history_convs = conversations[:-1]

    # Separate music and text history
    music_in_history = []
    text_in_history = []
    for turn in history_convs:
        if turn["role"] == "music":
            music_in_history.append(turn["content"])
        elif turn["role"] in ("user", "assistant"):
            text_in_history.append(turn["content"])

    # Build retrieval query
    history_parts = [goal, culture]

    music_slice = music_in_history if args.hist_turns == 0 else music_in_history[-args.hist_turns:]
    for tid in music_slice:
        history_parts.append(get_track_text(tid))

    if args.text_turns > 0:
        history_parts.extend(text_in_history[-args.text_turns:])

    history_parts.append(user_query)
    retrieval_query = " ".join(p for p in history_parts if p)

    # Retrieve, excluding already-seen tracks
    if music_in_history:
        retrieve_k = args.topk + len(music_in_history) * 3
        candidates = retrieve_bm25(retrieval_query, topk=retrieve_k)
        seen = set(music_in_history)
        candidates = [t for t in candidates if t not in seen]
    else:
        candidates = retrieve_bm25(retrieval_query, topk=args.topk)
    predicted_track_ids = candidates[:args.topk]

    top = predicted_track_ids[0] if predicted_track_ids else ""
    row = metadata_dict.get(top, {})
    name = (row.get("track_name") or ["this track"])[0]
    artist = (row.get("artist_name") or ["the artist"])[0]
    response = f'I recommend "{name}" by {artist} based on your preferences.'

    inference_results.append({
        "session_id": session_id,
        "user_id": user_id,
        "turn_number": turn_number,
        "predicted_track_ids": predicted_track_ids,
        "predicted_response": response,
    })

Path(args.out_dir).mkdir(parents=True, exist_ok=True)
out_path = Path(args.out_dir) / f"{args.tid}.json"
with open(out_path, "w") as f:
    json.dump(inference_results, f, ensure_ascii=False, indent=2)

print(f"Saved {len(inference_results):,} predictions to {out_path}")
