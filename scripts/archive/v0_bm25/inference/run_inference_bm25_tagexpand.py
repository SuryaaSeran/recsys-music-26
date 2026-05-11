"""
BM25 retrieval with tag expansion for history tracks.

Same as run_inference_bm25.py but includes full metadata (tags) for
previously recommended tracks in the retrieval query.

Usage:
    python scripts/run_inference_bm25_tagexpand.py [--sessions 0] [--tid bm25_tagexpand_v1]
"""
import argparse
import json
import os
from pathlib import Path
from tqdm import tqdm

import bm25s
from datasets import load_dataset, concatenate_datasets

parser = argparse.ArgumentParser()
parser.add_argument("--sessions", type=int, default=0)
parser.add_argument("--tid", default="bm25_tagexpand_v1")
parser.add_argument("--split", default="test")
parser.add_argument("--out_dir", default="exp/inference/devset")
parser.add_argument("--topk", type=int, default=20)
parser.add_argument("--hist_turns", type=int, default=4, help="How many music history turns to include (0=all)")
parser.add_argument("--text_turns", type=int, default=4, help="How many text history turns to include")
parser.add_argument("--no_exclude_seen", action="store_true", help="Do NOT exclude already-recommended tracks")
parser.add_argument("--dataset", default="talkpl-ai/TalkPlayData-Challenge-Dataset", help="HuggingFace dataset name")
args = parser.parse_args()

CACHE_PATH = "cache/bm25/track_metadata"
CORPUS_TYPES = ["track_name", "artist_name", "album_name", "release_date", "tag_list"]

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


def retrieve_bm25(query: str, topk: int = 20) -> list[str]:
    tokens = bm25s.tokenize([query.lower()])
    results = bm25_model.retrieve(tokens, k=topk, return_as="tuple")
    return [track_ids[int(i)] for i in results.documents[0]]


print(f"Loading dataset {args.dataset} split={args.split}...")
ds = load_dataset(args.dataset)[args.split]
sessions = list(ds)
if args.sessions > 0:
    sessions = sessions[:args.sessions]

print(f"Running inference on {len(sessions)} sessions...")
inference_results = []

for item in tqdm(sessions, desc="Sessions"):
    session_id = item["session_id"]
    user_id = item["user_id"]
    goal = item.get("conversation_goal", {}).get("listener_goal", "")
    culture = item.get("user_profile", {}).get("preferred_musical_culture", "")
    conversations = item["conversations"]

    for target_turn in sorted(set(c["turn_number"] for c in conversations)):
        music_turns = [c for c in conversations if c["turn_number"] == target_turn and c["role"] == "music"]
        if not music_turns:
            continue
        user_turns = [c for c in conversations if c["turn_number"] == target_turn and c["role"] == "user"]
        if not user_turns:
            continue
        user_query = user_turns[0]["content"]

        # Build history with tag expansion for music turns
        history_parts = [goal, culture]
        music_in_history = []
        text_in_history = []
        for turn in conversations:
            if turn["turn_number"] >= target_turn:
                break
            if turn["role"] == "music":
                music_in_history.append(turn["content"])
            elif turn["role"] in ("user", "assistant"):
                text_in_history.append(turn["content"])

        # Include last N music tracks with full metadata (name + artist + tags)
        music_slice = music_in_history if args.hist_turns == 0 else music_in_history[-args.hist_turns:]
        for tid in music_slice:
            history_parts.append(get_track_text(tid))

        # Include last N text turns (0 = skip)
        if args.text_turns > 0:
            history_parts.extend(text_in_history[-args.text_turns:])

        history_parts.append(user_query)
        retrieval_query = " ".join(p for p in history_parts if p)

        exclude_seen = not args.no_exclude_seen
        if exclude_seen and music_in_history:
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
            "turn_number": target_turn,
            "predicted_track_ids": predicted_track_ids,
            "predicted_response": response,
        })

Path(args.out_dir).mkdir(parents=True, exist_ok=True)
out_path = Path(args.out_dir) / f"{args.tid}.json"
with open(out_path, "w") as f:
    json.dump(inference_results, f, ensure_ascii=False, indent=2)

print(f"Saved {len(inference_results):,} predictions to {out_path}")
