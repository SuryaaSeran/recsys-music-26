"""
BM25-based inference for Music-CRS.
Uses BM25 over track metadata for retrieval + base Qwen for response.
Much more accurate than semantic ID approach for named-entity queries.

Usage:
    python scripts/run_inference_bm25.py [--sessions 100] [--tid bm25_v1]
"""
import argparse
import json
import os
import sys
from pathlib import Path
from tqdm import tqdm

# Add baseline module to path
sys.path.insert(0, str(Path(__file__).parent.parent / "music-crs-baselines"))

import bm25s
from datasets import load_dataset, concatenate_datasets

parser = argparse.ArgumentParser()
parser.add_argument("--sessions", type=int, default=0, help="0=all")
parser.add_argument("--tid", default="bm25_v1")
parser.add_argument("--split", default="test")
parser.add_argument("--out_dir", default="exp/inference/devset")
parser.add_argument("--no_response", action="store_true")
parser.add_argument("--topk", type=int, default=20)
args = parser.parse_args()

# ── Build BM25 index ──────────────────────────────────────────────────────────
CACHE_DIR = "cache/bm25"
CORPUS_TYPES = ["track_name", "artist_name", "album_name", "release_date", "tag_list"]
CACHE_PATH = f"{CACHE_DIR}/track_metadata"

print("Loading track metadata...")
meta_ds = load_dataset("talkpl-ai/TalkPlayData-Challenge-Track-Metadata")
all_tracks = concatenate_datasets([meta_ds["all_tracks"], meta_ds["test_tracks"]])
metadata_dict = {row["track_id"]: row for row in all_tracks}
track_ids = list(metadata_dict.keys())

print(f"Tracks: {len(track_ids):,}")

def stringify_metadata(row: dict) -> str:
    parts = []
    for field in CORPUS_TYPES:
        val = row.get(field, "")
        if isinstance(val, list):
            val = " ".join(val)
        if val:
            parts.append(str(val))
    return " ".join(parts).lower()


if os.path.exists(f"{CACHE_PATH}/index.npz"):
    print("Loading cached BM25 index...")
    bm25_model = bm25s.BM25.load(CACHE_PATH, load_corpus=False)
    with open(f"{CACHE_PATH}/track_ids.json") as f:
        track_ids = json.load(f)
else:
    print("Building BM25 index (first run)...")
    corpus = [stringify_metadata(metadata_dict[tid]) for tid in track_ids]
    corpus_tokens = bm25s.tokenize(corpus)
    bm25_model = bm25s.BM25()
    bm25_model.index(corpus_tokens)
    os.makedirs(CACHE_PATH, exist_ok=True)
    bm25_model.save(CACHE_PATH)
    with open(f"{CACHE_PATH}/track_ids.json", "w") as f:
        json.dump(track_ids, f)
    print("BM25 index built and cached.")


def retrieve_bm25(query: str, topk: int = 20) -> list[str]:
    tokens = bm25s.tokenize([query.lower()])
    doc_scores = bm25_model.retrieve(tokens, k=topk, return_as="tuple")
    indices = doc_scores.documents[0]  # numpy array of int indices
    return [track_ids[int(i)] for i in indices]


# ── Load response model ───────────────────────────────────────────────────────
resp_model = resp_tokenizer = None
if not args.no_response:
    from mlx_lm import load, generate
    print("Loading Qwen response model...")
    resp_model, resp_tokenizer = load("models/qwen_sid_patched")


def generate_response(chat_history: list, top_tracks: list[str]) -> str:
    if resp_model is None:
        tid = top_tracks[0] if top_tracks else ""
        row = metadata_dict.get(tid, {})
        name = (row.get("track_name") or ["this track"])[0]
        artist = (row.get("artist_name") or ["the artist"])[0]
        return f'I recommend "{name}" by {artist}. It should match your preferences perfectly.'

    from mlx_lm import generate

    tid = top_tracks[0] if top_tracks else ""
    row = metadata_dict.get(tid, {})
    name = (row.get("track_name") or ["this track"])[0]
    artist = (row.get("artist_name") or ["the artist"])[0]

    messages = [
        {"role": "system", "content": "You are a friendly music recommendation assistant. Give a brief (2-3 sentence) enthusiastic recommendation."},
    ]
    messages.extend(chat_history[-4:])
    messages.append({
        "role": "user",
        "content": f'Recommended track: "{name}" by {artist}. Please recommend it briefly.',
    })
    prompt = resp_tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    out = generate(resp_model, resp_tokenizer, prompt=prompt, max_tokens=80).strip()
    return out or f'I recommend "{name}" by {artist}!'


# ── Dataset ───────────────────────────────────────────────────────────────────
print(f"Loading test dataset (split={args.split})...")
ds = load_dataset("talkpl-ai/TalkPlayData-Challenge-Dataset")[args.split]
sessions = list(ds)
if args.sessions > 0:
    sessions = sessions[:args.sessions]

print(f"Running inference on {len(sessions)} sessions...")

inference_results = []

for item in tqdm(sessions, desc="Sessions"):
    session_id = item["session_id"]
    user_id = item["user_id"]
    conversations = item["conversations"]

    for target_turn in sorted(set(c["turn_number"] for c in conversations)):
        # Need a music turn at this number (otherwise nothing to predict)
        music_turns = [c for c in conversations if c["turn_number"] == target_turn and c["role"] == "music"]
        if not music_turns:
            continue

        # User query at this turn
        user_turns = [c for c in conversations if c["turn_number"] == target_turn and c["role"] == "user"]
        if not user_turns:
            continue
        user_query = user_turns[0]["content"]

        # Chat history (text only, no music track IDs)
        history = []
        for turn in conversations:
            if turn["turn_number"] >= target_turn:
                break
            if turn["role"] == "music":
                row = metadata_dict.get(turn["content"], {})
                name = (row.get("track_name") or ["a track"])[0]
                artist = (row.get("artist_name") or ["unknown"])[0]
                history.append({"role": "assistant", "content": f'Recommended: "{name}" by {artist}'})
            else:
                history.append({"role": turn["role"], "content": turn["content"]})

        # Build retrieval query: concatenate all context
        retrieval_text = " ".join([
            item.get("conversation_goal", {}).get("listener_goal", ""),
            item.get("user_profile", {}).get("preferred_musical_culture", ""),
        ] + [h["content"] for h in history[-4:]] + [user_query])

        predicted_track_ids = retrieve_bm25(retrieval_text, topk=args.topk)
        predicted_response = generate_response(history, predicted_track_ids)

        inference_results.append({
            "session_id": session_id,
            "user_id": user_id,
            "turn_number": target_turn,
            "predicted_track_ids": predicted_track_ids,
            "predicted_response": predicted_response,
        })

# ── Save ──────────────────────────────────────────────────────────────────────
Path(args.out_dir).mkdir(parents=True, exist_ok=True)
out_path = Path(args.out_dir) / f"{args.tid}.json"
with open(out_path, "w") as f:
    json.dump(inference_results, f, ensure_ascii=False, indent=2)

print(f"Saved {len(inference_results):,} predictions to {out_path}")
