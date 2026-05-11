"""
Hybrid retrieval: BM25 + CF-BPR for Music-CRS.

For warm users (have CF embeddings):
  BM25 top-200 candidates, reranked by CF similarity
For cold users:
  BM25 top-20 directly

Usage:
    python scripts/run_inference_hybrid.py [--sessions 50] [--tid hybrid_v1]
"""
import argparse
import json
import os
from pathlib import Path
from collections import defaultdict

import bm25s
import numpy as np
from datasets import load_dataset, concatenate_datasets
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument("--sessions", type=int, default=0)
parser.add_argument("--tid", default="hybrid_v1")
parser.add_argument("--split", default="test")
parser.add_argument("--out_dir", default="exp/inference/devset")
parser.add_argument("--topk", type=int, default=20)
parser.add_argument("--bm25_candidates", type=int, default=200)
parser.add_argument("--cf_weight", type=float, default=0.4, help="CF score weight in hybrid (0=BM25 only)")
args = parser.parse_args()

# ── Track metadata + BM25 index ───────────────────────────────────────────────
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

track_id_to_idx = {tid: i for i, tid in enumerate(track_ids)}

# ── CF-BPR embeddings ─────────────────────────────────────────────────────────
print("Loading track CF-BPR embeddings from challenge file...")
# Extract CF-BPR slice from challenge track embeddings
# Modality ordering: CLAP(512) + SigLIP2(768) + CF-BPR(128) + 3xQwen3(1024)
# CF-BPR: dims 1280:1408

track_emb_raw = np.load("data/challenge_track_embeddings.npy", mmap_mode="r")
cf_slice_raw = track_emb_raw[:, 1280:1408].copy()  # (46424, 128)
cf_norms = np.linalg.norm(cf_slice_raw, axis=1, keepdims=True)
track_cf_normed = cf_slice_raw / (cf_norms + 1e-8)  # L2-normalize

# Load track IDs for the challenge embeddings file
with open("data/challenge_track_ids.txt") as f:
    emb_track_ids = [l.strip() for l in f]
emb_idx_map = {tid: i for i, tid in enumerate(emb_track_ids)}  # track_id → row in track_cf_normed

print(f"Track CF embeddings: {track_cf_normed.shape}")

print("Loading user CF-BPR embeddings...")
user_ds = load_dataset("talkpl-ai/TalkPlayData-Challenge-User-Embeddings")
user_cf_map = {}  # user_id → normalized 128d CF embedding
for split_name in ["test_warm", "test_cold", "train"]:
    for row in user_ds[split_name]:
        cf = row.get("cf-bpr")
        if cf and any(cf):
            v = np.array(cf, dtype=np.float32)
            v = v / (np.linalg.norm(v) + 1e-8)
            user_cf_map[row["user_id"]] = v

print(f"User CF embeddings loaded: {len(user_cf_map)} users")

# ── Dataset ───────────────────────────────────────────────────────────────────
print(f"Loading dataset split={args.split}...")
ds = load_dataset("talkpl-ai/TalkPlayData-Challenge-Dataset")[args.split]
sessions = list(ds)
if args.sessions > 0:
    sessions = sessions[:args.sessions]

def retrieve_bm25_candidates(query: str, k: int) -> list[str]:
    tokens = bm25s.tokenize([query.lower()])
    results = bm25_model.retrieve(tokens, k=k, return_as="tuple")
    return [track_ids[int(i)] for i in results.documents[0]]


def hybrid_retrieve(query: str, user_id: str, topk: int = 20) -> list[str]:
    # BM25 candidates
    candidates = retrieve_bm25_candidates(query, k=args.bm25_candidates)

    user_cf = user_cf_map.get(user_id)
    if user_cf is None or args.cf_weight == 0:
        return candidates[:topk]

    # BM25 scores (rank-based)
    n = len(candidates)
    bm25_scores = {tid: (n - i) / n for i, tid in enumerate(candidates)}

    # CF scores: dot product of user CF with track CF for candidates
    cf_scores = {}
    for tid in candidates:
        emb_i = emb_idx_map.get(tid)
        if emb_i is not None:
            cf_scores[tid] = float(np.dot(user_cf, track_cf_normed[emb_i]))
        else:
            cf_scores[tid] = 0.0

    # Normalize CF scores to [0,1]
    if cf_scores:
        cf_vals = list(cf_scores.values())
        cf_min, cf_max = min(cf_vals), max(cf_vals)
        cf_range = cf_max - cf_min + 1e-8
        cf_scores = {k: (v - cf_min) / cf_range for k, v in cf_scores.items()}

    # Combine
    w = args.cf_weight
    combined = {
        tid: (1 - w) * bm25_scores.get(tid, 0) + w * cf_scores.get(tid, 0)
        for tid in candidates
    }
    ranked = sorted(combined.keys(), key=lambda t: combined[t], reverse=True)
    return ranked[:topk]


# ── Inference ─────────────────────────────────────────────────────────────────
print(f"Running inference on {len(sessions)} sessions (cf_weight={args.cf_weight})...")
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

        # Full conversation history
        history_parts = [goal, culture]
        for turn in conversations:
            if turn["turn_number"] >= target_turn:
                break
            if turn["role"] == "music":
                row = metadata_dict.get(turn["content"], {})
                name = (row.get("track_name") or [""])[0]
                artist = (row.get("artist_name") or [""])[0]
                history_parts.append(f"{name} {artist}")
            elif turn["role"] in ("user", "assistant"):
                history_parts.append(turn["content"])

        history_parts.append(user_query)
        retrieval_query = " ".join(p for p in history_parts if p)

        predicted_track_ids = hybrid_retrieve(retrieval_query, user_id, topk=args.topk)

        # Simple template response
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

# ── Save ──────────────────────────────────────────────────────────────────────
Path(args.out_dir).mkdir(parents=True, exist_ok=True)
out_path = Path(args.out_dir) / f"{args.tid}.json"
with open(out_path, "w") as f:
    json.dump(inference_results, f, ensure_ascii=False, indent=2)

print(f"Saved {len(inference_results):,} predictions to {out_path}")
