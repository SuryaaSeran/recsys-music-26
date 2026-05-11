"""
Best retrieval: BM25 + tag expansion + exclude-seen + optional CF reranking.

Usage:
    python scripts/run_inference_best_v3.py [--cf_weight 0.3] [--tid v3_cf30]
"""
import argparse
import json
import numpy as np
from pathlib import Path
from tqdm import tqdm

import bm25s
from datasets import load_dataset, concatenate_datasets

parser = argparse.ArgumentParser()
parser.add_argument("--sessions", type=int, default=0)
parser.add_argument("--tid", default="best_v3")
parser.add_argument("--split", default="test")
parser.add_argument("--dataset", default="talkpl-ai/TalkPlayData-Challenge-Dataset")
parser.add_argument("--out_dir", default="exp/inference/devset")
parser.add_argument("--topk", type=int, default=20)
parser.add_argument("--bm25_candidates", type=int, default=200)
parser.add_argument("--cf_weight", type=float, default=0.0)
parser.add_argument("--hist_turns", type=int, default=4)
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
    bm25_track_ids = json.load(f)

if args.cf_weight > 0:
    print("Loading CF-BPR embeddings...")
    track_emb_raw = np.load("data/challenge_track_embeddings.npy", mmap_mode="r")
    cf_slice_raw = track_emb_raw[:, 1280:1408].copy()
    cf_norms = np.linalg.norm(cf_slice_raw, axis=1, keepdims=True)
    track_cf_normed = cf_slice_raw / (cf_norms + 1e-8)
    with open("data/challenge_track_ids.txt") as f:
        emb_track_ids = [l.strip() for l in f]
    emb_idx_map = {tid: i for i, tid in enumerate(emb_track_ids)}

    print("Loading user CF-BPR embeddings...")
    user_ds = load_dataset("talkpl-ai/TalkPlayData-Challenge-User-Embeddings")
    user_cf_map = {}
    for split_name in ["test_warm", "test_cold", "train"]:
        for row in user_ds[split_name]:
            cf = row.get("cf-bpr")
            if cf and any(cf):
                v = np.array(cf, dtype=np.float32)
                v = v / (np.linalg.norm(v) + 1e-8)
                user_cf_map[row["user_id"]] = v
    print(f"User CF embeddings: {len(user_cf_map)}")


def get_track_text(tid: str) -> str:
    row = metadata_dict.get(tid, {})
    name = (row.get("track_name") or [""])[0]
    artist = (row.get("artist_name") or [""])[0]
    tags = " ".join(row.get("tag_list") or [])
    return f"{name} {artist} {tags}".strip()


def retrieve_bm25(query: str, topk: int) -> list[str]:
    tokens = bm25s.tokenize([query.lower()])
    results = bm25_model.retrieve(tokens, k=topk, return_as="tuple")
    return [bm25_track_ids[int(i)] for i in results.documents[0]]


print(f"Loading {args.dataset} split={args.split}...")
ds = load_dataset(args.dataset)[args.split]
sessions = list(ds)
if args.sessions > 0:
    sessions = sessions[:args.sessions]

print(f"Running {len(sessions)} sessions (cf_weight={args.cf_weight})...")
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

        music_in_history = []
        text_in_history = []
        for turn in conversations:
            if turn["turn_number"] >= target_turn:
                break
            if turn["role"] == "music":
                music_in_history.append(turn["content"])
            elif turn["role"] in ("user", "assistant"):
                text_in_history.append(turn["content"])

        history_parts = [goal, culture]
        music_slice = music_in_history if args.hist_turns == 0 else music_in_history[-args.hist_turns:]
        for tid in music_slice:
            history_parts.append(get_track_text(tid))
        if args.text_turns > 0:
            history_parts.extend(text_in_history[-args.text_turns:])
        history_parts.append(user_query)
        retrieval_query = " ".join(p for p in history_parts if p)

        # BM25 candidates with exclusion buffer
        retrieve_k = args.bm25_candidates + len(music_in_history) * 3
        candidates = retrieve_bm25(retrieval_query, topk=retrieve_k)

        # Exclude seen tracks
        if music_in_history:
            seen = set(music_in_history)
            candidates = [t for t in candidates if t not in seen]

        candidates = candidates[:args.bm25_candidates]

        if args.cf_weight > 0:
            user_cf = user_cf_map.get(user_id)
            if user_cf is not None:
                n = len(candidates)
                bm25_scores = {tid: (n - i) / n for i, tid in enumerate(candidates)}
                cf_scores = {}
                for tid in candidates:
                    idx = emb_idx_map.get(tid)
                    cf_scores[tid] = float(np.dot(user_cf, track_cf_normed[idx])) if idx is not None else 0.0

                cf_vals = list(cf_scores.values())
                cf_min, cf_max = min(cf_vals), max(cf_vals)
                cf_range = cf_max - cf_min + 1e-8
                cf_scores = {k: (v - cf_min) / cf_range for k, v in cf_scores.items()}

                w = args.cf_weight
                combined = {t: (1 - w) * bm25_scores[t] + w * cf_scores.get(t, 0) for t in candidates}
                candidates = sorted(combined, key=lambda t: combined[t], reverse=True)

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
