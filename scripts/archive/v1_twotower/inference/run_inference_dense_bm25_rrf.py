"""
Dense + BM25 hybrid with Reciprocal Rank Fusion.

Two parallel recall paths:
  1. BM25 top-bm25_pool (lexical recall, full query)
  2. Dense top-dense_pool (cosine over all 47k tracks, compact query)

Both pools merged, scored by RRF: 1/(k+dense_rank) + 1/(k+bm25_rank).
Exclude seen tracks, take top-20.

Usage:
    python scripts/run_inference_dense_bm25_rrf.py \
        --bm25_pool 500 --dense_pool 200 --rrf_k 60 \
        --tid dense_bm25_rrf_d200 --sessions 0
"""
import argparse
import json
import numpy as np
from pathlib import Path

import bm25s
from datasets import load_dataset, concatenate_datasets
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument("--model", default="models/twotower_v3/final")
parser.add_argument("--index_dir", default="cache/twotower_v3")
parser.add_argument("--bm25_pool", type=int, default=500)
parser.add_argument("--dense_pool", type=int, default=200)
parser.add_argument("--rrf_k", type=int, default=60)
parser.add_argument("--sessions", type=int, default=0)
parser.add_argument("--tid", default="dense_bm25_rrf")
parser.add_argument("--out_dir", default="exp/inference/devset")
parser.add_argument("--topk", type=int, default=20)
parser.add_argument("--hist_turns", type=int, default=4)
parser.add_argument("--text_turns", type=int, default=4)
args = parser.parse_args()

BM25_CACHE = "cache/bm25/track_metadata"

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
bm25_model = bm25s.BM25.load(BM25_CACHE, load_corpus=False)
with open(f"{BM25_CACHE}/track_ids.json") as f:
    bm25_track_ids = json.load(f)


def retrieve_bm25(query, topk):
    tokens = bm25s.tokenize([query.lower()])
    results = bm25_model.retrieve(tokens, k=topk, return_as="tuple")
    return [bm25_track_ids[int(i)] for i in results.documents[0]]


print(f"Loading two-tower model: {args.model}")
tower_model = SentenceTransformer(args.model)

print(f"Loading dense track index from {args.index_dir}...")
dense_embs = np.load(f"{args.index_dir}/track_embeddings.npy")
with open(f"{args.index_dir}/track_ids.json") as f:
    dense_ids = json.load(f)
dense_id_to_idx = {tid: i for i, tid in enumerate(dense_ids)}
print(f"  Track index: {dense_embs.shape}")

print("Loading dev sessions...")
ds = load_dataset("talkpl-ai/TalkPlayData-Challenge-Dataset")
sessions = list(ds["test"])
if args.sessions > 0:
    sessions = sessions[:args.sessions]

print(f"Running {len(sessions)} sessions (bm25_pool={args.bm25_pool}, dense_pool={args.dense_pool}, rrf_k={args.rrf_k})...")
inference_results = []

for item in tqdm(sessions, desc="Sessions"):
    session_id = item["session_id"]
    user_id = item["user_id"]
    goal = item.get("conversation_goal", {}).get("listener_goal", "")
    culture = item.get("user_profile", {}).get("preferred_musical_culture", "")
    conversations = item["conversations"]

    music_in_history = []
    text_in_history = []

    for turn in conversations:
        if turn["role"] != "music":
            if turn["role"] in ("user", "assistant"):
                text_in_history.append(turn["content"])
            continue

        turn_number = turn["turn_number"]

        # Compact dense query (user request first, matches training)
        latest_user = text_in_history[-1] if text_in_history else ""
        dense_parts = [latest_user, goal, culture]
        for tid in music_in_history[-2:]:
            na = get_track_name_artist(tid)
            if na:
                dense_parts.append(na)
        dense_query = " ".join(p for p in dense_parts if p)

        # Full BM25 query (tags + full history)
        bm25_parts = [goal, culture]
        for tid in music_in_history[-args.hist_turns:]:
            bm25_parts.append(get_track_text(tid))
        bm25_parts.extend(text_in_history[-args.text_turns:])
        bm25_query = " ".join(p for p in bm25_parts if p)

        seen = set(music_in_history)

        # --- Recall path 1: BM25 ---
        bm25_cands_raw = retrieve_bm25(bm25_query, topk=args.bm25_pool + len(seen) * 3)
        bm25_cands = [t for t in bm25_cands_raw if t not in seen][:args.bm25_pool]
        bm25_rank = {tid: r for r, tid in enumerate(bm25_cands)}

        # --- Recall path 2: Dense (full cosine over all tracks) ---
        query_emb = tower_model.encode(dense_query, normalize_embeddings=True, convert_to_numpy=True)
        all_cos = dense_embs @ query_emb  # (N_tracks,)
        # Get top dense_pool + buffer for seen filtering
        top_dense_idx = np.argsort(-all_cos)
        dense_cands = []
        for idx in top_dense_idx:
            tid = dense_ids[idx]
            if tid not in seen:
                dense_cands.append(tid)
                if len(dense_cands) >= args.dense_pool:
                    break
        dense_rank = {tid: r for r, tid in enumerate(dense_cands)}

        # --- Merge pools ---
        bm25_set = set(bm25_cands)
        dense_set = set(dense_cands)
        all_cands = bm25_cands + [t for t in dense_cands if t not in bm25_set]

        if not all_cands:
            inference_results.append({
                "session_id": session_id, "user_id": user_id,
                "turn_number": turn_number,
                "predicted_track_ids": [], "predicted_response": "No recommendation.",
            })
            music_in_history.append(turn["content"])
            continue

        # --- RRF scoring ---
        k = args.rrf_k
        n = len(all_cands)
        scored = []
        for tid in all_cands:
            br = bm25_rank.get(tid, n)
            dr = dense_rank.get(tid, n)
            rrf_score = 1.0 / (k + dr) + 1.0 / (k + br)
            scored.append((tid, rrf_score))

        scored.sort(key=lambda x: -x[1])
        predicted = [tid for tid, _ in scored[:args.topk]]

        top = predicted[0] if predicted else ""
        row = metadata_dict.get(top, {})
        name = (row.get("track_name") or ["this track"])[0]
        artist = (row.get("artist_name") or ["the artist"])[0]

        inference_results.append({
            "session_id": session_id, "user_id": user_id,
            "turn_number": turn_number,
            "predicted_track_ids": predicted,
            "predicted_response": f'I recommend "{name}" by {artist} based on your request.',
        })
        music_in_history.append(turn["content"])

Path(args.out_dir).mkdir(parents=True, exist_ok=True)
out_path = Path(args.out_dir) / f"{args.tid}.json"
with open(out_path, "w") as f:
    json.dump(inference_results, f, ensure_ascii=False, indent=2)
print(f"Saved {len(inference_results):,} predictions to {out_path}")
