"""
Hybrid BM25 + two-tower retrieval inference.

Pipeline:
  1. BM25 top-100 candidates (recall layer)
  2. Encode query with two-tower query encoder
  3. Rerank BM25 candidates by cosine similarity to query embedding
  4. Exclude seen tracks, take top-20

Usage:
    python scripts/run_inference_twotower.py \
        --model models/twotower_v1/final \
        --tid twotower_v1 \
        --sessions 0 \
        --bm25_pool 100 \
        --dense_weight 0.5
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
parser.add_argument("--model", default="models/twotower_v1/final")
parser.add_argument("--index_dir", default="cache/twotower")
parser.add_argument("--sessions", type=int, default=0)
parser.add_argument("--tid", default="twotower_v1")
parser.add_argument("--out_dir", default="exp/inference/devset")
parser.add_argument("--topk", type=int, default=20)
parser.add_argument("--bm25_pool", type=int, default=100)
parser.add_argument("--dense_weight", type=float, default=0.5,
                    help="Score = dense_weight*cosine + (1-dense_weight)*bm25_rank_score")
parser.add_argument("--dense_pool", type=int, default=0,
                    help="Add top-N from dense-only retrieval to BM25 pool (0=disabled)")
parser.add_argument("--hist_turns", type=int, default=4)
parser.add_argument("--text_turns", type=int, default=4)
args = parser.parse_args()

BM25_CACHE = "cache/bm25/track_metadata"

print("Loading track metadata...")
meta_ds = load_dataset("talkpl-ai/TalkPlayData-Challenge-Track-Metadata")
all_tracks = concatenate_datasets([meta_ds["all_tracks"], meta_ds["test_tracks"]])
metadata_dict = {row["track_id"]: row for row in all_tracks}


def get_track_text(tid: str) -> str:
    row = metadata_dict.get(tid, {})
    name = (row.get("track_name") or [""])[0]
    artist = (row.get("artist_name") or [""])[0]
    tags = " ".join(row.get("tag_list") or [])
    return f"{name} {artist} {tags}".strip()


print("Loading BM25 index...")
bm25_model = bm25s.BM25.load(BM25_CACHE, load_corpus=False)
with open(f"{BM25_CACHE}/track_ids.json") as f:
    bm25_track_ids = json.load(f)


def retrieve_bm25(query: str, topk: int) -> list[str]:
    tokens = bm25s.tokenize([query.lower()])
    results = bm25_model.retrieve(tokens, k=topk, return_as="tuple")
    return [bm25_track_ids[int(i)] for i in results.documents[0]]


print(f"Loading two-tower model: {args.model}")
tower_model = SentenceTransformer(args.model)

print(f"Loading dense track index from {args.index_dir}...")
dense_embeddings = np.load(f"{args.index_dir}/track_embeddings.npy")
with open(f"{args.index_dir}/track_ids.json") as f:
    dense_track_ids = json.load(f)
dense_id_to_idx = {tid: i for i, tid in enumerate(dense_track_ids)}
print(f"  Index: {dense_embeddings.shape}")

print("Loading dev sessions...")
ds = load_dataset("talkpl-ai/TalkPlayData-Challenge-Dataset")
sessions = list(ds["test"])
if args.sessions > 0:
    sessions = sessions[:args.sessions]

print(f"Running {len(sessions)} sessions...")
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

        # Build query (mirrors run_inference_bm25_tagexpand.py)
        parts = [goal, culture]
        music_slice = music_in_history[-args.hist_turns:] if args.hist_turns > 0 else music_in_history
        for tid in music_slice:
            parts.append(get_track_text(tid))
        if args.text_turns > 0:
            parts.extend(text_in_history[-args.text_turns:])
        query = " ".join(p for p in parts if p)

        # BM25 recall layer
        retrieve_k = args.bm25_pool + len(music_in_history) * 3
        bm25_candidates = retrieve_bm25(query, topk=retrieve_k)

        # Exclude seen tracks
        seen = set(music_in_history)
        bm25_candidates = [t for t in bm25_candidates if t not in seen]
        bm25_candidates = bm25_candidates[:args.bm25_pool]

        if not bm25_candidates:
            inference_results.append({
                "session_id": session_id,
                "user_id": user_id,
                "turn_number": turn_number,
                "predicted_track_ids": [],
                "predicted_response": "I couldn't find a recommendation.",
            })
            music_in_history.append(turn["content"])
            continue

        # Encode query with two-tower model
        query_emb = tower_model.encode(
            query, normalize_embeddings=True, convert_to_numpy=True
        )  # (dim,)

        # Optional: full dense retrieval pass to expand candidates
        all_candidates = list(bm25_candidates)
        if args.dense_pool > 0:
            dense_scores_all = dense_embeddings @ query_emb  # (N,)
            top_dense_idx = np.argpartition(-dense_scores_all, args.dense_pool)[:args.dense_pool]
            top_dense_idx = top_dense_idx[np.argsort(-dense_scores_all[top_dense_idx])]
            dense_only = [dense_track_ids[i] for i in top_dense_idx
                          if dense_track_ids[i] not in seen and dense_track_ids[i] not in set(bm25_candidates)]
            all_candidates = all_candidates + dense_only

        # Get dense embeddings for candidates (those in index)
        candidate_indices = [dense_id_to_idx[t] for t in all_candidates if t in dense_id_to_idx]
        candidates_in_index = [t for t in all_candidates if t in dense_id_to_idx]
        candidates_not_in_index = [t for t in all_candidates if t not in dense_id_to_idx]

        if candidate_indices:
            cand_embs = dense_embeddings[candidate_indices]  # (n, dim)
            cosine_scores = cand_embs @ query_emb             # (n,)
        else:
            cosine_scores = np.array([])

        # BM25 rank score: reciprocal rank (1.0 for rank 1, 0.5 for rank 2, ...)
        bm25_rank_scores_dict = {
            tid: 1.0 / (rank + 1)
            for rank, tid in enumerate(all_candidates)
        }

        # Combine scores
        scored = []
        for i, tid in enumerate(candidates_in_index):
            dense_s = float(cosine_scores[i]) if len(cosine_scores) > i else 0.0
            bm25_s = bm25_rank_scores_dict.get(tid, 0.0)
            combined = args.dense_weight * dense_s + (1 - args.dense_weight) * bm25_s
            scored.append((tid, combined))

        # Tracks not in dense index: use BM25 rank score only
        for tid in candidates_not_in_index:
            bm25_s = bm25_rank_scores_dict.get(tid, 0.0)
            scored.append((tid, (1 - args.dense_weight) * bm25_s))

        scored.sort(key=lambda x: -x[1])
        predicted_track_ids = [tid for tid, _ in scored[:args.topk]]

        # Simple template response
        top = predicted_track_ids[0] if predicted_track_ids else ""
        row = metadata_dict.get(top, {})
        name = (row.get("track_name") or ["this track"])[0]
        artist = (row.get("artist_name") or ["the artist"])[0]
        response = f'I recommend "{name}" by {artist} based on your request.'

        inference_results.append({
            "session_id": session_id,
            "user_id": user_id,
            "turn_number": turn_number,
            "predicted_track_ids": predicted_track_ids,
            "predicted_response": response,
        })

        music_in_history.append(turn["content"])

Path(args.out_dir).mkdir(parents=True, exist_ok=True)
out_path = Path(args.out_dir) / f"{args.tid}.json"
with open(out_path, "w") as f:
    json.dump(inference_results, f, ensure_ascii=False, indent=2)

print(f"Saved {len(inference_results):,} predictions to {out_path}")
