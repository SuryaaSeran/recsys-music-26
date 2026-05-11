"""
Option 1: Dense retrieval using precomputed Qwen3 metadata embeddings.

Pipeline:
  1. Encode query with Qwen3-Embedding-0.6B (same model used for track metadata)
  2. Retrieve top-K by cosine similarity against metadata-qwen3 slice (dims 3456:4480)
  3. Combine with BM25 top-100 (union, rerank by combined score)
  4. Exclude seen tracks, take top-20

Usage:
    python scripts/run_inference_qwen3_dense.py \
        --tid qwen3_dense_v1 \
        --sessions 0
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
parser.add_argument("--sessions", type=int, default=0)
parser.add_argument("--tid", default="qwen3_dense_v1")
parser.add_argument("--out_dir", default="exp/inference/devset")
parser.add_argument("--topk", type=int, default=20)
parser.add_argument("--bm25_pool", type=int, default=100)
parser.add_argument("--dense_pool", type=int, default=100)
parser.add_argument("--dense_weight", type=float, default=0.5)
parser.add_argument("--hist_turns", type=int, default=4)
parser.add_argument("--text_turns", type=int, default=4)
parser.add_argument("--use_instruction", action="store_true",
                    help="Prepend task instruction to query (Qwen3 instruction format)")
args = parser.parse_args()

BM25_CACHE = "cache/bm25/track_metadata"
CHALLENGE_EMB = "data/challenge_track_embeddings.npy"
CHALLENGE_IDS = "data/challenge_track_ids.txt"
# metadata-qwen3_embedding_0.6b is the last 1024 dims
QWEN3_SLICE = (3456, 4480)

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


print("Loading challenge Qwen3 metadata embeddings...")
challenge_emb_full = np.load(CHALLENGE_EMB, mmap_mode="r")
qwen3_embs = np.array(challenge_emb_full[:, QWEN3_SLICE[0]:QWEN3_SLICE[1]], dtype=np.float32)
with open(CHALLENGE_IDS) as f:
    challenge_ids = [l.strip() for l in f]
challenge_id_to_idx = {tid: i for i, tid in enumerate(challenge_ids)}
print(f"  Qwen3 slice: {qwen3_embs.shape}")

# Normalize for cosine similarity
norms = np.linalg.norm(qwen3_embs, axis=1, keepdims=True)
norms = np.where(norms == 0, 1, norms)
qwen3_embs = qwen3_embs / norms

print("Loading Qwen3-Embedding-0.6B model...")
qwen3_model = SentenceTransformer("Qwen/Qwen3-Embedding-0.6B")

INSTRUCTION = "Instruct: Given a music conversation, retrieve the most relevant track\nQuery: "

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

        # Build query
        parts = [goal, culture]
        music_slice = music_in_history[-args.hist_turns:] if args.hist_turns > 0 else music_in_history
        for tid in music_slice:
            parts.append(get_track_text(tid))
        if args.text_turns > 0:
            parts.extend(text_in_history[-args.text_turns:])
        raw_query = " ".join(p for p in parts if p)

        # BM25 candidates
        retrieve_k = args.bm25_pool + len(music_in_history) * 3
        bm25_candidates = retrieve_bm25(raw_query, topk=retrieve_k)
        seen = set(music_in_history)
        bm25_candidates = [t for t in bm25_candidates if t not in seen][:args.bm25_pool]

        # Qwen3 dense candidates
        query_text = (INSTRUCTION + raw_query) if args.use_instruction else raw_query
        query_emb = qwen3_model.encode(
            query_text, normalize_embeddings=True, convert_to_numpy=True
        )

        # Full dense retrieval pass
        dense_scores_all = qwen3_embs @ query_emb  # (46424,)
        top_dense_indices = np.argpartition(-dense_scores_all, args.dense_pool)[:args.dense_pool]
        top_dense_indices = top_dense_indices[np.argsort(-dense_scores_all[top_dense_indices])]
        dense_candidates = [
            challenge_ids[i] for i in top_dense_indices
            if challenge_ids[i] not in seen
        ]

        # Merge candidates
        all_candidates_set = set(bm25_candidates)
        all_candidates = list(bm25_candidates)
        for tid in dense_candidates:
            if tid not in all_candidates_set and tid not in seen:
                all_candidates.append(tid)
                all_candidates_set.add(tid)

        # Score: dense_weight * cosine + (1-dense_weight) * bm25_reciprocal_rank
        bm25_rr = {tid: 1.0 / (r + 1) for r, tid in enumerate(bm25_candidates)}
        dense_score_map = {challenge_ids[i]: float(dense_scores_all[i]) for i in top_dense_indices}

        scored = []
        for tid in all_candidates:
            d_score = dense_score_map.get(tid, 0.0)
            b_score = bm25_rr.get(tid, 0.0)
            combined = args.dense_weight * d_score + (1 - args.dense_weight) * b_score
            scored.append((tid, combined))

        scored.sort(key=lambda x: -x[1])
        predicted_track_ids = [tid for tid, _ in scored[:args.topk]]

        top = predicted_track_ids[0] if predicted_track_ids else ""
        row = metadata_dict.get(top, {})
        name = (row.get("track_name") or ["this track"])[0]
        artist = (row.get("artist_name") or ["the artist"])[0]

        inference_results.append({
            "session_id": session_id,
            "user_id": user_id,
            "turn_number": turn_number,
            "predicted_track_ids": predicted_track_ids,
            "predicted_response": f'I recommend "{name}" by {artist} based on your request.',
        })

        music_in_history.append(turn["content"])

Path(args.out_dir).mkdir(parents=True, exist_ok=True)
out_path = Path(args.out_dir) / f"{args.tid}.json"
with open(out_path, "w") as f:
    json.dump(inference_results, f, ensure_ascii=False, indent=2)

print(f"Saved {len(inference_results):,} predictions to {out_path}")
