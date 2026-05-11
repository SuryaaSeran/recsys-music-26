"""
Dense hybrid retrieval: BM25 candidates reranked by sentence-transformer similarity.

For each turn:
  1. BM25 top-K candidates from track metadata
  2. Encode query with all-MiniLM-L6-v2
  3. Cosine similarity against prebuilt track embeddings
  4. Combine BM25 rank score + dense similarity score

Usage:
    python scripts/run_inference_dense_hybrid.py [--sessions 0] [--tid dense_hybrid_v1]
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
parser.add_argument("--tid", default="dense_hybrid_v1")
parser.add_argument("--split", default="test")
parser.add_argument("--out_dir", default="exp/inference/devset")
parser.add_argument("--topk", type=int, default=20)
parser.add_argument("--bm25_candidates", type=int, default=200)
parser.add_argument("--dense_weight", type=float, default=0.5, help="Dense score weight (0=BM25 only, 1=dense only)")
parser.add_argument("--query_strategy", default="goal_culture_query_x2",
                    help="query_only|goal_query|goal_culture_query|goal_culture_query_x2|all_context")
args = parser.parse_args()

# ── Track metadata ────────────────────────────────────────────────────────────
print("Loading track metadata...")
meta_ds = load_dataset("talkpl-ai/TalkPlayData-Challenge-Track-Metadata")
all_tracks = concatenate_datasets([meta_ds["all_tracks"], meta_ds["test_tracks"]])
metadata_dict = {row["track_id"]: row for row in all_tracks}

# ── BM25 index ────────────────────────────────────────────────────────────────
CACHE_PATH = "cache/bm25/track_metadata"
print("Loading BM25 index...")
bm25_model = bm25s.BM25.load(CACHE_PATH, load_corpus=False)
with open(f"{CACHE_PATH}/track_ids.json") as f:
    bm25_track_ids = json.load(f)

# ── Dense index ───────────────────────────────────────────────────────────────
DENSE_DIR = Path("cache/dense")
print("Loading dense embeddings...")
track_embeddings = np.load(DENSE_DIR / "track_embeddings.npy")  # (N, 384), L2-normalized
with open(DENSE_DIR / "track_ids.json") as f:
    dense_track_ids = json.load(f)
dense_idx_map = {tid: i for i, tid in enumerate(dense_track_ids)}

print(f"Dense index: {track_embeddings.shape}")

# ── Sentence transformer ──────────────────────────────────────────────────────
print("Loading sentence transformer...")
st_model = SentenceTransformer("all-MiniLM-L6-v2")

# ── Dataset ───────────────────────────────────────────────────────────────────
print(f"Loading dataset split={args.split}...")
ds = load_dataset("talkpl-ai/TalkPlayData-Challenge-Dataset")[args.split]
sessions = list(ds)
if args.sessions > 0:
    sessions = sessions[:args.sessions]


def build_query(goal, culture, history, user_query):
    s = args.query_strategy
    if s == "query_only":
        return user_query
    elif s == "query_x3":
        return f"{user_query} {user_query} {user_query}"
    elif s == "goal_query":
        return f"{goal} {user_query}"
    elif s == "goal_culture_query":
        return f"{goal} {culture} {user_query}"
    elif s == "goal_culture_query_x2":
        return f"{goal} {culture} {user_query} {user_query}"
    elif s == "all_context":
        hist_text = " ".join(h["content"] for h in history[-4:])
        return f"{goal} {culture} {hist_text} {user_query}"
    elif s == "user_turns_only":
        user_hist = " ".join(h["content"] for h in history if h["role"] == "user")
        return f"{goal} {user_hist} {user_query}"
    else:
        return f"{goal} {culture} {user_query} {user_query}"


def retrieve_bm25_candidates(query: str, k: int) -> list[str]:
    tokens = bm25s.tokenize([query.lower()])
    results = bm25_model.retrieve(tokens, k=k, return_as="tuple")
    return [bm25_track_ids[int(i)] for i in results.documents[0]]


def hybrid_retrieve(query: str, topk: int = 20) -> list[str]:
    candidates = retrieve_bm25_candidates(query, k=args.bm25_candidates)

    # BM25 rank-based scores
    n = len(candidates)
    bm25_scores = {tid: (n - i) / n for i, tid in enumerate(candidates)}

    if args.dense_weight == 0:
        return candidates[:topk]

    # Dense query embedding
    q_emb = st_model.encode(query, normalize_embeddings=True)  # (384,)

    # Dense scores for candidates
    dense_scores = {}
    for tid in candidates:
        idx = dense_idx_map.get(tid)
        if idx is not None:
            dense_scores[tid] = float(np.dot(q_emb, track_embeddings[idx]))
        else:
            dense_scores[tid] = 0.0

    # Normalize dense scores to [0, 1]
    vals = list(dense_scores.values())
    d_min, d_max = min(vals), max(vals)
    d_range = d_max - d_min + 1e-8
    dense_scores = {k: (v - d_min) / d_range for k, v in dense_scores.items()}

    w = args.dense_weight
    combined = {
        tid: (1 - w) * bm25_scores[tid] + w * dense_scores.get(tid, 0.0)
        for tid in candidates
    }
    ranked = sorted(combined, key=lambda t: combined[t], reverse=True)
    return ranked[:topk]


# ── Inference ─────────────────────────────────────────────────────────────────
print(f"Running inference on {len(sessions)} sessions (dense_weight={args.dense_weight}, strategy={args.query_strategy})...")
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

        history = []
        for turn in conversations:
            if turn["turn_number"] >= target_turn:
                break
            if turn["role"] == "music":
                row = metadata_dict.get(turn["content"], {})
                name = (row.get("track_name") or [""])[0]
                artist = (row.get("artist_name") or [""])[0]
                history.append({"role": "assistant", "content": f"{name} {artist}"})
            else:
                history.append({"role": turn["role"], "content": turn["content"]})

        query = build_query(goal, culture, history, user_query)
        predicted_track_ids = hybrid_retrieve(query, topk=args.topk)

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
