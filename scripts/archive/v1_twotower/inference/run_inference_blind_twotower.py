"""
Blind set inference with two-tower v3 model.

Uses BM25 top-500 recall + two-tower dense reranking.
Format: conversations[-1] is the user query (no ground truth).

Usage:
    python scripts/run_inference_blind_twotower.py \
        --model models/twotower_v3/final \
        --tid blind_a_twotower_v3 \
        --out_dir exp/inference/blind_a
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
parser.add_argument("--dataset", default="talkpl-ai/TalkPlayData-Challenge-Blind-A")
parser.add_argument("--split", default="test")
parser.add_argument("--bm25_pool", type=int, default=500)
parser.add_argument("--dense_weight", type=float, default=0.7)
parser.add_argument("--tid", default="blind_a_twotower_v3")
parser.add_argument("--out_dir", default="exp/inference/blind_a")
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

print(f"Loading {args.dataset} split={args.split}...")
ds = load_dataset(args.dataset)[args.split]
sessions = list(ds)
print(f"Running {len(sessions)} sessions...")

inference_results = []

for item in tqdm(sessions, desc="Sessions"):
    session_id = item["session_id"]
    user_id = item["user_id"]
    goal = item.get("conversation_goal", {}).get("listener_goal", "")
    culture = item.get("user_profile", {}).get("preferred_musical_culture", "")
    conversations = item["conversations"]

    user_query = conversations[-1]["content"]
    turn_number = conversations[-1]["turn_number"]
    history_convs = conversations[:-1]

    music_in_history = []
    text_in_history = []
    for turn in history_convs:
        if turn["role"] == "music":
            music_in_history.append(turn["content"])
        elif turn["role"] in ("user", "assistant"):
            text_in_history.append(turn["content"])

    seen = set(music_in_history)

    # Compact dense query (user request first, matches training)
    dense_parts = [user_query, goal, culture]
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
    bm25_parts.append(user_query)
    bm25_query = " ".join(p for p in bm25_parts if p)

    # BM25 recall
    retrieve_k = args.bm25_pool + len(seen) * 3
    bm25_cands = retrieve_bm25(bm25_query, topk=retrieve_k)
    bm25_cands = [t for t in bm25_cands if t not in seen][:args.bm25_pool]

    if not bm25_cands:
        inference_results.append({
            "session_id": session_id, "user_id": user_id,
            "turn_number": turn_number,
            "predicted_track_ids": [], "predicted_response": "No recommendation.",
        })
        continue

    # Dense scoring
    query_emb = tower_model.encode(dense_query, normalize_embeddings=True, convert_to_numpy=True)
    cand_indices = [dense_id_to_idx[t] for t in bm25_cands if t in dense_id_to_idx]
    cands_in_idx = [t for t in bm25_cands if t in dense_id_to_idx]
    cands_not_in_idx = [t for t in bm25_cands if t not in dense_id_to_idx]

    cos_scores = dense_embs[cand_indices] @ query_emb if cand_indices else np.array([])

    bm25_rr = {tid: 1.0 / (r + 1) for r, tid in enumerate(bm25_cands)}

    scored = []
    for i, tid in enumerate(cands_in_idx):
        cs = float(cos_scores[i]) if i < len(cos_scores) else 0.0
        bs = bm25_rr.get(tid, 0.0)
        scored.append((tid, args.dense_weight * cs + (1 - args.dense_weight) * bs))
    for tid in cands_not_in_idx:
        scored.append((tid, (1 - args.dense_weight) * bm25_rr.get(tid, 0.0)))

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

Path(args.out_dir).mkdir(parents=True, exist_ok=True)
out_path = Path(args.out_dir) / f"{args.tid}.json"
with open(out_path, "w") as f:
    json.dump(inference_results, f, ensure_ascii=False, indent=2)
print(f"Saved {len(inference_results):,} predictions to {out_path}")
