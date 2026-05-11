"""
Three-stage retrieval: BM25 → two-tower reranking → cross-encoder reranking.

1. BM25 top-bm25_pool candidates (lexical recall)
2. Two-tower reranks → top-ce_pool (dense recall + rank)
3. Cross-encoder reranks → top-20 (joint query+track scoring)

Usage:
    python scripts/run_inference_twotower_ce.py \
        --model models/twotower_v3/final \
        --ce_model cross-encoder/ms-marco-MiniLM-L-6-v2 \
        --bm25_pool 500 --dense_weight 0.7 --ce_pool 50 \
        --sessions 200 --tid twotower_ce_200
"""
import argparse
import json
import numpy as np
from pathlib import Path

import bm25s
from datasets import load_dataset, concatenate_datasets
from sentence_transformers import SentenceTransformer
from sentence_transformers.cross_encoder import CrossEncoder
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument("--model", default="models/twotower_v3/final")
parser.add_argument("--index_dir", default="cache/twotower_v3")
parser.add_argument("--ce_model", default="cross-encoder/ms-marco-MiniLM-L-6-v2")
parser.add_argument("--sessions", type=int, default=0)
parser.add_argument("--tid", default="twotower_ce")
parser.add_argument("--out_dir", default="exp/inference/devset")
parser.add_argument("--topk", type=int, default=20)
parser.add_argument("--bm25_pool", type=int, default=500)
parser.add_argument("--dense_weight", type=float, default=0.7)
parser.add_argument("--ce_pool", type=int, default=50)
parser.add_argument("--ce_batch_size", type=int, default=32)
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
dense_embeddings = np.load(f"{args.index_dir}/track_embeddings.npy")
with open(f"{args.index_dir}/track_ids.json") as f:
    dense_track_ids = json.load(f)
dense_id_to_idx = {tid: i for i, tid in enumerate(dense_track_ids)}
print(f"  Index: {dense_embeddings.shape}")

print(f"Loading cross-encoder: {args.ce_model}")
ce_model = CrossEncoder(args.ce_model, max_length=512)

print("Loading dev sessions...")
ds = load_dataset("talkpl-ai/TalkPlayData-Challenge-Dataset")
sessions = list(ds["test"])
if args.sessions > 0:
    sessions = sessions[:args.sessions]

print(f"Running {len(sessions)} sessions (bm25={args.bm25_pool}, ce_pool={args.ce_pool})...")
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

        latest_user = text_in_history[-1] if text_in_history else ""
        dense_parts = [latest_user, goal, culture]
        for tid in music_in_history[-2:]:
            na = get_track_name_artist(tid)
            if na:
                dense_parts.append(na)
        compact_query = " ".join(p for p in dense_parts if p)

        bm25_parts = [goal, culture]
        for tid in music_in_history[-args.hist_turns:]:
            bm25_parts.append(get_track_text(tid))
        bm25_parts.extend(text_in_history[-args.text_turns:])
        bm25_query = " ".join(p for p in bm25_parts if p)

        seen = set(music_in_history)

        # Stage 1: BM25 recall
        bm25_cands = retrieve_bm25(bm25_query, topk=args.bm25_pool + len(seen) * 3)
        bm25_cands = [t for t in bm25_cands if t not in seen][:args.bm25_pool]

        if not bm25_cands:
            inference_results.append({
                "session_id": session_id, "user_id": user_id,
                "turn_number": turn_number,
                "predicted_track_ids": [], "predicted_response": "No recommendation.",
            })
            music_in_history.append(turn["content"])
            continue

        # Stage 2: two-tower reranking → top-ce_pool
        query_emb = tower_model.encode(
            compact_query, normalize_embeddings=True, convert_to_numpy=True
        )
        cand_indices = [dense_id_to_idx[t] for t in bm25_cands if t in dense_id_to_idx]
        cands_in_idx = [t for t in bm25_cands if t in dense_id_to_idx]
        cands_not_in_idx = [t for t in bm25_cands if t not in dense_id_to_idx]

        bm25_rr = {tid: 1.0 / (r + 1) for r, tid in enumerate(bm25_cands)}
        scored = []
        if cand_indices:
            cos_scores = dense_embeddings[cand_indices] @ query_emb
            for i, tid in enumerate(cands_in_idx):
                scored.append((tid, args.dense_weight * float(cos_scores[i]) + (1 - args.dense_weight) * bm25_rr[tid]))
        for tid in cands_not_in_idx:
            scored.append((tid, (1 - args.dense_weight) * bm25_rr[tid]))

        scored.sort(key=lambda x: -x[1])
        ce_candidates = [tid for tid, _ in scored[:args.ce_pool]]

        # Stage 3: cross-encoder reranking
        cand_texts = [get_track_text(t) for t in ce_candidates]
        pairs = [(compact_query, text) for text in cand_texts]
        ce_scores = ce_model.predict(pairs, batch_size=args.ce_batch_size)

        ce_scored = sorted(zip(ce_candidates, ce_scores.tolist()), key=lambda x: -x[1])
        predicted = [tid for tid, _ in ce_scored[:args.topk]]

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
