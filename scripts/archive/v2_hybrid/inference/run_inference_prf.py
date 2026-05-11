"""
BM25 + dense retrieval with pseudo-relevance feedback (PRF).

1. BM25 top-bm25_pool candidates (standard)
2. Dense top-3 from BM25 pool → extract tags
3. Append tags to BM25 query → re-retrieve BM25 top-prf_expand
4. Merge original + new BM25 candidates (up to bm25_pool + prf_expand)
5. Dense rerank merged pool → hybrid score → top-20

Goal: improve pool recall by using tag expansion from top-ranked candidates.

Usage:
    python scripts/run_inference_prf.py \
        --model models/twotower_v3/final \
        --sessions 200 --tid prf_v1_200
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
parser.add_argument("--sessions", type=int, default=0)
parser.add_argument("--tid", default="prf_v1")
parser.add_argument("--out_dir", default="exp/inference/devset")
parser.add_argument("--topk", type=int, default=20)
parser.add_argument("--bm25_pool", type=int, default=500)
parser.add_argument("--prf_top", type=int, default=3, help="Dense top-K from BM25 pool used for tag expansion")
parser.add_argument("--prf_expand", type=int, default=200, help="Additional BM25 candidates from expanded query")
parser.add_argument("--dense_weight", type=float, default=0.7)
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


def get_track_tags(tid):
    row = metadata_dict.get(tid, {})
    return " ".join(row.get("tag_list") or [])


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

print("Loading dev sessions...")
ds = load_dataset("talkpl-ai/TalkPlayData-Challenge-Dataset")
sessions = list(ds["test"])
if args.sessions > 0:
    sessions = sessions[:args.sessions]

print(f"Running {len(sessions)} sessions (bm25={args.bm25_pool}, prf_top={args.prf_top}, prf_expand={args.prf_expand})...")
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
        dense_query = " ".join(p for p in dense_parts if p)

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

        # Stage 2: Dense rerank BM25 → pick top-prf_top for PRF
        query_emb = tower_model.encode(
            dense_query, normalize_embeddings=True, convert_to_numpy=True
        )
        cand_indices = [dense_id_to_idx[t] for t in bm25_cands if t in dense_id_to_idx]
        cands_in_idx = [t for t in bm25_cands if t in dense_id_to_idx]

        if cand_indices:
            cos_scores = dense_embeddings[cand_indices] @ query_emb
            top_indices = np.argsort(cos_scores)[::-1][:args.prf_top]
            prf_tids = [cands_in_idx[i] for i in top_indices]
        else:
            prf_tids = []

        # Stage 3: PRF expansion - append tags from top candidates to BM25 query
        prf_tags = " ".join(get_track_tags(t) for t in prf_tids if get_track_tags(t))
        if prf_tags:
            expanded_bm25_query = bm25_query + " " + prf_tags
            extra_cands = retrieve_bm25(expanded_bm25_query, topk=args.prf_expand + len(seen) * 3)
            extra_cands = [t for t in extra_cands if t not in seen][:args.prf_expand]
            # Merge: original BM25 + new candidates not already in pool
            bm25_set = set(bm25_cands)
            merged_cands = bm25_cands + [t for t in extra_cands if t not in bm25_set]
        else:
            merged_cands = bm25_cands

        # Stage 4: Dense rerank merged pool → hybrid score
        all_indices = [dense_id_to_idx[t] for t in merged_cands if t in dense_id_to_idx]
        all_in_idx = [t for t in merged_cands if t in dense_id_to_idx]
        all_not_in_idx = [t for t in merged_cands if t not in dense_id_to_idx]

        if all_indices:
            all_cos = dense_embeddings[all_indices] @ query_emb
        else:
            all_cos = np.array([])

        # BM25 RR only for original BM25 candidates (merged candidates have no BM25 rank)
        bm25_rr = {tid: 1.0 / (r + 1) for r, tid in enumerate(bm25_cands)}

        scored = []
        for i, tid in enumerate(all_in_idx):
            dense_s = float(all_cos[i]) if i < len(all_cos) else 0.0
            bm25_s = bm25_rr.get(tid, 0.0)
            scored.append((tid, args.dense_weight * dense_s + (1 - args.dense_weight) * bm25_s))
        for tid in all_not_in_idx:
            bm25_s = bm25_rr.get(tid, 0.0)
            scored.append((tid, (1 - args.dense_weight) * bm25_s))

        scored.sort(key=lambda x: -x[1])
        predicted_track_ids = [tid for tid, _ in scored[:args.topk]]

        top = predicted_track_ids[0] if predicted_track_ids else ""
        row = metadata_dict.get(top, {})
        name = (row.get("track_name") or ["this track"])[0]
        artist = (row.get("artist_name") or ["the artist"])[0]

        inference_results.append({
            "session_id": session_id, "user_id": user_id,
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
