"""
Hybrid recall: BM25 top-K + semantic cluster top-C, reranked by fine-tuned two-tower v3.

Two parallel recall paths:
  1. BM25 top-bm25_pool (lexical recall, full query)
  2. Semantic cluster top-top_clusters (dense recall, compact query)

Both pools merged, scored by: dense_weight * cosine + (1-dense_weight) * bm25_reciprocal_rank
Exclude seen tracks, take top-20.

Usage:
    python scripts/run_inference_hybrid_recall.py \
        --k 300 --top_clusters 5 --bm25_pool 500 --dense_weight 0.7 \
        --tid hybrid_k300_c5_w07 --sessions 0
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
parser.add_argument("--k", type=int, default=300, help="Codebook K value")
parser.add_argument("--top_clusters", type=int, default=5)
parser.add_argument("--bm25_pool", type=int, default=500)
parser.add_argument("--dense_weight", type=float, default=0.7)
parser.add_argument("--rrf", action="store_true", help="Use Reciprocal Rank Fusion instead of weighted sum")
parser.add_argument("--rrf_k", type=int, default=60, help="RRF constant k")
parser.add_argument("--sessions", type=int, default=0)
parser.add_argument("--tid", default="hybrid_k300_c5_w07")
parser.add_argument("--out_dir", default="exp/inference/devset")
parser.add_argument("--topk", type=int, default=20)
parser.add_argument("--hist_turns", type=int, default=4)
parser.add_argument("--text_turns", type=int, default=4)
parser.add_argument("--measure_recall", action="store_true",
                    help="Measure pool recall vs BM25-only (slower)")
args = parser.parse_args()

BM25_CACHE = "cache/bm25/track_metadata"
codebook_dir = f"cache/semantic_codebook_k{args.k}"

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

print(f"Loading semantic codebook K={args.k} from {codebook_dir}...")
centroids = np.load(f"{codebook_dir}/centroids.npy")  # (K, 384), normalized
with open(f"{codebook_dir}/cluster_to_tracks.json") as f:
    cluster_to_tracks = {int(k): v for k, v in json.load(f).items()}
print(f"  Centroids: {centroids.shape}")

print("Loading dev sessions...")
ds = load_dataset("talkpl-ai/TalkPlayData-Challenge-Dataset")
sessions = list(ds["test"])
if args.sessions > 0:
    sessions = sessions[:args.sessions]

print(f"Running {len(sessions)} sessions (K={args.k}, top_clusters={args.top_clusters}, bm25_pool={args.bm25_pool}, w={args.dense_weight})...")
inference_results = []

# Pool recall diagnostics
recall_hits = 0  # gold in merged pool
recall_bm25_only = 0  # gold in BM25-only pool
recall_cluster_only = 0  # gold added by cluster (not in BM25)
total_recall_turns = 0

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
        gold_tid = turn["content"]  # for recall measurement only

        # Compact dense query (user request first, fits in 256 tokens)
        latest_user = text_in_history[-1] if text_in_history else ""
        dense_parts = [latest_user, goal, culture]
        for tid in music_in_history[-2:]:
            na = get_track_name_artist(tid)
            if na:
                dense_parts.append(na)
        dense_query = " ".join(p for p in dense_parts if p)

        # Full BM25 query (tags, longer history — for lexical recall)
        bm25_parts = [goal, culture]
        for tid in music_in_history[-args.hist_turns:]:
            bm25_parts.append(get_track_text(tid))
        bm25_parts.extend(text_in_history[-args.text_turns:])
        bm25_query = " ".join(p for p in bm25_parts if p)

        seen = set(music_in_history)

        # --- Recall path 1: BM25 ---
        retrieve_k = args.bm25_pool + len(seen) * 3
        bm25_cands = retrieve_bm25(bm25_query, topk=retrieve_k)
        bm25_cands = [t for t in bm25_cands if t not in seen][:args.bm25_pool]
        bm25_set = set(bm25_cands)

        # --- Recall path 2: Semantic clusters ---
        query_emb = tower_model.encode(dense_query, normalize_embeddings=True, convert_to_numpy=True)
        cluster_scores = centroids @ query_emb  # (K,)
        top_cluster_ids = np.argsort(-cluster_scores)[:args.top_clusters]

        cluster_cands = []
        cluster_only_cands = []
        for cid in top_cluster_ids:
            for tid in cluster_to_tracks[int(cid)]:
                if tid not in seen and tid not in bm25_set:
                    cluster_only_cands.append(tid)
                elif tid not in seen and tid in bm25_set:
                    pass  # already in BM25 pool
        # Deduplicate cluster-only candidates
        seen_cluster = set()
        cluster_only_deduped = []
        for tid in cluster_only_cands:
            if tid not in seen_cluster:
                seen_cluster.add(tid)
                cluster_only_deduped.append(tid)

        # Merged candidate pool
        all_cands = bm25_cands + cluster_only_deduped

        # --- Pool recall measurement ---
        if args.measure_recall:
            total_recall_turns += 1
            if gold_tid in bm25_set:
                recall_bm25_only += 1
            if gold_tid in set(all_cands):
                recall_hits += 1
            if gold_tid in set(cluster_only_deduped):
                recall_cluster_only += 1

        if not all_cands:
            inference_results.append({
                "session_id": session_id, "user_id": user_id,
                "turn_number": turn_number,
                "predicted_track_ids": [], "predicted_response": "No recommendation.",
            })
            music_in_history.append(gold_tid)
            continue

        # --- Score all candidates ---
        # Cosine similarity
        cand_indices = [dense_id_to_idx[t] for t in all_cands if t in dense_id_to_idx]
        cands_in_idx = [t for t in all_cands if t in dense_id_to_idx]
        cands_not_in_idx = [t for t in all_cands if t not in dense_id_to_idx]

        if cand_indices:
            cos_scores = dense_embs[cand_indices] @ query_emb
        else:
            cos_scores = np.array([])

        # BM25 reciprocal rank (0 for cluster-only candidates in weighted mode)
        bm25_rank = {tid: r for r, tid in enumerate(bm25_cands)}

        if args.rrf:
            # Reciprocal Rank Fusion: rank all candidates by cosine, combine with BM25 rank
            cos_map = {tid: float(cos_scores[i]) for i, tid in enumerate(cands_in_idx)}
            # Sort all_cands by cosine descending to get dense rank
            dense_sorted = sorted(all_cands, key=lambda t: -cos_map.get(t, -1.0))
            dense_rank = {tid: r for r, tid in enumerate(dense_sorted)}
            k = args.rrf_k
            scored = []
            for tid in all_cands:
                dr = dense_rank.get(tid, len(all_cands))
                br = bm25_rank.get(tid, len(all_cands))
                rrf_score = 1.0 / (k + dr) + 1.0 / (k + br)
                scored.append((tid, rrf_score))
        else:
            bm25_rr = {tid: 1.0 / (r + 1) for r, tid in bm25_rank.items()}
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
        music_in_history.append(gold_tid)

Path(args.out_dir).mkdir(parents=True, exist_ok=True)
out_path = Path(args.out_dir) / f"{args.tid}.json"
with open(out_path, "w") as f:
    json.dump(inference_results, f, ensure_ascii=False, indent=2)
print(f"Saved {len(inference_results):,} predictions to {out_path}")

if args.measure_recall and total_recall_turns > 0:
    print(f"\nPOOL RECALL ({total_recall_turns} turns):")
    print(f"  BM25-only pool:    {100*recall_bm25_only/total_recall_turns:.1f}%")
    print(f"  Merged pool:       {100*recall_hits/total_recall_turns:.1f}%")
    print(f"  Added by clusters: {100*recall_cluster_only/total_recall_turns:.1f}%")
