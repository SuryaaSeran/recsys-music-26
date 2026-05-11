"""
Phase 1 diagnostic: Can semantic ID buckets recall gold tracks?

Measures three strategies:
  1. Oracle pair: use gold (coarse,fine) + neighbor expansion
  2. BM25-derived pair: use majority code among BM25 top-K tracks
  3. Random pair: baseline

Also reports gold rank within its bucket by popularity and BM25.

Usage:
    python scripts/eval_semantic_candidate_recall.py --sessions 200
"""
import argparse
import json
import pickle
import numpy as np
from collections import Counter, defaultdict
from pathlib import Path

import bm25s
from datasets import load_dataset, concatenate_datasets

parser = argparse.ArgumentParser()
parser.add_argument("--sessions", type=int, default=200)
parser.add_argument("--bm25_topk", type=int, default=100, help="BM25 candidates for code derivation")
parser.add_argument("--n_coarse_neighbors", type=int, default=5)
parser.add_argument("--n_fine_neighbors", type=int, default=5)
args = parser.parse_args()

CACHE_PATH = "cache/bm25/track_metadata"

print("Loading codebook...")
with open("data/codebook_v2.pkl", "rb") as f:
    cb = pickle.load(f)

track_to_codes = cb["track_to_codes"]
codes_to_tracks = cb["codes_to_tracks"]
km1 = cb["km1"]
km2 = cb["km2"]

# Precompute coarse and fine neighbor lists (by centroid distance)
coarse_centers = km1.cluster_centers_  # (128, 256)
fine_centers = km2.cluster_centers_    # (128, 256)

from sklearn.metrics.pairwise import cosine_similarity

coarse_sim = cosine_similarity(coarse_centers)
fine_sim = cosine_similarity(fine_centers)

# For each coarse cluster, sorted neighbors (excluding self)
coarse_neighbors = {}
for c in range(128):
    order = np.argsort(-coarse_sim[c])
    coarse_neighbors[c] = [x for x in order if x != c]

fine_neighbors = {}
for f in range(128):
    order = np.argsort(-fine_sim[f])
    fine_neighbors[f] = [x for x in order if x != f]


def expand_candidates(pairs: list[tuple], n_coarse_nb: int, n_fine_nb: int) -> list[str]:
    """Expand a list of (coarse,fine) pairs with neighbors, return track IDs."""
    expanded_pairs = set()
    for c, f in pairs:
        expanded_pairs.add((c, f))
        for nc in coarse_neighbors[c][:n_coarse_nb]:
            expanded_pairs.add((nc, f))
        for nf in fine_neighbors[f][:n_fine_nb]:
            expanded_pairs.add((c, nf))
    tracks = []
    seen = set()
    for pair in expanded_pairs:
        for tid in codes_to_tracks.get(pair, []):
            if tid not in seen:
                seen.add(tid)
                tracks.append(tid)
    return tracks


print("Loading track metadata...")
meta_ds = load_dataset("talkpl-ai/TalkPlayData-Challenge-Track-Metadata")
all_tracks = concatenate_datasets([meta_ds["all_tracks"], meta_ds["test_tracks"]])
metadata_dict = {row["track_id"]: row for row in all_tracks}

# Popularity proxy: number of tags (more tags = more popular/described)
track_popularity = {
    tid: len(row.get("tag_list") or [])
    for tid, row in metadata_dict.items()
}


def get_track_text(tid: str) -> str:
    row = metadata_dict.get(tid, {})
    name = (row.get("track_name") or [""])[0]
    artist = (row.get("artist_name") or [""])[0]
    tags = " ".join(row.get("tag_list") or [])
    return f"{name} {artist} {tags}".strip()


print("Loading BM25 index...")
bm25_model = bm25s.BM25.load(CACHE_PATH, load_corpus=False)
with open(f"{CACHE_PATH}/track_ids.json") as f:
    track_ids_list = json.load(f)


def retrieve_bm25(query: str, topk: int) -> list[str]:
    tokens = bm25s.tokenize([query.lower()])
    results = bm25_model.retrieve(tokens, k=topk, return_as="tuple")
    return [track_ids_list[int(i)] for i in results.documents[0]]


print("Loading conversations...")
ds = load_dataset("talkpl-ai/TalkPlayData-Challenge-Dataset")
sessions = list(ds["test"])
if args.sessions > 0:
    sessions = sessions[:args.sessions]

# ---- Metrics accumulators ----
oracle_in_bucket = 0        # gold track in gold pair bucket (should be ~100%)
oracle_in_expanded = 0      # gold track in expanded candidate pool
oracle_pool_sizes = []

bm25_code_hits = 0          # BM25-derived code matches gold code
bm25_derived_in_expanded = 0

total_turns = 0
not_in_codebook = 0

gold_rank_by_popularity = []  # rank of gold within its bucket
gold_rank_by_bm25 = []

bucket_sizes = []

print(f"Running diagnostics on {len(sessions)} sessions...")

for item in sessions:
    goal = item.get("conversation_goal", {}).get("listener_goal", "")
    culture = item.get("user_profile", {}).get("preferred_musical_culture", "")
    conversations = item["conversations"]

    music_in_history = []
    text_in_history = []

    for turn in conversations:
        if turn["role"] == "music":
            gold_tid = turn["content"]
            total_turns += 1

            # Check codebook coverage
            if gold_tid not in track_to_codes:
                not_in_codebook += 1
                music_in_history.append(gold_tid)
                continue

            gold_code = track_to_codes[gold_tid]  # (coarse, fine)

            # Build BM25 query (same as run_inference_bm25_tagexpand.py)
            history_parts = [goal, culture]
            for tid in music_in_history[-4:]:
                history_parts.append(get_track_text(tid))
            history_parts.extend(text_in_history[-4:])
            # user turn text not available here (we predict before seeing it)
            # use the most recent assistant response as proxy
            query = " ".join(p for p in history_parts if p) or "music"

            # Oracle: expand from gold code
            bucket_tracks = codes_to_tracks.get(gold_code, [])
            bucket_sizes.append(len(bucket_tracks))
            oracle_in_bucket += (gold_tid in bucket_tracks)

            expanded = expand_candidates(
                [gold_code],
                args.n_coarse_neighbors,
                args.n_fine_neighbors,
            )
            oracle_in_expanded += (gold_tid in expanded)
            oracle_pool_sizes.append(len(expanded))

            # Gold rank within bucket by popularity (lower is better)
            bucket_sorted_pop = sorted(
                bucket_tracks,
                key=lambda t: track_popularity.get(t, 0),
                reverse=True,
            )
            if gold_tid in bucket_sorted_pop:
                rank = bucket_sorted_pop.index(gold_tid) + 1
                gold_rank_by_popularity.append(rank)

            # BM25-derived code: majority code among BM25 top-K results
            bm25_results = retrieve_bm25(query, topk=args.bm25_topk)
            code_votes = Counter(
                track_to_codes[t]
                for t in bm25_results
                if t in track_to_codes
            )
            if code_votes:
                bm25_pred_code = code_votes.most_common(1)[0][0]
                if bm25_pred_code == gold_code:
                    bm25_code_hits += 1

                bm25_expanded = expand_candidates(
                    [bm25_pred_code],
                    args.n_coarse_neighbors,
                    args.n_fine_neighbors,
                )
                bm25_derived_in_expanded += (gold_tid in bm25_expanded)

            # Gold rank within bucket by BM25 score
            bucket_bm25_ranked = [t for t in bm25_results if t in bucket_tracks]
            remaining = [t for t in bucket_tracks if t not in bucket_bm25_ranked]
            bucket_bm25_order = bucket_bm25_ranked + remaining
            if gold_tid in bucket_bm25_order:
                rank_bm25 = bucket_bm25_order.index(gold_tid) + 1
                gold_rank_by_bm25.append(rank_bm25)

            music_in_history.append(gold_tid)

        elif turn["role"] in ("user", "assistant"):
            text_in_history.append(turn["content"])

# ---- Report ----
print(f"\n{'='*60}")
print(f"SEMANTIC ID CANDIDATE RECALL DIAGNOSTIC")
print(f"{'='*60}")
print(f"Sessions: {len(sessions)}  |  Turns: {total_turns}  |  Not in codebook: {not_in_codebook} ({100*not_in_codebook/max(total_turns,1):.1f}%)")
print()
print(f"CODEBOOK BUCKET SIZES (for gold tracks):")
if bucket_sizes:
    bs = sorted(bucket_sizes)
    n = len(bs)
    print(f"  min={bs[0]}, median={bs[n//2]}, p90={bs[int(n*0.9)]}, max={bs[-1]}")
    print(f"  avg={np.mean(bs):.1f}")
print()
print(f"ORACLE STRATEGY (gold (coarse,fine) pair + {args.n_coarse_neighbors} coarse + {args.n_fine_neighbors} fine neighbors):")
valid = total_turns - not_in_codebook
if valid > 0:
    print(f"  Gold track in own bucket: {oracle_in_bucket}/{valid} = {100*oracle_in_bucket/valid:.1f}%")
    print(f"  Gold track in expanded pool: {oracle_in_expanded}/{valid} = {100*oracle_in_expanded/valid:.1f}%")
    if oracle_pool_sizes:
        ps = sorted(oracle_pool_sizes)
        print(f"  Pool size: min={ps[0]}, median={ps[len(ps)//2]}, p90={ps[int(len(ps)*0.9)]}, max={ps[-1]}, avg={np.mean(ps):.0f}")
print()
print(f"BM25-DERIVED CODE STRATEGY (majority code from BM25 top-{args.bm25_topk}):")
if valid > 0:
    print(f"  BM25 code == gold code: {bm25_code_hits}/{valid} = {100*bm25_code_hits/valid:.1f}%")
    print(f"  Gold track in BM25-derived expanded pool: {bm25_derived_in_expanded}/{valid} = {100*bm25_derived_in_expanded/valid:.1f}%")
print()
print(f"GOLD RANK WITHIN BUCKET:")
if gold_rank_by_popularity:
    rp = sorted(gold_rank_by_popularity)
    print(f"  By popularity: median={rp[len(rp)//2]}, p90={rp[int(len(rp)*0.9)]}, mean={np.mean(rp):.1f}")
if gold_rank_by_bm25:
    rb = sorted(gold_rank_by_bm25)
    print(f"  By BM25 order: median={rb[len(rb)//2]}, p90={rb[int(len(rb)*0.9)]}, mean={np.mean(rb):.1f}")
print()
print(f"GATE CHECK:")
print(f"  Required: oracle_in_expanded >= 95%  |  Got: {100*oracle_in_expanded/max(valid,1):.1f}%")
print(f"  Required: bm25_code_hit >= 5%       |  Got: {100*bm25_code_hits/max(valid,1):.1f}%")
print()
if oracle_in_expanded / max(valid, 1) < 0.70:
    print("VERDICT: FAIL. Oracle recall < 70%. Semantic IDs cannot work with this codebook.")
elif bm25_code_hits / max(valid, 1) < 0.05:
    print("VERDICT: FAIL. BM25-derived code accuracy < 5%. No reliable code prediction method found.")
else:
    print("VERDICT: PASS. Proceed to Phase 2 (multi-code SFT training).")
