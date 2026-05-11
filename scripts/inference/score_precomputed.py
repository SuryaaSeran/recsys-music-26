"""
Fast weight-tuning using precomputed embeddings.

Computes all signal scores once, then evaluates any weight combination in seconds.

Usage:
    # Single eval
    python scripts/score_precomputed.py \
        --w_tt 0.35 --w_cf 0.12 --w_qwen_meta 0.30 --w_clap 0.10 --w_bm25 0.13

    # Grid search over weights
    python scripts/score_precomputed.py --grid_search [--bm25_norm]
"""
import argparse
import json
import numpy as np
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("--cache_dir",    default="cache/dev_embeddings")
parser.add_argument("--w_tt",         type=float, default=0.35)
parser.add_argument("--w_cf",         type=float, default=0.12)
parser.add_argument("--w_qwen_meta",  type=float, default=0.30)
parser.add_argument("--w_clap",       type=float, default=0.10)
parser.add_argument("--w_bm25",       type=float, default=0.13)
parser.add_argument("--w_attrs_hist",   type=float, default=0.0)
parser.add_argument("--w_qwen_lyrics", type=float, default=0.0)
parser.add_argument("--bm25_norm",    action="store_true")
parser.add_argument("--grid_search",  action="store_true")
parser.add_argument("--topk",         type=int,   default=20)
args = parser.parse_args()

CACHE = Path(args.cache_dir)

print("Loading precomputed turn data...")
with open(CACHE / "turns.json") as f:
    turns_meta = json.load(f)

bm25_cands  = np.load(CACHE / "bm25_cands.npy")    # (N, 500) int32: BM25 track indices
bm25_scores = np.load(CACHE / "bm25_scores.npy")   # (N, 500) float32: raw BM25 scores
tt_q        = np.load(CACHE / "tt_embs.npy")        # (N, 384)
qwen_q      = np.load(CACHE / "qwen_embs.npy")      # (N, 1024)
clap_q      = np.load(CACHE / "clap_embs.npy")      # (N, 512)
cf_u        = np.load(CACHE / "cf_user.npy")        # (N, 128)
ah_q        = np.load(CACHE / "attrs_hist.npy")     # (N, 1024) avg of last 4 played track attr embs

N, BM25_POOL = bm25_cands.shape
has_cf = np.array([m["has_cf"] for m in turns_meta], dtype=bool)

bm25_valid = bm25_cands >= 0  # (N, 500)

print("Loading track embeddings...")
tt_trk     = np.load("cache/twotower_v3/track_embeddings.npy")    # (M, 384)
qwen_trk   = np.load("cache/qwen3_meta/track_embeddings.npy")     # (M, 1024)
lyrics_trk = np.load("cache/qwen3_lyrics/track_embeddings.npy")   # (M, 1024)
attr_trk   = np.load("cache/qwen3_attr/track_embeddings.npy")     # (M, 1024)
clap_trk   = np.load("cache/clap/track_embeddings.npy")           # (M, 512)
cf_trk   = np.load("cache/cf_bpr/track_embeddings.npy")         # (M, 128)

# Build cross-index mappings: bm25 index → per-signal index
with open("cache/bm25/track_metadata/track_ids.json") as f:
    bm25_ids = json.load(f)
with open("cache/twotower_v3/track_ids.json") as f:
    tt_ids = json.load(f)
with open("cache/qwen3_meta/track_ids.json") as f:
    qwen_ids = json.load(f)
with open("cache/qwen3_lyrics/track_ids.json") as f:
    lyrics_ids = json.load(f)
with open("cache/qwen3_attr/track_ids.json") as f:
    attr_ids = json.load(f)
with open("cache/clap/track_ids.json") as f:
    clap_ids = json.load(f)
with open("cache/cf_bpr/track_ids.json") as f:
    cf_ids = json.load(f)

tt_id2idx     = {t: i for i, t in enumerate(tt_ids)}
qwen_id2idx   = {t: i for i, t in enumerate(qwen_ids)}
lyrics_id2idx = {t: i for i, t in enumerate(lyrics_ids)}
attr_id2idx   = {t: i for i, t in enumerate(attr_ids)}
clap_id2idx   = {t: i for i, t in enumerate(clap_ids)}
cf_id2idx     = {t: i for i, t in enumerate(cf_ids)}

# Map BM25 track indices to each signal's track indices
bm25_to_tt     = np.array([tt_id2idx.get(t, -1)     for t in bm25_ids], dtype=np.int32)
bm25_to_qwen   = np.array([qwen_id2idx.get(t, -1)   for t in bm25_ids], dtype=np.int32)
bm25_to_lyrics = np.array([lyrics_id2idx.get(t, -1) for t in bm25_ids], dtype=np.int32)
bm25_to_attr   = np.array([attr_id2idx.get(t, -1)   for t in bm25_ids], dtype=np.int32)
bm25_to_clap   = np.array([clap_id2idx.get(t, -1)   for t in bm25_ids], dtype=np.int32)
bm25_to_cf     = np.array([cf_id2idx.get(t, -1)     for t in bm25_ids], dtype=np.int32)

# Gold track positions in bm25_cands
bm25_id_to_pos = {tid: i for i, tid in enumerate(bm25_ids)}
gold_bm25_idx = np.array([bm25_id_to_pos.get(m["gold"], -1) for m in turns_meta], dtype=np.int32)

# Vectorized: find position of gold in each turn's bm25_cands
expanded_gold = gold_bm25_idx[:, np.newaxis]    # (N, 1)
matches = (bm25_cands == expanded_gold) & bm25_valid  # (N, 500) bool
has_match = matches.any(axis=1)                  # (N,)
first_match = matches.argmax(axis=1)             # (N,) - col index of first match
gold_in_cands = np.where(has_match, first_match, -1).astype(np.int32)

turn_idx = np.arange(N)

# Compute all candidate signal scores (N, 500) for each signal -- done once
print("Computing TT candidate scores...")
safe_cands_bm25 = np.where(bm25_valid, bm25_cands, 0)
cand_tt_idx = bm25_to_tt[safe_cands_bm25]   # (N, 500)
cand_tt_valid = (cand_tt_idx >= 0) & bm25_valid
safe_tt = np.where(cand_tt_valid, cand_tt_idx, 0)
# (N, 500, 384) @ (N, 384) → batch dot: use einsum for memory efficiency
# Process in batches of 500 turns to avoid OOM
BATCH = 500
tt_cos = np.zeros((N, BM25_POOL), dtype=np.float32)
for b in range(0, N, BATCH):
    e = min(b + BATCH, N)
    tt_cos[b:e] = (tt_trk[safe_tt[b:e]] * tt_q[b:e, np.newaxis, :]).sum(-1)
tt_cos = np.where(cand_tt_valid, tt_cos, 0.0)

print("Computing Qwen3 candidate scores...")
cand_qwen_idx = bm25_to_qwen[safe_cands_bm25]
cand_qwen_valid = (cand_qwen_idx >= 0) & bm25_valid
safe_qwen = np.where(cand_qwen_valid, cand_qwen_idx, 0)
qm_cos = np.zeros((N, BM25_POOL), dtype=np.float32)
for b in range(0, N, BATCH):
    e = min(b + BATCH, N)
    qm_cos[b:e] = (qwen_trk[safe_qwen[b:e]] * qwen_q[b:e, np.newaxis, :]).sum(-1)
qm_cos = np.where(cand_qwen_valid, qm_cos, 0.0)

print("Computing Qwen3 lyrics candidate scores...")
cand_lyrics_idx = bm25_to_lyrics[safe_cands_bm25]
cand_lyrics_valid = (cand_lyrics_idx >= 0) & bm25_valid
safe_lyrics = np.where(cand_lyrics_valid, cand_lyrics_idx, 0)
lyrics_cos = np.zeros((N, BM25_POOL), dtype=np.float32)
for b in range(0, N, BATCH):
    e = min(b + BATCH, N)
    lyrics_cos[b:e] = (lyrics_trk[safe_lyrics[b:e]] * qwen_q[b:e, np.newaxis, :]).sum(-1)
lyrics_cos = np.where(cand_lyrics_valid, lyrics_cos, 0.0)

print("Computing attrs_hist (style) candidate scores...")
cand_attr_idx = bm25_to_attr[safe_cands_bm25]
# Only score when there's a non-zero history vector for the turn
ah_q_norm = np.linalg.norm(ah_q, axis=1)
turn_has_hist = ah_q_norm > 1e-6
cand_attr_valid = (cand_attr_idx >= 0) & bm25_valid & turn_has_hist[:, np.newaxis]
safe_attr = np.where(cand_attr_valid, cand_attr_idx, 0)
ah_cos = np.zeros((N, BM25_POOL), dtype=np.float32)
for b in range(0, N, BATCH):
    e = min(b + BATCH, N)
    ah_cos[b:e] = (attr_trk[safe_attr[b:e]] * ah_q[b:e, np.newaxis, :]).sum(-1)
ah_cos = np.where(cand_attr_valid, ah_cos, 0.0)

print("Computing CLAP candidate scores...")
cand_clap_idx = bm25_to_clap[safe_cands_bm25]
cand_clap_valid = (cand_clap_idx >= 0) & bm25_valid
safe_clap = np.where(cand_clap_valid, cand_clap_idx, 0)
clap_cos = np.zeros((N, BM25_POOL), dtype=np.float32)
for b in range(0, N, BATCH):
    e = min(b + BATCH, N)
    clap_cos[b:e] = (clap_trk[safe_clap[b:e]] * clap_q[b:e, np.newaxis, :]).sum(-1)
clap_cos = np.where(cand_clap_valid, clap_cos, 0.0)

print("Computing CF candidate scores...")
cand_cf_idx = bm25_to_cf[safe_cands_bm25]
cand_cf_valid = (cand_cf_idx >= 0) & bm25_valid & has_cf[:, np.newaxis]
safe_cf = np.where(cand_cf_valid, cand_cf_idx, 0)
cf_cos = np.zeros((N, BM25_POOL), dtype=np.float32)
for b in range(0, N, BATCH):
    e = min(b + BATCH, N)
    cf_cos[b:e] = (cf_trk[safe_cf[b:e]] * cf_u[b:e, np.newaxis, :]).sum(-1)
cf_cos = np.where(cand_cf_valid, cf_cos, 0.0)

# BM25 signal (precompute base)
ranks_1d = np.arange(BM25_POOL, dtype=np.float32)
bm25_rr = np.where(bm25_valid, 1.0 / (ranks_1d + 1.0), 0.0)  # (N, 500)

max_s = bm25_scores[:, 0:1].copy()
max_s[max_s < 1e-8] = 1.0
bm25_norm_sig = np.where(bm25_valid, bm25_scores / max_s, 0.0)

# Vectorized evaluation helper
valid_mask  = gold_in_cands >= 0  # turns where gold is in BM25 pool
safe_gpos   = np.where(valid_mask, gold_in_cands, 0)
arange_n    = np.arange(N)
topk        = args.topk
log2_ranks  = None  # computed lazily

def evaluate(w_tt, w_cf, w_qm, w_clap, w_bm25, w_lyrics=0.0, w_ah=0.0, bm25_norm=False):
    bm25_sig = bm25_norm_sig if bm25_norm else bm25_rr
    total = w_tt*tt_cos + w_qm*qm_cos + w_clap*clap_cos + w_cf*cf_cos + w_bm25*bm25_sig
    if w_lyrics > 0:
        total = total + w_lyrics*lyrics_cos
    if w_ah > 0:
        total = total + w_ah*ah_cos
    gold_s = total[arange_n, safe_gpos]  # (N,) — garbage where not valid
    # Count how many candidates score strictly higher than gold (= rank - 1)
    ranks = (total[valid_mask] > gold_s[valid_mask, np.newaxis]).sum(axis=1) + 1  # (N_valid,)
    hit_mask = ranks <= topk
    ndcg = float(np.where(hit_mask, 1.0 / np.log2(ranks + 1), 0.0).sum()) / N
    hit_rate = float(hit_mask.sum()) / N
    return ndcg, hit_rate


if args.grid_search:
    print("Grid searching weights (refined + attrs_hist, normalized BM25 only)...")
    results = []
    for w_tt in [0.28, 0.32, 0.36]:
        for w_qm in [0.32, 0.36, 0.40, 0.44]:
            for w_lyrics in [0.06, 0.08, 0.10]:
                for w_ah in [0.0, 0.03, 0.05, 0.08]:
                    for w_cf in [0.08, 0.10, 0.13]:
                        for w_clap in [0.03, 0.05, 0.08]:
                            for w_bm25 in [0.20, 0.24, 0.28]:
                                ndcg, hit = evaluate(w_tt, w_cf, w_qm, w_clap, w_bm25, w_lyrics, w_ah, True)
                                results.append((ndcg, hit, w_tt, w_cf, w_qm, w_clap, w_bm25, w_lyrics, w_ah))

    results.sort(reverse=True)
    print(f"\nTop 15 configs (evaluated on all {N} turns):")
    for r in results[:15]:
        ndcg, hit, w_tt, w_cf, w_qm, w_clap, w_bm25, w_lyrics, w_ah = r
        print(f"  nDCG@20={ndcg:.4f} Hit@20={hit:.3f} | "
              f"tt={w_tt:.2f} cf={w_cf:.2f} qm={w_qm:.2f} lyrics={w_lyrics:.2f} ah={w_ah:.2f} clap={w_clap:.2f} bm25={w_bm25:.2f}")
else:
    bm25_norm = args.bm25_norm
    ndcg, hit = evaluate(args.w_tt, args.w_cf, args.w_qwen_meta, args.w_clap, args.w_bm25, args.w_qwen_lyrics, bm25_norm)
    print(f"nDCG@20: {ndcg:.4f}  Hit@20: {hit:.3f}")
    print(f"Weights: tt={args.w_tt} cf={args.w_cf} qm={args.w_qwen_meta} clap={args.w_clap} bm25={args.w_bm25} norm={bm25_norm}")
