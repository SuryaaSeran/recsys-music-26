"""
Fast pool-recall sweep: how many turns have gold in the candidate pool
under different expansion configs? Pure recall — no scoring, no nDCG.

Reads exp/analysis/recall_audit.jsonl (one row per turn with bm25_rank +
each dense signal's global rank of gold). Computes union recall in O(N)
per config — entire sweep finishes in seconds.

Output: stdout table + exp/analysis/pool_recall_sweep.txt

Usage:
    python scripts/inference/sweep_pool_recall.py
"""
import json
import numpy as np
from pathlib import Path
from itertools import product

AUDIT = Path("exp/analysis/recall_audit.jsonl")
OUT   = Path("exp/analysis/pool_recall_sweep.txt")

print(f"Loading {AUDIT}...")
recs = [json.loads(l) for l in AUDIT.open()]
N = len(recs)
print(f"Turns: {N}")

def col(key):
    return np.array([(r[key] if r[key] is not None else -1) for r in recs], dtype=np.int32)

bm25 = col("bm25_rank")
tt   = col("tt_rank")
qm   = col("qm_rank")
ql   = col("ql_rank")
cf   = col("cf_rank")
warm = np.array([r["has_cf"] for r in recs])

baseline = ((bm25 > 0) & (bm25 <= 500)).mean()

def union_recall(bm25_K, tt_K, qm_K, ql_K, cf_K_warm):
    hit = (bm25 > 0) & (bm25 <= bm25_K)
    if tt_K:    hit |= (tt   > 0) & (tt   <= tt_K)
    if qm_K:    hit |= (qm   > 0) & (qm   <= qm_K)
    if ql_K:    hit |= (ql   > 0) & (ql   <= ql_K)
    if cf_K_warm: hit |= (cf > 0) & (cf <= cf_K_warm) & warm
    return float(hit.mean())

def pool_size(bm25_K, tt_K, qm_K, ql_K, cf_K_warm):
    """Approximate candidate count (upper bound; ignores overlap)."""
    return bm25_K + tt_K + qm_K + ql_K + cf_K_warm

lines = []
def out(s):
    lines.append(s); print(s)

out(f"Baseline BM25@500 only: {baseline:.4f}\n")

# Configs to sweep — emphasize small pools first (low-noise expansion)
out(f"{'config':<55s}  {'recall':>7s}  {'lift':>7s}  {'pool_max':>8s}")
out("-" * 86)

configs = [
    # bm25, tt, qm, ql, cf
    ("BM25@500",                              500,   0,   0,   0,   0),
    ("BM25@500 + TT@50",                      500,  50,   0,   0,   0),
    ("BM25@500 + TT@100",                     500, 100,   0,   0,   0),
    ("BM25@500 + TT@200",                     500, 200,   0,   0,   0),
    ("BM25@500 + TT@500",                     500, 500,   0,   0,   0),
    ("BM25@500 + TT@1000",                    500,1000,   0,   0,   0),
    ("BM25@500 + QM@100",                     500,   0, 100,   0,   0),
    ("BM25@500 + QM@200",                     500,   0, 200,   0,   0),
    ("BM25@500 + QM@500",                     500,   0, 500,   0,   0),
    ("BM25@500 + CF@100 (warm)",              500,   0,   0,   0, 100),
    ("BM25@500 + CF@200 (warm)",              500,   0,   0,   0, 200),
    ("BM25@500 + CF@500 (warm)",              500,   0,   0,   0, 500),
    ("BM25@500 + TT@100 + QM@100",            500, 100, 100,   0,   0),
    ("BM25@500 + TT@200 + QM@100",            500, 200, 100,   0,   0),
    ("BM25@500 + TT@200 + QM@200",            500, 200, 200,   0,   0),
    ("BM25@500 + TT@500 + QM@200",            500, 500, 200,   0,   0),
    ("BM25@500 + TT@500 + QM@500",            500, 500, 500,   0,   0),
    ("BM25@500 + TT@200 + QM@100 + CF@100",   500, 200, 100,   0, 100),
    ("BM25@500 + TT@500 + QM@200 + CF@200",   500, 500, 200,   0, 200),
    ("BM25@500 + TT@500 + QM@200 + QL@200",   500, 500, 200, 200,   0),
    ("BM25@500 + TT@500 + QM@500 + QL@500 + CF@500", 500, 500, 500, 500, 500),
    ("BM25@500 + TT@1000 + QM@500 + CF@500",  500,1000, 500,   0, 500),
    ("BM25@1000",                             1000,  0,   0,   0,   0),
    ("BM25@1000 + TT@500",                    1000,500,   0,   0,   0),
    ("BM25@1000 + TT@1000 + QM@500 + CF@500", 1000,1000,500,   0, 500),
]
for name, *ks in configs:
    r = union_recall(*ks)
    lift = r - baseline
    out(f"{name:<55s}  {r:7.4f}  {lift:+7.4f}  {pool_size(*ks):>8d}")

out("\n--- breakdown by query bucket (BM25@500 + TT@100) ---")
buckets = np.array([r["bucket"] for r in recs])
for b in ["specific","mood","lyrics","more_like_this","history_driven","generic"]:
    m = buckets == b
    if m.sum() == 0: continue
    base = ((bm25 > 0) & (bm25 <= 500) & m).sum() / m.sum()
    plus = (((bm25 > 0) & (bm25 <= 500)) | ((tt > 0) & (tt <= 100))) & m
    out(f"  {b:18s} N={m.sum():4d}  base={base:.4f}  +TT@100={plus.sum()/m.sum():.4f}  lift={plus.sum()/m.sum()-base:+.4f}")

out("\n--- warm vs cold (BM25@500 + TT@200 + CF@100-warm) ---")
for name, m in [("warm", warm), ("cold", ~warm)]:
    base = ((bm25 > 0) & (bm25 <= 500) & m).sum() / m.sum()
    hit = ((bm25 > 0) & (bm25 <= 500)) | ((tt > 0) & (tt <= 200))
    if name == "warm":
        hit |= (cf > 0) & (cf <= 100) & warm
    plus = (hit & m).sum() / m.sum()
    out(f"  {name:6s} N={m.sum():4d}  base={base:.4f}  expanded={plus:.4f}  lift={plus-base:+.4f}")

OUT.parent.mkdir(parents=True, exist_ok=True)
OUT.write_text("\n".join(lines) + "\n")
print(f"\nWrote {OUT}")
