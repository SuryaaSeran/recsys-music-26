"""Offline sweep of alpha (LTR-prior blend) x rerank_k over saved rerank scores.

Reads a score-log produced by rerank_qwen3.py --save_scores (no model needed),
applies the blend for each (alpha, rerank_k), and reports flat-mean nDCG@20 on
a blind-sim spec. Lets us tune the reranker without rerunning the slow model.

Usage:
    python scripts/inference/sweep_rerank_blend.py \
        --scores exp/inference/devset/<scores>.json \
        --spec plan/DEV_BLINDSIM_100.json
"""
import argparse, json, math
from datasets import load_dataset

ap = argparse.ArgumentParser()
ap.add_argument("--scores", required=True, help="score-log from rerank_qwen3 --save_scores")
ap.add_argument("--spec", default="plan/DEV_BLINDSIM_100.json")
ap.add_argument("--split", default="test")
ap.add_argument("--k", type=int, default=20, help="nDCG@k")
ap.add_argument("--alphas", default="0,0.1,0.2,0.3,0.5,0.7,1.0")
ap.add_argument("--rerank_ks", default="20,30,50")
args = ap.parse_args()

score_log = json.load(open(args.scores))  # "sid|tn" -> [[tid, score], ...] in LTR order
spec = json.load(open(args.spec))
want = {(s["session_id"], s["turn_number"]) for s in spec}

# gold per (sid, tn)
ds = load_dataset("talkpl-ai/TalkPlayData-Challenge-Dataset")[args.split]
sids = {s["session_id"] for s in spec}
gold = {}
for item in ds:
    if item["session_id"] not in sids:
        continue
    for t in item["conversations"]:
        if t["role"] == "music" and (item["session_id"], t["turn_number"]) in want:
            gold[(item["session_id"], t["turn_number"])] = t["content"]


def ndcg(order_ids, g, k):
    top = order_ids[:k]
    return 1.0 / math.log2(top.index(g) + 2) if g in top else 0.0


def blended_order(entries, alpha, rk):
    """entries: [[tid, score], ...] in LTR order. Return reordered tids (head only)."""
    head = entries[:rk]
    n = len(head)
    if n == 0:
        return []
    scs = [e[1] for e in head]
    lo, hi = min(scs), max(scs)
    span = (hi - lo) or 1.0
    scored = []
    for i, (tid, sc) in enumerate(head):
        ltr_s = (n - i) / n
        rr_s = (sc - lo) / span
        scored.append((alpha * ltr_s + (1 - alpha) * rr_s, tid))
    scored.sort(key=lambda x: x[0], reverse=True)
    return [tid for _, tid in scored]


alphas = [float(a) for a in args.alphas.split(",")]
rerank_ks = [int(x) for x in args.rerank_ks.split(",")]

# Baseline (pure LTR order, no rerank) for reference
base = []
for s in spec:
    key = (s["session_id"], s["turn_number"])
    ent = score_log.get(f"{key[0]}|{key[1]}")
    g = gold.get(key)
    if ent is None or g is None:
        continue
    base.append(ndcg([e[0] for e in ent], g, args.k))
base_mean = sum(base) / len(base) if base else 0.0
print(f"spec: {args.spec}  turns scored: {len(base)}")
print(f"LTR baseline nDCG@{args.k} (from score-log order): {base_mean:.4f}\n")

print(f"{'alpha':>6} | " + " | ".join(f"k={k:<4}" for k in rerank_ks))
print("-" * (8 + 9 * len(rerank_ks)))
best = (None, -1)
for a in alphas:
    row = []
    for rk in rerank_ks:
        scores = []
        for s in spec:
            key = (s["session_id"], s["turn_number"])
            ent = score_log.get(f"{key[0]}|{key[1]}")
            g = gold.get(key)
            if ent is None or g is None:
                continue
            order = blended_order(ent, a, rk)
            scores.append(ndcg(order, g, args.k))
        m = sum(scores) / len(scores) if scores else 0.0
        row.append(m)
        if m > best[1]:
            best = ((a, rk), m)
    print(f"{a:>6.2f} | " + " | ".join(f"{v:.4f}" for v in row))

print(f"\nBest: alpha={best[0][0]}, rerank_k={best[0][1]} -> nDCG@{args.k}={best[1]:.4f} "
      f"(baseline {base_mean:.4f}, delta {best[1]-base_mean:+.4f})")
