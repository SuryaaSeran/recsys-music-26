"""
Compare multiple LTR/model prediction files on the 200-session golden holdout.

Reports per-model:
  - nDCG@{1,10,20}  (macro, turn-position then across positions)
  - Hit@{1,10,20}   (fraction of turns where gold in top-k)
  - Pool recall     (fraction of turns where gold appears anywhere in the predicted list)
  - Mean gold rank  (mean rank of gold when it appears)

For each model also shows the first --examples turns: session_id, turn, gold,
gold rank, top-5 predicted IDs.

Usage:
    python scripts/inference/compare_golden.py \
        --golden plan/GOLDEN_HOLDOUT_SESSIONS.json \
        --pred   exp/inference/devset/phase_a_baseline.json \
                 exp/inference/devset/phase_b_ltr.json \
        --labels "Phase A (0.1646 best)" "Phase B (reg)" \
        --examples 5

Pool recall is measured against the full length of predicted_track_ids, so pass
a top-150 file to get a meaningful pool recall number. For top-20 files,
pool_recall == hit@20.
"""
import argparse
import json
import math
from collections import defaultdict
from pathlib import Path

from datasets import load_dataset

parser = argparse.ArgumentParser()
parser.add_argument("--golden", default="plan/GOLDEN_HOLDOUT_SESSIONS.json")
parser.add_argument("--pred",   nargs="+", required=True,
                    help="One or more prediction JSON files to compare.")
parser.add_argument("--labels", nargs="*", default=None,
                    help="Human-readable labels for each --pred file (same order).")
parser.add_argument("--split",  default="test")
parser.add_argument("--examples", type=int, default=3,
                    help="Number of example turns to print per model.")
args = parser.parse_args()

# ── helpers ──────────────────────────────────────────────────────────────────

def ndcg_at_k(pred, gold, k):
    for i, p in enumerate(pred[:k], start=1):
        if p == gold:
            return (1.0 / math.log2(i + 1)) / (1.0 / math.log2(2))
    return 0.0

def hit_at_k(pred, gold, k):
    return 1 if gold in pred[:k] else 0

def gold_rank(pred, gold):
    try:
        return pred.index(gold) + 1   # 1-based
    except ValueError:
        return None

# ── load golden session IDs ───────────────────────────────────────────────────

with open(args.golden) as f:
    gdata = json.load(f)
golden_ids = set(gdata["golden_200"])
print(f"Golden holdout: {len(golden_ids)} sessions")

# ── load ground truth (once) ─────────────────────────────────────────────────

print("Loading ground truth from HuggingFace dataset...")
ds = load_dataset("talkpl-ai/TalkPlayData-Challenge-Dataset")[args.split]

gt = {}  # (session_id, turn_number) -> gold_track_id
for item in ds:
    sid = item["session_id"]
    if sid not in golden_ids:
        continue
    for turn in item["conversations"]:
        if turn["role"] != "music":
            continue
        gt[(sid, turn["turn_number"])] = turn["content"]

print(f"  Golden turns loaded: {len(gt)}")

# ── compare each prediction file ─────────────────────────────────────────────

labels = args.labels or args.pred

header_printed = False

for pred_path, label in zip(args.pred, labels):
    with open(pred_path) as f:
        preds = json.load(f)

    # filter to golden sessions
    golden_preds = [p for p in preds if p["session_id"] in golden_ids]
    pred_lookup = {(p["session_id"], p["turn_number"]): p["predicted_track_ids"]
                   for p in golden_preds}

    by_turn = defaultdict(lambda: {
        "ndcg@1": [], "ndcg@10": [], "ndcg@20": [],
        "hit@1": 0, "hit@10": 0, "hit@20": 0,
        "pool_hits": 0, "gold_ranks": [], "n": 0,
    })

    all_entries = []   # for example printing

    for (sid, tnum), gold in gt.items():
        if (sid, tnum) not in pred_lookup:
            continue
        pred = pred_lookup[(sid, tnum)]
        b = by_turn[tnum]
        b["ndcg@1"].append(ndcg_at_k(pred, gold, 1))
        b["ndcg@10"].append(ndcg_at_k(pred, gold, 10))
        b["ndcg@20"].append(ndcg_at_k(pred, gold, 20))
        b["hit@1"]  += hit_at_k(pred, gold, 1)
        b["hit@10"] += hit_at_k(pred, gold, 10)
        b["hit@20"] += hit_at_k(pred, gold, 20)
        gr = gold_rank(pred, gold)
        if gr is not None:
            b["pool_hits"] += 1
            b["gold_ranks"].append(gr)
        b["n"] += 1
        all_entries.append((sid, tnum, gold, pred, gr))

    # ── metrics ──────────────────────────────────────────────────────────────

    def pt_mean(field):
        vals = [sum(by_turn[t][field]) / len(by_turn[t][field])
                for t in sorted(by_turn) if by_turn[t]["n"] > 0]
        return sum(vals) / len(vals) if vals else 0.0

    def total(field):
        return sum(by_turn[t][field] for t in by_turn)

    total_n        = sum(b["n"] for b in by_turn.values())
    all_gold_ranks = [r for b in by_turn.values() for r in b["gold_ranks"]]
    pool_recall    = total("pool_hits") / total_n if total_n else 0.0
    mean_rank      = sum(all_gold_ranks) / len(all_gold_ranks) if all_gold_ranks else float("nan")
    pred_len       = len(all_entries[0][3]) if all_entries else "?"

    print()
    print(f"=== {label}  [{Path(pred_path).name}]  (list_len={pred_len}) ===")
    print(f"  Turns evaluated : {total_n}")
    print(f"  nDCG@1          : {pt_mean('ndcg@1'):.4f}")
    print(f"  nDCG@10         : {pt_mean('ndcg@10'):.4f}")
    print(f"  nDCG@20         : {pt_mean('ndcg@20'):.4f}")
    print(f"  Hit@1           : {total('hit@1')}/{total_n}  ({total('hit@1')/total_n:.3f})")
    print(f"  Hit@10          : {total('hit@10')}/{total_n}  ({total('hit@10')/total_n:.3f})")
    print(f"  Hit@20          : {total('hit@20')}/{total_n}  ({total('hit@20')/total_n:.3f})")
    print(f"  Recall@list({pred_len:>3}): {total('pool_hits')}/{total_n}  ({pool_recall:.3f})"
          f"   <- gold anywhere in predicted list; == Hit@20 when list_len==20")
    print(f"  Mean gold rank  : {mean_rank:.1f}  (when found)")

    if args.examples > 0:
        print(f"\n  --- {args.examples} example turns ---")
        for sid, tnum, gold, pred, gr in all_entries[:args.examples]:
            rank_str = f"rank={gr}" if gr is not None else "NOT IN LIST"
            print(f"  [{sid[:8]}..] turn={tnum}  gold={gold}  {rank_str}")
            print(f"    top-5 pred: {pred[:5]}")
