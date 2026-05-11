"""
Local evaluator matching the official music-crs-evaluator semantics.

Computes per-turn macro-averaged nDCG@{1,10,20} (mean per turn-position, then
mean across turn-positions) plus catalog_diversity and lexical_diversity
(Distinct-2). Enforces the official rule that predicted_track_ids contain no
duplicates.

Usage:
    python scripts/inference/evaluate_local.py --pred exp/inference/devset/<tid>.json
    python scripts/inference/evaluate_local.py --pred ... --hit  # also print Hit@k counts
"""
import argparse
import json
import math
from collections import defaultdict
from datasets import load_dataset

parser = argparse.ArgumentParser()
parser.add_argument("--pred", required=True)
parser.add_argument("--split", default="test")
parser.add_argument("--sessions", type=int, default=0, help="0=all")
parser.add_argument("--hit", action="store_true",
                    help="Also print Hit@1/10/20 counts (not in official scoring).")
parser.add_argument("--strict", action="store_true", default=True,
                    help="Raise on duplicate track IDs (official behaviour). Use --no-strict to warn instead.")
parser.add_argument("--no-strict", dest="strict", action="store_false")
args = parser.parse_args()


def ndcg_at_k(pred, gold, k):
    pred = pred[:k]
    dcg = 0.0
    for i, p in enumerate(pred, start=1):
        rel = 1 if p == gold else 0
        dcg += rel / math.log2(i + 1)
    idcg = 1.0 / math.log2(2)  # one relevant item
    return dcg / idcg if idcg else 0.0


def hit_at_k(pred, gold, k):
    return 1 if gold in pred[:k] else 0


# Load predictions
with open(args.pred) as f:
    preds = json.load(f)

pred_lookup = {}
for p in preds:
    key = (p["session_id"], p["turn_number"])
    tracks = p["predicted_track_ids"]
    if len(tracks) != len(set(tracks)):
        msg = f"duplicate track_ids in {key}"
        if args.strict:
            raise ValueError("Predictions should be unique. Duplicates detected: " + msg)
        print("WARN:", msg)
    pred_lookup[key] = (tracks, p.get("predicted_response", ""))

# Load ground truth
ds = load_dataset("talkpl-ai/TalkPlayData-Challenge-Dataset")[args.split]
sessions = list(ds)
if args.sessions > 0:
    sessions = sessions[:args.sessions]

# Per-turn-number accumulation (matches official macro semantics)
by_turn = defaultdict(lambda: {"ndcg@1": [], "ndcg@10": [], "ndcg@20": [],
                                "hit@1": 0, "hit@10": 0, "hit@20": 0, "n": 0})

all_recommended = []
all_responses = []

for item in sessions:
    session_id = item["session_id"]
    for turn in item["conversations"]:
        if turn["role"] != "music":
            continue
        gold = turn["content"]
        tnum = turn["turn_number"]
        key = (session_id, tnum)
        if key not in pred_lookup:
            continue
        pred, resp = pred_lookup[key]

        b = by_turn[tnum]
        b["ndcg@1"].append(ndcg_at_k(pred, gold, 1))
        b["ndcg@10"].append(ndcg_at_k(pred, gold, 10))
        b["ndcg@20"].append(ndcg_at_k(pred, gold, 20))
        b["hit@1"]  += hit_at_k(pred, gold, 1)
        b["hit@10"] += hit_at_k(pred, gold, 10)
        b["hit@20"] += hit_at_k(pred, gold, 20)
        b["n"]      += 1

        all_recommended.extend(pred)
        all_responses.append(resp)

# Macro per-turn means, then mean across turns
def per_turn_then_mean(field):
    vals = [sum(by_turn[t][field]) / len(by_turn[t][field])
            for t in sorted(by_turn) if by_turn[t]["n"] > 0]
    return sum(vals) / len(vals) if vals else 0.0


ndcg1  = per_turn_then_mean("ndcg@1")
ndcg10 = per_turn_then_mean("ndcg@10")
ndcg20 = per_turn_then_mean("ndcg@20")

# Diversity
catalog_size = 47071  # all_tracks
catalog_diversity = len(set(all_recommended)) / catalog_size if all_recommended else 0.0

bigrams = set()
total_bg = 0
for r in all_responses:
    toks = (r or "").lower().split()
    for i in range(len(toks) - 1):
        bigrams.add((toks[i], toks[i + 1]))
        total_bg += 1
lex_diversity = len(bigrams) / total_bg if total_bg else 0.0

total = sum(b["n"] for b in by_turn.values())
print(f"Predictions file: {args.pred}")
print(f"Total prediction points: {total}")
print(f"nDCG@1:  {ndcg1:.4f}")
print(f"nDCG@10: {ndcg10:.4f}")
print(f"nDCG@20: {ndcg20:.4f}")
print(f"catalog_diversity: {catalog_diversity:.4f}")
print(f"lexical_diversity: {lex_diversity:.4f}")
if args.hit:
    h1  = sum(b["hit@1"]  for b in by_turn.values())
    h10 = sum(b["hit@10"] for b in by_turn.values())
    h20 = sum(b["hit@20"] for b in by_turn.values())
    print(f"Hit@1:   {h1}/{total} = {100*h1/total:.1f}%")
    print(f"Hit@10:  {h10}/{total} = {100*h10/total:.1f}%")
    print(f"Hit@20:  {h20}/{total} = {100*h20/total:.1f}%")
