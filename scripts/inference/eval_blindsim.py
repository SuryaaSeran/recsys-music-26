"""Flat-mean nDCG@20 over a fixed (session, turn) spec — blind-A-aligned.

Matches blind A semantics: one turn per session, flat average over the set
(NOT per-turn-number macro). Single gold track per turn.

Usage:
    python scripts/inference/eval_blindsim.py --pred <pred>.json \
        --spec plan/DEV_BLINDSIM_100.json
"""
import argparse, json, math
from datasets import load_dataset

ap = argparse.ArgumentParser()
ap.add_argument("--pred", required=True)
ap.add_argument("--spec", default="plan/DEV_BLINDSIM_100.json")
ap.add_argument("--k", type=int, default=20)
ap.add_argument("--split", default="test")
ap.add_argument("--by_turn", action="store_true", help="Also print per-turn breakdown.")
args = ap.parse_args()

spec = json.load(open(args.spec))
want = {(s["session_id"], s["turn_number"]) for s in spec}

# Gold: the music-turn content at (session_id, turn_number)
ds = load_dataset("talkpl-ai/TalkPlayData-Challenge-Dataset")[args.split]
gold = {}
sids = {s["session_id"] for s in spec}
for item in ds:
    if item["session_id"] not in sids:
        continue
    for t in item["conversations"]:
        if t["role"] == "music" and (item["session_id"], t["turn_number"]) in want:
            gold[(item["session_id"], t["turn_number"])] = t["content"]

preds = {(p["session_id"], p["turn_number"]): p["predicted_track_ids"]
         for p in json.load(open(args.pred))}

def ndcg_at_k(pred_ids, gold_tid, k):
    top = pred_ids[:k]
    if gold_tid in top:
        return 1.0 / math.log2(top.index(gold_tid) + 2)
    return 0.0

scores = []
by_turn = {}
missing = 0
for s in spec:
    key = (s["session_id"], s["turn_number"])
    g = gold.get(key)
    pids = preds.get(key)
    if g is None or pids is None:
        missing += 1
        continue
    nd = ndcg_at_k(pids, g, args.k)
    scores.append(nd)
    by_turn.setdefault(s["turn_number"], []).append(nd)

mean = sum(scores) / len(scores) if scores else 0.0
hit = sum(1 for s in scores if s > 0) / len(scores) if scores else 0.0
print(f"pred: {args.pred}")
print(f"spec: {args.spec}  ({len(spec)} pairs, {missing} missing)")
print(f"nDCG@{args.k} (flat mean over {len(scores)}): {mean:.4f}")
print(f"Hit@{args.k}: {hit:.3f}")
if args.by_turn:
    print("per-turn:")
    for t in sorted(by_turn):
        v = by_turn[t]
        print(f"  turn {t}: nDCG@{args.k}={sum(v)/len(v):.4f}  (n={len(v)})")
