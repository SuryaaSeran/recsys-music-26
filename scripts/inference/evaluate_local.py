"""
Local nDCG@20 evaluator for Music-CRS predictions.
Computes nDCG@{1,10,20} against ground-truth "music" turns in the dataset.

Usage:
    python scripts/evaluate_local.py --pred exp/inference/devset/test_run.json [--split test] [--sessions 50]
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
args = parser.parse_args()

# Load predictions
with open(args.pred) as f:
    preds = json.load(f)

pred_lookup = {}
for p in preds:
    key = (p["session_id"], p["turn_number"])
    pred_lookup[key] = p["predicted_track_ids"]

# Load ground truth
ds = load_dataset("talkpl-ai/TalkPlayData-Challenge-Dataset")[args.split]
sessions = list(ds)
if args.sessions > 0:
    sessions = sessions[:args.sessions]

def dcg(rank: int) -> float:
    return 1.0 / math.log2(rank + 1) if rank >= 1 else 0.0

ndcg1_scores = []
ndcg10_scores = []
ndcg20_scores = []
hit1_count = 0
hit10_count = 0
hit20_count = 0
total = 0

for item in sessions:
    session_id = item["session_id"]
    conversations = item["conversations"]

    for turn in conversations:
        if turn["role"] != "music":
            continue
        gold_track = turn["content"]
        target_turn = turn["turn_number"]
        key = (session_id, target_turn)

        if key not in pred_lookup:
            continue

        predicted = pred_lookup[key]
        total += 1

        # Find rank of gold track
        rank = None
        for i, tid in enumerate(predicted, 1):
            if tid == gold_track:
                rank = i
                break

        ideal_dcg20 = dcg(1)  # only 1 relevant item

        if rank is not None:
            gain = dcg(rank)
            ndcg1_scores.append(gain / ideal_dcg20 if rank == 1 else 0.0)
            ndcg10_scores.append(gain / ideal_dcg20 if rank <= 10 else 0.0)
            ndcg20_scores.append(gain / ideal_dcg20 if rank <= 20 else 0.0)
            if rank == 1:
                hit1_count += 1
            if rank <= 10:
                hit10_count += 1
            hit20_count += 1
        else:
            ndcg1_scores.append(0.0)
            ndcg10_scores.append(0.0)
            ndcg20_scores.append(0.0)

n = len(ndcg20_scores)
print(f"Predictions file: {args.pred}")
print(f"Evaluated sessions: {len([item for item in sessions if (item['session_id'], 1) in pred_lookup])}")
print(f"Total prediction points: {total}")
print(f"nDCG@1:  {sum(ndcg1_scores)/n:.4f}  (Hit@1:  {hit1_count}/{n} = {100*hit1_count/n:.1f}%)")
print(f"nDCG@10: {sum(ndcg10_scores)/n:.4f}  (Hit@10: {hit10_count}/{n} = {100*hit10_count/n:.1f}%)")
print(f"nDCG@20: {sum(ndcg20_scores)/n:.4f}  (Hit@20: {hit20_count}/{n} = {100*hit20_count/n:.1f}%)")
