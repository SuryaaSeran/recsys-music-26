"""
Quick ablation: test different BM25 query construction strategies.
Evaluates on full test set (or --sessions N).

Usage:
    python scripts/bm25_query_ablation.py [--sessions 0]
"""
import argparse
import json
import math
import os
from pathlib import Path
from collections import defaultdict

import bm25s
from datasets import load_dataset, concatenate_datasets

parser = argparse.ArgumentParser()
parser.add_argument("--sessions", type=int, default=0)
args = parser.parse_args()

CACHE_PATH = "cache/bm25/track_metadata"
CORPUS_TYPES = ["track_name", "artist_name", "album_name", "release_date", "tag_list"]

print("Loading track metadata...")
meta_ds = load_dataset("talkpl-ai/TalkPlayData-Challenge-Track-Metadata")
all_tracks = concatenate_datasets([meta_ds["all_tracks"], meta_ds["test_tracks"]])
metadata_dict = {row["track_id"]: row for row in all_tracks}

print("Loading BM25 index...")
bm25_model = bm25s.BM25.load(CACHE_PATH, load_corpus=False)
with open(f"{CACHE_PATH}/track_ids.json") as f:
    track_ids = json.load(f)


def retrieve_bm25(query: str, topk: int = 20) -> list[str]:
    tokens = bm25s.tokenize([query.lower()])
    results = bm25_model.retrieve(tokens, k=topk, return_as="tuple")
    return [track_ids[int(i)] for i in results.documents[0]]


def evaluate(preds: list[dict]) -> dict:
    ds = load_dataset("talkpl-ai/TalkPlayData-Challenge-Dataset")["test"]
    sessions = list(ds)
    if args.sessions > 0:
        sessions = sessions[:args.sessions]

    pred_lookup = {}
    for p in preds:
        pred_lookup[(p["session_id"], p["turn_number"])] = p["predicted_track_ids"]

    def dcg(rank):
        return 1.0 / math.log2(rank + 1) if rank >= 1 else 0.0

    scores = {"ndcg1": [], "ndcg10": [], "ndcg20": [], "hit20": 0, "total": 0}
    for item in sessions:
        for turn in item["conversations"]:
            if turn["role"] != "music":
                continue
            gold = turn["content"]
            key = (item["session_id"], turn["turn_number"])
            if key not in pred_lookup:
                continue
            predicted = pred_lookup[key]
            scores["total"] += 1
            rank = next((i + 1 for i, t in enumerate(predicted) if t == gold), None)
            ideal = dcg(1)
            scores["ndcg1"].append(dcg(rank) / ideal if rank == 1 else 0.0)
            scores["ndcg10"].append(dcg(rank) / ideal if rank and rank <= 10 else 0.0)
            scores["ndcg20"].append(dcg(rank) / ideal if rank and rank <= 20 else 0.0)
            if rank and rank <= 20:
                scores["hit20"] += 1

    n = len(scores["ndcg20"])
    return {
        "ndcg@20": sum(scores["ndcg20"]) / n if n else 0,
        "ndcg@10": sum(scores["ndcg10"]) / n if n else 0,
        "hit@20": scores["hit20"] / n if n else 0,
        "n": n,
    }


print("Loading dataset...")
ds = load_dataset("talkpl-ai/TalkPlayData-Challenge-Dataset")["test"]
sessions = list(ds)
if args.sessions > 0:
    sessions = sessions[:args.sessions]


def run_strategy(name: str, build_query_fn):
    preds = []
    for item in sessions:
        sid = item["session_id"]
        uid = item["user_id"]
        goal = item.get("conversation_goal", {}).get("listener_goal", "")
        culture = item.get("user_profile", {}).get("preferred_musical_culture", "")
        convs = item["conversations"]

        for target_turn in sorted(set(c["turn_number"] for c in convs)):
            music_turns = [c for c in convs if c["turn_number"] == target_turn and c["role"] == "music"]
            if not music_turns:
                continue
            user_turns = [c for c in convs if c["turn_number"] == target_turn and c["role"] == "user"]
            if not user_turns:
                continue
            user_query = user_turns[0]["content"]

            history = []
            for turn in convs:
                if turn["turn_number"] >= target_turn:
                    break
                if turn["role"] == "music":
                    row = metadata_dict.get(turn["content"], {})
                    name_t = (row.get("track_name") or [""])[0]
                    artist = (row.get("artist_name") or [""])[0]
                    history.append({"role": "assistant", "content": f"{name_t} {artist}"})
                else:
                    history.append({"role": turn["role"], "content": turn["content"]})

            query = build_query_fn(goal, culture, history, user_query)
            tids = retrieve_bm25(query, topk=20)
            preds.append({"session_id": sid, "user_id": uid, "turn_number": target_turn, "predicted_track_ids": tids})

    return preds


strategies = {
    "query_only": lambda g, c, h, q: q,
    "query_x3": lambda g, c, h, q: f"{q} {q} {q}",
    "goal_query": lambda g, c, h, q: f"{g} {q}",
    "goal_culture_query": lambda g, c, h, q: f"{g} {c} {q}",
    "goal_culture_query_x2": lambda g, c, h, q: f"{g} {c} {q} {q}",
    "all_context": lambda g, c, h, q: " ".join([g, c] + [x["content"] for x in h[-4:]] + [q]),
    "no_hist_goal_culture": lambda g, c, h, q: f"{g} {c} {q}",
    "user_turns_only": lambda g, c, h, q: " ".join([g] + [x["content"] for x in h if x["role"] == "user"] + [q]),
    "query_plus_user_hist": lambda g, c, h, q: " ".join([x["content"] for x in h[-2:] if x["role"] == "user"] + [q, q]),
}

print(f"\nRunning ablation on {len(sessions)} sessions...\n")
results = {}
for strat_name, fn in strategies.items():
    print(f"  Strategy: {strat_name}...", flush=True)
    preds = run_strategy(strat_name, fn)

    # Quick nDCG@20 computation inline
    pred_lookup = {(p["session_id"], p["turn_number"]): p["predicted_track_ids"] for p in preds}
    hit20 = total = 0
    ndcg20_sum = 0.0
    for item in sessions:
        for turn in item["conversations"]:
            if turn["role"] != "music":
                continue
            gold = turn["content"]
            key = (item["session_id"], turn["turn_number"])
            if key not in pred_lookup:
                continue
            predicted = pred_lookup[key]
            total += 1
            rank = next((i + 1 for i, t in enumerate(predicted) if t == gold), None)
            if rank and rank <= 20:
                hit20 += 1
                ndcg20_sum += 1.0 / math.log2(rank + 1)

    ndcg20 = ndcg20_sum / total if total else 0
    hit_pct = hit20 / total * 100 if total else 0
    results[strat_name] = {"ndcg@20": ndcg20, "hit@20%": hit_pct}
    print(f"    nDCG@20={ndcg20:.4f}  Hit@20={hit_pct:.1f}%")

print("\n=== Summary ===")
for name, r in sorted(results.items(), key=lambda x: -x[1]["ndcg@20"]):
    print(f"  {name:30s}  nDCG@20={r['ndcg@20']:.4f}  Hit@20={r['hit@20%']:.1f}%")
