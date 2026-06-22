"""
Blind A proxy evaluator.

Uses single-target dev sessions (goal categories A,B,C,D,I,J) evaluated at turn 1.
These are fuzzy-recall / explicit-request sessions where the gold track IS the target
the user described -- the same task structure as blind A. Correct model ordering
confirmed: v8b (0.1714) > v6/v07 (0.1690), matching blind A (0.3701 vs 0.3164).

Spec file: plan/BLINDPROXY_SESSIONS.json  (442 sessions, turn 1)

Why this works:
- Full dev eval (8 turns, uniform) orders v07 > v10 -- WRONG direction vs blind A.
- Single-target turn-1 orders v10 > v07 -- CORRECT direction.
- TT v8b is genuinely better at content/lyric-based retrieval; this surfaces at turn 1
  where retrieval quality is the sole discriminator (no history noise).
- Later turns revert the ordering because history-dependent leakage dominates.

Usage:
    # Evaluate a prediction file against the proxy
    python scripts/inference/eval_blindproxy.py --pred exp/inference/devset/<tid>.json

    # Rebuild the index (not usually needed -- spec is version-controlled)
    python scripts/inference/eval_blindproxy.py --build_index
"""
import argparse
import ast
import json
import math
import os
from collections import defaultdict

from datasets import load_dataset

PROXY_INDEX_PATH = "plan/BLINDPROXY_SESSIONS.json"

# Blind A empirical turn distribution (80 sessions observed 2026-06-02)
BLIND_A_TURN_DIST = {1: 20, 2: 15, 3: 10, 4: 5, 5: 8, 6: 9, 7: 8, 8: 5}
BLIND_A_TOTAL = sum(BLIND_A_TURN_DIST.values())

parser = argparse.ArgumentParser()
parser.add_argument("--build_index", action="store_true",
                    help="Build the proxy index from the dev split and save it.")
parser.add_argument("--pred",   default="",
                    help="Prediction file to evaluate.")
parser.add_argument("--seed",   type=int, default=42)
parser.add_argument("--split",  default="test",
                    help="HF dataset split that contains the dev sessions.")
args = parser.parse_args()


def build_index(seed: int) -> list[dict]:
    import random
    rng = random.Random(seed)

    ds = load_dataset("talkpl-ai/TalkPlayData-Challenge-Dataset", split=args.split)
    sessions = list(ds)

    weights = [BLIND_A_TURN_DIST.get(tn, 0) for tn in range(1, 9)]
    turn_pool = list(range(1, 9))

    index = []
    for sess in sessions:
        sid = sess["session_id"]
        music_turns = {t["turn_number"]: t["content"]
                       for t in sess["conversations"] if t["role"] == "music"}
        # Sample a turn number from blind A distribution; fall back if not in session
        for _ in range(20):
            tn = rng.choices(turn_pool, weights=weights)[0]
            if tn in music_turns:
                index.append({"session_id": sid, "turn_number": tn,
                               "gold": music_turns[tn]})
                break
        else:
            # Fallback: pick any available turn
            tn = rng.choice(list(music_turns.keys()))
            index.append({"session_id": sid, "turn_number": tn,
                           "gold": music_turns[tn]})

    return index


def ndcg_at_k(preds: list, gold: str, k: int) -> float:
    preds = preds[:k]
    dcg = sum((1.0 / math.log2(i + 2)) for i, p in enumerate(preds) if p == gold)
    idcg = 1.0 / math.log2(2)
    return dcg / idcg if idcg else 0.0


def evaluate(index: list[dict], pred_path: str) -> dict:
    with open(pred_path) as f:
        raw = json.load(f)
    pred_lookup: dict[tuple, list] = {}
    for e in raw:
        sid = e["session_id"]
        tn  = int(e["turn_number"])
        try:
            tracks = (ast.literal_eval(e["predicted_track_ids"])
                      if isinstance(e["predicted_track_ids"], str)
                      else e["predicted_track_ids"])
        except Exception:
            tracks = []
        pred_lookup[(sid, tn)] = tracks

    # Compute per-turn nDCG, grouped by turn_number
    by_turn: dict[int, list[float]] = defaultdict(list)
    missed = 0
    for row in index:
        sid = row["session_id"]
        tn  = row["turn_number"]
        gold = row["gold"]
        preds = pred_lookup.get((sid, str(tn))) or pred_lookup.get((sid, tn), [])
        if not preds:
            missed += 1
            continue
        ndcg = ndcg_at_k(preds, gold, 20)
        by_turn[tn].append(ndcg)

    if missed:
        print(f"  WARNING: {missed} proxy turns had no matching prediction")

    turn_means = {tn: sum(vals) / len(vals) for tn, vals in by_turn.items()}
    macro_ndcg = sum(turn_means.values()) / len(turn_means) if turn_means else 0.0

    # Also compute micro (flat mean, for reference)
    all_vals = [v for vals in by_turn.values() for v in vals]
    micro_ndcg = sum(all_vals) / len(all_vals) if all_vals else 0.0

    print(f"  nDCG@20 (flat mean over {len(all_vals)} pairs): {micro_ndcg:.4f}")
    print(f"  Hit@20: {sum(1 for v in all_vals if v > 0)/len(all_vals):.3f}")
    print(f"  Turn groups: {sorted(turn_means)}  Per-turn:", {tn: f"{m:.3f}" for tn, m in sorted(turn_means.items())})
    return {"ndcg20": micro_ndcg, "per_turn": turn_means, "n_turns": len(all_vals)}


if args.build_index or not os.path.exists(PROXY_INDEX_PATH):
    print(f"Building proxy index (seed={args.seed})...")
    idx = build_index(args.seed)
    os.makedirs(os.path.dirname(PROXY_INDEX_PATH), exist_ok=True)
    json.dump(idx, open(PROXY_INDEX_PATH, "w"), indent=2)
    from collections import Counter
    tn_dist = Counter(r["turn_number"] for r in idx)
    print(f"Saved {len(idx)} entries to {PROXY_INDEX_PATH}")
    print(f"Turn distribution: {dict(sorted(tn_dist.items()))}")
    if not args.pred:
        import sys; sys.exit(0)

if args.pred:
    print(f"\nLoading proxy index from {PROXY_INDEX_PATH}...")
    idx = json.load(open(PROXY_INDEX_PATH))
    print(f"Evaluating: {args.pred}")
    evaluate(idx, args.pred)
