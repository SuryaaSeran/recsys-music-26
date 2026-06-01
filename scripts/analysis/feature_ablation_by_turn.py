"""
Per-turn-position feature importance analysis.

Loads an LTR feature dump, splits turns into early/late buckets, and trains
LightGBM on each bucket to reveal which features matter for which turn
positions. Helps identify blind-safe vs blind-unsafe features (blind evaluates
only the last turn per session).

Usage:
    python scripts/analysis/feature_ablation_by_turn.py \
        --features exp/analysis/ltr_phase_b_train_features.npz
"""
import argparse
import json
from pathlib import Path

import numpy as np
import lightgbm as lgb

parser = argparse.ArgumentParser()
parser.add_argument("--features", required=True)
parser.add_argument("--n_folds", type=int, default=5)
parser.add_argument("--num_leaves", type=int, default=31)
parser.add_argument("--lr", type=float, default=0.08)
parser.add_argument("--lambda_l2", type=float, default=0.1)
parser.add_argument("--num_iter", type=int, default=500)
parser.add_argument("--early_stop", type=int, default=50)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--out", default="", help="Save analysis JSON to this path.")
args = parser.parse_args()

print(f"Loading {args.features}...")
data = np.load(args.features, allow_pickle=True)
X = data["X"].astype(np.float32)
y = data["y"].astype(np.int32)
group = data["group"].astype(np.int32)
feature_cols = list(data["feature_cols"])
n_turns = int(group.max()) + 1

sidecar = Path(args.features).with_suffix(".meta.json")
with open(sidecar) as f:
    meta = json.load(f)
turn_meta = meta["turn_meta"]
assert len(turn_meta) == n_turns

turn_numbers = np.array([m["turn_number"] for m in turn_meta], dtype=np.int32)

# Build session -> turns mapping
session_to_turns: dict[str, list[int]] = {}
for t_idx, m in enumerate(turn_meta):
    session_to_turns.setdefault(m["session_id"], []).append(t_idx)
sessions = list(session_to_turns.keys())

print(f"  X: {X.shape}  turns: {n_turns}  features: {len(feature_cols)}")
print(f"  sessions: {len(sessions)}")
print(f"  turn_number distribution: {dict(zip(*np.unique(turn_numbers, return_counts=True)))}")

# Define buckets
BUCKETS = {
    "all":   lambda tn: True,
    "early": lambda tn: tn <= 4,
    "late":  lambda tn: tn >= 5,
    "last":  None,  # special: last turn per session
}


def get_turn_set(bucket_name: str) -> set[int]:
    if bucket_name == "last":
        out = set()
        for sid, tidxs in session_to_turns.items():
            best = max(tidxs, key=lambda t: turn_numbers[t])
            out.add(best)
        return out
    fn = BUCKETS[bucket_name]
    return {t for t in range(n_turns) if fn(turn_numbers[t])}


def slice_for_turns(turn_idxs):
    turn_idxs = sorted(turn_idxs)
    row_mask = np.isin(group, turn_idxs)
    Xs = X[row_mask]
    ys = y[row_mask]
    g_local = group[row_mask]
    sizes, last = [], -1
    for tid in g_local:
        if tid != last:
            sizes.append(0)
            last = tid
        sizes[-1] += 1
    return Xs, ys, np.asarray(sizes, dtype=np.int32)


def train_and_get_importance(turn_set: set[int], label: str):
    """Train LightGBM on given turns, return (cv_ndcg20, feature_gain_dict)."""
    # Filter sessions that have turns in this set
    bucket_session_turns = {}
    for sid, tidxs in session_to_turns.items():
        kept = [t for t in tidxs if t in turn_set]
        if kept:
            bucket_session_turns[sid] = kept
    bucket_sessions = list(bucket_session_turns.keys())

    if len(bucket_sessions) < args.n_folds:
        print(f"  {label}: too few sessions ({len(bucket_sessions)}), skipping")
        return 0.0, {}

    rng = np.random.default_rng(args.seed)
    rng.shuffle(bucket_sessions)
    folds = [bucket_sessions[i::args.n_folds] for i in range(args.n_folds)]

    params = {
        "objective": "lambdarank",
        "metric": "ndcg",
        "ndcg_eval_at": [20],
        "label_gain": [0, 1],
        "learning_rate": args.lr,
        "num_leaves": args.num_leaves,
        "lambda_l2": args.lambda_l2,
        "min_data_in_leaf": 20,
        "min_sum_hessian_in_leaf": 0.1,
        "feature_fraction": 0.8,
        "bagging_fraction": 0.8,
        "bagging_freq": 1,
        "path_smooth": 1.0,
        "force_col_wise": True,
        "verbose": -1,
        "seed": args.seed,
        "num_threads": 4,
    }

    fold_ndcg = []
    all_gain = np.zeros(len(feature_cols))

    for fi in range(args.n_folds):
        val_sids = set(folds[fi])
        train_turns = [t for s in bucket_sessions if s not in val_sids
                       for t in bucket_session_turns[s]]
        val_turns = [t for s in val_sids for t in bucket_session_turns[s]]

        Xtr, ytr, gtr = slice_for_turns(train_turns)
        Xva, yva, gva = slice_for_turns(val_turns)
        if yva.sum() == 0:
            continue

        dtr = lgb.Dataset(Xtr, label=ytr, group=gtr, feature_name=feature_cols)
        dva = lgb.Dataset(Xva, label=yva, group=gva, feature_name=feature_cols,
                          reference=dtr)

        booster = lgb.train(
            params, dtr, num_boost_round=args.num_iter,
            valid_sets=[dva], valid_names=["val"],
            callbacks=[
                lgb.early_stopping(args.early_stop, verbose=False),
                lgb.log_evaluation(period=0),
            ],
        )
        fold_ndcg.append(booster.best_score["val"]["ndcg@20"])
        all_gain += booster.feature_importance(importance_type="gain")

    if not fold_ndcg:
        return 0.0, {}

    mean_ndcg = float(np.mean(fold_ndcg))
    gain_dict = {n: float(g) for n, g in zip(feature_cols, all_gain / len(fold_ndcg))}
    n_turns_used = len(turn_set)
    print(f"  {label}: {n_turns_used} turns, {len(bucket_sessions)} sessions, "
          f"CV ndcg@20 = {mean_ndcg:.4f}")
    return mean_ndcg, gain_dict


# Run per bucket
results = {}
for bucket_name in BUCKETS:
    print(f"\nBucket: {bucket_name}")
    turn_set = get_turn_set(bucket_name)
    ndcg, gains = train_and_get_importance(turn_set, bucket_name)
    results[bucket_name] = {"ndcg20": ndcg, "gains": gains}

# Print comparison table
print(f"\n{'='*90}")
print(f"{'Feature':<30}  {'All':>8}  {'Early':>8}  {'Late':>8}  {'Last':>8}  {'Verdict'}")
print(f"{'='*90}")

for feat in feature_cols:
    g_all = results["all"]["gains"].get(feat, 0)
    g_early = results["early"]["gains"].get(feat, 0)
    g_late = results["late"]["gains"].get(feat, 0)
    g_last = results["last"]["gains"].get(feat, 0)

    # Verdict: blind-safe if late gain >= early gain, blind-unsafe otherwise
    if g_all == 0:
        verdict = "ZERO"
    elif g_late > 0 and g_late >= g_early * 0.8:
        verdict = "safe"
    elif g_early > g_late * 2 and g_late < g_all * 0.3:
        verdict = "UNSAFE"
    else:
        verdict = "mixed"

    print(f"  {feat:<28}  {g_all:>8.0f}  {g_early:>8.0f}  {g_late:>8.0f}  "
          f"{g_last:>8.0f}  {verdict}")

print(f"\n{'='*90}")
print(f"Bucket nDCG@20:")
for b in ["all", "early", "late", "last"]:
    print(f"  {b:<8}: {results[b]['ndcg20']:.4f}")

if args.out:
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to {out_path}")
