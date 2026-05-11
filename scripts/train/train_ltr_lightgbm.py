"""
Train a LightGBM LambdaMART ranker on the source-aware feature dump.

Input  : NPZ from `run_inference_fusion_recall_expansion.py --write_features <path.npz>`
Output : a LightGBM booster `.txt` + a small JSON sidecar with feature names.

Splits sessions (not turns) into k folds. Trains lambdarank with
group=turn, eval at ndcg@20. Final model is refit on all folds with
best_iteration from the CV.

Usage:
    python scripts/train/train_ltr_lightgbm.py \
        --features exp/analysis/ltr_dev_features.npz \
        --out models/ltr/ltr_v1.txt \
        --n_folds 5 --num_leaves 63 --lr 0.05 --num_iter 1000
"""
import argparse
import json
from pathlib import Path

import numpy as np
import lightgbm as lgb

parser = argparse.ArgumentParser()
parser.add_argument("--features", required=True)
parser.add_argument("--out",      required=True)
parser.add_argument("--n_folds",  type=int, default=5)
parser.add_argument("--num_leaves", type=int, default=63)
parser.add_argument("--lr",       type=float, default=0.05)
parser.add_argument("--num_iter", type=int, default=1000)
parser.add_argument("--early_stop", type=int, default=50)
parser.add_argument("--seed",     type=int, default=42)
args = parser.parse_args()

print(f"Loading {args.features}...")
data = np.load(args.features, allow_pickle=True)
X = data["X"].astype(np.float32)
y = data["y"].astype(np.int32)
group = data["group"].astype(np.int32)
feature_cols = list(data["feature_cols"])
print(f"  X: {X.shape}  y: {y.shape}  groups: {group.max() + 1}  features: {feature_cols}")

# Load turn metadata (sidecar) for session-stratified folds
sidecar = Path(args.features).with_suffix(".meta.json")
with open(sidecar) as f:
    meta = json.load(f)
turn_meta = meta["turn_meta"]
assert len(turn_meta) == group.max() + 1

# Session -> turn indices
session_to_turns = {}
for t_idx, m in enumerate(turn_meta):
    session_to_turns.setdefault(m["session_id"], []).append(t_idx)
sessions = list(session_to_turns.keys())

rng = np.random.default_rng(args.seed)
rng.shuffle(sessions)
folds = [sessions[i::args.n_folds] for i in range(args.n_folds)]

# Build per-turn boundaries for group sizes (LightGBM wants group sizes per turn, contiguous)
# Our X/y/group is row-ordered by turn (chunks were appended in turn order), so groups are
# already contiguous. Just need per-turn counts.
turn_sizes = np.bincount(group)  # length = num turns
n_turns = len(turn_sizes)

def slice_for_turns(turn_idxs):
    """Return (X_slice, y_slice, group_sizes) for the given turn indices, preserving order."""
    turn_idxs = sorted(turn_idxs)
    row_mask = np.isin(group, turn_idxs)
    # Slice rows
    Xs = X[row_mask]
    ys = y[row_mask]
    # group sizes in the order they appear in Xs (i.e. ascending turn_idx)
    g_local = group[row_mask]
    sizes = []
    last = -1
    for tid in g_local:
        if tid != last:
            sizes.append(0)
            last = tid
        sizes[-1] += 1
    return Xs, ys, np.asarray(sizes, dtype=np.int32)

fold_best_iters = []
fold_ndcgs = []

for fi in range(args.n_folds):
    val_sessions = set(folds[fi])
    train_sessions = [s for f in (folds[:fi] + folds[fi + 1:]) for s in f]
    train_turns = [t for s in train_sessions for t in session_to_turns[s]]
    val_turns   = [t for s in val_sessions   for t in session_to_turns[s]]

    Xtr, ytr, gtr = slice_for_turns(train_turns)
    Xva, yva, gva = slice_for_turns(val_turns)

    dtr = lgb.Dataset(Xtr, label=ytr, group=gtr, feature_name=feature_cols)
    dva = lgb.Dataset(Xva, label=yva, group=gva, feature_name=feature_cols, reference=dtr)

    params = {
        "objective": "lambdarank",
        "metric": "ndcg",
        "ndcg_eval_at": [20, 10, 1],
        "learning_rate": args.lr,
        "num_leaves": args.num_leaves,
        "min_data_in_leaf": 50,
        "feature_fraction": 0.9,
        "bagging_fraction": 0.9,
        "bagging_freq": 1,
        "lambdarank_truncation_level": 20,
        "verbose": -1,
        "seed": args.seed,
    }

    print(f"\n--- Fold {fi+1}/{args.n_folds}  train_turns={len(train_turns)}  val_turns={len(val_turns)} ---")
    booster = lgb.train(
        params, dtr,
        num_boost_round=args.num_iter,
        valid_sets=[dva], valid_names=["val"],
        callbacks=[lgb.early_stopping(args.early_stop), lgb.log_evaluation(period=50)],
    )
    best = booster.best_iteration
    val_ndcg20 = booster.best_score["val"]["ndcg@20"]
    fold_best_iters.append(best)
    fold_ndcgs.append(val_ndcg20)
    print(f"Fold {fi+1}  best_iter={best}  ndcg@20={val_ndcg20:.4f}")

print("\nCV ndcg@20 per fold:", [round(v, 4) for v in fold_ndcgs])
print(f"CV mean ndcg@20: {np.mean(fold_ndcgs):.4f}  (std {np.std(fold_ndcgs):.4f})")
print(f"CV mean best_iter: {int(np.mean(fold_best_iters))}")

# Refit on all data using mean best_iter
final_iter = int(np.mean(fold_best_iters))
print(f"\nRefitting on all turns to iter {final_iter}...")
all_turns = list(range(n_turns))
Xall, yall, gall = slice_for_turns(all_turns)
dfull = lgb.Dataset(Xall, label=yall, group=gall, feature_name=feature_cols)
final = lgb.train(
    {**{"objective": "lambdarank", "metric": "ndcg", "ndcg_eval_at": [20],
        "learning_rate": args.lr, "num_leaves": args.num_leaves,
        "min_data_in_leaf": 50, "feature_fraction": 0.9, "bagging_fraction": 0.9,
        "bagging_freq": 1, "lambdarank_truncation_level": 20, "verbose": -1,
        "seed": args.seed}},
    dfull,
    num_boost_round=final_iter,
    callbacks=[lgb.log_evaluation(period=50)],
)

out = Path(args.out)
out.parent.mkdir(parents=True, exist_ok=True)
final.save_model(str(out))
side = out.with_suffix(".meta.json")
with open(side, "w") as f:
    json.dump({"feature_cols": feature_cols,
               "cv_ndcg20_folds": fold_ndcgs,
               "cv_ndcg20_mean": float(np.mean(fold_ndcgs)),
               "final_iter": final_iter,
               "params": {"num_leaves": args.num_leaves, "lr": args.lr}}, f, indent=2)
print(f"Saved booster: {out}\nSaved meta: {side}")
print(f"Top feature gains:")
imp = final.feature_importance(importance_type="gain")
for name, score in sorted(zip(feature_cols, imp), key=lambda x: -x[1])[:10]:
    print(f"  {name:<14}  {score:.0f}")
