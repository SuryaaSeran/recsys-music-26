"""
Train a LightGBM LambdaMART ranker on the source-aware feature dump.

Input  : NPZ from run_inference_fusion_recall_expansion.py --write_features
Output : LightGBM booster .txt + JSON sidecar with CV results and feature gains.

Session-stratified k-fold CV. Final model refits on all data at mean best_iter.

--sweep mode runs a full hyperparameter grid (num_leaves x lr x lambda_l2 x
min_data_in_leaf), picks best CV ndcg@20, then refits.

Usage:
    # single config
    python scripts/train/train_ltr_lightgbm.py \
        --features exp/analysis/ltr_phase_b_train_features.npz \
        --out models/ltr/ltr_phase_b_nl31_lr0p08.txt \
        --n_folds 5 --num_leaves 31 --lr 0.08 --lambda_l2 0.1

    # full sweep
    python scripts/train/train_ltr_lightgbm.py \
        --features exp/analysis/ltr_phase_b_train_features.npz \
        --out models/ltr/ltr_phase_b_sweep_best.txt \
        --n_folds 5 --sweep
"""
import argparse
import json
from pathlib import Path

import numpy as np
import lightgbm as lgb

parser = argparse.ArgumentParser()
parser.add_argument("--features",             required=True)
parser.add_argument("--out",                  required=True)
parser.add_argument("--n_folds",              type=int,   default=5)
# single-config knobs (also used as sweep defaults where not varied)
parser.add_argument("--num_leaves",           type=int,   default=31)
parser.add_argument("--lr",                   type=float, default=0.08)
parser.add_argument("--lambda_l2",            type=float, default=0.1)
parser.add_argument("--min_data_in_leaf",     type=int,   default=20)
parser.add_argument("--feature_fraction",     type=float, default=0.8)
parser.add_argument("--bagging_fraction",     type=float, default=0.8)
parser.add_argument("--min_sum_hessian",      type=float, default=0.1,
                    help="Min sum of hessians in leaf. Prevents splitting on noise "
                         "when positives are sparse (1:2500+ ratio).")
parser.add_argument("--path_smooth",          type=float, default=1.0,
                    help="Path smoothing — regularises splits on low-frequency "
                         "features like popularity=0 or track_year=0.")
parser.add_argument("--truncation_level",     type=int,   default=30,
                    help="LambdaRank gradient truncation level. Wider than @20 "
                         "pushes borderline items into top-20 from above.")
parser.add_argument("--num_iter",             type=int,   default=1000)
parser.add_argument("--early_stop",           type=int,   default=75)
parser.add_argument("--seed",                 type=int,   default=42)
parser.add_argument("--sweep",                action="store_true")
parser.add_argument("--save_sweep_dir", default=None,
                    help="If set, save each sweep booster to this directory.")
parser.add_argument("--soft_labels",    action="store_true",
                    help="Use graded labels (0/1/2) with label_gain=[0,1,3]. "
                         "Feature dump must have been built with --soft_labels.")
parser.add_argument("--poly_feats",     action="store_true",
                    help="Expand features with pairwise interaction columns computed "
                         "at load time. No re-dump needed.")
args = parser.parse_args()

# ── Load data ─────────────────────────────────────────────────────────────────

print(f"Loading {args.features}...")
data = np.load(args.features, allow_pickle=True)
X     = data["X"].astype(np.float32)
y     = data["y"].astype(np.int32)
group = data["group"].astype(np.int32)
feature_cols = list(data["feature_cols"])
n_turns      = int(group.max()) + 1

print(f"  X: {X.shape}  turns: {n_turns}  features: {len(feature_cols)}")
print(f"  positives: {int(y.sum())}  pos_rate: {y.mean():.5f}  "
      f"mean_pool: {X.shape[0]/n_turns:.1f}")

# ── Soft labels ───────────────────────────────────────────────────────────────
# With --soft_labels the dump stores 0/1/2; validate and set label_gain.
# Without it, y is binary 0/1 and label_gain uses the default [0,1].
if args.soft_labels:
    unique_labels = set(int(v) for v in np.unique(y))
    if not unique_labels <= {0, 1, 2}:
        raise ValueError(f"--soft_labels set but unexpected label values: {unique_labels}. "
                         "Re-run feature dump with --soft_labels.")
    print(f"  soft labels: {dict(zip(*np.unique(y, return_counts=True)))}")
    # label_gain: DCG gain for label 0,1,2 → 0, 1, 3
    # This tells LightGBM: same-artist (1) is worth 1 gain unit; gold (2) is worth 3.
    LABEL_GAIN = [0, 1, 3]
else:
    LABEL_GAIN = [0, 1]

# ── Polynomial feature interactions ──────────────────────────────────────────
# Computed from the base features at load time. Indices into feature_cols:
INTERACTION_PAIRS = [
    # (feat_a, feat_b, name)  — only pairs that are semantically meaningful
    ("tt_cos",      "bm25_signal",    "tt_x_bm25"),
    ("tt_rank_sig", "bm25_origin",    "ttrank_x_bm25orig"),
    ("tt_cos",      "tt_rank_sig",    "tt_x_ttrank"),
    ("qm_cos",      "bm25_signal",    "qm_x_bm25"),
    ("artist_sig",  "artist_origin",  "artist_x_orig"),
    ("nn_sig",      "tt_cos",         "nn_x_tt"),
    ("collab_rank_sig", "collab_score", "collab_rank_x_score"),
    ("popularity",  "tt_cos",         "pop_x_tt"),
    ("popularity",  "bm25_signal",    "pop_x_bm25"),
]

if args.poly_feats:
    col_idx = {name: i for i, name in enumerate(feature_cols)}
    new_cols = []
    for fa, fb, name in INTERACTION_PAIRS:
        ia, ib = col_idx.get(fa), col_idx.get(fb)
        if ia is None or ib is None:
            print(f"  poly_feats: skipping {name} (missing column {fa!r} or {fb!r})")
            continue
        interaction = X[:, ia] * X[:, ib]
        new_cols.append((name, interaction))
    if new_cols:
        X = np.hstack([X] + [c[:, None] for _, c in new_cols]).astype(np.float32)
        feature_cols = feature_cols + [n for n, _ in new_cols]
        print(f"  poly_feats: added {len(new_cols)} interaction columns → {len(feature_cols)} total")

sidecar = Path(args.features).with_suffix(".meta.json")
with open(sidecar) as f:
    meta = json.load(f)
turn_meta = meta["turn_meta"]
assert len(turn_meta) == n_turns, f"meta/npz mismatch: {len(turn_meta)} vs {n_turns}"

session_to_turns: dict[str, list[int]] = {}
for t_idx, m in enumerate(turn_meta):
    session_to_turns.setdefault(m["session_id"], []).append(t_idx)
sessions = list(session_to_turns.keys())
print(f"  sessions: {len(sessions)}")

rng = np.random.default_rng(args.seed)
rng.shuffle(sessions)
folds = [sessions[i::args.n_folds] for i in range(args.n_folds)]

# Validate fold balance
print(f"\nFold sizes: {[len(f) for f in folds]} sessions")

# ── Helpers ───────────────────────────────────────────────────────────────────

def slice_for_turns(turn_idxs: list[int]):
    """Row-slice X/y for given turn indices; return (X, y, group_sizes)."""
    turn_idxs = sorted(turn_idxs)
    row_mask  = np.isin(group, turn_idxs)
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


def build_params(nl, lr, l2, mdil):
    return {
        "objective":                    "lambdarank",
        "metric":                       "ndcg",
        "ndcg_eval_at":                 [20, 10, 1],
        "label_gain":                   LABEL_GAIN,
        "learning_rate":                lr,
        "num_leaves":                   nl,
        "min_data_in_leaf":             mdil,
        "min_sum_hessian_in_leaf":      args.min_sum_hessian,
        "feature_fraction":             args.feature_fraction,
        "bagging_fraction":             args.bagging_fraction,
        "bagging_freq":                 1,
        "lambda_l2":                    l2,
        "path_smooth":                  args.path_smooth,
        "lambdarank_truncation_level":  args.truncation_level,
        "force_col_wise":               True,
        "verbose":                      -1,
        "seed":                         args.seed,
        "num_threads":                  4,
    }


def run_cv(nl, lr, l2, mdil, tag="", save_dir=None):
    """
    Session-stratified k-fold CV.
    Returns (mean_ndcg20, mean_best_iter, fold_results_list).
    fold_results_list: [{ndcg1, ndcg10, ndcg20, best_iter, n_train, n_val, pos_val}]
    """
    params = build_params(nl, lr, l2, mdil)
    fold_results = []

    for fi in range(args.n_folds):
        val_sessions   = set(folds[fi])
        train_sessions = [s for f in (folds[:fi] + folds[fi+1:]) for s in f]
        train_turns = [t for s in train_sessions for t in session_to_turns[s]]
        val_turns   = [t for s in val_sessions   for t in session_to_turns[s]]

        Xtr, ytr, gtr = slice_for_turns(train_turns)
        Xva, yva, gva = slice_for_turns(val_turns)

        # Guard: skip fold if no positives in val
        if yva.sum() == 0:
            print(f"  fold {fi+1}: WARNING no positives in val — skipping")
            continue

        dtr = lgb.Dataset(Xtr, label=ytr, group=gtr, feature_name=feature_cols)
        dva = lgb.Dataset(Xva, label=yva, group=gva, feature_name=feature_cols,
                          reference=dtr)

        booster = lgb.train(
            params, dtr,
            num_boost_round=args.num_iter,
            valid_sets=[dva], valid_names=["val"],
            callbacks=[
                lgb.early_stopping(args.early_stop, verbose=False),
                lgb.log_evaluation(period=100),
            ],
        )
        scores = booster.best_score["val"]
        res = {
            "ndcg20":    scores["ndcg@20"],
            "ndcg10":    scores.get("ndcg@10", 0.0),
            "ndcg1":     scores.get("ndcg@1",  0.0),
            "best_iter": booster.best_iteration,
            "n_train":   len(train_turns),
            "n_val":     len(val_turns),
            "pos_val":   int(yva.sum()),
        }
        fold_results.append(res)
        print(f"  fold {fi+1}  iter={res['best_iter']:4d}  "
              f"ndcg@1={res['ndcg1']:.4f}  ndcg@10={res['ndcg10']:.4f}  "
              f"ndcg@20={res['ndcg20']:.4f}  "
              f"val_turns={res['n_val']}  val_pos={res['pos_val']}")

        if save_dir:
            Path(save_dir).mkdir(parents=True, exist_ok=True)
            booster.save_model(str(Path(save_dir) / f"{tag}_fold{fi+1}.txt"))

    if not fold_results:
        raise RuntimeError("All folds skipped — no positives.")

    ndcg20s = [r["ndcg20"] for r in fold_results]
    iters   = [r["best_iter"] for r in fold_results]
    mean_ndcg = float(np.mean(ndcg20s))
    mean_iter = int(round(float(np.mean(iters))))
    return mean_ndcg, mean_iter, fold_results


# ── Sweep or single config ────────────────────────────────────────────────────

if args.sweep:
    # Grid: num_leaves x lr x lambda_l2 x min_data_in_leaf
    grid = [
        # (nl,  lr,    l2,   mdil)
        (31,  0.08, 0.0,  20),
        (31,  0.08, 0.1,  20),
        (31,  0.05, 0.1,  20),
        (63,  0.08, 0.0,  20),
        (63,  0.08, 0.1,  20),
        (63,  0.05, 0.1,  20),
        (63,  0.05, 0.1,  50),
        (127, 0.05, 0.1,  20),
        (127, 0.03, 0.1,  20),
        (127, 0.05, 0.1,  50),
    ]
    all_results = []
    for nl, lr, l2, mdil in grid:
        tag = f"nl{nl}_lr{str(lr).replace('.','p')}_l2{str(l2).replace('.','p')}_mdil{mdil}"
        print(f"\n{'='*60}")
        print(f"Sweep: nl={nl}  lr={lr}  l2={l2}  mdil={mdil}")
        mean_ndcg, mean_iter, fold_results = run_cv(
            nl, lr, l2, mdil, tag=tag, save_dir=args.save_sweep_dir
        )
        std = float(np.std([r["ndcg20"] for r in fold_results]))
        print(f"  CV ndcg@20: {mean_ndcg:.4f}  std={std:.4f}  mean_iter={mean_iter}")
        all_results.append({
            "nl": nl, "lr": lr, "l2": l2, "mdil": mdil,
            "cv_ndcg20": mean_ndcg, "cv_std": std,
            "mean_iter": mean_iter,
            "fold_results": fold_results,
        })

    all_results.sort(key=lambda r: -r["cv_ndcg20"])
    print(f"\n{'='*60}")
    print("Sweep results (best first):")
    for r in all_results:
        print(f"  nl={r['nl']:3d}  lr={r['lr']:.3f}  l2={r['l2']:.2f}  "
              f"mdil={r['mdil']:3d}  cv_ndcg@20={r['cv_ndcg20']:.4f}  "
              f"std={r['cv_std']:.4f}  iter={r['mean_iter']}")

    best = all_results[0]
    final_nl   = best["nl"]
    final_lr   = best["lr"]
    final_l2   = best["l2"]
    final_mdil = best["mdil"]
    final_iter = best["mean_iter"]
    cv_ndcg    = best["cv_ndcg20"]
    fold_results = best["fold_results"]
    sweep_log  = all_results
    print(f"\nBest: nl={final_nl}  lr={final_lr}  l2={final_l2}  "
          f"mdil={final_mdil}  ndcg@20={cv_ndcg:.4f}")

else:
    nl   = args.num_leaves
    lr   = args.lr
    l2   = args.lambda_l2
    mdil = args.min_data_in_leaf
    print(f"\n--- Single config: nl={nl}  lr={lr}  l2={l2}  mdil={mdil} ---")
    cv_ndcg, final_iter, fold_results = run_cv(nl, lr, l2, mdil)
    final_nl, final_lr, final_l2, final_mdil = nl, lr, l2, mdil
    std = float(np.std([r["ndcg20"] for r in fold_results]))
    print(f"\nCV ndcg@20: {cv_ndcg:.4f}  std={std:.4f}  mean_iter={final_iter}")
    sweep_log = None

# ── Refit on all data ─────────────────────────────────────────────────────────

print(f"\nRefitting on all {n_turns} turns  iter={final_iter} "
      f"nl={final_nl}  lr={final_lr}  l2={final_l2}  mdil={final_mdil} ...")
all_turns = list(range(n_turns))
Xall, yall, gall = slice_for_turns(all_turns)
dfull = lgb.Dataset(Xall, label=yall, group=gall, feature_name=feature_cols)

final = lgb.train(
    build_params(final_nl, final_lr, final_l2, final_mdil),
    dfull,
    num_boost_round=final_iter,
    callbacks=[lgb.log_evaluation(period=50)],
)

# ── Save ──────────────────────────────────────────────────────────────────────

out = Path(args.out)
out.parent.mkdir(parents=True, exist_ok=True)
final.save_model(str(out))

# Feature importance (gain and split)
gain_imp  = final.feature_importance(importance_type="gain")
split_imp = final.feature_importance(importance_type="split")
feat_gain  = sorted(zip(feature_cols, gain_imp,  split_imp), key=lambda x: -x[1])
feat_split = sorted(zip(feature_cols, gain_imp,  split_imp), key=lambda x: -x[2])

print(f"\nTop-10 by gain:")
for name, gain, spl in feat_gain[:10]:
    print(f"  {name:<28}  gain={gain:.0f}  splits={spl}")
print(f"\nTop-10 by split count:")
for name, gain, spl in feat_split[:10]:
    print(f"  {name:<28}  gain={gain:.0f}  splits={spl}")

# Zero-importance features (sanity check for new features)
zero = [(n, g, s) for n, g, s in zip(feature_cols, gain_imp, split_imp)
        if g == 0 and s == 0]
if zero:
    print(f"\nWARNING: {len(zero)} features with zero importance: "
          f"{[n for n,_,_ in zero]}")

side = out.with_suffix(".meta.json")
with open(side, "w") as f:
    json.dump({
        "feature_cols":   feature_cols,
        "cv_ndcg20_mean": cv_ndcg,
        "cv_ndcg20_std":  float(np.std([r["ndcg20"] for r in fold_results])),
        "fold_results":   fold_results,
        "final_iter":     final_iter,
        "params": {
            "num_leaves":           final_nl,
            "lr":                   final_lr,
            "lambda_l2":            final_l2,
            "min_data_in_leaf":     final_mdil,
            "min_sum_hessian":      args.min_sum_hessian,
            "path_smooth":          args.path_smooth,
            "feature_fraction":     args.feature_fraction,
            "bagging_fraction":     args.bagging_fraction,
            "truncation_level":     args.truncation_level,
        },
        "feature_importance": {
            "gain":  {n: int(g) for n, g, _ in feat_gain},
            "split": {n: int(s) for n, _, s in feat_split},
        },
        **({"sweep_results": sweep_log} if sweep_log else {}),
    }, f, indent=2)

print(f"\nSaved booster : {out}")
print(f"Saved meta    : {side}")
