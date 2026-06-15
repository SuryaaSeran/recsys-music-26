#!/usr/bin/env python3
"""Fast failure guard for LTR booster before dev eval.

Checks:
  1. Booster file exists and size > 0
  2. Meta JSON has final CV nDCG + best_iteration
  3. Feature count matches expected
  4. No all-zero or NaN semantic feature columns in training data
  5. Feature importances — semantic features not dead

Exit code 0 = all clear. Exit code 1 = problem found.
"""
import argparse
import json
import sys
from pathlib import Path

import lightgbm as lgb
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--booster", required=True)
ap.add_argument("--features", required=True, help="NPZ feature dump used for training")
ap.add_argument("--expected_features", type=int, default=64)
ap.add_argument("--sem_feature_prefix", default="cand_sem",
                help="Prefix of semantic features to check for non-zero importance")
ap.add_argument("--sem_ids_dir", default="cache/semantic_ids/runC2_attributes_L2C64",
                help="Semantic IDs directory — used for check 7 (mapping version match)")
args = ap.parse_args()

PASS = "\033[92mPASS\033[0m"
FAIL = "\033[91mFAIL\033[0m"
WARN = "\033[93mWARN\033[0m"
ok = True

def check(label, cond, detail="", warn_only=False):
    global ok
    tag = PASS if cond else (WARN if warn_only else FAIL)
    print(f"[{tag}] {label}" + (f" — {detail}" if detail else ""))
    if not cond and not warn_only:
        ok = False

# 1. Booster file exists and non-empty
bpath = Path(args.booster)
check("Booster file exists", bpath.exists(), str(bpath))
if bpath.exists():
    check("Booster file size > 0", bpath.stat().st_size > 0,
          f"{bpath.stat().st_size:,} bytes")

# 2. Meta JSON
meta_path = bpath.with_suffix(".meta.json")
cv_ndcg = None
if meta_path.exists():
    meta = json.load(open(meta_path))
    cv_ndcg = meta.get("cv_ndcg20") or meta.get("cv_ndcg20_mean")
    best_iter = meta.get("best_iteration") or meta.get("mean_iter") or meta.get("final_iter")
    check("Meta has cv_ndcg20", cv_ndcg is not None, f"cv_ndcg20={cv_ndcg}")
    check("Meta has best_iteration", best_iter is not None, f"best_iter={best_iter}")
    check("CV nDCG@20 > 0.30", cv_ndcg is not None and cv_ndcg > 0.30,
          f"cv_ndcg20={cv_ndcg}")
else:
    check("Meta JSON exists", False, str(meta_path), warn_only=True)

# 3. Load booster and check features
if bpath.exists() and bpath.stat().st_size > 0:
    try:
        bst = lgb.Booster(model_file=str(bpath))
        feat_names = bst.feature_name()
        n_feat = len(feat_names)
        check(f"Feature count = {args.expected_features}", n_feat == args.expected_features,
              f"got {n_feat}")

        # 5. Feature importances
        gains = bst.feature_importance(importance_type="gain")
        gain_map = dict(zip(feat_names, gains))

        sem_feats = {k: v for k, v in gain_map.items() if args.sem_feature_prefix in k}
        total_gain = sum(gains)

        print(f"\n  Semantic feature importances (gain):")
        for fname, g in sorted(sem_feats.items(), key=lambda x: -x[1]):
            pct = 100 * g / total_gain if total_gain > 0 else 0
            print(f"    {fname:<35} {g:>10,.0f}  ({pct:.2f}%)")

        sem_total = sum(sem_feats.values())
        sem_pct = 100 * sem_total / total_gain if total_gain > 0 else 0
        check("Semantic features have non-zero total importance",
              sem_total > 0, f"{sem_pct:.2f}% of total gain")
        check("At least 2 semantic features with gain > 0",
              sum(1 for v in sem_feats.values() if v > 0) >= 2,
              f"{sum(1 for v in sem_feats.values() if v > 0)} non-zero", warn_only=True)

        # Top-10 overall
        top10 = sorted(gain_map.items(), key=lambda x: -x[1])[:10]
        print(f"\n  Top-10 features by gain:")
        for fname, g in top10:
            pct = 100 * g / total_gain if total_gain > 0 else 0
            print(f"    {fname:<35} {g:>10,.0f}  ({pct:.2f}%)")

        zero_imp = [k for k, v in gain_map.items() if v == 0]
        print(f"\n  Zero-importance features ({len(zero_imp)}): {zero_imp}")
        check("Fewer than 6 zero-importance features", len(zero_imp) < 6,
              f"got {len(zero_imp)}", warn_only=True)

    except Exception as e:
        check("Booster loads cleanly", False, str(e))

# 4. Feature dump: check semantic columns for NaN / all-zero
feat_path = Path(args.features)
if feat_path.exists():
    npz = np.load(str(feat_path), allow_pickle=True)
    X = npz["X"]
    meta_feat = json.load(open(str(feat_path).replace(".npz", ".meta.json")))
    feat_cols = meta_feat.get("feature_cols", [])

    sem_indices = [i for i, n in enumerate(feat_cols) if args.sem_feature_prefix in n]
    if sem_indices:
        sem_block = X[:, sem_indices]
        n_nan = int(np.isnan(sem_block).sum())
        n_inf = int(np.isinf(sem_block).sum())
        all_zero_cols = [feat_cols[i] for idx, i in enumerate(sem_indices)
                         if np.all(sem_block[:, idx] == 0)]
        check("No NaN in semantic feature columns", n_nan == 0, f"{n_nan} NaNs")
        check("No Inf in semantic feature columns", n_inf == 0, f"{n_inf} Infs")
        check("No all-zero semantic feature columns", len(all_zero_cols) == 0,
              f"all-zero: {all_zero_cols}", warn_only=len(all_zero_cols) <= 1)
        # fraction non-zero
        for idx, col_i in enumerate(sem_indices):
            col = sem_block[:, idx]
            frac_nz = (col != 0).mean()
            print(f"    {feat_cols[col_i]:<35} non-zero={frac_nz:.3%}")
    else:
        check("Semantic columns found in dump", False,
              f"prefix '{args.sem_feature_prefix}' not in {feat_cols[:5]}...")
else:
    check("Feature NPZ exists", False, str(feat_path), warn_only=True)

# 6. Candidate pool count / no Stage 3 contamination
#    Verify the feature dump has expected row count (≈ 77–78M for 6K sessions, no Stage 3).
if feat_path.exists():
    npz2 = np.load(str(feat_path), allow_pickle=True)
    n_rows = npz2["X"].shape[0]
    # Stage 3 expansion adds ~3484 cands/turn × 24718 turns ≈ +86M rows
    # Without Stage 3 we expect ~77–80M rows
    in_expected = 70_000_000 < n_rows < 85_000_000
    check("Row count consistent with no-Stage-3 pool",
          in_expected, f"{n_rows:,} rows")

# 7. Semantic ID mapping version match
#    Compare track_ids.npy fingerprint between sem_ids_dir (eval) and
#    what the feature dump meta records as semantic_ids_dir.
import hashlib
sem_dir = Path(args.sem_ids_dir)
tid_path = sem_dir / "track_ids.npy"
if tid_path.exists() and feat_path.exists():
    meta_feat2 = json.load(open(str(feat_path).replace(".npz", ".meta.json")))
    dump_sem_dir = meta_feat2.get("semantic_ids_dir", "")
    # Check the paths match (same directory name = same artifact)
    same_dir = Path(dump_sem_dir).name == sem_dir.name if dump_sem_dir else False
    check("Feature dump uses same sem_ids_dir as eval",
          same_dir,
          f"dump={Path(dump_sem_dir).name if dump_sem_dir else 'NOT RECORDED'} eval={sem_dir.name}")

    # Fingerprint the track_ids mapping itself
    tids_bytes = np.load(str(tid_path), allow_pickle=True).tobytes()
    tid_hash = hashlib.md5(tids_bytes).hexdigest()[:8]
    codes_bytes = np.load(str(sem_dir / "semantic_ids.npy")).tobytes()
    codes_hash = hashlib.md5(codes_bytes).hexdigest()[:8]
    print(f"    Semantic mapping fingerprint: track_ids={tid_hash} codes={codes_hash}")
    print(f"    (record these and verify they match at eval time)")
else:
    check("Semantic ID files exist for version check", False,
          f"{tid_path}", warn_only=True)

print()
if ok:
    print(f"[{PASS}] All checks passed — safe to run dev eval")
    sys.exit(0)
else:
    print(f"[{FAIL}] One or more checks failed — fix before dev eval")
    sys.exit(1)
