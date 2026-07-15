"""
Hyperparameter sweep for the LTR LambdaMART trainer.

Drives `scripts/train/train_ltr_lightgbm.py` over a small grid of
`num_leaves` x `lr`, captures the printed CV ndcg@20 mean per config,
and writes a CSV summary.

Usage:
    python scripts/train/sweep_ltr_lightgbm.py \
        --features exp/analysis/ltr_train_features.npz \
        --out_dir models/ltr/sweep \
        --csv exp/analysis/ltr_sweep.csv
"""
import argparse
import csv
import re
import subprocess
import sys
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("--features", required=True)
parser.add_argument("--out_dir", default="models/ltr/sweep")
parser.add_argument("--csv",     default="exp/analysis/ltr_sweep.csv")
parser.add_argument("--n_folds", type=int, default=5)
parser.add_argument("--num_iter", type=int, default=1000)
parser.add_argument("--early_stop", type=int, default=50)
args = parser.parse_args()

GRID = [
    (nl, lr)
    for nl in (31, 63, 127)
    for lr in (0.03, 0.05, 0.08)
]

Path(args.out_dir).mkdir(parents=True, exist_ok=True)
Path(args.csv).parent.mkdir(parents=True, exist_ok=True)

CV_FOLD_RE   = re.compile(r"CV ndcg@20 per fold:\s*(\[[^\]]+\])")
CV_MEAN_RE   = re.compile(r"CV mean ndcg@20:\s*([0-9.]+)\s*\(std\s*([0-9.]+)\)")
BEST_ITER_RE = re.compile(r"CV mean best_iter:\s*([0-9]+)")

rows = []
for nl, lr in GRID:
    tag = f"nl{nl}_lr{lr:g}".replace(".", "p")
    out = Path(args.out_dir) / f"ltr_{tag}.txt"
    cmd = [
        sys.executable, "scripts/train/train_ltr_lightgbm.py",
        "--features", args.features,
        "--out", str(out),
        "--n_folds", str(args.n_folds),
        "--num_leaves", str(nl),
        "--lr", str(lr),
        "--num_iter", str(args.num_iter),
        "--early_stop", str(args.early_stop),
    ]
    print(f"\n=== Config {tag}: num_leaves={nl} lr={lr} ===")
    result = subprocess.run(cmd, capture_output=True, text=True)
    log = result.stdout + "\n" + result.stderr
    if result.returncode != 0:
        print(f"FAILED (rc={result.returncode}):\n{log[-2000:]}")
        rows.append({"num_leaves": nl, "lr": lr, "cv_mean": "", "cv_std": "",
                     "best_iter": "", "model": str(out), "status": "fail"})
        continue
    m_mean = CV_MEAN_RE.search(log)
    m_iter = BEST_ITER_RE.search(log)
    if not m_mean:
        print(f"Could not parse CV mean. Tail:\n{log[-1500:]}")
        rows.append({"num_leaves": nl, "lr": lr, "cv_mean": "", "cv_std": "",
                     "best_iter": "", "model": str(out), "status": "parse_fail"})
        continue
    cv_mean = float(m_mean.group(1))
    cv_std  = float(m_mean.group(2))
    best_it = int(m_iter.group(1)) if m_iter else -1
    print(f"  CV mean nDCG@20: {cv_mean:.4f} (std {cv_std:.4f})  best_iter={best_it}")
    rows.append({"num_leaves": nl, "lr": lr, "cv_mean": cv_mean, "cv_std": cv_std,
                 "best_iter": best_it, "model": str(out), "status": "ok"})

# Write CSV
with open(args.csv, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=["num_leaves", "lr", "cv_mean", "cv_std",
                                       "best_iter", "model", "status"])
    w.writeheader()
    w.writerows(rows)

# Print best
ok = [r for r in rows if r["status"] == "ok"]
if ok:
    best = max(ok, key=lambda r: r["cv_mean"])
    print(f"\n=== Best ===  num_leaves={best['num_leaves']}  lr={best['lr']}  "
          f"cv_mean={best['cv_mean']:.4f}  best_iter={best['best_iter']}  "
          f"model={best['model']}")
print(f"\nWrote {len(rows)} rows to {args.csv}")
