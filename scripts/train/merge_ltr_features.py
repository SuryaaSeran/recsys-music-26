"""Merge two or more LTR feature NPZ files produced by parallel feature dump workers.

Groups are re-indexed to be contiguous across shards. Feature columns must match.

Usage:
    python scripts/train/merge_ltr_features.py \
        exp/analysis/features_shard0.npz \
        exp/analysis/features_shard1.npz \
        --out exp/analysis/features_merged.npz
"""
import argparse
import json
import numpy as np
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("shards", nargs="+", help="NPZ shard files to merge, in order.")
parser.add_argument("--out", required=True, help="Output merged NPZ path.")
args = parser.parse_args()

Xs, ys, gs, metas = [], [], [], []
feature_cols = None
group_offset = 0

for path in args.shards:
    p = Path(path)
    print(f"Loading {p}...")
    data = np.load(p, allow_pickle=True)
    X = data["X"].astype(np.float32)
    y = data["y"].astype(np.int32)
    g = data["group"].astype(np.int32) + group_offset
    fc = list(data["feature_cols"])

    if feature_cols is None:
        feature_cols = fc
    else:
        assert fc == feature_cols, f"Feature column mismatch: {fc} vs {feature_cols}"

    sidecar = p.with_suffix(".meta.json")
    with open(sidecar) as f:
        meta = json.load(f)
    turn_meta = meta["turn_meta"]

    Xs.append(X); ys.append(y); gs.append(g); metas.extend(turn_meta)
    group_offset += int(g.max()) + 1
    print(f"  {X.shape[0]:,} rows  {int(g.max())+1 - (g.min()):,} groups")

X_out = np.concatenate(Xs, axis=0)
y_out = np.concatenate(ys, axis=0)
g_out = np.concatenate(gs, axis=0)

out = Path(args.out)
out.parent.mkdir(parents=True, exist_ok=True)
np.savez_compressed(out, X=X_out, y=y_out, group=g_out,
                    feature_cols=np.array(feature_cols))
sidecar_out = out.with_suffix(".meta.json")
with open(sidecar_out, "w") as f:
    json.dump({"feature_cols": feature_cols,
               "n_turns": int(g_out.max()) + 1,
               "n_rows": int(X_out.shape[0]),
               "turn_meta": metas}, f)

print(f"\nMerged: {X_out.shape} -> {out}")
print(f"Sidecar: {sidecar_out}")
print(f"Total turns: {int(g_out.max()) + 1}  pos_rate: {y_out.mean():.5f}")
