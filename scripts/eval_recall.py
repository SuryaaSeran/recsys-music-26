"""Evaluate fusion recall on dev: recall@K over music turns.

  python scripts/eval_recall.py --n 1000 --device mps
"""
import argparse
import glob
import json
import os
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

from src.recall.fusion import FusionRetriever, render_query

HUB = os.path.expanduser("~/.cache/huggingface/hub")
DS = "datasets--talkpl-ai--TalkPlayData-Challenge-Dataset"
KS = (20, 50, 100, 200, 500, 1000)


def dev_turns(n, seed=0):
    paths = sorted(glob.glob(f"{HUB}/{DS}/snapshots/*/data/test-*.parquet"))
    df = pd.concat([pd.read_parquet(p) for p in paths], ignore_index=True)
    out = []
    for r in df.to_dict("records"):
        conv = list(r["conversations"])
        for i, t in enumerate(conv):
            if t["role"] == "music":
                out.append((render_query(r, conv[:i]), t["content"]))
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(out), min(n, len(out)), replace=False)
    return [out[i] for i in idx]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=1000)
    ap.add_argument("--device", default="mps")
    ap.add_argument("--k", type=int, default=1000)
    a = ap.parse_args()

    turns = dev_turns(a.n)
    queries = [q for q, _ in turns]
    golds = [g for _, g in turns]

    r = FusionRetriever.build(device=a.device)
    pools = r.retrieve(queries, k=a.k, per_source_k=a.k)

    rec = {k: 0 for k in KS if k <= a.k}
    for pool, gold in zip(pools, golds):
        if gold in pool:
            p = pool.index(gold)
            for k in rec:
                if p < k:
                    rec[k] += 1
    n = len(golds)
    report = {f"recall@{k}": round(rec[k] / n, 4) for k in rec}
    report["n"] = n
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
