"""Isolated CPU benchmark for BM25 retrieve and LightGBM scoring.

Kept separate from bench_cpu_components.py because importing torch/transformers
in the same process as bm25s + lightgbm triggered a native-threadpool deadlock
on this machine. This process imports neither.
"""
import json
import os
import resource
import statistics
import time
from pathlib import Path

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
os.chdir(ROOT)
N = 20

BM25_QUERY = (
    "play highly popular alternative rock tracks Anglo-American Rock "
    "Heart-Shaped Box Nirvana grunge alternative rock 90s Fluorescent Adolescent "
    "Arctic Monkeys indie rock alternative 2000s another highly popular alternative rock track"
)


def timeit(fn, n=N, warmup=2):
    for _ in range(warmup):
        fn()
    ts = []
    for _ in range(n):
        t0 = time.perf_counter()
        fn()
        ts.append(time.perf_counter() - t0)
    ts.sort()
    return {"median_ms": round(1000 * statistics.median(ts), 1),
            "p90_ms": round(1000 * ts[int(0.9 * len(ts)) - 1], 1), "n": n}


results = {}

import bm25s

bm25 = bm25s.BM25.load("cache/bm25/track_metadata", load_corpus=False)
tok = bm25s.tokenize([BM25_QUERY.lower()], show_progress=False)
results["bm25_retrieve_top500"] = timeit(
    lambda: bm25.retrieve(tok, k=500, show_progress=False))
print("bm25", results["bm25_retrieve_top500"], flush=True)

import lightgbm as lgb

booster = lgb.Booster(model_file="models/ltr/ltr_v8d_s3cap_nl31_lr0p08.txt")
X = np.random.rand(3100, booster.num_feature()).astype(np.float32)
results[f"ltr_score_3100cands_{booster.num_feature()}feat"] = timeit(
    lambda: booster.predict(X))
print("ltr", results[f"ltr_score_3100cands_{booster.num_feature()}feat"], flush=True)

results["rss_mb"] = round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6, 0)
print("rss MB (bm25+ltr only):", results["rss_mb"], flush=True)

out = ROOT / "exp/analysis/bench_bm25_ltr.json"
out.parent.mkdir(parents=True, exist_ok=True)
json.dump(results, open(out, "w"), indent=2)
print("wrote", out, flush=True)
