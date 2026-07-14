"""CPU-only micro-benchmarks for every inference component of the v2 Blind B
system (v8d recall + LTR rescore). All torch models are forced to device=cpu so
the numbers hold on any commodity CPU box, not just Apple Silicon with MPS.

Usage:
  OMP_NUM_THREADS=4 python scripts/analysis/bench_cpu_components.py

Reports median / p90 latency over N repeats per component plus artifact sizes
and process RSS after all loads.
"""
import json
import os

# Avoid the HuggingFace fast-tokenizer Rust threadpool deadlock (0% CPU hang
# during .encode) and unbounded thread contention on CPU.
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import resource
import statistics
import time
from pathlib import Path

import numpy as np

try:
    import torch
    torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "4")))
except Exception:
    pass

ROOT = Path(__file__).resolve().parents[2]
os.chdir(ROOT)

SKIP_QWEN_ENCODE = os.environ.get("SKIP_QWEN_ENCODE") == "1"

N = 20  # repeats per component

ANCHOR = (
    "query: [PROFILE] 25-34 · GB · female · Anglo-American Rock · en "
    "[GOAL] play highly popular alternative rock tracks  (broad discovery) "
    "[T1] USER: Play Heart-Shaped Box by Nirvana | REC: Heart-Shaped Box – Nirvana | "
    "ASST: Channeling that raw In Utero-era grunge energy for you. | REACTION: liked "
    "[T2] USER: Great! What other popular alternative rock tracks do you recommend? | "
    "REC: Fluorescent Adolescent – Arctic Monkeys | ASST: Since the grunge landed, here "
    "is a 2007 indie-rock staple with the same guitar-forward drive. | REACTION: liked "
    "[NOW] USER: Another solid track. Can you recommend another highly popular "
    "alternative rock track from the same era?"
)
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
    return {
        "median_ms": round(1000 * statistics.median(ts), 1),
        "p90_ms": round(1000 * ts[int(0.9 * len(ts)) - 1], 1),
        "n": n,
    }


results = {}

# ---------------------------------------------------------------- hardware
import subprocess

def _sysctl(key):
    try:
        return subprocess.run(["sysctl", "-n", key], capture_output=True,
                              text=True, timeout=5).stdout.strip()
    except Exception:
        return "?"

results["hardware"] = {
    "chip": _sysctl("machdep.cpu.brand_string"),
    "mem_bytes": _sysctl("hw.memsize"),
    "ncpu": _sysctl("hw.ncpu"),
    "omp_threads": os.environ.get("OMP_NUM_THREADS", "default"),
}
print(json.dumps({"hardware": results["hardware"]}, indent=2))

# ---------------------------------------------------------------- sizes
def du(path):
    p = Path(path)
    if not p.exists():
        return None
    if p.is_file():
        return p.stat().st_size
    return sum(f.stat().st_size for f in p.rglob("*") if f.is_file())


sizes = {}
for name, path in {
    "tt_model (e5-base LoRA-merged)": "models/twotower_v8d/final",
    "tt_index (47071x768 f32)": "cache/twotower_v8d",
    "qwen3_meta index": "cache/qwen3_meta",
    "qwen3_lyrics index": "cache/qwen3_lyrics",
    "clap index": "cache/clap",
    "cooccur table": "cache/cooccur/next_song_leakfree_6k_excluded.npz",
    "ltr s3cap (67 feat)": "models/ltr/ltr_v8d_s3cap_nl31_lr0p08.txt",
    "ltr tier1 (60 feat)": "models/ltr/ltr_v8d_tier1_nl31_lr0p08.txt",
    "sasrec ckpt": "models/sasrec/sasrec_runC2_L2C64/best_model.pth",
    "semantic ids": "cache/semantic_ids/runC2_attributes_L2C64",
    "bm25 index": "cache/bm25/track_metadata",
}.items():
    b = du(path)
    sizes[name] = f"{b/1e6:.1f} MB" if b else "MISSING"
results["artifact_sizes"] = sizes
print(json.dumps({"artifact_sizes": sizes}, indent=2))

# ---------------------------------------------------------------- TT encode (CPU)
from sentence_transformers import SentenceTransformer  # noqa: E402

tt = SentenceTransformer("models/twotower_v8d/final", device="cpu")
results["tt_query_encode_cpu"] = timeit(
    lambda: tt.encode([ANCHOR], normalize_embeddings=True, show_progress_bar=False)
)
print("tt_query_encode_cpu", results["tt_query_encode_cpu"])

# ---------------------------------------------------------------- passage encode (index build extrapolation)
PASSAGE = ("passage: D Is For Dangerous by Arctic Monkeys | Album: Favourite Worst "
           "Nightmare | Tags: indie rock alternative 2000s garage rock | 2007")
results["tt_passage_encode_batch32_cpu"] = timeit(
    lambda: tt.encode([PASSAGE] * 32, normalize_embeddings=True, show_progress_bar=False),
    n=5,
)
print("tt_passage_encode_batch32_cpu", results["tt_passage_encode_batch32_cpu"])

# ---------------------------------------------------------------- ANN brute force
tt_embs = np.load("cache/twotower_v8d/track_embeddings.npy")  # (47071, 768) f32
q = tt.encode([ANCHOR], normalize_embeddings=True, show_progress_bar=False)[0]


def ann():
    sims = tt_embs @ q
    idx = np.argpartition(-sims, 2000)[:2000]
    idx[np.argsort(-sims[idx])]


results["ann_matmul_top2000 (47071x768)"] = timeit(ann)
print("ann", results["ann_matmul_top2000 (47071x768)"])

# ---------------------------------------------------------------- BM25
# BM25/LTR are placed BEFORE the Qwen block: Qwen3-Embedding-0.6B's custom
# modeling code deadlocks (0% CPU) on CPU encode in this environment, so the
# reliable components must not sit behind it. Qwen is loaded last and guarded.
import bm25s  # noqa: E402

bm25 = bm25s.BM25.load("cache/bm25/track_metadata", load_corpus=False)
tok = bm25s.tokenize([BM25_QUERY.lower()], show_progress=False)
results["bm25_retrieve_top500"] = timeit(
    lambda: bm25.retrieve(tok, k=500, show_progress=False)
)
print("bm25", results["bm25_retrieve_top500"])

# ---------------------------------------------------------------- LightGBM score
import lightgbm as lgb  # noqa: E402

booster = lgb.Booster(model_file="models/ltr/ltr_v8d_s3cap_nl31_lr0p08.txt")
X = np.random.rand(3100, booster.num_feature()).astype(np.float32)
results[f"ltr_score_3100cands_{booster.num_feature()}feat"] = timeit(
    lambda: booster.predict(X)
)
print("ltr", results[f"ltr_score_3100cands_{booster.num_feature()}feat"])

# RSS with e5-base + BM25 + LTR loaded (Qwen NOT yet loaded)
results["rss_mb_without_qwen"] = round(
    resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6, 0)
print("rss (no qwen) MB:", results["rss_mb_without_qwen"])

# write partial results now, so a Qwen hang cannot lose the reliable numbers
out = ROOT / "exp/analysis/bench_cpu_components.json"
out.parent.mkdir(parents=True, exist_ok=True)
json.dump(results, open(out, "w"), indent=2)
print(f"wrote partial {out}")

# ---------------------------------------------------------------- Qwen3 (guarded)
SKIP_QWEN = os.environ.get("SKIP_QWEN") == "1"
qwen_embs = np.load("cache/qwen3_meta/track_embeddings.npy")

if not SKIP_QWEN:
    qwen = SentenceTransformer("Qwen/Qwen3-Embedding-0.6B", trust_remote_code=True, device="cpu")
    QWEN_INSTR = "Instruct: Given a music listener's request, retrieve relevant music tracks\nQuery: "
    if SKIP_QWEN_ENCODE:
        results["qwen3_0.6b_query_encode_cpu"] = "skipped (SKIP_QWEN_ENCODE=1)"
        print("qwen3 encode skipped; model still loaded for RSS realism")
        qq = qwen_embs[0] / np.linalg.norm(qwen_embs[0])
    else:
        results["qwen3_0.6b_query_encode_cpu"] = timeit(
            lambda: qwen.encode([QWEN_INSTR + BM25_QUERY], show_progress_bar=False), n=10
        )
        print("qwen3", results["qwen3_0.6b_query_encode_cpu"])
        qq = qwen.encode([QWEN_INSTR + BM25_QUERY], show_progress_bar=False)[0]
        qq = qq / np.linalg.norm(qq)
else:
    results["qwen3_0.6b_query_encode_cpu"] = "skipped (SKIP_QWEN=1, model not loaded)"
    print("qwen3 fully skipped")
    qq = qwen_embs[0] / np.linalg.norm(qwen_embs[0])


def qwen_ann():
    sims = qwen_embs @ qq
    np.argpartition(-sims, 500)[:500]


results[f"qwen_ann_top500 ({qwen_embs.shape[0]}x{qwen_embs.shape[1]})"] = timeit(qwen_ann)
print("qwen_ann", results[f"qwen_ann_top500 ({qwen_embs.shape[0]}x{qwen_embs.shape[1]})"])

# ---------------------------------------------------------------- final RSS
rss_mb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6  # bytes on macOS
results["process_peak_rss_mb"] = round(rss_mb, 0)
print("peak RSS MB:", results["process_peak_rss_mb"])

json.dump(results, open(out, "w"), indent=2)
print(f"\nwrote {out}")
