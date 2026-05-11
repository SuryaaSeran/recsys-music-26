"""
Recall audit: where does the gold track sit in BM25 vs each dense signal?

Reads cache/dev_embeddings/ (precomputed per-turn query embeddings + BM25 cands)
and computes the global rank of the gold track in each signal's index. Also
classifies each turn into a query bucket and warm/cold for segmented analysis.

No reranking, no fusion, no scoring changes. Pure measurement.

Outputs:
  exp/analysis/recall_audit.jsonl    -- one record per turn
  exp/analysis/recall_audit_summary.txt -- aggregated Recall@K tables

Usage:
    python scripts/inference/audit_recall.py
"""
import json
import re
import numpy as np
from pathlib import Path
from collections import defaultdict

from datasets import load_dataset

CACHE = Path("cache/dev_embeddings")
OUT_DIR = Path("exp/analysis")
OUT_DIR.mkdir(parents=True, exist_ok=True)

K_LIST = [50, 100, 200, 500, 1000]
DENSE_K = max(K_LIST)


def load_idx(path):
    with open(path) as f:
        ids = json.load(f)
    return ids, {tid: i for i, tid in enumerate(ids)}


print("Loading dev_embeddings cache...")
with open(CACHE / "turns.json") as f:
    turns = json.load(f)
bm25_cands = np.load(CACHE / "bm25_cands.npy")    # (N, 500) int32 indices into bm25_track_ids
tt_q       = np.load(CACHE / "tt_embs.npy")
qwen_q     = np.load(CACHE / "qwen_embs.npy")
clap_q     = np.load(CACHE / "clap_embs.npy")
cf_q       = np.load(CACHE / "cf_user.npy")
N = len(turns)
print(f"Turns: {N}")

print("Loading track-side indexes...")
with open("cache/bm25/track_metadata/track_ids.json") as f:
    bm25_track_ids = json.load(f)
bm25_set = set(bm25_track_ids)

tt_embs = np.load("cache/twotower_v3/track_embeddings.npy")
tt_ids, tt_id2idx = load_idx("cache/twotower_v3/track_ids.json")

qwen_meta_embs = np.load("cache/qwen3_meta/track_embeddings.npy")
qwen_meta_ids, qwen_meta_id2idx = load_idx("cache/qwen3_meta/track_ids.json")

qwen_lyrics_embs = np.load("cache/qwen3_lyrics/track_embeddings.npy")
qwen_lyrics_ids, qwen_lyrics_id2idx = load_idx("cache/qwen3_lyrics/track_ids.json")

clap_embs = np.load("cache/clap/track_embeddings.npy")
clap_ids, clap_id2idx = load_idx("cache/clap/track_ids.json")

cf_track_embs = np.load("cache/cf_bpr/track_embeddings.npy")
cf_track_ids, cf_track_id2idx = load_idx("cache/cf_bpr/track_ids.json")


# -- query-bucket classifier (recover latest_user + music_history from dev set) --
print("Loading dev sessions for query-bucket classification...")
ds = load_dataset("talkpl-ai/TalkPlayData-Challenge-Dataset")
sessions = list(ds["test"])

ARTIST_RE = re.compile(r'"[^"]+"|\bby\s+\w+', re.IGNORECASE)
MOOD_WORDS = {
    "sad","happy","chill","energetic","calm","party","sleep","study","workout",
    "mellow","upbeat","melancholy","romantic","angry","nostalgic","dreamy","relaxing",
    "soft","hard","loud","quiet","intense","dark","bright","cheerful",
}
LYRICS_WORDS = {"lyrics","meaning","sing","words","sung","singing"}
SIM_PHRASES = ["like this","similar","more like","same vibe","another one","like that"]


def classify(latest_user: str, n_history: int) -> str:
    q = (latest_user or "").lower()
    if any(p in q for p in SIM_PHRASES):
        return "more_like_this"
    if any(w in q.split() for w in LYRICS_WORDS):
        return "lyrics"
    if ARTIST_RE.search(latest_user or ""):
        return "specific"
    toks = set(re.findall(r"\w+", q))
    if toks & MOOD_WORDS:
        return "mood"
    if n_history >= 2:
        return "history_driven"
    return "generic"


# Walk dev set in same order as precompute to attach (bucket, n_history) per turn
print("Aligning bucket/history info to cached turns...")
bucket_per_turn = [None] * N
n_hist_per_turn = [0] * N
idx = 0
for item in sessions:
    convs = item["conversations"]
    music_history: list[str] = []
    text_history:  list[str] = []
    for turn in convs:
        if turn["role"] != "music":
            if turn["role"] in ("user","assistant"):
                text_history.append(turn["content"])
            continue
        latest_user = text_history[-1] if text_history else ""
        bucket_per_turn[idx] = classify(latest_user, len(music_history))
        n_hist_per_turn[idx] = len(music_history)
        music_history.append(turn["content"])
        idx += 1
assert idx == N, f"alignment mismatch: {idx} != {N}"


# -- compute BM25 rank of gold per turn --
print("Computing BM25 ranks...")
bm25_rank = np.full(N, -1, dtype=np.int32)  # -1 = miss
gold_in_bm25_index = np.zeros(N, dtype=bool)
for i, t in enumerate(turns):
    gold = t["gold"]
    gold_in_bm25_index[i] = gold in bm25_set
    cands_idx = bm25_cands[i]
    # find index where bm25_track_ids[cands_idx[j]] == gold
    # cheaper: build set of cand tids
    for j, tk_idx in enumerate(cands_idx):
        if tk_idx < 0:
            break
        if bm25_track_ids[int(tk_idx)] == gold:
            bm25_rank[i] = j + 1
            break


# -- compute global dense rank of gold per signal --
def dense_ranks(query_embs, track_embs, gold_idxs, batch=256, signal_name=""):
    """For each turn, return rank (1-indexed) of gold_idxs[i] under cosine vs all tracks.
    -1 if gold not in this index (gold_idxs[i] == -1)."""
    N = len(query_embs)
    ranks = np.full(N, -1, dtype=np.int32)
    for s in range(0, N, batch):
        e = min(N, s + batch)
        sub_idxs = gold_idxs[s:e]
        valid = sub_idxs >= 0
        if not valid.any():
            continue
        scores = query_embs[s:e] @ track_embs.T  # (b, N_track)
        for k in range(e - s):
            gi = sub_idxs[k]
            if gi < 0:
                continue
            gs = scores[k, gi]
            ranks[s + k] = int((scores[k] > gs).sum()) + 1
        if (s // batch) % 4 == 0:
            print(f"  {signal_name}: {e}/{N}")
    return ranks


def gold_idx_array(id2idx):
    arr = np.full(N, -1, dtype=np.int64)
    for i, t in enumerate(turns):
        arr[i] = id2idx.get(t["gold"], -1)
    return arr


print("TT global ranks...")
tt_rank = dense_ranks(tt_q, tt_embs, gold_idx_array(tt_id2idx), batch=512, signal_name="tt")

print("Qwen-meta global ranks...")
qm_rank = dense_ranks(qwen_q, qwen_meta_embs, gold_idx_array(qwen_meta_id2idx), batch=128, signal_name="qm")

print("Qwen-lyrics global ranks...")
ql_rank = dense_ranks(qwen_q, qwen_lyrics_embs, gold_idx_array(qwen_lyrics_id2idx), batch=128, signal_name="ql")

print("CLAP global ranks...")
clap_rank = dense_ranks(clap_q, clap_embs, gold_idx_array(clap_id2idx), batch=256, signal_name="clap")

print("CF global ranks (warm only)...")
cf_gold_idx = gold_idx_array(cf_track_id2idx)
# zero out cold-start turns (cf_q == 0)
cf_q_norms = np.linalg.norm(cf_q, axis=1)
warm_mask = cf_q_norms > 1e-6
cf_gold_idx_masked = cf_gold_idx.copy()
cf_gold_idx_masked[~warm_mask] = -1
cf_rank = dense_ranks(cf_q, cf_track_embs, cf_gold_idx_masked, batch=512, signal_name="cf")


# -- write per-turn records --
print("Writing per-turn JSONL...")
records = []
for i, t in enumerate(turns):
    records.append({
        "session_id":  t["session_id"],
        "user_id":     t["user_id"],
        "turn_number": t["turn_number"],
        "gold":        t["gold"],
        "has_cf":      bool(t.get("has_cf", False)),
        "n_history":   int(n_hist_per_turn[i]),
        "bucket":      bucket_per_turn[i],
        "gold_in_bm25_index": bool(gold_in_bm25_index[i]),
        "bm25_rank":   int(bm25_rank[i]) if bm25_rank[i] > 0 else None,
        "tt_rank":     int(tt_rank[i])   if tt_rank[i]   > 0 else None,
        "qm_rank":     int(qm_rank[i])   if qm_rank[i]   > 0 else None,
        "ql_rank":     int(ql_rank[i])   if ql_rank[i]   > 0 else None,
        "clap_rank":   int(clap_rank[i]) if clap_rank[i] > 0 else None,
        "cf_rank":     int(cf_rank[i])   if cf_rank[i]   > 0 else None,
    })

with open(OUT_DIR / "recall_audit.jsonl", "w") as f:
    for r in records:
        f.write(json.dumps(r) + "\n")
print(f"Wrote {len(records)} rows to {OUT_DIR / 'recall_audit.jsonl'}")


# -- aggregate Recall@K tables --
def recall_at(rank_arr, ks, mask=None):
    if mask is None:
        mask = np.ones(len(rank_arr), dtype=bool)
    n = mask.sum()
    if n == 0:
        return {k: 0.0 for k in ks}
    return {k: float(((rank_arr > 0) & (rank_arr <= k) & mask).sum()) / n for k in ks}


def union_recall(ks, ranks_list, mask=None):
    """Recall where gold appears in top-K of ANY of the given rank arrays.
    For BM25 rank, K_bm25 is fixed at 500 (pool); dense Ks vary."""
    if mask is None:
        mask = np.ones(N, dtype=bool)
    n = mask.sum()
    if n == 0:
        return {k: 0.0 for k in ks}
    out = {}
    for k in ks:
        any_in = np.zeros(N, dtype=bool)
        for r in ranks_list:
            any_in |= (r > 0) & (r <= k)
        out[k] = float((any_in & mask).sum()) / n
    return out


arr = lambda key: np.array([rec[key] if rec[key] is not None else -1 for rec in records], dtype=np.int32)
bm25_a = arr("bm25_rank")
tt_a   = arr("tt_rank")
qm_a   = arr("qm_rank")
ql_a   = arr("ql_rank")
clap_a = arr("clap_rank")
cf_a   = arr("cf_rank")

buckets_arr = np.array([rec["bucket"] for rec in records])
warm_arr    = np.array([rec["has_cf"] for rec in records])
hist_arr    = np.array([rec["n_history"] for rec in records])
in_bm25_idx = np.array([rec["gold_in_bm25_index"] for rec in records])


lines = []
def section(title): lines.append(f"\n=== {title} ===")
def row(name, d, ks):
    cells = "  ".join(f"@{k}={d[k]:.3f}" for k in ks)
    lines.append(f"  {name:18s} {cells}")


section("Overall recall (N={} turns)".format(N))
lines.append(f"  gold in BM25 index:     {in_bm25_idx.mean():.3f}  (rest are unreachable by BM25 entirely)")
lines.append(f"  warm-start fraction:    {warm_arr.mean():.3f}")
row("BM25 (pool=500)", recall_at(bm25_a, [500]), [500])
row("TT (global)",     recall_at(tt_a,   K_LIST), K_LIST)
row("Qwen-meta",       recall_at(qm_a,   K_LIST), K_LIST)
row("Qwen-lyrics",     recall_at(ql_a,   K_LIST), K_LIST)
row("CLAP",            recall_at(clap_a, K_LIST), K_LIST)
row("CF (warm only)",  recall_at(cf_a,   K_LIST, mask=warm_arr), K_LIST)


section("Oracle union recall (BM25@500 OR dense@K)")
# For each K, union BM25-top-500 with each dense top-K
bm25_in500 = (bm25_a > 0) & (bm25_a <= 500)
for combo_name, ranks in [
    ("BM25 only",                        []),
    ("BM25 + TT",                        [tt_a]),
    ("BM25 + Qwen-meta",                 [qm_a]),
    ("BM25 + Qwen-lyrics",               [ql_a]),
    ("BM25 + CLAP",                      [clap_a]),
    ("BM25 + CF (warm)",                 [cf_a]),
    ("BM25 + TT + Qwen-meta",            [tt_a, qm_a]),
    ("BM25 + TT + QM + QL",              [tt_a, qm_a, ql_a]),
    ("BM25 + ALL (TT,QM,QL,CLAP,CF)",    [tt_a, qm_a, ql_a, clap_a, cf_a]),
]:
    line_cells = []
    for k in K_LIST:
        any_in = bm25_in500.copy()
        for r in ranks:
            any_in |= (r > 0) & (r <= k)
        line_cells.append(f"@{k}={any_in.mean():.3f}")
    lines.append(f"  {combo_name:32s} " + "  ".join(line_cells))


section("Recall by query bucket (Union BM25@500 + each dense @100)")
for bucket in ["specific","mood","lyrics","more_like_this","history_driven","generic"]:
    m = buckets_arr == bucket
    if m.sum() == 0:
        continue
    bm25_recall = (bm25_in500 & m).sum() / m.sum()
    union_all = bm25_in500.copy()
    for r in [tt_a, qm_a, ql_a, clap_a, cf_a]:
        union_all |= (r > 0) & (r <= 100)
    lines.append(
        f"  {bucket:18s} N={m.sum():4d}  BM25@500={bm25_recall:.3f}  "
        f"TT@100={((tt_a>0)&(tt_a<=100)&m).sum()/m.sum():.3f}  "
        f"QM@100={((qm_a>0)&(qm_a<=100)&m).sum()/m.sum():.3f}  "
        f"QL@100={((ql_a>0)&(ql_a<=100)&m).sum()/m.sum():.3f}  "
        f"Union={(union_all & m).sum()/m.sum():.3f}"
    )


section("Recall by warm/cold (Union BM25@500 + dense @100)")
for name, m in [("warm", warm_arr), ("cold", ~warm_arr)]:
    if m.sum() == 0:
        continue
    bm25_recall = (bm25_in500 & m).sum() / m.sum()
    union_all = bm25_in500.copy()
    for r in [tt_a, qm_a, ql_a, clap_a, cf_a]:
        union_all |= (r > 0) & (r <= 100)
    lines.append(
        f"  {name:6s} N={m.sum():4d}  BM25@500={bm25_recall:.3f}  Union={((union_all)&m).sum()/m.sum():.3f}"
    )


section("Recall by history length (Union BM25@500 + dense @100)")
for name, m in [
    ("no_history",  hist_arr == 0),
    ("hist 1-2",    (hist_arr >= 1) & (hist_arr <= 2)),
    ("hist 3-5",    (hist_arr >= 3) & (hist_arr <= 5)),
    ("hist 6+",     hist_arr >= 6),
]:
    if m.sum() == 0:
        continue
    bm25_recall = (bm25_in500 & m).sum() / m.sum()
    union_all = bm25_in500.copy()
    for r in [tt_a, qm_a, ql_a, clap_a, cf_a]:
        union_all |= (r > 0) & (r <= 100)
    lines.append(
        f"  {name:12s} N={m.sum():4d}  BM25@500={bm25_recall:.3f}  Union={((union_all)&m).sum()/m.sum():.3f}"
    )


section("BM25 misses: where does each dense signal place gold?")
miss = ~bm25_in500
lines.append(f"  BM25 misses: {miss.sum()} turns ({miss.mean():.1%})")
for name, r in [("TT", tt_a), ("QM", qm_a), ("QL", ql_a), ("CLAP", clap_a), ("CF", cf_a)]:
    found = (r > 0) & (r <= 500) & miss
    found100 = (r > 0) & (r <= 100) & miss
    lines.append(f"  {name:5s} rescues @100={found100.sum()} ({found100.sum()/max(miss.sum(),1):.1%})  "
                 f"@500={found.sum()} ({found.sum()/max(miss.sum(),1):.1%})")

text = "\n".join(lines)
print(text)
with open(OUT_DIR / "recall_audit_summary.txt", "w") as f:
    f.write(text + "\n")
print(f"\nWrote summary to {OUT_DIR / 'recall_audit_summary.txt'}")
