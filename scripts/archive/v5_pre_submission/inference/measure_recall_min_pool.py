"""
Min-pool recall sweep.

Goal: find the smallest *deduped* total pool that hits the recall target.

Pool sources, all unioned and deduped per turn:
  - BM25 top-N (N in BM25_NS)
  - Artist expansion (on/off, capped per artist)
  - TT-v6 query top-K (K in TT_KS)
  - Last-track-NN in TT space: for each of last LAST_TRACKS_N played tracks,
    add its top-M nearest tracks (M in LAST_NN_MS)

For every config, reports:
  - mean deduped pool size
  - overall recall (gold in pool)
  - per-bucket recall (specific / mood / lyrics / more_like_this / history_driven / generic)

Output: exp/analysis/recall_min_pool_grid.txt
"""
import json
import re
from pathlib import Path
from itertools import product

import bm25s
import numpy as np
from datasets import load_dataset, concatenate_datasets
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

# --- Config grid ---
BM25_NS         = [300, 500, 750]
ARTIST_FLAGS    = [False, True]
ARTIST_CAP      = 50               # max tracks added per mentioned artist
TT_KS           = [0, 250, 500, 1000]
LAST_NN_MS      = [0, 50, 100]     # per-track NN depth
LAST_TRACKS_N   = 2                # use last N played tracks for NN

BM25_CACHE  = "cache/bm25/track_metadata"
TT_MODEL    = "models/twotower_v6/final"
TT_INDEX    = "cache/twotower_v6"
OUT_PATH    = "exp/analysis/recall_min_pool_grid.txt"

BUCKET_KEYWORDS = {
    # rough query bucketing from audit_recall.py convention; replicated lightweight
    "specific":       ["who sings", "song by", "track by", "by ", "any songs by"],
    "mood":           ["mood", "vibe", "feel", "feeling", "chill", "upbeat", "sad", "happy"],
    "lyrics":         ["lyric", "lyrics", "the line", "the words"],
    "more_like_this": ["more like", "similar to", "like this", "more of"],
    "history_driven": [],   # assigned when no other bucket but history exists
    "generic":        [],   # default
}

def bucket_for(latest_user: str, has_history: bool) -> str:
    tl = (latest_user or "").lower()
    for b in ("specific", "lyrics", "more_like_this", "mood"):
        if any(k in tl for k in BUCKET_KEYWORDS[b]):
            return b
    if has_history:
        return "history_driven"
    return "generic"


# --- Load metadata + artist dict ---
print("Loading track metadata...")
meta_ds = load_dataset("talkpl-ai/TalkPlayData-Challenge-Track-Metadata")
all_tracks = concatenate_datasets([meta_ds["all_tracks"], meta_ds["test_tracks"]])
metadata_dict = {row["track_id"]: row for row in all_tracks}

artist_to_tids: dict[str, list[str]] = {}
for tid, row in metadata_dict.items():
    for a in (row.get("artist_name") or []):
        k = a.strip().lower()
        if k:
            artist_to_tids.setdefault(k, []).append(tid)
known_artists = sorted(artist_to_tids.keys(), key=len, reverse=True)
print(f"Known artists: {len(known_artists):,}")

# Pre-sort artist tracks by track_id stable order (cap deterministic)
for k in artist_to_tids:
    artist_to_tids[k] = artist_to_tids[k][:ARTIST_CAP]

_artist_re_cache = {}
def find_mentioned_artists(text: str) -> list[str]:
    tl = text.lower()
    found = []
    for a in known_artists:
        if a in tl:
            found.append(a)
            tl = tl.replace(a, " " * len(a))
    return found

def get_track_text(tid):
    row = metadata_dict.get(tid, {})
    return " ".join(filter(None, [
        (row.get("track_name") or [""])[0],
        (row.get("artist_name") or [""])[0],
        " ".join(row.get("tag_list") or []),
    ]))

def get_track_name_artist(tid):
    row = metadata_dict.get(tid, {})
    return f"{(row.get('track_name') or [''])[0]} {(row.get('artist_name') or [''])[0]}".strip()


# --- Load indexes ---
print("Loading BM25 index...")
bm25_model = bm25s.BM25.load(BM25_CACHE, load_corpus=False)
with open(f"{BM25_CACHE}/track_ids.json") as f:
    bm25_track_ids = json.load(f)
bm25_tid2idx = {t: i for i, t in enumerate(bm25_track_ids)}

print(f"Loading TT model + index: {TT_MODEL}")
tt_model = SentenceTransformer(TT_MODEL)
tt_track_embs = np.load(f"{TT_INDEX}/track_embeddings.npy").astype(np.float32)
with open(f"{TT_INDEX}/track_ids.json") as f:
    tt_track_ids = json.load(f)
tt_id2idx = {t: i for i, t in enumerate(tt_track_ids)}

# normalize once (TT is usually already L2-normalized but be safe)
norms = np.linalg.norm(tt_track_embs, axis=1, keepdims=True)
norms[norms < 1e-8] = 1.0
tt_track_embs = tt_track_embs / norms


# --- Build turns ---
print("Loading dev sessions...")
ds = load_dataset("talkpl-ai/TalkPlayData-Challenge-Dataset")
sessions = list(ds["test"])

all_turns = []
for sess in tqdm(sessions, desc="Collecting turns"):
    goal = (sess.get("conversation_goal", {}) or {}).get("listener_goal", "") or ""
    culture = (sess.get("user_profile", {}) or {}).get("preferred_musical_culture", "") or ""
    music_history, text_history, all_text = [], [], []
    for turn in sess["conversations"]:
        if turn["role"] == "music":
            gold = turn["content"]
            latest_user = text_history[-1] if text_history else ""

            bm25_parts = [goal, culture]
            for tid in music_history[-4:]:
                bm25_parts.append(get_track_text(tid))
            bm25_parts.extend(text_history[-4:])
            bm25_query = " ".join(p for p in bm25_parts if p)

            tt_parts = [latest_user, goal, culture]
            for tid in music_history[-2:]:
                na = get_track_name_artist(tid)
                if na: tt_parts.append(na)
            tt_query = " ".join(p for p in tt_parts if p)

            mentioned = set()
            for txt in all_text:
                for a in find_mentioned_artists(txt):
                    mentioned.add(a)
            for tid in music_history:
                for a in (metadata_dict.get(tid, {}).get("artist_name") or []):
                    mentioned.add(a.strip().lower())

            all_turns.append({
                "gold": gold,
                "bm25_query": bm25_query,
                "tt_query": tt_query,
                "seen": list(music_history),
                "mentioned_artists": list(mentioned),
                "last_tracks": list(music_history[-LAST_TRACKS_N:]),
                "bucket": bucket_for(latest_user, bool(music_history)),
            })
            music_history.append(gold)
        else:
            text_history.append(turn["content"])
            all_text.append(turn["content"])

N = len(all_turns)
print(f"Total turns: {N}")


# --- BM25 retrieval (one big retrieve at max N) ---
MAX_BM25 = max(BM25_NS)
print(f"BM25 retrieval @{MAX_BM25}...")
bm25_lists = []   # list of (tids_in_rank_order) per turn
for t in tqdm(all_turns, desc="BM25"):
    tokens = bm25s.tokenize([t["bm25_query"].lower()], show_progress=False)
    seen = set(t["seen"])
    res = bm25_model.retrieve(
        tokens, k=MAX_BM25 + len(seen) * 3, return_as="tuple", show_progress=False
    )
    cands = []
    for idx in res.documents[0]:
        tid = bm25_track_ids[int(idx)]
        if tid not in seen:
            cands.append(tid)
        if len(cands) >= MAX_BM25:
            break
    bm25_lists.append(cands)


# --- TT query encoding + full sims (compute top-MAX_K once) ---
MAX_TT_K = max(TT_KS) if max(TT_KS) > 0 else 1
print("Encoding TT queries...")
tt_q_embs = tt_model.encode(
    [t["tt_query"] for t in all_turns],
    batch_size=128, show_progress_bar=True,
    normalize_embeddings=True, convert_to_numpy=True,
).astype(np.float32)

print(f"Computing TT query top-{MAX_TT_K}...")
tt_topk_per_turn = np.zeros((N, MAX_TT_K), dtype=np.int32)
batch = 256
for i in tqdm(range(0, N, batch), desc="TT sims"):
    sims = tt_q_embs[i:i+batch] @ tt_track_embs.T
    top = np.argpartition(-sims, MAX_TT_K, axis=1)[:, :MAX_TT_K]
    for j in range(top.shape[0]):
        order = np.argsort(-sims[j, top[j]])
        tt_topk_per_turn[i+j] = top[j][order]


# --- Last-track-NN: for each played track id present in tt index, precompute top-MAX_NN ---
MAX_NN = max(LAST_NN_MS) if max(LAST_NN_MS) > 0 else 0
last_nn_cache: dict[str, np.ndarray] = {}
if MAX_NN > 0:
    needed = set()
    for t in all_turns:
        for tid in t["last_tracks"]:
            if tid in tt_id2idx:
                needed.add(tid)
    print(f"Computing last-track-NN top-{MAX_NN} for {len(needed):,} distinct tracks...")
    needed_list = list(needed)
    idxs = np.array([tt_id2idx[t] for t in needed_list], dtype=np.int64)
    bsz = 256
    for s in tqdm(range(0, len(idxs), bsz), desc="NN sims"):
        chunk_idx = idxs[s:s+bsz]
        sims = tt_track_embs[chunk_idx] @ tt_track_embs.T
        # exclude self
        for r, gi in enumerate(chunk_idx):
            sims[r, gi] = -1e9
        top = np.argpartition(-sims, MAX_NN, axis=1)[:, :MAX_NN]
        for r in range(top.shape[0]):
            order = np.argsort(-sims[r, top[r]])
            last_nn_cache[needed_list[s + r]] = top[r][order]


# --- Per-turn pool builders ---
def pool_for_config(turn_idx: int, bm25_n: int, use_artist: bool,
                    tt_k: int, last_nn_m: int) -> set[str]:
    t = all_turns[turn_idx]
    seen = set(t["seen"])
    pool: set[str] = set()

    # BM25
    pool.update(bm25_lists[turn_idx][:bm25_n])

    # Artist expansion
    if use_artist:
        for a in t["mentioned_artists"]:
            for tid in artist_to_tids.get(a, ()):
                if tid not in seen:
                    pool.add(tid)

    # TT query top-K
    if tt_k > 0:
        for idx in tt_topk_per_turn[turn_idx, :tt_k]:
            tid = tt_track_ids[int(idx)]
            if tid not in seen:
                pool.add(tid)

    # Last-track-NN
    if last_nn_m > 0:
        for src_tid in t["last_tracks"]:
            arr = last_nn_cache.get(src_tid)
            if arr is None:
                continue
            for idx in arr[:last_nn_m]:
                tid = tt_track_ids[int(idx)]
                if tid not in seen:
                    pool.add(tid)

    pool.discard("")  # safety
    return pool


# --- Sweep ---
configs = list(product(BM25_NS, ARTIST_FLAGS, TT_KS, LAST_NN_MS))
print(f"\nSweeping {len(configs)} configs over {N} turns...")

bucket_names = ["specific", "mood", "lyrics", "more_like_this", "history_driven", "generic"]
turn_bucket = [t["bucket"] for t in all_turns]
gold_list = [t["gold"] for t in all_turns]

rows = []
for (bn, art, tk, nm) in tqdm(configs, desc="Configs"):
    sizes = np.zeros(N, dtype=np.int32)
    hits = np.zeros(N, dtype=bool)
    bucket_hits = {b: [0, 0] for b in bucket_names}  # [hit, total]
    for i in range(N):
        pool = pool_for_config(i, bn, art, tk, nm)
        sizes[i] = len(pool)
        hit = gold_list[i] in pool
        hits[i] = hit
        b = turn_bucket[i]
        bucket_hits[b][1] += 1
        if hit:
            bucket_hits[b][0] += 1
    rows.append({
        "bm25_n": bn, "artist": art, "tt_k": tk, "last_nn": nm,
        "mean_pool": float(sizes.mean()),
        "recall": float(hits.mean()),
        "buckets": {b: (bucket_hits[b][0] / bucket_hits[b][1] if bucket_hits[b][1] else 0.0)
                    for b in bucket_names},
    })


# --- Output ---
Path(OUT_PATH).parent.mkdir(parents=True, exist_ok=True)
rows_sorted = sorted(rows, key=lambda r: r["mean_pool"])

lines = []
lines.append(f"Min-pool recall sweep  N={N} turns  TT_model={TT_MODEL}\n")
lines.append(f"Grid: BM25_NS={BM25_NS}  ARTIST={ARTIST_FLAGS}  TT_KS={TT_KS}  LAST_NN_MS={LAST_NN_MS}  last_tracks={LAST_TRACKS_N}  artist_cap={ARTIST_CAP}\n\n")
hdr = (f"{'bm25':>5} {'art':>4} {'tt_k':>5} {'lnn':>4} | "
       f"{'pool':>6} {'recall':>7} | "
       + " ".join(f"{b[:6]:>6}" for b in bucket_names))
lines.append(hdr + "\n")
lines.append("-" * len(hdr) + "\n")
for r in rows_sorted:
    line = (f"{r['bm25_n']:>5} {('Y' if r['artist'] else '.'):>4} "
            f"{r['tt_k']:>5} {r['last_nn']:>4} | "
            f"{r['mean_pool']:>6.0f} {r['recall']:>7.4f} | "
            + " ".join(f"{r['buckets'][b]:>6.3f}" for b in bucket_names))
    lines.append(line + "\n")

# Pareto frontier (max recall per increasing pool size)
lines.append("\nPareto frontier (recall vs mean pool size):\n")
best_recall = -1.0
for r in rows_sorted:
    if r["recall"] > best_recall:
        best_recall = r["recall"]
        lines.append(
            f"  pool={r['mean_pool']:>6.0f}  recall={r['recall']:.4f}  "
            f"(bm25={r['bm25_n']} artist={'Y' if r['artist'] else '.'} "
            f"tt_k={r['tt_k']} last_nn={r['last_nn']})\n"
        )

with open(OUT_PATH, "w") as f:
    f.writelines(lines)
print(f"\nSaved: {OUT_PATH}")
print("".join(lines[-15:]))
