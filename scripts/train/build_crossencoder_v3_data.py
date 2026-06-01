"""Build cross-encoder v3 training data with Phase A pool-aware hard negatives.

Differences from v2:
  - **Hard negatives drawn from the actual Phase A pool** per turn, not BM25.
    This is the key fix for the v2 distribution mismatch.
  - **Multi-turn query template** with [TURN-3]/[TURN-2]/[TURN-1] tags so the
    cross-encoder learns recency weighting. Track-side metadata format
    unchanged from v2 (name, artist, album, top-12 tags, year).

Each TRAIN turn -> one listwise group with (positive + N pool negatives).

Output: data/crossencoder_v3/{train,valid}.jsonl
Each row: {"query": str, "docs": list[str](length 1+N), "labels": list[float]}

Usage:
    python scripts/train/build_crossencoder_v3_data.py
"""
import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import argparse
import json
import random
import re
from pathlib import Path

import bm25s
import numpy as np
from datasets import load_dataset, concatenate_datasets
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

# ── Phase A pool config (matches inference / recall_gap diagnostic) ──────────
TT_MODEL   = "models/twotower_v6/final"
TT_INDEX   = "cache/twotower_v6"
BM25_CACHE = "cache/bm25/track_metadata"
BM25_POOL  = 500
TT_POOL    = 2000      # NOTE: matches stage-1 retrain (was 1000 in v2)
QWEN_POOL  = 500
CF_POOL    = 200
LAST_NN_K  = 100
LAST_NN_SRC = 2
SESSION_MEAN_K = 100
SESSION_MEAN_N = 4
ARTIST_CAP = 50
COOCCUR_TABLE = "cache/cooccur/next_song_leakfree.npz"
COOCCUR_KS = [300, 150, 50]

parser = argparse.ArgumentParser()
parser.add_argument("--hard_negs", type=int, default=5,
                    help="Pool-sampled negatives per turn.")
parser.add_argument("--valid_frac", type=float, default=0.01)
parser.add_argument("--out_dir", default="data/crossencoder_v3")
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--max_turns_per_session", type=int, default=2)
parser.add_argument("--max_sessions", type=int, default=0,
                    help="If >0, cap TRAIN sessions for a quick run.")
args = parser.parse_args()

random.seed(args.seed)

_FILLER = re.compile(
    r"\b(can you|could you|would you|please|i want|i'd like|i would like|"
    r"i need|i'm looking for|i am looking for|something that(?:'s| is)|something|"
    r"recommend(?:ation)?|suggest(?:ion)?|play(?: me)?|find me|show me|give me|"
    r"how about|what about|i feel like(?: listening to)?|i(?:'m| am) in the mood for|"
    r"do you have|do you know)\b",
    re.IGNORECASE,
)
def clean_query(t: str) -> str:
    return re.sub(r"\s+", " ", _FILLER.sub(" ", t)).strip()

# ── Catalog ───────────────────────────────────────────────────────────────────
print("Loading track metadata...")
meta_ds = load_dataset("talkpl-ai/TalkPlayData-Challenge-Track-Metadata")
all_tracks = concatenate_datasets([meta_ds["all_tracks"], meta_ds["test_tracks"]])
metadata_dict = {row["track_id"]: row for row in all_tracks}
all_track_ids = list(metadata_dict.keys())


def get_track_text(tid: str) -> str:
    row = metadata_dict.get(tid, {})
    name   = (row.get("track_name")  or [""])[0]
    artist = (row.get("artist_name") or [""])[0]
    album  = (row.get("album_name")  or [""])[0]
    tags   = " ".join((row.get("tag_list") or [])[:12])
    release = row.get("release_date") or ""
    year = str(release)[:4] if release else ""
    parts = [name]
    if artist: parts.append(f"by {artist}")
    if album:  parts.append(f"| Album: {album}")
    if tags:   parts.append(f"| Tags: {tags}")
    if year:   parts.append(f"| {year}")
    return " ".join(parts).strip()


def get_track_name_artist(tid: str) -> str:
    row = metadata_dict.get(tid, {})
    name   = (row.get("track_name")  or [""])[0]
    artist = (row.get("artist_name") or [""])[0]
    return f"{name} by {artist}".strip()


print("Building artist catalog...")
artist_buckets: dict[str, list] = {}
for _tid, _row in metadata_dict.items():
    _pop = float(_row.get("popularity") or 0.0)
    for _a in (_row.get("artist_name") or []):
        _k = _a.strip().lower()
        if _k: artist_buckets.setdefault(_k, []).append((_pop, _tid))
artist_to_tids: dict[str, list] = {}
for _k, _bucket in artist_buckets.items():
    _bucket.sort(key=lambda x: -x[0])
    artist_to_tids[_k] = [t for _, t in _bucket[: ARTIST_CAP]]
known_artists = sorted(artist_to_tids.keys(), key=len, reverse=True)


def find_mentioned_artists(text: str) -> list[str]:
    if not text: return []
    tl = text.lower()
    out = []
    for a in known_artists:
        if a in tl:
            out.append(a); tl = tl.replace(a, " " * len(a))
    return out


# ── BM25 ──────────────────────────────────────────────────────────────────────
print("Loading BM25...")
bm25_model = bm25s.BM25.load(BM25_CACHE, load_corpus=False)
with open(f"{BM25_CACHE}/track_ids.json") as f:
    bm25_track_ids = json.load(f)


def retrieve_bm25(q: str, topk: int) -> list[str]:
    tokens = bm25s.tokenize([q.lower()])
    results = bm25_model.retrieve(tokens, k=topk, return_as="tuple")
    return [bm25_track_ids[int(i)] for i in results.documents[0]]


# ── TT ────────────────────────────────────────────────────────────────────────
print(f"Loading TT model: {TT_MODEL}")
tt_model = SentenceTransformer(TT_MODEL)
print(f"Loading TT index: {TT_INDEX}")
tt_embs = np.load(f"{TT_INDEX}/track_embeddings.npy")
with open(f"{TT_INDEX}/track_ids.json") as f:
    tt_ids = json.load(f)
tt_id2idx = {t: i for i, t in enumerate(tt_ids)}

# ── Qwen3 meta (SKIPPED in v3 data builder for speed; the pool still
#    contains TT+BM25+CF+cooccur+NN+artist which is ~85% of Phase A) ─────────
# The Qwen3 expansion adds ~500 candidates/turn that mostly overlap with TT.
# Encoding it per turn dominates wall time (~3 s/turn). For training data
# only (CE never sees Qwen3-ranking features directly), the slight pool
# distribution difference is acceptable.

# ── CF ────────────────────────────────────────────────────────────────────────
print("Loading CF-BPR embeddings...")
cf_track_embs = np.load("cache/cf_bpr/track_embeddings.npy")
with open("cache/cf_bpr/track_ids.json") as f:
    cf_track_ids = json.load(f)
with open("cache/user_cf_bpr.json") as f:
    user_cf_raw = json.load(f)
user_cf = {uid: np.array(v, dtype=np.float32) for uid, v in user_cf_raw.items() if v}

# ── Cooccur ───────────────────────────────────────────────────────────────────
print(f"Loading cooccur: {COOCCUR_TABLE}")
co = np.load(COOCCUR_TABLE, allow_pickle=True)
co_ids = co["track_ids"]
co_neigh = co["neigh_ids"]
co_id2idx = {str(t): i for i, t in enumerate(co_ids)}


# ── Pool builder (same shape as run_inference_fusion_recall_expansion.py) ────
def build_phase_a_pool(item, latest_user, music_history, text_history):
    user_id = item["user_id"]
    user_emb = user_cf.get(user_id)
    goal = (item.get("conversation_goal") or {}).get("listener_goal", "") or ""
    culture = (item.get("user_profile") or {}).get("preferred_musical_culture", "") or ""
    category = (item.get("conversation_goal") or {}).get("category", "") or ""
    specificity = (item.get("conversation_goal") or {}).get("specificity", "") or ""
    age_group = (item.get("user_profile") or {}).get("age_group", "") or ""
    country = (item.get("user_profile") or {}).get("country_name", "") or ""

    tt_parts = [latest_user, goal, culture]
    for tid in music_history[-2:]:
        na = get_track_name_artist(tid)
        if na: tt_parts.append(na)
    tt_query = " ".join(p for p in tt_parts if p)

    bm25_parts = [goal, culture]
    for tid in music_history[-4:]:
        bm25_parts.append(get_track_text(tid))
    bm25_parts.extend(text_history[-4:])
    bm25_query = " ".join(p for p in bm25_parts if p)

    cleaned = clean_query(latest_user) or latest_user
    sem_parts = [cleaned, goal, culture]
    for tid in music_history[-2:]:
        na = get_track_name_artist(tid)
        if na: sem_parts.append(na)
    semantic_query = " ".join(p for p in sem_parts if p)

    tt_emb = tt_model.encode(tt_query, normalize_embeddings=True, convert_to_numpy=True)
    tt_all = tt_embs @ tt_emb
    cf_all = (cf_track_embs @ user_emb) if user_emb is not None else None

    seen = set(music_history)
    retrieve_k = BM25_POOL + len(seen) * 3
    raw_tids = retrieve_bm25(bm25_query, topk=retrieve_k)
    bm25_cands = [t for t in raw_tids if t not in seen][: BM25_POOL]

    cands = list(bm25_cands)
    cands_set = set(cands)

    def add_topk(scores, ids_list, k):
        if k <= 0 or scores is None: return
        top = np.argpartition(scores, -k)[-k:]
        top = top[np.argsort(scores[top])[::-1]]
        for idx in top:
            tid = ids_list[int(idx)]
            if tid not in cands_set and tid not in seen:
                cands.append(tid); cands_set.add(tid)

    add_topk(tt_all, tt_ids, TT_POOL)
    if cf_all is not None:
        add_topk(cf_all, cf_track_ids, CF_POOL)

    # artist expansion
    mentioned: dict[str, str] = {}
    for txt in text_history:
        for a in find_mentioned_artists(txt):
            mentioned.setdefault(a, "user_text")
    for hist_tid in music_history:
        for a in (metadata_dict.get(hist_tid, {}).get("artist_name") or []):
            k = a.strip().lower()
            if k and k not in mentioned: mentioned[k] = "played_track_artist"
    for a in mentioned:
        for tid in artist_to_tids.get(a, ()):
            if tid in seen or tid in cands_set: continue
            cands.append(tid); cands_set.add(tid)

    # session NN
    for pos, k_nn in enumerate([LAST_NN_K] * LAST_NN_SRC):
        if k_nn <= 0 or pos >= len(music_history): continue
        src_tid = music_history[-(pos + 1)]
        src_idx = tt_id2idx.get(src_tid)
        if src_idx is None: continue
        sims = tt_embs @ tt_embs[src_idx]
        top = np.argpartition(sims, -k_nn - 1)[-(k_nn + 1):]
        top = top[np.argsort(sims[top])[::-1]]
        taken = 0
        for idx in top:
            tid = tt_ids[int(idx)]
            if tid == src_tid or tid in seen: continue
            if tid not in cands_set:
                cands.append(tid); cands_set.add(tid)
            taken += 1
            if taken >= k_nn: break

    # session mean NN
    if SESSION_MEAN_K > 0 and music_history:
        hist_idxs = [tt_id2idx[t] for t in music_history[-SESSION_MEAN_N:]
                     if t in tt_id2idx]
        if hist_idxs:
            mean_vec = tt_embs[hist_idxs].mean(axis=0)
            n = np.linalg.norm(mean_vec)
            if n > 1e-8:
                mean_vec /= n
                sims = tt_embs @ mean_vec
                top = np.argpartition(sims, -SESSION_MEAN_K - 1)[-(SESSION_MEAN_K + 1):]
                top = top[np.argsort(sims[top])[::-1]]
                taken = 0
                for idx in top:
                    tid = tt_ids[int(idx)]
                    if tid in seen: continue
                    if tid not in cands_set:
                        cands.append(tid); cands_set.add(tid)
                    taken += 1
                    if taken >= SESSION_MEAN_K: break

    # cooccur
    if music_history:
        for pos, k_co in enumerate(COOCCUR_KS):
            if k_co <= 0 or pos >= len(music_history): break
            src_tid = music_history[-(pos + 1)]
            src_idx = co_id2idx.get(src_tid)
            if src_idx is None: continue
            neighs = co_neigh[src_idx]
            taken = 0
            for rank in range(len(neighs)):
                if taken >= k_co: break
                nidx = int(neighs[rank])
                if nidx < 0: break
                tid = str(co_ids[nidx])
                if tid in seen: continue
                if tid not in cands_set:
                    cands.append(tid); cands_set.add(tid)
                taken += 1
    return cands


# ── Build multi-turn query ──────────────────────────────────────────────────-─
def build_multiturn_query(text_history_user_only: list[str], goal: str, culture: str) -> str:
    """Last 3 user turns tagged. Older empty if fewer than 3."""
    last3 = text_history_user_only[-3:]
    while len(last3) < 3:
        last3 = [""] + last3
    parts = [
        f"[TURN-3] {last3[0]}".strip(),
        f"[TURN-2] {last3[1]}".strip(),
        f"[TURN-1] {last3[2]}".strip(),
    ]
    tail = []
    if goal:    tail.append(f"Goal: {goal}")
    if culture: tail.append(f"Culture: {culture}")
    return " ".join(parts + tail).strip()


# ── Sessions ──────────────────────────────────────────────────────────────────
print("Loading TRAIN sessions...")
ds = load_dataset("talkpl-ai/TalkPlayData-Challenge-Dataset", split="train")
sessions = list(ds)
random.shuffle(sessions)
if args.max_sessions > 0:
    sessions = sessions[: args.max_sessions]
n_valid = int(len(sessions) * args.valid_frac)
valid_sessions = sessions[:n_valid]
train_sessions = sessions[n_valid:]
print(f"Train sessions: {len(train_sessions)}, Valid sessions: {len(valid_sessions)}")


def build_rows(sessions: list, desc: str) -> list[dict]:
    rows = []
    for item in tqdm(sessions, desc=desc):
        music_history: list[str] = []
        text_history:  list[str] = []
        text_history_user_only: list[str] = []
        per_session: list[dict] = []

        goal = (item.get("conversation_goal") or {}).get("listener_goal", "") or ""
        culture = (item.get("user_profile") or {}).get("preferred_musical_culture", "") or ""

        for turn in item["conversations"]:
            role = turn["role"]
            if role == "music":
                gold = turn["content"]
                gold_text = get_track_text(gold)
                if not gold_text:
                    music_history.append(gold); continue

                latest_user = text_history[-1] if text_history else ""
                if not latest_user:
                    music_history.append(gold); continue

                # build the Phase A pool for this turn
                pool = build_phase_a_pool(item, latest_user, music_history, text_history)
                # restrict to candidates that are not gold/seen and not empty text
                cand_pool = [t for t in pool
                             if t != gold and t not in set(music_history) and get_track_text(t)]
                if len(cand_pool) < args.hard_negs:
                    music_history.append(gold); continue

                neg_tids = random.sample(cand_pool, args.hard_negs)
                query = build_multiturn_query(text_history_user_only, goal, culture)
                docs = [gold_text] + [get_track_text(t) for t in neg_tids]
                labels = [1.0] + [0.0] * args.hard_negs

                per_session.append({"query": query, "docs": docs, "labels": labels})
                music_history.append(gold)
            elif role in ("user", "assistant"):
                text_history.append(turn["content"])
                if role == "user":
                    text_history_user_only.append(turn["content"])

        if args.max_turns_per_session > 0 and len(per_session) > args.max_turns_per_session:
            per_session = random.sample(per_session, args.max_turns_per_session)
        rows.extend(per_session)
    return rows


print("Building TRAIN rows...")
train_rows = build_rows(train_sessions, "train")
print("Building VALID rows...")
valid_rows = build_rows(valid_sessions, "valid")

print(f"Train groups: {len(train_rows):,}  (group_size={args.hard_negs + 1})")
print(f"Valid groups: {len(valid_rows):,}")

out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
for name, rows in [("train", train_rows), ("valid", valid_rows)]:
    p = out / f"{name}.jsonl"
    with open(p, "w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"Saved {len(rows):,} rows to {p}")

print("\nSample:")
if train_rows:
    ex = train_rows[0]
    print("  query    :", ex["query"][:300])
    print("  positive :", ex["docs"][0][:120])
    print("  negative :", ex["docs"][1][:120])
    print("  labels   :", ex["labels"])
