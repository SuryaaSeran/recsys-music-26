"""
Diagnose the 17% unreachable turns in the dev set.

For each turn where gold is NOT in the current Phase A pool, records:
  - Turn context (position, history length, cold user)
  - Gold track features (popularity, has_tags)
  - Gold rank in each full signal (TT, BM25, Qwen-meta)
  - Whether gold artist appears anywhere in conversation text

Phase A pool config (matches current best system):
  BM25@500 + artist + TT@1000 + NN(k=100,src=2) + qwen@500 + CF@200
  + session_mean@100 + cooccur(300,150,50)

Usage:
    python scripts/inference/analyze_recall_gap.py              # full 1000 sessions
    python scripts/inference/analyze_recall_gap.py --sessions 200  # fast estimate
"""
import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import argparse
import json
import re
import random
from collections import defaultdict, Counter

import numpy as np
import bm25s
import torch
from datasets import load_dataset, concatenate_datasets
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument("--sessions", type=int, default=0, help="0 = all dev sessions")
parser.add_argument("--out", default="exp/analysis/recall_gap.json",
                    help="Save per-turn records here for deeper analysis")
parser.add_argument("--tt_model", default="models/twotower_v6/final")
parser.add_argument("--tt_index", default="cache/twotower_v6")
parser.add_argument("--tt_query_prefix", default="",
                    help="Prefix prepended to the TT query (e.g. Qwen3 'Instruct: ...\\nQuery: ').")
args = parser.parse_args()

# ── Phase A pool config (hardcoded to current best) ──────────────────────────
TT_MODEL   = args.tt_model
TT_INDEX   = args.tt_index
BM25_CACHE = "cache/bm25/track_metadata"
BM25_POOL  = 500
TT_POOL    = 1000
QWEN_POOL  = 500
CF_POOL    = 200
LAST_NN_K  = 100
LAST_NN_SRC = 2
SESSION_MEAN_K = 100
SESSION_MEAN_N = 4
ARTIST_CAP = 50
COOCCUR_TABLE = "cache/cooccur/next_song_leakfree.npz"
COOCCUR_KS = [300, 150, 50]
BM25_DIAG_DEPTH = 5000   # BM25 retrieval depth for rank lookup on unreachable turns
# ─────────────────────────────────────────────────────────────────────────────

_FILLER = re.compile(
    r"\b(can you|could you|would you|please|i want|i'd like|i would like|"
    r"i need|i'm looking for|i am looking for|something that(?:'s| is)|something|"
    r"recommend(?:ation)?|suggest(?:ion)?|play(?: me)?|find me|show me|give me|"
    r"how about|what about|i feel like(?: listening to)?|i(?:'m| am) in the mood for|"
    r"do you have|do you know)\b",
    re.IGNORECASE,
)
def clean_query(text):
    return re.sub(r"\s+", " ", _FILLER.sub(" ", text)).strip()

# ── Load metadata ─────────────────────────────────────────────────────────────
print("Loading track metadata...")
meta_ds = load_dataset("talkpl-ai/TalkPlayData-Challenge-Track-Metadata")
all_tracks = concatenate_datasets([meta_ds["all_tracks"], meta_ds["test_tracks"]])
metadata_dict = {row["track_id"]: row for row in all_tracks}

def get_track_name_artist(tid):
    row = metadata_dict.get(tid, {})
    name   = (row.get("track_name")  or [""])[0]
    artist = (row.get("artist_name") or [""])[0]
    return f"{name} {artist}".strip()

def get_track_text(tid):
    row = metadata_dict.get(tid, {})
    name   = (row.get("track_name")  or [""])[0]
    artist = (row.get("artist_name") or [""])[0]
    tags   = " ".join(row.get("tag_list") or [])
    return f"{name} {artist} {tags}".strip()

# Artist -> tracks dict (same as inference script)
print("Building artist catalog...")
artist_buckets: dict[str, list] = {}
for _tid, _row in metadata_dict.items():
    _pop = float(_row.get("popularity") or 0.0)
    for _a in (_row.get("artist_name") or []):
        _k = _a.strip().lower()
        if _k:
            artist_buckets.setdefault(_k, []).append((_pop, _tid))
artist_to_tids: dict[str, list] = {}
for _k, _bucket in artist_buckets.items():
    _bucket.sort(key=lambda x: -x[0])
    artist_to_tids[_k] = [t for _, t in _bucket[:ARTIST_CAP]]
known_artists = sorted(artist_to_tids.keys(), key=len, reverse=True)
print(f"  {len(known_artists):,} artists in catalog")

def find_mentioned_artists(text):
    if not text:
        return []
    tl = text.lower()
    out = []
    for a in known_artists:
        if a in tl:
            out.append(a)
            tl = tl.replace(a, " " * len(a))
    return out

# ── Load BM25 ─────────────────────────────────────────────────────────────────
print("Loading BM25 index...")
bm25_model = bm25s.BM25.load(BM25_CACHE, load_corpus=False)
with open(f"{BM25_CACHE}/track_ids.json") as f:
    bm25_track_ids = json.load(f)
bm25_tid_set = set(bm25_track_ids)

def retrieve_bm25(query, topk):
    tokens = bm25s.tokenize([query.lower()])
    results = bm25_model.retrieve(tokens, k=topk, return_as="tuple")
    tids = [bm25_track_ids[int(i)] for i in results.documents[0]]
    scores = [float(s) for s in results.scores[0]]
    return tids, scores

# ── Load TT ───────────────────────────────────────────────────────────────────
print(f"Loading TT model: {TT_MODEL}")
tt_model = SentenceTransformer(TT_MODEL)
print(f"Loading TT index: {TT_INDEX}")
tt_embs = np.load(f"{TT_INDEX}/track_embeddings.npy")
with open(f"{TT_INDEX}/track_ids.json") as f:
    tt_ids = json.load(f)
tt_id2idx = {t: i for i, t in enumerate(tt_ids)}
print(f"  TT index: {tt_embs.shape}")

# ── Load Qwen3 ────────────────────────────────────────────────────────────────
print("Loading Qwen3-Embedding-0.6B...")
from sentence_transformers import SentenceTransformer as _ST
qwen_model = _ST("Qwen/Qwen3-Embedding-0.6B", trust_remote_code=True)
QWEN_INSTR = "Instruct: Given a music conversation, retrieve the track that best fits.\nQuery: "

print("Loading Qwen3 metadata index...")
qwen_meta_embs = np.load("cache/qwen3_meta/track_embeddings.npy")
with open("cache/qwen3_meta/track_ids.json") as f:
    qwen_meta_ids = json.load(f)
qwen_meta_id2idx = {t: i for i, t in enumerate(qwen_meta_ids)}
print(f"  Qwen-meta index: {qwen_meta_embs.shape}")

# ── Load CF ───────────────────────────────────────────────────────────────────
print("Loading CF-BPR embeddings...")
cf_track_embs = np.load("cache/cf_bpr/track_embeddings.npy")
with open("cache/cf_bpr/track_ids.json") as f:
    cf_track_ids = json.load(f)
cf_track_id2idx = {t: i for i, t in enumerate(cf_track_ids)}
with open("cache/user_cf_bpr.json") as f:
    user_cf_raw = json.load(f)
user_cf = {uid: np.array(v, dtype=np.float32) for uid, v in user_cf_raw.items() if v}
print(f"  CF track index: {cf_track_embs.shape}  users: {len(user_cf):,}")

# ── Load CLAP (not needed for pool building, skip) ───────────────────────────

# ── Load co-occurrence ────────────────────────────────────────────────────────
print(f"Loading co-occurrence table: {COOCCUR_TABLE}")
cooccur_data = np.load(COOCCUR_TABLE, allow_pickle=True)
cooccur_track_ids = cooccur_data["track_ids"]
cooccur_neigh_ids = cooccur_data["neigh_ids"]
cooccur_neigh_w   = cooccur_data["neigh_w"]
cooccur_tid2idx   = {str(t): i for i, t in enumerate(cooccur_track_ids)}
print(f"  co-occur shape: {cooccur_neigh_ids.shape}")

# ── Load sessions ─────────────────────────────────────────────────────────────
print("Loading dev sessions...")
ds = load_dataset("talkpl-ai/TalkPlayData-Challenge-Dataset")
sessions = list(ds["test"])
if args.sessions > 0:
    sessions = sessions[:args.sessions]
print(f"  {len(sessions)} sessions")

# ── helper: build pool identical to Phase A inference ────────────────────────
def add_topk(scores_arr, ids_list, k, src_label, cands, cands_set, sources,
             tt_rank_map, qm_rank_map):
    if k <= 0 or scores_arr is None:
        return
    top = np.argpartition(scores_arr, -k)[-k:]
    top = top[np.argsort(scores_arr[top])[::-1]]
    for rank, idx in enumerate(top):
        tid = ids_list[int(idx)]
        if tid not in cands_set:
            cands.append(tid); cands_set.add(tid)
            sources[tid] = set()
        sources[tid].add(src_label)
        if src_label == "tt" and tid not in tt_rank_map:
            tt_rank_map[tid] = rank
        if src_label == "qm" and tid not in qm_rank_map:
            qm_rank_map[tid] = rank

# ── Main loop ─────────────────────────────────────────────────────────────────
records = []       # one record per unreachable turn
total_turns = 0
reachable   = 0

for item in tqdm(sessions, desc="Sessions"):
    session_id  = item["session_id"]
    user_id     = item["user_id"]
    goal        = item.get("conversation_goal", {}).get("listener_goal", "")
    culture     = item.get("user_profile", {}).get("preferred_musical_culture", "")
    goal_cat    = item.get("conversation_goal", {}).get("goal_category", "")
    goal_spec   = item.get("conversation_goal", {}).get("goal_specificity", "")
    age_group   = item.get("user_profile", {}).get("age_group", "")
    country     = item.get("user_profile", {}).get("country", "")
    conversations = item["conversations"]

    user_emb = user_cf.get(user_id)

    music_history: list[str] = []
    text_history:  list[str] = []

    for turn in conversations:
        if turn["role"] != "music":
            if turn["role"] in ("user", "assistant"):
                text_history.append(turn["content"])
            continue

        gold = turn["content"]
        turn_number = turn["turn_number"]
        seen = set(music_history)
        total_turns += 1

        # ── Build queries (same as inference) ────────────────────────────────
        latest_user = text_history[-1] if text_history else ""
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

        # ── Encode ───────────────────────────────────────────────────────────
        tt_emb   = tt_model.encode(args.tt_query_prefix + tt_query, normalize_embeddings=True, convert_to_numpy=True)
        qwen_emb = qwen_model.encode(QWEN_INSTR + semantic_query,
                                     normalize_embeddings=True, convert_to_numpy=True)

        tt_all = tt_embs @ tt_emb
        qm_all = qwen_meta_embs @ qwen_emb
        cf_all = (cf_track_embs @ user_emb) if user_emb is not None else None

        # ── BM25 recall ──────────────────────────────────────────────────────
        retrieve_k = BM25_POOL + len(seen) * 3
        raw_tids, raw_scores = retrieve_bm25(bm25_query, topk=retrieve_k)
        filtered = [(t, s) for t, s in zip(raw_tids, raw_scores) if t not in seen][:BM25_POOL]
        bm25_cands = [t for t, _ in filtered]

        cands = list(bm25_cands)
        cands_set = set(cands)
        sources: dict[str, set] = {t: {"bm25"} for t in cands}
        tt_rank_map: dict[str, int] = {}
        qm_rank_map: dict[str, int] = {}
        artist_rank_map: dict[str, int] = {}
        nn_rank_map: dict[str, int] = {}
        collab_rank_map: dict[str, int] = {}

        # ── Global dense expansion ────────────────────────────────────────────
        add_topk(tt_all, tt_ids, TT_POOL, "tt", cands, cands_set, sources,
                 tt_rank_map, qm_rank_map)
        add_topk(qm_all, qwen_meta_ids, QWEN_POOL, "qm", cands, cands_set, sources,
                 tt_rank_map, qm_rank_map)
        if cf_all is not None:
            add_topk(cf_all, cf_track_ids, CF_POOL, "cf", cands, cands_set, sources,
                     tt_rank_map, qm_rank_map)

        # ── Artist expansion ──────────────────────────────────────────────────
        mentioned: dict[str, str] = {}
        for txt in text_history:
            for a in find_mentioned_artists(txt):
                mentioned.setdefault(a, "user_text")
        for hist_tid in music_history:
            for a in (metadata_dict.get(hist_tid, {}).get("artist_name") or []):
                k = a.strip().lower()
                if k and k not in mentioned:
                    mentioned[k] = "played_track_artist"
        for a, src in mentioned.items():
            for rank, tid in enumerate(artist_to_tids.get(a, ())):
                if tid in seen:
                    continue
                if tid not in cands_set:
                    cands.append(tid); cands_set.add(tid); sources[tid] = set()
                sources[tid].add("artist")
                if tid not in artist_rank_map or rank < artist_rank_map[tid]:
                    artist_rank_map[tid] = rank

        # ── Session NN expansion ──────────────────────────────────────────────
        nn_ks = [LAST_NN_K] * LAST_NN_SRC
        for pos, k_nn in enumerate(nn_ks):
            if k_nn <= 0 or pos >= len(music_history):
                continue
            src_tid = music_history[-(pos + 1)]
            src_idx = tt_id2idx.get(src_tid)
            if src_idx is None:
                continue
            sims = tt_embs @ tt_embs[src_idx]
            top = np.argpartition(sims, -k_nn - 1)[-(k_nn + 1):]
            top = top[np.argsort(sims[top])[::-1]]
            rank = 0
            for idx in top:
                tid = tt_ids[int(idx)]
                if tid == src_tid or tid in seen:
                    continue
                if tid not in cands_set:
                    cands.append(tid); cands_set.add(tid); sources[tid] = set()
                sources[tid].add("nn")
                if tid not in nn_rank_map or rank < nn_rank_map[tid]:
                    nn_rank_map[tid] = rank
                rank += 1
                if rank >= k_nn:
                    break

        # ── Session mean NN ───────────────────────────────────────────────────
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
                        if tid in seen:
                            continue
                        if tid not in cands_set:
                            cands.append(tid); cands_set.add(tid); sources[tid] = set()
                        sources[tid].add("mean_nn")
                        taken += 1
                        if taken >= SESSION_MEAN_K:
                            break

        # ── Co-occurrence ─────────────────────────────────────────────────────
        if music_history:
            for pos, k_co in enumerate(COOCCUR_KS):
                if k_co <= 0 or pos >= len(music_history):
                    break
                src_tid = music_history[-(pos + 1)]
                src_idx = cooccur_tid2idx.get(src_tid)
                if src_idx is None:
                    continue
                neighs = cooccur_neigh_ids[src_idx]
                taken = 0
                for rank in range(len(neighs)):
                    if taken >= k_co:
                        break
                    nidx = int(neighs[rank])
                    if nidx < 0:
                        break
                    tid = str(cooccur_track_ids[nidx])
                    if tid in seen:
                        continue
                    if tid not in cands_set:
                        cands.append(tid); cands_set.add(tid); sources[tid] = set()
                    sources[tid].add("collab")
                    if tid not in collab_rank_map or rank < collab_rank_map[tid]:
                        collab_rank_map[tid] = rank
                    taken += 1

        # ── Check reachability ────────────────────────────────────────────────
        if gold in cands_set:
            reachable += 1
            music_history.append(gold)
            continue

        # ── Gold is unreachable — diagnose ────────────────────────────────────
        gold_row = metadata_dict.get(gold, {})
        gold_artists_raw = gold_row.get("artist_name") or []
        gold_artists_lc  = [a.strip().lower() for a in gold_artists_raw]
        gold_pop = float(gold_row.get("popularity") or 0.0)
        gold_has_tags = bool(gold_row.get("tag_list"))

        # Full TT rank (0-indexed, lower = better)
        gold_tt_idx = tt_id2idx.get(gold)
        if gold_tt_idx is not None:
            gold_tt_rank = int(np.sum(tt_all > tt_all[gold_tt_idx]))
        else:
            gold_tt_rank = len(tt_ids)  # not embedded → max rank

        # Full Qwen-meta rank
        gold_qm_idx = qwen_meta_id2idx.get(gold)
        if gold_qm_idx is not None:
            gold_qm_rank = int(np.sum(qm_all > qm_all[gold_qm_idx]))
        else:
            gold_qm_rank = len(qwen_meta_ids)

        # Full CF rank (warm users only)
        if cf_all is not None:
            gold_cf_idx = cf_track_id2idx.get(gold)
            if gold_cf_idx is not None:
                gold_cf_rank = int(np.sum(cf_all > cf_all[gold_cf_idx]))
            else:
                gold_cf_rank = len(cf_track_ids)
        else:
            gold_cf_rank = None  # cold user

        # BM25 rank (expensive: retrieve deep)
        bm25_diag_tids, _ = retrieve_bm25(bm25_query, topk=BM25_DIAG_DEPTH)
        if gold in bm25_diag_tids:
            gold_bm25_rank = bm25_diag_tids.index(gold)
        else:
            gold_bm25_rank = BM25_DIAG_DEPTH  # not in top-5000

        # Co-occurrence rank (best across last-3 source tracks)
        gold_cooccur_rank = None
        for pos in range(min(3, len(music_history))):
            src_tid = music_history[-(pos + 1)]
            src_idx = cooccur_tid2idx.get(src_tid)
            if src_idx is None:
                continue
            neighs = cooccur_neigh_ids[src_idx]
            for rank in range(len(neighs)):
                nidx = int(neighs[rank])
                if nidx < 0:
                    break
                if str(cooccur_track_ids[nidx]) == gold:
                    if gold_cooccur_rank is None or rank < gold_cooccur_rank:
                        gold_cooccur_rank = rank
                    break

        # Artist-in-conversation check.
        # past_conv_text = only turns BEFORE this music turn (what the system sees).
        # full_conv_text = full session (includes future turns, diagnostic only).
        past_conv_text = " ".join(text_history).lower()
        full_conv_text = " ".join(
            t["content"] for t in conversations if t["role"] in ("user", "assistant")
        ).lower()
        artist_in_past_conv = any(a in past_conv_text for a in gold_artists_lc)
        artist_in_full_conv = any(a in full_conv_text for a in gold_artists_lc)
        artist_caught_by_system = any(a in mentioned for a in gold_artists_lc)

        # Bucket classification (uses past context only)
        if len(music_history) == 0:
            bucket = "cold_start"
        elif artist_in_past_conv and not artist_caught_by_system:
            bucket = "entity_link_gap"
        elif gold_tt_rank <= 3000:
            bucket = "mid_range_tt"        # TT@2000-3000 would rescue
        elif gold_bm25_rank < 1000 and gold_tt_rank > 5000:
            bucket = "bm25_only"           # BM25@1000 would rescue, TT can't
        elif gold_tt_rank > 5000 and gold_qm_rank > 5000 and gold_bm25_rank >= 1000:
            bucket = "truly_unreachable"   # no current signal can reach with bigger K
        else:
            bucket = "high_rank_tt"        # rank 3001-5000

        records.append({
            "session_id":       session_id,
            "turn_number":      turn_number,
            "turn_position":    len(music_history) + 1,  # 1-indexed position in session
            "music_history_len": len(music_history),
            "text_history_len": len(text_history),
            "cold_user":        user_emb is None,
            "gold":             gold,
            "gold_popularity":  gold_pop,
            "gold_has_tags":    gold_has_tags,
            "gold_tt_rank":     gold_tt_rank,
            "gold_qm_rank":     gold_qm_rank,
            "gold_cf_rank":     gold_cf_rank,
            "gold_bm25_rank":   gold_bm25_rank,
            "gold_cooccur_rank": gold_cooccur_rank,
            "artist_in_past_conv": artist_in_past_conv,
            "artist_in_full_conv": artist_in_full_conv,
            "artist_caught":    artist_caught_by_system,
            "bucket":           bucket,
            "pool_size":        len(cands),
        })

        music_history.append(gold)

# ── Analysis report ────────────────────────────────────────────────────────────
n_unreachable = len(records)
print(f"\n{'='*60}")
print(f"RECALL GAP ANALYSIS")
print(f"Sessions: {len(sessions)}   Turns: {total_turns}")
print(f"Reachable:   {reachable:,} / {total_turns:,}  ({100*reachable/total_turns:.1f}%)")
print(f"Unreachable: {n_unreachable:,} / {total_turns:,}  ({100*n_unreachable/total_turns:.1f}%)")
print(f"{'='*60}\n")

# Bucket summary
print("── Bucket breakdown ─────────────────────────────────────────")
bucket_counts = Counter(r["bucket"] for r in records)
for bucket, count in sorted(bucket_counts.items(), key=lambda x: -x[1]):
    pct_of_unreachable = 100 * count / n_unreachable
    pct_of_total = 100 * count / total_turns
    print(f"  {bucket:<22} {count:>5}  ({pct_of_unreachable:5.1f}% of unreachable, "
          f"{pct_of_total:4.1f}% of all turns)")

# Turn-position histogram
print("\n── Turn position (first=cold-start risk) ────────────────────")
pos_all = Counter(min(r["turn_position"], 8) for r in records)
pos_total = Counter()
for item in ds["test"]:  # recount total per position for denominator
    for t in item["conversations"]:
        if t["role"] == "music":
            pos_total[min(t["turn_number"], 8)] += 1
for pos in sorted(pos_all):
    label = f"turn {pos}" if pos < 8 else "turn 8+"
    cnt = pos_all[pos]
    denom = pos_total.get(pos, 1)
    print(f"  {label}: {cnt:>4} unreachable / {denom:>5} total  "
          f"({100*cnt/denom:5.1f}% unreachable rate)")

# Gold TT rank distribution
print("\n── Gold TT rank distribution (unreachable turns) ────────────")
ranks = [r["gold_tt_rank"] for r in records]
thresholds = [500, 1000, 1500, 2000, 3000, 5000, 10000]
for k in thresholds:
    cnt = sum(1 for x in ranks if x <= k)
    print(f"  TT rank <= {k:>5}: {cnt:>4} / {n_unreachable}  ({100*cnt/n_unreachable:5.1f}%)")
cnt_max = sum(1 for x in ranks if x >= len(tt_ids))
print(f"  Not in TT index:  {cnt_max:>4} / {n_unreachable}  ({100*cnt_max/n_unreachable:5.1f}%)")

# BM25 rank distribution
print("\n── Gold BM25 rank distribution (unreachable turns) ──────────")
bm25_ranks = [r["gold_bm25_rank"] for r in records]
for k in [200, 500, 1000, 2000, 5000]:
    cnt = sum(1 for x in bm25_ranks if x < k)
    print(f"  BM25 rank < {k:>4}: {cnt:>4} / {n_unreachable}  ({100*cnt/n_unreachable:5.1f}%)")
cnt_absent = sum(1 for x in bm25_ranks if x >= BM25_DIAG_DEPTH)
print(f"  Not in top-{BM25_DIAG_DEPTH}:  {cnt_absent:>4} / {n_unreachable}  ({100*cnt_absent/n_unreachable:5.1f}%)")

# Entity linking
print("\n── Artist mention analysis ───────────────────────────────────")
in_past  = sum(1 for r in records if r["artist_in_past_conv"])
in_full  = sum(1 for r in records if r["artist_in_full_conv"])
caught   = sum(1 for r in records if r["artist_caught"])
gap      = sum(1 for r in records if r["artist_in_past_conv"] and not r["artist_caught"])
future_only = sum(1 for r in records if r["artist_in_full_conv"] and not r["artist_in_past_conv"])
no_match = sum(1 for r in records if not r["artist_in_full_conv"])
print(f"  Artist in PAST context (verbatim): {in_past}  ({100*in_past/n_unreachable:.1f}%)")
print(f"    of which caught by current system: {caught}")
print(f"    entity linking gap (in past context, not caught): {gap}")
print(f"  Artist in FUTURE context only: {future_only}  ({100*future_only/n_unreachable:.1f}%)  [not fixable with linking]")
print(f"  Artist NOT in session at all: {no_match}  ({100*no_match/n_unreachable:.1f}%)")

# Popularity
print("\n── Gold track popularity ─────────────────────────────────────")
pops = [r["gold_popularity"] for r in records]
all_pops = [float(row.get("popularity") or 0.0) for row in metadata_dict.values()]
print(f"  Unreachable gold: mean={np.mean(pops):.3f}  median={np.median(pops):.3f}  "
      f"p10={np.percentile(pops,10):.3f}")
print(f"  Full catalog:     mean={np.mean(all_pops):.3f}  median={np.median(all_pops):.3f}  "
      f"p10={np.percentile(all_pops,10):.3f}")
zero_pop = sum(1 for p in pops if p == 0.0)
print(f"  Zero popularity:  {zero_pop} / {n_unreachable}  ({100*zero_pop/n_unreachable:.1f}%)")

# Has tags
no_tags = sum(1 for r in records if not r["gold_has_tags"])
print(f"\n── Gold track has no tags: {no_tags} / {n_unreachable}  ({100*no_tags/n_unreachable:.1f}%)")

# Save records
import json, pathlib
pathlib.Path(args.out).parent.mkdir(parents=True, exist_ok=True)
with open(args.out, "w") as f:
    json.dump(records, f, indent=2)
print(f"\nSaved {len(records)} records to {args.out}")
