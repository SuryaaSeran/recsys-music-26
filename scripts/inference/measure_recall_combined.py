"""
Measure pool recall with all improvements combined:
  BM25@500 + artist expansion + TT-v6@K

Also sweeps K to find the smallest K that crosses 80%.

Usage:
    python scripts/inference/measure_recall_combined.py
"""
import json
import numpy as np
from pathlib import Path

import bm25s
from datasets import load_dataset, concatenate_datasets
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

BM25_CACHE = "cache/bm25/track_metadata"
TT_MODEL   = "models/twotower_v6/final"
TT_INDEX   = "cache/twotower_v6"
BM25_POOL  = 500
K_LIST     = [100, 250, 500, 750, 1000, 1500]

print("Loading track metadata...")
meta_ds = load_dataset("talkpl-ai/TalkPlayData-Challenge-Track-Metadata")
all_tracks = concatenate_datasets([meta_ds["all_tracks"], meta_ds["test_tracks"]])
metadata_dict = {row["track_id"]: row for row in all_tracks}

artist_to_tids: dict[str, list[str]] = {}
for tid, row in metadata_dict.items():
    for a in (row.get("artist_name") or []):
        key = a.strip().lower()
        if key:
            artist_to_tids.setdefault(key, []).append(tid)
known_artists = sorted(artist_to_tids.keys(), key=len, reverse=True)
print(f"Known artists: {len(known_artists):,}")

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

print("Loading BM25 index...")
bm25_model = bm25s.BM25.load(BM25_CACHE, load_corpus=False)
with open(f"{BM25_CACHE}/track_ids.json") as f:
    bm25_track_ids = json.load(f)

print(f"Loading TT model + index...")
tt_model = SentenceTransformer(TT_MODEL)
tt_track_embs = np.load(f"{TT_INDEX}/track_embeddings.npy")
with open(f"{TT_INDEX}/track_ids.json") as f:
    tt_track_ids = json.load(f)
tt_id2idx = {tid: i for i, tid in enumerate(tt_track_ids)}

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
            # artist expansion: all text + played artists
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
            })
            music_history.append(gold)
        else:
            text_history.append(turn["content"])
            all_text.append(turn["content"])

print(f"Total turns: {len(all_turns)}")

# BM25
print("BM25 retrieval...")
bm25_pool_per_turn = []
for t in tqdm(all_turns, desc="BM25"):
    tokens = bm25s.tokenize([t["bm25_query"].lower()], show_progress=False)
    res = bm25_model.retrieve(tokens, k=BM25_POOL + 50, return_as="tuple", show_progress=False)
    seen = set(t["seen"])
    cands = []
    for idx in res.documents[0]:
        tid = bm25_track_ids[int(idx)]
        if tid not in seen and tid not in cands:
            cands.append(tid)
        if len(cands) >= BM25_POOL:
            break
    bm25_pool_per_turn.append(set(cands))

# Artist expansion
print("Artist expansion...")
expanded_pool_per_turn = []
for i, t in enumerate(all_turns):
    pool = set(bm25_pool_per_turn[i])
    seen = set(t["seen"])
    for a in t["mentioned_artists"]:
        for tid in artist_to_tids.get(a, []):
            if tid not in seen:
                pool.add(tid)
    expanded_pool_per_turn.append(pool)

# TT queries
print("Encoding TT queries...")
tt_q_embs = tt_model.encode(
    [t["tt_query"] for t in all_turns],
    batch_size=128, show_progress_bar=True,
    normalize_embeddings=True, convert_to_numpy=True,
)

# TT top-K (compute max K needed)
max_k = max(K_LIST)
print(f"Computing TT rankings (top-{max_k})...")
tt_top_sets = [set() for _ in all_turns]
batch = 256
for i in range(0, len(all_turns), batch):
    sims = tt_q_embs[i:i+batch] @ tt_track_embs.T
    top = np.argpartition(-sims, max_k, axis=1)[:, :max_k]
    for j in range(top.shape[0]):
        order = np.argsort(-sims[j, top[j]])
        tt_top_sets[i+j] = set(top[j][order].tolist())

N = len(all_turns)
print(f"\n=== Combined Pool Recall (N={N}) ===")
print(f"BM25@{BM25_POOL} only:           {sum(t['gold'] in bm25_pool_per_turn[i] for i,t in enumerate(all_turns))}/{N} = {sum(t['gold'] in bm25_pool_per_turn[i] for i,t in enumerate(all_turns))/N:.4f}")
print(f"BM25+artist only:       {sum(t['gold'] in expanded_pool_per_turn[i] for i,t in enumerate(all_turns))}/{N} = {sum(t['gold'] in expanded_pool_per_turn[i] for i,t in enumerate(all_turns))/N:.4f}")

for K in K_LIST:
    # TT top-K uses track index positions
    union_hits = 0
    for i, t in enumerate(all_turns):
        if t["gold"] in expanded_pool_per_turn[i]:
            union_hits += 1
            continue
        g = tt_id2idx.get(t["gold"])
        if g is not None:
            # check if g is in top-K of tt_top_sets[i]
            # need to recompute per K -- re-use stored sims would be better but we stored sets
            # Instead recompute here cleanly
            pass
    # Recompute properly with K cutoff
    union_hits = 0
    tt_top_k_per_turn = []
    batch2 = 512
    for i in range(0, len(all_turns), batch2):
        sims = tt_q_embs[i:i+batch2] @ tt_track_embs.T
        top = np.argpartition(-sims, K, axis=1)[:, :K]
        for j in range(top.shape[0]):
            tt_top_k_per_turn.append(set(top[j].tolist()))
    for i, t in enumerate(all_turns):
        if t["gold"] in expanded_pool_per_turn[i]:
            union_hits += 1
            continue
        g = tt_id2idx.get(t["gold"])
        if g is not None and g in tt_top_k_per_turn[i]:
            union_hits += 1
    print(f"BM25+artist+TT@{K:4d}:   {union_hits}/{N} = {union_hits/N:.4f}  (pool≈{BM25_POOL}+artist+{K})")

Path("exp/analysis").mkdir(parents=True, exist_ok=True)
with open("exp/analysis/recall_v6_combined.txt", "w") as f:
    f.write("Combined: BM25@500 + artist expansion + TT-v6\n")
print("\nDone.")
