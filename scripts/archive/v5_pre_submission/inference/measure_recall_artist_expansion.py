"""
Measure pool recall with artist-based expansion.

When any conversation turn (user or previously played tracks) explicitly mentions
an artist that exists in the catalog, add ALL that artist's tracks to the pool.

This directly targets "more songs by X" and "find me Y by artist Z" queries.

Usage:
    python scripts/inference/measure_recall_artist_expansion.py
"""
import json
import re
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
TT_K       = 500
LABEL      = "v6_artist_expansion"

print("Loading track metadata...")
meta_ds = load_dataset("talkpl-ai/TalkPlayData-Challenge-Track-Metadata")
all_tracks = concatenate_datasets([meta_ds["all_tracks"], meta_ds["test_tracks"]])
metadata_dict = {row["track_id"]: row for row in all_tracks}

# Build artist -> track_ids lookup (lowercase)
artist_to_tids: dict[str, list[str]] = {}
for tid, row in metadata_dict.items():
    for a in (row.get("artist_name") or []):
        key = a.strip().lower()
        if key:
            artist_to_tids.setdefault(key, []).append(tid)

# Build set of known artist names sorted by length descending (longer match first)
known_artists = sorted(artist_to_tids.keys(), key=len, reverse=True)
print(f"Known artists: {len(known_artists):,}")


def find_mentioned_artists(text: str) -> list[str]:
    """Return catalog artist names found verbatim in text (case-insensitive)."""
    tl = text.lower()
    found = []
    for a in known_artists:
        if a in tl:
            found.append(a)
            # mask to avoid sub-matches
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

print(f"Loading TT model: {TT_MODEL}")
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
    music_history = []
    text_history = []
    all_text_so_far = []  # every user/asst utterance so far for artist detection

    for turn in sess["conversations"]:
        if turn["role"] == "music":
            gold = turn["content"]
            latest_user = text_history[-1] if text_history else ""

            # BM25 query
            bm25_parts = [goal, culture]
            for tid in music_history[-4:]:
                bm25_parts.append(get_track_text(tid))
            bm25_parts.extend(text_history[-4:])
            bm25_query = " ".join(p for p in bm25_parts if p)

            # TT query
            tt_parts = [latest_user, goal, culture]
            for tid in music_history[-2:]:
                na = get_track_name_artist(tid)
                if na: tt_parts.append(na)
            tt_query = " ".join(p for p in tt_parts if p)

            # Artist detection: scan all text so far + played track artists
            mentioned = set()
            for txt in all_text_so_far:
                for a in find_mentioned_artists(txt):
                    mentioned.add(a)
            # Also add artists of played tracks
            for tid in music_history:
                row = metadata_dict.get(tid, {})
                for a in (row.get("artist_name") or []):
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
            all_text_so_far.append(turn["content"])

print(f"Total turns: {len(all_turns)}")

# --- BM25 retrieval ---
print("BM25 retrieval...")
bm25_pool_per_turn = []
for t in tqdm(all_turns, desc="BM25"):
    tokens = bm25s.tokenize([t["bm25_query"].lower()], show_progress=False)
    res = bm25_model.retrieve(tokens, k=BM25_POOL + 50, return_as="tuple", show_progress=False)
    seen = set(t["seen"])
    cands = []
    for idx in res.documents[0]:
        tid = bm25_track_ids[int(idx)]
        if tid not in seen:
            cands.append(tid)
        if len(cands) >= BM25_POOL:
            break
    bm25_pool_per_turn.append(set(cands))

# --- Artist expansion ---
print("Applying artist expansion...")
artist_expanded_pool_per_turn = []
for i, t in enumerate(all_turns):
    base = set(bm25_pool_per_turn[i])
    seen = set(t["seen"])
    for a in t["mentioned_artists"]:
        for tid in artist_to_tids.get(a, []):
            if tid not in seen:
                base.add(tid)
    artist_expanded_pool_per_turn.append(base)

# --- TT encoding ---
print("Encoding TT queries...")
tt_q_embs = tt_model.encode(
    [t["tt_query"] for t in all_turns],
    batch_size=128, show_progress_bar=True,
    normalize_embeddings=True, convert_to_numpy=True,
)

# --- TT top-K ---
print("Computing TT rankings...")
top_k_max = TT_K
tt_top_idx = np.zeros((len(all_turns), top_k_max), dtype=np.int32)
batch = 256
for i in range(0, len(all_turns), batch):
    sims = tt_q_embs[i:i+batch] @ tt_track_embs.T
    top = np.argpartition(-sims, top_k_max, axis=1)[:, :top_k_max]
    for j in range(top.shape[0]):
        order = np.argsort(-sims[j, top[j]])
        tt_top_idx[i+j] = top[j][order]

# --- Recall measurement ---
N = len(all_turns)
bm25_hits     = sum(1 for i, t in enumerate(all_turns) if t["gold"] in bm25_pool_per_turn[i])
artist_hits   = sum(1 for i, t in enumerate(all_turns) if t["gold"] in artist_expanded_pool_per_turn[i])

tt_hits = 0
for i, t in enumerate(all_turns):
    g = tt_id2idx.get(t["gold"])
    if g is not None and g in tt_top_idx[i]:
        tt_hits += 1

union_base = 0
union_artist = 0
for i, t in enumerate(all_turns):
    gold_in_bm25 = t["gold"] in bm25_pool_per_turn[i]
    gold_in_artist = t["gold"] in artist_expanded_pool_per_turn[i]
    g = tt_id2idx.get(t["gold"])
    gold_in_tt = g is not None and g in tt_top_idx[i]
    if gold_in_bm25 or gold_in_tt:
        union_base += 1
    if gold_in_artist or gold_in_tt:
        union_artist += 1

print(f"\n=== Pool Recall (N={N} turns) ===")
print(f"BM25@{BM25_POOL} only:                     {bm25_hits}/{N} = {bm25_hits/N:.4f}")
print(f"BM25@{BM25_POOL} + artist expansion:        {artist_hits}/{N} = {artist_hits/N:.4f}")
print(f"TT-v6@{TT_K} only:                         {tt_hits}/{N} = {tt_hits/N:.4f}")
print(f"BM25@{BM25_POOL} + TT@{TT_K}:              {union_base}/{N} = {union_base/N:.4f}")
print(f"BM25@{BM25_POOL} + artist + TT@{TT_K}:     {union_artist}/{N} = {union_artist/N:.4f}")

# How many turns benefited from artist expansion?
artist_gain = sum(1 for i, t in enumerate(all_turns)
                  if t["gold"] not in bm25_pool_per_turn[i]
                  and t["gold"] in artist_expanded_pool_per_turn[i])
print(f"\nArtist expansion rescued {artist_gain} BM25 misses ({artist_gain/N:.3f})")
turns_with_artists = sum(1 for t in all_turns if t["mentioned_artists"])
print(f"Turns with ≥1 mentioned artist: {turns_with_artists}/{N} ({turns_with_artists/N:.3f})")

Path("exp/analysis").mkdir(parents=True, exist_ok=True)
with open(f"exp/analysis/recall_{LABEL}.txt", "w") as f:
    f.write(f"BM25@{BM25_POOL}: {bm25_hits/N:.4f}\n")
    f.write(f"BM25+artist: {artist_hits/N:.4f}\n")
    f.write(f"BM25+TT@{TT_K}: {union_base/N:.4f}\n")
    f.write(f"BM25+artist+TT@{TT_K}: {union_artist/N:.4f}\n")
print(f"Saved to exp/analysis/recall_{LABEL}.txt")
