"""
Precompute per-turn query embeddings and BM25 candidates for all dev sessions.

Saves a cache that allows rapid (seconds) weight-tuning without rerunning models.

Outputs:
  cache/dev_embeddings/turns.json     -- session_id, turn_number, user_id, gold
  cache/dev_embeddings/bm25_cands.npy -- (N_turns, bm25_pool) int32 BM25 candidate indices into track_ids
  cache/dev_embeddings/bm25_scores.npy -- (N_turns, bm25_pool) float32 raw BM25 scores
  cache/dev_embeddings/tt_embs.npy   -- (N_turns, 384) TwoTower query embeddings
  cache/dev_embeddings/qwen_embs.npy -- (N_turns, 1024) Qwen3 query embeddings
  cache/dev_embeddings/clap_embs.npy -- (N_turns, 512) CLAP query embeddings
  cache/dev_embeddings/cf_user.npy   -- (N_turns, 128) CF-BPR user embeddings (zeros if cold-start)
  cache/dev_embeddings/seen_mask.npy -- (N_turns, bm25_pool) bool mask of seen tracks in BM25 result

Usage:
    python scripts/precompute_embeddings.py
"""
import json
import re
import numpy as np
from pathlib import Path

import bm25s
import torch
from datasets import load_dataset, concatenate_datasets
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

BM25_CACHE = "cache/bm25/track_metadata"
BM25_POOL  = 500
OUT_DIR    = Path("cache/dev_embeddings")

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


print("Loading track metadata...")
meta_ds = load_dataset("talkpl-ai/TalkPlayData-Challenge-Track-Metadata")
all_tracks = concatenate_datasets([meta_ds["all_tracks"], meta_ds["test_tracks"]])
metadata_dict = {row["track_id"]: row for row in all_tracks}

def get_track_text(tid):
    row = metadata_dict.get(tid, {})
    return " ".join(filter(None, [
        (row.get("track_name") or [""])[0],
        (row.get("artist_name") or [""])[0],
        " ".join(row.get("tag_list") or []),
    ]))

def get_track_name_artist(tid):
    row = metadata_dict.get(tid, {})
    return " ".join(filter(None, [
        (row.get("track_name") or [""])[0],
        (row.get("artist_name") or [""])[0],
    ]))


print("Loading BM25 index...")
bm25_model = bm25s.BM25.load(BM25_CACHE, load_corpus=False)
with open(f"{BM25_CACHE}/track_ids.json") as f:
    bm25_track_ids = json.load(f)
bm25_id2idx = {tid: i for i, tid in enumerate(bm25_track_ids)}

print("Loading TwoTower model...")
tt_model = SentenceTransformer("models/twotower_v3/final")

print("Loading Qwen3-Embedding-0.6B...")
qwen_model = SentenceTransformer("Qwen/Qwen3-Embedding-0.6B", trust_remote_code=True)
QWEN_INSTR = "Instruct: Given a music listener's request, retrieve relevant music tracks\nQuery: "

print("Loading LAION CLAP model (HTSAT-tiny)...")
import laion_clap
clap_model = laion_clap.CLAP_Module(enable_fusion=False, amodel="HTSAT-tiny")
clap_model.load_ckpt(verbose=False)
clap_model.eval()

print("Loading Qwen3 attributes index (for style-history signal)...")
qwen_attr_embs = np.load("cache/qwen3_attr/track_embeddings.npy")
with open("cache/qwen3_attr/track_ids.json") as f:
    qwen_attr_ids = json.load(f)
qwen_attr_id2idx = {tid: i for i, tid in enumerate(qwen_attr_ids)}

print("Loading CF-BPR user embeddings...")
with open("cache/user_cf_bpr.json") as f:
    user_cf_raw = json.load(f)
user_cf = {}
for uid, vec in user_cf_raw.items():
    if not vec or len(vec) != 128:
        continue
    v = np.array(vec, dtype=np.float32)
    n = np.linalg.norm(v)
    if n > 1e-8:
        user_cf[uid] = v / n

print("Loading dev sessions...")
ds = load_dataset("talkpl-ai/TalkPlayData-Challenge-Dataset")
sessions = list(ds["test"])

# First pass: count turns
n_turns = sum(
    sum(1 for t in item["conversations"] if t["role"] == "music")
    for item in sessions
)
print(f"Total turns: {n_turns}")

# Allocate output arrays
turns_meta = []  # list of dicts: session_id, turn_number, user_id, gold
bm25_cands_arr   = np.full((n_turns, BM25_POOL), -1, dtype=np.int32)
bm25_scores_arr  = np.zeros((n_turns, BM25_POOL), dtype=np.float32)
tt_embs_arr      = np.zeros((n_turns, 384),       dtype=np.float32)
qwen_embs_arr    = np.zeros((n_turns, 1024),      dtype=np.float32)
clap_embs_arr    = np.zeros((n_turns, 512),       dtype=np.float32)
cf_user_arr      = np.zeros((n_turns, 128),       dtype=np.float32)
attrs_hist_arr   = np.zeros((n_turns, 1024),      dtype=np.float32)  # avg attrs of last 4 tracks

turn_idx = 0
for item in tqdm(sessions, desc="Sessions"):
    session_id   = item["session_id"]
    user_id      = item["user_id"]
    goal         = item.get("conversation_goal", {}).get("listener_goal", "")
    culture      = item.get("user_profile", {}).get("preferred_musical_culture", "")
    conversations = item["conversations"]

    user_emb_vec = user_cf.get(user_id)
    if user_emb_vec is not None:
        pass  # store per-turn below

    music_history = []
    text_history  = []

    for turn in conversations:
        if turn["role"] != "music":
            if turn["role"] in ("user", "assistant"):
                text_history.append(turn["content"])
            continue

        gold        = turn["content"]
        turn_number = turn["turn_number"]
        seen        = set(music_history)

        # BM25 query
        bm25_parts = [goal, culture]
        for tid in music_history[-4:]:
            bm25_parts.append(get_track_text(tid))
        bm25_parts.extend(text_history[-4:])
        bm25_query = " ".join(p for p in bm25_parts if p)

        # TwoTower compact query
        latest_user = text_history[-1] if text_history else ""
        tt_parts = [latest_user, goal, culture]
        for tid in music_history[-2:]:
            na = get_track_name_artist(tid)
            if na:
                tt_parts.append(na)
        tt_query = " ".join(p for p in tt_parts if p)

        # Qwen3/CLAP semantic query
        cleaned = clean_query(latest_user) or latest_user
        sem_parts = [cleaned, goal, culture]
        for tid in music_history[-2:]:
            na = get_track_name_artist(tid)
            if na:
                sem_parts.append(na)
        semantic_query = " ".join(p for p in sem_parts if p)

        # BM25 retrieval
        tokens = bm25s.tokenize([bm25_query.lower()], show_progress=False)
        retrieve_k = BM25_POOL + len(seen) * 3
        results = bm25_model.retrieve(tokens, k=retrieve_k, return_as="tuple", show_progress=False)
        raw_tids   = [bm25_track_ids[int(i)] for i in results.documents[0]]
        raw_scores = list(results.scores[0])
        filtered   = [(t, s) for t, s in zip(raw_tids, raw_scores) if t not in seen][:BM25_POOL]

        for j, (tid, score) in enumerate(filtered):
            idx = bm25_id2idx.get(tid, -1)
            bm25_cands_arr[turn_idx, j]  = idx
            bm25_scores_arr[turn_idx, j] = score

        # Encode queries
        tt_emb = tt_model.encode(tt_query, normalize_embeddings=True, convert_to_numpy=True)
        tt_embs_arr[turn_idx] = tt_emb

        qwen_emb = qwen_model.encode(
            QWEN_INSTR + semantic_query,
            normalize_embeddings=True, convert_to_numpy=True,
        )
        qwen_embs_arr[turn_idx] = qwen_emb

        with torch.no_grad():
            clap_raw = clap_model.get_text_embedding([semantic_query], use_tensor=True)
        clap_emb = clap_raw[0].cpu().numpy().astype(np.float32)
        norm = np.linalg.norm(clap_emb)
        clap_embs_arr[turn_idx] = clap_emb / max(norm, 1e-8)

        if user_emb_vec is not None:
            cf_user_arr[turn_idx] = user_emb_vec

        # Style-history: avg attrs embedding of last 4 played tracks
        if music_history:
            hist_vecs = []
            for hist_tid in music_history[-4:]:
                idx_ah = qwen_attr_id2idx.get(hist_tid)
                if idx_ah is not None:
                    hist_vecs.append(qwen_attr_embs[idx_ah])
            if hist_vecs:
                avg = np.mean(hist_vecs, axis=0).astype(np.float32)
                n_avg = np.linalg.norm(avg)
                if n_avg > 1e-8:
                    attrs_hist_arr[turn_idx] = avg / n_avg

        turns_meta.append({
            "session_id":  session_id,
            "user_id":     user_id,
            "turn_number": turn_number,
            "gold":        gold,
            "has_cf":      user_emb_vec is not None,
        })

        music_history.append(gold)
        turn_idx += 1

OUT_DIR.mkdir(parents=True, exist_ok=True)
print(f"Saving {turn_idx} turns to {OUT_DIR}...")

with open(OUT_DIR / "turns.json", "w") as f:
    json.dump(turns_meta, f)

np.save(OUT_DIR / "bm25_cands.npy",   bm25_cands_arr[:turn_idx])
np.save(OUT_DIR / "bm25_scores.npy",  bm25_scores_arr[:turn_idx])
np.save(OUT_DIR / "tt_embs.npy",      tt_embs_arr[:turn_idx])
np.save(OUT_DIR / "qwen_embs.npy",    qwen_embs_arr[:turn_idx])
np.save(OUT_DIR / "clap_embs.npy",    clap_embs_arr[:turn_idx])
np.save(OUT_DIR / "cf_user.npy",      cf_user_arr[:turn_idx])
np.save(OUT_DIR / "attrs_hist.npy",   attrs_hist_arr[:turn_idx])

print("Done.")
