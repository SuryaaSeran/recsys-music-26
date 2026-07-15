"""
Multi-signal fusion inference: BM25 + CF-BPR + Qwen3-metadata + CLAP + two-tower.

Retrieval: BM25 top-bm25_pool candidates (unchanged from v3)
Scoring signals:
  1. two-tower v3 cosine     (384-dim, fine-tuned on task)
  2. CF-BPR user-track sim   (128-dim, collaborative filtering)
  3. Qwen3 metadata cosine   (1024-dim, semantic metadata match)
  4. CLAP text-audio cosine  (512-dim, audio-semantic match)
  5. BM25 reciprocal rank

Query cleaning for Qwen3/CLAP: strip conversational filler, keep music descriptors.

Usage:
    python scripts/run_inference_fusion.py --sessions 200 --tid fusion_v1_200
"""
import argparse
import json
import re
import numpy as np
from pathlib import Path

import bm25s
import torch
from datasets import load_dataset, concatenate_datasets
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument("--tt_model",    default="models/twotower_v3/final")
parser.add_argument("--tt_index",    default="cache/twotower_v3")
parser.add_argument("--sessions",    type=int,   default=0)
parser.add_argument("--tid",         default="fusion_v1")
parser.add_argument("--out_dir",     default="exp/inference/devset")
parser.add_argument("--topk",        type=int,   default=20)
parser.add_argument("--bm25_pool",   type=int,   default=500)
parser.add_argument("--hist_turns",  type=int,   default=4)
parser.add_argument("--text_turns",  type=int,   default=4)
# Signal weights
parser.add_argument("--w_tt",          type=float, default=0.35, help="Two-tower weight")
parser.add_argument("--w_cf",          type=float, default=0.08, help="CF-BPR weight")
parser.add_argument("--w_qwen_meta",   type=float, default=0.20, help="Qwen3 metadata weight")
parser.add_argument("--w_qwen_attr",   type=float, default=0.10, help="Qwen3 attributes weight")
parser.add_argument("--w_qwen_lyrics", type=float, default=0.10, help="Qwen3 lyrics weight")
parser.add_argument("--w_clap",        type=float, default=0.07, help="CLAP audio weight")
parser.add_argument("--w_bm25",        type=float, default=0.10, help="BM25 RR weight")
parser.add_argument("--cf_pool",       type=int,   default=0,   help="CF-BPR retrieval: add top-N user-affinity tracks to pool. 0=disabled.")
parser.add_argument("--qwen_pool",     type=int,   default=0,   help="Qwen3 retrieval: add top-N semantic tracks to pool. 0=disabled.")
parser.add_argument("--tt_pool",       type=int,   default=0,   help="TwoTower dense retrieval: add top-N to pool. 0=disabled.")
parser.add_argument("--bm25_norm",     action="store_true", help="Use normalized BM25 score instead of reciprocal rank.")
parser.add_argument("--sem_hist",      type=int,   default=2,   help="Last N track names to append to Qwen3/CLAP semantic query.")
parser.add_argument("--w_attrs_hist",  type=float, default=0.0, help="Weight for style-history signal: avg attrs embedding of last N played tracks vs candidate attrs.")
parser.add_argument("--attrs_hist_n",  type=int,   default=4,   help="Last N tracks to average for attrs-history signal.")
args = parser.parse_args()

BM25_CACHE = "cache/bm25/track_metadata"

# Conversational filler patterns to strip before Qwen3/CLAP encoding
_FILLER = re.compile(
    r"\b(can you|could you|would you|please|i want|i'd like|i would like|"
    r"i need|i'm looking for|i am looking for|something that(?:'s| is)|something|"
    r"recommend(?:ation)?|suggest(?:ion)?|play(?: me)?|find me|show me|give me|"
    r"how about|what about|i feel like(?: listening to)?|i(?:'m| am) in the mood for|"
    r"do you have|do you know)\b",
    re.IGNORECASE,
)

def clean_query(text: str) -> str:
    text = _FILLER.sub(" ", text)
    return re.sub(r"\s+", " ", text).strip()


print("Loading track metadata...")
meta_ds = load_dataset("talkpl-ai/TalkPlayData-Challenge-Track-Metadata")
all_tracks = concatenate_datasets([meta_ds["all_tracks"], meta_ds["test_tracks"]])
metadata_dict = {row["track_id"]: row for row in all_tracks}


def get_track_text(tid: str) -> str:
    row = metadata_dict.get(tid, {})
    name   = (row.get("track_name")   or [""])[0]
    artist = (row.get("artist_name")  or [""])[0]
    tags   = " ".join(row.get("tag_list") or [])
    return f"{name} {artist} {tags}".strip()


def get_track_name_artist(tid: str) -> str:
    row = metadata_dict.get(tid, {})
    name   = (row.get("track_name")  or [""])[0]
    artist = (row.get("artist_name") or [""])[0]
    return f"{name} {artist}".strip()


print("Building artist->tracks lookup...")
artist_to_tids: dict[str, list[str]] = {}
for _tid, _row in metadata_dict.items():
    for _a in (_row.get("artist_name") or []):
        _key = _a.strip().lower()
        if _key:
            artist_to_tids.setdefault(_key, []).append(_tid)
_known_artists = sorted(artist_to_tids.keys(), key=len, reverse=True)

def find_mentioned_artists(text: str) -> list[str]:
    tl = text.lower()
    found = []
    for a in _known_artists:
        if a in tl:
            found.append(a)
            tl = tl.replace(a, " " * len(a))
    return found

print("Loading BM25 index...")
bm25_model = bm25s.BM25.load(BM25_CACHE, load_corpus=False)
with open(f"{BM25_CACHE}/track_ids.json") as f:
    bm25_track_ids = json.load(f)


def retrieve_bm25(query: str, topk: int) -> tuple[list[str], list[float]]:
    tokens = bm25s.tokenize([query.lower()])
    results = bm25_model.retrieve(tokens, k=topk, return_as="tuple")
    tids = [bm25_track_ids[int(i)] for i in results.documents[0]]
    scores = [float(s) for s in results.scores[0]]
    return tids, scores


print(f"Loading two-tower model: {args.tt_model}")
tt_model = SentenceTransformer(args.tt_model)

print(f"Loading two-tower index: {args.tt_index}")
tt_embs = np.load(f"{args.tt_index}/track_embeddings.npy")
with open(f"{args.tt_index}/track_ids.json") as f:
    tt_ids = json.load(f)
tt_id2idx = {tid: i for i, tid in enumerate(tt_ids)}

print("Loading Qwen3-Embedding-0.6B...")
qwen_model = SentenceTransformer("Qwen/Qwen3-Embedding-0.6B", trust_remote_code=True)
QWEN_INSTR = "Instruct: Given a music listener's request, retrieve relevant music tracks\nQuery: "

print("Loading Qwen3 metadata index...")
qwen_meta_embs = np.load("cache/qwen3_meta/track_embeddings.npy")
with open("cache/qwen3_meta/track_ids.json") as f:
    qwen_meta_ids = json.load(f)
qwen_meta_id2idx = {tid: i for i, tid in enumerate(qwen_meta_ids)}

print("Loading Qwen3 attributes index...")
qwen_attr_embs = np.load("cache/qwen3_attr/track_embeddings.npy")
with open("cache/qwen3_attr/track_ids.json") as f:
    qwen_attr_ids = json.load(f)
qwen_attr_id2idx = {tid: i for i, tid in enumerate(qwen_attr_ids)}

print("Loading Qwen3 lyrics index...")
qwen_lyrics_embs = np.load("cache/qwen3_lyrics/track_embeddings.npy")
with open("cache/qwen3_lyrics/track_ids.json") as f:
    qwen_lyrics_ids = json.load(f)
qwen_lyrics_id2idx = {tid: i for i, tid in enumerate(qwen_lyrics_ids)}

print("Loading LAION CLAP model (HTSAT-tiny)...")
import laion_clap
clap_model = laion_clap.CLAP_Module(enable_fusion=False, amodel="HTSAT-tiny")
clap_model.load_ckpt(verbose=False)
clap_model.eval()

print("Loading CLAP audio index...")
clap_embs = np.load("cache/clap/track_embeddings.npy")
with open("cache/clap/track_ids.json") as f:
    clap_ids = json.load(f)
clap_id2idx = {tid: i for i, tid in enumerate(clap_ids)}

print("Loading CF-BPR embeddings...")
cf_track_embs = np.load("cache/cf_bpr/track_embeddings.npy")
with open("cache/cf_bpr/track_ids.json") as f:
    cf_track_ids = json.load(f)
cf_track_id2idx = {tid: i for i, tid in enumerate(cf_track_ids)}

with open("cache/user_cf_bpr.json") as f:
    user_cf_raw = json.load(f)
# Pre-normalise user vectors (skip empty/missing)
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
if args.sessions > 0:
    sessions = sessions[:args.sessions]

print(
    f"Running {len(sessions)} sessions  "
    f"w_tt={args.w_tt} w_cf={args.w_cf} "
    f"w_qwen_meta={args.w_qwen_meta} w_qwen_attr={args.w_qwen_attr} w_qwen_lyrics={args.w_qwen_lyrics} "
    f"w_clap={args.w_clap} w_bm25={args.w_bm25}"
)

inference_results = []

for item in tqdm(sessions, desc="Sessions"):
    session_id  = item["session_id"]
    user_id     = item["user_id"]
    goal        = item.get("conversation_goal", {}).get("listener_goal", "")
    culture     = item.get("user_profile", {}).get("preferred_musical_culture", "")
    conversations = item["conversations"]

    user_emb = user_cf.get(user_id)   # None if cold-start user

    music_history: list[str] = []
    text_history:  list[str] = []
    all_text_so_far: list[str] = []

    for turn in conversations:
        if turn["role"] != "music":
            if turn["role"] in ("user", "assistant"):
                text_history.append(turn["content"])
                all_text_so_far.append(turn["content"])
            continue

        turn_number = turn["turn_number"]
        seen = set(music_history)

        # --- Query construction ---
        latest_user = text_history[-1] if text_history else ""

        # Two-tower compact query (matches training format)
        tt_parts = [latest_user, goal, culture]
        for tid in music_history[-2:]:
            na = get_track_name_artist(tid)
            if na:
                tt_parts.append(na)
        tt_query = " ".join(p for p in tt_parts if p)

        # BM25 full query
        bm25_parts = [goal, culture]
        for tid in music_history[-args.hist_turns:]:
            bm25_parts.append(get_track_text(tid))
        bm25_parts.extend(text_history[-args.text_turns:])
        bm25_query = " ".join(p for p in bm25_parts if p)

        # Qwen3/CLAP query: clean conversational filler + add last N track context
        cleaned = clean_query(latest_user) or latest_user
        sem_parts = [cleaned, goal, culture]
        for tid in music_history[-args.sem_hist:]:
            na = get_track_name_artist(tid)
            if na:
                sem_parts.append(na)
        semantic_query = " ".join(p for p in sem_parts if p)

        # --- BM25 recall ---
        retrieve_k = args.bm25_pool + len(seen) * 3
        raw_tids, raw_scores = retrieve_bm25(bm25_query, topk=retrieve_k)
        filtered = [(t, s) for t, s in zip(raw_tids, raw_scores) if t not in seen][:args.bm25_pool]
        cands = [t for t, _ in filtered]
        bm25_scores = [s for _, s in filtered]

        # --- Artist expansion: add all tracks by mentioned artists ---
        mentioned_artists: set[str] = set()
        for txt in all_text_so_far:
            for a in find_mentioned_artists(txt):
                mentioned_artists.add(a)
        for tid in music_history:
            for a in (metadata_dict.get(tid, {}).get("artist_name") or []):
                mentioned_artists.add(a.strip().lower())
        cands_set = set(cands)
        for a in mentioned_artists:
            for tid in artist_to_tids.get(a, []):
                if tid not in cands_set and tid not in seen:
                    cands.append(tid)
                    bm25_scores.append(0.0)
                    cands_set.add(tid)

        # --- CF-BPR retrieval: add user-affinity tracks to pool ---
        if args.cf_pool > 0 and user_emb is not None:
            cf_all = cf_track_embs @ user_emb
            top_cf_idx = np.argpartition(cf_all, -args.cf_pool)[-args.cf_pool:]
            top_cf_idx = top_cf_idx[np.argsort(cf_all[top_cf_idx])[::-1]]
            bm25_set = set(cands)
            for idx in top_cf_idx:
                tid = cf_track_ids[int(idx)]
                if tid not in bm25_set and tid not in seen:
                    cands.append(tid)
                    bm25_scores.append(0.0)

        if not cands:
            inference_results.append({
                "session_id": session_id, "user_id": user_id,
                "turn_number": turn_number,
                "predicted_track_ids": [], "predicted_response": "No recommendation.",
            })
            music_history.append(turn["content"])
            continue

        if args.bm25_norm:
            max_s = bm25_scores[0] if bm25_scores and bm25_scores[0] > 1e-8 else 1.0
            bm25_rr = {tid: s / max_s for tid, s in zip(cands, bm25_scores)}
        else:
            bm25_rr = {tid: 1.0 / (r + 1) for r, tid in enumerate(cands)}

        # --- Encode queries ---
        tt_emb = tt_model.encode(tt_query, normalize_embeddings=True, convert_to_numpy=True)

        qwen_emb = qwen_model.encode(
            QWEN_INSTR + semantic_query,
            normalize_embeddings=True, convert_to_numpy=True,
        )

        with torch.no_grad():
            clap_raw = clap_model.get_text_embedding([semantic_query], use_tensor=True)
        clap_emb = clap_raw[0].cpu().numpy().astype(np.float32)
        clap_emb = clap_emb / max(np.linalg.norm(clap_emb), 1e-8)

        # --- Qwen3 dense retrieval: add semantic candidates not in BM25 pool ---
        if args.qwen_pool > 0:
            qwen_all = qwen_meta_embs @ qwen_emb
            top_q_idx = np.argpartition(qwen_all, -args.qwen_pool)[-args.qwen_pool:]
            top_q_idx = top_q_idx[np.argsort(qwen_all[top_q_idx])[::-1]]
            bm25_set = set(cands)
            for idx in top_q_idx:
                tid = qwen_meta_ids[int(idx)]
                if tid not in bm25_set and tid not in seen:
                    cands.append(tid)
                    bm25_scores.append(0.0)

        # --- TwoTower dense retrieval: add dense-top candidates to pool ---
        if args.tt_pool > 0:
            tt_all_idx = tt_embs @ tt_emb
            top_tt_idx = np.argpartition(tt_all_idx, -args.tt_pool)[-args.tt_pool:]
            top_tt_idx = top_tt_idx[np.argsort(tt_all_idx[top_tt_idx])[::-1]]
            bm25_set = set(cands)
            for idx in top_tt_idx:
                tid = tt_ids[int(idx)]
                if tid not in bm25_set and tid not in seen:
                    cands.append(tid)
                    bm25_scores.append(0.0)

        # --- Style-history signal: avg attrs embedding of last N played tracks ---
        attrs_hist_emb = None
        if args.w_attrs_hist > 0 and music_history:
            hist_vecs = []
            for hist_tid in music_history[-args.attrs_hist_n:]:
                idx_ah = qwen_attr_id2idx.get(hist_tid)
                if idx_ah is not None:
                    hist_vecs.append(qwen_attr_embs[idx_ah])
            if hist_vecs:
                avg = np.mean(hist_vecs, axis=0)
                n = np.linalg.norm(avg)
                if n > 1e-8:
                    attrs_hist_emb = avg / n

        # --- Score all candidates: precompute full-index scores, then index ---
        tt_all   = tt_embs        @ tt_emb    # (N_tt,)
        qm_all   = qwen_meta_embs @ qwen_emb  # (N_qwen,)
        ql_all   = (qwen_lyrics_embs @ qwen_emb) if args.w_qwen_lyrics > 0 else None
        clap_all = clap_embs      @ clap_emb  # (N_clap,)
        cf_all   = (cf_track_embs @ user_emb) if user_emb is not None else None
        ah_all   = (qwen_attr_embs @ attrs_hist_emb) if attrs_hist_emb is not None else None

        n_cands = len(cands)
        total_arr = np.zeros(n_cands, dtype=np.float32)
        for i, tid in enumerate(cands):
            bm25_s = bm25_rr.get(tid, 0.0)
            idx_tt = tt_id2idx.get(tid)
            idx_qm = qwen_meta_id2idx.get(tid)
            idx_ql = qwen_lyrics_id2idx.get(tid) if ql_all is not None else None
            idx_c  = clap_id2idx.get(tid)
            idx_cf = cf_track_id2idx.get(tid) if user_emb is not None else None
            idx_ah = qwen_attr_id2idx.get(tid) if attrs_hist_emb is not None else None

            total_arr[i] = (
                args.w_tt         * (float(tt_all[idx_tt])     if idx_tt is not None else 0.0) +
                args.w_qwen_meta  * (float(qm_all[idx_qm])     if idx_qm is not None else 0.0) +
                args.w_qwen_lyrics * (float(ql_all[idx_ql])    if idx_ql is not None and ql_all is not None else 0.0) +
                args.w_clap       * (float(clap_all[idx_c])    if idx_c  is not None else 0.0) +
                args.w_cf         * (float(cf_all[idx_cf])     if idx_cf is not None and cf_all is not None else 0.0) +
                args.w_attrs_hist * (float(ah_all[idx_ah])     if idx_ah is not None and ah_all is not None else 0.0) +
                args.w_bm25       * bm25_s
            )

        top_idx = np.argsort(total_arr)[::-1][:args.topk]
        predicted_track_ids = [cands[i] for i in top_idx]

        top = predicted_track_ids[0] if predicted_track_ids else ""
        row = metadata_dict.get(top, {})
        name   = (row.get("track_name")  or ["this track"])[0]
        artist = (row.get("artist_name") or ["the artist"])[0]

        inference_results.append({
            "session_id": session_id, "user_id": user_id,
            "turn_number": turn_number,
            "predicted_track_ids": predicted_track_ids,
            "predicted_response": f'I recommend "{name}" by {artist} based on your request.',
        })
        music_history.append(turn["content"])

Path(args.out_dir).mkdir(parents=True, exist_ok=True)
out_path = Path(args.out_dir) / f"{args.tid}.json"
with open(out_path, "w") as f:
    json.dump(inference_results, f, ensure_ascii=False, indent=2)
print(f"Saved {len(inference_results):,} predictions to {out_path}")
