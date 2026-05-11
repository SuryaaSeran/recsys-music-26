"""
Blind-set fusion inference: BM25 + CF-BPR + Qwen3-metadata + CLAP + two-tower.

Same scoring as run_inference_fusion.py but for blind_a format:
  - conversations[-1] is the user query (no ground truth)
  - history is all prior turns

Usage:
    python scripts/run_inference_blind_fusion.py \
        --tid blind_a_fusion_v6 \
        --w_tt 0.35 --w_cf 0.12 --w_qwen_meta 0.30 --w_clap 0.10 --w_bm25 0.13
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
parser.add_argument("--dataset",     default="talkpl-ai/TalkPlayData-Challenge-Blind-A")
parser.add_argument("--split",       default="test")
parser.add_argument("--tid",         default="blind_a_fusion")
parser.add_argument("--out_dir",     default="exp/inference/blind_a")
parser.add_argument("--topk",        type=int,   default=20)
parser.add_argument("--bm25_pool",   type=int,   default=500)
parser.add_argument("--hist_turns",  type=int,   default=4)
parser.add_argument("--text_turns",  type=int,   default=4)
parser.add_argument("--w_tt",           type=float, default=0.35)
parser.add_argument("--w_cf",           type=float, default=0.12)
parser.add_argument("--w_qwen_meta",    type=float, default=0.30)
parser.add_argument("--w_qwen_lyrics",  type=float, default=0.0)
parser.add_argument("--w_clap",         type=float, default=0.10)
parser.add_argument("--w_bm25",         type=float, default=0.13)
parser.add_argument("--bm25_norm",      action="store_true")
parser.add_argument("--sem_hist",       type=int,   default=2)
args = parser.parse_args()

BM25_CACHE = "cache/bm25/track_metadata"

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
user_cf = {}
for uid, vec in user_cf_raw.items():
    if not vec or len(vec) != 128:
        continue
    v = np.array(vec, dtype=np.float32)
    n = np.linalg.norm(v)
    if n > 1e-8:
        user_cf[uid] = v / n

print(f"Loading {args.dataset} split={args.split}...")
ds = load_dataset(args.dataset)[args.split]
sessions = list(ds)
print(f"Running {len(sessions)} sessions  w_tt={args.w_tt} w_cf={args.w_cf} "
      f"w_qwen_meta={args.w_qwen_meta} w_clap={args.w_clap} w_bm25={args.w_bm25}")

inference_results = []

for item in tqdm(sessions, desc="Sessions"):
    session_id   = item["session_id"]
    user_id      = item["user_id"]
    goal         = item.get("conversation_goal", {}).get("listener_goal", "")
    culture      = item.get("user_profile", {}).get("preferred_musical_culture", "")
    conversations = item["conversations"]

    user_query   = conversations[-1]["content"]
    turn_number  = conversations[-1]["turn_number"]
    history_convs = conversations[:-1]

    music_history: list[str] = []
    text_history:  list[str] = []
    for turn in history_convs:
        if turn["role"] == "music":
            music_history.append(turn["content"])
        elif turn["role"] in ("user", "assistant"):
            text_history.append(turn["content"])

    seen = set(music_history)
    user_emb = user_cf.get(user_id)

    # Two-tower compact query
    tt_parts = [user_query, goal, culture]
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
    bm25_parts.append(user_query)
    bm25_query = " ".join(p for p in bm25_parts if p)

    # Qwen3/CLAP semantic query
    cleaned = clean_query(user_query) or user_query
    sem_parts = [cleaned, goal, culture]
    for tid in music_history[-args.sem_hist:]:
        na = get_track_name_artist(tid)
        if na:
            sem_parts.append(na)
    semantic_query = " ".join(p for p in sem_parts if p)

    # BM25 recall
    retrieve_k = args.bm25_pool + len(seen) * 3
    raw_tids, raw_scores = retrieve_bm25(bm25_query, topk=retrieve_k)
    filtered = [(t, s) for t, s in zip(raw_tids, raw_scores) if t not in seen][:args.bm25_pool]
    cands = [t for t, _ in filtered]
    bm25_scores = [s for _, s in filtered]

    if not cands:
        inference_results.append({
            "session_id": session_id, "user_id": user_id,
            "turn_number": turn_number,
            "predicted_track_ids": [], "predicted_response": "No recommendation.",
        })
        continue

    if args.bm25_norm:
        max_s = bm25_scores[0] if bm25_scores and bm25_scores[0] > 1e-8 else 1.0
        bm25_rr = {tid: s / max_s for tid, s in zip(cands, bm25_scores)}
    else:
        bm25_rr = {tid: 1.0 / (r + 1) for r, tid in enumerate(cands)}

    # Encode queries
    tt_emb = tt_model.encode(tt_query, normalize_embeddings=True, convert_to_numpy=True)

    qwen_emb = qwen_model.encode(
        QWEN_INSTR + semantic_query,
        normalize_embeddings=True, convert_to_numpy=True,
    )

    with torch.no_grad():
        clap_raw = clap_model.get_text_embedding([semantic_query], use_tensor=True)
    clap_emb = clap_raw[0].cpu().numpy().astype(np.float32)
    clap_emb = clap_emb / max(np.linalg.norm(clap_emb), 1e-8)

    # Precompute full-index scores
    tt_all   = tt_embs        @ tt_emb
    qm_all   = qwen_meta_embs @ qwen_emb
    ql_all   = (qwen_lyrics_embs @ qwen_emb) if args.w_qwen_lyrics > 0 else None
    clap_all = clap_embs      @ clap_emb
    cf_all   = (cf_track_embs @ user_emb) if user_emb is not None else None

    n_cands = len(cands)
    total_arr = np.zeros(n_cands, dtype=np.float32)
    for i, tid in enumerate(cands):
        bm25_s = bm25_rr.get(tid, 0.0)
        idx_tt = tt_id2idx.get(tid)
        idx_qm = qwen_meta_id2idx.get(tid)
        idx_ql = qwen_lyrics_id2idx.get(tid) if ql_all is not None else None
        idx_c  = clap_id2idx.get(tid)
        idx_cf = cf_track_id2idx.get(tid) if user_emb is not None else None

        total_arr[i] = (
            args.w_tt         * (float(tt_all[idx_tt])   if idx_tt is not None else 0.0) +
            args.w_qwen_meta  * (float(qm_all[idx_qm])   if idx_qm is not None else 0.0) +
            args.w_qwen_lyrics * (float(ql_all[idx_ql])  if idx_ql is not None and ql_all is not None else 0.0) +
            args.w_clap       * (float(clap_all[idx_c])  if idx_c  is not None else 0.0) +
            args.w_cf         * (float(cf_all[idx_cf])   if idx_cf is not None and cf_all is not None else 0.0) +
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

Path(args.out_dir).mkdir(parents=True, exist_ok=True)
out_path = Path(args.out_dir) / f"{args.tid}.json"
with open(out_path, "w") as f:
    json.dump(inference_results, f, ensure_ascii=False, indent=2)
print(f"Saved {len(inference_results):,} predictions to {out_path}")
