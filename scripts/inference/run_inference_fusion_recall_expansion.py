"""
Fusion inference with recall expansion.

Same scoring as run_inference_fusion.py, but adds dense candidates to the BM25
pool and gives them a configurable BM25 floor so they're not auto-penalized.

Pool = BM25 top-N
       + TT global top-tt_pool
       + Qwen-meta global top-qwen_pool
       + CF global top-cf_pool   (only for warm users; cold-start gets 0)
       (CLAP / lyrics expansion intentionally not added — audit shows minimal recall lift.)

Each candidate is scored by the full fusion (tt, qwen_meta, qwen_lyrics, clap,
cf, bm25). For candidates not found by BM25 itself, bm25_signal = bm25_missing_floor
instead of 0.0, so dense-only candidates start on equal footing.

Usage:
    python scripts/inference/run_inference_fusion_recall_expansion.py \
        --tid fusion_recall_tt100_floor005 \
        --tt_pool 100 --bm25_missing_floor 0.05 \
        --w_tt 0.32 --w_cf 0.10 --w_qwen_meta 0.40 --w_qwen_lyrics 0.08 \
        --w_clap 0.05 --w_bm25 0.24 --bm25_norm
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
parser.add_argument("--tid",         default="fusion_recall_v1")
parser.add_argument("--out_dir",     default="exp/inference/devset")
parser.add_argument("--topk",        type=int,   default=20)
parser.add_argument("--bm25_pool",   type=int,   default=500)
parser.add_argument("--hist_turns",  type=int,   default=4)
parser.add_argument("--text_turns",  type=int,   default=4)
parser.add_argument("--sem_hist",    type=int,   default=2)
# Signal weights (best-known fusion config defaults)
parser.add_argument("--w_tt",          type=float, default=0.32)
parser.add_argument("--w_cf",          type=float, default=0.10)
parser.add_argument("--w_qwen_meta",   type=float, default=0.40)
parser.add_argument("--w_qwen_attr",   type=float, default=0.0)
parser.add_argument("--w_qwen_lyrics", type=float, default=0.08)
parser.add_argument("--w_clap",        type=float, default=0.05)
parser.add_argument("--w_bm25",        type=float, default=0.24)
parser.add_argument("--w_attrs_hist",  type=float, default=0.0)
parser.add_argument("--attrs_hist_n",  type=int,   default=4)
parser.add_argument("--bm25_norm",     action="store_true", default=True,
                    help="Normalized BM25 score (s/s_max). On by default.")
parser.add_argument("--no_bm25_norm",  dest="bm25_norm", action="store_false")
# Recall expansion
parser.add_argument("--tt_pool",       type=int,   default=100,
                    help="Add TT global top-K to the pool. 0=disabled.")
parser.add_argument("--qwen_pool",     type=int,   default=0,
                    help="Add Qwen-meta global top-K to the pool. 0=disabled.")
parser.add_argument("--cf_pool",       type=int,   default=0,
                    help="Add CF global top-K to the pool (warm users only). 0=disabled.")
parser.add_argument("--bm25_missing_floor", type=float, default=0.05,
                    help="BM25 signal value assigned to candidates not in BM25 pool.")
# Artist + history-NN expansion + source-aware features
parser.add_argument("--artist_expansion", action="store_true", default=False,
                    help="Union tracks of any catalog artist verbatim-mentioned in conversation or in played-track artists.")
parser.add_argument("--artist_cap", type=int, default=50,
                    help="Max tracks added per artist via expansion (deterministic by metadata order).")
parser.add_argument("--last_nn_k", type=int, default=0,
                    help="Per-track TT-NN expansion depth. 0=disabled.")
parser.add_argument("--last_nn_src", type=int, default=2,
                    help="Use last-N played tracks as NN sources.")
parser.add_argument("--w_tt_rank",  type=float, default=0.0,
                    help="Weight on 1/log2(tt_rank+2) for candidates in the TT@K pool.")
parser.add_argument("--w_artist",   type=float, default=0.0,
                    help="Weight on artist_expansion hit flag.")
parser.add_argument("--w_nn",       type=float, default=0.0,
                    help="Weight on 1/log2(nn_rank+2) for last-track-NN candidates.")
parser.add_argument("--w_bm25_origin", type=float, default=0.0,
                    help="Bonus added to BM25-origin candidates (preservation feature).")
parser.add_argument("--write_provenance", default="",
                    help="If set, write per-turn provenance JSONL to this path.")
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
    return re.sub(r"\s+", " ", _FILLER.sub(" ", text)).strip()


print("Loading track metadata...")
meta_ds = load_dataset("talkpl-ai/TalkPlayData-Challenge-Track-Metadata")
all_tracks = concatenate_datasets([meta_ds["all_tracks"], meta_ds["test_tracks"]])
metadata_dict = {row["track_id"]: row for row in all_tracks}

# Artist -> tracks dictionary (lowercased, capped, deterministic order)
artist_to_tids: dict[str, list[str]] = {}
if args.artist_expansion:
    for _tid, _row in metadata_dict.items():
        for _a in (_row.get("artist_name") or []):
            _k = _a.strip().lower()
            if _k:
                artist_to_tids.setdefault(_k, []).append(_tid)
    for _k in artist_to_tids:
        artist_to_tids[_k] = artist_to_tids[_k][:args.artist_cap]
known_artists = sorted(artist_to_tids.keys(), key=len, reverse=True)
print(f"Artist dict: {len(known_artists):,} artists (expansion={'on' if args.artist_expansion else 'off'})")

def find_mentioned_artists(text: str) -> list[tuple[str, str]]:
    """Return [(artist, match_source)] for catalog artists verbatim in text."""
    if not args.artist_expansion or not text:
        return []
    tl = text.lower()
    out = []
    for a in known_artists:
        if a in tl:
            out.append(a)
            tl = tl.replace(a, " " * len(a))
    return out

def get_track_text(tid):
    row = metadata_dict.get(tid, {})
    name   = (row.get("track_name")  or [""])[0]
    artist = (row.get("artist_name") or [""])[0]
    tags   = " ".join(row.get("tag_list") or [])
    return f"{name} {artist} {tags}".strip()

def get_track_name_artist(tid):
    row = metadata_dict.get(tid, {})
    name   = (row.get("track_name")  or [""])[0]
    artist = (row.get("artist_name") or [""])[0]
    return f"{name} {artist}".strip()


print("Loading BM25 index...")
bm25_model = bm25s.BM25.load(BM25_CACHE, load_corpus=False)
with open(f"{BM25_CACHE}/track_ids.json") as f:
    bm25_track_ids = json.load(f)

def retrieve_bm25(query, topk):
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

print("Loading LAION CLAP...")
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

print("Loading dev sessions...")
ds = load_dataset("talkpl-ai/TalkPlayData-Challenge-Dataset")
sessions = list(ds["test"])
if args.sessions > 0:
    sessions = sessions[:args.sessions]

print(
    f"Running {len(sessions)} sessions  "
    f"bm25_pool={args.bm25_pool} tt_pool={args.tt_pool} qwen_pool={args.qwen_pool} cf_pool={args.cf_pool}  "
    f"floor={args.bm25_missing_floor}  bm25_norm={args.bm25_norm}\n"
    f"weights: tt={args.w_tt} cf={args.w_cf} qm={args.w_qwen_meta} ql={args.w_qwen_lyrics} "
    f"clap={args.w_clap} bm25={args.w_bm25} ah={args.w_attrs_hist}"
)

inference_results = []

prov_fh = None
if args.write_provenance:
    Path(args.write_provenance).parent.mkdir(parents=True, exist_ok=True)
    prov_fh = open(args.write_provenance, "w")
    print(f"Writing provenance to {args.write_provenance}")

for item in tqdm(sessions, desc="Sessions"):
    session_id  = item["session_id"]
    user_id     = item["user_id"]
    goal        = item.get("conversation_goal", {}).get("listener_goal", "")
    culture     = item.get("user_profile", {}).get("preferred_musical_culture", "")
    conversations = item["conversations"]

    user_emb = user_cf.get(user_id)

    music_history: list[str] = []
    text_history:  list[str] = []

    for turn in conversations:
        if turn["role"] != "music":
            if turn["role"] in ("user", "assistant"):
                text_history.append(turn["content"])
            continue

        turn_number = turn["turn_number"]
        seen = set(music_history)

        latest_user = text_history[-1] if text_history else ""

        # tt compact query
        tt_parts = [latest_user, goal, culture]
        for tid in music_history[-2:]:
            na = get_track_name_artist(tid)
            if na: tt_parts.append(na)
        tt_query = " ".join(p for p in tt_parts if p)

        # bm25 long query
        bm25_parts = [goal, culture]
        for tid in music_history[-args.hist_turns:]:
            bm25_parts.append(get_track_text(tid))
        bm25_parts.extend(text_history[-args.text_turns:])
        bm25_query = " ".join(p for p in bm25_parts if p)

        # semantic query (cleaned)
        cleaned = clean_query(latest_user) or latest_user
        sem_parts = [cleaned, goal, culture]
        for tid in music_history[-args.sem_hist:]:
            na = get_track_name_artist(tid)
            if na: sem_parts.append(na)
        semantic_query = " ".join(p for p in sem_parts if p)

        # --- BM25 recall ---
        retrieve_k = args.bm25_pool + len(seen) * 3
        raw_tids, raw_scores = retrieve_bm25(bm25_query, topk=retrieve_k)
        filtered = [(t, s) for t, s in zip(raw_tids, raw_scores) if t not in seen][:args.bm25_pool]
        bm25_cands = [t for t, _ in filtered]
        bm25_scores = [s for _, s in filtered]

        # bm25 native rr-or-norm signal (only for tracks BM25 actually retrieved)
        if args.bm25_norm:
            max_s = bm25_scores[0] if bm25_scores and bm25_scores[0] > 1e-8 else 1.0
            bm25_native_sig = {tid: s / max_s for tid, s in zip(bm25_cands, bm25_scores)}
        else:
            bm25_native_sig = {tid: 1.0 / (r + 1) for r, tid in enumerate(bm25_cands)}

        cands = list(bm25_cands)
        cands_set = set(cands)
        sources: dict[str, set] = {tid: {"bm25"} for tid in cands}
        tt_rank_map: dict[str, int] = {}
        artist_src_map: dict[str, str] = {}
        artist_rank_map: dict[str, int] = {}    # min rank within any matched artist's catalog
        nn_src_map: dict[str, str] = {}
        nn_rank_map: dict[str, int] = {}        # min rank across NN source tracks

        # --- Encode queries (needed for expansion + scoring) ---
        tt_emb = tt_model.encode(tt_query, normalize_embeddings=True, convert_to_numpy=True)
        qwen_emb = qwen_model.encode(QWEN_INSTR + semantic_query,
                                     normalize_embeddings=True, convert_to_numpy=True)
        with torch.no_grad():
            clap_raw = clap_model.get_text_embedding([semantic_query], use_tensor=True)
        clap_emb = clap_raw[0].cpu().numpy().astype(np.float32)
        clap_emb = clap_emb / max(np.linalg.norm(clap_emb), 1e-8)

        # full-index dot products (used for both expansion and scoring)
        tt_all   = tt_embs        @ tt_emb
        qm_all   = qwen_meta_embs @ qwen_emb
        ql_all   = (qwen_lyrics_embs @ qwen_emb) if args.w_qwen_lyrics > 0 else None
        clap_all = clap_embs      @ clap_emb
        cf_all   = (cf_track_embs @ user_emb) if user_emb is not None else None

        # --- Recall expansion ---
        def add_topk(scores_arr, ids_list, k, src_label):
            if k <= 0 or scores_arr is None:
                return
            top = np.argpartition(scores_arr, -k)[-k:]
            top = top[np.argsort(scores_arr[top])[::-1]]
            for rank, idx in enumerate(top):
                tid = ids_list[int(idx)]
                if tid in seen:
                    continue
                if tid not in cands_set:
                    cands.append(tid); cands_set.add(tid)
                    sources[tid] = set()
                sources[tid].add(src_label)
                if src_label == "tt" and tid not in tt_rank_map:
                    tt_rank_map[tid] = rank

        add_topk(tt_all,   tt_ids,         args.tt_pool, "tt")
        add_topk(qm_all,   qwen_meta_ids,  args.qwen_pool, "qm")
        if cf_all is not None:
            add_topk(cf_all, cf_track_ids,  args.cf_pool, "cf")

        # Artist expansion
        if args.artist_expansion:
            mentioned: dict[str, str] = {}  # artist -> match_source
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
                    artist_src_map.setdefault(tid, src)
                    if tid not in artist_rank_map or rank < artist_rank_map[tid]:
                        artist_rank_map[tid] = rank

        # Last-track NN expansion (TT space)
        if args.last_nn_k > 0 and music_history:
            for src_tid in music_history[-args.last_nn_src:]:
                src_idx = tt_id2idx.get(src_tid)
                if src_idx is None:
                    continue
                sims = tt_embs @ tt_embs[src_idx]
                sims[src_idx] = -1e9
                top = np.argpartition(-sims, args.last_nn_k)[:args.last_nn_k]
                top = top[np.argsort(-sims[top])]
                for rank, idx in enumerate(top):
                    tid = tt_ids[int(idx)]
                    if tid in seen:
                        continue
                    if tid not in cands_set:
                        cands.append(tid); cands_set.add(tid); sources[tid] = set()
                    sources[tid].add("nn")
                    nn_src_map.setdefault(tid, src_tid)
                    if tid not in nn_rank_map or rank < nn_rank_map[tid]:
                        nn_rank_map[tid] = rank

        if not cands:
            inference_results.append({
                "session_id": session_id, "user_id": user_id,
                "turn_number": turn_number,
                "predicted_track_ids": [], "predicted_response": "No recommendation.",
            })
            music_history.append(turn["content"])
            continue

        # --- attrs-history (optional) ---
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
        ah_all = (qwen_attr_embs @ attrs_hist_emb) if attrs_hist_emb is not None else None

        # --- Score every candidate (BM25 floor for non-native) ---
        n_cands = len(cands)
        total_arr = np.zeros(n_cands, dtype=np.float32)
        for i, tid in enumerate(cands):
            bm25_s = bm25_native_sig.get(tid, args.bm25_missing_floor)
            idx_tt = tt_id2idx.get(tid)
            idx_qm = qwen_meta_id2idx.get(tid)
            idx_ql = qwen_lyrics_id2idx.get(tid) if ql_all is not None else None
            idx_c  = clap_id2idx.get(tid)
            idx_cf = cf_track_id2idx.get(tid) if cf_all is not None else None
            idx_ah = qwen_attr_id2idx.get(tid) if ah_all is not None else None

            tt_rank = tt_rank_map.get(tid)
            tt_rank_sig = (1.0 / np.log2(tt_rank + 2.0)) if tt_rank is not None else 0.0
            artist_rank = artist_rank_map.get(tid)
            artist_sig  = (1.0 / np.log2(artist_rank + 2.0)) if artist_rank is not None else 0.0
            nn_rank = nn_rank_map.get(tid)
            nn_sig  = (1.0 / np.log2(nn_rank + 2.0)) if nn_rank is not None else 0.0
            bm25_origin_sig = 1.0 if "bm25" in sources.get(tid, ()) else 0.0

            total_arr[i] = (
                args.w_tt          * (float(tt_all[idx_tt])   if idx_tt is not None else 0.0) +
                args.w_qwen_meta   * (float(qm_all[idx_qm])   if idx_qm is not None else 0.0) +
                args.w_qwen_lyrics * (float(ql_all[idx_ql])   if idx_ql is not None and ql_all is not None else 0.0) +
                args.w_clap        * (float(clap_all[idx_c])  if idx_c  is not None else 0.0) +
                args.w_cf          * (float(cf_all[idx_cf])   if idx_cf is not None and cf_all is not None else 0.0) +
                args.w_attrs_hist  * (float(ah_all[idx_ah])   if idx_ah is not None and ah_all is not None else 0.0) +
                args.w_bm25        * bm25_s +
                args.w_tt_rank     * tt_rank_sig +
                args.w_artist      * artist_sig +
                args.w_nn          * nn_sig +
                args.w_bm25_origin * bm25_origin_sig
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

        if prov_fh is not None:
            gold = turn["content"]
            bm25_rank = bm25_cands.index(gold) if gold in bm25_cands else None
            tt_rank_gold = tt_rank_map.get(gold)
            srcs = sorted(sources.get(gold, ()))
            # final rank (1-based) of gold in scored pool
            order = np.argsort(total_arr)[::-1]
            try:
                gold_idx_in_cands = cands.index(gold)
                final_rank_gold = int(np.where(order == gold_idx_in_cands)[0][0]) + 1
                final_score_gold = float(total_arr[gold_idx_in_cands])
            except ValueError:
                final_rank_gold = None
                final_score_gold = None
            prov_fh.write(json.dumps({
                "session_id": session_id,
                "turn_number": turn_number,
                "user_id": user_id,
                "gold": gold,
                "found_in_pool": gold in cands_set,
                "found_by": srcs,
                "pool_size": len(cands),
                "bm25_rank": bm25_rank,
                "tt_rank": tt_rank_gold,
                "artist_match_source": artist_src_map.get(gold),
                "nn_source_track": nn_src_map.get(gold),
                "final_rank": final_rank_gold,
                "final_score": final_score_gold,
                "top20_predicted": predicted_track_ids[:20],
            }) + "\n")

        music_history.append(turn["content"])

Path(args.out_dir).mkdir(parents=True, exist_ok=True)
out_path = Path(args.out_dir) / f"{args.tid}.json"
with open(out_path, "w") as f:
    json.dump(inference_results, f, ensure_ascii=False, indent=2)
print(f"Saved {len(inference_results):,} predictions to {out_path}")
if prov_fh is not None:
    prov_fh.close()
