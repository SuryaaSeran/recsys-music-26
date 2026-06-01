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
import math
import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import argparse
import json
import re
import numpy as np
from pathlib import Path

# Import lightgbm BEFORE torch -- on macOS, importing it after torch causes a
# silent OpenMP-related abort when other heavy native libs (e.g. CLAP via torch
# / transformers) are loaded.
try:
    import lightgbm as _lgb_preload  # noqa: F401
except ImportError:
    _lgb_preload = None

import bm25s
import torch
from datasets import load_dataset, concatenate_datasets
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument("--tt_model",    default="models/twotower_v3/final")
parser.add_argument("--tt_index",    default="cache/twotower_v3")
parser.add_argument("--tt_query_prefix", default="",
                    help="Prefix prepended to the TT query before encoding (e.g. Qwen3 'Instruct: ...\\nQuery: ').")
parser.add_argument("--sessions",    type=int,   default=0)
parser.add_argument("--session_ids_file", default="",
                    help="JSON file with a list of session_ids (or object with a key matching "
                         "'golden_200' or first list value). Only those sessions are processed.")
parser.add_argument("--split",       default="test",
                    help="Which dataset split to run on (test / train / etc).")
parser.add_argument("--dataset",     default="talkpl-ai/TalkPlayData-Challenge-Dataset",
                    help="HF dataset path. Use talkpl-ai/TalkPlayData-Challenge-Blind-A for Blind A.")
parser.add_argument("--blind_mode",  action="store_true",
                    help="Predict only the final music turn per session (turn_number = conversations[-1].turn_number). Use with --dataset talkpl-ai/TalkPlayData-Challenge-Blind-A.")
parser.add_argument("--shuffle_seed", type=int, default=-1,
                    help=">=0: shuffle sessions before taking --sessions slice (deterministic).")
parser.add_argument("--session_offset", type=int, default=0,
                    help="Skip first N sessions after shuffle before taking --sessions slice. "
                         "Use with --sessions to split a large dump across parallel workers: "
                         "worker 0: --sessions 3000 --session_offset 0, "
                         "worker 1: --sessions 3000 --session_offset 3000.")
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
parser.add_argument("--ql_pool",       type=int,   default=0,
                    help="Add Qwen-lyrics global top-K to the pool. 0=disabled. "
                         "Rescues tracks described by lyrics/mood that BM25 misses (~6.7%% of BM25 misses at top-500).")
parser.add_argument("--bm25_sharp_pool", type=int, default=0,
                    help="Add a second BM25 retrieval using only latest_user+goal (no track history) top-K. "
                         "Targets mood/vibe turns where history text dilutes query keywords. 0=disabled.")
parser.add_argument("--bm25_entity_pool", type=int, default=0,
                    help="Add a focused BM25 retrieval using only extracted catalog entities (artist names "
                         "and quoted strings) from the latest user message. Targets exact-match sessions "
                         "where the user names a specific track/album/artist. 0=disabled.")
parser.add_argument("--cf_pool",       type=int,   default=0,
                    help="Add CF global top-K to the pool (warm users only). 0=disabled.")
parser.add_argument("--adaptive_pool_threshold", type=float, default=0.0,
                    help="When BM25 top normalized score >= this value, suppress CF, "
                         "cooccurrence, and session-mean for this turn (exact-match mode). "
                         "Addresses Phase D regression on specific-goal sessions. 0=disabled.")
parser.add_argument("--bm25_missing_floor", type=float, default=0.05,
                    help="BM25 signal value assigned to candidates not in BM25 pool.")
# Artist + history-NN expansion + source-aware features
parser.add_argument("--artist_expansion", action="store_true", default=False,
                    help="Union tracks of any catalog artist verbatim-mentioned in conversation or in played-track artists.")
parser.add_argument("--artist_cap", type=int, default=50,
                    help="Max tracks added per artist via expansion (deterministic by metadata order).")
parser.add_argument("--last_nn_k", type=int, default=0,
                    help="Per-track TT-NN expansion depth (uniform across last_nn_src). 0=disabled.")
parser.add_argument("--last_nn_src", type=int, default=2,
                    help="Use last-N played tracks as NN sources.")
parser.add_argument("--session_nn_ks", default="",
                    help="Comma list of per-position NN depths, newest first (overrides --last_nn_k). "
                         "Example: '300,200,100' = top-300 NN of last track, top-200 of prev2, top-100 of prev3.")
parser.add_argument("--session_mean_k", type=int, default=0,
                    help="Add top-K NN of mean-session vector (TT mean of last --session_mean_n tracks).")
parser.add_argument("--session_mean_n", type=int, default=4,
                    help="Number of recent tracks averaged for mean-session vector.")
parser.add_argument("--cooccur_table", default="",
                    help="Path to co-occurrence .npz built by scripts/train/build_cooccur_table.py.")
parser.add_argument("--cooccur_ks", default="",
                    help="Comma list of per-position co-occur depths, newest first. Example: '300,150,50'.")
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
parser.add_argument("--write_features", default="",
                    help="If set, write per-candidate feature rows to this NPZ path for LTR training.")
parser.add_argument("--soft_labels", action="store_true",
                    help="If set, use graded labels: 2=gold, 1=same-artist-as-gold, 0=other. "
                         "Requires label_gain=[0,1,3] in the LTR trainer. Default: binary 0/1.")
parser.add_argument("--progress_aware", action="store_true",
                    help="Use goal_progress_assessment from dataset. Gold tracks rated "
                         "DOES_NOT_MOVE_TOWARD_GOAL get label=0 (treated as negatives). "
                         "Turns with no gold in pool are always included regardless.")
parser.add_argument("--skip_no_progress", action="store_true",
                    help="Drop turns where gold is DOES_NOT_MOVE_TOWARD_GOAL entirely "
                         "from the feature dump (no rows emitted). More aggressive than "
                         "--progress_aware which keeps the turn but zeros the gold label.")
parser.add_argument("--use_goal_progress", action="store_true",
                    help="H1+H3: use goal_progress_assessments at inference to filter "
                         "rejected tracks from retrieval seeds (H1) and optionally modulate "
                         "the goal query slot (H3). Reads goal_progress_assessments from dataset.")
parser.add_argument("--infer_progress_labels", action="store_true",
                    help="If goal_progress_assessments are absent or incomplete, infer "
                         "MOVES_TOWARD_GOAL / DOES_NOT_MOVE_TOWARD_GOAL labels from user "
                         "follow-up messages (rule-based). Enables H1+H3 on blind test sessions "
                         "that do not carry gold progress labels. Implies --use_goal_progress.")
parser.add_argument("--rejection_drop_threshold", type=int, default=0,
                    help="H3a: drop the static goal string from all query types when "
                         "n_consecutive_rejections >= this value. 0=disabled.")
parser.add_argument("--goal_substitute_positive", action="store_true",
                    help="H3b: when at least one prior MOVES_TOWARD_GOAL track exists, "
                         "substitute its name+artist into the goal slot of all query types. "
                         "Takes precedence over --rejection_drop_threshold when a positive exists.")
parser.add_argument("--ltr_model", default="",
                    help="If set, score with this LightGBM booster instead of the linear fusion.")
parser.add_argument("--ltr_neural", default="",
                    help="If set, score with this PyTorch MLP directory (from train_ltr_neural.py) "
                         "instead of the linear fusion. Mutually exclusive with --ltr_model.")
# TT query richness (set >0 to include extra context matching v8 training format)
parser.add_argument("--tt_text_turns", type=int, default=0,
                    help="Prior text turns (user+assistant, before latest_user) to append to the TT query. "
                         "0=v6 compact. Set 3 for v8 nomic-embed.")
parser.add_argument("--tt_hist_turns", type=int, default=2,
                    help="Number of recently played tracks to append to the TT query. "
                         "2=v6 compact (name/artist only). Set 4 for v8 nomic-embed (full track text).")
args = parser.parse_args()

if args.ltr_model and args.ltr_neural:
    raise ValueError("--ltr_model and --ltr_neural are mutually exclusive.")

BM25_CACHE = "cache/bm25/track_metadata"

_FILLER = re.compile(
    r"\b(can you|could you|would you|please|i want|i'd like|i would like|"
    r"i need|i'm looking for|i am looking for|something that(?:'s| is)|something|"
    r"recommend(?:ation)?|suggest(?:ion)?|play(?: me)?|find me|show me|give me|"
    r"how about|what about|i feel like(?: listening to)?|i(?:'m| am) in the mood for|"
    r"do you have|do you know)\b",
    re.IGNORECASE,
)
# User intent detection patterns (proxy for goal_progress_assessment)
# Rule-based progress label inference (Track 2a, plan/09_generalization_routing.md)
# Classifies user follow-up messages to infer MOVES/DOES_NOT_MOVE without gold labels.
_PROGRESS_POSITIVE_RE = re.compile(
    r"\b(yes\b|yeah|yep|perfect|exactly|that'?s (it|exactly|right|perfect)|love (it|this)|"
    r"great (choice|pick|selection)?|awesome|fantastic|excellent|found (it|what)|"
    r"keep (going|them coming|it up)|more like (this|that)|that'?s the (one|song|track)|"
    r"this is (perfect|it|exactly)|that is (perfect|it|exactly))\b",
    re.IGNORECASE,
)
_PROGRESS_NEGATIVE_RE = re.compile(
    r"\b(not (quite|really|this|that|it)|something (different|else|more|other)|"
    r"try (something|another|a different)|rather (have|get|hear)|instead|"
    r"don'?t (want|like|think)|that'?s not|not what i|too (slow|fast|heavy|light|pop|dark|upbeat|sad|old|new)|"
    r"i was (thinking|hoping|looking for)|i'?m (looking for|hoping for|thinking more))\b",
    re.IGNORECASE,
)

def _infer_progress_label(user_followup: str) -> str:
    """Classify a user follow-up message as MOVES or DOES_NOT_MOVE (or empty if unclear)."""
    if not user_followup:
        return ""
    if _PROGRESS_POSITIVE_RE.search(user_followup):
        return "MOVES_TOWARD_GOAL"
    if _PROGRESS_NEGATIVE_RE.search(user_followup):
        return "DOES_NOT_MOVE_TOWARD_GOAL"
    return ""

_NEGATION = re.compile(
    r"\b(not what|not quite|not really|not exactly|doesn't|don't|didn't|"
    r"isn't|wasn't|that's not|but i(?:'m| am)|but i want|but i(?:'d| would)|"
    r"too much|too little|wrong|different|instead|rather|no[,.]|nope|"
    r"not the|without|less|more of a|looking for something)\b",
    re.IGNORECASE,
)
_FOLLOWUP = re.compile(
    r"\b(more like|more of|another|similar|same|again|keep|continue|"
    r"along those lines|in that vein|that direction|like that|"
    r"the same|one more|next|also|too)\b",
    re.IGNORECASE,
)

def clean_query(text: str) -> str:
    return re.sub(r"\s+", " ", _FILLER.sub(" ", text)).strip()


print("Loading track metadata...")
meta_ds = load_dataset("talkpl-ai/TalkPlayData-Challenge-Track-Metadata")
all_tracks = concatenate_datasets([meta_ds["all_tracks"], meta_ds["test_tracks"]])
metadata_dict = {row["track_id"]: row for row in all_tracks}

# Precompute popularity percentile lookup (rank-percentile, 0-1)
_pop_vals = []
for _tid, _row in metadata_dict.items():
    _p = float(_row.get("popularity") or 0.0)
    _pop_vals.append((_p, _tid))
_pop_vals.sort(key=lambda x: x[0])
popularity_pctile: dict[str, float] = {}
_n_tracks = len(_pop_vals)
for _rank, (_p, _tid) in enumerate(_pop_vals):
    popularity_pctile[_tid] = _rank / max(_n_tracks - 1, 1)
del _pop_vals
print(f"Popularity percentile lookup: {len(popularity_pctile):,} tracks")

# Goal category integer encoding — built after sessions load (see below sessions assignment)

# Artist -> tracks dictionary (lowercased, capped, deterministic order)
artist_to_tids: dict[str, list[str]] = {}
if args.artist_expansion:
    # Sort each artist's catalog by popularity desc so rank 0 = most popular track.
    # Falls back to 0.0 if popularity is missing.
    artist_buckets: dict[str, list[tuple[float, str]]] = {}
    for _tid, _row in metadata_dict.items():
        _pop = float(_row.get("popularity") or 0.0)
        for _a in (_row.get("artist_name") or []):
            _k = _a.strip().lower()
            if _k:
                artist_buckets.setdefault(_k, []).append((_pop, _tid))
    for _k, _bucket in artist_buckets.items():
        _bucket.sort(key=lambda x: -x[0])
        artist_to_tids[_k] = [t for _, t in _bucket[:args.artist_cap]]
known_artists = sorted(artist_to_tids.keys(), key=len, reverse=True)
print(f"Artist dict: {len(known_artists):,} artists (expansion={'on' if args.artist_expansion else 'off'})")

# For entity extraction: always build a flat artist name set (>=4 chars to reduce false positives).
# This is independent of --artist_expansion so entity BM25 works without full artist expansion.
_entity_artist_names: list[str] = sorted(
    {_a.strip().lower() for _tid, _row in metadata_dict.items()
     for _a in (_row.get("artist_name") or []) if _a and len(_a.strip()) >= 4},
    key=len, reverse=True
)
_QUOTED_RE = re.compile(r'[“”‘’"\']([\w][^“”‘’"\']{2,59})[“”‘’"\']')

def extract_query_entities(text: str) -> str:
    """Extract catalog artist names and quoted strings from query text.
    Returns a focused query string for entity-targeted BM25 retrieval."""
    if not text:
        return ""
    parts: list[str] = []
    tl = text.lower()
    # Artist names (catalog match, longest first to avoid partial matches)
    for a in _entity_artist_names:
        if a in tl:
            parts.append(a)
            tl = tl.replace(a, " " * len(a))
    # Quoted strings (likely track/album names)
    for m in _QUOTED_RE.finditer(text):
        q = m.group(1).strip()
        if len(q) > 2:
            parts.append(q)
    return " ".join(parts)

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

# Co-occurrence table (optional)
cooccur_track_ids = None
cooccur_tid2idx: dict[str, int] = {}
cooccur_neigh_ids = None
cooccur_neigh_w = None
if args.cooccur_table:
    print(f"Loading co-occurrence table: {args.cooccur_table}")
    _z = np.load(args.cooccur_table, allow_pickle=True)
    cooccur_track_ids = _z["track_ids"]
    cooccur_neigh_ids = _z["neigh_ids"]
    cooccur_neigh_w   = _z["neigh_w"]
    cooccur_tid2idx = {str(t): i for i, t in enumerate(cooccur_track_ids.tolist())}
    nz = (cooccur_neigh_ids[:, 0] >= 0).sum()
    print(f"  table shape={cooccur_neigh_ids.shape}  rows-with-neighbours={nz}")

# Parse comma-list flags
def _parse_ks(s: str) -> list[int]:
    return [int(x) for x in s.split(",") if x.strip()] if s else []

session_nn_ks_list = _parse_ks(args.session_nn_ks)
cooccur_ks_list    = _parse_ks(args.cooccur_ks)

print(f"Loading dataset: {args.dataset} [{args.split}]")
ds = load_dataset(args.dataset)
sessions = list(ds[args.split])
if args.shuffle_seed >= 0:
    import random as _r
    _r.Random(args.shuffle_seed).shuffle(sessions)
if args.session_ids_file:
    import json as _sjson
    with open(args.session_ids_file) as _sf:
        _sd = _sjson.load(_sf)
    if isinstance(_sd, dict):
        # e.g. GOLDEN_HOLDOUT_SESSIONS.json with {"golden_200": [...], "eval_800": [...]}
        _sid_set = set(next(v for v in _sd.values() if isinstance(v, list)))
    else:
        _sid_set = set(_sd)
    sessions = [s for s in sessions if s["session_id"] in _sid_set]
    print(f"  session_ids_file: keeping {len(sessions)} sessions matching {args.session_ids_file}")
elif args.sessions > 0:
    sessions = sessions[args.session_offset: args.session_offset + args.sessions]
print(f"Using split={args.split}  n_sessions={len(sessions)}  shuffle_seed={args.shuffle_seed}")

# Goal category integer encoding — sorted for deterministic codes across any session ordering.
# 0 is reserved for unknown/missing.
_all_goal_cats = sorted({
    item.get("conversation_goal", {}).get("category", "")
    for item in sessions
} - {""})
GOAL_CATEGORY_MAP: dict[str, int] = {cat: i + 1 for i, cat in enumerate(_all_goal_cats)}

print(
    f"Running {len(sessions)} sessions  "
    f"bm25_pool={args.bm25_pool} tt_pool={args.tt_pool} qwen_pool={args.qwen_pool} cf_pool={args.cf_pool}  "
    f"floor={args.bm25_missing_floor}  bm25_norm={args.bm25_norm}\n"
    f"weights: tt={args.w_tt} cf={args.w_cf} qm={args.w_qwen_meta} ql={args.w_qwen_lyrics} "
    f"clap={args.w_clap} bm25={args.w_bm25} ah={args.w_attrs_hist}"
)

FEATURE_COLS = [
    "tt_cos", "qm_cos", "ql_cos", "clap_cos", "cf_cos",
    "bm25_signal", "tt_rank_sig", "artist_sig", "nn_sig",
    "bm25_origin", "artist_origin", "tt_origin", "nn_origin",
    "cold_user", "pool_size",
    # Stage 9 additions
    "qm_origin", "qm_rank_sig", "qm_only",
    "nn_source_count", "mean_nn_origin", "mean_nn_rank_sig",
    "dist_to_last", "dist_to_recent_mean",
    "collab_origin", "collab_rank_sig", "collab_score", "collab_source_count",
    # Phase B additions
    "popularity", "track_year",
    # Phase D: feature engineering v2
    "n_sources",            # count of retrieval sources that found this candidate
    "turn_number",          # position in conversation (1-indexed)
    "history_len",          # number of tracks played so far in this session
    "popularity_pctile",    # rank-percentile of popularity across catalog (0-1)
    "years_since_release",  # 2026 - release_year, 0 if missing
    "tag_overlap_count",    # number of candidate tags appearing in the BM25 query
    "query_len_tokens",     # word count of latest user message (query specificity proxy)
    "cf_dist_to_last",      # cosine to last played track in CF space (0 for cold users)
    "cf_dist_to_recent_mean",  # cosine to mean of recent tracks in CF space (0 for cold)
    "goal_category",        # integer-encoded conversation goal category
    # Phase D2: user intent signals (proxy for goal progress)
    "user_has_negation",    # 1.0 if latest user msg contains correction/negation words
    "user_has_followup",    # 1.0 if latest user msg is a continuation ("more", "another", "similar")
    "query_track_tag_sim",  # fraction of gold candidate's tags that appear in user query (per-candidate)
    # Phase F: turn-position-normalised source agreement + exact-match signal (indices 42-44)
    "n_sources_norm",         # n_sources / (1 + turn_number) — scale-invariant across positions
    "log1p_n_sources",        # log1p(n_sources) — dampens outsized n_sources dominance
    "bm25_top1",              # 1.0 if this candidate is the #1 BM25 result; rewards exact-match hits
    # Phase E: H2 history-based features (require --use_goal_progress; 0 otherwise; indices 44-47)
    "sim_to_pos_hist_mean",   # TT cosine between candidate and mean of MOVES_TOWARD prior tracks
    "sim_to_neg_hist_mean",   # TT cosine between candidate and mean of DOES_NOT_MOVE prior tracks
    "artist_in_rejected_set", # 1.0 if candidate artist appears in any prior DOES_NOT_MOVE track
    "n_rejected_in_history",  # count of DOES_NOT_MOVE turns so far, clipped at 10 then /10
]

ltr_booster = None
if args.ltr_model:
    if _lgb_preload is None:
        raise RuntimeError("--ltr_model requires `lightgbm` to be installed.")
    ltr_booster = _lgb_preload.Booster(model_file=args.ltr_model)
    n_booster_feats = ltr_booster.num_feature()
    print(f"Loaded LTR booster: {args.ltr_model}  ({n_booster_feats} features, FEATURE_COLS has {len(FEATURE_COLS)})")
    _LGB_POLY_PAIRS = [
        ("tt_cos",            "bm25_signal",    "tt_x_bm25"),
        ("tt_rank_sig",       "bm25_origin",    "ttrank_x_bm25orig"),
        ("tt_cos",            "tt_rank_sig",    "tt_x_ttrank"),
        ("qm_cos",            "bm25_signal",    "qm_x_bm25"),
        ("artist_sig",        "artist_origin",  "artist_x_orig"),
        ("nn_sig",            "tt_cos",         "nn_x_tt"),
        ("collab_rank_sig",   "collab_score",   "collab_rank_x_score"),
        ("popularity",        "tt_cos",         "pop_x_tt"),
        ("popularity",        "bm25_signal",    "pop_x_bm25"),
        ("tag_overlap_count", "bm25_signal",    "tagoverlap_x_bm25"),
        ("tag_overlap_count", "tt_cos",         "tagoverlap_x_tt"),
        ("cf_dist_to_last",   "cf_cos",         "cfdist_x_cfcos"),
        ("n_sources",         "tt_cos",         "nsrc_x_tt"),
        ("popularity_pctile", "tt_cos",         "poppctile_x_tt"),
    ]
    _lgb_use_poly = n_booster_feats > len(FEATURE_COLS)
    assert n_booster_feats <= len(FEATURE_COLS) + len(_LGB_POLY_PAIRS), \
        f"booster expects {n_booster_feats} features but max expandable is {len(FEATURE_COLS) + len(_LGB_POLY_PAIRS)}"

# ── Neural LTR model (PyTorch MLP) ───────────────────────────────────────────
_ltr_neural_model  = None
_ltr_neural_scaler = None
_ltr_neural_meta   = None
if args.ltr_neural:
    import json as _json
    import torch as _torch
    import torch.nn as _nn

    _nd = Path(args.ltr_neural)
    with open(_nd / "meta.json")   as _f: _ltr_neural_meta   = _json.load(_f)
    with open(_nd / "scaler.json") as _f: _ltr_neural_scaler = _json.load(_f)

    _hidden = _ltr_neural_meta["hidden"]
    _nf     = _ltr_neural_meta["n_feats"]
    _do     = _ltr_neural_meta.get("dropout", 0.1)

    class _MLP(_nn.Module):
        def __init__(self, n, h, d):
            super().__init__()
            layers, i = [], n
            for o in h:
                layers += [_nn.Linear(i, o), _nn.ReLU(), _nn.Dropout(d)]
                i = o
            layers.append(_nn.Linear(i, 1))
            self.net = _nn.Sequential(*layers)
        def forward(self, x): return self.net(x).squeeze(-1)

    _ltr_neural_model = _MLP(_nf, _hidden, _do)
    _ltr_neural_model.load_state_dict(
        _torch.load(_nd / "model.pt", map_location="cpu", weights_only=True)
    )
    _ltr_neural_model.eval()
    _neural_feat_cols = _ltr_neural_scaler["feature_cols"]
    _neural_mean = np.array(_ltr_neural_scaler["mean"], dtype=np.float32)
    _neural_std  = np.array(_ltr_neural_scaler["std"],  dtype=np.float32)

    # interaction pairs (must match train_ltr_neural.py)
    _NEURAL_PAIRS = [
        ("tt_cos",      "bm25_signal",     "tt_x_bm25"),
        ("tt_rank_sig", "bm25_origin",     "ttrank_x_bm25orig"),
        ("tt_cos",      "tt_rank_sig",     "tt_x_ttrank"),
        ("qm_cos",      "bm25_signal",     "qm_x_bm25"),
        ("artist_sig",  "artist_origin",   "artist_x_orig"),
        ("nn_sig",      "tt_cos",          "nn_x_tt"),
        ("collab_rank_sig", "collab_score","collab_rank_x_score"),
        ("popularity",  "tt_cos",          "pop_x_tt"),
        ("popularity",  "bm25_signal",     "pop_x_bm25"),
    ]
    _neural_use_poly = _ltr_neural_meta.get("poly_feats", False)

    print(f"Loaded neural LTR: {args.ltr_neural}  "
          f"({_nf} feats, poly={_neural_use_poly}, "
          f"CV ndcg@20={_ltr_neural_meta['cv_ndcg20_mean']:.4f})")

inference_results = []

prov_fh = None
if args.write_provenance:
    Path(args.write_provenance).parent.mkdir(parents=True, exist_ok=True)
    prov_fh = open(args.write_provenance, "w")
    print(f"Writing provenance to {args.write_provenance}")

# Progress-aware counters
_progress_total_turns = 0
_progress_rejected_turns = 0
_progress_skipped_turns = 0

# LTR feature dump buffers. X is written directly to a temp binary file to avoid
# accumulating GBs of arrays in RAM — peak memory = one turn's features at a time.
feat_chunks: list = []   # kept empty; used as sentinel only
label_chunks: list = []
group_chunks: list = []   # turn-index per row
turn_meta: list = []      # one per turn: {session_id, turn_number, gold}
turn_counter = [0]        # mutable counter for closures
_X_tmpfile = None
_X_tmppath = None
_X_n_rows = [0]
_X_n_feats = [0]
if args.write_features:
    import tempfile as _tempfile
    _X_tmppath = Path(args.write_features).with_suffix(".X.tmp")
    _X_tmpfile = open(_X_tmppath, "wb")
total_music_turns = 0
found_in_pool_count = 0

for item in tqdm(sessions, desc="Sessions"):
    session_id  = item["session_id"]
    user_id     = item["user_id"]
    goal        = item.get("conversation_goal", {}).get("listener_goal", "")
    culture     = item.get("user_profile", {}).get("preferred_musical_culture", "")
    _goal_cat   = item.get("conversation_goal", {}).get("category", "")
    goal_category_int = float(GOAL_CATEGORY_MAP.get(_goal_cat, 0))
    # Goal progress assessment per turn (for --progress_aware / --skip_no_progress / --use_goal_progress)
    _progress_by_turn: dict[int, str] = {}
    _use_any_progress = args.progress_aware or args.skip_no_progress or args.use_goal_progress or args.infer_progress_labels
    if _use_any_progress:
        for _a in (item.get("goal_progress_assessments") or []):
            _progress_by_turn[_a["turn_number"]] = _a.get("goal_progress_assessment", "")
    # --infer_progress_labels: fill in missing labels from user follow-up text.
    # Also implies --use_goal_progress so H1/H3 fire automatically.
    if args.infer_progress_labels:
        _convs_raw = item["conversations"]
        _last_music_tnum: int | None = None
        for _t in _convs_raw:
            if _t["role"] == "music" and _t.get("content"):
                _last_music_tnum = _t["turn_number"]
            elif _t["role"] == "user" and _last_music_tnum is not None:
                if _progress_by_turn.get(_last_music_tnum) is None or _progress_by_turn.get(_last_music_tnum) == "":
                    _inferred = _infer_progress_label(_t["content"])
                    if _inferred:
                        _progress_by_turn[_last_music_tnum] = _inferred
                _last_music_tnum = None  # reset: next user message is a new query
    conversations = item["conversations"]

    if args.blind_mode:
        # In blind mode there is no "music" turn to predict for in the input.
        # Move all existing music turns to the front (so they populate
        # music_history first), keep user/assistant turns in order so
        # text_history is built correctly, then append a synthetic music
        # turn at the end with the trigger turn_number so the existing
        # loop emits exactly one prediction per session.
        last_user_turn_number = conversations[-1]["turn_number"]
        music_turns = [t for t in conversations if t["role"] == "music"]
        text_turns  = [t for t in conversations if t["role"] != "music"]
        conversations = music_turns + text_turns + [{
            "role": "music",
            "turn_number": last_user_turn_number,
            "content": "",  # no gold; we are predicting it
        }]

    user_emb = user_cf.get(user_id)

    music_history: list[str] = []
    music_history_labels: list[str] = []  # parallel to music_history; assessment per turn
    text_history:  list[str] = []

    for turn in conversations:
        if turn["role"] != "music":
            if turn["role"] in ("user", "assistant"):
                text_history.append(turn["content"])
            continue

        # In blind_mode, real (historic) music turns carry gold content; they
        # represent the user's past plays, not turns we need to predict for.
        # Add them to music_history and skip the prediction logic. Only the
        # synthetic trailing music turn (content == "") triggers a prediction.
        if args.blind_mode and turn["content"]:
            music_history.append(turn["content"])
            music_history_labels.append(_progress_by_turn.get(turn["turn_number"], ""))
            continue

        turn_number = turn["turn_number"]
        seen = set(music_history)

        latest_user = text_history[-1] if text_history else ""

        # H1: positive_history — seed expansion only from non-rejected prior tracks.
        # Falls back to raw music_history when no positive exists (e.g. turn 1).
        _use_progress_at_inference = args.use_goal_progress or args.infer_progress_labels
        if _use_progress_at_inference:
            _pos_hist = [t for t, l in zip(music_history, music_history_labels)
                         if l != "DOES_NOT_MOVE_TOWARD_GOAL"] or music_history
        else:
            _pos_hist = music_history

        # H3: goal-slot modulation
        _goal_for_query = goal
        if _use_progress_at_inference and music_history_labels:
            # Most recent MOVES_TOWARD_GOAL track (for H3b substitution)
            _most_recent_pos_tid = next(
                (t for t, l in zip(reversed(music_history), reversed(music_history_labels))
                 if l == "MOVES_TOWARD_GOAL"), None)
            # Consecutive rejections from tail (for H3a drop)
            _n_consec_rej = 0
            for _l in reversed(music_history_labels):
                if _l == "DOES_NOT_MOVE_TOWARD_GOAL":
                    _n_consec_rej += 1
                else:
                    break
            if args.goal_substitute_positive and _most_recent_pos_tid:
                _prow = metadata_dict.get(_most_recent_pos_tid, {})
                _pname = (_prow.get("track_name") or [""])[0]
                _partist = (_prow.get("artist_name") or [""])[0]
                _goal_for_query = f"{_pname} by {_partist}" if _pname else goal
            elif args.rejection_drop_threshold > 0 and _n_consec_rej >= args.rejection_drop_threshold:
                _goal_for_query = ""

        # tt query — compact (v6) or rich (v8+)
        tt_parts = [latest_user, _goal_for_query, culture]
        if args.tt_text_turns > 0:
            # Prior text turns before latest_user (user+assistant interleaved)
            for txt in text_history[-(args.tt_text_turns + 1):-1]:
                if txt: tt_parts.append(txt)
        if args.tt_hist_turns > 2:
            # v8+: full track text (H1: use _pos_hist)
            for tid in _pos_hist[-args.tt_hist_turns:]:
                ft = get_track_text(tid)
                if ft: tt_parts.append(ft)
        else:
            # v6 compact: name+artist only (H1: use _pos_hist)
            for tid in _pos_hist[-args.tt_hist_turns:]:
                na = get_track_name_artist(tid)
                if na: tt_parts.append(na)
        tt_query = " ".join(p for p in tt_parts if p)

        # bm25 long query (H1: use _pos_hist for track history seeds)
        bm25_parts = [_goal_for_query, culture]
        for tid in _pos_hist[-args.hist_turns:]:
            bm25_parts.append(get_track_text(tid))
        bm25_parts.extend(text_history[-args.text_turns:])
        bm25_query = " ".join(p for p in bm25_parts if p)

        # semantic query (cleaned)
        cleaned = clean_query(latest_user) or latest_user
        sem_parts = [cleaned, _goal_for_query, culture]
        for tid in _pos_hist[-args.sem_hist:]:
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

        # Track the #1 BM25 result for the bm25_top1 feature (exact-match signal)
        _bm25_top1_tid: str | None = bm25_cands[0] if bm25_cands else None

        cands = list(bm25_cands)
        cands_set = set(cands)
        sources: dict[str, set] = {tid: {"bm25"} for tid in cands}
        tt_rank_map: dict[str, int] = {}
        qm_rank_map: dict[str, int] = {}        # rank in Qwen-Meta pool
        artist_src_map: dict[str, str] = {}
        artist_rank_map: dict[str, int] = {}    # min rank within any matched artist's catalog
        nn_src_map: dict[str, str] = {}
        nn_rank_map: dict[str, int] = {}        # min rank across NN source tracks
        nn_src_count: dict[str, int] = {}       # how many prior tracks NN'd this candidate
        mean_nn_rank_map: dict[str, int] = {}   # rank under mean-session-vec NN
        collab_rank_map: dict[str, int] = {}    # best (min) position across collab sources
        collab_score_map: dict[str, float] = {} # max decayed weight
        collab_src_count: dict[str, int] = {}   # how many source tracks contributed

        # --- Encode queries (needed for expansion + scoring) ---
        tt_emb = tt_model.encode(args.tt_query_prefix + tt_query, normalize_embeddings=True, convert_to_numpy=True)
        qwen_emb = qwen_model.encode(QWEN_INSTR + semantic_query,
                                     normalize_embeddings=True, convert_to_numpy=True)
        with torch.no_grad():
            clap_raw = clap_model.get_text_embedding([semantic_query], use_tensor=True)
        clap_emb = clap_raw[0].cpu().numpy().astype(np.float32)
        clap_emb = clap_emb / max(np.linalg.norm(clap_emb), 1e-8)

        # full-index dot products (used for both expansion and scoring)
        tt_all   = tt_embs        @ tt_emb
        qm_all   = qwen_meta_embs @ qwen_emb
        ql_all   = (qwen_lyrics_embs @ qwen_emb) if (args.w_qwen_lyrics > 0 or args.ql_pool > 0) else None
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
                if src_label == "qm" and tid not in qm_rank_map:
                    qm_rank_map[tid] = rank

        # Adaptive pool: when the query contains explicit entity mentions (exact-match mode),
        # suppress CF, cooccurrence, and session-mean to avoid mainstream-bias dilution.
        _exact_match_mode = (
            args.adaptive_pool_threshold > 0
            and bool(extract_query_entities(latest_user))
        )

        add_topk(tt_all,   tt_ids,         args.tt_pool,  "tt")
        add_topk(qm_all,   qwen_meta_ids,  args.qwen_pool, "qm")
        add_topk(ql_all,   qwen_lyrics_ids, args.ql_pool,  "ql")
        if cf_all is not None and not _exact_match_mode:
            add_topk(cf_all, cf_track_ids,  args.cf_pool, "cf")

        # Sharp BM25: re-query with only latest_user + goal (no track history).
        # Targets mood/vibe queries where track history text dilutes specific mood keywords.
        if args.bm25_sharp_pool > 0:
            bm25_sharp_query = " ".join(p for p in [latest_user, goal] if p)
            if bm25_sharp_query:
                sharp_k = args.bm25_sharp_pool + len(seen) * 3
                sharp_tids, _ = retrieve_bm25(bm25_sharp_query, topk=sharp_k)
                sharp_filtered = [t for t in sharp_tids if t not in seen][:args.bm25_sharp_pool]
                for rank, tid in enumerate(sharp_filtered):
                    if tid not in cands_set:
                        cands.append(tid); cands_set.add(tid)
                        sources[tid] = set()
                    sources[tid].add("bm25_sharp")

        # Entity BM25: focused query using catalog artist names + quoted strings from latest_user.
        # Targets sessions where the user explicitly names a specific track/album/artist.
        if args.bm25_entity_pool > 0:
            entity_query = extract_query_entities(latest_user)
            if entity_query:
                entity_k = args.bm25_entity_pool + len(seen) * 3
                entity_tids, _ = retrieve_bm25(entity_query, topk=entity_k)
                entity_filtered = [t for t in entity_tids if t not in seen][:args.bm25_entity_pool]
                for rank, tid in enumerate(entity_filtered):
                    if tid not in cands_set:
                        cands.append(tid); cands_set.add(tid)
                        sources[tid] = set()
                    sources[tid].add("bm25_entity")

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

        # Per-position session NN expansion (TT space).
        # If --session_nn_ks is given (e.g. "300,200,100"), each position uses its own K.
        # Otherwise --last_nn_k applies uniformly to the last --last_nn_src tracks.
        if session_nn_ks_list:
            nn_plan = [(_pos_hist[-(i+1)], session_nn_ks_list[i])
                       for i in range(min(len(session_nn_ks_list), len(_pos_hist)))
                       if session_nn_ks_list[i] > 0]
        elif args.last_nn_k > 0 and _pos_hist:
            nn_plan = [(t, args.last_nn_k) for t in _pos_hist[-args.last_nn_src:]]
        else:
            nn_plan = []

        for src_tid, k_nn in nn_plan:
            src_idx = tt_id2idx.get(src_tid)
            if src_idx is None:
                continue
            sims = tt_embs @ tt_embs[src_idx]
            sims[src_idx] = -1e9
            k_take = min(k_nn, len(sims) - 1)
            top = np.argpartition(-sims, k_take)[:k_take]
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
                nn_src_count[tid] = nn_src_count.get(tid, 0) + 1

        # Mean-session-vector NN expansion
        mean_session_vec = None
        if _pos_hist:
            hist_idxs = [tt_id2idx.get(t) for t in _pos_hist[-args.session_mean_n:]]
            hist_idxs = [i for i in hist_idxs if i is not None]
            if hist_idxs:
                v = tt_embs[hist_idxs].mean(axis=0)
                vn = np.linalg.norm(v)
                if vn > 1e-8:
                    mean_session_vec = (v / vn).astype(np.float32)
        if args.session_mean_k > 0 and mean_session_vec is not None and not _exact_match_mode:
            sims = tt_embs @ mean_session_vec
            for i in hist_idxs:
                sims[i] = -1e9
            k_take = min(args.session_mean_k, len(sims) - 1)
            top = np.argpartition(-sims, k_take)[:k_take]
            top = top[np.argsort(-sims[top])]
            for rank, idx in enumerate(top):
                tid = tt_ids[int(idx)]
                if tid in seen:
                    continue
                if tid not in cands_set:
                    cands.append(tid); cands_set.add(tid); sources[tid] = set()
                sources[tid].add("mean_nn")
                if tid not in mean_nn_rank_map or rank < mean_nn_rank_map[tid]:
                    mean_nn_rank_map[tid] = rank

        # Co-occurrence expansion (behavioural next-song table from TRAIN)
        if cooccur_neigh_ids is not None and cooccur_ks_list and music_history and not _exact_match_mode:
            for pos, k_co in enumerate(cooccur_ks_list):
                if k_co <= 0 or pos >= len(music_history):
                    break
                src_tid = music_history[-(pos + 1)]
                src_idx = cooccur_tid2idx.get(src_tid)
                if src_idx is None:
                    continue
                neighs = cooccur_neigh_ids[src_idx]
                ws     = cooccur_neigh_w[src_idx]
                taken = 0
                for rank in range(len(neighs)):
                    if taken >= k_co:
                        break
                    nidx = int(neighs[rank])
                    if nidx < 0:
                        break
                    tid = str(cooccur_track_ids[nidx])
                    w   = float(ws[rank])
                    if tid in seen:
                        continue
                    if tid not in cands_set:
                        cands.append(tid); cands_set.add(tid); sources[tid] = set()
                    sources[tid].add("collab")
                    if tid not in collab_rank_map or rank < collab_rank_map[tid]:
                        collab_rank_map[tid] = rank
                    if w > collab_score_map.get(tid, 0.0):
                        collab_score_map[tid] = w
                    collab_src_count[tid] = collab_src_count.get(tid, 0) + 1
                    taken += 1

        # Pool recall tracking
        gold_track = turn["content"]
        if gold_track:  # skip in blind_mode where the synthetic music turn has no gold
            total_music_turns += 1
            found_in_pool_count += int(gold_track in cands_set)

        # Distance arrays (for new ranking features)
        dist_to_last_arr = None
        if music_history:
            last_idx = tt_id2idx.get(music_history[-1])
            if last_idx is not None:
                dist_to_last_arr = tt_embs @ tt_embs[last_idx]
        dist_to_mean_arr = (tt_embs @ mean_session_vec) if mean_session_vec is not None else None

        # CF-space reference vectors for Phase D distance features.
        # Stored as unit vectors; per-candidate similarity computed inline (dot product
        # with one row) instead of materialising full (47K,) arrays that are ~47x wasteful
        # given a typical ~1K candidate pool.
        cf_last_vec = None
        cf_mean_unit_vec = None
        if music_history:
            last_cf_idx = cf_track_id2idx.get(music_history[-1])
            if last_cf_idx is not None:
                cf_last_vec = cf_track_embs[last_cf_idx]
            cf_hist_idxs = [cf_track_id2idx.get(t) for t in music_history[-args.session_mean_n:]]
            cf_hist_idxs = [i for i in cf_hist_idxs if i is not None]
            if cf_hist_idxs:
                cf_mean_vec = cf_track_embs[cf_hist_idxs].mean(axis=0)
                cf_mean_norm = np.linalg.norm(cf_mean_vec)
                if cf_mean_norm > 1e-8:
                    cf_mean_unit_vec = cf_mean_vec / cf_mean_norm

        # Tag overlap: precompute set of lowered query words for tag matching
        _bm25_query_words = set(bm25_query.lower().split())
        _query_len = len(latest_user.split())
        # User intent signals (Phase D2)
        _user_has_negation = 1.0 if _NEGATION.search(latest_user) else 0.0
        _user_has_followup = 1.0 if _FOLLOWUP.search(latest_user) else 0.0
        _latest_user_words = set(latest_user.lower().split())

        if not cands:
            inference_results.append({
                "session_id": session_id, "user_id": user_id,
                "turn_number": turn_number,
                "predicted_track_ids": [], "predicted_response": "No recommendation.",
            })
            music_history.append(turn["content"])
            music_history_labels.append(_progress_by_turn.get(turn_number, ""))
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

        # --- LTR feature matrix (built when dumping or when scoring with a booster) ---
        feat = None
        if args.write_features or ltr_booster is not None:
            gold_tid = turn["content"]
            # Check goal progress for this turn
            _turn_progress = _progress_by_turn.get(turn_number, "")
            _is_rejected_gold = (_turn_progress == "DOES_NOT_MOVE_TOWARD_GOAL")
            if _progress_by_turn:
                _progress_total_turns += 1
                if _is_rejected_gold:
                    _progress_rejected_turns += 1
            # --skip_no_progress: drop the entire turn from the dump
            if args.skip_no_progress and _is_rejected_gold and args.write_features:
                _progress_skipped_turns += 1
                music_history.append(turn["content"])
                music_history_labels.append(_progress_by_turn.get(turn_number, ""))
                continue
            cold_user_flag = 1.0 if cf_all is None else 0.0
            n_cands_local = len(cands)
            feat = np.zeros((n_cands_local, len(FEATURE_COLS)), dtype=np.float32)
            # Phase E: H2 — build history-mean embeddings from labeled prior tracks
            _pos_hist_mean_vec = None
            _neg_hist_mean_vec = None
            _neg_artist_set: set = set()
            _n_rejected_hist = 0
            if args.use_goal_progress and music_history_labels:
                _pos_embs, _neg_embs = [], []
                for _h_tid, _h_lbl in zip(music_history, music_history_labels):
                    _h_idx = tt_id2idx.get(_h_tid)
                    if _h_lbl == "MOVES_TOWARD_GOAL":
                        if _h_idx is not None:
                            _pos_embs.append(tt_embs[_h_idx])
                    elif _h_lbl == "DOES_NOT_MOVE_TOWARD_GOAL":
                        _n_rejected_hist += 1
                        if _h_idx is not None:
                            _neg_embs.append(tt_embs[_h_idx])
                        _hm = metadata_dict.get(_h_tid, {})
                        _ha = ((_hm.get("artist_name") or [""])[0] or "").lower()
                        if _ha:
                            _neg_artist_set.add(_ha)
                if _pos_embs:
                    _v = np.mean(np.stack(_pos_embs), axis=0)
                    _pos_hist_mean_vec = _v / (np.linalg.norm(_v) + 1e-8)
                if _neg_embs:
                    _v = np.mean(np.stack(_neg_embs), axis=0)
                    _neg_hist_mean_vec = _v / (np.linalg.norm(_v) + 1e-8)
            lbl  = np.zeros(n_cands_local, dtype=np.int8)
            # For soft labels: pre-compute gold artist for partial-credit assignment
            if args.soft_labels:
                _gmeta = metadata_dict.get(gold_tid, {})
                gold_artist = ((_gmeta.get("artist_name") or [""])[0] or "").lower()
            for i, tid in enumerate(cands):
                bm25_s = bm25_native_sig.get(tid, args.bm25_missing_floor)
                idx_tt = tt_id2idx.get(tid)
                idx_qm = qwen_meta_id2idx.get(tid)
                idx_ql = qwen_lyrics_id2idx.get(tid) if ql_all is not None else None
                idx_c  = clap_id2idx.get(tid)
                idx_cf = cf_track_id2idx.get(tid) if cf_all is not None else None
                tt_rank = tt_rank_map.get(tid)
                tt_rank_sig_f = (1.0 / np.log2(tt_rank + 2.0)) if tt_rank is not None else 0.0
                artist_rank = artist_rank_map.get(tid)
                artist_sig_f  = (1.0 / np.log2(artist_rank + 2.0)) if artist_rank is not None else 0.0
                nn_rank = nn_rank_map.get(tid)
                nn_sig_f  = (1.0 / np.log2(nn_rank + 2.0)) if nn_rank is not None else 0.0
                srcs = sources.get(tid, ())
                qm_rank = qm_rank_map.get(tid)
                qm_rank_sig_f = (1.0 / np.log2(qm_rank + 2.0)) if qm_rank is not None else 0.0
                qm_only_f = 1.0 if ("qm" in srcs and "bm25" not in srcs and "tt" not in srcs) else 0.0
                nn_src_cnt_f = float(nn_src_count.get(tid, 0))
                mean_nn_rank = mean_nn_rank_map.get(tid)
                mean_nn_rank_sig_f = (1.0 / np.log2(mean_nn_rank + 2.0)) if mean_nn_rank is not None else 0.0
                idx_tt_for_dist = idx_tt
                dist_last_f = float(dist_to_last_arr[idx_tt_for_dist]) if (dist_to_last_arr is not None and idx_tt_for_dist is not None) else 0.0
                dist_mean_f = float(dist_to_mean_arr[idx_tt_for_dist]) if (dist_to_mean_arr is not None and idx_tt_for_dist is not None) else 0.0
                collab_rank = collab_rank_map.get(tid)
                collab_rank_sig_f = (1.0 / np.log2(collab_rank + 2.0)) if collab_rank is not None else 0.0
                _meta = metadata_dict.get(tid, {})
                _pop_raw = _meta.get("popularity")
                _pop  = float(_pop_raw) if _pop_raw is not None else np.nan
                _rel  = _meta.get("release_date") or ""
                _year = float(str(_rel)[:4]) if _rel and str(_rel)[:4].isdigit() else np.nan
                # Phase D: new features
                _n_sources_f = float(len(srcs))
                _pop_pctile_f = popularity_pctile.get(tid, 0.0)
                _yrs_since_f = float(2026 - _year) if not np.isnan(_year) else np.nan
                _tags = _meta.get("tag_list") or []
                _tag_overlap_f = float(sum(1 for t in _tags if t.lower() in _bm25_query_words))
                # Per-candidate: fraction of this track's tags matching latest user message
                _tag_query_sim_f = 0.0
                if _tags:
                    _tag_query_sim_f = sum(1 for t in _tags if t.lower() in _latest_user_words) / len(_tags)
                # Phase E: H2 per-candidate history features
                _cand_artist = ((_meta.get("artist_name") or [""])[0] or "").lower()
                _sim_pos_f = float(tt_embs[idx_tt] @ _pos_hist_mean_vec) if (idx_tt is not None and _pos_hist_mean_vec is not None) else 0.0
                _sim_neg_f = float(tt_embs[idx_tt] @ _neg_hist_mean_vec) if (idx_tt is not None and _neg_hist_mean_vec is not None) else 0.0
                _artist_rejected_f = 1.0 if (_cand_artist and _cand_artist in _neg_artist_set) else 0.0
                _n_rej_norm_f = min(_n_rejected_hist, 10) / 10.0
                _cf_dist_last_f = 0.0
                _cf_dist_mean_f = 0.0
                if cf_last_vec is not None or cf_mean_unit_vec is not None:
                    _cf_idx = cf_track_id2idx.get(tid)
                    if _cf_idx is not None:
                        _cf_row = cf_track_embs[_cf_idx]
                        if cf_last_vec is not None:
                            _cf_dist_last_f = float(_cf_row @ cf_last_vec)
                        if cf_mean_unit_vec is not None:
                            _cf_dist_mean_f = float(_cf_row @ cf_mean_unit_vec)
                feat[i] = (
                    float(tt_all[idx_tt])   if idx_tt is not None else 0.0,
                    float(qm_all[idx_qm])   if idx_qm is not None else 0.0,
                    float(ql_all[idx_ql])   if idx_ql is not None and ql_all is not None else 0.0,
                    float(clap_all[idx_c])  if idx_c  is not None else 0.0,
                    float(cf_all[idx_cf])   if idx_cf is not None and cf_all is not None else 0.0,
                    bm25_s,
                    tt_rank_sig_f,
                    artist_sig_f,
                    nn_sig_f,
                    1.0 if "bm25"   in srcs else 0.0,
                    1.0 if "artist" in srcs else 0.0,
                    1.0 if "tt"     in srcs else 0.0,
                    1.0 if "nn"     in srcs else 0.0,
                    cold_user_flag,
                    float(n_cands_local),
                    1.0 if "qm" in srcs else 0.0,
                    qm_rank_sig_f,
                    qm_only_f,
                    nn_src_cnt_f,
                    1.0 if "mean_nn" in srcs else 0.0,
                    mean_nn_rank_sig_f,
                    dist_last_f,
                    dist_mean_f,
                    1.0 if "collab" in srcs else 0.0,
                    collab_rank_sig_f,
                    float(collab_score_map.get(tid, 0.0)),
                    float(collab_src_count.get(tid, 0)),
                    _pop,
                    _year,
                    # Phase D: feature engineering v2
                    _n_sources_f,
                    float(turn_number),
                    float(len(music_history)),
                    _pop_pctile_f,
                    _yrs_since_f,
                    _tag_overlap_f,
                    float(_query_len),
                    _cf_dist_last_f,
                    _cf_dist_mean_f,
                    goal_category_int,
                    # Phase D2: user intent signals
                    _user_has_negation,
                    _user_has_followup,
                    _tag_query_sim_f,
                    # Phase F: turn-position-normalised source agreement (must precede H2 to match 44-feat model)
                    _n_sources_f / (1.0 + float(turn_number)),
                    math.log1p(_n_sources_f),
                    1.0 if tid == _bm25_top1_tid else 0.0,  # bm25_top1
                    # Phase E: H2 history-based features
                    _sim_pos_f,
                    _sim_neg_f,
                    _artist_rejected_f,
                    _n_rej_norm_f,
                )
                if tid == gold_tid:
                    # --progress_aware: gold tracks rated DOES_NOT_MOVE get label 0
                    if args.progress_aware and _is_rejected_gold:
                        lbl[i] = 0  # treat rejected gold as negative
                    else:
                        lbl[i] = 2 if args.soft_labels else 1
                elif args.soft_labels and gold_artist:
                    cand_artist = ((_meta.get("artist_name") or [""])[0] or "").lower()
                    if cand_artist and cand_artist == gold_artist:
                        lbl[i] = 1  # same artist, partial credit
            if args.write_features:
                _feat_f32 = feat.astype(np.float32)
                _X_tmpfile.write(_feat_f32.tobytes())  # stream directly to disk
                _X_n_rows[0] += n_cands_local
                _X_n_feats[0] = _feat_f32.shape[1]
                label_chunks.append(lbl)
                group_chunks.append(np.full(n_cands_local, turn_counter[0], dtype=np.int32))
                turn_meta.append({"session_id": session_id, "turn_number": turn_number,
                                  "gold": gold_tid, "n_cands": n_cands_local,
                                  "cand_ids": cands})
                turn_counter[0] += 1

        if ltr_booster is not None and feat is not None:
            lgb_feat = feat
            if _lgb_use_poly:
                _cidx = {n: i for i, n in enumerate(FEATURE_COLS)}
                _extras = []
                for _fa, _fb, _ in _LGB_POLY_PAIRS:
                    _ia, _ib = _cidx.get(_fa), _cidx.get(_fb)
                    if _ia is not None and _ib is not None:
                        _extras.append(lgb_feat[:, _ia] * lgb_feat[:, _ib])
                if _extras:
                    lgb_feat = np.hstack([lgb_feat] + [c[:, None] for c in _extras]).astype(np.float32)
            total_arr = ltr_booster.predict(lgb_feat[:, :n_booster_feats]).astype(np.float32)

        if _ltr_neural_model is not None and feat is not None:
            import torch as _torch
            _base_feats = feat[:, :len(FEATURE_COLS)].astype(np.float32)
            if _neural_use_poly:
                _col_idx = {n: i for i, n in enumerate(FEATURE_COLS)}
                _extra = []
                for _fa, _fb, _ in _NEURAL_PAIRS:
                    _ia, _ib = _col_idx.get(_fa), _col_idx.get(_fb)
                    if _ia is not None and _ib is not None:
                        _extra.append(_base_feats[:, _ia] * _base_feats[:, _ib])
                if _extra:
                    _base_feats = np.hstack([_base_feats] + [c[:, None] for c in _extra])
            _norm = (_base_feats - _neural_mean[:_base_feats.shape[1]]) / _neural_std[:_base_feats.shape[1]]
            with _torch.no_grad():
                total_arr = _ltr_neural_model(
                    _torch.from_numpy(_norm)
                ).numpy().astype(np.float32)

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
                "qm_rank": qm_rank_map.get(gold),
                "nn_src_count": nn_src_count.get(gold, 0),
                "mean_nn_rank": mean_nn_rank_map.get(gold),
                "collab_rank": collab_rank_map.get(gold),
                "collab_score": collab_score_map.get(gold),
                "collab_src_count": collab_src_count.get(gold, 0),
                "artist_match_source": artist_src_map.get(gold),
                "nn_source_track": nn_src_map.get(gold),
                "final_rank": final_rank_gold,
                "final_score": final_score_gold,
                "top20_predicted": predicted_track_ids[:20],
            }) + "\n")

        music_history.append(turn["content"])
        music_history_labels.append(_progress_by_turn.get(turn_number, ""))

Path(args.out_dir).mkdir(parents=True, exist_ok=True)
out_path = Path(args.out_dir) / f"{args.tid}.json"
with open(out_path, "w") as f:
    json.dump(inference_results, f, ensure_ascii=False, indent=2)
print(f"Saved {len(inference_results):,} predictions to {out_path}")
print(f"Pool recall: {found_in_pool_count}/{total_music_turns} = {found_in_pool_count/max(total_music_turns,1):.4f}")
if prov_fh is not None:
    prov_fh.close()

if args.write_features and _X_n_rows[0] > 0:
    _X_tmpfile.close()
    out = Path(args.write_features)
    out.parent.mkdir(parents=True, exist_ok=True)
    total_rows = _X_n_rows[0]
    n_feats    = _X_n_feats[0]
    y = np.concatenate(label_chunks, axis=0)
    g = np.concatenate(group_chunks, axis=0)
    # Build NPZ by streaming X from the temp file — never holds full X in RAM.
    # y/group/feature_cols are small (~60MB total) and concatenated normally.
    import zipfile
    import numpy.lib.format as _fmt
    _CHUNK_BYTES = 64 * 1024 * 1024  # 64MB read buffer
    with zipfile.ZipFile(str(out), "w", compression=zipfile.ZIP_STORED,
                         allowZip64=True) as _zf:
        with _zf.open("X.npy", "w", force_zip64=True) as _buf:
            _hdr = {"descr": _fmt.dtype_to_descr(np.dtype("float32")),
                    "fortran_order": False,
                    "shape": (total_rows, n_feats)}
            _fmt.write_array_header_2_0(_buf, _hdr)
            with open(_X_tmppath, "rb") as _src:
                while True:
                    _block = _src.read(_CHUNK_BYTES)
                    if not _block:
                        break
                    _buf.write(_block)
        with _zf.open("y.npy", "w") as _buf:
            np.save(_buf, y)
        with _zf.open("group.npy", "w") as _buf:
            np.save(_buf, g)
        with _zf.open("feature_cols.npy", "w") as _buf:
            np.save(_buf, np.array(FEATURE_COLS))
    _X_tmppath.unlink()  # clean up temp file
    sidecar = out.with_suffix(".meta.json")
    with open(sidecar, "w") as f:
        json.dump({"feature_cols": FEATURE_COLS,
                   "n_turns": len(turn_meta),
                   "n_rows": total_rows,
                   "turn_meta": turn_meta}, f)
    print(f"Saved features: ({total_rows}, {n_feats}) -> {out}  (sidecar {sidecar})")
    if _progress_total_turns > 0:
        print(f"  progress_aware: {_progress_rejected_turns}/{_progress_total_turns} turns had "
              f"DOES_NOT_MOVE_TOWARD_GOAL gold ({_progress_rejected_turns/_progress_total_turns:.1%})")
        if _progress_skipped_turns > 0:
            print(f"  skip_no_progress: {_progress_skipped_turns} turns dropped entirely")
