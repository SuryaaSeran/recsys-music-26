"""
Measure pool recall@500 with v6 TT index, on all dev sessions.

Recomputes:
  - BM25@500 (using existing index + existing inference query format)
  - TT-v6 dense @K (re-encode queries with v6 model)
  - Union recall: BM25@500 + TT-v6@K (K=100, 250, 500, 1000)

Optionally adds Qwen-meta@K and CF@K from existing precomputed cache if present.

Usage:
    python scripts/inference/measure_recall_v6.py
    python scripts/inference/measure_recall_v6.py --bm25_pool 500 --tt_k 500
"""
import argparse
import json
import re
import numpy as np

_FILLER = re.compile(
    r"\b(can you|could you|would you|please|i want|i'd like|i would like|"
    r"i need|i'm looking for|i am looking for|something that(?:'s| is)|something|"
    r"recommend(?:ation)?|suggest(?:ion)?|play(?: me)?|find me|show me|give me|"
    r"how about|what about|i feel like(?: listening to)?|i(?:'m| am) in the mood for|"
    r"do you have|do you know)\b",
    re.IGNORECASE,
)

def clean_text(text):
    return re.sub(r"\s+", " ", _FILLER.sub(" ", text)).strip()
from pathlib import Path

import bm25s
import torch
from datasets import load_dataset, concatenate_datasets
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument("--bm25_pool", type=int, default=500)
parser.add_argument("--tt_k_list", default="100,250,500,1000")
parser.add_argument("--tt_model", default="models/twotower_v6/final")
parser.add_argument("--tt_index", default="cache/twotower_v6")
parser.add_argument("--bm25_cache", default="cache/bm25/track_metadata")
parser.add_argument("--hist_turns", type=int, default=4)
parser.add_argument("--text_turns", type=int, default=4)
parser.add_argument("--all_text", action="store_true", help="Use ALL user/asst turns in BM25 query, not just last N")
parser.add_argument("--clean_bm25", action="store_true", help="Strip conversational filler from user turns in BM25 query")
parser.add_argument("--multi_query", action="store_true", help="Run 2 BM25 queries (full + latest-user-only) and union")
parser.add_argument("--add_profile", action="store_true", help="Add age_group + country_name + goal category/specificity to BM25 query")
parser.add_argument("--label", default="v6_baseline")
args = parser.parse_args()

K_LIST = [int(k) for k in args.tt_k_list.split(",")]

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
    return f"{(row.get('track_name') or [''])[0]} {(row.get('artist_name') or [''])[0]}".strip()


print("Loading BM25 index...")
bm25_model = bm25s.BM25.load(args.bm25_cache, load_corpus=False)
with open(f"{args.bm25_cache}/track_ids.json") as f:
    bm25_track_ids = json.load(f)
bm25_id2idx = {tid: i for i, tid in enumerate(bm25_track_ids)}


print(f"Loading TT model: {args.tt_model}")
tt_model = SentenceTransformer(args.tt_model)

print(f"Loading TT track embeddings: {args.tt_index}")
tt_track_embs = np.load(f"{args.tt_index}/track_embeddings.npy")
with open(f"{args.tt_index}/track_ids.json") as f:
    tt_track_ids = json.load(f)
tt_id2idx = {tid: i for i, tid in enumerate(tt_track_ids)}


print("Loading dev sessions...")
ds = load_dataset("talkpl-ai/TalkPlayData-Challenge-Dataset")
sessions = list(ds["test"])
# Load second BM25 index if using v2 cache
bm25_cache2 = None
if args.bm25_cache != "cache/bm25/track_metadata":
    bm25_cache2_path = args.bm25_cache
else:
    bm25_cache2_path = None
print(f"Sessions: {len(sessions)}")


# --- collect all turns ---
all_turns = []
for sess in tqdm(sessions, desc="Collecting turns"):
    goal_obj = sess.get("conversation_goal", {}) or {}
    goal = goal_obj.get("listener_goal", "") or ""
    category = goal_obj.get("category", "") or ""
    specificity = goal_obj.get("specificity", "") or ""
    profile = sess.get("user_profile", {}) or {}
    culture = profile.get("preferred_musical_culture", "") or ""
    age_group = profile.get("age_group", "") or ""
    country = profile.get("country_name", "") or ""
    music_history = []
    text_history = []
    for turn in sess["conversations"]:
        if turn["role"] == "music":
            gold = turn["content"]
            latest_user = text_history[-1] if text_history else ""

            # BM25 query
            bm25_parts = [goal, culture]
            if args.add_profile:
                if category: bm25_parts.append(f"category {category}")
                if specificity: bm25_parts.append(specificity)
                if age_group: bm25_parts.append(age_group)
                if country: bm25_parts.append(country)
            for tid in music_history[-args.hist_turns:]:
                bm25_parts.append(get_track_text(tid))
            text_src = text_history if args.all_text else text_history[-args.text_turns:]
            for t in text_src:
                bm25_parts.append(clean_text(t) if args.clean_bm25 else t)
            bm25_query = " ".join(p for p in bm25_parts if p)

            # Multi-query: second query focused on latest user turn + goal
            bm25_query2 = None
            if args.multi_query and latest_user:
                u = clean_text(latest_user) if args.clean_bm25 else latest_user
                bm25_query2 = " ".join(p for p in [u, goal, culture] if p)

            # TT compact query (matches v6 training)
            tt_parts = [latest_user, goal, culture]
            for tid in music_history[-2:]:
                na = get_track_name_artist(tid)
                if na:
                    tt_parts.append(na)
            tt_query = " ".join(p for p in tt_parts if p)

            all_turns.append({
                "gold": gold,
                "bm25_query": bm25_query,
                "bm25_query2": bm25_query2,
                "tt_query": tt_query,
                "seen": list(music_history),
            })
            music_history.append(gold)
        else:
            text_history.append(turn["content"])

print(f"Total turns: {len(all_turns)}")


# --- BM25 retrieval ---
def bm25_retrieve(query, seen, pool_size):
    tokens = bm25s.tokenize([query.lower()], show_progress=False)
    res = bm25_model.retrieve(tokens, k=pool_size + 50, return_as="tuple", show_progress=False)
    cands = []
    for idx in res.documents[0]:
        tid = bm25_track_ids[int(idx)]
        if tid not in seen:
            cands.append(tid)
        if len(cands) >= pool_size:
            break
    return set(cands)

print("BM25 retrieval...")
bm25_pool_per_turn = []
for t in tqdm(all_turns, desc="BM25"):
    seen = set(t["seen"])
    pool = bm25_retrieve(t["bm25_query"], seen, args.bm25_pool)
    if t.get("bm25_query2"):
        pool2 = bm25_retrieve(t["bm25_query2"], seen, args.bm25_pool // 2)
        # union: keep up to bm25_pool unique unseen tracks
        combined = list(pool) + [x for x in pool2 if x not in pool]
        pool = set(combined[:args.bm25_pool])
    bm25_pool_per_turn.append(pool)


# --- TT-v6 query encoding ---
print("Encoding TT queries with v6...")
tt_queries = [t["tt_query"] for t in all_turns]
tt_q_embs = tt_model.encode(tt_queries, batch_size=128, show_progress_bar=True,
                             normalize_embeddings=True, convert_to_numpy=True)


# --- TT-v6 ranking per turn ---
print("Computing TT-v6 rankings...")
# tt_track_embs shape: (47k, 384), normalized; tt_q_embs: (N, 384), normalized
# similarity = q @ T.T
top_k_max = max(K_LIST)
tt_top_idx = np.zeros((len(all_turns), top_k_max), dtype=np.int32)
batch = 256
for i in tqdm(range(0, len(all_turns), batch), desc="TT cosine"):
    sims = tt_q_embs[i:i+batch] @ tt_track_embs.T  # (b, 47k)
    top = np.argpartition(-sims, top_k_max, axis=1)[:, :top_k_max]
    # sort within
    for j in range(top.shape[0]):
        order = np.argsort(-sims[j, top[j]])
        tt_top_idx[i+j] = top[j][order]


# --- recall measurement ---
print("\n=== Pool Recall (N={} turns) ===".format(len(all_turns)))
print(f"Label: {args.label}")
print(f"Config: BM25@{args.bm25_pool}, TT-v6@K, hist_turns={args.hist_turns}, text_turns={args.text_turns}")
print()

# BM25 only
bm25_hits = sum(1 for i, t in enumerate(all_turns) if t["gold"] in bm25_pool_per_turn[i])
print(f"BM25@{args.bm25_pool} only:                {bm25_hits}/{len(all_turns)} = {bm25_hits/len(all_turns):.4f}")

# TT only at each K
for K in K_LIST:
    tt_hits = 0
    for i, t in enumerate(all_turns):
        gold_idx = tt_id2idx.get(t["gold"])
        if gold_idx is None:
            continue
        if gold_idx in tt_top_idx[i, :K]:
            tt_hits += 1
    print(f"TT-v6@{K} only:                  {tt_hits}/{len(all_turns)} = {tt_hits/len(all_turns):.4f}")

# Union: BM25@500 + TT@K
print()
for K in K_LIST:
    union_hits = 0
    for i, t in enumerate(all_turns):
        if t["gold"] in bm25_pool_per_turn[i]:
            union_hits += 1
            continue
        gold_idx = tt_id2idx.get(t["gold"])
        if gold_idx is not None and gold_idx in tt_top_idx[i, :K]:
            union_hits += 1
    print(f"BM25@{args.bm25_pool} + TT-v6@{K}:        {union_hits}/{len(all_turns)} = {union_hits/len(all_turns):.4f}")

# Save for downstream
out = Path("exp/analysis") / f"recall_{args.label}.txt"
out.parent.mkdir(parents=True, exist_ok=True)
with open(out, "w") as f:
    f.write(f"Label: {args.label}\n")
    f.write(f"BM25@{args.bm25_pool}: {bm25_hits/len(all_turns):.4f}\n")
    for K in K_LIST:
        union = sum(1 for i, t in enumerate(all_turns)
                    if t["gold"] in bm25_pool_per_turn[i]
                    or (tt_id2idx.get(t["gold"]) is not None and tt_id2idx[t["gold"]] in tt_top_idx[i, :K]))
        f.write(f"BM25+TT@{K}: {union/len(all_turns):.4f}\n")
print(f"\nSaved summary to {out}")
