"""
BM25 + CF candidate expansion: add CF-similar tracks to BM25 pool, rank by BM25 score.

Approach:
1. BM25 top-N candidates (ranked by BM25)
2. CF top-M candidates (tracks with highest user CF similarity)
3. Union, ranked by BM25 position (CF additions get rank N+1, N+2, ...)
4. Exclude seen tracks, return top-20

Usage:
    python scripts/run_inference_cf_expand.py [--cf_extra 50] [--tid cf_expand_v1]
"""
import argparse
import json
import numpy as np
from pathlib import Path
from tqdm import tqdm

import bm25s
from datasets import load_dataset, concatenate_datasets

parser = argparse.ArgumentParser()
parser.add_argument("--sessions", type=int, default=0)
parser.add_argument("--tid", default="cf_expand_v1")
parser.add_argument("--split", default="test")
parser.add_argument("--out_dir", default="exp/inference/devset")
parser.add_argument("--topk", type=int, default=20)
parser.add_argument("--bm25_pool", type=int, default=200)
parser.add_argument("--cf_extra", type=int, default=50, help="Extra CF candidates to add")
parser.add_argument("--hist_turns", type=int, default=4)
parser.add_argument("--text_turns", type=int, default=4)
args = parser.parse_args()

CACHE_PATH = "cache/bm25/track_metadata"

print("Loading track metadata...")
meta_ds = load_dataset("talkpl-ai/TalkPlayData-Challenge-Track-Metadata")
all_tracks = concatenate_datasets([meta_ds["all_tracks"], meta_ds["test_tracks"]])
metadata_dict = {row["track_id"]: row for row in all_tracks}

print("Loading BM25 index...")
bm25_model = bm25s.BM25.load(CACHE_PATH, load_corpus=False)
with open(f"{CACHE_PATH}/track_ids.json") as f:
    bm25_track_ids = json.load(f)

print("Loading CF-BPR track embeddings...")
track_emb_raw = np.load("data/challenge_track_embeddings.npy", mmap_mode="r")
cf_slice = track_emb_raw[:, 1280:1408].copy()
cf_norms = np.linalg.norm(cf_slice, axis=1, keepdims=True)
track_cf = cf_slice / (cf_norms + 1e-8)
with open("data/challenge_track_ids.txt") as f:
    emb_track_ids = [l.strip() for l in f]
emb_idx_map = {tid: i for i, tid in enumerate(emb_track_ids)}
emb_id_list = list(emb_track_ids)

print("Loading user CF-BPR embeddings...")
user_ds = load_dataset("talkpl-ai/TalkPlayData-Challenge-User-Embeddings")
user_cf_map = {}
for split_name in ["test_warm", "test_cold", "train"]:
    for row in user_ds[split_name]:
        cf = row.get("cf-bpr")
        if cf and any(cf):
            v = np.array(cf, dtype=np.float32)
            v = v / (np.linalg.norm(v) + 1e-8)
            user_cf_map[row["user_id"]] = v
print(f"User CF: {len(user_cf_map)}")


def get_track_text(tid: str) -> str:
    row = metadata_dict.get(tid, {})
    name = (row.get("track_name") or [""])[0]
    artist = (row.get("artist_name") or [""])[0]
    tags = " ".join(row.get("tag_list") or [])
    return f"{name} {artist} {tags}".strip()


def retrieve_bm25(query: str, topk: int) -> list[str]:
    tokens = bm25s.tokenize([query.lower()])
    results = bm25_model.retrieve(tokens, k=topk, return_as="tuple")
    return [bm25_track_ids[int(i)] for i in results.documents[0]]


def get_cf_top(user_id: str, k: int) -> list[str]:
    user_cf = user_cf_map.get(user_id)
    if user_cf is None:
        return []
    scores = track_cf @ user_cf  # (N,)
    top_indices = np.argpartition(scores, -k)[-k:]
    top_indices = top_indices[np.argsort(scores[top_indices])[::-1]]
    return [emb_id_list[i] for i in top_indices]


print(f"Loading dataset split={args.split}...")
ds = load_dataset("talkpl-ai/TalkPlayData-Challenge-Dataset")[args.split]
sessions = list(ds)
if args.sessions > 0:
    sessions = sessions[:args.sessions]

print(f"Running {len(sessions)} sessions (cf_extra={args.cf_extra})...")
inference_results = []

for item in tqdm(sessions, desc="Sessions"):
    session_id = item["session_id"]
    user_id = item["user_id"]
    goal = item.get("conversation_goal", {}).get("listener_goal", "")
    culture = item.get("user_profile", {}).get("preferred_musical_culture", "")
    conversations = item["conversations"]

    for target_turn in sorted(set(c["turn_number"] for c in conversations)):
        music_turns = [c for c in conversations if c["turn_number"] == target_turn and c["role"] == "music"]
        if not music_turns:
            continue
        user_turns = [c for c in conversations if c["turn_number"] == target_turn and c["role"] == "user"]
        if not user_turns:
            continue
        user_query = user_turns[0]["content"]

        music_in_history = []
        text_in_history = []
        for turn in conversations:
            if turn["turn_number"] >= target_turn:
                break
            if turn["role"] == "music":
                music_in_history.append(turn["content"])
            elif turn["role"] in ("user", "assistant"):
                text_in_history.append(turn["content"])

        history_parts = [goal, culture]
        music_slice = music_in_history if args.hist_turns == 0 else music_in_history[-args.hist_turns:]
        for tid in music_slice:
            history_parts.append(get_track_text(tid))
        if args.text_turns > 0:
            history_parts.extend(text_in_history[-args.text_turns:])
        history_parts.append(user_query)
        retrieval_query = " ".join(p for p in history_parts if p)

        # BM25 candidates
        retrieve_k = args.bm25_pool + len(music_in_history) * 3
        bm25_cands = retrieve_bm25(retrieval_query, topk=retrieve_k)

        # CF expansion candidates
        cf_cands = get_cf_top(user_id, k=args.cf_extra + len(music_in_history) * 2)

        # Union: BM25 order first, then CF additions
        seen_in_cands = set(bm25_cands)
        cf_additions = [t for t in cf_cands if t not in seen_in_cands]
        all_candidates = bm25_cands + cf_additions[:args.cf_extra]

        # Exclude seen tracks
        if music_in_history:
            seen = set(music_in_history)
            all_candidates = [t for t in all_candidates if t not in seen]

        predicted_track_ids = all_candidates[:args.topk]

        top = predicted_track_ids[0] if predicted_track_ids else ""
        row = metadata_dict.get(top, {})
        name = (row.get("track_name") or ["this track"])[0]
        artist = (row.get("artist_name") or ["the artist"])[0]
        response = f'I recommend "{name}" by {artist} based on your preferences.'

        inference_results.append({
            "session_id": session_id,
            "user_id": user_id,
            "turn_number": target_turn,
            "predicted_track_ids": predicted_track_ids,
            "predicted_response": response,
        })

Path(args.out_dir).mkdir(parents=True, exist_ok=True)
out_path = Path(args.out_dir) / f"{args.tid}.json"
with open(out_path, "w") as f:
    json.dump(inference_results, f, ensure_ascii=False, indent=2)

print(f"Saved {len(inference_results):,} predictions to {out_path}")
