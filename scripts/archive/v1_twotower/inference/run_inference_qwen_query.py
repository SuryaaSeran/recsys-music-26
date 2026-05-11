"""
Qwen query reformulation + BM25 + tag expansion + exclude-seen.

Uses Qwen to extract specific artist names, genres, moods from the conversation,
then runs BM25 with both the original query and the extracted terms.

Usage:
    python scripts/run_inference_qwen_query.py --dataset talkpl-ai/TalkPlayData-Challenge-Blind-A --tid blind_a_qquery
"""
import argparse
import json
import re
from pathlib import Path
from tqdm import tqdm

import bm25s
from datasets import load_dataset, concatenate_datasets
from mlx_lm import load, generate

parser = argparse.ArgumentParser()
parser.add_argument("--sessions", type=int, default=0)
parser.add_argument("--tid", default="blind_a_qquery")
parser.add_argument("--split", default="test")
parser.add_argument("--dataset", default="talkpl-ai/TalkPlayData-Challenge-Blind-A")
parser.add_argument("--out_dir", default="exp/inference/blind_a")
parser.add_argument("--topk", type=int, default=20)
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
    track_ids_list = json.load(f)

print("Loading Qwen model...")
qwen_model, qwen_tokenizer = load("models/qwen_sid_patched")


def get_track_text(tid: str) -> str:
    row = metadata_dict.get(tid, {})
    name = (row.get("track_name") or [""])[0]
    artist = (row.get("artist_name") or [""])[0]
    tags = " ".join(row.get("tag_list") or [])
    return f"{name} {artist} {tags}".strip()


def retrieve_bm25(query: str, topk: int) -> list[str]:
    tokens = bm25s.tokenize([query.lower()])
    results = bm25_model.retrieve(tokens, k=topk, return_as="tuple")
    return [track_ids_list[int(i)] for i in results.documents[0]]


def extract_query_terms(goal: str, culture: str, conversation_summary: str, user_query: str) -> str:
    """Use Qwen to extract key retrieval terms from the conversation."""
    messages = [
        {
            "role": "system",
            "content": "Extract specific music search terms from the conversation. Output ONLY a short list of keywords: artist names, song names, genres, moods. No explanation. Max 20 words."
        },
        {
            "role": "user",
            "content": f"Goal: {goal}\nCulture: {culture}\nConversation: {conversation_summary}\nRequest: {user_query}"
        }
    ]
    prompt = qwen_tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    out = generate(qwen_model, qwen_tokenizer, prompt=prompt, max_tokens=40).strip()
    return out


print(f"Loading {args.dataset} split={args.split}...")
ds = load_dataset(args.dataset)[args.split]
sessions = list(ds)
if args.sessions > 0:
    sessions = sessions[:args.sessions]

print(f"Running {len(sessions)} sessions...")
inference_results = []

for item in tqdm(sessions, desc="Sessions"):
    session_id = item["session_id"]
    user_id = item["user_id"]
    goal = item.get("conversation_goal", {}).get("listener_goal", "")
    culture = item.get("user_profile", {}).get("preferred_musical_culture", "")
    conversations = item["conversations"]

    # Blind set: last conversation is the target
    user_query = conversations[-1]["content"]
    turn_number = conversations[-1]["turn_number"]
    history_convs = conversations[:-1]

    music_in_history = []
    text_in_history = []
    for turn in history_convs:
        if turn["role"] == "music":
            music_in_history.append(turn["content"])
        elif turn["role"] in ("user", "assistant"):
            text_in_history.append(turn["content"])

    # Standard BM25 query with tag expansion
    history_parts = [goal, culture]
    music_slice = music_in_history if args.hist_turns == 0 else music_in_history[-args.hist_turns:]
    for tid in music_slice:
        history_parts.append(get_track_text(tid))
    if args.text_turns > 0:
        history_parts.extend(text_in_history[-args.text_turns:])
    history_parts.append(user_query)
    standard_query = " ".join(p for p in history_parts if p)

    # Qwen-extracted query terms
    conv_summary = " ".join(text_in_history[-4:])
    qwen_terms = extract_query_terms(goal, culture, conv_summary, user_query)

    # Combined query: standard + Qwen terms (repeated to boost weight)
    combined_query = f"{standard_query} {qwen_terms} {qwen_terms}"

    # Retrieve with exclusion
    retrieve_k = args.topk + len(music_in_history) * 3
    candidates = retrieve_bm25(combined_query, topk=retrieve_k)

    if music_in_history:
        seen = set(music_in_history)
        candidates = [t for t in candidates if t not in seen]

    predicted_track_ids = candidates[:args.topk]

    # Generate response
    top = predicted_track_ids[0] if predicted_track_ids else ""
    row = metadata_dict.get(top, {})
    name = (row.get("track_name") or ["this track"])[0]
    artist = (row.get("artist_name") or ["the artist"])[0]

    rec_parts = []
    for tid in predicted_track_ids[:3]:
        r = metadata_dict.get(tid, {})
        n = (r.get("track_name") or ["?"])[0]
        a = (r.get("artist_name") or ["?"])[0]
        tags = ", ".join((r.get("tag_list") or [])[:5])
        rec_parts.append(f'"{n}" by {a} (tags: {tags})')
    recs_text = "\n".join(f"{i+1}. {r}" for i, r in enumerate(rec_parts))

    resp_messages = [
        {"role": "system", "content": "You are a friendly music assistant. Give a brief (2-3 sentence) recommendation that references the user's request."},
        {"role": "user", "content": user_query},
        {"role": "user", "content": f"Top recommendations:\n{recs_text}\nBriefly recommend the top track."},
    ]
    resp_prompt = qwen_tokenizer.apply_chat_template(resp_messages, tokenize=False, add_generation_prompt=True)
    try:
        response = generate(qwen_model, qwen_tokenizer, prompt=resp_prompt, max_tokens=100).strip()
        if not response:
            response = f'I recommend "{name}" by {artist} based on your preferences.'
    except Exception:
        response = f'I recommend "{name}" by {artist} based on your preferences.'

    inference_results.append({
        "session_id": session_id,
        "user_id": user_id,
        "turn_number": turn_number,
        "predicted_track_ids": predicted_track_ids,
        "predicted_response": response,
    })

Path(args.out_dir).mkdir(parents=True, exist_ok=True)
out_path = Path(args.out_dir) / f"{args.tid}.json"
with open(out_path, "w") as f:
    json.dump(inference_results, f, ensure_ascii=False, indent=2)

print(f"Saved {len(inference_results):,} predictions to {out_path}")
print("Sample response:", inference_results[0]["predicted_response"])
