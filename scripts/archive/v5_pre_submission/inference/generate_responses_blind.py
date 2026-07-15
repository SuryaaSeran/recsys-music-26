"""
Generate Qwen3 responses for Blind A predictions.
Reads existing predicted_track_ids and generates better responses using Qwen.

Usage:
    python scripts/generate_responses_blind.py --pred exp/inference/blind_a/blind_a_v2.json
"""
import argparse
import json
from pathlib import Path
from datasets import load_dataset, concatenate_datasets
from mlx_lm import load, generate
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument("--pred", required=True)
parser.add_argument("--out", default=None)
parser.add_argument("--max_tokens", type=int, default=120)
parser.add_argument("--dataset", default="talkpl-ai/TalkPlayData-Challenge-Blind-A")
parser.add_argument("--split", default="test")
args = parser.parse_args()

out_path = args.out or args.pred.replace(".json", "_qwen.json")

print("Loading Qwen model...")
model, tokenizer = load("models/qwen_sid_patched")

print("Loading track metadata...")
meta_ds = load_dataset("talkpl-ai/TalkPlayData-Challenge-Track-Metadata")
all_tracks = concatenate_datasets([meta_ds["all_tracks"], meta_ds["test_tracks"]])
metadata_dict = {row["track_id"]: row for row in all_tracks}

print(f"Loading {args.dataset}...")
ds = load_dataset(args.dataset)[args.split]
session_map = {item["session_id"]: item for item in ds}

with open(args.pred) as f:
    preds = json.load(f)

print(f"Generating responses for {len(preds)} predictions...")
results = []

for pred in tqdm(preds, desc="Generating"):
    sid = pred["session_id"]
    uid = pred["user_id"]
    tn = pred["turn_number"]
    top_tids = pred["predicted_track_ids"]

    item = session_map.get(sid, {})
    goal = item.get("conversation_goal", {}).get("listener_goal", "")
    culture = item.get("user_profile", {}).get("preferred_musical_culture", "")
    conversations = item.get("conversations", [])
    history_convs = conversations[:-1]
    user_query = conversations[-1]["content"] if conversations else ""

    # Build top 3 recommendations text
    rec_parts = []
    for tid in top_tids[:3]:
        row = metadata_dict.get(tid, {})
        name = (row.get("track_name") or ["?"])[0]
        artist = (row.get("artist_name") or ["?"])[0]
        tags = ", ".join((row.get("tag_list") or [])[:5])
        rec_parts.append(f'"{name}" by {artist} (tags: {tags})')

    recs_text = "\n".join(f"{i+1}. {r}" for i, r in enumerate(rec_parts))

    # Build chat history
    messages = [
        {"role": "system", "content": "You are a friendly music recommendation assistant. Give a brief (2-3 sentence) recommendation that references the user's request and explains why the top track fits."}
    ]

    # Add conversation history
    for turn in history_convs[-4:]:
        role = turn["role"]
        content = turn["content"]
        if role == "music":
            row = metadata_dict.get(content, {})
            name = (row.get("track_name") or ["a track"])[0]
            artist = (row.get("artist_name") or ["the artist"])[0]
            messages.append({"role": "assistant", "content": f'I recommend "{name}" by {artist}.'})
        elif role in ("user", "assistant"):
            messages.append({"role": role, "content": content})

    messages.append({"role": "user", "content": user_query})
    messages.append({
        "role": "user",
        "content": f"Based on the request, here are my recommendations:\n{recs_text}\n\nPlease give a brief recommendation response about the top track."
    })

    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    try:
        response = generate(model, tokenizer, prompt=prompt, max_tokens=args.max_tokens).strip()
        if not response:
            top = top_tids[0] if top_tids else ""
            row = metadata_dict.get(top, {})
            name = (row.get("track_name") or ["this track"])[0]
            artist = (row.get("artist_name") or ["the artist"])[0]
            response = f'I recommend "{name}" by {artist} based on your preferences.'
    except Exception:
        top = top_tids[0] if top_tids else ""
        row = metadata_dict.get(top, {})
        name = (row.get("track_name") or ["this track"])[0]
        artist = (row.get("artist_name") or ["the artist"])[0]
        response = f'I recommend "{name}" by {artist} based on your preferences.'

    results.append({
        "session_id": sid,
        "user_id": uid,
        "turn_number": tn,
        "predicted_track_ids": top_tids,
        "predicted_response": response,
    })

with open(out_path, "w") as f:
    json.dump(results, f, ensure_ascii=False, indent=2)

print(f"Saved to {out_path}")
print("Sample response:")
print(results[0]["predicted_response"])
