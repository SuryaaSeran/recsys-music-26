"""
MLX-based inference for Music-CRS competition.

Retrieval: plain-int adapter → coarse codes → popularity-ranked tracks from clusters
Response: base Qwen2.5-0.5B-Instruct (no adapter)
Output: exp/inference/devset/{tid}.json (standard competition format)

Usage:
    python scripts/run_inference_mlx.py [--adapter adapters/qwen_balanced_v1] [--sessions 50]
"""
import argparse
import json
import pickle
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Optional

import numpy as np
from datasets import load_dataset
from mlx_lm import load, generate
from mlx_lm.sample_utils import make_sampler
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument("--adapter", default="adapters/qwen_balanced_v1")
parser.add_argument("--sessions", type=int, default=0, help="0=all test sessions")
parser.add_argument("--samples_per_query", type=int, default=5)
parser.add_argument("--temp", type=float, default=1.0)
parser.add_argument("--split", default="test")
parser.add_argument("--out_dir", default="exp/inference/devset")
parser.add_argument("--tid", default="qwen_balanced_v1")
parser.add_argument("--no_response", action="store_true", help="skip response generation (faster)")
args = parser.parse_args()

# ── Load codebook ─────────────────────────────────────────────────────────────
print("Loading codebook...")
with open("data/codebook_v2.pkl", "rb") as f:
    cb = pickle.load(f)

track_to_codes = cb["track_to_codes"]
codes_to_tracks = cb["codes_to_tracks"]

# Build coarse retrieval index (coarse_code → sorted track list by popularity)
print("Loading coarse retrieval index...")
with open("data/coarse_retrieval_index.json") as f:
    coarse_index = json.load(f)  # str(coarse_code) → [track_id, ...]

# Track metadata for response generation
print("Loading track metadata...")
meta_ds = load_dataset("talkpl-ai/TalkPlayData-Challenge-Track-Metadata")["all_tracks"]
track_meta = {}
for row in meta_ds:
    tid = row["track_id"]
    name = row["track_name"][0] if row["track_name"] else "Unknown"
    artist = row["artist_name"][0] if row["artist_name"] else "Unknown"
    track_meta[tid] = {"name": name, "artist": artist}

print(f"Track metadata loaded: {len(track_meta):,} tracks")

# ── Load retrieval model ──────────────────────────────────────────────────────
print(f"Loading retrieval model (adapter: {args.adapter})...")
ret_model, ret_tokenizer = load("models/qwen_sid_patched", adapter_path=args.adapter)
ret_sampler = make_sampler(temp=args.temp)

# ── Optionally load response model (base, no adapter) ─────────────────────────
resp_model, resp_tokenizer = None, None
if not args.no_response:
    print("Loading response model (base Qwen)...")
    resp_model, resp_tokenizer = load("models/qwen_sid_patched")

# ── Load dataset ──────────────────────────────────────────────────────────────
print(f"Loading dataset split={args.split}...")
ds = load_dataset("talkpl-ai/TalkPlayData-Challenge-Dataset")[args.split]

# ── Helper functions ──────────────────────────────────────────────────────────

def build_system_prompt() -> str:
    return "You are an expert music recommendation assistant. Recommend tracks based on the user's preferences."


def build_retrieval_prompt(goal: str, culture: str, chat_history: list, user_query: str) -> str:
    """Format prompt for the retrieval model."""
    system = "You are a music retrieval model. Return only two integers separated by a space: coarse fine (both 0-127). Example: 62 45"
    prompt_text = f"""Goal: {goal}
User culture: {culture}
User request: {user_query}

Return only one integer pair."""
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": prompt_text},
    ]
    return ret_tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def retrieve_tracks(coarse_codes: list[int], n: int = 20) -> list[str]:
    """Get top-N tracks from given coarse clusters, de-duplicated."""
    all_tracks = []
    seen = set()
    for c in coarse_codes:
        for tid in coarse_index.get(str(c), []):
            if tid not in seen:
                seen.add(tid)
                all_tracks.append(tid)
    return all_tracks[:n]


def generate_predictions(goal: str, culture: str, chat_history: list, user_query: str) -> list[str]:
    """Generate top-20 track IDs via coarse code prediction."""
    prompt = build_retrieval_prompt(goal, culture, chat_history, user_query)

    coarse_codes = []
    for _ in range(args.samples_per_query):
        out = generate(ret_model, ret_tokenizer, prompt=prompt, max_tokens=10, sampler=ret_sampler).strip()
        m = re.fullmatch(r"(\d+)\s+(\d+)", out)
        if m:
            c, f = int(m.group(1)), int(m.group(2))
            if 0 <= c <= 127:
                coarse_codes.append(c)

    if not coarse_codes:
        # Fallback: use middle coarse clusters
        coarse_codes = [64, 32, 96]

    # Deduplicate coarse codes while preserving order (most predicted first)
    code_count = Counter(coarse_codes)
    unique_coarse = sorted(code_count.keys(), key=lambda c: -code_count[c])

    tracks = retrieve_tracks(unique_coarse[:5], n=20)

    # Pad if needed using popular tracks from any coarse code
    if len(tracks) < 20:
        for c in range(128):
            if len(tracks) >= 20:
                break
            for tid in coarse_index.get(str(c), [])[:5]:
                if tid not in tracks:
                    tracks.append(tid)

    return tracks[:20]


def generate_response(system_prompt: str, chat_history: list, top_track_id: str) -> str:
    """Generate conversational response about the top recommended track."""
    if resp_model is None:
        meta = track_meta.get(top_track_id, {})
        name = meta.get("name", "this track")
        artist = meta.get("artist", "the artist")
        return f"I recommend \"{name}\" by {artist}. It fits your musical taste and the vibe you're looking for."

    meta = track_meta.get(top_track_id, {})
    name = meta.get("name", "this track")
    artist = meta.get("artist", "the artist")

    messages = [{"role": "system", "content": system_prompt}]
    messages.extend(chat_history[-4:])  # last 4 turns for context
    messages.append({
        "role": "user",
        "content": f"Recommended track: \"{name}\" by {artist}. Please give a brief, enthusiastic recommendation response."
    })

    prompt = resp_tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    out = generate(resp_model, resp_tokenizer, prompt=prompt, max_tokens=80).strip()
    return out if out else f"I recommend \"{name}\" by {artist}!"


# ── Main inference loop ───────────────────────────────────────────────────────

sessions = list(ds)
if args.sessions > 0:
    sessions = sessions[:args.sessions]

print(f"Running inference on {len(sessions)} sessions...")

inference_results = []
for item in tqdm(sessions, desc="Sessions"):
    session_id = item["session_id"]
    user_id = item["user_id"]
    goal = item.get("conversation_goal", {}).get("listener_goal", "")
    culture = item.get("user_profile", {}).get("preferred_musical_culture", "")
    conversations = item["conversations"]

    # Find max turn number
    turn_numbers = sorted(set(c["turn_number"] for c in conversations))

    for target_turn in turn_numbers:
        # Build chat history up to (but not including) target turn
        history = []
        for turn in conversations:
            if turn["turn_number"] < target_turn:
                role = turn["role"]
                content = turn["content"]
                if role == "music":
                    meta = track_meta.get(content, {})
                    name = meta.get("name", content)
                    artist = meta.get("artist", "")
                    role = "assistant"
                    content = f"Recommended: \"{name}\" by {artist}"
                history.append({"role": role, "content": content})

        # Find user query at target turn
        user_turns = [c for c in conversations if c["turn_number"] == target_turn and c["role"] == "user"]
        if not user_turns:
            continue
        user_query = user_turns[0]["content"]

        # Check if there's a music turn at this turn number (if not, skip - no prediction needed)
        music_turns = [c for c in conversations if c["turn_number"] == target_turn and c["role"] == "music"]
        if not music_turns:
            continue

        # Generate top-20 track IDs
        predicted_track_ids = generate_predictions(goal, culture, history, user_query)

        # Generate response
        sys_prompt = build_system_prompt()
        predicted_response = generate_response(sys_prompt, history, predicted_track_ids[0] if predicted_track_ids else "")

        inference_results.append({
            "session_id": session_id,
            "user_id": user_id,
            "turn_number": target_turn,
            "predicted_track_ids": predicted_track_ids,
            "predicted_response": predicted_response,
        })

# ── Save ──────────────────────────────────────────────────────────────────────
Path(args.out_dir).mkdir(parents=True, exist_ok=True)
out_path = Path(args.out_dir) / f"{args.tid}.json"
with open(out_path, "w") as f:
    json.dump(inference_results, f, ensure_ascii=False, indent=2)

print(f"Saved {len(inference_results):,} predictions to {out_path}")
