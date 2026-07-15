"""Generate creative, mood-matched responses via HuggingFace Router (DeepSeek).

Does NOT rerank `predicted_track_ids`; only rewrites `predicted_response`.

For each record in the input predictions JSON:
  - Build a chat prompt with the last 3 user turns + the top-3 recommended
    tracks (name, artist, top tags) for context.
  - Ask the model to:
      1) infer the mood (sad, gym/workout, happy, road-trip, focus, party,
         chill, study, heartbreak, ...) from the user's request;
      2) write a 2-3 sentence response that names the top track, fits the
         inferred mood, and slips in one quirk (a sensory detail, a tempo
         callout, a "pair this with..." aside, a tiny lyric tease, etc.).
  - Cache per-turn results so reruns / retries are cheap.

Usage:
  export HF_TOKEN=hf_...
  python scripts/inference/generate_responses_hfrouter.py \
      --pred exp/inference/blind_a/<file>.json \
      --out  exp/inference/blind_a/<file>_hfresp.json \
      --model deepseek-ai/DeepSeek-V4-Flash:novita
"""
import argparse
import json
import os
import time
from pathlib import Path

import requests
from datasets import load_dataset, concatenate_datasets
from tqdm import tqdm

API_URL = "https://router.huggingface.co/v1/chat/completions"

parser = argparse.ArgumentParser()
parser.add_argument("--pred", required=True)
parser.add_argument("--out", default=None)
parser.add_argument("--model", default="deepseek-ai/DeepSeek-V4-Flash:novita")
parser.add_argument("--dataset", default="talkpl-ai/TalkPlayData-Challenge-Blind-A")
parser.add_argument("--split", default="test")
parser.add_argument("--top_show", type=int, default=3,
                    help="How many candidate tracks to expose to the model as context.")
parser.add_argument("--cache_dir", default=None,
                    help="One JSON per (session,turn). Defaults to cache/hfrouter_resp/<tid>/")
parser.add_argument("--max_tokens", type=int, default=160)
parser.add_argument("--temperature", type=float, default=0.85)
parser.add_argument("--retries", type=int, default=4)
parser.add_argument("--sleep_between", type=float, default=0.0,
                    help="Pause this many seconds between calls (rate-limit friendly).")
args = parser.parse_args()

def _load_token() -> str | None:
    tok = os.environ.get("HF_TOKEN")
    if tok: return tok
    # fall back to .env.local in repo root (gitignored)
    env_file = Path(__file__).resolve().parents[2] / ".env.local"
    if env_file.exists():
        for line in env_file.read_text().splitlines():
            if line.startswith("HF_TOKEN="):
                return line.split("=", 1)[1].strip()
    return None


token = _load_token()
if not token:
    raise SystemExit("HF_TOKEN not set (env var or .env.local); abort.")
session = requests.Session()
session.headers.update({"Authorization": f"Bearer {token}"})

out_path = args.out or args.pred.replace(".json", "_hfresp.json")
tid = Path(out_path).stem
cache_dir = Path(args.cache_dir or f"cache/hfrouter_resp/{tid}")
cache_dir.mkdir(parents=True, exist_ok=True)

# ── Catalog ──────────────────────────────────────────────────────────────────
print("Loading track metadata...")
meta_ds = load_dataset("talkpl-ai/TalkPlayData-Challenge-Track-Metadata")
all_tracks = concatenate_datasets([meta_ds["all_tracks"], meta_ds["test_tracks"]])
metadata_dict = {row["track_id"]: row for row in all_tracks}


def track_line(tid_: str) -> str:
    row = metadata_dict.get(tid_, {})
    name = (row.get("track_name")  or ["?"])[0]
    artist = (row.get("artist_name") or ["?"])[0]
    album = (row.get("album_name") or [""])[0]
    tags = ", ".join((row.get("tag_list") or [])[:6])
    year = ""
    rel = row.get("release_date") or ""
    if rel: year = str(rel)[:4]
    parts = [f'"{name}" by {artist}']
    if album: parts.append(f"Album: {album}")
    if tags:  parts.append(f"Tags: {tags}")
    if year:  parts.append(year)
    return " | ".join(parts)


def fallback_response(tid_: str) -> str:
    row = metadata_dict.get(tid_, {})
    name = (row.get("track_name")  or ["this track"])[0]
    artist = (row.get("artist_name") or ["the artist"])[0]
    return f'I recommend "{name}" by {artist} based on your request.'


# ── Sessions ─────────────────────────────────────────────────────────────────
print(f"Loading {args.dataset} [{args.split}]...")
ds = load_dataset(args.dataset)[args.split]
session_map = {item["session_id"]: item for item in ds}

print(f"Loading predictions: {args.pred}")
preds = json.load(open(args.pred))
print(f"  records: {len(preds)}")


# ── Prompt ───────────────────────────────────────────────────────────────────
SYSTEM = (
    "You are a music recommender with a memorable voice. "
    "You will receive the user's last few turns and three candidate tracks. "
    "First, silently identify the mood the user is in (sad, gym/workout, "
    "happy, road-trip / car drive, focus, party, chill, heartbreak, study, "
    "rage, nostalgia — pick the closest single mood). "
    "Then write a 2-3 sentence recommendation that:\n"
    " - Names the FIRST track in quotes and its artist.\n"
    " - Matches the inferred mood in tone and wording.\n"
    " - Slips in ONE quirk: a tempo callout, a sensory detail "
    "(bass weight, guitar tone, vocal grain), a 'pair this with...' "
    "aside, or a brief lyric tease — anything small and unexpected.\n"
    " - Does NOT use exclamation points more than once, no emojis, no "
    "markdown, no bullet lists. Plain prose only.\n"
    " - 2 to 3 sentences. No greetings, no sign-off."
)


def build_user_prompt(session: dict, turn_number: int, top_tids: list[str]) -> str:
    goal = (session.get("conversation_goal") or {}).get("listener_goal", "")
    culture = (session.get("user_profile") or {}).get("preferred_musical_culture", "")
    user_turns: list[str] = []
    for turn in session.get("conversations") or []:
        if turn.get("turn_number") == turn_number and turn.get("role") == "music":
            break
        if turn.get("role") == "user":
            user_turns.append(turn["content"])
    last3 = user_turns[-3:]
    while len(last3) < 3:
        last3 = [""] + last3
    lines = [
        "Last user turns (most recent last):",
        f"[TURN-3] {last3[0]}",
        f"[TURN-2] {last3[1]}",
        f"[TURN-1] {last3[2]}",
        "",
    ]
    meta = []
    if goal:    meta.append(f"Goal: {goal}")
    if culture: meta.append(f"Culture: {culture}")
    if meta:
        lines.append(" | ".join(meta)); lines.append("")
    lines.append(f"Top {len(top_tids)} candidate tracks (you must recommend #1):")
    for i, t in enumerate(top_tids, start=1):
        lines.append(f"{i}. {track_line(t)}")
    lines.append("")
    lines.append("Write the recommendation now.")
    return "\n".join(lines)


def call_router(system: str, user: str) -> str:
    payload = {
        "model": args.model,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user",   "content": user},
        ],
        "max_tokens": args.max_tokens,
        "temperature": args.temperature,
    }
    for attempt in range(args.retries):
        try:
            r = session.post(API_URL, json=payload, timeout=120)
            if r.status_code in (429, 500, 502, 503, 504):
                time.sleep(min(60, 2 ** attempt))
                continue
            r.raise_for_status()
            data = r.json()
            choices = data.get("choices") or []
            if not choices:
                time.sleep(min(60, 2 ** attempt)); continue
            msg = choices[0].get("message") or {}
            content = msg.get("content")
            if isinstance(content, list):
                content = "".join(blk.get("text", "") for blk in content
                                  if isinstance(blk, dict))
            if isinstance(content, str) and content.strip():
                return content.strip()
            time.sleep(min(60, 2 ** attempt))
        except Exception as e:
            time.sleep(min(60, 2 ** attempt))
            if attempt == args.retries - 1:
                print(f"  call failed after {args.retries}: {e}")
    return ""


# ── Loop ─────────────────────────────────────────────────────────────────────
results = []
n_cache_hits = 0
n_api = 0
n_fallback = 0
for p in tqdm(preds, desc="responses"):
    sid = p["session_id"]
    tn  = p["turn_number"]
    tids = p["predicted_track_ids"][: args.top_show]
    sess = session_map.get(sid)
    cache_file = cache_dir / f"{sid}__{tn}.json"

    response_text = None
    if cache_file.exists():
        try:
            response_text = json.loads(cache_file.read_text()).get("response")
            if response_text: n_cache_hits += 1
        except Exception:
            response_text = None

    if not response_text and sess is not None and tids:
        user_msg = build_user_prompt(sess, tn, tids)
        response_text = call_router(SYSTEM, user_msg)
        if response_text:
            n_api += 1
            cache_file.write_text(json.dumps({"response": response_text}, ensure_ascii=False))
        if args.sleep_between > 0:
            time.sleep(args.sleep_between)

    if not response_text:
        response_text = fallback_response(tids[0]) if tids else "Try this track."
        n_fallback += 1

    results.append({
        "session_id": sid,
        "user_id":    p["user_id"],
        "turn_number": tn,
        "predicted_track_ids": p["predicted_track_ids"],  # unchanged
        "predicted_response":  response_text,
    })

print(f"\ncache_hits={n_cache_hits}  api_calls={n_api}  fallbacks={n_fallback}")
Path(out_path).parent.mkdir(parents=True, exist_ok=True)
with open(out_path, "w") as f:
    json.dump(results, f, ensure_ascii=False, indent=2)
print(f"Saved {len(results)} predictions to {out_path}")
print("\nSample:")
for p in results[:3]:
    print(f"  sid={p['session_id'][:18]} turn={p['turn_number']}")
    print(f"    {p['predicted_response']}")
