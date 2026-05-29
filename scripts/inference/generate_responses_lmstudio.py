"""Generate Blind-A responses via a local LM Studio server.

POST /api/v1/chat on http://localhost:1234 (default LM Studio endpoint).
Request envelope: {model, system_prompt, input}.
Response envelope: {output:[{type:"message", content:"..."}], stats:{...}}.

Reads predictions JSON, regenerates `predicted_response` for every record
with the sharpened prompt below. `predicted_track_ids` are untouched.

Usage:
    python scripts/inference/generate_responses_lmstudio.py \
        --pred exp/inference/blind_a/blind_a_phase_a_tt2000.json \
        --out  exp/inference/blind_a/blind_a_phase_a_tt2000_lmresp.json
"""
import argparse
import hashlib
import json
import time
from pathlib import Path

import requests
from datasets import load_dataset, concatenate_datasets
from tqdm import tqdm

DEFAULT_API_URL = "http://localhost:1234/v1/chat/completions"
NATIVE_API_URL  = "http://localhost:1234/api/v1/chat"

parser = argparse.ArgumentParser()
parser.add_argument("--pred", required=True)
parser.add_argument("--out", default=None)
parser.add_argument("--api_url", default=None,
                    help="Override API endpoint. Defaults to native URL when --native_api is set, "
                         "otherwise OpenAI-compat /v1/chat/completions.")
parser.add_argument("--native_api", action="store_true", default=False,
                    help="Use the LM Studio native /api/v1/chat endpoint "
                         "(body: {model, system_prompt, input}) instead of OpenAI-compat.")
parser.add_argument("--model", default="google/gemma-4-e4b")
parser.add_argument("--dataset", default="talkpl-ai/TalkPlayData-Challenge-Blind-A")
parser.add_argument("--split", default="test")
parser.add_argument("--top_show", type=int, default=3)
parser.add_argument("--cache_dir", default=None)
parser.add_argument("--retries", type=int, default=3)
parser.add_argument("--sleep_between", type=float, default=0.0)
parser.add_argument("--timeout", type=int, default=120)
parser.add_argument("--limit", type=int, default=0,
                    help="If >0, only process the first N predictions (for A/B tests).")
args = parser.parse_args()

if args.api_url is None:
    args.api_url = NATIVE_API_URL if args.native_api else DEFAULT_API_URL

out_path = args.out or args.pred.replace(".json", "_lmresp.json")
tid = Path(out_path).stem
cache_dir = Path(args.cache_dir or f"cache/lmstudio_resp/{tid}")
cache_dir.mkdir(parents=True, exist_ok=True)

# ── Catalog ──────────────────────────────────────────────────────────────────
print("Loading track metadata...")
meta_ds = load_dataset("talkpl-ai/TalkPlayData-Challenge-Track-Metadata")
all_tracks = concatenate_datasets([meta_ds["all_tracks"], meta_ds["test_tracks"]])
metadata_dict = {row["track_id"]: row for row in all_tracks}


def track_line(t: str) -> str:
    row = metadata_dict.get(t, {})
    name = (row.get("track_name") or ["?"])[0]
    artist = (row.get("artist_name") or ["?"])[0]
    album = (row.get("album_name") or [""])[0]
    tags = ", ".join((row.get("tag_list") or [])[:8])
    year = ""
    rel = row.get("release_date") or ""
    if rel: year = str(rel)[:4]
    parts = [f'"{name}" by {artist}']
    if album: parts.append(f"Album: {album}")
    if tags:  parts.append(f"Tags: {tags}")
    if year:  parts.append(year)
    return " | ".join(parts)


def fallback(t: str) -> str:
    row = metadata_dict.get(t, {})
    name = (row.get("track_name") or ["this track"])[0]
    artist = (row.get("artist_name") or ["the artist"])[0]
    return f'I recommend "{name}" by {artist} based on your request.'


# ── Sessions ─────────────────────────────────────────────────────────────────
print(f"Loading {args.dataset} [{args.split}]...")
ds = load_dataset(args.dataset)[args.split]
session_map = {item["session_id"]: item for item in ds}

print(f"Loading predictions: {args.pred}")
preds = json.load(open(args.pred))
print(f"  records: {len(preds)}")


SYSTEM = (
    "You are a music recommendation assistant in a multi-turn conversation. "
    "You write the response the user sees when a track starts playing.\n"
    "\n"
    "Your reply MUST do four things:\n"
    "\n"
    "1. PERSONALIZE: Show you understood the specific thing the user asked for. "
    "Paraphrase their mood, artist, era, or feeling in your own words. If they "
    "said \"melancholic like Nils Frahm\", your reply must reference Frahm and "
    "melancholy. Generic replies fail.\n"
    "\n"
    "2. NAME THE TRACK: Write the track name in double quotes followed by "
    "\"by {artist}\". Use the exact name and artist from the metadata provided.\n"
    "\n"
    "3. EXPLAIN WITH EVIDENCE: Give one concrete reason from the track's metadata "
    "(year, album name, a specific tag, sonic character) that connects to the "
    "user's request. Not \"it has great vibes\" — say \"the 2007 Favourite Worst "
    "Nightmare production\" or \"the grunge and alternative rock tags\".\n"
    "\n"
    "4. CONNECT TO SESSION HISTORY: If the user has played tracks in this session, "
    "draw an explicit connection. \"Since you enjoyed Nirvana's grunge energy, "
    "this track channels a similar raw intensity.\" This is critical for "
    "personalisation scoring.\n"
    "\n"
    "Close with 1-2 sentences explaining what combination of the user's preferences, "
    "their listening history, and the track's characteristics made this the top pick.\n"
    "\n"
    "Match register to the mood: intimate for heartbreak, taut for workout, "
    "cinematic for road trips, hushed for late-night, brash for parties, "
    "snarling for rage, warm for nostalgia.\n"
    "\n"
    "HARD RULES:\n"
    "  - 3 to 5 sentences total. No 2-liners. No 6-liners.\n"
    "  - Plain prose. No headers, bullets, markdown, emojis, numbered lists.\n"
    "  - No greetings (\"Hi!\"), sign-offs (\"Enjoy!\"), or filler.\n"
    "  - Do NOT start with: \"Here's\", \"Here is\", \"I recommend\", "
    "\"Based on\", \"Absolutely\", \"Great choice\", \"Check out\", "
    "\"I've got\", \"Perfect for\", \"Built for\", \"Loaded\", \"Queued\", "
    "\"Spinning up\", \"This is\", \"These\", \"For when\", \"Threading\".\n"
    "  - Use the OPENING STYLE hint in the user message to begin your reply.\n"
    "  - NEVER invent track names, artists, lyrics, or albums. If the user "
    "named an artist not in the candidate list, reference them only as "
    "an influence or comparison.\n"
    "  - Begin with the first word of the actual reply.\n"
    "  - The response is about ONE track (#1). Do not list or mention multiple tracks.\n"
    "\n"
    "An AI judge will evaluate your response for personalisation and explanation "
    "quality. Generic, templated responses score zero."
)


# Opening "shapes" — picked deterministically per (session,turn) for diversity.
# Each one nudges the model into a different rhetorical move while still
# producing a personalized + explanatory reply.
STYLE_HINTS = [
    "Open by paraphrasing the user's request and lead into the track name "
    "(e.g. 'Sounds like you want...', 'Caught the ... ask...', 'Heard you "
    "on... — try ...').",
    "Open with a direct verb tied to the recommendation "
    "(e.g. 'Cued up ...', 'Putting on ...', 'Going to lean into ... with "
    "...', 'Reaching for ...').",
    "Open with the track name immediately, then explain why "
    "(e.g. '\"Track Name\" by Artist is the answer to ...').",
    "Open with a noun-first label of the user's vibe + dash to the track "
    "(e.g. 'Heartbreak hours — ', 'Drive-time drift — ', 'Workout fuel — ').",
    "Open with a sensory image tied to the user's mood, then introduce the "
    "track (e.g. 'Half-light, slow piano — \"Stay\" by Rihanna meets you "
    "there.').",
    "Open with 'For ...' that names the user's specific situation "
    "(e.g. 'For the slow-burn ache you described, ...', 'For the long "
    "drive ahead, ...').",
    "Open with 'If ...' tied to the user's stated need "
    "(e.g. 'If you want that 90s East Coast groove, ...').",
    "Open with 'You mentioned ...' or 'You said ...' to call back to "
    "their exact words, then introduce the track.",
    "Open with the artist or genre as a hook "
    "(e.g. 'Wood Brothers territory, so ...', 'Pure Britpop afterglow — ...').",
    "Open with a setting that matches the user's request "
    "(e.g. 'For a Sunday morning kitchen ...', 'For the last mile of the "
    "run, ...').",
    "Open with a contrast to a track already played this session "
    "(e.g. 'Shifting gears from the grunge energy of Nirvana, ...' or "
    "'Dialing back the intensity from before, ...'). Only use if tracks "
    "have been played.",
    "Open with a rhetorical question tied to the user's stated mood or use "
    "case (e.g. 'Looking for that late-night acoustic feel?' or 'Need "
    "something that hits as hard as the last track but slower?').",
    "Open with a time or place anchor that matches the user's context "
    "(e.g. 'For those quiet 2am moments ...' or 'Perfect for a long "
    "coastal drive — ...'). Keep it grounded in what the user described.",
]


def style_for(session_id: str, turn_number: int) -> str:
    key = f"{session_id}__{turn_number}".encode()
    idx = int.from_bytes(hashlib.sha256(key).digest()[:4], "big") % len(STYLE_HINTS)
    return STYLE_HINTS[idx]


def build_user(session: dict, turn_number: int, top_tids: list[str]) -> str:
    goal    = (session.get("conversation_goal") or {}).get("listener_goal", "")
    profile = session.get("user_profile") or {}
    culture    = profile.get("preferred_musical_culture", "") or ""
    age_group  = profile.get("age_group", "") or ""
    country    = profile.get("country_name", "") or ""
    gender     = profile.get("gender", "") or ""

    # Build full conversation history up to this turn (user + assistant + music)
    played_tids: list[str] = []
    history_lines: list[str] = []
    for turn in session.get("conversations") or []:
        if turn.get("turn_number") == turn_number and turn.get("role") == "music":
            break
        role = turn.get("role", "")
        content = turn.get("content", "")
        if role == "user":
            history_lines.append(f"[User] {content}")
        elif role == "assistant":
            history_lines.append(f"[Assistant] {content}")
        elif role == "music" and content:
            played_tids.append(content)
            history_lines.append(f"[Played] {track_line(content)}")

    lines = []

    # Conversation history (last 4 exchanges for context without overwhelming)
    if history_lines:
        lines.append("CONVERSATION HISTORY (most recent exchanges):")
        for h in history_lines[-8:]:   # up to 4 user+played pairs
            lines.append(h)
        lines.append("")

    # User profile
    profile_parts = []
    if goal:      profile_parts.append(f"Goal: {goal}")
    if culture:   profile_parts.append(f"Culture: {culture}")
    if age_group: profile_parts.append(f"Age: {age_group}")
    if country:   profile_parts.append(f"Country: {country}")
    if gender:    profile_parts.append(f"Gender: {gender}")
    if profile_parts:
        lines.append("USER PROFILE:")
        for p in profile_parts:
            lines.append(f"  {p}")
        lines.append("")

    # Tracks played this session (explicit material for connections)
    if played_tids:
        lines.append("TRACKS PLAYED THIS SESSION (draw connections if relevant):")
        for i, t in enumerate(played_tids[-4:], start=1):
            lines.append(f"  {i}. {track_line(t)}")
        lines.append("")

    # Top recommendation + context candidates
    lines.append("YOUR TOP RECOMMENDATION (write your response about this track):")
    if top_tids:
        lines.append(f"  1. {track_line(top_tids[0])}")
    if len(top_tids) > 1:
        lines.append("Also considered (for context only, do NOT mention in response):")
        for i, t in enumerate(top_tids[1:], start=2):
            lines.append(f"  {i}. {track_line(t)}")
    lines.append("")

    # One worked example in the user message (closer to generation = better following)
    lines.append("EXAMPLE OF A GOOD RESPONSE (different genre — do not copy phrasing):")
    lines.append(
        "User asked for \"something melancholic like Nils Frahm but with groove\": "
        "Heard you on Frahm with extra pulse, so cued up \"Says\" by Nils Frahm — "
        "same patient piano build but the rhythm thickens into a steady kick around "
        "the two-minute mark. The 2011 Erased Tapes recording has that signature "
        "half-pedal hum that fits late-night focus. Between the melancholy tag and "
        "the groove you described, this sat at the top of the list."
    )
    lines.append("")

    # Opening style hint
    lines.append(f"OPENING STYLE FOR THIS REPLY: {style_for(session['session_id'], turn_number)}")
    lines.append("")

    # Final instruction
    lines.append(
        "Write the recommendation reply for track #1 now. 3-5 sentences. "
        "Personalize to the user's specific request, name the track in double "
        "quotes with its artist, explain why it fits using a concrete metadata "
        "detail, reference their session history if relevant, and close with "
        "1-2 sentences of overall prediction reasoning."
    )
    return "\n".join(lines)


def call_lmstudio(system: str, user: str) -> str:
    if args.native_api:
        payload = {
            "model": args.model,
            "system_prompt": system,
            "input": user,
        }
    else:
        payload = {
            "model": args.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user",   "content": user},
            ],
            "max_tokens": 200,
            "temperature": 0.75,
        }
    for attempt in range(args.retries):
        try:
            r = requests.post(args.api_url, json=payload, timeout=args.timeout)
            r.raise_for_status()
            data = r.json()
            if args.native_api:
                output = data.get("output") or []
                content = next(
                    (blk.get("content") for blk in output
                     if isinstance(blk, dict) and blk.get("type") == "message"),
                    None,
                )
            else:
                choices = data.get("choices") or []
                if not choices:
                    time.sleep(min(20, 2 ** attempt)); continue
                msg = choices[0].get("message") or {}
                content = msg.get("content")  # NOT reasoning_content
                if isinstance(content, list):
                    content = "".join(blk.get("text", "") for blk in content
                                      if isinstance(blk, dict))
            if isinstance(content, str) and content.strip():
                return content.strip()
            time.sleep(min(20, 2 ** attempt))
        except Exception as e:
            time.sleep(min(20, 2 ** attempt))
            if attempt == args.retries - 1:
                print(f"  call failed after {args.retries}: {e}")
    return ""


def cache_path(p: dict) -> Path:
    return cache_dir / f"{p['session_id']}__{p['turn_number']}.json"


# ── Loop ─────────────────────────────────────────────────────────────────────
results = []
hits, api, fbk = 0, 0, 0
_preds_iter = preds[: args.limit] if args.limit > 0 else preds
print(f"  processing {len(_preds_iter)} of {len(preds)} records")
for p in tqdm(_preds_iter, desc="responses"):
    sid = p["session_id"]
    tn  = p["turn_number"]
    tids = p["predicted_track_ids"][: args.top_show]
    sess = session_map.get(sid)
    cf = cache_path(p)

    text = None
    if cf.exists():
        try:
            text = json.loads(cf.read_text()).get("response")
            if text: hits += 1
        except Exception:
            text = None

    if not text and sess is not None and tids:
        user_msg = build_user(sess, tn, tids)
        text = call_lmstudio(SYSTEM, user_msg)
        if text:
            api += 1
            cf.write_text(json.dumps({"response": text}, ensure_ascii=False))
        if args.sleep_between > 0:
            time.sleep(args.sleep_between)

    if not text:
        text = fallback(tids[0]) if tids else "Try this track."
        fbk += 1

    results.append({
        "session_id": sid,
        "user_id":    p["user_id"],
        "turn_number": tn,
        "predicted_track_ids": p["predicted_track_ids"],
        "predicted_response":  text,
    })

print(f"\ncache_hits={hits}  api_calls={api}  fallbacks={fbk}")
Path(out_path).parent.mkdir(parents=True, exist_ok=True)
with open(out_path, "w") as f:
    json.dump(results, f, ensure_ascii=False, indent=2)
print(f"Saved {len(results)} predictions to {out_path}")
print("\nSample:")
for p in results[:3]:
    print(f"  [{p['session_id'][:14]} turn={p['turn_number']}]")
    print(f"    {p['predicted_response']}")
