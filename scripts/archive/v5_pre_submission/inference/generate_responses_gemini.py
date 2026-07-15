"""Generate Blind-A responses via the Gemini API (google-genai).

Uses the same prompt structure as generate_responses_lmstudio.py.
Requires: pip install google-genai
Env var: GOOGLE_API_KEY

Usage:
    python scripts/inference/generate_responses_gemini.py \
        --pred exp/inference/blind_a/phase_d_v8b_6k_h1h3_blind_a.json \
        --out  exp/inference/blind_a/phase_d_v8b_6k_h1h3_blind_a_gemini.json
"""
import argparse
import hashlib
import json
import os
import re
import time
from pathlib import Path

from datasets import load_dataset, concatenate_datasets
from google import genai
from google.genai import types
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument("--pred", required=True)
parser.add_argument("--out", default=None)
parser.add_argument("--model", default="gemini-2.5-flash-preview-05-20")
parser.add_argument("--dataset", default="talkpl-ai/TalkPlayData-Challenge-Blind-A")
parser.add_argument("--split", default="test")
parser.add_argument("--top_show", type=int, default=3)
parser.add_argument("--cache_dir", default=None)
parser.add_argument("--retries", type=int, default=3)
parser.add_argument("--sleep_between", type=float, default=0.3)
parser.add_argument("--limit", type=int, default=0)
parser.add_argument("--temperature", type=float, default=0.8)
parser.add_argument("--no_search", action="store_true", default=False,
                    help="Disable Google Search grounding (faster, cheaper).")
args = parser.parse_args()

api_key = os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY")
if not api_key:
    raise RuntimeError("Set GOOGLE_API_KEY or GEMINI_API_KEY env var")

client = genai.Client(api_key=api_key)

out_path = args.out or args.pred.replace(".json", "_gemini.json")
tid = Path(out_path).stem
cache_dir = Path(args.cache_dir or f"cache/gemini_resp/{tid}")
cache_dir.mkdir(parents=True, exist_ok=True)

# ── Catalog ──────────────────────────────────────────────────────────────────
print("Loading track metadata...")
meta_ds = load_dataset("talkpl-ai/TalkPlayData-Challenge-Track-Metadata")
all_tracks = meta_ds["all_tracks"]
try:
    all_tracks = all_tracks.__class__.from_list(
        list(all_tracks) + list(meta_ds["test_tracks"])
    )
except Exception:
    from datasets import concatenate_datasets
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
    if rel:
        year = str(rel)[:4]
    parts = [f'"{name}" by {artist}']
    if album:
        parts.append(f"Album: {album}")
    if tags:
        parts.append(f"Tags: {tags}")
    if year:
        parts.append(year)
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
    "  - Plain prose only. No asterisks (*), underscores (_), hashes (#), "
    "no markdown, no bullet points, no numbered lists.\n"
    "  - No exclamation marks.\n"
    "  - No greetings (\"Hi!\"), sign-offs (\"Enjoy!\"), or filler.\n"
    "  - Do NOT start with: \"Here's\", \"Here is\", \"I recommend\", "
    "\"Based on\", \"Absolutely\", \"Great choice\", \"Check out\", "
    "\"I've got\", \"Perfect for\", \"Built for\", \"Loaded\", \"Queued\", "
    "\"Spinning up\", \"This is\", \"These\", \"For when\", \"Threading\", "
    "\"I've selected\", \"I have selected\", \"Let's try\", \"Let's dive\", "
    "\"How about\", \"Let me\", \"Wicked\".\n"
    "  - Use the OPENING STYLE hint in the user message to begin your reply.\n"
    "  - NEVER invent track names, artists, lyrics, or albums. Only use details "
    "that appear in the metadata provided.\n"
    "  - Begin with the first word of the actual reply.\n"
    "  - The response is about ONE track (#1). Do not mention other tracks.\n"
    "\n"
    "An AI judge will evaluate your response for personalisation and explanation "
    "quality. Generic, templated responses score zero."
)


EXAMPLE_POOL = [
    (
        "User asked for \"high-energy rap to run to\"",
        "Picking up the pace from the slower cuts you have been on, \"Power\" by "
        "Kanye West hits with a pounded drum loop and a King Crimson sample driving "
        "it forward. The 2010 My Beautiful Dark Twisted Fantasy production was built "
        "for momentum, all swagger and push. It carries the defiant energy of the rap "
        "tracks earlier in your session but tightens it for the run."
    ),
    (
        "User asked for \"something for 2am, headphones on\"",
        "Low light, a single voice — \"Nikes\" by Frank Ocean settles into exactly "
        "that hush. The hazy pitched-vocal opening drifts before the beat arrives, "
        "which suits the unwound late-night mood. It rewards an empty room."
    ),
    (
        "User asked for \"90s East Coast, raw, no polish\"",
        "\"Protect Ya Neck\" by Wu-Tang Clan is that rawness with nothing smoothed over. "
        "The 1993 Enter the Wu-Tang grit, traded verses over a stripped sample, is the "
        "blueprint for the boom-bap you keep returning to. After the cleaner cuts earlier, "
        "it swings the session back toward the dusty end you favor."
    ),
    (
        "User asked for \"a slow Sunday-morning feel\"",
        "\"Harvest Moon\" by Neil Young eases right into it. The 1992 recording leans "
        "on brushed drums and a drifting harmonica, warm and unhurried. Nothing about "
        "it asks for your full attention, which is the point."
    ),
    (
        "User asked for \"heavier, angrier than the last one\"",
        "\"Bulls on Parade\" by Rage Against the Machine answers with a scraping "
        "siren-like guitar and a rhythm section that never lets up. The 1996 Evil "
        "Empire cut pairs funk-metal groove with real fury, lining up with the harder "
        "edge you have been steering toward."
    ),
    (
        "User asked for \"melancholic like Nils Frahm but with a little groove\"",
        "\"An Ending (Ascent)\" by Brian Eno lands in the same quiet territory as "
        "Frahm — synth pads that hang rather than resolve, patient and still. The "
        "ambient and minimalism tags fit the breath-held quality you described. It "
        "sits as a companion to the slow piano played earlier rather than a departure."
    ),
    (
        "User asked for \"something that sounds like a film score, building tension\"",
        "\"Experience\" by Ludovico Einaudi starts with a single piano note and "
        "pulls in strings until the room fills. The neoclassical tag and the 2013 "
        "In a Time Lapse album both point toward exactly that cinematic architecture. "
        "After the more restrained pieces you played, the gradual build here earns it."
    ),
    (
        "User asked for \"upbeat, danceable, something from the 80s\"",
        "\"Don't You Want Me\" by The Human League hits that mark — synthesizers, "
        "a four-on-the-floor pulse, and a narrative push that keeps the floor moving. "
        "The 1981 Dare production still sounds polished decades later, which is exactly "
        "the era you were pointing toward."
    ),
]


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


def example_for(session_id: str, turn_number: int) -> str:
    key = f"ex__{session_id}__{turn_number}".encode()
    idx = int.from_bytes(hashlib.sha256(key).digest()[:4], "big") % len(EXAMPLE_POOL)
    situation, response = EXAMPLE_POOL[idx]
    return f"{situation}:\n{response}"


def build_user(session: dict, turn_number: int, top_tids: list) -> str:
    goal = (session.get("conversation_goal") or {}).get("listener_goal", "")
    profile = session.get("user_profile") or {}
    culture = profile.get("preferred_musical_culture", "") or ""
    age_group = profile.get("age_group", "") or ""
    country = profile.get("country_name", "") or ""
    gender = profile.get("gender", "") or ""

    played_tids = []
    history_lines = []
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

    if history_lines:
        lines.append("CONVERSATION HISTORY (most recent exchanges):")
        for h in history_lines[-8:]:
            lines.append(h)
        lines.append("")

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

    if played_tids:
        lines.append("TRACKS PLAYED THIS SESSION (draw connections if relevant):")
        for i, t in enumerate(played_tids[-4:], start=1):
            lines.append(f"  {i}. {track_line(t)}")
        lines.append("")

    lines.append("YOUR TOP RECOMMENDATION (write your response about this track):")
    if top_tids:
        lines.append(f"  1. {track_line(top_tids[0])}")
    if len(top_tids) > 1:
        lines.append("Also considered (for context only, do NOT mention in response):")
        for i, t in enumerate(top_tids[1:], start=2):
            lines.append(f"  {i}. {track_line(t)}")
    lines.append("")

    lines.append("EXAMPLE OF A GOOD RESPONSE (different genre — do not copy phrasing):")
    lines.append(example_for(session["session_id"], turn_number))
    lines.append("")

    lines.append(f"OPENING STYLE FOR THIS REPLY: {style_for(session['session_id'], turn_number)}")
    lines.append("")

    lines.append(
        "Write the recommendation reply for track #1 now. 3-5 sentences. "
        "Personalize to the user's specific request, name the track in double "
        "quotes with its artist, explain why it fits using a concrete metadata "
        "detail, reference their session history if relevant, and close with "
        "1-2 sentences of overall prediction reasoning."
    )
    return "\n".join(lines)


def strip_markdown(text: str) -> str:
    text = re.sub(r'\*([^*]+)\*', r'\1', text)
    text = re.sub(r'_([^_]+)_', r'\1', text)
    return text.strip()


def _parse_retry_delay(exc: Exception) -> float:
    """Extract retryDelay seconds from a 429 API exception, or return default."""
    msg = str(exc)
    import re as _re
    m = _re.search(r"'retryDelay':\s*'([0-9.]+)s'", msg)
    if m:
        return float(m.group(1)) + 2.0
    return 65.0


def call_gemini(system: str, user: str) -> str:
    tools = [] if args.no_search else [
        types.Tool(googleSearch=types.GoogleSearch())
    ]
    config = types.GenerateContentConfig(
        system_instruction=system,
        temperature=args.temperature,
        thinking_config=types.ThinkingConfig(thinking_budget=0),
        tools=tools if tools else None,
    )
    contents = [
        types.Content(
            role="user",
            parts=[types.Part.from_text(text=user)],
        )
    ]
    for attempt in range(args.retries):
        try:
            response_text = ""
            for chunk in client.models.generate_content_stream(
                model=args.model,
                contents=contents,
                config=config,
            ):
                if chunk.text:
                    response_text += chunk.text
            response_text = response_text.strip()
            if response_text:
                return strip_markdown(response_text)
            time.sleep(2)
        except Exception as e:
            delay = _parse_retry_delay(e)
            if attempt < args.retries - 1:
                print(f"  rate limited, waiting {delay:.0f}s (attempt {attempt+1}/{args.retries})")
                time.sleep(delay)
            else:
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
    tn = p["turn_number"]
    tids = p["predicted_track_ids"][: args.top_show]
    sess = session_map.get(sid)
    cf = cache_path(p)

    text = None
    if cf.exists():
        try:
            text = json.loads(cf.read_text()).get("response")
            if text:
                hits += 1
        except Exception:
            text = None

    if not text and sess is not None and tids:
        user_msg = build_user(sess, tn, tids)
        text = call_gemini(SYSTEM, user_msg)
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
        "user_id": p["user_id"],
        "turn_number": tn,
        "predicted_track_ids": p["predicted_track_ids"],
        "predicted_response": text,
    })

print(f"\ncache_hits={hits}  api_calls={api}  fallbacks={fbk}")
Path(out_path).parent.mkdir(parents=True, exist_ok=True)
with open(out_path, "w") as f:
    json.dump(results, f, ensure_ascii=False, indent=2)
print(f"Saved {len(results)} predictions to {out_path}")
print("\nSample:")
for r in results[:3]:
    print(f"  [{r['session_id'][:14]} turn={r['turn_number']}]")
    print(f"    {r['predicted_response']}")
