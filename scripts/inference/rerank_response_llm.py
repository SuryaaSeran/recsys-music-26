"""Stage 3: Anthropic Claude does one-call rerank + response per turn.

Reads a predictions JSON (top-K candidates per turn). For each turn:
  - Builds a chat prompt with the last 3 user turns and the K candidates
    (full metadata: name, artist, album, top tags, year).
  - Calls Claude with a strict JSON response format.
  - Parses `order` (rerank permutation) and `response` (text reply).
  - Falls back to the input order / a template response on parse failure.
  - Writes a new predictions JSON with `predicted_track_ids` reordered
    and `predicted_response` updated.

Per-turn results are cached as JSON files under --cache_dir so reruns are
free (cost control + recovery from crashes).

Usage:
    export ANTHROPIC_API_KEY=...
    python scripts/inference/rerank_response_llm.py \
        --pred exp/inference/devset/<stage2>.json \
        --out  exp/inference/devset/<stage2>_opus.json \
        --model claude-opus-4-7 --sessions 50 --concurrency 8
"""
import argparse
import asyncio
import json
import os
import re
import time
from pathlib import Path

from anthropic import AsyncAnthropic, APIError, APIStatusError
from datasets import load_dataset, concatenate_datasets
from tqdm.asyncio import tqdm

parser = argparse.ArgumentParser()
parser.add_argument("--pred", required=True)
parser.add_argument("--out", default=None)
parser.add_argument("--model", default="claude-opus-4-7")
parser.add_argument("--dataset", default="talkpl-ai/TalkPlayData-Challenge-Dataset")
parser.add_argument("--split", default="test")
parser.add_argument("--top_k", type=int, default=20)
parser.add_argument("--sessions", type=int, default=0,
                    help="0 = all sessions in the pred file; otherwise first N.")
parser.add_argument("--concurrency", type=int, default=8)
parser.add_argument("--max_tokens", type=int, default=500)
parser.add_argument("--cache_dir", default=None,
                    help="Per-turn JSON cache. Defaults to cache/llm_rerank/<tid>/.")
parser.add_argument("--temperature", type=float, default=0.0)
parser.add_argument("--retries", type=int, default=4)
args = parser.parse_args()

out_path = args.out or args.pred.replace(".json", "_opus.json")

# Derive a stable tid from the output path for cache dir
tid = Path(out_path).stem
cache_dir = Path(args.cache_dir or f"cache/llm_rerank/{tid}")
cache_dir.mkdir(parents=True, exist_ok=True)

if not os.environ.get("ANTHROPIC_API_KEY"):
    raise SystemExit("ANTHROPIC_API_KEY not set; aborting.")

client = AsyncAnthropic()

# ── Catalog ──────────────────────────────────────────────────────────────────
print("Loading track metadata...")
meta_ds = load_dataset("talkpl-ai/TalkPlayData-Challenge-Track-Metadata")
all_tracks = concatenate_datasets([meta_ds["all_tracks"], meta_ds["test_tracks"]])
metadata_dict = {row["track_id"]: row for row in all_tracks}


def candidate_line(tid: str) -> str:
    row = metadata_dict.get(tid, {})
    name   = (row.get("track_name")  or ["?"])[0]
    artist = (row.get("artist_name") or ["?"])[0]
    tags   = ", ".join((row.get("tag_list") or [])[:6])
    release = row.get("release_date") or ""
    year  = str(release)[:4] if release else ""
    pop   = float(row.get("popularity") or 0.0)
    if pop >= 0.7:   pop_label = "High"
    elif pop >= 0.3: pop_label = "Medium"
    elif pop > 0.0:  pop_label = "Low"
    else:            pop_label = ""
    pieces = [f'"{name}" by {artist}']
    if year:      pieces.append(f"Year: {year}")
    if pop_label: pieces.append(f"Popularity: {pop_label}")
    if tags:      pieces.append(f"Tags: {tags}")
    return " | ".join(pieces)


def top1_template(tid: str) -> str:
    row = metadata_dict.get(tid, {})
    name = (row.get("track_name")  or ["this track"])[0]
    artist = (row.get("artist_name") or ["the artist"])[0]
    return f'I recommend "{name}" by {artist} based on your request.'


# ── Sessions ─────────────────────────────────────────────────────────────────
print(f"Loading {args.dataset} [{args.split}]...")
ds = load_dataset(args.dataset)[args.split]
session_map = {item["session_id"]: item for item in ds}

print(f"Loading predictions: {args.pred}")
preds = json.load(open(args.pred))
if args.sessions > 0:
    keep, filt = set(), []
    for p in preds:
        if p["session_id"] not in keep:
            if len(keep) >= args.sessions:
                continue
            keep.add(p["session_id"])
        filt.append(p)
    preds = filt
    print(f"  Restricted to first {len(keep)} sessions ({len(preds)} turns).")


# ── Prompt ───────────────────────────────────────────────────────────────────
SYSTEM = (
    "You are a light-touch music recommendation reranker.\n"
    "\n"
    "The retrieval system has already ranked N candidate tracks roughly best-to-worst "
    "for this user's request. It is generally accurate. Your job is conservative "
    "correction only — do not overthink it:\n"
    "  - If a candidate is genuinely wrong (clearly wrong genre, mood, or era for the "
    "request), prune it from the list.\n"
    "  - If two adjacent candidates are obviously in the wrong relative order, swap them.\n"
    "  - Otherwise, keep the existing order.\n"
    "\n"
    "Output a single JSON object with one field and nothing else:\n"
    "  \"order\": a list of exactly 20 integers chosen from 1..N, best-first.\n"
    "            Omit any candidates that clearly do not belong.\n"
    "            If all N candidates fit, return the best 20 in rank order.\n"
    "\n"
    "Output VALID JSON only. No prose before or after. No markdown fences."
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
    meta_line = []
    if goal:    meta_line.append(f"Goal: {goal}")
    if culture: meta_line.append(f"Culture: {culture}")
    if meta_line:
        lines.append(" | ".join(meta_line)); lines.append("")
    lines.append(f"Candidates (retrieval order, length={len(top_tids)}):")
    for i, t in enumerate(top_tids, start=1):
        lines.append(f"{i}. {candidate_line(t)}")
    lines.append("")
    lines.append(
        f"Output the reranked order as JSON: {{\"order\": [20 integers from 1..{len(top_tids)}]}}."
        " Conservative: only move or prune when clearly needed."
    )
    return "\n".join(lines)


_JSON_BLOCK = re.compile(r"\{.*\}", re.DOTALL)


_SELECT = 20  # LLM selects this many from the N candidates


def parse_reply(text: str, n: int) -> list[int] | None:
    """Return 0-indexed order list of length _SELECT, or None on parse/validation failure."""
    if not text: return None
    m = _JSON_BLOCK.search(text)
    if not m: return None
    try:
        d = json.loads(m.group(0))
    except json.JSONDecodeError:
        return None
    order = d.get("order")
    if not isinstance(order, list):
        return None
    try:
        order = [int(x) for x in order]
    except (TypeError, ValueError):
        return None
    seen = set()
    valid = []
    for v in order:
        i = v - 1
        if 0 <= i < n and i not in seen:
            valid.append(i); seen.add(i)
    return valid if len(valid) == _SELECT else None


async def call_claude(system: str, user: str) -> str:
    for attempt in range(args.retries):
        try:
            resp = await client.messages.create(
                model=args.model,
                max_tokens=args.max_tokens,
                temperature=args.temperature,
                system=system,
                messages=[{"role": "user", "content": user}],
            )
            return "".join(block.text for block in resp.content
                           if getattr(block, "type", "") == "text")
        except (APIStatusError, APIError) as e:
            delay = min(60, 2 ** attempt)
            await asyncio.sleep(delay)
            if attempt == args.retries - 1:
                print(f"  API failed after {args.retries}: {e}")
                return ""
    return ""


def cache_path(p: dict) -> Path:
    return cache_dir / f"{p['session_id']}__{p['turn_number']}.json"


async def process_one(p: dict, sem: asyncio.Semaphore) -> dict:
    sid = p["session_id"]
    top_tids = p["predicted_track_ids"][: args.top_k]
    session = session_map.get(sid)
    if session is None or len(top_tids) < 2:
        return p

    cp = cache_path(p)
    if cp.exists():
        cached = json.loads(cp.read_text())
        order = cached.get("order")
    else:
        async with sem:
            user = build_user_prompt(session, p["turn_number"], top_tids)
            raw = await call_claude(SYSTEM, user)
        order = parse_reply(raw, len(top_tids))
        cp.write_text(json.dumps({"raw": raw, "order": order}, ensure_ascii=False))

    if order:
        new_top = [top_tids[i] for i in order]
        # append any candidates not selected by LLM (in original rank order)
        selected = set(new_top)
        for t in top_tids:
            if t not in selected:
                new_top.append(t)
    else:
        new_top = list(top_tids)

    tail = p["predicted_track_ids"][args.top_k:]
    new_full = new_top + [t for t in tail if t not in set(new_top)]
    return {
        "session_id": sid, "user_id": p["user_id"],
        "turn_number": p["turn_number"],
        "predicted_track_ids": new_full,
        "predicted_response":  top1_template(new_full[0] if new_full else ""),
    }


async def main():
    sem = asyncio.Semaphore(args.concurrency)
    tasks = [process_one(p, sem) for p in preds]
    results = []
    for fut in tqdm.as_completed(tasks, total=len(tasks), desc="rerank"):
        results.append(await fut)
    # preserve original order
    by_key = {(r["session_id"], r["turn_number"]): r for r in results}
    ordered = [by_key[(p["session_id"], p["turn_number"])] for p in preds]
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(ordered, f, ensure_ascii=False, indent=2)
    n_reranked = sum(1 for r in results
                     if r["predicted_track_ids"] != next(
                         (p["predicted_track_ids"][:args.top_k]
                          for p in preds
                          if p["session_id"] == r["session_id"]
                          and p["turn_number"] == r["turn_number"]), []))
    print(f"\nTurns reranked (order changed): {n_reranked}/{len(results)}")
    print(f"Cache dir: {cache_dir}")
    print(f"Saved {out_path}")


asyncio.run(main())
