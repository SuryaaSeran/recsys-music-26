"""
Conservative LLM reranker using a local LM Studio (OpenAI-compatible) model.

Takes a prediction JSON with top-N candidates per turn, calls the local model to
select/reorder the best 20, writes a new prediction JSON with top-20 per turn.

Usage:
    python scripts/inference/rerank_lmstudio.py \
        --pred exp/inference/devset/phase_b_reg_200sess_top25.json \
        --out  exp/inference/devset/phase_b_reg_200sess_top25_gemma.json \
        --top_k 25 \
        --model google/gemma-4-e4b \
        --api_url http://localhost:1234/v1/chat/completions

Then evaluate both:
    python scripts/inference/evaluate_local.py --pred <out>
"""
import argparse
import json
import re
import time
from pathlib import Path

import requests
from datasets import load_dataset, concatenate_datasets

parser = argparse.ArgumentParser()
parser.add_argument("--pred",    required=True)
parser.add_argument("--out",     required=True)
parser.add_argument("--top_k",   type=int, default=25)
parser.add_argument("--api_url", default="http://localhost:1234/v1/chat/completions")
parser.add_argument("--model",   default="google/gemma-4-e4b")
parser.add_argument("--temperature", type=float, default=0.1)
parser.add_argument("--max_tokens",  type=int,   default=300)
parser.add_argument("--timeout",     type=int,   default=60)
parser.add_argument("--retries",     type=int,   default=3)
parser.add_argument("--dataset",     default="talkpl-ai/TalkPlayData-Challenge-Dataset")
parser.add_argument("--split",       default="test")
parser.add_argument("--cache_dir",   default="")
args = parser.parse_args()

preds = json.load(open(args.pred))
tid   = Path(args.out).stem
cache_dir = Path(args.cache_dir or f"cache/lmstudio_rerank/{tid}")
cache_dir.mkdir(parents=True, exist_ok=True)

print("Loading track metadata...")
meta_ds   = load_dataset("talkpl-ai/TalkPlayData-Challenge-Track-Metadata")
all_tracks = concatenate_datasets([meta_ds["all_tracks"], meta_ds["test_tracks"]])
metadata_dict = {row["track_id"]: row for row in all_tracks}

print(f"Loading {args.dataset} [{args.split}]...")
ds = load_dataset(args.dataset)[args.split]
session_map = {item["session_id"]: item for item in ds}

_SELECT = 20   # LLM must output exactly this many


def candidate_line(tid: str) -> str:
    row = metadata_dict.get(tid, {})
    name   = (row.get("track_name")  or ["?"])[0]
    artist = (row.get("artist_name") or ["?"])[0]
    tags   = ", ".join((row.get("tag_list") or [])[:6])
    release = row.get("release_date") or ""
    year   = str(release)[:4] if release else ""
    pop    = float(row.get("popularity") or 0.0)
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
    name   = (row.get("track_name")  or ["this track"])[0]
    artist = (row.get("artist_name") or ["the artist"])[0]
    return f'I recommend "{name}" by {artist} based on your request.'


SYSTEM = (
    "You are a conservative music recommendation reranker. "
    "The retrieval system has already ranked these candidates well. "
    "Your ONLY job: identify if there is an absolute need to prune or reorder. "
    "Do NOT rerank just to rerank. Do NOT prune just to prune. "
    "Only act if a candidate is clearly wrong (totally wrong genre/mood/era for this specific request). "
    "Output VALID JSON only — no prose, no markdown:\n"
    '  {"order": [exactly 20 integers from 1..N, best-first]}\n'
    "If the existing order is good, return it unchanged."
)


def build_prompt(session: dict, turn_number: int, top_tids: list[str]) -> str:
    conversations = session.get("conversations") or []
    # collect user turns and previous music suggestion before this turn
    user_turns, prev_suggestion = [], ""
    for t in conversations:
        tn = t.get("turn_number", 99)
        if tn >= turn_number:
            break
        if t.get("role") == "user":
            user_turns.append(t["content"])
        elif t.get("role") == "music":
            prev_suggestion = t.get("content", "")

    last_turn = user_turns[-1] if user_turns else ""

    lines = [
        f'User request: "{last_turn}"',
        "",
        f"Ranked candidates (already ordered best-to-worst):",
    ]
    for i, t in enumerate(top_tids, start=1):
        lines.append(f"{i}. {candidate_line(t)}")
    lines += [
        "",
        f'Output JSON: {{"order": [exactly 20 integers from 1..{len(top_tids)}]}}',
    ]
    return "\n".join(lines)


_JSON_BLOCK = re.compile(r"\{.*?\}", re.DOTALL)


def parse_reply(text: str, n: int) -> list[int] | None:
    if not text:
        return None
    m = _JSON_BLOCK.search(text)
    if not m:
        return None
    try:
        d = json.loads(m.group(0))
    except json.JSONDecodeError:
        return None
    order = d.get("order")
    if not isinstance(order, list):
        return None
    seen, valid = set(), []
    for v in order:
        try:
            i = int(v) - 1
        except (TypeError, ValueError):
            continue
        if 0 <= i < n and i not in seen:
            valid.append(i); seen.add(i)
    return valid if len(valid) == _SELECT else None


def call_lmstudio(system: str, user: str) -> str:
    # Supports both /api/v1/chat and /v1/chat/completions formats.
    use_lmstudio_api = "/api/v1/chat" in args.api_url
    if use_lmstudio_api:
        payload = {"model": args.model, "system_prompt": system, "input": user}
    else:
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
            r = requests.post(args.api_url, json=payload, timeout=args.timeout)
            r.raise_for_status()
            data = r.json()
            if use_lmstudio_api:
                outputs = data.get("output") or []
                content = next((o["content"] for o in outputs
                                if o.get("type") == "message"), "")
            else:
                choices = data.get("choices") or []
                if not choices:
                    time.sleep(min(10, 2 ** attempt)); continue
                msg     = choices[0].get("message") or {}
                content = msg.get("content", "")
                if isinstance(content, list):
                    content = "".join(b.get("text", "") for b in content
                                      if isinstance(b, dict))
            if isinstance(content, str) and content.strip():
                return content.strip()
            time.sleep(min(10, 2 ** attempt))
        except Exception as e:
            time.sleep(min(10, 2 ** attempt))
            if attempt == args.retries - 1:
                print(f"  call failed: {e}")
    return ""


def cache_path(p: dict) -> Path:
    return cache_dir / f"{p['session_id']}__{p['turn_number']}.json"


results  = []
n_ok, n_fail, n_cached = 0, 0, 0

for idx, p in enumerate(preds):
    sid     = p["session_id"]
    tn      = p["turn_number"]
    top_tids = p["predicted_track_ids"][: args.top_k]
    session = session_map.get(sid)

    if session is None or len(top_tids) < _SELECT:
        results.append(p); n_fail += 1; continue

    cp = cache_path(p)
    if cp.exists():
        cached = json.loads(cp.read_text())
        order  = cached.get("order")
        n_cached += 1
    else:
        user  = build_prompt(session, tn, top_tids)
        raw   = call_lmstudio(SYSTEM, user)
        order = parse_reply(raw, len(top_tids))
        cp.write_text(json.dumps({"raw": raw, "order": order}, ensure_ascii=False))

    if order:
        new_top = [top_tids[i] for i in order]
        selected = set(new_top)
        for t in top_tids:
            if t not in selected:
                new_top.append(t)
        n_ok += 1
    else:
        new_top = list(top_tids)
        n_fail  += 1

    tail    = p["predicted_track_ids"][args.top_k:]
    new_full = new_top + [t for t in tail if t not in set(new_top)]
    results.append({
        "session_id": sid, "user_id": p["user_id"],
        "turn_number": tn,
        "predicted_track_ids": new_full,
        "predicted_response":  top1_template(new_full[0] if new_full else ""),
    })

    if (idx + 1) % 100 == 0:
        print(f"  [{idx+1}/{len(preds)}]  ok={n_ok}  fail={n_fail}  cached={n_cached}")

Path(args.out).parent.mkdir(parents=True, exist_ok=True)
with open(args.out, "w") as f:
    json.dump(results, f, ensure_ascii=False, indent=2)

print(f"\nDone. ok={n_ok}  fail={n_fail}  cached={n_cached}")
print(f"Saved: {args.out}")
