"""Step 0 + 1: Generate L0 bucket member lists and (optionally) LLM descriptions.

Step 0 (always): Build bucket_members.json — top-N tracks per L0 bucket with metadata.
Step 1 (needs API): Call an LLM to write a 2-3 sentence description per bucket.

Usage:
    # Step 0 only (no API):
    python scripts/train/generate_bucket_descriptions.py \
        --sids_dir cache/semantic_ids/runF_v8e_L2C64

    # Step 0 + 1 with Anthropic:
    python scripts/train/generate_bucket_descriptions.py \
        --sids_dir cache/semantic_ids/runF_v8e_L2C64 \
        --describe --api anthropic
"""
import argparse
import json
import os
from collections import defaultdict
from pathlib import Path

import numpy as np
from datasets import concatenate_datasets, load_dataset
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument("--sids_dir", required=True)
parser.add_argument("--top_n", type=int, default=25,
                    help="Tracks per bucket to include in prompt")
parser.add_argument("--describe", action="store_true",
                    help="Call LLM to generate descriptions (needs --api)")
parser.add_argument("--api", choices=["anthropic", "gemini"], default="anthropic")
parser.add_argument("--model", default=None,
                    help="Override model. Anthropic default: claude-haiku-4-5-20251001")
args = parser.parse_args()

sids_dir = Path(args.sids_dir)
codes = np.load(sids_dir / "semantic_ids.npy")   # (N, L)
tids  = np.load(sids_dir / "track_ids.npy", allow_pickle=True).tolist()

# Build L0 → track_ids mapping
l0_to_tids: dict[int, list[str]] = defaultdict(list)
for i, (tid, code_row) in enumerate(zip(tids, codes)):
    l0_to_tids[int(code_row[0])].append(tid)

print(f"Loaded {len(tids):,} tracks, {len(l0_to_tids)} L0 buckets")

# Load track metadata
print("Loading track metadata...")
meta_ds = load_dataset("talkpl-ai/TalkPlayData-Challenge-Track-Metadata")
all_tracks = concatenate_datasets([meta_ds["all_tracks"], meta_ds["test_tracks"]])
metadata: dict[str, dict] = {row["track_id"]: row for row in all_tracks}


def track_summary(tid: str) -> str:
    row = metadata.get(tid, {})
    name   = (row.get("track_name") or ["?"])[0]
    artist = (row.get("artist_name") or ["?"])[0]
    tags   = ", ".join((row.get("tag_list") or [])[:6])
    year   = str(row.get("release_date") or "")[:4]
    parts = [f'"{name}" by {artist}']
    if tags: parts.append(f"Tags: {tags}")
    if year: parts.append(year)
    return " | ".join(parts)


# Build bucket member dicts
bucket_members: dict[str, list[str]] = {}
for l0 in sorted(l0_to_tids):
    bucket_tids = l0_to_tids[l0]
    # Sample up to top_n tracks for the description prompt
    sample = bucket_tids[:args.top_n]
    bucket_members[str(l0)] = [track_summary(t) for t in sample]

out_members = sids_dir / "bucket_members.json"
out_members.write_text(json.dumps(bucket_members, indent=2, ensure_ascii=False))
print(f"Saved bucket member lists → {out_members}")

if not args.describe:
    print("Run with --describe to generate LLM descriptions.")
    raise SystemExit(0)


# ── LLM description generation ───────────────────────────────────────────────

PROMPT_TEMPLATE = """I have a music catalog. The following tracks have been grouped into a single semantic cluster by an embedding model:

{track_list}

Write a 2-3 sentence description of what musical style, mood, era, or genre defines this cluster. Be specific — mention genre, mood, tempo, and era where applicable. Do not mention specific track or artist names. Return only the description, no preamble."""


def build_prompt(l0: int) -> str:
    tracks = bucket_members[str(l0)]
    numbered = "\n".join(f"{i+1}. {t}" for i, t in enumerate(tracks))
    return PROMPT_TEMPLATE.format(track_list=numbered)


def call_anthropic(prompt: str, model: str) -> str:
    import anthropic
    client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
    msg = client.messages.create(
        model=model,
        max_tokens=256,
        messages=[{"role": "user", "content": prompt}],
    )
    return msg.content[0].text.strip()


def call_gemini(prompt: str, model: str) -> str:
    from google import genai
    from google.genai import types
    client = genai.Client(api_key=os.environ["GOOGLE_API_KEY"])
    resp = client.models.generate_content(
        model=model,
        contents=prompt,
        config=types.GenerateContentConfig(
            thinking_config=types.ThinkingConfig(thinking_budget=0),
            max_output_tokens=256,
        ),
    )
    return resp.text.strip()


if args.api == "anthropic":
    model = args.model or "claude-haiku-4-5-20251001"
    call_llm = lambda p: call_anthropic(p, model)
else:
    model = args.model or "gemini-2.0-flash"
    call_llm = lambda p: call_gemini(p, model)

print(f"Generating descriptions with {args.api} / {model}...")

descriptions: dict[str, str] = {}
out_desc = sids_dir / "bucket_descriptions.json"

# Resume if partially done
if out_desc.exists():
    descriptions = json.loads(out_desc.read_text())
    print(f"  Resuming: {len(descriptions)}/64 already done")

for l0 in tqdm(sorted(l0_to_tids.keys()), desc="buckets"):
    key = str(l0)
    if key in descriptions:
        continue
    prompt = build_prompt(l0)
    try:
        desc = call_llm(prompt)
        descriptions[key] = desc
    except Exception as e:
        print(f"  bucket {l0} failed: {e}")
        descriptions[key] = ""
    # Save incrementally
    out_desc.write_text(json.dumps(descriptions, indent=2, ensure_ascii=False))

print(f"Saved descriptions → {out_desc}")
print(f"Done: {sum(1 for v in descriptions.values() if v)}/{len(descriptions)} successful")
