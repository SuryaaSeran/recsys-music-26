"""Listwise LLM reranker over an existing predictions JSON.

Reads predictions produced by run_inference_fusion_recall_expansion.py
(top-20 track IDs per turn). For each turn, asks a local Qwen via mlx-lm
to re-rank the candidates given the conversation context and writes a new
predictions JSON. Falls back to the original ordering on parse failure so
the rescorer cannot drop below the LTR baseline by a parse bug alone.

Usage:
    python scripts/inference/rescore_with_llm.py \\
        --pred exp/inference/devset/phase_a_ltr_retrained.json \\
        --model models/qwen_sid_patched \\
        --sessions 50
"""
import argparse
import json
import re
from pathlib import Path

from datasets import load_dataset, concatenate_datasets
from mlx_lm import load, generate
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument("--pred", required=True)
parser.add_argument("--out", default=None,
                    help="Output JSON path. Defaults to <pred>_llm.json.")
parser.add_argument("--model", default="models/qwen_sid_patched")
parser.add_argument("--dataset", default="talkpl-ai/TalkPlayData-Challenge-Dataset")
parser.add_argument("--split", default="test")
parser.add_argument("--top_k", type=int, default=20)
parser.add_argument("--sessions", type=int, default=0,
                    help="0 = all sessions in the predictions file; otherwise the first N session_ids.")
parser.add_argument("--max_tokens", type=int, default=80)
parser.add_argument("--verbose_every", type=int, default=200)
args = parser.parse_args()

out_path = args.out or args.pred.replace(".json", "_llm.json")

# ── Load Qwen ────────────────────────────────────────────────────────────────
print(f"Loading LLM: {args.model}")
model, tokenizer = load(args.model)

# ── Load track metadata ──────────────────────────────────────────────────────
print("Loading track metadata...")
meta_ds = load_dataset("talkpl-ai/TalkPlayData-Challenge-Track-Metadata")
all_tracks = concatenate_datasets([meta_ds["all_tracks"], meta_ds["test_tracks"]])
metadata_dict = {row["track_id"]: row for row in all_tracks}


def get_candidate_text(tid: str) -> str:
    row = metadata_dict.get(tid, {})
    name = (row.get("track_name")  or ["?"])[0]
    artist = (row.get("artist_name") or ["?"])[0]
    tags = ", ".join((row.get("tag_list") or [])[:5])
    if tags:
        return f'"{name}" by {artist} (tags: {tags})'
    return f'"{name}" by {artist}'


def get_track_name_artist(tid: str) -> str:
    row = metadata_dict.get(tid, {})
    name = (row.get("track_name")  or ["?"])[0]
    artist = (row.get("artist_name") or ["?"])[0]
    return f'"{name}" by {artist}'


# ── Load sessions ────────────────────────────────────────────────────────────
print(f"Loading {args.dataset} [{args.split}]...")
ds = load_dataset(args.dataset)[args.split]
session_map = {item["session_id"]: item for item in ds}

# ── Load predictions ─────────────────────────────────────────────────────────
print(f"Loading predictions: {args.pred}")
with open(args.pred) as f:
    preds = json.load(f)

if args.sessions > 0:
    keep_ids: set = set()
    filtered = []
    for p in preds:
        if p["session_id"] not in keep_ids:
            if len(keep_ids) >= args.sessions:
                continue
            keep_ids.add(p["session_id"])
        filtered.append(p)
    preds = filtered
    print(f"  Restricted to first {len(keep_ids)} sessions ({len(preds)} turns).")


SYSTEM = (
    "You are a music recommendation ranker. Given a user's conversation "
    "context and a numbered list of candidate tracks, re-rank the "
    "candidates from best to worst for what the user is asking for. "
    "Output ONLY a JSON list of the candidate numbers in best-to-worst "
    "order, e.g. [3,7,1,12,4,...]. Do not add any other text."
)


def build_prompt_messages(session: dict, top_tids: list[str]) -> list[dict]:
    goal = (session.get("conversation_goal") or {}).get("listener_goal", "")
    culture = (session.get("user_profile") or {}).get("preferred_musical_culture", "")
    conversations = session.get("conversations") or []

    music_history: list[str] = []
    text_history:  list[str] = []
    for turn in conversations:
        role = turn.get("role")
        if role == "music":
            music_history.append(turn["content"])
        elif role in ("user", "assistant"):
            text_history.append(turn["content"])

    convo_lines = []
    if goal:
        convo_lines.append(f"User's overall goal: {goal}")
    if culture:
        convo_lines.append(f"Preferred culture: {culture}")
    if music_history:
        recent_played = [get_track_name_artist(t) for t in music_history[-4:]]
        convo_lines.append("Recently played: " + " | ".join(recent_played))
    if text_history:
        convo_lines.append("Recent conversation:")
        for t in text_history[-4:]:
            convo_lines.append(f"- {t}")

    convo_block = "\n".join(convo_lines) if convo_lines else "(no context)"
    cand_block = "\n".join(
        f"{i+1}. {get_candidate_text(tid)}" for i, tid in enumerate(top_tids)
    )
    user_msg = (
        f"{convo_block}\n\n"
        f"Candidates:\n{cand_block}\n\n"
        f"Return a JSON list of the {len(top_tids)} candidate numbers in "
        f"best-to-worst order."
    )
    return [
        {"role": "system", "content": SYSTEM},
        {"role": "user",   "content": user_msg},
    ]


_JSON_LIST = re.compile(r"\[(?:\s*-?\d+\s*,?)+\]")
_INT       = re.compile(r"-?\d+")


def parse_ordering(text: str, n: int) -> list[int] | None:
    """Return a 0-indexed permutation of length <=n with no duplicates, or
    None on parse failure."""
    if not text:
        return None
    m = _JSON_LIST.search(text)
    if not m:
        # accept any space/comma-separated integer list as fallback
        nums = [int(x) for x in _INT.findall(text)]
    else:
        nums = [int(x) for x in _INT.findall(m.group(0))]
    if not nums:
        return None
    seen = set()
    out = []
    for v in nums:
        i = v - 1  # 1-indexed -> 0-indexed
        if 0 <= i < n and i not in seen:
            out.append(i); seen.add(i)
        if len(out) >= n:
            break
    return out or None


# ── Rerank loop ──────────────────────────────────────────────────────────────
results = []
parse_failures = 0
order_changes = 0
for k, pred in enumerate(tqdm(preds, desc="rerank")):
    sid     = pred["session_id"]
    uid     = pred["user_id"]
    tn      = pred["turn_number"]
    top_tids = pred["predicted_track_ids"][: args.top_k]

    session = session_map.get(sid)
    if session is None or len(top_tids) < 2:
        results.append({
            "session_id": sid, "user_id": uid, "turn_number": tn,
            "predicted_track_ids": pred["predicted_track_ids"],
            "predicted_response":  pred.get("predicted_response", ""),
        })
        continue

    messages = build_prompt_messages(session, top_tids)
    prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    try:
        raw = generate(model, tokenizer, prompt=prompt, max_tokens=args.max_tokens)
    except Exception:
        raw = ""

    order = parse_ordering(raw, len(top_tids))
    if order is None:
        parse_failures += 1
        new_top = list(top_tids)
    else:
        new_top = [top_tids[i] for i in order]
        # append any candidates the model didn't mention, in original order
        for i, tid in enumerate(top_tids):
            if tid not in new_top:
                new_top.append(tid)
        if new_top != list(top_tids):
            order_changes += 1

    # if the original prediction was longer than top_k, append the tail unchanged
    tail = pred["predicted_track_ids"][args.top_k:]
    new_full = new_top + [t for t in tail if t not in new_top]

    results.append({
        "session_id": sid, "user_id": uid, "turn_number": tn,
        "predicted_track_ids": new_full,
        "predicted_response":  pred.get("predicted_response", ""),
    })

    if args.verbose_every > 0 and (k + 1) % args.verbose_every == 0:
        rate = 100 * parse_failures / (k + 1)
        ch   = 100 * order_changes  / (k + 1)
        print(f"  [{k+1}/{len(preds)}] parse_fail={rate:.1f}%  order_changes={ch:.1f}%")

print(f"\nParse failures: {parse_failures}/{len(preds)} ({100*parse_failures/max(len(preds),1):.1f}%)")
print(f"Order changes:  {order_changes}/{len(preds)} ({100*order_changes/max(len(preds),1):.1f}%)")

Path(out_path).parent.mkdir(parents=True, exist_ok=True)
with open(out_path, "w") as f:
    json.dump(results, f, ensure_ascii=False, indent=2)
print(f"\nSaved {len(results)} predictions to {out_path}")
