"""
Build two-tower v8 training dataset for intfloat/multilingual-e5-large LoRA fine-tuning.

Anchor format (E5 prefixes, token-budget-aware, no truncation):
  query: {latest_user} Goal: {goal} Type: {cat} {spec} {culture} {age} {country}
         [+ up to 4 full track texts, most recent first, if they fit]
         [+ up to 3 prior text turns, most recent first, only if budget remains]

Documents (positives + negatives): "passage: {name} by {artist} | Album: ... | Tags: ... | year"

No-truncation guarantee: each optional component (tracks, prior turns) is added only if
its token count fits within the remaining budget (MAX_ANCHOR_TOKENS=510). Components that
don't fit are skipped entirely. Prior turns have lower priority than tracks and are dropped
first when the budget is tight.

Changes from v6/nomic:
  - E5 prefixes: query: / passage: (not search_query: / search_document:)
  - Tokenizer-aware greedy anchor builder
  - All 15K TRAIN sessions; optional --exclude_n/--exclude_seed for LTR leak prevention

Usage:
    python scripts/train/build_twotower_v8_data.py
    python scripts/train/build_twotower_v8_data.py \\
        --out_dir data/twotower_v8 --hard_negs 5 \\
        --exclude_n 2000 --exclude_seed 42
"""
import argparse
import json
import random
from pathlib import Path

import bm25s
from datasets import load_dataset, concatenate_datasets
from tqdm import tqdm
from transformers import AutoTokenizer

E5_QUERY_PREFIX = "query: "
E5_DOC_PREFIX = "passage: "
MAX_ANCHOR_TOKENS = 510   # 512 - 2 for [CLS]/[SEP]

parser = argparse.ArgumentParser()
parser.add_argument("--hard_negs", type=int, default=5)
parser.add_argument("--bm25_pool", type=int, default=100)
parser.add_argument("--valid_frac", type=float, default=0.05)
parser.add_argument("--out_dir", default="data/twotower_v8")
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--exclude_seed", type=int, default=-1,
                    help="If >=0, shuffle TRAIN with this seed and skip first --exclude_n sessions.")
parser.add_argument("--exclude_n", type=int, default=0)
parser.add_argument("--max_tracks", type=int, default=4,
                    help="Max recent played tracks to greedily include in anchor.")
parser.add_argument("--max_prior_turns", type=int, default=3,
                    help="Max prior text turns to greedily include in anchor (lowest priority).")
parser.add_argument("--drop_rejected", action="store_true",
                    help="Fully exclude turns where gold is DOES_NOT_MOVE_TOWARD_GOAL "
                         "(instead of 60%% probabilistic drop). Keeps rejected tracks as "
                         "negatives for later turns.")
parser.add_argument("--more_hard_negs", type=int, default=0,
                    help="If >0, include this many BM25 hard negatives as explicit negative "
                         "columns (negative_1..negative_N). The trainer can use these via "
                         "--use_hard_neg. Default 0 = same as before (negatives in JSONL "
                         "but trainer ignores all but negative_1).")
args = parser.parse_args()

random.seed(args.seed)
CACHE_PATH = "cache/bm25/track_metadata"

PROGRESS_WEIGHT = {
    "MOVES_TOWARD_GOAL": 1.0,
    "DOES_NOT_MOVE_TOWARD_GOAL": 0.4,
    None: 1.0,
}

print("Loading E5 tokenizer...")
tokenizer = AutoTokenizer.from_pretrained("intfloat/multilingual-e5-large")


def count_tokens(text: str) -> int:
    return len(tokenizer.encode(text, add_special_tokens=False))


print("Loading track metadata...")
meta_ds = load_dataset("talkpl-ai/TalkPlayData-Challenge-Track-Metadata")
all_tracks = concatenate_datasets([meta_ds["all_tracks"], meta_ds["test_tracks"]])
metadata_dict = {row["track_id"]: row for row in all_tracks}
all_track_ids = list(metadata_dict.keys())


def get_track_text(tid: str) -> str:
    row = metadata_dict.get(tid, {})
    name = (row.get("track_name") or [""])[0]
    artist = (row.get("artist_name") or [""])[0]
    album = (row.get("album_name") or [""])[0]
    tags = " ".join((row.get("tag_list") or [])[:12])
    release_date = row.get("release_date") or ""
    year = str(release_date)[:4] if release_date else ""
    parts = [name]
    if artist:
        parts.append(f"by {artist}")
    if album:
        parts.append(f"| Album: {album}")
    if tags:
        parts.append(f"| Tags: {tags}")
    if year:
        parts.append(f"| {year}")
    return " ".join(parts).strip()


def build_anchor(latest_user: str, listener_goal: str, category: str, specificity: str,
                 culture: str, age_group: str, country: str,
                 text_hist: list[str], music_hist: list[tuple]) -> str:
    # Core (always included): user request first, then session metadata
    core_parts = [E5_QUERY_PREFIX + latest_user]
    if listener_goal:
        core_parts.append(f"Goal: {listener_goal}")
    if category or specificity:
        core_parts.append(f"Type: {category} {specificity}".strip())
    if culture:
        core_parts.append(culture)
    if age_group or country:
        core_parts.append(f"{age_group} {country}".strip())

    base = " ".join(p for p in core_parts if p)
    budget = MAX_ANCHOR_TOKENS - count_tokens(base)

    # Greedily add tracks (most recent first, highest priority)
    for tid, _ in reversed(music_hist[-args.max_tracks:]):
        ft = get_track_text(tid)
        if not ft:
            continue
        cost = count_tokens(" " + ft)
        if budget >= cost:
            base += " " + ft
            budget -= cost

    # Greedily add prior text turns (most recent first, lowest priority)
    prior_turns = text_hist[-(args.max_prior_turns + 1):-1]
    for txt in reversed(prior_turns):
        if not txt:
            continue
        cost = count_tokens(" " + txt)
        if budget >= cost:
            base += " " + txt
            budget -= cost

    return base.strip()


print("Loading BM25 index...")
bm25_model = bm25s.BM25.load(CACHE_PATH, load_corpus=False)
with open(f"{CACHE_PATH}/track_ids.json") as f:
    track_ids_list = json.load(f)


def retrieve_bm25(query: str, topk: int) -> list[str]:
    tokens = bm25s.tokenize([query.lower()])
    results = bm25_model.retrieve(tokens, k=topk, return_as="tuple")
    return [track_ids_list[int(i)] for i in results.documents[0]]


print("Loading TRAIN conversations...")
ds = load_dataset("talkpl-ai/TalkPlayData-Challenge-Dataset", split="train")
sessions = list(ds)

if args.exclude_seed >= 0 and args.exclude_n > 0:
    rng = random.Random(args.exclude_seed)
    rng.shuffle(sessions)
    held_out = sessions[: args.exclude_n]
    sessions = sessions[args.exclude_n :]
    print(f"Holding out {len(held_out)} sessions (seed {args.exclude_seed}); "
          f"using {len(sessions)} for v8 training.")

random.shuffle(sessions)
n_valid = int(len(sessions) * args.valid_frac)
valid_sessions = sessions[:n_valid]
train_sessions = sessions[n_valid:]
print(f"Train sessions: {len(train_sessions)}, Valid sessions: {len(valid_sessions)}")


def build_examples(sessions: list) -> list[dict]:
    examples = []
    truncated = 0
    for item in tqdm(sessions, desc="Building pairs"):
        goal = item.get("conversation_goal", {}) or {}
        listener_goal = goal.get("listener_goal", "") or ""
        category = goal.get("category", "") or ""
        specificity = goal.get("specificity", "") or ""

        profile = item.get("user_profile", {}) or {}
        culture = profile.get("preferred_musical_culture", "") or ""
        age_group = profile.get("age_group", "") or ""
        country = profile.get("country_name", "") or ""

        # Re-key by T-1: gpa at turn T judges the rec made at T-1, so
        # progress_by_turn[turn_num] = assessment for the rec at turn_num.
        progress_by_turn = {
            a["turn_number"] - 1: a["goal_progress_assessment"]
            for a in (item.get("goal_progress_assessments") or [])
        }

        conversations = item["conversations"]
        music_in_history = []   # list of (tid, turn_number)
        text_in_history = []    # list of user/assistant message strings
        rejected_tids = set()

        for turn in conversations:
            role = turn["role"]
            turn_num = turn["turn_number"]

            if role == "music":
                gold_tid = turn["content"]
                gold_text = get_track_text(gold_tid)
                if not gold_text.strip():
                    music_in_history.append((gold_tid, turn_num))
                    continue

                progress = progress_by_turn.get(turn_num)
                weight = PROGRESS_WEIGHT[progress]

                # --drop_rejected: skip DOES_NOT_MOVE turns entirely (but still
                # add gold to rejected_tids for use as negatives in later turns)
                if args.drop_rejected and progress == "DOES_NOT_MOVE_TOWARD_GOAL":
                    music_in_history.append((gold_tid, turn_num))
                    rejected_tids.add(gold_tid)
                    continue

                latest_user = text_in_history[-1] if text_in_history else ""

                anchor = build_anchor(
                    latest_user, listener_goal, category, specificity,
                    culture, age_group, country,
                    text_in_history, music_in_history,
                )
                if not anchor:
                    music_in_history.append((gold_tid, turn_num))
                    continue

                # Verify no truncation happened (should never trigger with greedy builder)
                n_tok = count_tokens(anchor)
                if n_tok > 512:
                    truncated += 1

                positive = E5_DOC_PREFIX + gold_text

                # Mixed negatives
                seen = set(t for t, _ in music_in_history) | {gold_tid}
                negatives = []

                # 1. BM25 top-100 negatives (up to 2)
                bm25_results = retrieve_bm25(anchor, topk=args.bm25_pool + 1)
                bm25_negs = [t for t in bm25_results if t not in seen]
                for tid in bm25_negs[:2]:
                    txt = get_track_text(tid)
                    if txt.strip():
                        negatives.append(E5_DOC_PREFIX + txt)

                # 2. Rejected track from history (up to 1)
                rejected_pool = [t for t in rejected_tids if t not in seen]
                if rejected_pool:
                    tid = random.choice(rejected_pool)
                    txt = get_track_text(tid)
                    if txt.strip():
                        negatives.append(E5_DOC_PREFIX + txt)

                # 3. Random negatives (fill remaining slots up to hard_negs)
                all_neg_pool = [t for t in all_track_ids if t not in seen]
                random.shuffle(all_neg_pool)
                for tid in all_neg_pool:
                    if len(negatives) >= args.hard_negs:
                        break
                    txt = get_track_text(tid)
                    if txt.strip():
                        negatives.append(E5_DOC_PREFIX + txt)

                ex = {"anchor": anchor, "positive": positive, "weight": weight}
                for i, neg in enumerate(negatives[:args.hard_negs]):
                    ex[f"negative_{i+1}"] = neg
                examples.append(ex)

                music_in_history.append((gold_tid, turn_num))
                if progress == "DOES_NOT_MOVE_TOWARD_GOAL":
                    rejected_tids.add(gold_tid)

            elif role in ("user", "assistant"):
                text_in_history.append(turn["content"])

    if truncated:
        print(f"  WARNING: {truncated} anchors exceeded 512 tokens despite greedy builder")
    return examples


train_examples = build_examples(train_sessions)
valid_examples = build_examples(valid_sessions)

print(f"Train examples: {len(train_examples):,}")
print(f"Valid examples: {len(valid_examples):,}")

weights = [e["weight"] for e in train_examples]
w_counts = {1.0: weights.count(1.0), 0.4: weights.count(0.4)}
print(f"Weight distribution: {w_counts}")


def apply_weights(examples: list[dict]) -> list[dict]:
    """Approximate per-sample weights by randomly dropping low-weight examples."""
    out = []
    for ex in examples:
        w = ex.get("weight", 1.0)
        if w >= 1.0 or random.random() < w:
            out.append(ex)
    return out


train_examples = apply_weights(train_examples)
print(f"After weight filtering: {len(train_examples):,} train examples")

out_dir = Path(args.out_dir)
out_dir.mkdir(parents=True, exist_ok=True)

for name, exs in [("train", train_examples), ("valid", valid_examples)]:
    path = out_dir / f"{name}.jsonl"
    with open(path, "w") as f:
        for ex in exs:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")
    print(f"Saved {len(exs):,} examples to {path}")

print("\nSample anchor (first 300 chars):")
ex = train_examples[0]
print(f"  {ex['anchor'][:300]}")
print(f"  positive: {ex['positive'][:120]}")
print(f"  weight: {ex['weight']}")
if "negative_1" in ex:
    print(f"  negative_1: {ex['negative_1'][:120]}")
n_tok = count_tokens(ex["anchor"])
print(f"  anchor tokens: {n_tok}")
