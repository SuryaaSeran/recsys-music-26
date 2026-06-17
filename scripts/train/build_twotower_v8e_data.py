"""Build two-tower v8e training data.

Extends v8d with listener thought (T_t^l) in history blocks.

Anchor format (role-tagged, what blind eval exposes at target turn t):

    query: [PROFILE] {age_group} · {country_code} · {gender} · {culture} · {language}
    [GOAL] {listener_goal}  ({specificity})
    [T1] USER: ... | REC: title – artist | ASST: ... | REACTION: liked
    [T2] USER: ... | REC: ... | ASST: ... | REACTION: rejected | LISTENER: <thought>
    [NOW] USER: <Q_t>

Listener thought (user role `thought` field, T_t^l):
  - Available in both train data and blind A at eval time (turn 2+)
  - Truncated to first 60 tokens to fit within 510-token budget
  - Added as "| LISTENER: ..." slot after REACTION, only when non-empty

Music role thought (T_t^r): excluded — null in blind A.
Token budget 510 (E5 max - 2).

Positives:
  P_{t+1} = MOVES         → weight 1.0
  P_{t+1} = DOES_NOT      → weight 0.3 (subsampled)
  M_8 (no P_9)            → weight 1.0 (neutral)

Hard negatives (per anchor, in order of value):
  1. Confirmed in-session rejections (P_{i+1} = DOES_NOT, M_i)
  2. (HH/LH only) BM25 top-K aggressive hard negs
  3. Artist-repeat distractor (any tid by an artist already recommended)
  4. Random catalog fill to --hard_negs

False-negative protection:
  - Never use a track that was MOVES anywhere in the same session as a negative
  - LL/HL specificity: skip BM25 same-session positional negs (rely on
    explicit DOES_NOT + in-batch + random)

Usage:
    python scripts/train/build_twotower_v8d_data.py \\
        --out_dir data/twotower_v8d \\
        --hard_negs 5 --valid_frac 0.10 \\
        --exclude_n 6000 --exclude_seed 42
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
MAX_ANCHOR_TOKENS = 510

PROGRESS_WEIGHT = {
    "MOVES_TOWARD_GOAL": 1.0,
    "DOES_NOT_MOVE_TOWARD_GOAL": 0.3,
    None: 1.0,
}

REACTION_LABEL = {
    "MOVES_TOWARD_GOAL": "liked",
    "DOES_NOT_MOVE_TOWARD_GOAL": "rejected",
    None: "unknown",
}

parser = argparse.ArgumentParser()
parser.add_argument("--out_dir", default="data/twotower_v8e")
parser.add_argument("--hard_negs", type=int, default=5)
parser.add_argument("--bm25_pool", type=int, default=100)
parser.add_argument("--valid_frac", type=float, default=0.10)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--exclude_n", type=int, default=6000)
parser.add_argument("--exclude_seed", type=int, default=42)
parser.add_argument("--base_model", default="intfloat/multilingual-e5-base")
args = parser.parse_args()

random.seed(args.seed)

print(f"Loading tokenizer: {args.base_model}")
tokenizer = AutoTokenizer.from_pretrained(args.base_model)


def count_tokens(text: str) -> int:
    return len(tokenizer.encode(text, add_special_tokens=False))


print("Loading track metadata...")
meta_ds = load_dataset("talkpl-ai/TalkPlayData-Challenge-Track-Metadata")
all_tracks = concatenate_datasets([meta_ds["all_tracks"], meta_ds["test_tracks"]])
metadata_dict = {row["track_id"]: row for row in all_tracks}
all_track_ids = list(metadata_dict.keys())
artist_to_tids: dict[str, list[str]] = {}
for tid, row in metadata_dict.items():
    art = ((row.get("artist_name") or [""])[0] or "").lower()
    if art:
        artist_to_tids.setdefault(art, []).append(tid)


def get_track_text(tid: str) -> str:
    """Full track text for item-tower (passage:) encoding."""
    row = metadata_dict.get(tid, {})
    name = (row.get("track_name") or [""])[0]
    artist = (row.get("artist_name") or [""])[0]
    album = (row.get("album_name") or [""])[0]
    tags = " ".join((row.get("tag_list") or [])[:12])
    year = str(row.get("release_date") or "")[:4]
    parts = [name]
    if artist: parts.append(f"by {artist}")
    if album: parts.append(f"| Album: {album}")
    if tags: parts.append(f"| Tags: {tags}")
    if year: parts.append(f"| {year}")
    return " ".join(parts).strip()


def get_track_short(tid: str) -> str:
    """Compact 'title – artist' for the REC: slot in the anchor."""
    row = metadata_dict.get(tid, {})
    name = (row.get("track_name") or [""])[0]
    artist = (row.get("artist_name") or [""])[0]
    if name and artist:
        return f"{name} – {artist}"
    return (name or artist or "").strip()


def get_track_artist(tid: str) -> str:
    row = metadata_dict.get(tid, {})
    return ((row.get("artist_name") or [""])[0] or "").lower()


def build_anchor(profile: dict, goal_text: str, specificity: str,
                 history_blocks: list, current_query: str) -> str:
    """Compose anchor in role-tagged format with greedy token budget."""
    profile_parts = []
    for k in ("age_group", "country_code", "gender", "culture", "language"):
        v = profile.get(k, "")
        if v: profile_parts.append(str(v))
    profile_line = "[PROFILE] " + " · ".join(profile_parts) if profile_parts else "[PROFILE]"

    goal_line = f"[GOAL] {goal_text}".strip()
    if specificity:
        goal_line += f"  ({specificity})"

    now_line = f"[NOW] USER: {current_query}".strip()

    # Always-present core
    core_text = profile_line + "\n" + goal_line + "\n" + now_line
    budget = MAX_ANCHOR_TOKENS - count_tokens(E5_QUERY_PREFIX + core_text)

    # Greedy history insertion: most recent first, full → without ASST → minimal
    added = []
    for hb in reversed(history_blocks):
        lt = hb.get("listener_thought", "")
        lt_slot = f" | LISTENER: {lt}" if lt else ""
        full = (f"[T{hb['turn']}] USER: {hb['user']} | REC: {hb['rec']} "
                f"| ASST: {hb['asst']} | REACTION: {hb['reaction']}{lt_slot}")
        short = (f"[T{hb['turn']}] USER: {hb['user']} | REC: {hb['rec']} "
                 f"| REACTION: {hb['reaction']}{lt_slot}")
        minimal = (f"[T{hb['turn']}] USER: {hb['user']} "
                   f"| REC: {hb['rec']} | REACTION: {hb['reaction']}{lt_slot}")
        for cand in (full, short, minimal):
            cost = count_tokens("\n" + cand)
            if budget >= cost:
                added.append(cand)
                budget -= cost
                break

    added.reverse()
    parts = [profile_line, goal_line] + added + [now_line]
    return E5_QUERY_PREFIX + "\n".join(parts)


print("Loading BM25 index...")
bm25_model = bm25s.BM25.load("cache/bm25/track_metadata", load_corpus=False)
with open("cache/bm25/track_metadata/track_ids.json") as f:
    track_ids_list = json.load(f)


def retrieve_bm25(query: str, topk: int) -> list[str]:
    tokens = bm25s.tokenize([query.lower()])
    results = bm25_model.retrieve(tokens, k=topk, return_as="tuple")
    return [track_ids_list[int(i)] for i in results.documents[0]]


print("Loading TRAIN conversations...")
ds = load_dataset("talkpl-ai/TalkPlayData-Challenge-Dataset", split="train")
sessions = list(ds)

if args.exclude_n > 0:
    rng = random.Random(args.exclude_seed)
    rng.shuffle(sessions)
    held_out = sessions[:args.exclude_n]
    sessions = sessions[args.exclude_n:]
    print(f"Held out {len(held_out)} sessions (seed {args.exclude_seed}); "
          f"using {len(sessions)} for v8d.")

random.shuffle(sessions)
n_valid = int(len(sessions) * args.valid_frac)
valid_sessions = sessions[:n_valid]
train_sessions = sessions[n_valid:]
print(f"Train: {len(train_sessions)}  Valid: {len(valid_sessions)}")


def build_examples(sessions: list) -> list[dict]:
    examples = []
    weight_counts = {1.0: 0, 0.3: 0}

    for item in tqdm(sessions, desc="Building"):
        profile_raw = item.get("user_profile", {}) or {}
        profile = {
            "age_group": profile_raw.get("age_group", "") or "",
            "country_code": profile_raw.get("country_code", "") or "",
            "gender": profile_raw.get("gender", "") or "",
            "culture": profile_raw.get("preferred_musical_culture", "") or "",
            "language": profile_raw.get("preferred_language", "") or "",
        }
        goal = item.get("conversation_goal", {}) or {}
        goal_text = goal.get("listener_goal", "") or ""
        specificity = goal.get("specificity", "") or ""

        # Re-keyed progress: progress_by_turn[T] = reaction to rec at turn T
        progress_by_turn = {
            a["turn_number"] - 1: a["goal_progress_assessment"]
            for a in (item.get("goal_progress_assessments") or [])
        }

        # Collate turn-aligned data (user/music/assistant per turn)
        turn_data: dict[int, dict] = {}
        for t in item["conversations"]:
            tn = t["turn_number"]
            slot = turn_data.setdefault(tn, {"user": "", "music": "", "asst": "", "user_thought": ""})
            if t["role"] == "user":
                slot["user"] = t["content"] or ""
                slot["user_thought"] = t.get("thought") or ""
            elif t["role"] == "music":
                slot["music"] = t["content"] or ""
            elif t["role"] == "assistant":
                slot["asst"] = t["content"] or ""

        # MOVES-protection: tracks that were MOVES anywhere in this session
        moves_tids = {
            td["music"] for tn, td in turn_data.items()
            if td["music"] and progress_by_turn.get(tn) == "MOVES_TOWARD_GOAL"
        }

        history_blocks: list[dict] = []
        rejected_tids: set[str] = set()
        artist_history: set[str] = set()
        aggressive = specificity in ("HH", "LH")

        for tn in sorted(turn_data):
            td = turn_data[tn]
            gold_tid = td["music"]
            if not gold_tid:
                continue
            gold_text = get_track_text(gold_tid)
            if not gold_text.strip():
                # still update history for later turns
                _maybe_add_history(history_blocks, rejected_tids, artist_history,
                                   tn, td, gold_tid, progress_by_turn)
                continue

            # Build anchor (history reflects only turns < tn)
            anchor = build_anchor(profile, goal_text, specificity,
                                  history_blocks, td["user"])

            positive = E5_DOC_PREFIX + gold_text
            weight = PROGRESS_WEIGHT.get(progress_by_turn.get(tn), 1.0)

            # ── Hard negative mining ──────────────────────────────────────
            seen = {gold_tid}
            negatives = []

            # 1) Confirmed rejections from earlier in this session
            rej_pool = [t for t in rejected_tids if t not in seen and t not in moves_tids]
            random.shuffle(rej_pool)
            for tid in rej_pool[:2]:
                txt = get_track_text(tid)
                if txt.strip():
                    negatives.append(E5_DOC_PREFIX + txt)
                    seen.add(tid)

            # 2) BM25 hard negs (aggressive only for HH/LH)
            if aggressive:
                bm25_results = retrieve_bm25(anchor, topk=args.bm25_pool + 1)
                for tid in bm25_results:
                    if len(negatives) >= 3:
                        break
                    if tid in seen or tid in moves_tids:
                        continue
                    txt = get_track_text(tid)
                    if txt.strip():
                        negatives.append(E5_DOC_PREFIX + txt)
                        seen.add(tid)

            # 3) Artist-repeat distractor (any specificity — explicit signal)
            ar_candidates = []
            for art in artist_history:
                for tid in artist_to_tids.get(art, [])[:30]:
                    if tid in seen or tid in moves_tids:
                        continue
                    ar_candidates.append(tid)
                    if len(ar_candidates) >= 20:
                        break
                if len(ar_candidates) >= 20:
                    break
            if ar_candidates:
                tid = random.choice(ar_candidates)
                txt = get_track_text(tid)
                if txt.strip():
                    negatives.append(E5_DOC_PREFIX + txt)
                    seen.add(tid)

            # 4) Random catalog fill
            rand_pool = all_track_ids[:]
            random.shuffle(rand_pool)
            for tid in rand_pool:
                if len(negatives) >= args.hard_negs:
                    break
                if tid in seen or tid in moves_tids:
                    continue
                txt = get_track_text(tid)
                if txt.strip():
                    negatives.append(E5_DOC_PREFIX + txt)
                    seen.add(tid)

            ex = {
                "anchor": anchor,
                "positive": positive,
                "weight": weight,
                "specificity": specificity,
                "turn": tn,
            }
            for i, neg in enumerate(negatives[:args.hard_negs]):
                ex[f"negative_{i+1}"] = neg
            examples.append(ex)
            weight_counts[weight] = weight_counts.get(weight, 0) + 1

            _maybe_add_history(history_blocks, rejected_tids, artist_history,
                               tn, td, gold_tid, progress_by_turn)

    print(f"  Weight distribution: {weight_counts}")
    return examples


def _maybe_add_history(history_blocks, rejected_tids, artist_history,
                       tn, td, gold_tid, progress_by_turn):
    """Append the just-emitted turn to history for future turns."""
    rec_short = get_track_short(gold_tid)
    reaction = REACTION_LABEL.get(progress_by_turn.get(tn), "unknown")
    # Truncate listener thought to first sentence (≤200 chars)
    raw_thought = td.get("user_thought") or ""
    if raw_thought:
        end = raw_thought.find(". ")
        raw_thought = raw_thought[:end + 1] if 0 < end < 200 else raw_thought[:200]
    history_blocks.append({
        "turn": tn,
        "user": td["user"],
        "rec": rec_short,
        "asst": td["asst"],
        "reaction": reaction,
        "listener_thought": raw_thought,
    })
    art = get_track_artist(gold_tid)
    if art:
        artist_history.add(art)
    if progress_by_turn.get(tn) == "DOES_NOT_MOVE_TOWARD_GOAL":
        rejected_tids.add(gold_tid)


train_examples = build_examples(train_sessions)
valid_examples = build_examples(valid_sessions)

print(f"Train: {len(train_examples):,}  Valid: {len(valid_examples):,}")

# Apply weighted-positive subsampling
filtered_train = []
for ex in train_examples:
    if ex["weight"] >= 1.0 or random.random() < ex["weight"]:
        filtered_train.append(ex)
print(f"After subsampling (DOES_NOT @ 0.3): {len(filtered_train):,} train examples")

out_dir = Path(args.out_dir)
out_dir.mkdir(parents=True, exist_ok=True)

with open(out_dir / "train.jsonl", "w") as f:
    for ex in filtered_train:
        f.write(json.dumps(ex) + "\n")
with open(out_dir / "valid.jsonl", "w") as f:
    for ex in valid_examples:
        f.write(json.dumps(ex) + "\n")

print(f"Saved {len(filtered_train):,} examples to {out_dir / 'train.jsonl'}")
print(f"Saved {len(valid_examples):,} examples to {out_dir / 'valid.jsonl'}")

if filtered_train:
    ex0 = filtered_train[0]
    print(f"\nSample anchor (first 600 chars):\n{ex0['anchor'][:600]}\n")
    print(f"  positive: {ex0['positive'][:120]}")
    print(f"  weight: {ex0['weight']}  specificity: {ex0['specificity']}  turn: {ex0['turn']}")
    print(f"  anchor tokens: {count_tokens(ex0['anchor'])}")
