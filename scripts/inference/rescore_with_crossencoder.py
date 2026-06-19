"""Cross-encoder pointwise rescorer over an existing predictions JSON.

Re-orders the top-K candidates per turn using a pretrained or fine-tuned
cross-encoder (default: cross-encoder/ms-marco-MiniLM-L-12-v2). Output JSON
keeps the same schema; only predicted_track_ids is reshuffled inside the
top-K. The tail (positions K..) is preserved.

Usage:
    python scripts/inference/rescore_with_crossencoder.py \\
        --pred exp/inference/devset/phase_a_ltr_retrained.json \\
        --model cross-encoder/ms-marco-MiniLM-L-12-v2 \\
        --sessions 50
"""
import argparse
import json
from pathlib import Path

import numpy as np
from datasets import load_dataset, concatenate_datasets
from sentence_transformers import CrossEncoder
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument("--pred", required=True)
parser.add_argument("--out", default=None)
parser.add_argument("--model", default="cross-encoder/ms-marco-MiniLM-L-12-v2")
parser.add_argument("--dataset", default="talkpl-ai/TalkPlayData-Challenge-Dataset")
parser.add_argument("--split", default="test")
parser.add_argument("--top_k", type=int, default=20)
parser.add_argument("--sessions", type=int, default=0)
parser.add_argument("--batch_size", type=int, default=32)
parser.add_argument("--blend", type=float, default=0.0,
                    help="If >0, final score = blend*ltr_rank_sig + (1-blend)*ce_score. "
                         "Preserves some LTR signal.")
parser.add_argument("--query_template", default="default", choices=["default", "multi_turn", "v8d"],
                    help="default = legacy v2 anchor; multi_turn = [TURN-N] (CE v3); "
                         "v8d = role-tagged anchor with reactions+listener (CE v4).")
args = parser.parse_args()

out_path = args.out or args.pred.replace(".json", "_ce.json")

_ce_maxlen = 512 if args.query_template == "v8d" else 256
print(f"Loading cross-encoder: {args.model} (max_length={_ce_maxlen})")
ce = CrossEncoder(args.model, max_length=_ce_maxlen)

print("Loading track metadata...")
meta_ds = load_dataset("talkpl-ai/TalkPlayData-Challenge-Track-Metadata")
all_tracks = concatenate_datasets([meta_ds["all_tracks"], meta_ds["test_tracks"]])
metadata_dict = {row["track_id"]: row for row in all_tracks}


def get_candidate_text(tid: str) -> str:
    # Matches build_crossencoder_v4_data.get_track_text (top-12 tags + year)
    row = metadata_dict.get(tid, {})
    name   = (row.get("track_name")  or ["?"])[0]
    artist = (row.get("artist_name") or ["?"])[0]
    album  = (row.get("album_name")  or [""])[0]
    tags   = " ".join((row.get("tag_list") or [])[:12])
    release = row.get("release_date") or ""
    year = str(release)[:4] if release else ""
    parts = [name, f"by {artist}"]
    if album: parts.append(f"| Album: {album}")
    if tags:  parts.append(f"| Tags: {tags}")
    if year:  parts.append(f"| {year}")
    return " ".join(parts)


def get_track_name_artist(tid: str) -> str:
    row = metadata_dict.get(tid, {})
    name   = (row.get("track_name")  or ["?"])[0]
    artist = (row.get("artist_name") or ["?"])[0]
    return f'"{name}" by {artist}'


print(f"Loading {args.dataset} [{args.split}]...")
ds = load_dataset(args.dataset)[args.split]
session_map = {item["session_id"]: item for item in ds}

print(f"Loading predictions: {args.pred}")
preds = json.load(open(args.pred))

if args.sessions > 0:
    keep: set = set()
    filt = []
    for p in preds:
        if p["session_id"] not in keep:
            if len(keep) >= args.sessions:
                continue
            keep.add(p["session_id"])
        filt.append(p)
    preds = filt
    print(f"  Restricted to first {len(keep)} sessions ({len(preds)} turns).")


def build_query(session: dict, turn_number: int) -> str:
    """Build the rescorer query: goal + culture + last-4 user/assistant text
    + last-2 played tracks (compact). Mirrors the inference TT query shape."""
    goal = (session.get("conversation_goal") or {}).get("listener_goal", "")
    culture = (session.get("user_profile") or {}).get("preferred_musical_culture", "")
    music_history: list[str] = []
    text_history:  list[str] = []
    for turn in session.get("conversations") or []:
        if turn.get("turn_number") == turn_number and turn.get("role") == "music":
            break
        role = turn.get("role")
        if role == "music":
            music_history.append(turn["content"])
        elif role in ("user", "assistant"):
            text_history.append(turn["content"])

    parts = []
    latest_user = text_history[-1] if text_history else ""
    if latest_user: parts.append(latest_user)
    if goal:    parts.append(f"Goal: {goal}")
    if culture: parts.append(culture)
    for tid in music_history[-2:]:
        na = get_track_name_artist(tid)
        if na: parts.append(na)
    for t in text_history[-3:-1]:  # 2 history msgs before the latest user
        parts.append(t)
    return " ".join(parts)


def build_query_multiturn(session: dict, turn_number: int) -> str:
    """Multi-turn query matching CE v3 training format:
    [TURN-3] {oldest user turn} [TURN-2] {middle} [TURN-1] {most recent}
    Goal: ... Culture: ...
    Older user turns are left empty if fewer than 3 exist."""
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
    parts = [
        f"[TURN-3] {last3[0]}".strip(),
        f"[TURN-2] {last3[1]}".strip(),
        f"[TURN-1] {last3[2]}".strip(),
    ]
    if goal:    parts.append(f"Goal: {goal}")
    if culture: parts.append(f"Culture: {culture}")
    return " ".join(parts).strip()


_V8D_REACTION = {"MOVES_TOWARD_GOAL": "liked", "DOES_NOT_MOVE_TOWARD_GOAL": "rejected"}
_v8d_tok = None

def build_query_v8d(session: dict, turn_number: int) -> str:
    """v8d role-tagged anchor matching build_crossencoder_v4_data.py / inference."""
    global _v8d_tok
    if _v8d_tok is None:
        from transformers import AutoTokenizer
        _v8d_tok = AutoTokenizer.from_pretrained("intfloat/multilingual-e5-base")
    def ctok(t): return len(_v8d_tok.encode(t, add_special_tokens=False))

    prof = session.get("user_profile") or {}
    parts = [str(prof.get(k, "") or "") for k in
             ("age_group", "country_code", "gender",
              "preferred_musical_culture", "preferred_language")]
    profile_line = "[PROFILE] " + " · ".join(p for p in parts if p)
    goal_obj = session.get("conversation_goal") or {}
    goal = goal_obj.get("listener_goal", "") or ""
    spec = goal_obj.get("specificity", "") or ""
    goal_line = f"[GOAL] {goal}".strip() + (f"  ({spec})" if spec else "")

    progress = {a["turn_number"] - 1: a["goal_progress_assessment"]
                for a in (session.get("goal_progress_assessments") or [])}

    mh, mturns, mlabels, text_hist, thought_hist = [], [], [], [], []
    latest_user = ""
    for turn in session.get("conversations") or []:
        if turn.get("turn_number") == turn_number and turn.get("role") == "music":
            break
        role = turn.get("role")
        if role == "music" and turn.get("content"):
            mh.append(turn["content"]); mturns.append(turn["turn_number"])
            mlabels.append(progress.get(turn["turn_number"], ""))
        elif role in ("user", "assistant"):
            text_hist.append(turn.get("content") or "")
            thought_hist.append((turn.get("thought") or "") if role == "user" else "")
            if role == "user":
                latest_user = turn.get("content") or ""

    now_line = f"[NOW] USER: {latest_user}".strip()
    core = profile_line + "\n" + goal_line + "\n" + now_line
    budget = 510 - ctok("query: " + core)
    th = thought_hist[:-1]

    blocks = []
    for i, (tid, tn) in enumerate(zip(mh, mturns)):
        rec = get_track_name_artist(tid).strip('"')
        rxn = _V8D_REACTION.get(mlabels[i] if i < len(mlabels) else "", "unknown")
        umsg = text_hist[2*i]   if 2*i   < len(text_hist) - 1 else ""
        amsg = text_hist[2*i+1] if 2*i+1 < len(text_hist) - 1 else ""
        lt = (th[2*i] if 2*i < len(th) else "") or ""
        if lt:
            end = lt.find(". "); lt = lt[:end+1] if 0 < end < 200 else lt[:200]
        blocks.append({"turn": tn, "user": umsg, "rec": rec, "asst": amsg, "reaction": rxn, "lt": lt})

    def cands(hb):
        ls = f" | LISTENER: {hb['lt']}" if hb["lt"] else ""
        return (f"[T{hb['turn']}] USER: {hb['user']} | REC: {hb['rec']} | ASST: {hb['asst']} | REACTION: {hb['reaction']}{ls}",
                f"[T{hb['turn']}] USER: {hb['user']} | REC: {hb['rec']} | REACTION: {hb['reaction']}{ls}",
                f"[T{hb['turn']}] USER: {hb['user']} | REC: {hb['rec']} | REACTION: {hb['reaction']}")
    def insert(hb, bud):
        for c in cands(hb):
            cost = ctok("\n" + c)
            if bud >= cost: return c, bud - cost
        return None, bud

    added_rest = []
    if blocks:
        first, rest = blocks[0], blocks[1:]
        for hb in reversed(rest):
            txt, budget = insert(hb, budget)
            if txt: added_rest.append(txt)
        added_rest.reverse()
        ftxt, budget = insert(first, budget)
        added = ([ftxt] if ftxt else []) + added_rest
    else:
        added = []
    return "query: " + "\n".join([profile_line, goal_line] + added + [now_line])


def make_query(session, turn_number):
    if args.query_template == "v8d":
        return build_query_v8d(session, turn_number)
    if args.query_template == "multi_turn":
        return build_query_multiturn(session, turn_number)
    return build_query(session, turn_number)


# ── Rerank loop ──────────────────────────────────────────────────────────────
results = []
order_changes = 0
for p in tqdm(preds, desc="rerank"):
    sid = p["session_id"]
    top_tids = p["predicted_track_ids"][: args.top_k]
    session = session_map.get(sid)
    if session is None or len(top_tids) < 2:
        results.append(p); continue

    query = make_query(session, p["turn_number"])
    cand_texts = [get_candidate_text(t) for t in top_tids]
    pairs = [(query, c) for c in cand_texts]
    scores = ce.predict(pairs, batch_size=args.batch_size, show_progress_bar=False)
    scores = np.asarray(scores, dtype=np.float32)

    if args.blend > 0.0:
        ltr_sig = np.array(
            [1.0 / np.log2(i + 2) for i in range(len(top_tids))], dtype=np.float32
        )
        s_min, s_max = scores.min(), scores.max()
        if s_max - s_min > 1e-6:
            scores_norm = (scores - s_min) / (s_max - s_min)
        else:
            scores_norm = np.zeros_like(scores)
        final = args.blend * ltr_sig + (1 - args.blend) * scores_norm
    else:
        final = scores

    order = np.argsort(-final).tolist()
    new_top = [top_tids[i] for i in order]
    if new_top != list(top_tids):
        order_changes += 1
    tail = p["predicted_track_ids"][args.top_k:]
    new_full = new_top + [t for t in tail if t not in new_top]
    results.append({
        "session_id": sid, "user_id": p["user_id"],
        "turn_number": p["turn_number"],
        "predicted_track_ids": new_full,
        "predicted_response":  p.get("predicted_response", ""),
    })

print(f"Order changes: {order_changes}/{len(preds)} ({100*order_changes/max(len(preds),1):.1f}%)")
Path(out_path).parent.mkdir(parents=True, exist_ok=True)
json.dump(results, open(out_path, "w"), ensure_ascii=False, indent=2)
print(f"Saved {len(results)} predictions to {out_path}")
