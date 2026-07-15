"""Build a self-contained rerank-input file for an external GPU (e.g. Kaggle 8B).

Bakes in all local preprocessing so the remote reranker needs no HF datasets:
  - future-release prune (drop candidates released after session_date)
  - per-turn query (short history: only MOVES_TOWARD_GOAL plays, name by artist)
  - candidate docs (name, artist, album, tags, year) in LTR order

Output: list of
  {session_id, user_id, turn_number, session_date, query,
   candidates: [[track_id, doc], ...]}   # LTR order, future-pruned, top-N

Usage:
    python scripts/inference/build_rerank_input.py \
        --pred exp/inference/blind_a/blind_a_v8bh1h3_top100.json \
        --dataset talkpl-ai/TalkPlayData-Challenge-Blind-A --split test \
        --out exp/inference/blind_a/blind_a_rerank_input.json --keep 100
"""
import argparse, json
from datasets import load_dataset, concatenate_datasets

ap = argparse.ArgumentParser()
ap.add_argument("--pred", required=True, help="top-N candidate pred (emit_topk).")
ap.add_argument("--dataset", default="talkpl-ai/TalkPlayData-Challenge-Blind-A")
ap.add_argument("--split", default="test")
ap.add_argument("--out", required=True)
ap.add_argument("--keep", type=int, default=100, help="candidates kept per turn (post-prune).")
ap.add_argument("--max_hist_tracks", type=int, default=4)
args = ap.parse_args()

print("loading metadata...")
meta = load_dataset("talkpl-ai/TalkPlayData-Challenge-Track-Metadata")
md = {r["track_id"]: r for r in concatenate_datasets([meta["all_tracks"], meta["test_tracks"]])}


def cand_doc(tid):
    r = md.get(tid, {})
    name = (r.get("track_name") or ["?"])[0]
    artist = (r.get("artist_name") or ["?"])[0]
    album = (r.get("album_name") or [""])[0]
    tags = ", ".join((r.get("tag_list") or [])[:6])
    year = str(r.get("release_date") or "")[:4]
    parts = [f"{name} by {artist}"]
    if album: parts.append(f"album {album}")
    if tags: parts.append(f"tags {tags}")
    if year and year != "None": parts.append(year)
    return "; ".join(parts)


def hist_doc(tid):
    r = md.get(tid, {})
    return f"{(r.get('track_name') or ['?'])[0]} by {(r.get('artist_name') or ['?'])[0]}"


def release_date(tid):
    return str((md.get(tid, {}) or {}).get("release_date") or "")[:10]


print(f"loading {args.dataset}[{args.split}]...")
ds = load_dataset(args.dataset)[args.split]
session_map = {it["session_id"]: it for it in ds}


def build_query(item, turn_number):
    goal = (item.get("conversation_goal") or {}).get("listener_goal", "") or ""
    culture = (item.get("user_profile") or {}).get("preferred_musical_culture", "") or ""
    # Re-key by T-1: gpa at turn T judges the rec made at T-1.
    progress = {a["turn_number"] - 1: (a.get("goal_progress_assessment") or "")
                for a in (item.get("goal_progress_assessments") or [])}
    have_progress = any(v for v in progress.values())
    latest_user, played = "", []
    for turn in item.get("conversations") or []:
        if turn.get("turn_number") == turn_number and turn.get("role") == "music":
            break
        if turn.get("role") == "user":
            latest_user = turn.get("content", "") or latest_user
        elif turn.get("role") == "music" and turn.get("content"):
            tn = turn.get("turn_number")
            if have_progress and progress.get(tn) != "MOVES_TOWARD_GOAL":
                continue
            played.append(turn["content"])
    parts = []
    if latest_user: parts.append(f"Request: {latest_user}")
    if goal: parts.append(f"Goal: {goal}")
    if culture: parts.append(f"Preferred culture: {culture}")
    if played:
        recent = "; ".join(hist_doc(t) for t in played[-args.max_hist_tracks:])
        parts.append(f"Recently played: {recent}")
    return " | ".join(parts)


preds = json.load(open(args.pred))
out = []
n_pruned = 0
for p in preds:
    sid, tn = p["session_id"], p["turn_number"]
    item = session_map.get(sid, {})
    sdate = item.get("session_date", "") or ""
    cands = p["predicted_track_ids"]
    # future-release prune
    if sdate:
        kept = [t for t in cands if not (release_date(t) and release_date(t) > sdate)]
        n_pruned += len(cands) - len(kept)
        cands = kept
    cands = cands[: args.keep]
    out.append({
        "session_id": sid,
        "user_id": p.get("user_id", ""),
        "turn_number": tn,
        "session_date": sdate,
        "query": build_query(item, tn),
        "candidates": [[t, cand_doc(t)] for t in cands],
    })

with open(args.out, "w") as f:
    json.dump(out, f, ensure_ascii=False, indent=2)
print(f"Wrote {len(out)} turns to {args.out}")
print(f"future-release pruned: {n_pruned} ({n_pruned/max(1,len(out)):.1f}/turn)")
print(f"cand depth per turn: min={min(len(r['candidates']) for r in out)} "
      f"max={max(len(r['candidates']) for r in out)}")
print(f"sample query: {out[0]['query'][:160]}")
