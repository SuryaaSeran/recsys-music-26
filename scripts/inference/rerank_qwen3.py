"""Last-stage reranker using Qwen3-Reranker (cross-encoder) over LTR top-K.

Reads a prediction file whose `predicted_track_ids` hold the LTR-ranked top-N
(N >= rerank_k; produce with run_inference --emit_topk 100). For each turn it
rebuilds the conversation query, scores the top-K candidates with the
Qwen3 cross-encoder, reorders them, and writes a fresh top-20 prediction file.
Candidates beyond K keep their LTR order and are appended after the reranked head.

Focus metric: nDCG@20. Responses are passed through untouched.

Usage:
    python scripts/inference/rerank_qwen3.py \
        --pred exp/inference/devset/<topk100>.json \
        --out  exp/inference/devset/<topk100>_qwen3rr.json \
        --rerank_k 50 --model Qwen/Qwen3-Reranker-0.6B

Then score:
    python scripts/inference/evaluate_local.py --pred <out>.json
"""
import argparse
import json
import time
from pathlib import Path

import torch
from datasets import load_dataset, concatenate_datasets
from sentence_transformers import CrossEncoder
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument("--pred", required=True, help="Input pred with top-N candidate IDs.")
parser.add_argument("--out", default=None)
parser.add_argument("--model", default="Qwen/Qwen3-Reranker-4B",
                    help="Qwen/Qwen3-Reranker-4B (8GB, default) or -0.6B (1.2GB, safe). "
                         "8B (16GB) does not fit 16GB RAM.")
parser.add_argument("--rerank_k", type=int, default=50,
                    help="Rerank the top-K LTR candidates per turn (50/75/100).")
parser.add_argument("--final_k", type=int, default=20,
                    help="Length of output predicted_track_ids (eval uses 20).")
parser.add_argument("--dataset", default="talkpl-ai/TalkPlayData-Challenge-Dataset")
parser.add_argument("--split", default="test",
                    help="Dataset split holding session context. The dev eval set "
                         "is the 'test' split (1000 sessions).")
parser.add_argument("--batch_size", type=int, default=32)
parser.add_argument("--max_hist_tracks", type=int, default=4,
                    help="Recent played tracks to include in the query context.")
parser.add_argument("--hist_doc_mode", default="full", choices=["full", "short", "none"],
                    help="How to render 'Recently played' history in the query: "
                         "full=name/album/tags/year, short=name by artist, none=omit. "
                         "Long history slows the model and hurt turn-8 nDCG.")
parser.add_argument("--alpha", type=float, default=0.0,
                    help="Blend weight on the LTR rank prior. "
                         "final = alpha*ltr_rank_score + (1-alpha)*norm_rerank_score. "
                         "alpha=0 -> pure rerank (most aggressive); 1.0 -> keep LTR order.")
parser.add_argument("--save_scores", default="",
                    help="If set, write per-candidate rerank scores to this JSON so "
                         "alpha/rerank_k can be swept offline without rerunning the model.")
parser.add_argument("--limit", type=int, default=0)
args = parser.parse_args()

out_path = args.out or args.pred.replace(".json", "_qwen3rr.json")

# Music-domain reranking instruction. Qwen3-Reranker injects this into its chat
# template via the prompt mechanism; a task-specific instruction beats the
# default web-search instruction for this domain.
INSTRUCTION = (
    "Given a user's music listening request and the conversation so far, "
    "rank the candidate track by how well it satisfies the request and "
    "continues the listening session."
)

# ── Models / catalog ─────────────────────────────────────────────────────────
print("Loading track metadata...")
meta_ds = load_dataset("talkpl-ai/TalkPlayData-Challenge-Track-Metadata")
all_tracks = concatenate_datasets([meta_ds["all_tracks"], meta_ds["test_tracks"]])
metadata_dict = {row["track_id"]: row for row in all_tracks}


def track_doc(tid: str) -> str:
    """Full candidate document: name, artist, album, tags, year."""
    row = metadata_dict.get(tid, {})
    name = (row.get("track_name") or ["?"])[0]
    artist = (row.get("artist_name") or ["?"])[0]
    album = (row.get("album_name") or [""])[0]
    tags = ", ".join((row.get("tag_list") or [])[:6])
    year = str(row.get("release_date") or "")[:4]
    parts = [f"{name} by {artist}"]
    if album:
        parts.append(f"album {album}")
    if tags:
        parts.append(f"tags {tags}")
    if year and year != "None":
        parts.append(year)
    return "; ".join(parts)


def hist_doc(tid: str) -> str:
    """Short history rendering: name by artist only (keeps query compact)."""
    row = metadata_dict.get(tid, {})
    name = (row.get("track_name") or ["?"])[0]
    artist = (row.get("artist_name") or ["?"])[0]
    return f"{name} by {artist}"


print(f"Loading {args.dataset} [{args.split}]...")
ds = load_dataset(args.dataset)[args.split]
session_map = {item["session_id"]: item for item in ds}


def session_date_of(session_id: str) -> str:
    item = session_map.get(session_id)
    return (item or {}).get("session_date", "") or ""


def prune_future_tracks(cand_ids: list, session_date: str) -> list:
    """Drop candidates whose release_date is after the session date (causality fix).
    Tracks with unknown/missing release_date are kept (cannot verify)."""
    if not session_date:
        return cand_ids
    kept = []
    for t in cand_ids:
        rd = str((metadata_dict.get(t, {}) or {}).get("release_date") or "")[:10]
        if rd and rd > session_date:
            continue  # released after the session happened -> impossible answer
        kept.append(t)
    return kept


def build_query(session_id: str, turn_number: int) -> str:
    """Rebuild the conversation query up to (not including) the target music turn.

    The "Recently played" context only includes tracks whose turn was assessed
    MOVES_TOWARD_GOAL (positive signal). If no progress assessments exist
    (e.g. blind sessions), all played tracks are included.
    """
    item = session_map.get(session_id)
    if item is None:
        return ""
    goal = (item.get("conversation_goal") or {}).get("listener_goal", "") or ""
    culture = (item.get("user_profile") or {}).get("preferred_musical_culture", "") or ""

    # Per-turn progress map: turn_number -> assessment string
    progress = {a["turn_number"]: (a.get("goal_progress_assessment") or "")
                for a in (item.get("goal_progress_assessments") or [])}
    have_progress = any(v for v in progress.values())

    latest_user = ""
    played: list[str] = []
    for turn in item.get("conversations") or []:
        if turn.get("turn_number") == turn_number and turn.get("role") == "music":
            break
        role = turn.get("role", "")
        if role == "user":
            latest_user = turn.get("content", "") or latest_user
        elif role == "music" and turn.get("content"):
            tn = turn.get("turn_number")
            # Only keep positively-received plays as context (when we have labels)
            if have_progress and progress.get(tn) != "MOVES_TOWARD_GOAL":
                continue
            played.append(turn["content"])
    parts = []
    if latest_user:
        parts.append(f"Request: {latest_user}")
    if goal:
        parts.append(f"Goal: {goal}")
    if culture:
        parts.append(f"Preferred culture: {culture}")
    if played and args.hist_doc_mode != "none":
        render = track_doc if args.hist_doc_mode == "full" else hist_doc
        recent = "; ".join(render(t) for t in played[-args.max_hist_tracks:])
        parts.append(f"Recently played: {recent}")
    return " | ".join(parts)


print(f"Loading model {args.model}...")
_t0 = time.time()
device = "mps" if torch.backends.mps.is_available() else "cpu"
model = CrossEncoder(
    args.model,
    prompts={"music_rerank": INSTRUCTION},
    default_prompt_name="music_rerank",
    device=device,
    model_kwargs={"torch_dtype": torch.float16} if device == "mps" else {},
)
print(f"  loaded in {time.time() - _t0:.0f}s on {device}")

preds = json.load(open(args.pred))
if args.limit > 0:
    preds = preds[: args.limit]
print(f"Reranking {len(preds)} turns, top-{args.rerank_k} each...")

# ── Rerank loop ──────────────────────────────────────────────────────────────
results = []
score_log = {}
n_changed = 0
n_pruned_total = 0
for p in tqdm(preds, desc="rerank"):
    sid = p["session_id"]
    tn = p["turn_number"]
    cand_ids = p.get("predicted_track_ids") or []

    # Causality prune: drop tracks released after the session date.
    _sdate = session_date_of(sid)
    _before = len(cand_ids)
    cand_ids = prune_future_tracks(cand_ids, _sdate)
    n_pruned_total += _before - len(cand_ids)

    head = cand_ids[: args.rerank_k]
    tail = cand_ids[args.rerank_k:]

    if len(head) <= 1:
        new_ids = cand_ids[: args.final_k]
    else:
        query = build_query(sid, tn)
        pairs = [(query, track_doc(t)) for t in head]
        scores = [float(s) for s in
                  model.predict(pairs, batch_size=args.batch_size, show_progress_bar=False)]

        # Blend rerank score with the LTR rank prior.
        #   ltr_s: position-based prior, 1.0 for rank-0 down to ~0 for rank K-1
        #   rr_s : min-max normalized rerank score in [0,1]
        n = len(head)
        lo, hi = min(scores), max(scores)
        span = (hi - lo) or 1.0
        blended = []
        for i in range(n):
            ltr_s = (n - i) / n
            rr_s = (scores[i] - lo) / span
            blended.append(args.alpha * ltr_s + (1.0 - args.alpha) * rr_s)
        order = sorted(range(n), key=lambda i: blended[i], reverse=True)
        reranked_head = [head[i] for i in order]
        if reranked_head[: args.final_k] != head[: args.final_k]:
            n_changed += 1
        new_ids = (reranked_head + tail)[: args.final_k]

        if args.save_scores:
            score_log[f"{sid}|{tn}"] = [[head[i], scores[i]] for i in range(n)]

    results.append({
        "session_id": sid,
        "user_id": p.get("user_id", ""),
        "turn_number": tn,
        "predicted_track_ids": new_ids,
        "predicted_response": p.get("predicted_response", ""),
    })

Path(out_path).parent.mkdir(parents=True, exist_ok=True)
with open(out_path, "w") as f:
    json.dump(results, f, ensure_ascii=False, indent=2)
print(f"\nSaved {len(results)} preds to {out_path}")
print(f"Top-{args.final_k} order changed on {n_changed}/{len(results)} turns "
      f"({100*n_changed/max(1,len(results)):.0f}%)")
print(f"Future-release candidates pruned: {n_pruned_total} "
      f"({n_pruned_total/max(1,len(results)):.1f} per turn)")

if args.save_scores:
    with open(args.save_scores, "w") as f:
        json.dump(score_log, f)
    print(f"Saved per-candidate rerank scores to {args.save_scores} "
          f"({len(score_log)} turns) -- sweep alpha/rerank_k offline.")
