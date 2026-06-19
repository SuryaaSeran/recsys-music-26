"""Option A: centroid-matching bucket recall (zero training).

For each L0 bucket, compute its centroid = mean of v8e track embeddings of
member tracks. At each dev turn, encode the conversation query with the v8e
two-tower, cosine-match to the 64 centroids, take top-k buckets.

Measures bucket recall@k = fraction of dev turns where the gold track's L0
bucket is in the top-k predicted buckets. Target: >90% @ k=3.

Compares directly against SASRec baseline (run eval_bucket_recall.py --mode sasrec).

Usage:
  python scripts/inference/eval_bucket_recall_centroid.py \
    --tt_model models/twotower_v8e/final \
    --emb cache/twotower_v8e/track_embeddings.npy \
    --tids cache/twotower_v8e/track_ids.json \
    --sids_dir cache/semantic_ids/runF_v8e_L2C64 \
    --n_sessions 1000 --top_k 3
"""
import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from datasets import load_dataset
from tqdm import tqdm

ap = argparse.ArgumentParser()
ap.add_argument("--tt_model",   default="models/twotower_v8e/final")
ap.add_argument("--emb",        default="cache/twotower_v8e/track_embeddings.npy")
ap.add_argument("--tids",       default="cache/twotower_v8e/track_ids.json")
ap.add_argument("--sids_dir",   default="cache/semantic_ids/runF_v8e_L2C64")
ap.add_argument("--n_sessions", type=int, default=1000)
ap.add_argument("--top_k",      type=int, default=3)
ap.add_argument("--use_moves_only", action="store_true", default=True,
                help="(unused here; centroid uses query not history)")
args = ap.parse_args()

# ── Load embeddings + codebook ────────────────────────────────────────────────
print("Loading v8e embeddings + codebook...")
emb  = np.load(args.emb)                       # (N, 768), L2-normalised
with open(args.tids) as f:
    emb_tids = json.load(f)
tid_to_idx = {t: i for i, t in enumerate(emb_tids)}

sids_dir = Path(args.sids_dir)
codes  = np.load(sids_dir / "semantic_ids.npy")
ctids  = np.load(sids_dir / "track_ids.npy", allow_pickle=True).tolist()
tid_to_l0 = {t: int(c[0]) for t, c in zip(ctids, codes)}

# ── Compute 64 L0 centroids in v8e space ──────────────────────────────────────
print("Computing L0 centroids...")
bucket_vecs = defaultdict(list)
for t, l0 in tid_to_l0.items():
    if t in tid_to_idx:
        bucket_vecs[l0].append(emb[tid_to_idx[t]])

n_buckets = max(bucket_vecs) + 1
centroids = np.zeros((n_buckets, emb.shape[1]), dtype=np.float32)
for l0, vecs in bucket_vecs.items():
    c = np.mean(vecs, axis=0)
    c = c / (np.linalg.norm(c) + 1e-8)         # re-normalise centroid
    centroids[l0] = c
print(f"  {n_buckets} centroids, dim {centroids.shape[1]}")

# ── Track-id → short text (for REC: slot) ─────────────────────────────────────
from datasets import concatenate_datasets
print("Loading track metadata for anchor REC slots...")
meta_ds = load_dataset("talkpl-ai/TalkPlayData-Challenge-Track-Metadata")
all_tracks = concatenate_datasets([meta_ds["all_tracks"], meta_ds["test_tracks"]])
metadata = {row["track_id"]: row for row in all_tracks}
def track_short(tid):
    row = metadata.get(tid, {})
    name   = (row.get("track_name")  or [""])[0]
    artist = (row.get("artist_name") or [""])[0]
    return f"{name} – {artist}" if name and artist else (name or artist or "")

# ── Load v8e query encoder ────────────────────────────────────────────────────
print(f"Loading v8e two-tower: {args.tt_model}")
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer
tt = SentenceTransformer(args.tt_model)
tok = AutoTokenizer.from_pretrained("intfloat/multilingual-e5-base")
def count_tokens(text): return len(tok.encode(text, add_special_tokens=False))

MAX_TOK = 510
V8D_REACTION = {"MOVES_TOWARD_GOAL": "liked", "DOES_NOT_MOVE_TOWARD_GOAL": "rejected"}

def build_anchor_v8d(profile, goal_text, specificity, music_hist_v8d,
                     music_labels, text_history, current_query, count_fn,
                     user_thought_history=None):
    """Inlined copy of the v8e anchor builder (T1-protected, LISTENER slot)."""
    parts = [str(profile.get(k, "")) for k in
             ("age_group", "country_code", "gender", "culture", "language")]
    profile_line = "[PROFILE] " + " · ".join(p for p in parts if p)
    goal_line = f"[GOAL] {goal_text}".strip()
    if specificity:
        goal_line += f"  ({specificity})"
    now_line = f"[NOW] USER: {current_query}".strip()
    core = profile_line + "\n" + goal_line + "\n" + now_line
    budget = MAX_TOK - count_fn("query: " + core)

    blocks = []
    for i, (tid, tn) in enumerate(music_hist_v8d):
        rec = track_short(tid)
        rxn = V8D_REACTION.get(music_labels[i] if i < len(music_labels) else "", "unknown")
        umsg = text_history[2*i]   if 2*i   < len(text_history) else ""
        amsg = text_history[2*i+1] if 2*i+1 < len(text_history) else ""
        lt = (user_thought_history[2*i] if user_thought_history and 2*i < len(user_thought_history) else "") or ""
        if lt:
            end = lt.find(". ")
            lt = lt[:end+1] if 0 < end < 200 else lt[:200]
        blocks.append({"turn": tn, "user": umsg, "rec": rec, "asst": amsg, "reaction": rxn, "lt": lt})

    def cands(hb):
        ls = f" | LISTENER: {hb['lt']}" if hb["lt"] else ""
        return (f"[T{hb['turn']}] USER: {hb['user']} | REC: {hb['rec']} | ASST: {hb['asst']} | REACTION: {hb['reaction']}{ls}",
                f"[T{hb['turn']}] USER: {hb['user']} | REC: {hb['rec']} | REACTION: {hb['reaction']}{ls}",
                f"[T{hb['turn']}] USER: {hb['user']} | REC: {hb['rec']} | REACTION: {hb['reaction']}")

    def insert(hb, bud):
        for c in cands(hb):
            cost = count_fn("\n" + c)
            if bud >= cost:
                return c, bud - cost
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

# ── Load dev sessions ─────────────────────────────────────────────────────────
print(f"Loading {args.n_sessions} dev sessions...")
ds = load_dataset("talkpl-ai/TalkPlayData-Challenge-Dataset", split="test")
sessions = list(ds)[:args.n_sessions]

REACTION = {"MOVES_TOWARD_GOAL": "liked", "DOES_NOT_MOVE_TOWARD_GOAL": "rejected"}

# ── Build all query anchors first (batch encode) ─────────────────────────────
print("Building query anchors...")
anchors, golds = [], []

for item in tqdm(sessions, desc="sessions"):
    prof_raw = item.get("user_profile", {}) or {}
    profile = {
        "age_group":    prof_raw.get("age_group", "") or "",
        "country_code": prof_raw.get("country_code", "") or "",
        "gender":       prof_raw.get("gender", "") or "",
        "culture":      prof_raw.get("preferred_musical_culture", "") or "",
        "language":     prof_raw.get("preferred_language", "") or "",
    }
    goal = item.get("conversation_goal", {}) or {}
    goal_text = goal.get("listener_goal", "") or ""
    spec      = goal.get("specificity", "") or ""

    progress = {a["turn_number"] - 1: a["goal_progress_assessment"]
                for a in (item.get("goal_progress_assessments") or [])}

    turn_data = {}
    for t in item["conversations"]:
        tn = t["turn_number"]
        slot = turn_data.setdefault(tn, {"user": "", "music": "", "asst": "", "thought": ""})
        if t["role"] == "user":
            slot["user"] = t["content"] or ""
            slot["thought"] = t.get("thought") or ""
        elif t["role"] == "music":
            slot["music"] = t["content"] or ""
        elif t["role"] == "assistant":
            slot["asst"] = t["content"] or ""

    music_hist, music_turns, music_labels = [], [], []
    text_hist, thought_hist = [], []

    for tn in sorted(turn_data):
        td = turn_data[tn]
        gold_tid = td["music"]
        # Append user text/thought before processing this turn's query
        if td["user"]:
            text_hist.append(td["user"])
            thought_hist.append(td["thought"])
        if td["asst"]:
            text_hist.append(td["asst"])
            thought_hist.append("")

        if gold_tid and gold_tid in tid_to_l0:
            latest_user = td["user"]
            mh = list(zip(music_hist, music_turns))
            anchor = build_anchor_v8d(
                profile, goal_text, spec, mh, music_labels,
                text_hist[:-1] if text_hist else [], latest_user, count_tokens,
                user_thought_history=thought_hist[:-1] if thought_hist else [],
            )
            anchors.append(anchor)
            golds.append(tid_to_l0[gold_tid])

        if gold_tid:
            music_hist.append(gold_tid)
            music_turns.append(tn)
            music_labels.append(progress.get(tn, ""))

print(f"  {len(anchors)} query turns")

# ── Encode + match ────────────────────────────────────────────────────────────
print("Encoding queries...")
q_emb = tt.encode(anchors, batch_size=32, normalize_embeddings=True,
                  show_progress_bar=True)

print("Matching to centroids...")
sims = q_emb @ centroids.T                      # (Q, 64)
topk_buckets = np.argsort(-sims, axis=1)[:, :args.top_k]

# ── Metrics ───────────────────────────────────────────────────────────────────
correct, recall_k = 0, 0
per_bucket_total   = defaultdict(int)
per_bucket_hit     = defaultdict(int)
for i, gold in enumerate(golds):
    pred = topk_buckets[i].tolist()
    if pred[0] == gold:    correct  += 1
    if gold in pred:       recall_k += 1
    per_bucket_total[gold] += 1
    if gold in pred:       per_bucket_hit[gold] += 1

total = len(golds)
print(f"\n=== Centroid Bucket Recall (top-{args.top_k}) ===")
print(f"  Top-1 accuracy: {correct}/{total} = {100*correct/total:.1f}%")
print(f"  Recall @ {args.top_k}:  {recall_k}/{total} = {100*recall_k/total:.1f}%")
print(f"  (Target: >90% recall @ {args.top_k})")

# Recall at higher k for reference
for k in (5, 8, 10):
    topk = np.argsort(-sims, axis=1)[:, :k]
    rk = sum(1 for i, g in enumerate(golds) if g in topk[i].tolist())
    print(f"  Recall @ {k}: {100*rk/total:.1f}%")

worst = sorted(per_bucket_total, key=lambda b: per_bucket_hit.get(b,0)/per_bucket_total[b])[:5]
print(f"\n  Worst-5 buckets by recall@{args.top_k}:")
for b in worst:
    r = 100 * per_bucket_hit.get(b,0) / per_bucket_total[b]
    print(f"    bucket {b:>2}: {r:.1f}%  ({per_bucket_total[b]} turns)")
