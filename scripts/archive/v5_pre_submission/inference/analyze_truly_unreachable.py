"""Profile the 391 'truly_unreachable' turns from the recall-gap diagnostic.

Reads exp/analysis/recall_gap_full.json, filters to bucket==truly_unreachable,
enriches each record with TRAIN-side stats (play counts, cooccur successor
presence, artist catalog size, metadata coverage, BM25 token overlap) and
prints aggregate tables.
"""
import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import json
import re
from collections import Counter
from pathlib import Path

import numpy as np
from datasets import load_dataset, concatenate_datasets
from tqdm import tqdm


RECORDS_PATH = "exp/analysis/recall_gap_full.json"
OUT_PATH     = "exp/analysis/truly_unreachable_features.json"
COOCCUR_PATH = "cache/cooccur/next_song_leakfree.npz"

# ── Load records ──────────────────────────────────────────────────────────────
print(f"Loading {RECORDS_PATH}")
recs_all = json.load(open(RECORDS_PATH))
hard = [r for r in recs_all if r["bucket"] == "truly_unreachable"]
print(f"  total unreachable: {len(recs_all)}")
print(f"  truly_unreachable: {len(hard)}")
assert len(hard) == 391, f"expected 391 hard records, got {len(hard)}"

# ── Catalog metadata ──────────────────────────────────────────────────────────
print("Loading track metadata...")
meta_ds = load_dataset("talkpl-ai/TalkPlayData-Challenge-Track-Metadata")
all_tracks = concatenate_datasets([meta_ds["all_tracks"], meta_ds["test_tracks"]])
meta = {row["track_id"]: row for row in all_tracks}

artist_buckets: dict[str, list] = {}
for _tid, _row in meta.items():
    for _a in (_row.get("artist_name") or []):
        k = _a.strip().lower()
        if k:
            artist_buckets.setdefault(k, []).append(_tid)
print(f"  artists: {len(artist_buckets):,}  tracks: {len(meta):,}")

# ── TRAIN stats ───────────────────────────────────────────────────────────────
print("Loading TRAIN sessions for play counts...")
train = load_dataset("talkpl-ai/TalkPlayData-Challenge-Dataset", split="train")
play_counts: Counter = Counter()
artist_play_counts: Counter = Counter()
for s in tqdm(train, desc="train"):
    for t in s["conversations"]:
        if t["role"] == "music":
            tid = t["content"]
            play_counts[tid] += 1
            for a in (meta.get(tid, {}).get("artist_name") or []):
                artist_play_counts[a.strip().lower()] += 1
print(f"  unique tracks played in TRAIN: {len(play_counts):,}")

# ── Cooccur successor set ─────────────────────────────────────────────────────
print(f"Loading cooccur table {COOCCUR_PATH}")
co = np.load(COOCCUR_PATH, allow_pickle=True)
co_ids = co["track_ids"]
co_neigh = co["neigh_ids"]
co_id2idx = {str(t): i for i, t in enumerate(co_ids)}
successor_set: set[int] = set()
for row in co_neigh:
    for nidx in row:
        if nidx >= 0:
            successor_set.add(int(nidx))
print(f"  tracks ever appearing as a successor: {len(successor_set):,} / {len(co_ids):,}")

# ── BM25 query reconstruction ─────────────────────────────────────────────────
print("Loading dev sessions (for BM25 query reconstruction)...")
dev = load_dataset("talkpl-ai/TalkPlayData-Challenge-Dataset", split="test")
sess_by_id = {s["session_id"]: s for s in dev}

_TOK = re.compile(r"[a-z0-9]+")
def tokens(text: str) -> set:
    return set(_TOK.findall((text or "").lower()))

def get_track_text(tid: str) -> str:
    row = meta.get(tid, {})
    name   = (row.get("track_name")  or [""])[0]
    artist = (row.get("artist_name") or [""])[0]
    tags   = " ".join(row.get("tag_list") or [])
    return f"{name} {artist} {tags}".strip()

def build_bm25_query(session, current_turn_number):
    goal     = session.get("conversation_goal", {}).get("listener_goal", "")
    culture  = session.get("user_profile", {}).get("preferred_musical_culture", "")
    music_history, text_history = [], []
    for t in session["conversations"]:
        if t.get("turn_number") == current_turn_number and t["role"] == "music":
            break
        if t["role"] == "music":
            music_history.append(t["content"])
        elif t["role"] in ("user", "assistant"):
            text_history.append(t["content"])
    parts = [goal, culture]
    for tid in music_history[-4:]:
        parts.append(get_track_text(tid))
    parts.extend(text_history[-4:])
    return " ".join(p for p in parts if p)

# ── Enrich ────────────────────────────────────────────────────────────────────
print("Enriching records...")
for r in tqdm(hard, desc="enrich"):
    tid = r["gold"]
    row = meta.get(tid, {})
    artists = [a.strip().lower() for a in (row.get("artist_name") or [])]
    r["train_plays"] = play_counts.get(tid, 0)
    r["artist_train_plays_max"] = max((artist_play_counts.get(a, 0) for a in artists), default=0)
    r["catalog_artist_size_max"] = max((len(artist_buckets.get(a, [])) for a in artists), default=0)
    r["appears_as_successor"] = (
        co_id2idx.get(tid) is not None and co_id2idx[tid] in successor_set
    )
    r["tag_count"] = len(row.get("tag_list") or [])
    r["has_album"] = bool((row.get("album_name") or [""])[0])
    r["has_year"]  = bool(row.get("release_year"))
    r["in_catalog"] = tid in meta

    sess = sess_by_id.get(r["session_id"])
    if sess is not None:
        q = build_bm25_query(sess, r["turn_number"])
        gtxt = get_track_text(tid)
        qt, gt = tokens(q), tokens(gtxt)
        r["bm25_token_overlap"] = len(qt & gt)
        r["bm25_query_tokens"]  = len(qt)
        r["gold_meta_tokens"]   = len(gt)
        r["goal_category"]  = sess.get("conversation_goal", {}).get("goal_category", "")
        r["goal_specificity"] = sess.get("conversation_goal", {}).get("goal_specificity", "")
        r["culture"]        = sess.get("user_profile", {}).get("preferred_musical_culture", "")
        r["age_group"]      = sess.get("user_profile", {}).get("age_group", "")
        r["country"]        = sess.get("user_profile", {}).get("country", "")

# ── Aggregate report ──────────────────────────────────────────────────────────
n = len(hard)
print(f"\n{'='*60}")
print(f"TRULY_UNREACHABLE PATTERN ANALYSIS  ({n} turns)")
print(f"{'='*60}")

def pct(c): return f"{c}/{n} ({100*c/n:.1f}%)"

# Catalog presence
print("\n── Catalog / index presence ─────────────────────────────────")
print(f"  In catalog metadata: {pct(sum(1 for r in hard if r['in_catalog']))}")
print(f"  Cold catalog (0 TRAIN plays): {pct(sum(1 for r in hard if r['train_plays']==0))}")
print(f"  Never a successor in cooccur: {pct(sum(1 for r in hard if not r['appears_as_successor']))}")
print(f"  Singleton artist (catalog size <= 1): {pct(sum(1 for r in hard if r['catalog_artist_size_max']<=1))}")
print(f"  Artist with 0 TRAIN plays: {pct(sum(1 for r in hard if r['artist_train_plays_max']==0))}")

# Train plays distribution
plays = np.array([r["train_plays"] for r in hard])
all_plays = np.array(list(play_counts.values()))
print("\n── TRAIN play counts ────────────────────────────────────────")
print(f"  hard:    p10={int(np.percentile(plays,10))}  p50={int(np.percentile(plays,50))}  p90={int(np.percentile(plays,90))}  mean={plays.mean():.1f}")
print(f"  TRAIN catalog (played at least once): p50={int(np.percentile(all_plays,50))}  p90={int(np.percentile(all_plays,90))}")
for thr in [0, 1, 2, 5, 10, 50]:
    c = (plays <= thr).sum()
    print(f"  train_plays <= {thr:>2}: {pct(int(c))}")

# Popularity
pops = np.array([r["gold_popularity"] for r in hard])
all_pops = np.array([float(row.get("popularity") or 0.0) for row in meta.values()])
print("\n── Popularity ────────────────────────────────────────────────")
print(f"  hard:    p10={np.percentile(pops,10):.1f}  p50={np.percentile(pops,50):.1f}  p90={np.percentile(pops,90):.1f}  mean={pops.mean():.1f}")
print(f"  catalog: p10={np.percentile(all_pops,10):.1f}  p50={np.percentile(all_pops,50):.1f}  p90={np.percentile(all_pops,90):.1f}  mean={all_pops.mean():.1f}")
print(f"  popularity == 0: {pct(int((pops==0).sum()))}")

# Metadata coverage
print("\n── Metadata coverage ────────────────────────────────────────")
print(f"  No tags (tag_count==0): {pct(sum(1 for r in hard if r['tag_count']==0))}")
print(f"  No album: {pct(sum(1 for r in hard if not r['has_album']))}")
print(f"  No release year: {pct(sum(1 for r in hard if not r['has_year']))}")
tag_counts = np.array([r["tag_count"] for r in hard])
print(f"  tag_count p10/50/90: {int(np.percentile(tag_counts,10))}/{int(np.percentile(tag_counts,50))}/{int(np.percentile(tag_counts,90))}")

# Lexical overlap
print("\n── BM25 lexical overlap (query tokens ∩ gold metadata tokens) ─")
overlaps = np.array([r.get("bm25_token_overlap", 0) for r in hard])
print(f"  overlap p10/50/90: {int(np.percentile(overlaps,10))}/{int(np.percentile(overlaps,50))}/{int(np.percentile(overlaps,90))}")
print(f"  overlap == 0: {pct(int((overlaps==0).sum()))}")
print(f"  overlap <= 1: {pct(int((overlaps<=1).sum()))}")

# Goal / culture distribution
def show_dist(name, key, top=8):
    cnt = Counter(r.get(key, "") for r in hard)
    overall = Counter()
    for s in dev:
        for t in s["conversations"]:
            if t["role"] == "music":
                overall[s.get("conversation_goal", {}).get(key, "") if key.startswith("goal") or key=="goal_specificity"
                        else s.get("user_profile", {}).get(key, "")] += 1
    print(f"\n  ── {name} ──")
    total_dev = sum(overall.values())
    for k, c in cnt.most_common(top):
        ov = overall.get(k, 0)
        hard_share = c/n
        dev_share = ov/total_dev if total_dev else 0
        ratio = hard_share/dev_share if dev_share else float("inf")
        print(f"    {k!r:<30} hard={c:>3} ({100*hard_share:4.1f}%)  dev={100*dev_share:4.1f}%  ratio={ratio:.2f}")

show_dist("Goal category", "goal_category")
show_dist("Goal specificity", "goal_specificity")
show_dist("Culture", "culture")
show_dist("Country", "country")
show_dist("Age group", "age_group")

# Save enriched records
Path(OUT_PATH).parent.mkdir(parents=True, exist_ok=True)
json.dump(hard, open(OUT_PATH, "w"), indent=2)
print(f"\nSaved enriched records: {OUT_PATH}")
