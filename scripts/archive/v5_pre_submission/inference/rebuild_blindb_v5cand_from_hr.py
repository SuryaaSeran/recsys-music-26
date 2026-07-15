"""Reconstruct the submission-format prediction file for the v5-cand (no-bucket)
Blind B candidate from the corrected human-readable file, after a validation
pass (done in another session) re-ranked 4 sessions and tightened 1 response.

The human-readable file only carries title/artist/year/genres, not track_id, so
track_ids are recovered by matching (title, artist) back against the ORIGINAL
per-session candidate pool (scoped per session — the pool never changes here,
only the order and the response text). This also acts as a structural check:
if a title/artist pair does not resolve to exactly one track_id within that
session's original 20, the script aborts instead of guessing.

Input:
  /Users/stealthmacmini/.claude/uploads/95ce8712-e6ed-46cb-814b-7ef94e493b61/
    ee1b0804-BLIND_B_v5cand_predictions_human_readable_1.json  (corrected, 80 sessions)
  exp/inference/blind_b/blind_b_v8d_tier1_nobucket_resp.json    (original, track_ids)

Output:
  exp/inference/blind_b/blind_b_v8d_tier1_nobucket_resp_v2.json
"""
import json
from pathlib import Path
from collections import Counter

ROOT = Path(__file__).resolve().parents[2]

HR_PATH = Path("/Users/stealthmacmini/.claude/uploads/95ce8712-e6ed-46cb-814b-7ef94e493b61/"
               "ee1b0804-BLIND_B_v5cand_predictions_human_readable_1.json")
ORIG_PATH = ROOT / "exp/inference/blind_b/blind_b_v8d_tier1_nobucket_resp.json"
OUT_PATH = ROOT / "exp/inference/blind_b/blind_b_v8d_tier1_nobucket_resp_v2.json"

meta_ds = None  # loaded lazily only if needed for tie-breaking


def load_meta():
    from datasets import load_dataset, concatenate_datasets
    meta_ds = load_dataset("talkpl-ai/TalkPlayData-Challenge-Track-Metadata")
    splits = [meta_ds["all_tracks"]]
    if "test_tracks" in meta_ds:
        splits.append(meta_ds["test_tracks"])
    all_tracks = concatenate_datasets(splits) if len(splits) > 1 else splits[0]
    return {r["track_id"]: r for r in all_tracks}


hr = json.load(open(HR_PATH))
orig = json.load(open(ORIG_PATH))
orig_by_sid = {r["session_id"]: r for r in orig}
hr_by_sid = {s["session_id"]: s for s in hr}

assert set(hr_by_sid) == set(orig_by_sid), "session set mismatch"

meta = None
out = []
rank1_changed = []
resp_changed = []
order_changed_count = 0

for sid, orig_r in orig_by_sid.items():
    hr_s = hr_by_sid[sid]
    orig_ids = orig_r["predicted_track_ids"]

    # Build (title, artist) -> [track_id,...] map scoped to this session's
    # original 20 candidates (lazy meta load only on first use).
    if meta is None:
        meta = load_meta()
    key_to_ids = {}
    for tid in orig_ids:
        r = meta.get(tid, {})
        name = (r.get("track_name") or ["?"])[0]
        artist = (r.get("artist_name") or ["?"])[0]
        key_to_ids.setdefault((name, artist), []).append(tid)

    new_ids = []
    for i, t in enumerate(hr_s["predicted_tracks_top20"]):
        key = (t["title"], t["artist"])
        cands = key_to_ids.get(key)
        assert cands, f"{sid}: rank {i+1} title/artist {key!r} not found in original pool"
        assert len(cands) == 1, f"{sid}: rank {i+1} title/artist {key!r} ambiguous: {cands}"
        new_ids.append(cands[0])

    assert len(new_ids) == 20 and len(set(new_ids)) == 20, sid
    assert set(new_ids) == set(orig_ids), f"{sid}: track set changed, not just reordered"

    if new_ids != orig_ids:
        order_changed_count += 1
    if new_ids[0] != orig_ids[0]:
        rank1_changed.append(sid)

    new_resp = hr_s["generated_response"]
    if new_resp.strip() != orig_r["predicted_response"].strip():
        resp_changed.append(sid)

    out.append({
        "session_id": sid,
        "user_id": orig_r["user_id"],
        "turn_number": orig_r["turn_number"],
        "predicted_track_ids": new_ids,
        "predicted_response": new_resp,
    })

assert len(out) == 80
json.dump(out, open(OUT_PATH, "w"), indent=2, ensure_ascii=False)
print(f"wrote {OUT_PATH} (80 records)")
print(f"sessions with reordered track list: {order_changed_count}")
print(f"sessions with rank-1 changed: {len(rank1_changed)} -> {rank1_changed}")
print(f"sessions with response text changed: {len(resp_changed)} -> {resp_changed}")

wc = [len(r["predicted_response"].split()) for r in out]
print(f"resp words: mean={sum(wc)/len(wc):.1f} min={min(wc)} max={max(wc)}")

all_ids = [t for r in out for t in r["predicted_track_ids"]]
print(f"CatDiv: {len(set(all_ids))/len(all_ids):.4f} ({len(set(all_ids))} unique / {len(all_ids)})")


def bigrams(s):
    toks = s.lower().split()
    return list(zip(toks, toks[1:]))


allb = [b for r in out for b in bigrams(r["predicted_response"])]
print(f"LexDiv (distinct-2): {len(set(allb))/len(allb):.4f}")

openers = Counter(" ".join(r["predicted_response"].split()[:4]) for r in out)
print(f"distinct 4-word openers: {len(openers)}/80")
