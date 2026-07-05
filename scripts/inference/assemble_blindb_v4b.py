"""Assemble Blind B v4b: v2 tracks (untouched LTR top-20, best real-blind 0.49)
+ v4 sharpened Opus responses (avg 78 words, LexDiv 0.752).

Rationale: every track-list manipulation regressed nDCG (4 confirmations:
max_per_artist, boost_named_track, pivot_suppress on dev-sim; v3 real blind
0.49 -> 0.48). The judge is text-only, so responses are the only positive-EV
lever. v4b isolates the pure judge upside with zero nDCG risk.
"""
import json
import math
from pathlib import Path
from collections import Counter

ROOT = Path(__file__).resolve().parents[2]

v2 = json.load(open(ROOT / "exp/inference/blind_b/blind_b_v8d_s3cap_v2.json"))
v4 = json.load(open(ROOT / "exp/inference/blind_b/blind_b_v8d_s3cap_v4.json"))
v4r = {x["session_id"]: x["predicted_response"] for x in v4}
v4t = {x["session_id"]: x["predicted_track_ids"] for x in v4}

out = []
n_rank1_diff = 0
for rec in v2:
    sid = rec["session_id"]
    assert sid in v4r, sid
    if v4t[sid][0] != rec["predicted_track_ids"][0]:
        n_rank1_diff += 1
    out.append({
        "session_id": sid,
        "user_id": rec["user_id"],
        "turn_number": rec["turn_number"],
        "predicted_track_ids": rec["predicted_track_ids"],
        "predicted_response": v4r[sid],
    })

assert len(out) == 80
for r in out:
    ids = r["predicted_track_ids"]
    assert len(ids) == 20 and len(set(ids)) == 20, r["session_id"]
    assert r["predicted_response"].strip(), r["session_id"]

json.dump(out, open(ROOT / "exp/inference/blind_b/blind_b_v8d_s3cap_v4b.json", "w"),
          indent=2, ensure_ascii=False)
print(f"v4b written. 80 records OK. rank1 differs v4 vs v2 in {n_rank1_diff} sessions")

wc = [len(r["predicted_response"].split()) for r in out]
print(f"resp words: mean={sum(wc)/len(wc):.1f} min={min(wc)} max={max(wc)}")

# CatalogDiversity: unique tracks / total slots
all_ids = [t for r in out for t in r["predicted_track_ids"]]
print(f"CatDiv: {len(set(all_ids))/len(all_ids):.4f} ({len(set(all_ids))} unique / {len(all_ids)})")

# LexicalDiversity: distinct-2 over responses (as used before)
def bigrams(s):
    toks = s.lower().split()
    return list(zip(toks, toks[1:]))
allb = [b for r in out for b in bigrams(r["predicted_response"])]
print(f"LexDiv (distinct-2): {len(set(allb))/len(allb):.4f}")

# distinct openers
openers = Counter(" ".join(r["predicted_response"].split()[:4]) for r in out)
print(f"distinct 4-word openers: {len(openers)}/80")
