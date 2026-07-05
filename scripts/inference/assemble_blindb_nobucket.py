"""Assemble the no-bucket Blind B candidate submission: merge the 4 hand-written
response batches onto exp/inference/blind_b/blind_b_v8d_tier1_nobucket.json.

Output: exp/inference/blind_b/blind_b_v8d_tier1_nobucket_resp.json
"""
import json
from pathlib import Path
from collections import Counter

ROOT = Path(__file__).resolve().parents[2]

preds = json.load(open(ROOT / "exp/inference/blind_b/blind_b_v8d_tier1_nobucket.json"))

resp = {}
for b in range(4):
    for r in json.load(open(f"/tmp/blindb_nobucket_resp_batch{b}.json")):
        resp[r["session_id"]] = r["predicted_response"]

out = []
for p in preds:
    sid = p["session_id"]
    assert sid in resp, f"missing response for {sid}"
    ids = p["predicted_track_ids"]
    assert len(ids) == 20 and len(set(ids)) == 20, sid
    assert resp[sid].strip(), sid
    out.append({
        "session_id": sid,
        "user_id": p["user_id"],
        "turn_number": p["turn_number"],
        "predicted_track_ids": ids,
        "predicted_response": resp[sid],
    })

assert len(out) == 80
outp = ROOT / "exp/inference/blind_b/blind_b_v8d_tier1_nobucket_resp.json"
json.dump(out, open(outp, "w"), indent=2, ensure_ascii=False)
print(f"wrote {outp}  (80 records)")

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

excl = sum(1 for r in out if "!" in r["predicted_response"])
print(f"responses with exclamation marks: {excl}")
