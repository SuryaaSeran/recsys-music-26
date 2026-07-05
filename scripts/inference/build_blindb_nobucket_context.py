"""Build per-session context for hand-writing responses on the no-bucket
Blind B candidate config (exp/inference/blind_b/blind_b_v8d_tier1_nobucket.json).

Same style target as v2/v4: 4-5 substantive sentences, ~78 words, 2-3 concrete
metadata evidence points (tag/album/year tied to the request), continuity
callback to a prior played track when history exists, honest framing when the
exact requested track is not the top pick. COLD sessions: anchor personalization
to the request + history only, no profile/demographic references.

Output: exp/inference/blind_b/blindb_nobucket_context.json
"""
import json
from pathlib import Path
from datasets import load_dataset, concatenate_datasets

ROOT = Path(__file__).resolve().parents[2]

preds = json.load(open(ROOT / "exp/inference/blind_b/blind_b_v8d_tier1_nobucket.json"))
sessions_ds = load_dataset("talkpl-ai/TalkPlayData-Challenge-Blind-B")["test"]
sessions = {s["session_id"]: s for s in sessions_ds}

meta_ds = load_dataset("talkpl-ai/TalkPlayData-Challenge-Track-Metadata")
_meta_splits = [meta_ds["all_tracks"]]
if "test_tracks" in meta_ds:
    _meta_splits.append(meta_ds["test_tracks"])
all_tracks = concatenate_datasets(_meta_splits) if len(_meta_splits) > 1 else _meta_splits[0]
meta = {r["track_id"]: r for r in all_tracks}


def track_line(tid):
    r = meta.get(tid, {})
    name = (r.get("track_name") or ["?"])[0]
    artist = (r.get("artist_name") or ["?"])[0]
    album = (r.get("album_name") or [""])[0]
    year = str(r.get("release_date", "") or "")[:4]
    tags = ", ".join((r.get("tag_list") or [])[:6])
    return {"track_id": tid, "name": name, "artist": artist, "album": album,
            "year": year, "tags": tags}


out = []
for p in preds:
    sid = p["session_id"]
    s = sessions[sid]
    conv = s["conversations"]
    turns = [{"role": c["role"], "text": c["content"]} for c in conv]
    cold = not s.get("user_id")
    profile = s.get("user_profile") if not cold else None
    top20 = [track_line(t) for t in p["predicted_track_ids"]]
    out.append({
        "session_id": sid,
        "turn_number": p["turn_number"],
        "cold": cold,
        "user_profile": profile,
        "conversation": turns,
        "top20": top20,
    })

outp = ROOT / "exp/inference/blind_b/blindb_nobucket_context.json"
json.dump(out, open(outp, "w"), indent=2, ensure_ascii=False)
print(f"wrote {outp}  ({len(out)} sessions, {sum(1 for o in out if o['cold'])} cold)")
