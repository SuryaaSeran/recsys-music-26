"""Build a human-readable version of a Blind B prediction file: track IDs
resolved to title/artist/year/genres, plus full conversation + response, so
results can be checked (e.g. from a different session) without querying the
track metadata dataset directly.

Usage:
  python scripts/inference/build_human_readable_submission.py <pred.json> <out.json>

Schema per session (matches wiki/BLIND_B_v1_predictions_human_readable.json):
  session_id, turn_number, cold_start, conversation, generated_response,
  predicted_tracks_top20 [{rank, title, artist, year, genres}]
"""
import json
import sys
from pathlib import Path
from datasets import load_dataset, concatenate_datasets

ROOT = Path(__file__).resolve().parents[2]

pred_path = Path(sys.argv[1])
out_path = Path(sys.argv[2])

preds = json.load(open(pred_path))

sessions_ds = load_dataset("talkpl-ai/TalkPlayData-Challenge-Blind-B")["test"]
sessions = {s["session_id"]: s for s in sessions_ds}

meta_ds = load_dataset("talkpl-ai/TalkPlayData-Challenge-Track-Metadata")
_meta_splits = [meta_ds["all_tracks"]]
if "test_tracks" in meta_ds:
    _meta_splits.append(meta_ds["test_tracks"])
all_tracks = concatenate_datasets(_meta_splits) if len(_meta_splits) > 1 else _meta_splits[0]
meta = {r["track_id"]: r for r in all_tracks}


def track_info(rank, tid):
    r = meta.get(tid, {})
    return {
        "rank": rank,
        "title": (r.get("track_name") or ["?"])[0],
        "artist": (r.get("artist_name") or ["?"])[0],
        "year": str(r.get("release_date", "") or "")[:4],
        "genres": (r.get("tag_list") or [])[:10],
    }


out = []
for p in preds:
    sid = p["session_id"]
    s = sessions[sid]
    cold = not s.get("user_id")
    conv = [{"role": c["role"], "text": c["content"]} for c in s["conversations"]]
    top20 = [track_info(i + 1, tid) for i, tid in enumerate(p["predicted_track_ids"])]
    out.append({
        "session_id": sid,
        "turn_number": p["turn_number"],
        "cold_start": cold,
        "conversation": conv,
        "generated_response": p["predicted_response"],
        "predicted_tracks_top20": top20,
    })

out_path.parent.mkdir(parents=True, exist_ok=True)
json.dump(out, open(out_path, "w"), indent=2, ensure_ascii=False)
print(f"wrote {out_path}  ({len(out)} sessions)")
