"""
Build a coarse-level retrieval index: for each coarse code, list all track IDs
sorted by their number of conversations (popularity).

Output: data/coarse_retrieval_index.json
  { "0": ["track_id1", "track_id2", ...], ... }  # tracks sorted by popularity
"""
import json
import pickle
from collections import Counter, defaultdict
from pathlib import Path

from datasets import load_dataset

with open("data/codebook_v2.pkl", "rb") as f:
    cb = pickle.load(f)

track_to_codes = cb["track_to_codes"]
codes_to_tracks = cb["codes_to_tracks"]

# Count how many conversations each track appears in (as a proxy for popularity)
print("Counting track popularity from conversation dataset...")
ds = load_dataset("talkpl-ai/TalkPlayData-Challenge-Dataset")["train"]
track_popularity = Counter()

for row in ds:
    for turn in row["conversations"]:
        if turn.get("role") == "music":
            tid = str(turn.get("content", "")).strip()
            track_popularity[tid] += 1

print(f"Total unique tracks in conversations: {len(track_popularity)}")

# Build coarse index: coarse_code -> sorted list of track IDs
coarse_to_tracks = defaultdict(list)
for track_id, (c, f) in track_to_codes.items():
    coarse_to_tracks[c].append(track_id)

# Sort each coarse bucket by popularity (descending)
coarse_index = {}
for c in range(128):
    tracks = coarse_to_tracks.get(c, [])
    tracks_sorted = sorted(tracks, key=lambda t: track_popularity.get(t, 0), reverse=True)
    coarse_index[str(c)] = tracks_sorted

# Stats
sizes = [len(v) for v in coarse_index.values()]
print(f"Coarse codes with tracks: {sum(1 for s in sizes if s > 0)}/128")
print(f"Avg tracks/coarse: {sum(sizes)/len(sizes):.0f}")
print(f"Min/Max tracks/coarse: {min(sizes)}/{max(sizes)}")

out = Path("data/coarse_retrieval_index.json")
with open(out, "w") as f:
    json.dump(coarse_index, f)
print(f"Saved to {out}")
