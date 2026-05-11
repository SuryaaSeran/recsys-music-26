"""
Rebuild BM25 index with field weighting.

Changes from v1:
  - track_name repeated 3x  (boost exact title match)
  - artist_name repeated 2x (boost artist match)
  - album_name kept 1x
  - tags limited to top-20, each repeated 1x (reduce noise from 100+ tags)
  - release_year extracted from release_date
  - bm25s stopwords enabled

Output: cache/bm25/track_metadata_v2/

Usage:
    python scripts/train/build_bm25_v2.py
"""
import json
import os
from pathlib import Path

import bm25s
from datasets import load_dataset, concatenate_datasets
from tqdm import tqdm

OUT_DIR = "cache/bm25/track_metadata_v2"

print("Loading track metadata...")
meta_ds = load_dataset("talkpl-ai/TalkPlayData-Challenge-Track-Metadata")
all_tracks = concatenate_datasets([meta_ds["all_tracks"], meta_ds["test_tracks"]])
metadata_dict = {row["track_id"]: row for row in all_tracks}
track_ids = list(metadata_dict.keys())
print(f"Tracks: {len(track_ids):,}")


def stringify_v2(row: dict) -> str:
    name = (row.get("track_name") or [""])[0].strip()
    artist = (row.get("artist_name") or [""])[0].strip()
    album = (row.get("album_name") or [""])[0].strip()
    tags = (row.get("tag_list") or [])[:20]
    release_date = str(row.get("release_date") or "")
    year = release_date[:4] if len(release_date) >= 4 else ""

    parts = []
    # Field weighting via repetition
    if name:
        parts.extend([name, name, name])     # 3x
    if artist:
        parts.extend([artist, artist])       # 2x
    if album:
        parts.append(album)                  # 1x
    if year:
        parts.append(year)
    parts.extend(tags)                       # 1x each, top-20
    return " ".join(parts).lower()


print("Building corpus...")
corpus = [stringify_v2(metadata_dict[tid]) for tid in tqdm(track_ids)]

print("Tokenizing (with stopwords)...")
corpus_tokens = bm25s.tokenize(corpus, stopwords="en", show_progress=True)

print("Indexing...")
bm25_model = bm25s.BM25(k1=1.5, b=0.75)
bm25_model.index(corpus_tokens)

Path(OUT_DIR).mkdir(parents=True, exist_ok=True)
bm25_model.save(OUT_DIR)
with open(f"{OUT_DIR}/track_ids.json", "w") as f:
    json.dump(track_ids, f)

print(f"Saved BM25 v2 index to {OUT_DIR}")
print("Done.")
