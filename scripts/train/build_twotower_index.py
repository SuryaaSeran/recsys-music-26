"""
Encode all tracks with the fine-tuned document encoder and save to disk.

Output: cache/twotower/track_embeddings.npy + track_ids.json

Usage:
    python scripts/build_twotower_index.py --model models/twotower_v1/final
"""
import argparse
import json
import numpy as np
from pathlib import Path

from datasets import load_dataset, concatenate_datasets
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument("--model", default="models/twotower_v1/final")
parser.add_argument("--out_dir", default="cache/twotower")
parser.add_argument("--batch_size", type=int, default=512)
args = parser.parse_args()

print("Loading track metadata...")
meta_ds = load_dataset("talkpl-ai/TalkPlayData-Challenge-Track-Metadata")
all_tracks = concatenate_datasets([meta_ds["all_tracks"], meta_ds["test_tracks"]])
metadata_dict = {row["track_id"]: row for row in all_tracks}


def get_track_text(tid: str) -> str:
    row = metadata_dict.get(tid, {})
    name = (row.get("track_name") or [""])[0]
    artist = (row.get("artist_name") or [""])[0]
    album = (row.get("album_name") or [""])[0]
    tags = " ".join((row.get("tag_list") or [])[:12])
    release_date = row.get("release_date") or ""
    year = str(release_date)[:4] if release_date else ""
    parts = [name]
    if artist:
        parts.append(f"by {artist}")
    if album:
        parts.append(f"| Album: {album}")
    if tags:
        parts.append(f"| Tags: {tags}")
    if year:
        parts.append(f"| {year}")
    return " ".join(parts).strip()


print(f"Loading model: {args.model}")
model = SentenceTransformer(args.model)

track_ids = list(metadata_dict.keys())
track_texts = [get_track_text(tid) for tid in track_ids]

print(f"Encoding {len(track_ids):,} tracks...")
embeddings = model.encode(
    track_texts,
    batch_size=args.batch_size,
    show_progress_bar=True,
    normalize_embeddings=True,
    convert_to_numpy=True,
)

out_dir = Path(args.out_dir)
out_dir.mkdir(parents=True, exist_ok=True)

np.save(out_dir / "track_embeddings.npy", embeddings)
with open(out_dir / "track_ids.json", "w") as f:
    json.dump(track_ids, f)

print(f"Saved {embeddings.shape} embeddings to {out_dir}")
print(f"Shape: {embeddings.shape}, dtype: {embeddings.dtype}")
