"""
Build dense track embedding index using sentence-transformers.
Uses all-MiniLM-L6-v2 to embed track metadata (title + artist + tags).
Output: cache/dense/track_embeddings.npy and cache/dense/track_ids.json

Usage:
    python scripts/build_dense_index.py
"""
import json
import os
import numpy as np
from pathlib import Path
from datasets import load_dataset, concatenate_datasets
from sentence_transformers import SentenceTransformer

CACHE_DIR = Path("cache/dense")
MODEL_NAME = "all-MiniLM-L6-v2"

print(f"Loading model: {MODEL_NAME}...")
model = SentenceTransformer(MODEL_NAME)

print("Loading track metadata...")
meta_ds = load_dataset("talkpl-ai/TalkPlayData-Challenge-Track-Metadata")
all_tracks = concatenate_datasets([meta_ds["all_tracks"], meta_ds["test_tracks"]])

track_ids = []
sentences = []
for row in all_tracks:
    tid = row["track_id"]
    name = " ".join(row.get("track_name") or [])
    artist = " ".join(row.get("artist_name") or [])
    album = " ".join(row.get("album_name") or [])
    tags = " ".join(row.get("tag_list") or [])
    text = f"{name} {artist} {album} {tags}".strip()
    track_ids.append(tid)
    sentences.append(text)

print(f"Encoding {len(sentences):,} tracks...")
embeddings = model.encode(
    sentences,
    batch_size=512,
    show_progress_bar=True,
    normalize_embeddings=True,  # L2 normalize for cosine similarity
    convert_to_numpy=True,
)

print(f"Embedding shape: {embeddings.shape}")

CACHE_DIR.mkdir(parents=True, exist_ok=True)
np.save(CACHE_DIR / "track_embeddings.npy", embeddings)
with open(CACHE_DIR / "track_ids.json", "w") as f:
    json.dump(track_ids, f)

print(f"Saved to {CACHE_DIR}")
