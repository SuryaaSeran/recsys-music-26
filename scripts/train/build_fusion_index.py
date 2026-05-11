"""
Build numpy caches for fusion inference from HuggingFace embedding datasets.

Outputs:
  cache/cf_bpr/track_{ids,embeddings}  - 128-dim CF-BPR track embeddings
  cache/clap/track_{ids,embeddings}    - 512-dim LAION CLAP audio embeddings
  cache/qwen3_meta/track_{ids,embeddings} - 1024-dim Qwen3 metadata embeddings
  cache/qwen3_attr/track_{ids,embeddings} - 1024-dim Qwen3 attributes embeddings
  cache/user_cf_bpr.json              - {user_id: [128-dim vector]}
"""
import json
import numpy as np
from pathlib import Path
from datasets import load_dataset, concatenate_datasets

print("Loading Track Embeddings dataset...")
te = load_dataset("talkpl-ai/TalkPlayData-Challenge-Track-Embeddings")
all_te = concatenate_datasets([te["all_tracks"], te["test_tracks"]])
print(f"  {len(all_te)} tracks")

track_ids = [row["track_id"] for row in all_te]

print("Extracting embeddings (filling missing with zeros)...")

def safe_emb(rows, key, dim):
    out = np.zeros((len(rows), dim), dtype=np.float32)
    for i, row in enumerate(rows):
        v = row[key]
        if v and len(v) == dim:
            out[i] = v
    return out

cf_bpr      = safe_emb(all_te, "cf-bpr",                           128)
clap        = safe_emb(all_te, "audio-laion_clap",                 512)
qwen_meta   = safe_emb(all_te, "metadata-qwen3_embedding_0.6b",   1024)
qwen_attr   = safe_emb(all_te, "attributes-qwen3_embedding_0.6b", 1024)
qwen_lyrics = safe_emb(all_te, "lyrics-qwen3_embedding_0.6b",     1024)

print(f"  cf_bpr:       {cf_bpr.shape}")
print(f"  clap:         {clap.shape}")
print(f"  qwen_meta:    {qwen_meta.shape}")
print(f"  qwen_attr:    {qwen_attr.shape}")
print(f"  qwen_lyrics:  {qwen_lyrics.shape}")

def normalize(x: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    return x / np.maximum(norms, 1e-8)

cf_bpr      = normalize(cf_bpr)
clap        = normalize(clap)
qwen_meta   = normalize(qwen_meta)
qwen_attr   = normalize(qwen_attr)
qwen_lyrics = normalize(qwen_lyrics)

for name, emb in [
    ("cf_bpr",       cf_bpr),
    ("clap",         clap),
    ("qwen3_meta",   qwen_meta),
    ("qwen3_attr",   qwen_attr),
    ("qwen3_lyrics", qwen_lyrics),
]:
    out = Path(f"cache/{name}")
    out.mkdir(parents=True, exist_ok=True)
    np.save(out / "track_embeddings.npy", emb)
    with open(out / "track_ids.json", "w") as f:
        json.dump(track_ids, f)
    print(f"Saved cache/{name}: {emb.shape}")

print("\nLoading User Embeddings dataset...")
ue = load_dataset("talkpl-ai/TalkPlayData-Challenge-User-Embeddings")
all_ue = concatenate_datasets([ue["train"], ue["test_warm"], ue["test_cold"]])
print(f"  {len(all_ue)} users")

user_cf = {row["user_id"]: row["cf-bpr"] for row in all_ue}
with open("cache/user_cf_bpr.json", "w") as f:
    json.dump(user_cf, f)
print(f"Saved cache/user_cf_bpr.json: {len(user_cf)} users")

print("\nDone.")
