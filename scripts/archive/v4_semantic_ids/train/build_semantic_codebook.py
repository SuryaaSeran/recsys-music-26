"""
Build K-means semantic codebook from fine-tuned two-tower track embeddings.

Clusters are text-predictable (built from music retrieval embeddings, not audio/CF).
Target: K=200-500 clusters, ~100-250 tracks/cluster.

Usage:
    python scripts/build_semantic_codebook.py --k 300 --model_dir cache/twotower_v3
"""
import argparse
import json
import numpy as np
from pathlib import Path
from sklearn.cluster import MiniBatchKMeans

parser = argparse.ArgumentParser()
parser.add_argument("--k", type=int, default=300)
parser.add_argument("--model_dir", default="cache/twotower_v3")
parser.add_argument("--out_dir", default=None)
parser.add_argument("--seed", type=int, default=42)
args = parser.parse_args()

out_dir = args.out_dir or f"cache/semantic_codebook_k{args.k}"

print(f"Loading embeddings from {args.model_dir}...")
embeddings = np.load(f"{args.model_dir}/track_embeddings.npy")
with open(f"{args.model_dir}/track_ids.json") as f:
    track_ids = json.load(f)
print(f"  Embeddings: {embeddings.shape}")

print(f"Running MiniBatchKMeans K={args.k}...")
km = MiniBatchKMeans(n_clusters=args.k, random_state=args.seed, batch_size=4096, n_init=5, verbose=0)
km.fit(embeddings)
assignments = km.labels_  # (n_tracks,)
centroids = km.cluster_centers_  # (K, 384)

# Normalize centroids for cosine similarity
norms = np.linalg.norm(centroids, axis=1, keepdims=True)
norms = np.where(norms == 0, 1, norms)
centroids = centroids / norms

# Build cluster → track_ids map
cluster_to_tracks = {i: [] for i in range(args.k)}
for idx, cluster_id in enumerate(assignments):
    cluster_to_tracks[int(cluster_id)].append(track_ids[idx])

track_to_cluster = {track_ids[idx]: int(cid) for idx, cid in enumerate(assignments)}

# Stats
sizes = [len(v) for v in cluster_to_tracks.values()]
sizes.sort()
n = len(sizes)
print(f"Cluster size distribution: min={sizes[0]}, median={sizes[n//2]}, p90={sizes[int(n*0.9)]}, max={sizes[-1]}, mean={np.mean(sizes):.0f}")
print(f"Empty clusters: {sum(1 for s in sizes if s == 0)}")

Path(out_dir).mkdir(parents=True, exist_ok=True)
np.save(f"{out_dir}/centroids.npy", centroids)
np.save(f"{out_dir}/assignments.npy", assignments)
with open(f"{out_dir}/cluster_to_tracks.json", "w") as f:
    json.dump(cluster_to_tracks, f)
with open(f"{out_dir}/track_to_cluster.json", "w") as f:
    json.dump(track_to_cluster, f)
with open(f"{out_dir}/meta.json", "w") as f:
    json.dump({"k": args.k, "n_tracks": len(track_ids), "model_dir": args.model_dir}, f)

print(f"Saved codebook to {out_dir}/")
print(f"  centroids.npy: {centroids.shape}")
print(f"  cluster_to_tracks: {args.k} clusters")
