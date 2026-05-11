
"""
Build Semantic IDs from pre-computed track embeddings.

Input:
  data/track_embeddings.npy
  data/track_ids.txt

Output:
  data/codebook.pkl
"""

import argparse
import pickle
from pathlib import Path

import numpy as np
import yaml
from loguru import logger
from sklearn.cluster import MiniBatchKMeans


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="config/train.yaml")
    return p.parse_args()


def load_config(path):
    with open(path) as f:
        return yaml.safe_load(f)


def build_codebook(embeddings: np.ndarray, n_coarse: int, n_fine: int):
    logger.info(f"Fitting level-1 KMeans: {n_coarse} clusters on {len(embeddings):,} tracks...")

    km1 = MiniBatchKMeans(
        n_clusters=n_coarse,
        random_state=42,
        batch_size=4096,
        n_init=3,
        max_iter=100,
        reassignment_ratio=0.01,
        verbose=0,
    )
    codes1 = km1.fit_predict(embeddings)

    logger.info(f"Fitting level-2 KMeans: {n_fine} clusters on residuals...")

    residuals = embeddings - km1.cluster_centers_[codes1]

    km2 = MiniBatchKMeans(
        n_clusters=n_fine,
        random_state=42,
        batch_size=4096,
        n_init=3,
        max_iter=100,
        reassignment_ratio=0.01,
        verbose=0,
    )
    codes2 = km2.fit_predict(residuals)

    return km1, km2, codes1, codes2


def main():
    args = parse_args()
    cfg = load_config(args.config)

    emb_path = Path(cfg["embedding_path"])
    id_path = Path(cfg["track_ids_path"])

    logger.info(f"Loading embeddings from {emb_path}...")
    embeddings = np.load(emb_path).astype(np.float32)

    logger.info(f"Loading track IDs from {id_path}...")
    with open(id_path) as f:
        track_ids = [line.strip() for line in f if line.strip()]

    assert len(track_ids) == len(embeddings), (
        f"Mismatch: {len(track_ids)} track IDs vs {len(embeddings)} embeddings"
    )

    logger.info(f"Loaded {len(track_ids):,} tracks, embedding dim={embeddings.shape[1]}")

    logger.info("Normalizing embeddings...")
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    embeddings = embeddings / (norms + 1e-8)

    km1, km2, codes1, codes2 = build_codebook(
        embeddings,
        n_coarse=cfg["n_coarse"],
        n_fine=cfg["n_fine"],
    )

    track_to_codes = {
        tid: (int(c1), int(c2))
        for tid, c1, c2 in zip(track_ids, codes1, codes2)
    }

    codes_to_tracks = {}
    for tid, pair in track_to_codes.items():
        codes_to_tracks.setdefault(pair, []).append(tid)

    valid_coarse = set(int(x) for x in codes1.tolist())
    valid_pairs = set(track_to_codes.values())

    codebook = {
        "km1": km1,
        "km2": km2,
        "track_to_codes": track_to_codes,
        "codes_to_tracks": codes_to_tracks,
        "valid_coarse": valid_coarse,
        "valid_pairs": valid_pairs,
        "n_coarse": cfg["n_coarse"],
        "n_fine": cfg["n_fine"],
    }

    save_path = Path(cfg["codebook_save_path"])
    save_path.parent.mkdir(parents=True, exist_ok=True)

    with open(save_path, "wb") as f:
        pickle.dump(codebook, f)

    logger.success(f"Codebook saved to {save_path}")
    logger.info(f"Unique coarse codes used: {len(valid_coarse)}/{cfg['n_coarse']}")
    logger.info(f"Unique (coarse, fine) pairs: {len(valid_pairs)}")

    avg_bucket = len(track_ids) / len(valid_pairs)
    logger.info(f"Avg tracks per bucket: {avg_bucket:.2f}")

    bucket_sizes = [len(v) for v in codes_to_tracks.values()]
    logger.info(f"Min bucket size: {min(bucket_sizes)}")
    logger.info(f"Max bucket size: {max(bucket_sizes)}")


if __name__ == "__main__":
    main()
