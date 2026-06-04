"""Build the per-modality float16 cache. Run once.

  python scripts/build_cache.py [--pca 256] [--force]
"""
import argparse
import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from src.rqvae.cache import build_cache

PCA_TARGETS = [
    "attributes-qwen3_embedding_0.6b",
    "lyrics-qwen3_embedding_0.6b",
    "metadata-qwen3_embedding_0.6b",
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default="data/cache")
    ap.add_argument("--pca", type=int, help="PCA dim applied to the 1024-d qwen3 modalities")
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()

    pca = {m: a.pca for m in PCA_TARGETS} if a.pca else None
    stats = build_cache(out_dir=a.out_dir, pca=pca, force=a.force)
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
