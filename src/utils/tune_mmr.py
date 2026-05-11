"""
Utility: Tune MMR lambda on the Dev set.

Sweeps lambda from 0.0 to 1.0 and reports nDCG@20 + catalog diversity
for each value. Pick the lambda that gives the best combined score.

Usage:
    # First generate dev predictions with lambda=1.0 (no diversity):
    python src/infer/run_inference.py --split dev --mmr_lambda 1.0

    # Then tune:
    python src/utils/tune_mmr.py \
        --predictions exp/predictions_dev.json \
        --ground_truth data/ground_truth_dev.json \
        --embeddings data/TalkPlayData-2/track_embeddings.npy \
        --track_ids data/TalkPlayData-2/track_ids.txt
"""

import argparse
import json

import numpy as np
from loguru import logger

from src.infer.mmr import mmr_rerank


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--predictions",  required=True)
    p.add_argument("--ground_truth", required=True)
    p.add_argument("--embeddings",   required=True)
    p.add_argument("--track_ids",    required=True)
    p.add_argument("--lambdas",      nargs="+", type=float,
                   default=[0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
    return p.parse_args()


def ndcg_at_k(recommended: list[str], relevant: set[str], k: int = 20) -> float:
    dcg, idcg = 0.0, 0.0
    for i, tid in enumerate(recommended[:k]):
        if tid in relevant:
            dcg += 1.0 / np.log2(i + 2)
    for i in range(min(k, len(relevant))):
        idcg += 1.0 / np.log2(i + 2)
    return dcg / idcg if idcg > 0 else 0.0


def catalog_coverage(all_recommendations: list[list[str]], catalog_size: int) -> float:
    unique_recommended = set(tid for recs in all_recommendations for tid in recs)
    return len(unique_recommended) / catalog_size


def main():
    args = parse_args()

    with open(args.predictions) as f:
        predictions = json.load(f)
    with open(args.ground_truth) as f:
        ground_truth = json.load(f)

    # Build GT lookup: (session_id, turn_id) → set of track_ids
    gt_lookup = {}
    for entry in ground_truth:
        key = (entry["session_id"], entry["turn_id"])
        gt_lookup[key] = set(entry.get("track_ids", []))

    # Load embeddings
    logger.info("Loading embeddings...")
    embs_raw = np.load(args.embeddings)
    with open(args.track_ids) as f:
        all_track_ids = [l.strip() for l in f]
    track_embeddings = dict(zip(all_track_ids, embs_raw))
    catalog_size = len(all_track_ids)

    # Store original ranked lists (lambda=1.0 = original order)
    original_tracks = {
        (p["session_id"], p["turn_id"]): p["track_ids"]
        for p in predictions
    }

    logger.info(f"Sweeping lambda in {args.lambdas}")
    logger.info(f"{'lambda':>8}  {'nDCG@20':>10}  {'Coverage':>10}  {'Combined':>10}")
    logger.info("-" * 45)

    best_lambda, best_combined = 0.5, -1.0

    for lam in args.lambdas:
        ndcg_scores, all_recs = [], []

        for pred in predictions:
            key = (pred["session_id"], pred["turn_id"])
            raw_tracks = original_tracks[key]
            rel_scores = list(range(len(raw_tracks), 0, -1))

            reranked = mmr_rerank(
                candidates=raw_tracks,
                embeddings=track_embeddings,
                relevance_scores=rel_scores,
                lambda_=lam,
                top_k=20,
            )
            all_recs.append(reranked)

            gt = gt_lookup.get(key, set())
            ndcg_scores.append(ndcg_at_k(reranked, gt, k=20))

        mean_ndcg = float(np.mean(ndcg_scores))
        coverage  = catalog_coverage(all_recs, catalog_size)
        combined  = 0.7 * mean_ndcg + 0.3 * coverage   # adjust weights as needed

        marker = " ← best" if combined > best_combined else ""
        logger.info(f"{lam:>8.1f}  {mean_ndcg:>10.4f}  {coverage:>10.4f}  {combined:>10.4f}{marker}")

        if combined > best_combined:
            best_combined = combined
            best_lambda   = lam

    logger.success(f"\nBest lambda = {best_lambda} (combined score = {best_combined:.4f})")
    logger.info(f"Set mmr_lambda: {best_lambda} in config/train.yaml")


if __name__ == "__main__":
    main()
