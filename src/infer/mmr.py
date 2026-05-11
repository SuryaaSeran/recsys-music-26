"""
MMR (Maximal Marginal Relevance) post-processing for catalog diversity.

After beam search returns 20 ranked tracks, MMR reorders them to balance:
  - Relevance  (model's original ranking score)
  - Diversity  (dissimilarity to already-selected tracks)

This directly improves the catalog diversity metric with zero impact on
the underlying retrieval model. Tune lambda on the Dev split.

lambda=1.0 → pure relevance (original order)
lambda=0.0 → pure diversity
lambda=0.5 → balanced (default, good starting point)
"""

import numpy as np


def mmr_rerank(
    candidates: list[str],
    embeddings: dict[str, np.ndarray],
    relevance_scores: list[float],
    lambda_: float = 0.5,
    top_k: int = 20,
) -> list[str]:
    """
    Re-rank a list of candidate track IDs using MMR.

    Args:
        candidates:        Ordered list of track_ids (model's output, best first)
        embeddings:        Dict mapping track_id → embedding vector (for similarity)
        relevance_scores:  Model relevance score for each candidate (same order)
        lambda_:           Trade-off between relevance and diversity (0–1)
        top_k:             Number of tracks to return

    Returns:
        Re-ranked list of track_ids, length = min(top_k, len(candidates))
    """
    if not candidates:
        return []

    # Normalise relevance scores to [0, 1]
    scores = np.array(relevance_scores, dtype=np.float32)
    if scores.max() > scores.min():
        scores = (scores - scores.min()) / (scores.max() - scores.min())

    # Filter to candidates that have embeddings
    valid = [(tid, sc) for tid, sc in zip(candidates, scores) if tid in embeddings]
    if not valid:
        return candidates[:top_k]

    track_ids, rel_scores = zip(*valid)
    vecs = np.stack([embeddings[tid] for tid in track_ids])

    # Normalise embeddings
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    vecs = vecs / (norms + 1e-8)

    selected_indices = []
    remaining = list(range(len(track_ids)))

    for _ in range(min(top_k, len(track_ids))):
        if not remaining:
            break

        if not selected_indices:
            # First item: pick highest relevance
            best = max(remaining, key=lambda i: rel_scores[i])
        else:
            # MMR score for each remaining candidate
            sel_vecs = vecs[selected_indices]

            mmr_scores = []
            for i in remaining:
                # Max similarity to already-selected tracks
                sim = float(np.max(vecs[i] @ sel_vecs.T))
                mmr = lambda_ * rel_scores[i] - (1 - lambda_) * sim
                mmr_scores.append((i, mmr))

            best = max(mmr_scores, key=lambda x: x[1])[0]

        selected_indices.append(best)
        remaining.remove(best)

    return [track_ids[i] for i in selected_indices]
