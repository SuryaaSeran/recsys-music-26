"""
Codebook revamp using talkpl-ai/TalkPlayData-Challenge-Track-Embeddings.

Multimodal embeddings (audio CLAP + image SigLIP2 + CF-BPR + Qwen3 text x3)
concatenated and PCA-reduced to 256 dims, then two-level KMeans (128x128).

UUID track IDs match the metadata and conversation datasets directly.

Output: data/codebook_v2.pkl  (same interface as codebook.pkl)
"""

import pickle
from pathlib import Path

import numpy as np
from datasets import load_dataset
from loguru import logger
from sklearn.cluster import MiniBatchKMeans
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import normalize

OUT_PATH = Path("data/codebook_v2.pkl")
EMB_PATH = Path("data/challenge_track_embeddings.npy")
IDS_PATH = Path("data/challenge_track_ids.txt")

N_COARSE = 128
N_FINE = 128
SVD_DIM = 256

MODALITIES = [
    "audio-laion_clap",
    "image-siglip2",
    "cf-bpr",
    "attributes-qwen3_embedding_0.6b",
    "lyrics-qwen3_embedding_0.6b",
    "metadata-qwen3_embedding_0.6b",
]


def main():
    logger.info("Loading TalkPlayData-Challenge-Track-Embeddings (all_tracks)...")
    ds = load_dataset("talkpl-ai/TalkPlayData-Challenge-Track-Embeddings")["all_tracks"]
    logger.info(f"Rows: {len(ds):,} | Columns: {ds.column_names}")

    track_ids_raw = list(map(str, ds["track_id"]))

    logger.info("Building valid mask (all modalities present)...")
    valid_mask = [
        all(ds[col][i] and len(ds[col][i]) > 0 for col in MODALITIES)
        for i in range(len(ds))
    ]
    n_valid = sum(valid_mask)
    logger.info(f"Valid rows: {n_valid:,} / {len(ds):,} ({n_valid*100/len(ds):.1f}%)")
    track_ids = [tid for tid, ok in zip(track_ids_raw, valid_mask) if ok]

    logger.info("Concatenating modality embeddings...")
    arrays = []
    for col in MODALITIES:
        col_data = [ds[col][i] for i, ok in enumerate(valid_mask) if ok]
        arr = np.vstack([np.array(row, dtype=np.float32) for row in col_data])
        norms = np.linalg.norm(arr, axis=1, keepdims=True)
        arr = arr / (norms + 1e-8)
        logger.info(f"  {col}: {arr.shape}")
        arrays.append(arr)

    embeddings = np.concatenate(arrays, axis=1).astype(np.float32)
    logger.info(f"Concatenated shape: {embeddings.shape}")

    # Final L2-normalize
    embeddings = normalize(embeddings).astype(np.float32)

    EMB_PATH.parent.mkdir(parents=True, exist_ok=True)
    np.save(EMB_PATH, embeddings)
    IDS_PATH.write_text("\n".join(track_ids))
    logger.info(f"Saved raw embeddings to {EMB_PATH}")

    logger.info(f"Reducing to {SVD_DIM} dims with TruncatedSVD...")
    svd = TruncatedSVD(n_components=SVD_DIM, random_state=42)
    reduced = svd.fit_transform(embeddings).astype(np.float32)
    reduced = normalize(reduced).astype(np.float32)
    logger.info(f"Reduced shape: {reduced.shape}")

    logger.info(f"Fitting level-1 KMeans: {N_COARSE} coarse clusters...")
    km1 = MiniBatchKMeans(
        n_clusters=N_COARSE,
        random_state=42,
        batch_size=4096,
        n_init=5,
        max_iter=200,
        reassignment_ratio=0.01,
    )
    codes1 = km1.fit_predict(reduced)

    logger.info(f"Fitting level-2 KMeans: {N_FINE} fine clusters on residuals...")
    residuals = normalize(reduced - km1.cluster_centers_[codes1]).astype(np.float32)
    km2 = MiniBatchKMeans(
        n_clusters=N_FINE,
        random_state=42,
        batch_size=4096,
        n_init=5,
        max_iter=200,
        reassignment_ratio=0.01,
    )
    codes2 = km2.fit_predict(residuals)

    track_to_codes = {
        tid: (int(c1), int(c2))
        for tid, c1, c2 in zip(track_ids, codes1, codes2)
    }

    codes_to_tracks: dict[tuple, list] = {}
    for tid, pair in track_to_codes.items():
        codes_to_tracks.setdefault(pair, []).append(tid)

    valid_coarse = set(int(x) for x in codes1)
    valid_pairs = set(track_to_codes.values())

    codebook = {
        "source": "challenge_multimodal",
        "modalities": MODALITIES,
        "svd": svd,
        "km1": km1,
        "km2": km2,
        "track_to_codes": track_to_codes,
        "codes_to_tracks": codes_to_tracks,
        "valid_coarse": valid_coarse,
        "valid_pairs": valid_pairs,
        "n_coarse": N_COARSE,
        "n_fine": N_FINE,
    }

    with open(OUT_PATH, "wb") as f:
        pickle.dump(codebook, f)

    bucket_sizes = [len(v) for v in codes_to_tracks.values()]
    logger.success(f"Codebook v2 saved to {OUT_PATH}")
    logger.info(f"Unique coarse codes: {len(valid_coarse)}/{N_COARSE}")
    logger.info(f"Unique pairs: {len(valid_pairs)}")
    logger.info(f"Avg bucket size: {len(track_ids) / len(valid_pairs):.2f}")
    logger.info(f"Max bucket size: {max(bucket_sizes)}")
    logger.info(f"Min bucket size: {min(bucket_sizes)}")


if __name__ == "__main__":
    main()
