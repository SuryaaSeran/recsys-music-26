"""Per-modality on-disk cache for RQ-VAE training (the efficiency layer).

Read embeddings once via src.tracks, L2-normalize present rows, optionally PCA, write
compact float16 artifacts. Training never touches parquet again after this.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from src.tracks import load_track_embeddings

DEFAULT_DIR = "data/cache"


@dataclass
class CachedModality:
    track_ids: list[str]
    matrix: np.ndarray      # [N, d] float16, L2-normalized present rows (0 where absent)
    present: np.ndarray     # [N] bool


def _l2_normalize(mat: np.ndarray, present: np.ndarray) -> np.ndarray:
    out = mat.astype(np.float32, copy=True)
    norms = np.linalg.norm(out, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    out = out / norms
    out[~present] = 0.0
    return out


def _fit_pca(x: np.ndarray, k: int):
    from sklearn.decomposition import PCA

    p = PCA(n_components=k, random_state=0).fit(x)
    return p.components_.astype(np.float32), p.mean_.astype(np.float32)


def _apply_pca(mat: np.ndarray, present: np.ndarray, components: np.ndarray, mean: np.ndarray) -> np.ndarray:
    out = np.zeros((mat.shape[0], components.shape[0]), dtype=np.float32)
    out[present] = (mat[present] - mean) @ components.T
    return out


def build_cache(modalities: list[str] | None = None, out_dir: str = DEFAULT_DIR,
                pca: dict[str, int] | None = None, force: bool = False) -> dict:
    from src.rqvae.config import ID_MODALITY_ORDER

    mods = modalities or ID_MODALITY_ORDER
    pca = pca or {}
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    te = load_track_embeddings(mods)
    track_ids = te.track_ids
    (out / "track_ids.json").write_text(json.dumps(track_ids))

    stats: dict[str, dict] = {}
    for m in mods:
        mat_path = out / f"{m}.f16.npy"
        if mat_path.exists() and not force:
            present = np.load(out / f"{m}.present.npy")
            mat = np.load(mat_path, mmap_mode="r")
            stats[m] = {"dim": int(mat.shape[1]), "present": int(present.sum()),
                        "pca": pca.get(m), "skipped": True}
            continue

        present = te.present[m]
        norm = _l2_normalize(te.matrices[m], present)
        k = pca.get(m)
        if k:
            comp, mean = _fit_pca(norm[present], k)
            norm = _apply_pca(norm, present, comp, mean)
            norm = _l2_normalize(norm, present)
            np.savez(out / f"pca_{m}.npz", components=comp, mean=mean)

        np.save(mat_path, norm.astype(np.float16))
        np.save(out / f"{m}.present.npy", present)
        stats[m] = {"dim": int(norm.shape[1]), "present": int(present.sum()), "pca": k}

    (out / "norm_stats.json").write_text(json.dumps(stats, indent=2))
    return stats


def load_cached(modality: str, out_dir: str = DEFAULT_DIR) -> CachedModality:
    out = Path(out_dir)
    track_ids = json.loads((out / "track_ids.json").read_text())
    matrix = np.load(out / f"{modality}.f16.npy")
    present = np.load(out / f"{modality}.present.npy")
    return CachedModality(track_ids, matrix, present)


def load_track_ids(out_dir: str = DEFAULT_DIR) -> list[str]:
    return json.loads((Path(out_dir) / "track_ids.json").read_text())
