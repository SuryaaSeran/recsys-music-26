"""Per-modality RQ-VAE configuration (Stage 1 semantic IDs).

Tiered design: cf-bpr (collaborative signal) gets the finest residual depth (L=4)
and a smaller latent; the four content modalities get L=3. All codebooks K=256.
The final semantic ID concatenates each modality's code tuple in ID_MODALITY_ORDER.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from src.tracks import EMB_DIMS

# Fixed concatenation order of per-modality code tuples in the final semantic ID.
# Load-bearing for Stage 2: do NOT reorder after IDs are exported.
ID_MODALITY_ORDER = [
    "cf-bpr",
    "audio-laion_clap",
    "attributes-qwen3_embedding_0.6b",
    "lyrics-qwen3_embedding_0.6b",
    "metadata-qwen3_embedding_0.6b",
]

# Reserved per-position code for a track that lacks a given modality. Safe because
# a real argmin index is always >= 0.
MISSING_CODE = -1


@dataclass
class ModalityConfig:
    name: str
    input_dim: int                      # EMB_DIMS[name], or PCA target dim if pca set
    embed_dim: int = 256                # RQ-VAE latent / codebook vector dim
    hidden_dims: tuple = (512, 256)
    codebook_size: int = 256
    n_layers: int = 3                   # residual quantization depth
    commitment_weight: float = 0.25
    kmeans_init: bool = True
    epochs: int = 200
    batch_size: int = 4096
    lr: float = 1e-4
    pca: int | None = None              # offline PCA target dim, None = off
    seed: int = 0


# Per-modality overrides applied on top of ModalityConfig defaults.
_OVERRIDES: dict[str, dict] = {
    "cf-bpr": dict(embed_dim=128, hidden_dims=(256, 128), n_layers=4),
    "audio-laion_clap": dict(),
    "attributes-qwen3_embedding_0.6b": dict(),
    "lyrics-qwen3_embedding_0.6b": dict(),
    "metadata-qwen3_embedding_0.6b": dict(),
}


def default_configs(pca: dict[str, int] | None = None) -> dict[str, ModalityConfig]:
    """Build the tiered config for each modality in ID_MODALITY_ORDER.

    pca: optional {modality: target_dim}; sets input_dim to the PCA dim for those.
    """
    pca = pca or {}
    cfgs: dict[str, ModalityConfig] = {}
    for name in ID_MODALITY_ORDER:
        ov = dict(_OVERRIDES[name])
        k = pca.get(name)
        input_dim = k if k else EMB_DIMS[name]
        cfgs[name] = ModalityConfig(name=name, input_dim=input_dim, pca=k, **ov)
    return cfgs
