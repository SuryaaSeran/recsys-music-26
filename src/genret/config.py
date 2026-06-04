"""Stage A (generative candidate generator) configuration."""
from __future__ import annotations

from dataclasses import dataclass, field

# Generator base. Meta's gated repo is 403 for this account; this ungated mirror is
# identical weights.
BASE_MODEL = "unsloth/Llama-3.2-1B-Instruct"

# cf-bpr branch: columns 0:4 of exp/ids/per_modality_codes.npy, codebook size 256.
CF_LEVELS = 4
CF_CODEBOOK = 256

# New-token string formats (per-level offset so a code is a distinct token per level).
CF_TOKEN = "<cf_{level}_{code}>"          # 4 * 256 = 1024 tokens
HIST_OPEN = "<hist>"
HIST_CLOSE = "</hist>"
GEN = "<gen>"                              # decode-start marker; target tuple follows
STRUCTURAL = [HIST_OPEN, HIST_CLOSE, GEN]

# Artifact paths (relative to repo root).
CODES_NPY = "exp/ids/per_modality_codes.npy"
TRACK_IDS_JSON = "data/cache/track_ids.json"


@dataclass
class GenRetConfig:
    base: str = BASE_MODEL
    data_dir: str = "exp/genret/data"
    ckpt_dir: str = "exp/genret/ckpt"
    device: str = "auto"            # auto -> mps if available else cpu
    # data
    with_history: bool = True
    max_recent_turns: int = 3
    max_ctx_tokens: int = 1024
    # train
    dtype: str = "bfloat16"      # bf16 on MPS: ~half the memory, no paging
    epochs: int = 3
    lr: float = 2e-4
    batch_size: int = 2
    grad_accum: int = 16
    train_max_len: int = 256     # left-truncate (keeps target at the end)
    empty_cache_every: int = 1   # MPS: release cache each step or it grows -> paging
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    seed: int = 0
    # generate
    pool_size: int = 200
    num_beams: int = 200
