"""Extend a pretrained tokenizer with cf-bpr semantic-ID tokens.

1024 content tokens `<cf_L_c>` (level L in 0..3, code c in 0..255) plus structural
specials. A code is a DISTINCT token per level (per-level offset), matching the
residual-codebook design.
"""
from __future__ import annotations

import numpy as np

from src.genret.config import (CF_CODEBOOK, CF_LEVELS, CF_TOKEN, GEN, HIST_CLOSE,
                               HIST_OPEN, STRUCTURAL)


class SemTokenizer:
    """Owns the new-token vocabulary and cf-code <-> token-id mapping."""

    def __init__(self, tokenizer):
        self.tok = tokenizer
        cf_tokens = [CF_TOKEN.format(level=l, code=c)
                     for l in range(CF_LEVELS) for c in range(CF_CODEBOOK)]
        new = cf_tokens + STRUCTURAL
        # Add as additional special tokens so they are never split.
        added = tokenizer.add_special_tokens({"additional_special_tokens": new})
        self.n_added = added

        # grid[L][c] -> token id ; and inverse for decoding.
        self.grid = np.empty((CF_LEVELS, CF_CODEBOOK), dtype=np.int64)
        self.tok2levelcode: dict[int, tuple[int, int]] = {}
        for l in range(CF_LEVELS):
            for c in range(CF_CODEBOOK):
                tid = tokenizer.convert_tokens_to_ids(CF_TOKEN.format(level=l, code=c))
                self.grid[l, c] = tid
                self.tok2levelcode[tid] = (l, c)
        self.hist_open_id = tokenizer.convert_tokens_to_ids(HIST_OPEN)
        self.hist_close_id = tokenizer.convert_tokens_to_ids(HIST_CLOSE)
        self.gen_id = tokenizer.convert_tokens_to_ids(GEN)
        self.eos_id = tokenizer.eos_token_id

        cf_ids = self.grid.reshape(-1)
        self.cf_lo = int(cf_ids.min())
        self.cf_hi = int(cf_ids.max())
        # All newly added ids (cf + structural), contiguous block at the end.
        all_new = list(cf_ids) + [self.hist_open_id, self.hist_close_id, self.gen_id]
        self.new_lo = int(min(all_new))
        self.new_hi = int(max(all_new))

    # --- cf code <-> token ---
    def cf_codes_to_tokens(self, codes4) -> list[int]:
        return [int(self.grid[l, int(codes4[l])]) for l in range(CF_LEVELS)]

    def cf_codes_to_str(self, codes4) -> str:
        return "".join(CF_TOKEN.format(level=l, code=int(codes4[l])) for l in range(CF_LEVELS))

    def tokens_to_cf_codes(self, token_ids) -> tuple:
        return tuple(self.tok2levelcode[int(t)][1] for t in token_ids)

    def new_token_id_range(self) -> tuple[int, int]:
        """Contiguous (lo, hi_inclusive) of all added rows, for trainable masking."""
        return self.new_lo, self.new_hi
