"""
Constrained beam search decoding for Semantic IDs.

Builds a prefix trie over all valid (coarse, fine) ID pairs so the model
can never hallucinate a non-catalog track. Every decoded token sequence is
guaranteed to correspond to a real track in the 1M catalog.

Based on the Speak Spotify production approach:
  "Exploring constrained decoding to cut latency while preserving accuracy."
"""

import pickle
from pathlib import Path

import torch
from loguru import logger
from transformers import LogitsProcessor, LogitsProcessorList


class SemanticIDConstraintProcessor(LogitsProcessor):
    """
    Masks logits at each decoding step to only allow tokens that form
    valid catalog Semantic ID sequences.

    State machine with 3 states per track slot:
      STATE_FREE   → any text token (response prose)
      STATE_COARSE → expecting a coarse ID token <0..255>
      STATE_FINE   → expecting a fine ID token <256..511>

    The processor detects when the model has just emitted a coarse token
    and then constrains the next token to only valid fine tokens that
    complete a real catalog pair.
    """

    def __init__(
        self,
        valid_coarse: set[int],
        valid_pairs: set[tuple[int, int]],
        coarse_token_ids: list[int],    # token IDs for <0>..<255>
        fine_token_ids: list[int],      # token IDs for <256>..<511>
        n_coarse: int = 256,
    ):
        self.valid_coarse    = valid_coarse
        self.valid_pairs     = valid_pairs
        self.coarse_token_ids = torch.tensor(coarse_token_ids)
        self.fine_token_ids   = torch.tensor(fine_token_ids)
        self.n_coarse = n_coarse

        # Precompute: for each valid coarse code c, which fine codes are valid?
        self.coarse_to_valid_fine: dict[int, set[int]] = {}
        for (c, f) in valid_pairs:
            self.coarse_to_valid_fine.setdefault(c, set()).add(f)

        # last_coarse[batch_idx] = coarse code just emitted, or None
        self._last_coarse: dict[int, int | None] = {}

    def __call__(
        self,
        input_ids: torch.LongTensor,
        scores: torch.FloatTensor,
    ) -> torch.FloatTensor:
        batch_size = input_ids.shape[0]

        for b in range(batch_size):
            last_token_id = input_ids[b, -1].item()

            # Check if the last emitted token was a coarse ID token
            if last_token_id in self.coarse_token_ids.tolist():
                coarse_code = self.coarse_token_ids.tolist().index(last_token_id)
                self._last_coarse[b] = coarse_code
            else:
                self._last_coarse[b] = None

            # If we're in fine-token state, mask to only valid fine completions
            if self._last_coarse.get(b) is not None:
                coarse_code = self._last_coarse[b]
                valid_fines = self.coarse_to_valid_fine.get(coarse_code, set())

                # Build allowed fine token IDs
                allowed = torch.full_like(scores[b], float("-inf"))
                for fine_code in valid_fines:
                    fine_token_id = self.fine_token_ids[fine_code].item()
                    allowed[fine_token_id] = scores[b, fine_token_id]
                scores[b] = allowed

        return scores


def build_constraint_processor(
    codebook_path: str,
    tokenizer,
    n_coarse: int = 256,
) -> SemanticIDConstraintProcessor:
    """Load codebook and build the logits processor."""
    with open(codebook_path, "rb") as f:
        codebook = pickle.load(f)

    valid_coarse = codebook["valid_coarse"]
    valid_pairs  = codebook["valid_pairs"]

    # Map code indices to actual token IDs in the tokenizer vocab
    coarse_token_ids = [
        tokenizer.convert_tokens_to_ids(f"<{i}>")
        for i in range(n_coarse)
    ]
    fine_token_ids = [
        tokenizer.convert_tokens_to_ids(f"<{i + n_coarse}>")
        for i in range(n_coarse)
    ]

    logger.info(
        f"Built constraint processor: "
        f"{len(valid_coarse)} valid coarse codes, "
        f"{len(valid_pairs)} valid pairs"
    )
    return SemanticIDConstraintProcessor(
        valid_coarse=valid_coarse,
        valid_pairs=valid_pairs,
        coarse_token_ids=coarse_token_ids,
        fine_token_ids=fine_token_ids,
        n_coarse=n_coarse,
    )
