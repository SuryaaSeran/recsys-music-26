"""Lean trainable vocabulary for the Stage A generator.

Llama ties input embedding and LM head into one [V, H] matrix (264.8M params). To
train only the 1027 new cf-token rows without paying AdamW state for 264.8M params,
we freeze the full matrix as a buffer and hold the new rows as a small parameter.

- Input side: F.embedding for old ids on the frozen weight, new_emb for new ids,
  combined with torch.where (no in-place assign -> autograd/checkpoint safe).
- Output side: logits_old = h @ frozen[:new_lo].T (frozen has no grad, so grad flows
  to `h`/LoRA but not the weight) concatenated with logits_new = h @ new_emb.T.
  No detach: the old-vocab path must carry the softmax-normalization gradient to h.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class LeanVocab(nn.Module):
    def __init__(self, full_weight: torch.Tensor, new_lo: int):
        super().__init__()
        self.new_lo = new_lo
        self.register_buffer("frozen", full_weight.detach().clone(), persistent=False)
        self.frozen.requires_grad_(False)
        self.new_emb = nn.Parameter(full_weight[new_lo:].detach().clone())

    def embed(self, ids: torch.Tensor) -> torch.Tensor:
        frozen_e = F.embedding(ids, self.frozen)
        new_e = F.embedding((ids - self.new_lo).clamp_min(0), self.new_emb)
        is_new = (ids >= self.new_lo).unsqueeze(-1)
        return torch.where(is_new, new_e.to(frozen_e.dtype), frozen_e)

    def head(self, h: torch.Tensor) -> torch.Tensor:
        old = h @ self.frozen[:self.new_lo].t()      # grad to h, not weight (buffer)
        new = h @ self.new_emb.t()                    # grad to h and new_emb
        return torch.cat([old, new], dim=-1)


class _SplicedEmbedding(nn.Module):
    def __init__(self, lv: LeanVocab):
        super().__init__()
        self.lv = lv                                  # registers lv (and new_emb) once

    def forward(self, ids):
        return self.lv.embed(ids)


class _SplicedHead(nn.Module):
    def __init__(self, lv: LeanVocab):
        super().__init__()
        self._lv = [lv]                               # list -> not re-registered (no dup param)

    def forward(self, h):
        # Only ever called during generation (training uses lv.head on gathered positions).
        # Slice to the last position so prefill doesn't build [beams, prompt_len, vocab].
        if h.dim() == 3 and h.shape[1] > 1:
            h = h[:, -1:, :]
        return self._lv[0].head(h)


def attach_lean_vocab(model, new_lo: int) -> LeanVocab:
    """Replace tied embedding + lm_head with the spliced lean versions. Call after
    resize_token_embeddings and before get_peft_model. Works for both training
    (call lv.head on gathered hidden) and generation (generate() uses the spliced head)."""
    W = model.get_input_embeddings().weight.data
    lv = LeanVocab(W, new_lo)
    model.set_input_embeddings(_SplicedEmbedding(lv))
    model.lm_head = _SplicedHead(lv)
    model.config.tie_word_embeddings = False          # we manage the split explicitly
    return lv
