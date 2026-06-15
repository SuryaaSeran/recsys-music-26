"""Stage 3: SASRec semantic-bucket recall expansion.

At inference, given a conversation history (list of track_ids), predicts the
top-K most likely L0 semantic buckets for the next recommendation, then returns
all tracks whose semantic ID falls in any of those predicted buckets.

This is a pure recall-expansion module — it produces a set of candidate track_ids
to merge into the main fusion pool. No ranking is done here.

Usage:
    retriever = SemanticIDRetriever(
        sasrec_ckpt="models/sasrec/sasrec_runC2_L2C64/best_model.pth",
        sids_dir="cache/semantic_ids/runC2_attributes_L2C64",
        device="mps",
    )
    extra_candidates = retriever.expand(history_tids, top_k_l0=3)
"""
from __future__ import annotations

import json
import os
import sys
from collections import defaultdict
from pathlib import Path
from typing import Optional

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[2] / "third_party" / "semantic-ids-llm"
sys.path.insert(0, str(REPO_ROOT))
os.environ.setdefault("WANDB_MODE", "disabled")

from src.train_sasrec_semantic_id import SemanticSASRecConfig, SemanticSASRec  # noqa: E402


class SemanticIDRetriever:
    """Wraps trained SASRec to expand recall via predicted semantic-ID buckets."""

    def __init__(
        self,
        sasrec_ckpt: str,
        sids_dir: str,
        device: str = "mps",
        top_k_l0: int = 3,
    ):
        self.device = device
        self.top_k_l0 = top_k_l0

        # Load semantic IDs
        sids_dir = Path(sids_dir)
        meta = json.load(open(sids_dir / "meta.json"))
        self.K = meta["codebook_size"]
        self.L = meta["codebook_quantization_levels"]

        track_ids = np.load(sids_dir / "track_ids.npy", allow_pickle=True).tolist()
        codes = np.load(sids_dir / "semantic_ids.npy")  # (N, L)

        self.tid_to_codes: dict[str, tuple] = {
            t: tuple(int(c) for c in codes[i]) for i, t in enumerate(track_ids)
        }

        # Build L0 bucket → list of track_ids (for fast expansion)
        self.l0_to_tids: dict[int, list[str]] = defaultdict(list)
        for tid, cd in self.tid_to_codes.items():
            self.l0_to_tids[cd[0]].append(tid)

        # Load SASRec model — reconstruct arch from weight shapes
        # (config dict is not serialised in the checkpoint)
        state = torch.load(sasrec_ckpt, map_location="cpu", weights_only=False)
        sd = state["model_state_dict"]

        # Derive arch from saved tensor shapes
        token_emb_w = sd["token_emb.weight"]     # (vocab_size+1, input_dim)
        pos_emb_w   = sd["pos_emb.weight"]        # (max_tokens+1, hidden_dim)
        head_w      = sd["level_heads.0.weight"]  # (codebook_size, hidden_dim)
        attn_w      = sd["blocks.0.attn.c_attn.weight"]  # (3*hidden, hidden)

        saved_vocab = token_emb_w.shape[0] - 1    # exclude padding row
        saved_input_dim = token_emb_w.shape[1]
        saved_hidden = head_w.shape[1]
        saved_codebook = head_w.shape[0]
        saved_num_levels = saved_vocab // saved_codebook
        saved_max_tokens = pos_emb_w.shape[0] - 1
        saved_max_seq = saved_max_tokens // saved_num_levels
        # num_heads: hidden must be divisible; infer from attn projection rows
        # c_attn: (3*hidden, hidden) → num_heads can be any divisor of saved_hidden
        # we trained with num_heads=4, head_dim=64 → hidden=256; verify
        saved_num_blocks = sum(1 for k in sd if k.startswith("blocks.") and k.endswith(".ln_1.weight"))

        cfg = SemanticSASRecConfig()
        cfg.num_levels = saved_num_levels
        cfg.codebook_size = saved_codebook
        cfg.vocab_size = saved_vocab
        cfg.max_seq_length = saved_max_seq
        cfg.input_dim = saved_input_dim
        cfg.hidden_dim = saved_hidden
        # find valid num_heads that divides hidden
        for nh in [4, 8, 2, 1]:
            if saved_hidden % nh == 0:
                cfg.num_heads = nh
                cfg.head_dim = saved_hidden // nh
                break
        cfg.num_blocks = saved_num_blocks

        self.cfg = cfg
        self.model = SemanticSASRec(cfg)
        self.model.load_state_dict(sd, strict=True)
        self.model.to(device)
        self.model.eval()

    def _encode_history(self, history_tids: list[str]) -> Optional[torch.Tensor]:
        """Convert list of track_ids to input_ids tensor for SASRec.

        Returns None if history has no known semantic IDs.
        """
        tokens = []
        for tid in history_tids[-self.cfg.max_seq_length:]:
            cd = self.tid_to_codes.get(tid)
            if cd is None:
                continue
            # level offset encoding: L0 raw, L1 + K, L2 + 2K, ...
            for level, code in enumerate(cd):
                token_id = code + level * self.K
                tokens.append(token_id)
        if not tokens:
            return None
        # pad left to max_seq_length * num_levels
        max_len = self.cfg.max_seq_length * self.cfg.num_levels
        if len(tokens) > max_len:
            tokens = tokens[-max_len:]
        # left-pad with 0
        padded = [0] * (max_len - len(tokens)) + tokens
        return torch.tensor([padded], dtype=torch.long, device=self.device)

    def expand(
        self,
        history_tids: list[str],
        top_k_l0: Optional[int] = None,
        exclude_tids: Optional[set[str]] = None,
    ) -> tuple[list[str], dict[str, tuple[int, float]]]:
        """Return candidates from predicted top-K L0 buckets with calibration scores.

        Returns:
            (candidates, meta_map) where meta_map[tid] = (l0_rank, l0_prob)
            l0_rank: 0-indexed rank of the L0 bucket this candidate came from
            l0_prob: softmax probability of that L0 bucket
        """
        k = top_k_l0 if top_k_l0 is not None else self.top_k_l0
        if not history_tids:
            return [], {}

        input_ids = self._encode_history(history_tids)
        if input_ids is None:
            return [], {}

        with torch.no_grad():
            preds = self.model.predict_next_item(input_ids)
            l0_logits = preds["logits_l0"][0]  # (K,)
            top_result = torch.topk(l0_logits, min(k, self.K))
            top_l0 = top_result.indices.cpu().tolist()
            l0_probs = torch.softmax(l0_logits, dim=-1).cpu().tolist()

        result = []
        meta_map: dict[str, tuple[int, float]] = {}
        seen = set(exclude_tids) if exclude_tids else set()
        for rank, l0 in enumerate(top_l0):
            prob = l0_probs[l0]
            for tid in self.l0_to_tids.get(l0, []):
                if tid not in seen:
                    result.append(tid)
                    meta_map[tid] = (rank, prob)
                    seen.add(tid)
        return result, meta_map

    def predict_l0_distribution(self, history_tids: list[str]) -> list[tuple[int, float]]:
        """Return (l0_code, probability) sorted descending — for debugging/features."""
        if not history_tids:
            return []
        input_ids = self._encode_history(history_tids)
        if input_ids is None:
            return []
        with torch.no_grad():
            preds = self.model.predict_next_item(input_ids)
            l0_probs = torch.softmax(preds["logits_l0"][0], dim=-1).cpu().tolist()
        return sorted(enumerate(l0_probs), key=lambda x: -x[1])
