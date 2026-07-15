"""Stage 3 (LLM variant): predict L0 buckets from conversation context.

Drop-in replacement for SemanticIDRetriever that uses a fine-tuned Qwen3 LLM
instead of SASRec. The LLM sees the full conversation context (profile, goal,
history, listener thoughts) and predicts the most likely L0 bucket(s) for the
next track — goal-aware and conversation-aware, unlike SASRec.

Interface matches SemanticIDRetriever.expand() so it plugs into
run_inference_fusion_recall_expansion.py with a --llm_bucket_model flag.

Usage:
    retriever = SemanticIDLLMRetriever(
        llm_model="models/sid_qwen3_8b/merged",   # or base+adapter
        sids_dir="cache/semantic_ids/runF_v8e_L2C64",
        device="mps",
    )
    cands, meta = retriever.expand_from_context(
        profile, goal_text, specificity, history_blocks, current_query,
        top_k_l0=3, exclude_tids=pool_set,
    )
"""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Optional

import numpy as np
import torch


SYSTEM_QUERY = (
    "You are a music recommendation assistant. "
    "Given a conversation context, predict the L0 semantic cluster ID (0-63) "
    "that best matches the next recommended track. "
    "Output only the integer cluster ID, nothing else."
)


class SemanticIDLLMRetriever:
    """Wraps a fine-tuned Qwen3 LLM to expand recall via predicted L0 buckets."""

    def __init__(
        self,
        llm_model: str,
        sids_dir: str,
        device: str = "mps",
        adapter: Optional[str] = None,
        max_cands_per_bucket: int = 0,
    ):
        self.device = device

        # ── Load semantic IDs ─────────────────────────────────────────────────
        sids_dir = Path(sids_dir)
        codes = np.load(sids_dir / "semantic_ids.npy")
        tids  = np.load(sids_dir / "track_ids.npy", allow_pickle=True).tolist()
        self.tid_to_l0 = {t: int(c[0]) for t, c in zip(tids, codes)}
        self.l0_to_tids: dict[int, list[str]] = defaultdict(list)
        for t, c in zip(tids, codes):
            self.l0_to_tids[int(c[0])].append(t)
        self.n_buckets = len(self.l0_to_tids)

        # ── Load LLM ──────────────────────────────────────────────────────────
        from transformers import AutoModelForCausalLM, AutoTokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(llm_model, padding_side="left")
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model = AutoModelForCausalLM.from_pretrained(
            llm_model,
            torch_dtype=torch.float32 if device == "mps" else torch.bfloat16,
            device_map={"": device},
        )
        if adapter:
            from peft import PeftModel
            self.model = PeftModel.from_pretrained(self.model, adapter)
            self.model = self.model.merge_and_unload()
        self.model.eval()

        # Precompute token IDs for bucket digits 0-63 (first token of each)
        self.bucket_first_tok = {}
        for b in range(64):
            ids = self.tokenizer.encode(str(b), add_special_tokens=False)
            if ids:
                self.bucket_first_tok[b] = ids[0]

    def _build_prompt(self, profile_line: str, goal_line: str,
                      history_blocks: list[str], current_query: str) -> str:
        user_content = "\n".join(
            [profile_line, goal_line] + history_blocks
            + [f"[NOW] USER: {current_query}", "Predict the L0 cluster ID:"]
        )
        messages = [
            {"role": "system", "content": SYSTEM_QUERY},
            {"role": "user",   "content": user_content},
        ]
        return self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True)

    @torch.no_grad()
    def predict_top_k(self, prompt: str, k: int) -> list[tuple[int, float]]:
        """Return top-k (bucket, prob) using next-token logits over digit tokens."""
        ids = self.tokenizer(prompt, return_tensors="pt",
                             max_length=1024, truncation=True).to(self.device)
        out = self.model(**ids)
        logits = out.logits[0, -1, :]   # next-token logits

        # Score each bucket by its first-digit token logit
        scores = []
        for b, tok in self.bucket_first_tok.items():
            scores.append((b, logits[tok].item()))
        # Softmax over bucket scores for calibrated probs
        raw = torch.tensor([s for _, s in scores])
        probs = torch.softmax(raw, dim=-1).tolist()
        ranked = sorted(
            [(scores[i][0], probs[i]) for i in range(len(scores))],
            key=lambda x: -x[1],
        )
        return ranked[:k]

    def expand_from_context(
        self,
        profile_line: str,
        goal_line: str,
        history_blocks: list[str],
        current_query: str,
        top_k_l0: int = 3,
        exclude_tids: Optional[set[str]] = None,
        rejected_tids: Optional[set[str]] = None,
    ) -> tuple[list[str], dict[str, tuple[int, float]]]:
        """Return candidate tids from top-K predicted L0 buckets.

        Matches SemanticIDRetriever.expand() return signature:
            (candidates, meta_map) where meta_map[tid] = (l0_rank, l0_prob)
        """
        prompt = self._build_prompt(profile_line, goal_line, history_blocks, current_query)
        top_buckets = self.predict_top_k(prompt, top_k_l0 + len(rejected_tids or []))

        # Blacklist buckets containing rejected tracks
        rejected_l0: set[int] = set()
        if rejected_tids:
            for tid in rejected_tids:
                b = self.tid_to_l0.get(tid)
                if b is not None:
                    rejected_l0.add(b)
        top_buckets = [(b, p) for b, p in top_buckets if b not in rejected_l0][:top_k_l0]

        result = []
        meta_map: dict[str, tuple[int, float]] = {}
        seen = set(exclude_tids) if exclude_tids else set()
        for rank, (b, prob) in enumerate(top_buckets):
            for tid in self.l0_to_tids.get(b, []):
                if tid not in seen:
                    result.append(tid)
                    meta_map[tid] = (rank, prob)
                    seen.add(tid)
        return result, meta_map
