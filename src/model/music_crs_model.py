"""
Step 2: LLM with expanded Semantic ID vocabulary.

Wraps Llama-3.2-1B-Instruct (or any CausalLM) with:
  - 512 new special tokens: <0>..<255> (coarse) and <256>..<511> (fine)
  - LoRA adapters on attention projection layers
  - Frozen base model weights (only new embeddings + LoRA are trained)

This follows the Speak Spotify partial weight-freezing recipe:
  "We freeze the base LLM weights and original token embeddings,
   training only the new ID-token embeddings."
"""

from pathlib import Path
from typing import Optional

import torch
from loguru import logger
from peft import LoraConfig, TaskType, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig


# ── Token naming convention ──────────────────────────────────────────────────
# Coarse codes: <0> .. <255>   → token strings "<0>", "<1>", ..., "<255>"
# Fine codes:   <256> .. <511> → token strings "<256>", ..., "<511>"
# In prompts, a track with coarse=12, fine=47 appears as: "<12> <303>"
# (fine code is offset by n_coarse=256 to avoid collision)

N_COARSE = 256
N_FINE   = 256
TOTAL_ID_TOKENS = N_COARSE + N_FINE  # 512


def make_id_tokens() -> list[str]:
    return [f"<{i}>" for i in range(TOTAL_ID_TOKENS)]


def codes_to_token_str(coarse: int, fine: int) -> str:
    """Convert a (coarse, fine) code pair to the token string for the LLM."""
    return f"<{coarse}> <{fine + N_COARSE}>"


def token_str_to_codes(token_str: str) -> tuple[int, int]:
    """Parse '<12> <303>' back to (12, 47)."""
    parts = token_str.strip().split()
    coarse = int(parts[0].strip("<>"))
    fine   = int(parts[1].strip("<>")) - N_COARSE
    return coarse, fine


class MusicCRSModel:
    """
    Wraps a CausalLM with semantic ID vocabulary expansion and LoRA.

    Usage:
        model_wrapper = MusicCRSModel.from_pretrained("meta-llama/Llama-3.2-1B-Instruct", cfg)
        model   = model_wrapper.model
        tokenizer = model_wrapper.tokenizer
    """

    def __init__(self, model, tokenizer, cfg: dict):
        self.model = model
        self.tokenizer = tokenizer
        self.cfg = cfg
        self.id_tokens = make_id_tokens()

    @classmethod
    def from_pretrained(
        cls,
        model_name: str,
        cfg: dict,
        load_in_4bit: bool = False,
        checkpoint_path: Optional[str] = None,
    ) -> "MusicCRSModel":

        logger.info(f"Loading tokenizer from {model_name}...")
        tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "right"

        # ── Add semantic ID tokens ──────────────────────────────────────────
        id_tokens = make_id_tokens()
        n_added = tokenizer.add_special_tokens(
            {"additional_special_tokens": id_tokens}
        )
        logger.info(f"Added {n_added} semantic ID tokens to tokenizer vocabulary")

        # ── Load base model ─────────────────────────────────────────────────
        quant_config = None
        if load_in_4bit:
            quant_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True,
            )

        logger.info(f"Loading model {model_name}...")
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=quant_config,
            torch_dtype=torch.bfloat16,
            attn_implementation=cfg.get("attn_implementation", "eager"),
            device_map="auto",
        )

        # Resize embeddings to fit new tokens
        model.resize_token_embeddings(len(tokenizer))

        # ── Freeze base model; only train new token embeddings + LoRA ───────
        # Freeze everything first
        for param in model.parameters():
            param.requires_grad = False

        # Unfreeze only the newly added token embedding rows
        # (rows N_original .. N_original + 512)
        embed_weight = model.get_input_embeddings().weight
        n_original = len(tokenizer) - TOTAL_ID_TOKENS
        embed_weight.requires_grad = True  # gradient flows, but we'll mask old rows in optimizer

        # ── Apply LoRA ───────────────────────────────────────────────────────
        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=cfg.get("lora_r", 16),
            lora_alpha=cfg.get("lora_alpha", 32),
            target_modules=cfg.get("lora_target_modules", ["q_proj", "v_proj"]),
            lora_dropout=cfg.get("lora_dropout", 0.05),
            bias="none",
        )
        model = get_peft_model(model, lora_config)

        trainable, total = 0, 0
        for p in model.parameters():
            total += p.numel()
            if p.requires_grad:
                trainable += p.numel()
        logger.info(
            f"Trainable params: {trainable:,} / {total:,} "
            f"({100 * trainable / total:.2f}%)"
        )

        if checkpoint_path:
            logger.info(f"Loading checkpoint from {checkpoint_path}...")
            from peft import PeftModel
            model = PeftModel.from_pretrained(model, checkpoint_path)

        return cls(model, tokenizer, cfg)

    def save(self, path: str):
        Path(path).mkdir(parents=True, exist_ok=True)
        self.model.save_pretrained(path)
        self.tokenizer.save_pretrained(path)
        logger.info(f"Model saved to {path}")
