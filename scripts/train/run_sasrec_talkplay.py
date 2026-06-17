#!/usr/bin/env python3
"""Runner for SASRec_semantic_id on TalkPlay sequences (Run A semantic IDs).

Config:
  - num_levels = 2 (matches Run A RQ-VAE)
  - codebook_size = 64
  - vocab_size = 128 (auto)
  - max_seq_length = 8 (TalkPlay sessions have 8 music turns)

Usage:
    .venv/bin/python scripts/train/run_sasrec_talkplay.py --run_name sasrec_runA
"""
import argparse
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2] / "third_party" / "semantic-ids-llm"
sys.path.insert(0, str(REPO_ROOT))
os.environ.setdefault("WANDB_MODE", "disabled")
os.environ.setdefault("WANDB_DISABLED", "true")

import torch
from torch.utils.data import DataLoader

from src.train_sasrec_semantic_id import (  # noqa: E402
    SemanticSASRecConfig, SemanticSASRec,
    SemanticSequenceDataset, SemanticTrainingDataset,
    collate_semantic_batch, train_semantic_sasrec,
)
from src.device_manager import DeviceManager  # noqa: E402
from src.logger import setup_logger  # noqa: E402

import wandb  # noqa: E402

logger = setup_logger("run-sasrec-talkplay", log_to_file=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_levels", type=int, default=2)
    parser.add_argument("--codebook_size", type=int, default=64)
    parser.add_argument("--max_seq_length", type=int, default=8)
    parser.add_argument("--num_blocks", type=int, default=2,
                        help="Transformer blocks. 2 is enough for 8-item sequences.")
    parser.add_argument("--num_heads", type=int, default=4)
    parser.add_argument("--head_dim", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--run_name", default="sasrec_runA")
    parser.add_argument("--max_lr", type=float, default=1e-3)
    parser.add_argument("--data_path", default=None,
                        help="Path to sequence parquet. Defaults to the original runA parquet.")
    args = parser.parse_args()

    cfg = SemanticSASRecConfig()
    cfg.dataset = "TalkPlay"
    cfg.data_dir = REPO_ROOT / "data"
    cfg.data_path = Path(args.data_path) if args.data_path else \
        REPO_ROOT / "data" / "output" / "TalkPlay_sequences_with_semantic_ids_train.parquet"
    cfg.checkpoint_dir = Path("models/sasrec") / args.run_name

    cfg.num_levels = args.num_levels
    cfg.codebook_size = args.codebook_size
    cfg.vocab_size = args.num_levels * args.codebook_size
    cfg.max_seq_length = args.max_seq_length

    cfg.num_blocks = args.num_blocks
    cfg.num_heads = args.num_heads
    cfg.head_dim = args.head_dim
    cfg.hidden_dim = args.head_dim * args.num_heads
    cfg.input_dim = cfg.hidden_dim  # keep input_dim aligned

    cfg.num_epochs = args.epochs
    cfg.batch_size = args.batch_size
    cfg.max_learning_rate = args.max_lr
    cfg.use_compile = False  # MPS doesn't support torch.compile well

    cfg.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    cfg.log_config()

    device_manager = DeviceManager(logger)
    device = device_manager.device
    logger.info(f"Device: {device}")

    wandb.init(project="sasrec-semantic", name=args.run_name, mode="disabled")

    base_dataset = SemanticSequenceDataset(cfg)

    model = SemanticSASRec(cfg)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Model params: {n_params:,}")

    train_semantic_sasrec(model=model, train_dataset=base_dataset, config=cfg, device=device)

    final = cfg.checkpoint_dir / "final_model.pth"
    torch.save({"model_state_dict": model.state_dict(), "config": cfg.__dict__}, final)
    logger.info(f"Saved final to {final}")


if __name__ == "__main__":
    main()
