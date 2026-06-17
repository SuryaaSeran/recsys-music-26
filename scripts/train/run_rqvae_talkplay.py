#!/usr/bin/env python3
"""Runner for eugeneyan/semantic-ids-llm train_rqvae.py on TalkPlay metadata-qwen3 embeddings.

Run A spec (per user):
  - embedding: metadata-qwen3_embedding_0.6b (1024-dim)
  - codebook_quantization_levels = 2
  - codebook_size = 64
  - use_kmeans_init = True
  - reset_unused_codes = True

Usage:
    .venv/bin/python scripts/train/run_rqvae_talkplay.py --levels 2 --codes 64 --run_name runA
"""
import argparse
import os
import sys
from pathlib import Path

# Make src/ importable from the vendored repo
REPO_ROOT = Path(__file__).resolve().parents[2] / "third_party" / "semantic-ids-llm"
sys.path.insert(0, str(REPO_ROOT))

# Disable wandb (eugeneyan's script calls wandb.init/log/log_artifact extensively).
os.environ.setdefault("WANDB_MODE", "disabled")
os.environ.setdefault("WANDB_DISABLED", "true")

import torch
from torch.utils.data import DataLoader

from src.train_rqvae import (  # noqa: E402
    RQVAEConfig, RQVAE, EmbeddingDataset, train_rqvae,
)
from src.device_manager import DeviceManager  # noqa: E402
from src.logger import setup_logger  # noqa: E402

import wandb  # noqa: E402

logger = setup_logger("run-rqvae-talkplay", log_to_file=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--levels", type=int, default=2)
    parser.add_argument("--codes", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=400,
                        help="Total epochs. With 47K items & batch 8192, "
                             "1 epoch ≈ 6 steps. 400 epochs ≈ 2400 steps.")
    parser.add_argument("--batch_size", type=int, default=8192)
    parser.add_argument("--run_name", default="runA")
    parser.add_argument("--max_lr", type=float, default=3e-4)
    parser.add_argument("--warmup_steps", type=int, default=200)
    parser.add_argument("--embeddings", default=None,
                        help="Path to parquet with 'parent_asin' and 'embedding' columns. "
                             "Defaults to the original Qwen3 1024-dim parquet.")
    parser.add_argument("--embedding_dim", type=int, default=None,
                        help="Embedding dimension. Inferred from parquet if not set.")
    args = parser.parse_args()

    cfg = RQVAEConfig()
    cfg.category = "TalkPlay"
    cfg.data_dir = REPO_ROOT / "data"
    cfg.embeddings_path = Path(args.embeddings) if args.embeddings else \
        REPO_ROOT / "data" / "output" / "TalkPlay_items_with_embeddings.parquet"
    cfg.checkpoint_dir = Path("models/rqvae") / args.run_name

    # Infer embedding dim from parquet if not specified
    if args.embedding_dim is not None:
        cfg.item_embedding_dim = args.embedding_dim
    else:
        import polars as pl
        _df = pl.read_parquet(str(cfg.embeddings_path), n_rows=1)
        cfg.item_embedding_dim = len(_df["embedding"][0])
    cfg.codebook_quantization_levels = args.levels
    cfg.codebook_size = args.codes
    cfg.use_kmeans_init = True
    cfg.reset_unused_codes = True
    cfg.use_ema_vq = False
    cfg.use_rotation_trick = True
    cfg.batch_size = args.batch_size
    cfg.num_epochs = args.epochs
    cfg.max_lr = args.max_lr
    cfg.warmup_steps = args.warmup_steps
    cfg.scheduler_type = "cosine_with_warmup"

    cfg.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    cfg.log_config()

    device_manager = DeviceManager(logger)
    device = device_manager.device
    logger.info(f"Device: {device}")

    # wandb is disabled via env var but the script still calls wandb.init etc.
    # We initialize in disabled mode so the calls are no-ops.
    wandb.init(project="rqvae", name=args.run_name, mode="disabled")

    dataset = EmbeddingDataset(str(cfg.embeddings_path))
    val_size = max(int(len(dataset) * cfg.val_split), 1)
    train_size = len(dataset) - val_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size], generator=torch.Generator().manual_seed(42)
    )
    logger.info(f"Train size: {len(train_dataset):,}  Val size: {len(val_dataset):,}")

    train_loader = DataLoader(
        train_dataset, batch_size=cfg.batch_size, shuffle=True,
        num_workers=0, pin_memory=False, drop_last=False,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=cfg.batch_size, shuffle=False,
        num_workers=0, pin_memory=False,
    )

    model = RQVAE(cfg)
    train_rqvae(model=model, data_loader=train_loader, val_loader=val_loader,
                config=cfg, device=device)

    final = cfg.checkpoint_dir / "final_model.pth"
    logger.info(f"Saving final to {final}")
    torch.save({"model_state_dict": model.state_dict(), "config": cfg.__dict__}, final)
    logger.info("Done.")


if __name__ == "__main__":
    main()
