"""Train the Stage A generator.

  python scripts/genret_train.py --overfit 200   # sanity: memorize a subset
  python scripts/genret_train.py                 # full train
"""
import argparse
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from src.genret.config import GenRetConfig
from src.genret.train import train


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int)
    ap.add_argument("--lr", type=float)
    ap.add_argument("--batch-size", type=int)
    ap.add_argument("--grad-accum", type=int)
    ap.add_argument("--ckpt-dir")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--overfit", type=int, default=0)
    a = ap.parse_args()

    cfg = GenRetConfig(device=a.device)
    if a.epochs: cfg.epochs = a.epochs
    if a.lr: cfg.lr = a.lr
    if a.batch_size: cfg.batch_size = a.batch_size
    if a.grad_accum: cfg.grad_accum = a.grad_accum
    if a.ckpt_dir: cfg.ckpt_dir = a.ckpt_dir
    train(cfg, overfit=a.overfit)


if __name__ == "__main__":
    main()
