"""Train per-modality RQ-VAE codebooks.

  python scripts/build_codebooks.py --all
  python scripts/build_codebooks.py --modality cf-bpr --epochs 50
Each modality is independent; run several at once for parallelism.
"""
import argparse
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from src.rqvae.config import ID_MODALITY_ORDER, default_configs
from src.rqvae.train import train_modality


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--modality")
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--epochs", type=int)
    ap.add_argument("--codebook-size", type=int)
    ap.add_argument("--n-layers", type=int)
    ap.add_argument("--cache-dir", default="data/cache")
    ap.add_argument("--out-dir", default="exp/codebooks")
    a = ap.parse_args()

    cfgs = default_configs()
    if a.all:
        names = ID_MODALITY_ORDER
    elif a.modality:
        names = [a.modality]
    else:
        ap.error("pass --modality NAME or --all")

    for name in names:
        cfg = cfgs[name]
        if a.epochs:
            cfg.epochs = a.epochs
        if a.codebook_size:
            cfg.codebook_size = a.codebook_size
        if a.n_layers:
            cfg.n_layers = a.n_layers
        print(f"=== {name}  input_dim={cfg.input_dim} K={cfg.codebook_size} "
              f"L={cfg.n_layers} epochs={cfg.epochs} ===")
        train_modality(cfg, a.device, a.cache_dir, a.out_dir)


if __name__ == "__main__":
    main()
