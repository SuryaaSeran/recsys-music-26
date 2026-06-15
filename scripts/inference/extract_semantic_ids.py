#!/usr/bin/env python3
"""Extract semantic IDs for all TalkPlay tracks using a trained RQ-VAE checkpoint.

Outputs:
  cache/semantic_ids/<run_name>/track_ids.npy       — list of track ids (str, order matches embeddings)
  cache/semantic_ids/<run_name>/semantic_ids.npy    — (N, L) int array of (l0, l1, ...) codes
  cache/semantic_ids/<run_name>/meta.json           — config + stats
"""
import argparse
import json
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2] / "third_party" / "semantic-ids-llm"
sys.path.insert(0, str(REPO_ROOT))
os.environ.setdefault("WANDB_MODE", "disabled")

import numpy as np
import polars as pl
import torch

from src.train_rqvae import RQVAEConfig, RQVAE  # noqa: E402


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", required=True,
                        help="Path to .pth checkpoint (e.g. models/rqvae/runA_metaqwen_L2C64/final_model.pth)")
    parser.add_argument("--parquet", default=str(REPO_ROOT / "data" / "output" / "TalkPlay_items_with_embeddings.parquet"))
    parser.add_argument("--out_dir", required=True,
                        help="Output dir, e.g. cache/semantic_ids/runA")
    parser.add_argument("--batch_size", type=int, default=8192)
    parser.add_argument("--device", default="mps")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading checkpoint: {args.ckpt}")
    state = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    cfg_dict = state["config"]
    cfg = RQVAEConfig()
    for k, v in cfg_dict.items():
        if hasattr(cfg, k):
            setattr(cfg, k, v)

    model = RQVAE(cfg)
    model.load_state_dict(state["model_state_dict"], strict=False)
    model.to(args.device)
    model.eval()

    print(f"Loading embeddings: {args.parquet}")
    df = pl.read_parquet(args.parquet)
    track_ids = df["parent_asin"].to_list()
    embeddings = torch.tensor(df["embedding"].to_list(), dtype=torch.float32)
    print(f"  {len(track_ids):,} rows, dim {embeddings.shape[1]}")

    all_codes = []
    with torch.no_grad():
        for i in range(0, len(embeddings), args.batch_size):
            batch = embeddings[i:i + args.batch_size].to(args.device)
            sids = model.encode_to_semantic_ids(batch)  # (B, L)
            all_codes.append(sids.cpu().numpy())
    codes = np.concatenate(all_codes, axis=0)  # (N, L)
    print(f"semantic IDs: {codes.shape}, dtype {codes.dtype}")

    np.save(out_dir / "track_ids.npy", np.array(track_ids))
    np.save(out_dir / "semantic_ids.npy", codes)

    # Stats per level
    L = codes.shape[1]
    n_unique_per_level = [int(np.unique(codes[:, l]).shape[0]) for l in range(L)]
    bucket_sizes_per_level = []
    for l in range(L):
        counts = np.bincount(codes[:, l], minlength=cfg.codebook_size)
        bucket_sizes_per_level.append({
            "n_used": int((counts > 0).sum()),
            "min": int(counts[counts > 0].min()) if (counts > 0).any() else 0,
            "max": int(counts.max()),
            "median": int(np.median(counts[counts > 0])) if (counts > 0).any() else 0,
            "mean": float(counts.mean()),
        })

    # Leaf bucket distribution (joint (l0, l1, ...))
    joint = ["_".join(str(c) for c in row) for row in codes]
    from collections import Counter
    leaf_counts = Counter(joint)
    leaf_sizes = sorted(leaf_counts.values())
    leaf_stats = {
        "n_unique_ids": len(leaf_counts),
        "n_possible_ids": cfg.codebook_size ** L,
        "tracks_per_id_min": leaf_sizes[0],
        "tracks_per_id_p25": leaf_sizes[len(leaf_sizes) // 4],
        "tracks_per_id_median": leaf_sizes[len(leaf_sizes) // 2],
        "tracks_per_id_p75": leaf_sizes[3 * len(leaf_sizes) // 4],
        "tracks_per_id_p99": leaf_sizes[int(0.99 * len(leaf_sizes))],
        "tracks_per_id_max": leaf_sizes[-1],
        "singletons": sum(1 for s in leaf_sizes if s == 1),
        "ids_with_ge10_tracks": sum(1 for s in leaf_sizes if s >= 10),
        "ids_with_ge50_tracks": sum(1 for s in leaf_sizes if s >= 50),
    }

    meta = {
        "ckpt": args.ckpt,
        "parquet": args.parquet,
        "n_tracks": len(track_ids),
        "codebook_quantization_levels": cfg.codebook_quantization_levels,
        "codebook_size": cfg.codebook_size,
        "item_embedding_dim": cfg.item_embedding_dim,
        "per_level": [
            {"level": l, "n_unique": n_unique_per_level[l], **bucket_sizes_per_level[l]}
            for l in range(L)
        ],
        "leaf_stats": leaf_stats,
    }
    with open(out_dir / "meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    print()
    print("=== Per-level stats ===")
    for s in meta["per_level"]:
        print(f"  L{s['level']}: {s['n_used']}/{cfg.codebook_size} codes used  "
              f"(min={s['min']} median={s['median']} max={s['max']} mean={s['mean']:.1f})")
    print()
    print("=== Leaf bucket distribution ===")
    for k, v in leaf_stats.items():
        print(f"  {k}: {v}")
    print()
    print(f"Saved to {out_dir}")


if __name__ == "__main__":
    main()
