"""Train one per-modality RQ-VAE branch. Branches are independent (parallel-safe)."""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import torch

from src.rqvae.cache import load_cached
from src.rqvae.config import ModalityConfig
from src.rqvae.model import RqVae, codebook_utilization


def resolve_device(pref: str = "auto") -> str:
    if pref != "auto":
        return pref
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def assign_all(model: RqVae, Xt: torch.Tensor, device: str, bs: int = 8192) -> np.ndarray:
    model.eval()
    out = []
    with torch.no_grad():
        for i in range(0, len(Xt), bs):
            out.append(model.assign(Xt[i:i + bs].to(device)).cpu().numpy())
    model.train()
    return np.concatenate(out).astype(np.int32)


def train_modality(cfg: ModalityConfig, device: str = "auto",
                   cache_dir: str = "data/cache", out_dir: str = "exp/codebooks") -> dict:
    device = resolve_device(device)
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)

    cm = load_cached(cfg.name, cache_dir)
    X = cm.matrix[cm.present].astype(np.float32)
    Xt = torch.from_numpy(X)
    n = len(X)

    model = RqVae(cfg.input_dim, cfg.embed_dim, tuple(cfg.hidden_dims),
                  cfg.codebook_size, cfg.n_layers, cfg.commitment_weight,
                  cfg.kmeans_init).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr)

    bs = cfg.batch_size
    log = []
    t0 = time.time()
    for epoch in range(cfg.epochs):
        perm = torch.randperm(n)
        agg = {"loss": 0.0, "recon": 0.0, "commit": 0.0}
        nb = 0
        for i in range(0, n, bs):
            idx = perm[i:i + bs]
            xb = Xt[idx].to(device)
            loss, recon, commit, _ = model(xb)
            opt.zero_grad()
            loss.backward()
            opt.step()
            agg["loss"] += float(loss)
            agg["recon"] += float(recon)
            agg["commit"] += float(commit)
            nb += 1
        rec = {k: round(v / nb, 5) for k, v in agg.items()}
        rec["epoch"] = epoch
        log.append(rec)
        if epoch % 25 == 0 or epoch == cfg.epochs - 1:
            print(f"  [{cfg.name}] epoch {epoch:3d}  loss {rec['loss']:.4f}  "
                  f"recon {rec['recon']:.4f}  commit {rec['commit']:.4f}")

    codes = assign_all(model, Xt, device)
    util = codebook_utilization(codes, cfg.codebook_size, cfg.n_layers)

    save_dir = Path(out_dir) / cfg.name
    save_dir.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": model.state_dict(), "cfg": vars(cfg)}, save_dir / "ckpt.pt")
    codebook = np.stack([l.embed.weight.detach().cpu().numpy() for l in model.layers])
    np.save(save_dir / "codebook.npy", codebook)
    (save_dir / "train_log.json").write_text(
        json.dumps({"cfg": vars(cfg), "minutes": round((time.time() - t0) / 60, 2),
                    "util": util, "log": log}, indent=2))

    print(f"  [{cfg.name}] done in {(time.time()-t0)/60:.1f} min  util "
          + " ".join(f"L{l}={util[f'layer{l}']['util']}" for l in range(cfg.n_layers)))
    return {"util": util, "final": log[-1]}
