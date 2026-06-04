"""Minimal RQ-VAE: MLP encoder/decoder + residual vector quantization.

Faithful to EdoardoBotta/RQ-VAE-Recommender: one codebook per residual layer, kmeans
init from the encoder-output distribution, straight-through estimator, commitment
loss. One RqVae is trained per modality.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class MLP(nn.Module):
    """Dense stack with ReLU between layers, no activation on the output layer."""

    def __init__(self, dims: list[int], out_dim: int):
        super().__init__()
        d = list(dims) + [out_dim]
        layers: list[nn.Module] = []
        for i in range(len(d) - 1):
            layers.append(nn.Linear(d[i], d[i + 1]))
            if i < len(d) - 2:
                layers.append(nn.ReLU())
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class Quantize(nn.Module):
    """A single residual-level codebook."""

    def __init__(self, embed_dim: int, n_embed: int, commitment_weight: float = 0.25,
                 kmeans_init: bool = True):
        super().__init__()
        self.embed_dim = embed_dim
        self.n_embed = n_embed
        self.beta = commitment_weight
        self.embed = nn.Embedding(n_embed, embed_dim)
        nn.init.normal_(self.embed.weight, std=embed_dim ** -0.5)
        self.register_buffer("_inited", torch.tensor(not kmeans_init))

    @torch.no_grad()
    def _kmeans_init(self, z: torch.Tensor):
        from sklearn.cluster import KMeans

        x = z.detach().to("cpu", torch.float32).numpy()
        if len(x) > 20000:
            idx = np.random.default_rng(0).choice(len(x), 20000, replace=False)
            x = x[idx]
        km = KMeans(n_clusters=self.n_embed, n_init=4, random_state=0).fit(x)
        centers = torch.from_numpy(km.cluster_centers_).to(self.embed.weight)
        self.embed.weight.data.copy_(centers)
        self._inited.fill_(True)

    def forward(self, z: torch.Tensor):
        if not bool(self._inited):
            self._kmeans_init(z)
        # L2 distance from each z to every codebook entry
        dist = (
            z.pow(2).sum(1, keepdim=True)
            - 2 * z @ self.embed.weight.t()
            + self.embed.weight.pow(2).sum(1)
        )
        ids = dist.argmin(1)
        e = self.embed(ids)
        commit = F.mse_loss(z.detach(), e) + self.beta * F.mse_loss(z, e.detach())
        q = z + (e - z).detach()  # straight-through estimator
        return q, ids, commit


class RqVae(nn.Module):
    def __init__(self, input_dim: int, embed_dim: int, hidden_dims: tuple,
                 codebook_size: int, n_layers: int, commitment_weight: float = 0.25,
                 kmeans_init: bool = True):
        super().__init__()
        self.n_layers = n_layers
        self.codebook_size = codebook_size
        hidden = list(hidden_dims)
        self.encoder = MLP([input_dim, *hidden], embed_dim)
        self.decoder = MLP([embed_dim, *hidden[::-1]], input_dim)
        self.layers = nn.ModuleList([
            Quantize(embed_dim, codebook_size, commitment_weight, kmeans_init)
            for _ in range(n_layers)
        ])

    def encode(self, x):
        res = self.encoder(x)
        q_sum = 0.0
        commit = 0.0
        ids_list = []
        for layer in self.layers:
            q, ids, c = layer(res)
            res = res - q
            q_sum = q_sum + q
            commit = commit + c
            ids_list.append(ids)
        codes = torch.stack(ids_list, dim=1)  # [B, n_layers]
        return q_sum, codes, commit

    def forward(self, x):
        q_sum, codes, commit = self.encode(x)
        recon = self.decoder(q_sum)
        recon_mse = F.mse_loss(recon, x)
        loss = recon_mse + commit
        return loss, recon_mse, commit, codes

    @torch.no_grad()
    def assign(self, x):
        _, codes, _ = self.encode(x)
        return codes


def codebook_utilization(codes: np.ndarray, codebook_size: int, n_layers: int) -> dict:
    """Per-layer used/dead code counts. `codes` is an int array [N, n_layers]."""
    out = {}
    for l in range(n_layers):
        used = int(np.unique(codes[:, l]).size)
        out[f"layer{l}"] = {
            "used": used,
            "dead": codebook_size - used,
            "util": round(used / codebook_size, 4),
        }
    return out
