"""Encode all tracks to semantic IDs and build the combined ID table + report."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from src.rqvae.cache import load_cached, load_track_ids
from src.rqvae.config import ID_MODALITY_ORDER, MISSING_CODE, default_configs
from src.rqvae.model import RqVae, codebook_utilization
from src.rqvae.train import assign_all, resolve_device


def load_model(cfg, ckpt_path: Path, device: str) -> RqVae:
    ck = torch.load(ckpt_path, map_location=device, weights_only=False)
    model = RqVae(cfg.input_dim, cfg.embed_dim, tuple(cfg.hidden_dims),
                  cfg.codebook_size, cfg.n_layers, cfg.commitment_weight,
                  kmeans_init=False).to(device)
    model.load_state_dict(ck["state_dict"])
    model.eval()
    return model


def encode_modality(cfg, cache_dir: str, ckpt_dir: str, device: str):
    cm = load_cached(cfg.name, cache_dir)
    model = load_model(cfg, Path(ckpt_dir) / cfg.name / "ckpt.pt", device)
    Xt = torch.from_numpy(cm.matrix.astype(np.float32))
    codes = assign_all(model, Xt, device)          # [N, n_layers] over ALL rows
    codes[~cm.present] = MISSING_CODE
    return codes, cm.present


def _coherence_samples(per_mod, present, track_ids, modality="cf-bpr", n=3, mates=8):
    from src.tracks import load_catalog

    cat = load_catalog()
    codes = per_mod[modality]
    idx_present = np.where(present[modality])[0]
    rng = np.random.default_rng(0)
    picks = rng.choice(idx_present, min(n, len(idx_present)), replace=False)
    samples = []
    for i in picks:
        l1 = int(codes[i, 0])
        bucket = idx_present[codes[idx_present, 0] == l1]
        mate_ids = [track_ids[j] for j in bucket[:mates]]
        samples.append({
            "modality": modality,
            "l1_code": l1,
            "bucket_size": int(len(bucket)),
            "track": cat[track_ids[i]].text() if track_ids[i] in cat else track_ids[i],
            "mates": [cat[t].text() for t in mate_ids if t in cat],
        })
    return samples


def build_report(full, per_mod, present, dedup, buckets, cfgs, track_ids, verify):
    rep = {"n_tracks": len(track_ids), "id_length": int(full.shape[1])}

    rep["utilization"] = {
        m: codebook_utilization(per_mod[m][present[m]], cfgs[m].codebook_size, cfgs[m].n_layers)
        for m in per_mod
    }

    sizes = np.array([len(v) for v in buckets.values()])
    rep["collisions"] = {
        "n_buckets": int(len(buckets)),
        "pct_size1": round(float((sizes == 1).mean()), 5),
        "max_bucket": int(sizes.max()),
        "n_tracks_dedup_gt0": int((dedup > 0).sum()),
    }
    rep["missing"] = {m: int((~present[m]).sum()) for m in present}
    if verify:
        rep["samples"] = _coherence_samples(per_mod, present, track_ids)
    return rep


def export(cache_dir="data/cache", ckpt_dir="exp/codebooks", out_dir="exp/ids",
           device="auto", verify=False) -> dict:
    device = resolve_device(device)
    cfgs = default_configs()
    track_ids = load_track_ids(cache_dir)
    N = len(track_ids)

    per_mod, present = {}, {}
    for name in ID_MODALITY_ORDER:
        codes, pres = encode_modality(cfgs[name], cache_dir, ckpt_dir, device)
        per_mod[name], present[name] = codes, pres

    full = np.concatenate([per_mod[m] for m in ID_MODALITY_ORDER], axis=1)  # [N, 16]

    # Collision buckets + dedup counter (stable catalog order).
    buckets: dict[tuple, list[str]] = {}
    dedup = np.zeros(N, dtype=np.int32)
    for i in range(N):
        key = tuple(int(x) for x in full[i])
        b = buckets.setdefault(key, [])
        dedup[i] = len(b)
        b.append(track_ids[i])

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    np.save(out / "per_modality_codes.npy", full)

    starts = np.cumsum([0] + [cfgs[m].n_layers for m in ID_MODALITY_ORDER])
    sem = {}
    for i, tid in enumerate(track_ids):
        entry = {m: [int(x) for x in full[i, starts[j]:starts[j + 1]]]
                 for j, m in enumerate(ID_MODALITY_ORDER)}
        entry["full"] = [int(x) for x in full[i]]
        entry["dedup"] = int(dedup[i])
        sem[tid] = entry
    (out / "semantic_ids.json").write_text(json.dumps(sem))
    (out / "codes_to_tracks.json").write_text(
        json.dumps({",".join(map(str, k)): v for k, v in buckets.items()}))

    report = build_report(full, per_mod, present, dedup, buckets, cfgs, track_ids, verify)
    (out / "report.json").write_text(json.dumps(report, indent=2))
    return report
