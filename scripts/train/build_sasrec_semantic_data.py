#!/usr/bin/env python3
"""Build SASRec_semantic_id training data from TalkPlay TRAIN sessions.

Output parquet schema (matches eugeneyan/semantic-ids-llm):
  - user_id (str)                        — TalkPlay session_id
  - semantic_id_sequence (list[str])     — per music turn, "<|sid_start|><|sid_l0|><|sid_(L1+K)|><|sid_end|>"
  - semantic_id_sequence_length (int)    — count of music turns kept

We use Run A semantic IDs (L=2, K=64). Each item is encoded as 2 tokens:
  L0 ∈ [0, 64)  L1 ∈ [64, 128)  (with level offset applied for the trainer's vocab)

Usage:
    python scripts/train/build_sasrec_semantic_data.py \\
      --sids_dir cache/semantic_ids/runA_metaqwen_L2C64 \\
      --out third_party/semantic-ids-llm/data/output/TalkPlay_sequences_with_semantic_ids_train.parquet
"""
import argparse
import json
from pathlib import Path

import numpy as np
import polars as pl
from datasets import load_dataset


def encode_item(l0: int, l1: int, codebook_size: int) -> str:
    # eugeneyan format: <|sid_start|><|sid_X|><|sid_Y|><|sid_end|>
    # X = l0  (raw, level 0)
    # Y = l1 + codebook_size  (level 1 offset)
    return f"<|sid_start|><|sid_{l0}|><|sid_{l1 + codebook_size}|><|sid_end|>"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sids_dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--out_eval", default=None,
                    help="Optional separate parquet for held-out eval sessions (dev split)")
    ap.add_argument("--min_seq", type=int, default=3,
                    help="Minimum number of music turns required (SASRec filter)")
    args = ap.parse_args()

    sids_dir = Path(args.sids_dir)
    meta = json.load(open(sids_dir / "meta.json"))
    K = meta["codebook_size"]
    L = meta["codebook_quantization_levels"]
    assert L == 2, f"This builder expects L=2 (got {L})"

    track_ids = np.load(sids_dir / "track_ids.npy", allow_pickle=True).tolist()
    codes = np.load(sids_dir / "semantic_ids.npy")
    tid_to_codes = {tid: (int(codes[i, 0]), int(codes[i, 1])) for i, tid in enumerate(track_ids)}
    print(f"Loaded {len(tid_to_codes):,} track→(L0,L1) mappings, K={K}")

    def build_split(split: str) -> pl.DataFrame:
        ds = load_dataset("talkpl-ai/TalkPlayData-Challenge-Dataset", split=split)
        rows = []
        dropped_short = 0
        dropped_missing = 0
        for item in ds:
            sid = item["session_id"]
            seq = []
            for t in item["conversations"]:
                if t["role"] != "music":
                    continue
                tid = t["content"]
                if not tid:
                    continue
                codes = tid_to_codes.get(tid)
                if codes is None:
                    continue
                seq.append(encode_item(codes[0], codes[1], K))
            if len(seq) < args.min_seq:
                dropped_short += 1
                continue
            rows.append({
                "user_id": sid,
                "semantic_id_sequence": seq,
                "semantic_id_sequence_length": len(seq),
            })
        print(f"  {split}: {len(rows):,} sessions (dropped {dropped_short} <{args.min_seq} items, {dropped_missing} missing codes)")
        return pl.DataFrame(rows)

    train_df = build_split("train")
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    train_df.write_parquet(out)
    print(f"Wrote train: {out}")

    if args.out_eval:
        eval_df = build_split("test")
        eval_out = Path(args.out_eval)
        eval_out.parent.mkdir(parents=True, exist_ok=True)
        eval_df.write_parquet(eval_out)
        print(f"Wrote eval: {eval_out}")

    # Stats
    lens = train_df["semantic_id_sequence_length"].to_list()
    print(f"Train length: min={min(lens)} median={int(np.median(lens))} max={max(lens)} mean={np.mean(lens):.1f}")


if __name__ == "__main__":
    main()
