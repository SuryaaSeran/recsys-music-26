"""
Utility: Convert TalkPlayData-2 HuggingFace format → local sessions.jsonl

TalkPlayData-2 on HuggingFace (talkpl-ai/TalkPlayData-2) uses a specific
schema. This script normalises it to the flat sessions.jsonl format our
Dataset class expects.

Usage:
    python src/utils/data_prep.py --split train --output_dir data/TalkPlayData-2
    python src/utils/data_prep.py --split dev   --output_dir data/TalkPlayData-2
"""

import argparse
import json
from pathlib import Path

from datasets import load_dataset
from loguru import logger
from tqdm import tqdm


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--split",      default="train", choices=["train", "dev", "blind_a", "blind_b"])
    p.add_argument("--output_dir", default="data/TalkPlayData-2")
    p.add_argument("--hf_dataset", default="talkpl-ai/TalkPlayData-2",
                   help="HuggingFace dataset identifier")
    return p.parse_args()


def convert_session(raw: dict) -> dict:
    """Normalise a raw HF example to our session format."""
    turns = []
    raw_turns = raw.get("conversations") or raw.get("turns") or []

    for i, t in enumerate(raw_turns):
        role = t.get("role", t.get("speaker", "user")).lower()
        role = "user" if "user" in role or "listener" in role else "system"

        turn = {
            "turn_id":  i + 1,
            "role":     role,
            "text":     t.get("content", t.get("text", "")),
        }
        # Ground truth tracks and response — only present in train/dev
        if "ground_truth_tracks" in t:
            turn["ground_truth_tracks"] = t["ground_truth_tracks"]
        if "response" in t:
            turn["response"] = t["response"]
        # Some datasets embed the GT in a separate field
        if "recommended_tracks" in raw and role == "system":
            rec = raw["recommended_tracks"]
            if isinstance(rec, dict):
                turn["ground_truth_tracks"] = rec.get(str(i + 1), [])

        turns.append(turn)

    return {
        "session_id":       raw.get("session_id", raw.get("id", f"s_{id(raw)}")),
        "user_profile":     raw.get("user_profile", raw.get("user", {})),
        "listening_history": raw.get("listening_history", raw.get("user_history", [])),
        "turns":            turns,
    }


def main():
    args = parse_args()
    output_dir = Path(args.output_dir) / args.split
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Loading {args.hf_dataset} split={args.split}...")
    # Map HF split names to dataset split keys
    split_map = {"train": "train", "dev": "validation", "blind_a": "test", "blind_b": "test_cold"}
    hf_split = split_map.get(args.split, args.split)

    try:
        ds = load_dataset(args.hf_dataset, split=hf_split)
    except Exception as e:
        logger.error(f"Failed to load dataset: {e}")
        logger.info("If the dataset requires authentication, run: huggingface-cli login")
        return

    out_file = output_dir / "sessions.jsonl"
    n_written = 0
    with open(out_file, "w") as f:
        for raw in tqdm(ds, desc=f"Converting {args.split}"):
            session = convert_session(raw)
            f.write(json.dumps(session) + "\n")
            n_written += 1

    logger.success(f"Wrote {n_written:,} sessions to {out_file}")


if __name__ == "__main__":
    main()
