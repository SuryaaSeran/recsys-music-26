"""
Build DPO preference pairs from the training annotations.

Two types of pairs:
  1. Retrieval pairs  — chosen: top-5 GT tracks as IDs; rejected: random non-GT tracks
  2. Response pairs   — chosen: GT response (specific); rejected: template response (generic)

The DPO trainer expects a HuggingFace Dataset with columns:
  prompt, chosen, rejected
"""

import json
import pickle
import random
from pathlib import Path

from datasets import Dataset as HFDataset
from loguru import logger

from src.model.music_crs_model import codes_to_token_str
from src.train.dataset import (
    SYSTEM_PROMPT,
    format_dialogue,
    format_history,
    format_user_profile,
)

GENERIC_RESPONSE_TEMPLATES = [
    "Here are some tracks you might enjoy.",
    "I recommend these songs based on your preferences.",
    "These tracks might suit your taste.",
    "Check out these songs!",
]


def build_dpo_dataset(
    data_path: str,
    split: str,
    codebook_path: str,
    cfg: dict,
) -> HFDataset:

    with open(codebook_path, "rb") as f:
        codebook = pickle.load(f)
    track_to_codes: dict = codebook["track_to_codes"]
    all_track_ids: list = list(track_to_codes.keys())

    session_file = Path(data_path) / split / "sessions.jsonl"
    pairs = []

    with open(session_file) as f:
        for line in f:
            session = json.loads(line)
            profile  = session.get("user_profile", {})
            history  = session.get("listening_history", [])
            turns    = session.get("turns", [])

            for i, turn in enumerate(turns):
                if turn.get("role") != "user":
                    continue
                gt_tracks = turn.get("ground_truth_tracks", [])
                response  = turn.get("response", "")
                if not gt_tracks or not response:
                    continue

                # Build shared prompt
                prompt = (
                    f"<|system|>\n{SYSTEM_PROMPT}\n"
                    f"{format_user_profile(profile)}\n"
                    f"{format_history(history, track_to_codes, cfg.get('max_history_tracks', 20))}\n"
                    f"{format_dialogue(turns, i)}\n"
                    f"<|assistant|>\n"
                )

                # ── Retrieval pair ──────────────────────────────────────────
                # Chosen: top-5 GT tracks as first IDs, padded
                chosen_ids = []
                for tid in gt_tracks[:5]:
                    if tid in track_to_codes:
                        c1, c2 = track_to_codes[tid]
                        chosen_ids.append(codes_to_token_str(c1, c2))
                while len(chosen_ids) < 20:
                    chosen_ids.append("<0> <256>")

                # Rejected: random non-GT tracks
                non_gt = [t for t in random.sample(all_track_ids, 100) if t not in gt_tracks]
                rejected_ids = []
                for tid in non_gt[:20]:
                    c1, c2 = track_to_codes[tid]
                    rejected_ids.append(codes_to_token_str(c1, c2))

                chosen_retrieval  = "\n".join(chosen_ids)  + f"\n[RESPONSE] {response}"
                rejected_retrieval = "\n".join(rejected_ids) + f"\n[RESPONSE] {response}"

                pairs.append({
                    "prompt": prompt,
                    "chosen": chosen_retrieval,
                    "rejected": rejected_retrieval,
                })

                # ── Response pair ───────────────────────────────────────────
                # Same IDs, but response quality differs
                id_block = "\n".join(chosen_ids)
                generic  = random.choice(GENERIC_RESPONSE_TEMPLATES)

                pairs.append({
                    "prompt": prompt,
                    "chosen":   id_block + f"\n[RESPONSE] {response}",
                    "rejected": id_block + f"\n[RESPONSE] {generic}",
                })

    logger.info(f"Built {len(pairs):,} DPO preference pairs from {split} split")
    return HFDataset.from_list(pairs)
