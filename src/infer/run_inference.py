"""
Step 4: Inference — constrained beam search → MMR → top-20 + response.

One forward pass produces both the ranked track list and the response text.
No separate retrieval index needed — the semantic ID vocabulary IS the index.

Usage:
    python src/infer/run_inference.py --config config/train.yaml --split dev
    python src/infer/run_inference.py --config config/train.yaml --split blind_a
"""

import argparse
import json
import pickle
from pathlib import Path

import numpy as np
import torch
import yaml
from loguru import logger
from tqdm import tqdm
from transformers import LogitsProcessorList

from src.infer.constrained_decoding import build_constraint_processor
from src.infer.mmr import mmr_rerank
from src.model.music_crs_model import MusicCRSModel, N_COARSE
from src.train.dataset import (
    SYSTEM_PROMPT,
    format_dialogue,
    format_history,
    format_user_profile,
)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config",      default="config/train.yaml")
    p.add_argument("--split",       default="dev", choices=["dev", "blind_a", "blind_b"])
    p.add_argument("--checkpoint",  default=None, help="Path to model checkpoint (default: dpo_final)")
    p.add_argument("--batch_size",  type=int, default=1)
    p.add_argument("--mmr_lambda",  type=float, default=None, help="Override MMR lambda from config")
    return p.parse_args()


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def load_sessions(data_path: str, split: str) -> list[dict]:
    """Load sessions for the given split."""
    session_file = Path(data_path) / split / "sessions.jsonl"
    sessions = []
    with open(session_file) as f:
        for line in f:
            sessions.append(json.loads(line))
    return sessions


def decode_ids_from_output(
    output_text: str,
    codebook: dict,
    tokenizer,
    n_coarse: int = N_COARSE,
) -> tuple[list[str], str]:
    """
    Parse model output into:
      - list of track_ids (top-20)
      - response text

    Model output format:
        <12> <303>
        <45> <289>
        ... (20 lines)
        [RESPONSE] Here are some tracks...
    """
    track_ids = []
    response_text = ""

    codes_to_tracks: dict = codebook["codes_to_tracks"]

    lines = output_text.strip().split("\n")
    for line in lines:
        line = line.strip()
        if line.startswith("[RESPONSE]"):
            response_text = line[len("[RESPONSE]"):].strip()
            break
        # Parse semantic ID pair
        parts = line.split()
        if len(parts) == 2:
            try:
                coarse = int(parts[0].strip("<>"))
                fine   = int(parts[1].strip("<>")) - n_coarse
                pair   = (coarse, fine)
                bucket = codes_to_tracks.get(pair, [])
                if bucket:
                    # Pick first track in bucket (or could rank by CF similarity)
                    track_ids.append(bucket[0])
            except (ValueError, IndexError):
                continue

    return track_ids, response_text


def run_inference(cfg: dict, args):
    # ── Load model ─────────────────────────────────────────────────────────
    checkpoint = args.checkpoint or str(Path(cfg["output_dir"]) / "dpo_final")
    logger.info(f"Loading model from {checkpoint}...")
    model_wrapper = MusicCRSModel.from_pretrained(
        model_name=cfg["lm_type"],
        cfg=cfg,
        checkpoint_path=checkpoint,
    )
    model     = model_wrapper.model.eval()
    tokenizer = model_wrapper.tokenizer

    # ── Load codebook ───────────────────────────────────────────────────────
    with open(cfg["codebook_save_path"], "rb") as f:
        codebook = pickle.load(f)
    track_to_codes: dict = codebook["track_to_codes"]

    # ── Load track embeddings for MMR ───────────────────────────────────────
    logger.info("Loading track embeddings for MMR...")
    track_embeddings_raw = np.load(cfg["embedding_path"])
    with open(cfg["track_ids_path"]) as f:
        all_track_ids = [l.strip() for l in f]
    track_embeddings = dict(zip(all_track_ids, track_embeddings_raw))

    # ── Build constraint processor ─────────────────────────────────────────
    constraint_processor = build_constraint_processor(
        codebook_path=cfg["codebook_save_path"],
        tokenizer=tokenizer,
        n_coarse=cfg.get("n_coarse", N_COARSE),
    )
    logits_processors = LogitsProcessorList([constraint_processor])

    # ── Load sessions ───────────────────────────────────────────────────────
    sessions = load_sessions(cfg["data_path"], args.split)
    logger.info(f"Loaded {len(sessions):,} sessions from {args.split} split")

    mmr_lambda = args.mmr_lambda if args.mmr_lambda is not None else cfg.get("mmr_lambda", 0.5)

    predictions = []

    for session in tqdm(sessions, desc="Sessions"):
        profile  = session.get("user_profile", {})
        history  = session.get("listening_history", [])
        turns    = session.get("turns", [])

        for i, turn in enumerate(turns):
            if turn.get("role") != "user":
                continue

            # Build prompt
            prompt = (
                f"<|system|>\n{SYSTEM_PROMPT}\n"
                f"{format_user_profile(profile)}\n"
                f"{format_history(history, track_to_codes, cfg.get('max_history_tracks', 20))}\n"
                f"{format_dialogue(turns, i)}\n"
                f"<|assistant|>\n"
            )

            inputs = tokenizer(
                prompt,
                return_tensors="pt",
                truncation=True,
                max_length=cfg.get("max_seq_length", 2048) - cfg.get("max_new_tokens", 300),
            ).to(model.device)

            with torch.no_grad():
                output_ids = model.generate(
                    **inputs,
                    num_beams=cfg.get("beam_size", 40),
                    num_beam_groups=cfg.get("num_beam_groups", 20),
                    diversity_penalty=cfg.get("diversity_penalty", 0.3),
                    max_new_tokens=cfg.get("max_new_tokens", 300),
                    logits_processor=logits_processors,
                    pad_token_id=tokenizer.eos_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                )

            # Decode only the generated part
            gen_ids = output_ids[0][inputs["input_ids"].shape[1]:]
            output_text = tokenizer.decode(gen_ids, skip_special_tokens=False)

            # Parse IDs and response
            raw_track_ids, response_text = decode_ids_from_output(
                output_text, codebook, tokenizer
            )

            # Fallback if decoding produced < 20 tracks
            if len(raw_track_ids) < 20:
                logger.warning(
                    f"Only decoded {len(raw_track_ids)} tracks for "
                    f"session={session['session_id']} turn={turn['turn_id']}, padding..."
                )
                raw_track_ids += all_track_ids[:20 - len(raw_track_ids)]

            # MMR reranking for catalog diversity
            relevance_scores = list(range(len(raw_track_ids), 0, -1))  # rank-based scores
            final_track_ids = mmr_rerank(
                candidates=raw_track_ids[:20],
                embeddings=track_embeddings,
                relevance_scores=relevance_scores,
                lambda_=mmr_lambda,
                top_k=20,
            )

            predictions.append({
                "session_id": session["session_id"],
                "turn_id":    turn["turn_id"],
                "track_ids":  final_track_ids[:20],
                "response":   response_text or "Here are some tracks you might enjoy.",
            })

    # ── Save predictions ────────────────────────────────────────────────────
    output_path = Path(cfg["predictions_dir"]) / f"predictions_{args.split}.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(predictions, f, indent=2)

    logger.success(f"Saved {len(predictions):,} predictions to {output_path}")
    logger.info(
        f"Next: python music-crs-evaluator/evaluate_devset.py "
        f"--predictions {output_path} --ground_truth data/ground_truth_{args.split}.json"
    )


def main():
    args = parse_args()
    cfg  = load_config(args.config)
    run_inference(cfg, args)


if __name__ == "__main__":
    main()
