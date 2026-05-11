"""
Step 3: Joint SFT training (retrieval + response in one model).

Training flow:
  1. SFT on (dialogue → semantic IDs + response) — joint loss
  2. DPO on ranked preference pairs to sharpen ranking & Distinct-2

Run:
    python src/train/train.py --config config/train.yaml
    python src/train/train.py --config config/train.yaml --stage dpo
"""

import argparse
from pathlib import Path

import torch
import yaml
from datasets import Dataset as HFDataset
from loguru import logger
from transformers import DataCollatorForSeq2Seq, TrainingArguments
from trl import DPOTrainer, SFTTrainer

from src.model.music_crs_model import MusicCRSModel
from src.train.dataset import MusicCRSDataset
from src.train.dpo_dataset import build_dpo_dataset


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="config/train.yaml")
    p.add_argument("--stage", choices=["sft", "dpo", "both"], default="both")
    p.add_argument("--resume_from_checkpoint", default=None)
    return p.parse_args()


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def run_sft(cfg: dict, model_wrapper: MusicCRSModel, resume: str | None = None):
    """Supervised fine-tuning: cross-entropy on ID tokens + response tokens."""
    logger.info("=" * 60)
    logger.info("Stage 1: SFT")
    logger.info("=" * 60)

    train_ds = MusicCRSDataset(
        data_path=cfg["data_path"],
        split=cfg["train_split"],
        codebook_path=cfg["codebook_save_path"],
        tokenizer=model_wrapper.tokenizer,
        cfg=cfg,
        english_mix_path=cfg.get("english_mix_path"),
    )
    dev_ds = MusicCRSDataset(
        data_path=cfg["data_path"],
        split=cfg["dev_split"],
        codebook_path=cfg["codebook_save_path"],
        tokenizer=model_wrapper.tokenizer,
        cfg=cfg,
    )

    output_dir = str(Path(cfg["output_dir"]) / "sft")
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=cfg.get("num_train_epochs", 3),
        per_device_train_batch_size=cfg.get("per_device_train_batch_size", 4),
        per_device_eval_batch_size=4,
        gradient_accumulation_steps=cfg.get("gradient_accumulation_steps", 8),
        learning_rate=cfg.get("learning_rate", 2e-4),
        lr_scheduler_type=cfg.get("lr_scheduler_type", "cosine"),
        warmup_ratio=cfg.get("warmup_ratio", 0.05),
        bf16=cfg.get("bf16", True),
        fp16=cfg.get("fp16", False),
        logging_steps=cfg.get("logging_steps", 50),
        save_steps=cfg.get("save_steps", 500),
        eval_strategy="steps",
        eval_steps=cfg.get("save_steps", 500),
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        dataloader_num_workers=cfg.get("dataloader_num_workers", 4),
        report_to="none",
        remove_unused_columns=False,
    )

    data_collator = DataCollatorForSeq2Seq(
        tokenizer=model_wrapper.tokenizer,
        model=model_wrapper.model,
        padding=True,
        pad_to_multiple_of=8,
        label_pad_token_id=-100,
    )

    trainer = SFTTrainer(
        model=model_wrapper.model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=dev_ds,
        data_collator=data_collator,
        tokenizer=model_wrapper.tokenizer,
    )

    logger.info("Starting SFT training...")
    trainer.train(resume_from_checkpoint=resume)

    sft_path = str(Path(cfg["output_dir"]) / "sft_final")
    model_wrapper.save(sft_path)
    logger.success(f"SFT complete. Model saved to {sft_path}")
    return sft_path


def run_dpo(cfg: dict, model_wrapper: MusicCRSModel, sft_checkpoint: str):
    """
    DPO fine-tuning to sharpen ranking and improve Distinct-2.

    Preference pairs are constructed from train annotations:
      - Retrieval pairs: top-5 ground-truth = chosen, rank 50+ = rejected
      - Response pairs: specific/grounded response = chosen, generic = rejected
    """
    logger.info("=" * 60)
    logger.info("Stage 2: DPO")
    logger.info("=" * 60)

    dpo_dataset = build_dpo_dataset(
        data_path=cfg["data_path"],
        split=cfg["train_split"],
        codebook_path=cfg["codebook_save_path"],
        cfg=cfg,
    )
    logger.info(f"DPO dataset: {len(dpo_dataset):,} preference pairs")

    output_dir = str(Path(cfg["output_dir"]) / "dpo")
    dpo_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=cfg.get("dpo_epochs", 1),
        per_device_train_batch_size=2,
        gradient_accumulation_steps=16,
        learning_rate=cfg.get("dpo_lr", 5e-5),
        bf16=cfg.get("bf16", True),
        logging_steps=50,
        save_steps=200,
        save_total_limit=1,
        report_to="none",
        remove_unused_columns=False,
    )

    dpo_trainer = DPOTrainer(
        model=model_wrapper.model,
        ref_model=None,           # use implicit ref (frozen copy)
        beta=cfg.get("dpo_beta", 0.1),
        args=dpo_args,
        train_dataset=dpo_dataset,
        tokenizer=model_wrapper.tokenizer,
        max_length=cfg.get("max_seq_length", 2048),
        max_prompt_length=1024,
    )

    logger.info("Starting DPO training...")
    dpo_trainer.train()

    dpo_path = str(Path(cfg["output_dir"]) / "dpo_final")
    model_wrapper.save(dpo_path)
    logger.success(f"DPO complete. Model saved to {dpo_path}")
    return dpo_path


def main():
    args = parse_args()
    cfg = load_config(args.config)

    logger.info(f"Loading model: {cfg['lm_type']}")
    model_wrapper = MusicCRSModel.from_pretrained(
        model_name=cfg["lm_type"],
        cfg=cfg,
        load_in_4bit=False,
        checkpoint_path=args.resume_from_checkpoint,
    )

    sft_path = None
    if args.stage in ("sft", "both"):
        sft_path = run_sft(cfg, model_wrapper, resume=args.resume_from_checkpoint)

    if args.stage in ("dpo", "both"):
        if sft_path is None:
            # Load from config if running DPO standalone
            sft_path = str(Path(cfg["output_dir"]) / "sft_final")
        run_dpo(cfg, model_wrapper, sft_path)

    logger.success("Training complete.")


if __name__ == "__main__":
    main()
