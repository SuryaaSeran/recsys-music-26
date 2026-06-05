"""Stage A training: LoRA (attention) + only the 1027 new cf-token embedding rows.

The full tied [V,H] embedding is frozen (a buffer in LeanVocab); the new rows are a
small parameter. Loss is computed by running the inner LlamaModel, gathering hidden
at the position before each target token, and applying the split head only there
(memory bounded by #targets, not T*V). No full-vocab logits, no grad checkpointing.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.genret.config import GenRetConfig
from src.genret.data import collate
from src.genret.model import attach_lean_vocab
from src.genret.tokens import SemTokenizer


def resolve_device(pref="auto"):
    if pref != "auto":
        return pref
    if torch.cuda.is_available():
        return "cuda"
    return "mps" if torch.backends.mps.is_available() else "cpu"


def build_model(cfg: GenRetConfig, device: str, resume_from: str | None = None):
    tok = AutoTokenizer.from_pretrained(cfg.base)
    sem = SemTokenizer(tok)
    raw = AutoModelForCausalLM.from_pretrained(cfg.base, dtype=getattr(torch, cfg.dtype))
    raw.resize_token_embeddings(len(tok), mean_resizing=False)  # mean_resizing=True OOMs MPS
    lv = attach_lean_vocab(raw, sem.new_lo)
    if resume_from:
        from peft import PeftModel
        ck = torch.load(Path(resume_from) / "new_token_embeddings.pt", map_location="cpu")
        lv.new_emb.data.copy_(ck["new_rows"])
        peft_model = PeftModel.from_pretrained(raw, resume_from, is_trainable=True)
    else:
        lora = LoraConfig(task_type="CAUSAL_LM", r=cfg.lora_r, lora_alpha=cfg.lora_alpha,
                          lora_dropout=cfg.lora_dropout,
                          target_modules=["q_proj", "k_proj", "v_proj", "o_proj"])
        peft_model = get_peft_model(raw, lora)      # injects LoRA in-place; freezes base
    lv.new_emb.requires_grad_(True)                 # re-enable our new rows
    raw.to(device)
    return raw, peft_model, lv, tok, sem


def loss_and_acc(raw, lv, batch):
    """Inner-model hidden -> split head on the position before each target -> CE."""
    out = raw.model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"])
    hidden = out.last_hidden_state                  # [B,T,H]
    mask = batch["labels"] != -100
    b_idx, t_idx = mask.nonzero(as_tuple=True)
    prev = hidden[b_idx, t_idx - 1]                 # logit at i-1 predicts token at i
    logits = lv.head(prev)                          # [Nt, V]
    tgt = batch["labels"][b_idx, t_idx]
    loss = F.cross_entropy(logits, tgt)
    acc = (logits.argmax(-1) == tgt).float().mean()
    return loss, acc


def train(cfg: GenRetConfig, overfit: int = 0, resume_from: str | None = None,
          start_epoch: int = 0):
    device = resolve_device(cfg.device)
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)

    raw, peft_model, lv, tok, sem = build_model(cfg, device, resume_from)
    pad_id = tok.eos_token_id

    data = [json.loads(l) for l in open(Path(cfg.data_dir) / "train.jsonl")]
    if overfit:
        data = data[:overfit]
    params = [p for p in raw.parameters() if p.requires_grad]
    n_train = sum(p.numel() for p in params)
    print(f"device={device} examples={len(data)} trainable={n_train/1e6:.2f}M "
          f"(new_emb {lv.new_emb.numel()/1e6:.2f}M + LoRA)")

    frozen_snap = lv.frozen.detach().clone()        # must never change
    opt = torch.optim.AdamW(params, lr=cfg.lr, weight_decay=0.0)

    bs, accum, ml = cfg.batch_size, cfg.grad_accum, cfg.train_max_len
    rng = np.random.default_rng(cfg.seed)
    out = Path(cfg.ckpt_dir)
    out.mkdir(parents=True, exist_ok=True)

    for epoch in range(start_epoch, start_epoch + cfg.epochs):
        raw.train()
        order = rng.permutation(len(data))
        t0 = time.time()
        run_loss, run_acc, nb = 0.0, 0.0, 0
        opt.zero_grad()
        for step, i in enumerate(range(0, len(order), bs)):
            ex = [data[j] for j in order[i:i + bs]]
            ex = [{"input_ids": e["input_ids"][-ml:], "labels": e["labels"][-ml:]} for e in ex]
            b = {k: v.to(device) for k, v in collate(ex, pad_id).items()}
            loss, acc = loss_and_acc(raw, lv, b)
            (loss / accum).backward()
            run_loss += loss.item(); run_acc += acc.item(); nb += 1
            if (step + 1) % accum == 0:
                opt.step(); opt.zero_grad()
            if device == "mps" and step % cfg.empty_cache_every == 0:
                torch.mps.empty_cache()
        opt.step(); opt.zero_grad()
        peak = torch.mps.driver_allocated_memory() / 1e9 if device == "mps" else 0
        print(f"epoch {epoch}: loss {run_loss/nb:.4f}  tgt_acc {run_acc/nb:.4f}  "
              f"{(time.time()-t0)/60:.1f} min  mps_peak {peak:.1f}GB")
        save_checkpoint(peft_model, lv, tok, out / f"epoch{epoch}")   # keep every epoch

    drift = (lv.frozen.detach() - frozen_snap).abs().max().item()
    print(f"frozen-embedding max drift (want 0): {drift:.2e}")
    return {"frozen_drift": drift}


def save_checkpoint(peft_model, lv, tok, out_dir):
    out = Path(out_dir)
    peft_model.save_pretrained(out)                  # LoRA adapter (adapter_config.json + weights)
    tok.save_pretrained(out)
    torch.save({"new_lo": lv.new_lo, "new_rows": lv.new_emb.detach().cpu()},
               out / "new_token_embeddings.pt")
