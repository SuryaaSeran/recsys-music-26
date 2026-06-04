# Stage A on Kaggle (GPU)

Trains the generative candidate generator on a T4/P100 (~5-10x faster than local MPS)
and prints dev recall@pool. Self-contained: only `genret_kaggle.py` + one uploaded file.

## 1. Upload the cf semantic IDs as a Kaggle dataset
From this repo, the file is `exp/ids/semantic_ids.json` (built by Stage 1). Create a new
Kaggle Dataset (e.g. `reccys-cf-ids`) and add that one file. It maps every track to its
cf-bpr 4-code tuple (the targets + the decode trie). ~tens of MB.

(If you also want the popularity cold-slice later, it's pulled from HF metadata at
runtime; nothing extra to upload.)

## 2. New Kaggle Notebook
- Settings: **Accelerator = GPU T4 x2** (or P100), **Internet = ON**.
- Add data: your `reccys-cf-ids` dataset, and upload `genret_kaggle.py` (or paste it).

## 3. Cells
```python
!pip -q install -U "transformers>=4.45" peft accelerate datasets

!python genret_kaggle.py \
    --sem /kaggle/input/reccys-cf-ids/semantic_ids.json \
    --out /kaggle/working/ckpt \
    --epochs 12 --batch-size 16 --grad-accum 2 \
    --eval-every 2 --eval-n 1000 --patience 3
```

## What it does
- Pulls sessions (`talkpl-ai/TalkPlayData-Challenge-Dataset`) and the base model
  (`unsloth/Llama-3.2-1B-Instruct`) from HF.
- **Per-epoch terminal resampling:** each epoch re-draws one terminal turn per session
  from the Blind A turn-depth histogram, so more epochs = more of the 121k turns seen
  (better generalization than repeating a fixed 15,199).
- Trains LoRA (q,k,v,o) + the 1027 new cf-token embedding rows only; base frozen
  (lean split-head, bf16).
- Every `--eval-every` epochs: trie-constrained beam=256 decode -> recall@{20,50,100,200}
  on `--eval-n` dev turns, with `ceiling`, `gold_first_token_in_pool`, `exact_top1`, and
  recall@200 by turn. Headline: `recall.200` vs `ceiling` (~0.983).
- **Early stopping** on recall@200: stops after `--patience` consecutive evals with no
  gain (`--min-delta`). The best checkpoint is kept in `ckpt/best/` regardless of when
  it occurred, so you can set `--epochs 12` and let it stop itself.

## Outputs (in /kaggle/working/ckpt)
- `ckpt/best/` -- best-by-recall@200 weights (adapter + `new_token_embeddings.pt` +
  tokenizer) and `best/eval.json`. **Use this one.**
- `ckpt/` -- latest epoch's weights; `eval_epochN.json` per eval.

## Knobs
- Faster/smaller: `--batch-size 8`. More data per epoch is automatic via resampling.
- T4 16GB fits bs=16 easily at bf16; bump to `--batch-size 24` on P100 if you want.
- `--epochs 8` is a starting point; watch `recall.200` across `eval_epoch*.json` and stop
  when it flattens.
