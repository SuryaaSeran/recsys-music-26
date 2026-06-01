# Plan: Source-aware ranking + Phase B + Phase C TT v8

---

## Status summary (as of 2026-05-28)

| Phase | Result | Note |
|---|---|---|
| A: Source-aware LTR | Done | 0.1646 nDCG@20 (27-feat, tt_pool=1000) |
| B: +popularity/year, tt_pool=2000 | Done | **0.1653** nDCG@20 (29-feat, reg booster) |
| B addon: LLM reranking | Dropped | All rerankers hurt or break even vs LTR |
| B addon: Neural LTR (ListNet MLP) | Dropped | Gradient signal too weak on large pools |
| C: TT v8 pool-aware negatives | Not started | Next up |

**Current best:** `models/ltr/ltr_phase_b_reg_nl31_lr0p08.txt`, 29 features,
`--tt_pool 2000`, full-dev nDCG@20 = 0.1653.

---

## Phase C: TT v8 — larger context window + LoRA fine-tuning (in progress, 2026-05-29)

### Root problem fixed

TT v6 uses `all-MiniLM-L6-v2` (256-token max). The rich query the system wants to encode
(latest user request + goal + culture + 4 track texts with tags + 3 prior turns) is
300-500 tokens median. MiniLM truncates from the right, cutting the user request when
it is placed last. v6 worked around this with a compact format that put the user request
first but omitted most conversational history. v8 removes this constraint entirely.

### Model: `intfloat/multilingual-e5-base`

| Property | TT v6 | TT v8 |
|---|---|---|
| Base model | all-MiniLM-L6-v2 | multilingual-e5-base |
| Architecture | BERT-6L | XLM-RoBERTa-base (12L) |
| Max tokens | 256 | 512 |
| Embedding dim | 384 | 768 |
| Total params | 22M | 279M |
| Fine-tuning | full (22M) | LoRA r=16 (885K = 0.32%) |
| Query prefix | none | `query: ` |
| Doc prefix | none | `passage: ` |

Why LoRA: full fine-tuning of 279M params OOMs on MPS (M4 16GB) because Adam optimizer
states alone require ~3x model weights (~3GB). LoRA drops optimizer memory to ~7MB.
Gradient checkpointing also required (`--gradient_checkpointing`).

Why not nomic-embed (8192-tok) or bge-base (512-tok, no prefix): both caused
`kIOGPUCommandBufferCallbackErrorOutOfMemory` on backward pass regardless of batch size
or gradient checkpointing. Root cause unclear (likely nomic-bert architecture triggers
Metal command buffer fragmentation). e5-base with LoRA does not reproduce this.

### Anchor format (no-truncation guarantee)

Data builder (`build_twotower_v8_data.py`) loads the E5 tokenizer and builds each
anchor greedily:

1. Core (always included, ~60-70 tokens): `query: {latest_user}` + goal + type +
   culture + age/country
2. Last 4 played tracks (full text with tags), most recent first — each added only if
   it fits within 510 remaining tokens
3. Last 3 prior text turns (user+assistant), most recent first — added only if budget
   remains

User request is always first, so if any truncation occurs (5% of samples have core
>510 tokens) it hits distant history, never the request.

Prefixes are baked into the JSONL: anchor starts with `query: `, positive/negatives
start with `passage: `. The inference script receives `--tt_query_prefix "query: "`.

### Training status (as of 2026-05-29)

Training data: 74,377 train / 5,272 valid examples (all 15K TRAIN sessions, excluding
2000 LTR seed sessions with `--exclude_seed 42 --exclude_n 2000`).

Eval loss checkpoints (still training):
| Step | Eval Loss |
|---|---|
| 500 | 0.677 |
| 1000 | 0.645 |
| 1500 | 0.632 |
| 2000 | 0.626 |

Consistently improving. Training at 16s/step (gradient checkpointing, batch_size=16,
grad_accum=4, effective batch 64). Expected finish: ~1.5h from step 2000.

### Scripts

- `scripts/train/build_twotower_v8_data.py` — E5 data builder with tokenizer-aware
  greedy anchor construction and no-truncation guarantee
- `scripts/train/train_twotower_lora.py` — LoRA trainer for XLM-R family; merges
  adapters on save so output is a vanilla SentenceTransformer
- `scripts/train/build_twotower_index.py` — add `--doc_prefix "passage: "` flag
- `scripts/inference/run_inference_fusion_recall_expansion.py` — add
  `--tt_query_prefix "query: " --tt_text_turns 3 --tt_hist_turns 4`

### Commands

```bash
# 1. Build training data (already done: data/twotower_v8/)
python scripts/train/build_twotower_v8_data.py \
  --out_dir data/twotower_v8 --hard_negs 5 \
  --exclude_n 2000 --exclude_seed 42

# 2. Train (in progress: models/twotower_v8/)
source .venv/bin/activate && python scripts/train/train_twotower_lora.py \
  --data_dir data/twotower_v8 --out_dir models/twotower_v8 \
  --epochs 2 --batch_size 16 --grad_accum 4 \
  --lr 1e-4 --warmup_steps 200 --gradient_checkpointing

# 3. Build index (after training completes)
source .venv/bin/activate && python scripts/train/build_twotower_index.py \
  --model models/twotower_v8/final --out_dir cache/twotower_v8 \
  --doc_prefix "passage: " --batch_size 32

# 4. Quick eval with existing LTR (old LTR weights, TT features will be miscalibrated
#    but gives a first-pass signal on whether recall improved)
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
python scripts/inference/run_inference_fusion_recall_expansion.py \
  --split test --tid phase_c_tt8_oldltr \
  --tt_model models/twotower_v8/final --tt_index cache/twotower_v8 \
  --tt_query_prefix "query: " --tt_text_turns 3 --tt_hist_turns 4 \
  --tt_pool 2000 --artist_expansion --last_nn_k 100 --last_nn_src 2 \
  --bm25_missing_floor 0.05 \
  --qwen_pool 500 --cf_pool 200 --session_mean_k 100 \
  --cooccur_table cache/cooccur/next_song_leakfree.npz --cooccur_ks 300,150,50 \
  --ltr_model models/ltr/ltr_phase_b_reg_nl31_lr0p08.txt

python scripts/inference/evaluate_local.py \
  --pred exp/inference/devset/phase_c_tt8_oldltr.json

# 5. Retrain LTR on v8 features (only if step 4 shows promise)
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
python scripts/inference/run_inference_fusion_recall_expansion.py \
  --split train --sessions 2000 --shuffle_seed 42 \
  --tid phase_c_ltr_features \
  --tt_model models/twotower_v8/final --tt_index cache/twotower_v8 \
  --tt_query_prefix "query: " --tt_text_turns 3 --tt_hist_turns 4 \
  --tt_pool 2000 --artist_expansion --last_nn_k 100 --last_nn_src 2 \
  --bm25_missing_floor 0.05 \
  --qwen_pool 500 --cf_pool 200 --session_mean_k 100 \
  --cooccur_table cache/cooccur/next_song_leakfree.npz --cooccur_ks 300,150,50 \
  --write_features exp/analysis/ltr_phase_c_train_features.npz

python scripts/train/train_ltr_lightgbm.py \
  --features exp/analysis/ltr_phase_c_train_features.npz \
  --out models/ltr/ltr_phase_c_nl31_lr0p08.txt \
  --n_folds 5 --num_leaves 31 --lr 0.08 --num_iter 1000 --early_stop 75 \
  --lambda_l2 0.1 --min_sum_hessian 0.1 --path_smooth 1.0 \
  --feature_fraction 0.8 --bagging_fraction 0.8 --truncation_level 30
```

### Gate

- Pool recall (v8) > 0.830 AND dev nDCG@20 > 0.1653 → retrain LTR on v8 features.
- If old LTR + v8 shows recall improvement but nDCG is flat: retrain LTR before
  concluding (old LTR was calibrated on v6 `tt_cos`/`tt_rank_sig` distributions).
- If pool recall is flat: the recall ceiling is a model-architecture constraint;
  accept 83% and focus elsewhere.

---

## Phase B: popularity + year features (concluded, 2026-05-28, best = 0.1653)

### What changed from Phase A (0.1646)

| Component | Phase A | Phase B |
|---|---|---|
| TT pool size | `--tt_pool 1000` | `--tt_pool 2000` |
| LTR features | 27 | 29 (+`popularity`, +`track_year`) |
| LTR booster | plain nl31 lr0.08 | regularized (lambda_l2=0.1, min_sum_hessian=0.1, path_smooth=1.0) |
| LLM rerank | none | TESTED, DROPPED (hurts) |
| Neural LTR | none | TESTED, DROPPED (gradient signal issue) |

### Results

- Phase B plain (no reg, 29-feat, tt_pool=2000): dev 0.1646 -- flat vs Phase A
- Phase B reg (l2+hessian+path_smooth): dev **0.1653** -- new best (+0.0007)
- Golden-200 verification: reg 0.1595 / Hit@20 542 vs Phase A 0.1582 / 528

### LLM reranking (dropped)

Tested three approaches; all failed to beat LTR:
1. CE v3 (bge-reranker-v2-m3, pool-aware negs, top-150): 0.1228 on dev -- large regression
2. Gemma-3-12b conservative prune (25→20, local LM Studio): 0.0812 vs baseline 0.0818 on 50-session pilot -- regression
3. Claude Opus listwise rerank: not tested (no API key)

Root cause: LightGBM LambdaMART with 29 engineered features already captures the signal
a reranker would use (tt_cos, cf_cos, dist_to_recent_mean dominate). Adding a reranker
on top adds noise without useful signal.

### Neural LTR (dropped)

ListNet MLP (256→128→64→1) on 29 features: loss stuck at log(pool_size) ≈ 7.96 across
all epochs. Root cause: `softmax(gains)` over ~2864-item pool with 1 positive gives
target probability ~1/2864 for most items; gradient at the positive is ~0.0006.
LightGBM LambdaMART handles this directly (pairwise, not listwise softmax).

---

## Phase A: Source-aware ranking (concluded, 2026-05-24, best = 0.1646)

LightGBM LambdaMART on 27 features (cosines + rank signals + source flags + collab).
Trained on 2000 TRAIN sessions (shuffle_seed=42), 5-fold session-stratified CV.
CV ndcg@20 = 0.3586 (std 0.0183). Dev nDCG@20 = 0.1646. Booster: `ltr_phase_a_nl31_lr0p08.txt`.

Full Phase A development history: `plan/archive/` (phases 01-07).
