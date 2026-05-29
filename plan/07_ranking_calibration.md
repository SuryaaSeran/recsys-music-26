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

## Phase C: TT v8 — pool-aware negatives + all TRAIN sessions (next)

### What the current TT model is and why it can be improved

**Current model: TT v6** (`models/twotower_v6/final`)
- Base: `sentence-transformers/all-MiniLM-L6-v2`, 384 dim, 256 token max
- Training data: `data/twotower_v6/` — 85,559 train / 6,072 valid examples
- Session coverage: subset of TRAIN sessions (not all 15,199)
- Anchor: `{latest user turn} {goal} {culture/type} {last 2 played tracks}`
- Track text: `name | artist | album | top-12 tags | year`
- Negatives: 2 random + 2 BM25@100 + 1 rejected (DOES_NOT_MOVE_TOWARD_GOAL)
- Loss: MultipleNegativesRankingLoss, 2 epochs, lr=2e-5, batch=32, 43 min total
- Pool recall at tt_pool=2000: ~83% (Phase B pool)

**TT v7 (tried, failed — dev nDCG@20 = 0.1584):**
- Base: `Qwen/Qwen3-Embedding-0.6B` (1024 dim, bigger model)
- All 15K TRAIN sessions + 47K cold catalog pairs
- 1 epoch only, 22K session examples (less than v6's 85K)
- Root cause: bigger model alone doesn't help; fewer session examples + cold-track
  dilution + 1 epoch = less session-signal per parameter

**Why TT v8 might help:**
Pool recall ceiling is 83%. TT is the largest single recall source. Better negatives
(sampled from the actual pool, not random BM25) sharpen the cosine margin so gold
ranks higher within the pool. The fix is not a bigger model; it's pool-aware negatives.

### Three changes from v6 → v8

| | v6 | v8 |
|---|---|---|
| Base model | all-MiniLM-L6-v2 (384d) | same |
| Sessions | subset of TRAIN | all 15,199 TRAIN; cap 3 turns/session; exclude LTR seed 2000 |
| Negatives | 2 random + 2 BM25@100 + 1 rejected | 2 random + 3 Phase B pool samples (not gold) |
| Epochs | 2 | 2 |

Pool-aware negatives: for each music turn in TRAIN, build the Phase B pool (BM25@500 +
artist + TT-v6@2000 + NN@100 + Qwen@500 + CF@200 + co-occurrence), sample 3 non-gold
candidates from the pool. These are the same hard candidates the model sees at eval time.

Excluding LTR seed sessions (shuffle_seed=42, first 2000): keeps the LTR booster
leak-free when we retrain it on v8-based features.

### Files

- `scripts/train/build_twotower_v8_data.py` (new) — derived from v7 data builder:
  - All TRAIN sessions, max 3 turns each, exclude seed 2000
  - Pool-aware hard negatives (import pool helpers from inference script or inline)
  - Output: `data/twotower_v8/{train,valid}.jsonl`
- `scripts/train/train_twotower_qwen.py` (reuse, same base model)
- `scripts/inference/build_twotower_index.py` (reuse)

### Commands

```bash
# 1. Build training data
python scripts/train/build_twotower_v8_data.py \
  --out_dir data/twotower_v8 \
  --hard_negs 3 --max_turns_per_session 3 \
  --exclude_seed 42 --exclude_n 2000

# 2. Train
python scripts/train/train_twotower_qwen.py \
  --base_model sentence-transformers/all-MiniLM-L6-v2 \
  --data_dir data/twotower_v8 \
  --out_dir models/twotower_v8 \
  --epochs 2 --lr 2e-5 --batch_size 32 --warmup_steps 200

# 3. Build index
python scripts/inference/build_twotower_index.py \
  --model models/twotower_v8/final \
  --out cache/twotower_v8

# 4. Dev inference (Phase B pool + Phase B reg booster, just swap TT index)
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
python scripts/inference/run_inference_fusion_recall_expansion.py \
  --tid phase_c_tt8 \
  --tt_model models/twotower_v8/final --tt_index cache/twotower_v8 \
  --tt_pool 2000 --artist_expansion --last_nn_k 100 --last_nn_src 2 \
  --bm25_missing_floor 0.05 \
  --qwen_pool 500 --cf_pool 200 --session_mean_k 100 \
  --cooccur_table cache/cooccur/next_song_leakfree.npz --cooccur_ks 300,150,50 \
  --ltr_model models/ltr/ltr_phase_b_reg_nl31_lr0p08.txt

python scripts/inference/evaluate_local.py --pred exp/inference/devset/phase_c_tt8.json
```

### Gate

- Pool recall (v8) > 0.830 AND dev nDCG@20 > 0.1653 → retrain LTR on v8 features,
  re-evaluate.
- Pool recall flat or worse → recall ceiling is a model-architecture constraint; accept
  83% and focus elsewhere.

### Cost

~1 h data build + ~45 min training + ~40 min index build + ~35 min dev inference = ~3 h.

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
