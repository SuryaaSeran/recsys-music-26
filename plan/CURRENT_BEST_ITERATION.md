# Current Best Iteration

Live snapshot. Update only when full 1000-session dev nDCG@20 strictly beats this.

## Best (as of 2026-05-29, Phase D 39-feat LTR)

- **Dev nDCG@20: 0.1684**
- nDCG@1 0.0599  |  nDCG@10 0.1464  |  catalog_div 0.5159  |  lex_div 0.2086
- Script: `scripts/inference/run_inference_fusion_recall_expansion.py` with Phase D flags (`--tt_pool 2000`)
- Run id: `phase_d_baseline_dev1000` (`exp/inference/devset/phase_d_baseline_dev1000.json`)
- Booster: `models/ltr/ltr_phase_d_nl31_lr0p08.txt` (LightGBM LambdaMART,
  39 features, num_leaves=31, lr=0.08, lambda_l2=0.1,
  min_sum_hessian=0.1, path_smooth=1.0, feature_fraction=0.8,
  bagging_fraction=0.8, truncation_level=30, mean_iter=81) trained on 2000 random
  TRAIN-split sessions (`--shuffle_seed 42`). 5-fold CV ndcg@20 = 0.3752
  (std=0.0045). No dev-set leakage.
- One-line reason it beat prior best: 10 new Phase D features (n_sources is
  dominant at gain 497k -- multi-source retrieval agreement) + pool recall lifted
  from 83.03% to 87.21% (+4.2pp). Poly variant (39+14 interactions) scored 0.1678
  -- baseline wins.

### New features vs Phase B (39 vs 29)

| Feature | Gain | Note |
|---|---|---|
| `n_sources` | 497,487 | Count of retrieval sources that agreed on candidate -- dominant |
| `tt_rank_sig` | 151,416 | (existing, #2) |
| `tt_cos` | 84,283 | (existing, #3) |
| `cf_cos` | 73,924 | (existing, #4) |
| `popularity_pctile` | 19,719 | NEW -- normalized rank percentile, better than raw |
| `turn_number` | 6,737 | NEW |
| `cf_dist_to_recent_mean` | 4,495 | NEW |
| `cf_dist_to_last` | 3,461 | NEW |
| `tag_overlap_count` | 1,614 | NEW |
| `query_len_tokens` | 1,266 | NEW |
| `years_since_release` | 924 | NEW |
| `history_len` | 754 | NEW -- replaces binary cold_user (which is now zero-importance) |
| `goal_category` | 228 | NEW |

Zero-importance (can be pruned): `nn_origin`, `cold_user`, `qm_only`.

### Retrieval pool

```
BM25@500
+ artist expansion (popularity-sorted catalog, --artist_cap 50)
+ TT-v6@2000
+ last-track-NN@100 in TT space (last_nn_src=2)
+ Qwen-meta global top-500
+ CF global top-200 (warm users only)
+ session-mean-vector NN top-100
+ co-occurrence top-300/150/50 (last 3 played tracks, leakfree table)
```

Pool recall: 0.8721 (up from 0.8303 in Phase B).

### Reproduction

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
python scripts/inference/run_inference_fusion_recall_expansion.py \
  --tid phase_d_baseline_dev1000 \
  --tt_model models/twotower_v6/final --tt_index cache/twotower_v6 \
  --tt_pool 2000 --artist_expansion --last_nn_k 100 --last_nn_src 2 \
  --bm25_missing_floor 0.05 \
  --qwen_pool 500 --cf_pool 200 --session_mean_k 100 \
  --cooccur_table cache/cooccur/next_song_leakfree.npz --cooccur_ks 300,150,50 \
  --ltr_model models/ltr/ltr_phase_d_nl31_lr0p08.txt
```

Feature re-dump + booster retraining:

```bash
# 1) feature dump from TRAIN sessions (39 feat, tt_pool 2000, TT v6)
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
python scripts/inference/run_inference_fusion_recall_expansion.py \
  --tid phase_d_ltr_features --split train --sessions 2000 --shuffle_seed 42 \
  --tt_model models/twotower_v6/final --tt_index cache/twotower_v6 \
  --tt_pool 2000 --artist_expansion --last_nn_k 100 --last_nn_src 2 \
  --bm25_missing_floor 0.05 \
  --qwen_pool 500 --cf_pool 200 --session_mean_k 100 \
  --cooccur_table cache/cooccur/next_song_leakfree.npz --cooccur_ks 300,150,50 \
  --write_features exp/analysis/ltr_phase_d_train_features.npz

# 2) train 39-feat LightGBM LambdaMART
python scripts/train/train_ltr_lightgbm.py \
  --features exp/analysis/ltr_phase_d_train_features.npz \
  --out models/ltr/ltr_phase_d_nl31_lr0p08.txt \
  --n_folds 5 --num_leaves 31 --lr 0.08 --num_iter 1000 --early_stop 75 \
  --lambda_l2 0.1 --min_sum_hessian 0.1 --path_smooth 1.0 \
  --feature_fraction 0.8 --bagging_fraction 0.8 --truncation_level 30
```

### Blind safety check

Pool-1000 (tt_pool=1000) golden-200 nDCG@20 = **0.1609** -- fails gate vs Phase A
(0.1646). Same pattern as Phase B: gains are pool-size-dependent. Use Phase A
pool config for blind submissions until v8 re-dump + retrain resolves this.

### TT v8b 6K results (2026-05-31, below gate)

6K session dump (skip_no_progress) → 27,166 clean groups → max_groups=12K subsample for LTR.
Root cause of regression vs v6: TT v8b pool recall 86.02% vs v6 87.21% (-1.2pp).
H2 features (Phase E) added but did not recover gap — retrieval ceiling is the bottleneck.

| Mode | Phase D v6 (best) | v8b 6K no H1H3 | v8b 6K H1+H3 | Phase E H1+H2+H3 |
|---|---:|---:|---:|---:|
| All turns | **0.1684** | 0.1608 | 0.1615 | 0.1583 |
| MOVES only | 0.1662 | 0.1566 | 0.1580 | 0.1543 |
| Last turn | **0.1650** | 0.1503 | 0.1477 | 0.1479 |
| Last+progress | **0.1731** | 0.1532 | 0.1526 | 0.1540 |

### Blind A scoreboard (best blind: v10 H1+H3)

| Version | Blind nDCG@20 | LLM Judge | Composite | Dev nDCG@20 | Retrieval | Response |
|---|---:|---:|---:|---:|---|---|
| **v10 H1+H3** | **0.3701** | **3.60** | **0.4504** | 0.1615 | v8b H1+H3 42-feat LTR | Gemma-3-12b native |
| v10c | 0.3701 | TBD | TBD | 0.1615 | v8b H1+H3 42-feat LTR | Gemma-3-12b v07-prompt |
| **v07** | **0.3164** | **4.40** | **0.4837** | **0.1684** | Phase D 39-feat LTR | Gemma-3-12b v07-prompt |
| v04 | 0.3709 | 1.10 | 0.2771 | 0.1646 | Phase A 27-feat LTR | DeepSeek V4 Flash |
| v06 | 0.3000 | — | — | 0.1653 | Phase B 29-feat LTR | Gemma-3-12b |

**Best blind nDCG@20: 0.3701 (v10 H1+H3)** — v8b retrieval with H1 seed filtering and
H3 goal substitution. Beats v07 by +0.0537 on retrieval but loses judge points (3.60 vs
4.40), so composite is 0.4504 vs v07's 0.4837.

**Best composite: 0.4837 (v07)** — Phase D 39-feat LTR + Gemma-3-12b v07-prompt.
Composite is judge-dominated; v07 wins despite weaker retrieval.

Next blind submissions ready:
- v12 = v8b H1+H3 retrieval + fresh Gemma native responses (asterisks stripped). Same
  retrieval as v10. At `submissions/v12_v8b_6k_H1H3_gemma12b_native/submission.zip`.
- v13 = Phase A pool retrieval + Gemma v07-prompt stripped. At
  `submissions/v13_phase_a_stripped_resp/submission.zip`.

Below-gate experiments (do not submit):
- Phase E 46-feat (H2 history features): dev 0.1583.
- Phase F 44-feat (n_sources_norm + log1p_n_sources): dev 0.1603, BA100 0.1108.
- Phase F 45-feat (+bm25_top1): dev 0.1602, BA100 0.1045 (sparsity hurts).
- Phase F adaptive pool, infer_progress, entity BM25: no measurable gains.
- Blend D(T1-3)+A(T4+) zero-retrain config: BA100 0.1213 last-turn, useful as proxy
  baseline only.

## Evaluation standard

The official evaluator is at `music-crs-evaluator/` (mirrored in this repo).
Numbers below are produced by that evaluator. Our local
`scripts/inference/evaluate_local.py` mirrors it (per-turn-number macro-mean,
no-duplicate check, plus catalog/lexical diversity). Reproduce ours with:

```bash
python scripts/inference/evaluate_local.py --pred exp/inference/devset/<tid>.json
```

## Organizer baselines (official scores, devset 1000 sessions)

Source: `music-crs-evaluator/exp/scores/devset/{random,popularity,llama1b_bm25_devset}.json`.

| Baseline | nDCG@1 | nDCG@10 | nDCG@20 | Catalog div. | Lexical div. |
|---|---:|---:|---:|---:|---:|
| Random              | 0.0000 | 0.0001 | 0.0001 | 0.9652 | 0.0000 |
| Popularity          | 0.0005 | 0.0018 | 0.0024 | 0.0004 | 0.0000 |
| LLaMA-1B + BM25     | 0.0098 | 0.0627 | 0.0815 | 0.3795 | 0.2558 |

## Response generation

Two paths are used depending on the run target.

### Dev (inside the inference loop)

`run_inference_fusion_recall_expansion.py` emits a fixed template per turn:

```python
response = f'I recommend "{name}" by {artist} based on your request.'
```

Lexical diversity (Distinct-2) on this template scores ~0.18 on the official
evaluator -- below the LLaMA-1B + BM25 baseline at 0.256.

### Blind (post-process step)

For blind submissions, `scripts/inference/generate_responses_blind.py` rewrites
each turn's response with an LLM. Active model: Gemma-3-12b via LM Studio native
API (`generate_responses_lmstudio.py --native_api`). Prompt updated 2026-05-28.

## Previous bests

| Date | nDCG@20 | Pool | Rescore | Note |
|---|---|---|---|---|
| 2026-05-28 | 0.1653 | Phase B pool (tt_pool=2000) | LTR Phase B reg (nl31 lr0.08, 29 feat) | `models/ltr/ltr_phase_b_reg_nl31_lr0p08.txt`. Poly variant 0.1678 also tested today but baseline wins. |
| 2026-05-27 | 0.1646 | Phase A pool (tt_pool=1000) | LTR Phase A (nl31 lr0.08, 27 feat) | `models/ltr/ltr_phase_a_nl31_lr0p08.txt`. |
| 2026-05-24 | 0.1609 | BM25@500 + artist + TT@1000 + NN@100 | LTR v3 (nl31 lr0.08, 73 trees, 15 feat) | `models/ltr/sweep/ltr_nl31_lr0p08.txt`. |
| 2026-05-15 | 0.1601 | BM25@500 + artist + TT@1000 + NN@100 | LTR v2 (nl63 lr0.05, 44 trees) | `models/ltr/ltr_v2_train.txt`. |
| 2026-05-11 | 0.1533 | BM25@500 + artist + TT@1000 + NN@100 | v13_tuned linear | Pool expansion (NN@100) over BM25-only pool. |
| 2026-05-05 | 0.1519 | BM25@500 only | v13_tuned weights | Blind file: `blind_a_fusion_v13_tuned_qwen.json`. |
