# Current Best Iteration

Live snapshot. Update only when full 1000-session dev nDCG@20 strictly beats this.

## Best (as of 2026-06-13, v8d Tier 1 gpa-aware + entity features)

- **Dev nDCG@20: 0.1864** (+0.0116 over prior v8d 0.1748)
- nDCG@1 0.0716  |  nDCG@10 0.1634  |  catalog_div 0.4814  |  lex_div 0.2045
- Run id: `v8d_tier1_dev1000` (`exp/inference/devset/v8d_tier1_dev1000.json`)
- TT model: `models/twotower_v8d/final` (unchanged from prev best)
- TT index: `cache/twotower_v8d` (unchanged)
- Booster: `models/ltr/ltr_v8d_tier1_nl31_lr0p08.txt` (LightGBM LambdaMART,
  **60 features** (was 50), nl=31, lr=0.08, mean_iter=111, 5-fold CV ndcg@20=0.3152
  std=0.0061)
- LTR trained on 6K TRAIN sessions (shuffle_seed=42), `--skip_no_progress` +
  `--use_goal_progress`, 24,718 clean groups after all-zero filter
- Co-occurrence table: `cache/cooccur/next_song_leakfree_6k_excluded.npz`
- Pool recall: 87.42% (dev 1000 sessions)

## Inference flags active

The flags that turn on the new behaviours:

| Flag | What it does |
|---|---|
| `--anchor_v8d` | Role-tagged anchor format |
| `--use_goal_progress` | H1 (filter rejected from seed history) + activates H2 features |
| `--goal_substitute_positive` | H3a: substitute goal slot with last MOVES track |
| `--rejection_drop_threshold 2` | H3b: drop goal entirely after 2 consecutive DOES_NOT |

## Tier 1 changes vs prior best (v8d at 0.1748)

| ID | Change | Effect on nDCG@20 (dev) |
|---|---|---:|
| T1.1 | `--use_goal_progress` enabled at inference + in training dump | activates 4 previously-zero H2 features + H1/H3 retrieval modulation |
| T1.2 | 7 new keyword-bucket features (era, genre, mood, instrument + per-candidate genre/era match) | small contribution; `user_has_negation`, `user_has_followup`, `q_has_era` ended up zero-importance |
| T1.3 | `artist_id`-based within-artist grouping (was lowercased `artist_name` strings) | disambiguates 2,163 distinct artists that shared name strings |
| T1.4 | 3 new album_id features (`same_album_as_last_history`, `n_same_album_in_history`, `album_in_recent_window`) | encodes LFM-2b session-pool coherence revealed by paper |
| (combined) | All four stacked into one LTR retrain | **+0.0116 dev / +0.0192 BLINDPROXY_MIXED / +0.034 Hit@20** |

## Blind sim scores (Tier 1)

| Spec | Pairs | v8d Tier 1 | v8d (prev best) |
|---|---:|---:|---:|
| DEV_BLINDSIM_MIXED (992 MOVES-positive turns) | 992 | **0.1988** | 0.1796 |
| BLINDPROXY turn-1 only | 442 | 0.1796 | 0.1799 |
| Hit@20 (MIXED) | 992 | **0.400** | 0.366 |

Per-turn breakdown (BLINDPROXY_MIXED):

| Turn | v8d (prev) | v8d Tier 1 | Δ |
|---|---:|---:|---:|
| 1 | 0.2308 | 0.2281 | -0.003 (H1/H2/H3 need history → no lift expected) |
| 2 | 0.1857 | 0.1985 | +0.013 |
| 3 | 0.1405 | 0.1555 | +0.015 |
| 4 | 0.1756 | 0.1843 | +0.009 |
| 5 | 0.1714 | **0.2123** | **+0.041** |
| 6 | 0.1722 | **0.2050** | **+0.033** |
| 7 | 0.1770 | **0.2112** | **+0.034** |

The gpa-aware machinery delivers the biggest gains on turns 5–7 where history is rich
and rejected/accepted artist signals matter most.

## Retrieval pool

```
BM25@500
+ artist expansion (popularity-sorted catalog, --artist_cap 50)
+ TT-v8d@2000 (role-tagged anchor)
+ last-track-NN@100 in TT space (last_nn_src=2)
+ Qwen-meta global top-500
+ CF global top-200 (warm users only)
+ session-mean-vector NN top-100
+ co-occurrence top-300/150/50 (leakfree_6k_excluded table)

inference modulations active:
  H1: rejected tracks stripped from seed history (--use_goal_progress)
  H3a: most-recent MOVES track substituted into goal slot (--goal_substitute_positive)
  H3b: goal dropped after 2 consecutive rejections (--rejection_drop_threshold 2)
```

## Reproduction

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
python scripts/inference/run_inference_fusion_recall_expansion.py \
  --tid v8d_tier1_dev1000 \
  --tt_model models/twotower_v8d/final --tt_index cache/twotower_v8d \
  --anchor_v8d \
  --tt_pool 2000 --artist_expansion --last_nn_k 100 --last_nn_src 2 \
  --bm25_missing_floor 0.05 \
  --qwen_pool 500 --cf_pool 200 --session_mean_k 100 \
  --cooccur_table cache/cooccur/next_song_leakfree_6k_excluded.npz \
  --cooccur_ks 300,150,50 \
  --use_goal_progress --goal_substitute_positive --rejection_drop_threshold 2 \
  --ltr_model models/ltr/ltr_v8d_tier1_nl31_lr0p08.txt
```

Feature re-dump (training):

```bash
python scripts/inference/run_inference_fusion_recall_expansion.py \
  --split train --sessions 6000 --shuffle_seed 42 \
  --tt_model models/twotower_v8d/final --tt_index cache/twotower_v8d \
  --anchor_v8d \
  --tt_pool 2000 --artist_expansion --last_nn_k 100 --last_nn_src 2 \
  --bm25_missing_floor 0.05 \
  --qwen_pool 500 --cf_pool 200 --session_mean_k 100 \
  --cooccur_table cache/cooccur/next_song_leakfree_6k_excluded.npz \
  --cooccur_ks 300,150,50 \
  --skip_no_progress --use_goal_progress \
  --write_features exp/analysis/ltr_v8d_tier1_6k_features.npz
```

LTR retrain:

```bash
python scripts/train/train_ltr_lightgbm.py \
  --features exp/analysis/ltr_v8d_tier1_6k_features.npz \
  --out models/ltr/ltr_v8d_tier1_nl31_lr0p08.txt \
  --n_folds 5 --num_leaves 31 --lr 0.08 --num_iter 1000 --early_stop 75 \
  --lambda_l2 0.1 --min_sum_hessian 0.1 --path_smooth 1.0 \
  --feature_fraction 0.8 --bagging_fraction 0.8 --truncation_level 30
```

## Known gaps / next steps

Tier 1 is shipped. Next moves from the architectural plan (`plan/we-have-discovered-this-dapper-hearth.md`):

- **T2.1 — Cross-encoder rerank** on top of Tier 1 base (proven +0.049 on Phase D). Model `models/crossencoder_v3` already trained; `rescore_with_crossencoder.py` ready.
- **T2.2 — Thought-augmented TT training** (paper insight): include Recsys thought text in positive at train time. Needs TT retrain (~12 hours).
- **T2.3 — Listener-simulation reranker**: predict P(MOVES | context, candidate) using gpa labels.
- **T2.4 — LLM entity extraction** from Q_t + R_t (Gemma local).

## Previous bests

| Date | nDCG@20 | TT model | LTR | Note |
|---|---|---|---|---|
| 2026-06-07 | 0.1748 | TT v8d (r=32, role-tagged anchor) | 50-feat ltr_v8d | v8d encoder upgrade |
| 2026-06-06 | 0.1729 | TT v8c (r=32) | 50-feat ltr_v8c_fixed | gpa-fix + co-occur leakage fix |
| 2026-05-29 | 0.1684 | TT v6 | 39-feat ltr_phase_d | Phase D feature engineering |
| 2026-05-28 | 0.1653 | TT v6 | 29-feat ltr_phase_b | Phase B regularization |
| 2026-05-27 | 0.1646 | TT v6 | 27-feat ltr_phase_a | Phase A LTR baseline |
| 2026-05-24 | 0.1609 | TT v6 | 15-feat ltr_v3 | LTR v3 sweep |
| 2026-05-15 | 0.1601 | TT v6 | 44-tree ltr_v2 | LTR v2 |
| 2026-05-11 | 0.1533 | TT v6 | v13_tuned linear | NN@100 pool expansion |
| 2026-05-05 | 0.1519 | TT v6 | v13_tuned linear | BM25@500 only |
