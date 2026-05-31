# Plan: Feature Engineering v2 + Next Directions (Phase D)

## Goal

Beat dev nDCG@20 0.1653 and blind nDCG@20 0.37 through feature engineering,
training improvements, and response generation. Deadline: June 30, 2026.

## Current State (updated 2026-05-30)

- Dev nDCG@20: **0.1684** (39-feat Phase D LTR, TT v6, tt_pool=2000) — current best
- Blind composite: **0.4837** (v07, Gemma-3-12b, judge 4.4/5)
- Pool recall: 87.21% on train; TT v8b gets 86.48% on dev
- TT v8b: trained (drop_rejected, 3 hard negs), index at `cache/twotower_v8b`
- 42 features: 39 Phase D + 3 Phase D2 (user_has_negation, user_has_followup, query_track_tag_sim)
- H1+H3 implemented: `--use_goal_progress --rejection_drop_threshold 3 --goal_substitute_positive`
- Evaluator: `--progress_only`, `--last_turn_only` flags added for clean signal
- **Running now:** 6K session feature dump with TT v8b + skip_no_progress
- Next: retrain LTR → eval all 4 modes → gate check

### v8b eval results (2026-05-30, below gate)

| Mode | Phase D (v6) | v8b+42feat LTR | v8b+H1+H3 | Note |
|---|---:|---:|---:|---|
| All turns (8K) | 0.1684 | 0.1682 | 0.1672 | gate: >0.1684 |
| MOVES only (6184) | 0.1662 | 0.1666 | 0.1665 | clean signal |
| Last turn (1K) | 0.1650 | 0.1600 | 0.1591 | |
| Last+progress (836) | 0.1731 | 0.1643 | **0.1655** | H1+H3 help here |

Root cause of regression: LTR trained on only 7,953 clean turns (early stop iter=78).
Fix: 6K sessions → ~21K clean turns. Then 15K → ~53K if needed.

## Key Constraints

1. Phase B features (popularity, track_year, tt_pool=2000) hurt blind 0.37 -> 0.30.
   New features must be structurally robust, not distribution-dependent.
2. Competition scores on 4 dimensions: nDCG@20, catalog diversity, lexical diversity,
   LLM-as-Judge. We are currently optimizing only nDCG@20.
3. Blind B includes cold-start stress test. CF-dependent features must degrade gracefully.
4. 391 turns (4.9%) are "truly unreachable" -- gold track not in any signal's top-5000.
   These are a hard ceiling.

---

## Track 1: LTR Feature Engineering (DONE, 2026-05-29)

### Results

- Baseline 39-feat: golden-200 0.1641, full-dev **0.1684** (beats Phase B 0.1653)
- Poly 39+14-feat: golden-200 0.1614, full-dev 0.1678 (worse than baseline -- dropped)
- Pool recall: 87.21% (up from 83.03%)
- Pool-1000 blind safety: 0.1609 (fails vs Phase A 0.1646 -- same pattern as Phase B)
- Winner: `models/ltr/ltr_phase_d_nl31_lr0p08.txt`
- Top feature: `n_sources` (gain 497k, 3x second place `tt_rank_sig`)
- Zero-importance: `nn_origin`, `cold_user`, `qm_only`

### Steps

1. [done] Code changes to inference script + LTR trainer
2. [done] Dump features from TRAIN sessions (2000, seed 42)
3. [done] Train LTR baseline: 39 features -- CV nDCG@20 0.3752, full-dev 0.1684
4. [skip] soft_labels -- re-dump required, deferred to v8 cycle
5. [done] Train LTR + poly_feats -- full-dev 0.1678, baseline wins
6. [skip] soft_labels + poly_feats -- deferred
7. [done] Evaluate on golden-200 + full dev, promote baseline
8. [done] Feature importance -- see CURRENT_BEST_ITERATION.md

## Track 1b: TT v8b + 42-feat progress-aware LTR (ACTIVE, 2026-05-30)

TT v8b: multilingual-e5-base LoRA r=16, trained with `--drop_rejected` + 3 hard negs.
42 features: 39 Phase D + 3 user-intent proxies (Phase D2).
Progress-aware training: only MOVES_TOWARD_GOAL + None/missing turns used (`--skip_no_progress`).
H1+H3 inference flags: seed filtering + goal-slot modulation.

### Steps

1. [done] Build TT v8b data with `--drop_rejected` (56,874 clean examples)
2. [done] Train TT v8b LoRA (2 epochs, 3 hard negs, final loss 1.979)
3. [done] Build TT v8b index (`cache/twotower_v8b`, 138MB, 47K tracks)
4. [done] Add Phase D2 features (42 total) + H1+H3 flags to inference script
5. [done] Add `--progress_only`, `--last_turn_only` to evaluator
6. [done] Add all-zero group filter to LTR trainer
7. [done] Store `cand_ids` in feature dump sidecar (enables incremental feature augmentation)
8. [done] 2K session dump + LTR retrain — below gate (0.1682). Root cause: only 7,953 clean turns, early stop iter=78.
9. **[running]** 6K session dump (`exp/analysis/ltr_phase_d_v8b_6k_features.npz`, ~3.3hrs)
10. [ ] Retrain LTR on 6K features → expect ~21K clean turns, later early stopping
11. [ ] Dev eval (4 modes) — gate: all-turns nDCG@20 > 0.1684
12. [ ] If passes: update CURRENT_BEST_ITERATION, build blind submission
13. [ ] If still below gate: try 15K sessions or isolate TT v8b regression

### Incremental feature augmentation (for future phases)

**Key insight:** if the TT model is unchanged, new features can be appended to an existing
dump without re-running the full inference pipeline. The sidecar now stores `cand_ids` per
turn (added 2026-05-30). Future steps for new-feature-only additions:
1. Load existing NPZ + sidecar (with `cand_ids`)
2. Compute only the new feature columns per candidate
3. `np.hstack` onto existing X, update `feature_cols`, save

**When re-dump IS required:** if TT model changes (all TT-derived features change), or if
pool composition changes (H1 seed filtering changes which candidates are retrieved).

### Reproduction (6K session run)

```bash
# Feature dump (running)
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
python scripts/inference/run_inference_fusion_recall_expansion.py \
  --tid phase_d_v8b_6k_ltr_features \
  --split train --sessions 6000 --shuffle_seed 42 \
  --tt_model models/twotower_v8b/final --tt_index cache/twotower_v8b \
  --tt_query_prefix "query: " \
  --tt_pool 2000 --artist_expansion --last_nn_k 100 --last_nn_src 2 \
  --bm25_missing_floor 0.05 \
  --qwen_pool 500 --cf_pool 200 --session_mean_k 100 \
  --cooccur_table cache/cooccur/next_song_leakfree.npz --cooccur_ks 300,150,50 \
  --skip_no_progress \
  --write_features exp/analysis/ltr_phase_d_v8b_6k_features.npz

# LTR retrain (after dump)
# Sparsity note: 1 positive per ~700 candidates (0.14% rate). LambdaMART handles this
# via pairwise gradients (700 pairs per group), NOT via scale_pos_weight/focal loss.
# Tighter regularization vs 2K run: lambda_l2 0.1->0.5, min_data_in_leaf 20->50,
# min_sum_hessian 0.1->0.5. Previous run early-stopped at iter=78 (overfitting signal).
python scripts/train/train_ltr_lightgbm.py \
  --features exp/analysis/ltr_phase_d_v8b_6k_features.npz \
  --out models/ltr/ltr_phase_d_v8b_6k_nl31_lr0p08.txt \
  --n_folds 5 --num_leaves 31 --lr 0.08 --num_iter 1000 --early_stop 75 \
  --lambda_l2 0.5 --min_sum_hessian 0.5 --min_data_in_leaf 50 --path_smooth 1.0 \
  --feature_fraction 0.8 --bagging_fraction 0.8 --truncation_level 30

# Dev eval (all 4 modes)
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
python scripts/inference/run_inference_fusion_recall_expansion.py \
  --tid phase_d_v8b_6k_dev1000 \
  --tt_model models/twotower_v8b/final --tt_index cache/twotower_v8b \
  --tt_query_prefix "query: " --tt_pool 2000 \
  --artist_expansion --last_nn_k 100 --last_nn_src 2 \
  --bm25_missing_floor 0.05 \
  --qwen_pool 500 --cf_pool 200 --session_mean_k 100 \
  --cooccur_table cache/cooccur/next_song_leakfree.npz --cooccur_ks 300,150,50 \
  --ltr_model models/ltr/ltr_phase_d_v8b_6k_nl31_lr0p08.txt
```

---

## Track 1c: Phase E — Goal Progress at Inference (PLANNED)

Implements Vedanth's `origin/vedanth/plan-8` proposals. Run AFTER Track 1b passes gate.

### What's already done vs what's new

| Change | Status |
|---|---|
| LTR: rejected gold label=0 (`--progress_aware` / `--skip_no_progress`) | Done |
| TT: drop rejected turns (`--drop_rejected`) | Done |
| Inference proxies (user_has_negation, user_has_followup, query_track_tag_sim) | Done (42 feat) |
| H1: filter rejected tracks from NN/session-mean/BM25 seeds (`--use_goal_progress`) | **Implemented, not yet in training** |
| H3: goal-slot modulation (`--rejection_drop_threshold`, `--goal_substitute_positive`) | **Implemented, not yet in training** |
| H2: 4 history features (sim_to_pos/neg_hist_mean, artist_in_rejected_set, n_rejected) | Not yet |

### H2 features to add (42 → 46)

- `sim_to_pos_hist_mean`: TT cosine to mean embedding of MOVES_TOWARD prior tracks
- `sim_to_neg_hist_mean`: TT cosine to mean embedding of DOES_NOT_MOVE prior tracks
- `artist_in_rejected_set`: 1.0 if candidate artist matches any prior rejected track's artist
- `n_rejected_in_history`: count of DOES_NOT_MOVE turns so far (clipped at 10)

### Steps

1. [ ] Add H2 features to FEATURE_COLS + candidate scoring loop in inference script
2. [ ] Feature dump with H1+H3+H2 active (use `--use_goal_progress --skip_no_progress`)
   — since H1 changes pool composition, full re-dump required (cannot use incremental augmentation)
3. [ ] Retrain LTR on 46-feature matrix
4. [ ] Dev eval H1+H2+H3 combined vs baseline
5. [ ] Ablation table (rows A-F per Vedanth's plan)
6. [ ] Gate: nDCG@20 > current best

---

## Track 2: Recall Improvement (2a + 2b IMPLEMENTED, 2026-05-29)

Pool recall is 83.03%. The 17% miss rate is split into:
- ~12% "soft unreachable": gold exists in some signal's top-5000 but not in our pool config
- ~5% "truly unreachable": gold not in any signal's top-5000 at all

### Implemented

**2a. BM25 sharp query — `--bm25_sharp_pool N`** [DONE, commit f89c9cf]
Second BM25 retrieval using only `latest_user + goal` (no track history, no text turns).
Targets mood/vibe turns where history text dilutes mood keywords.
Expected lift: +1-2% recall on mood/lyrics buckets. Cost: one extra BM25 call per turn.

**2b. Qwen-lyrics pool expansion — `--ql_pool N`** [DONE, commit f89c9cf]
Qwen-lyrics was scoring-only; now also expands the candidate pool. `ql_all` was already
computed when `w_qwen_lyrics > 0`; now also triggered when `ql_pool > 0`. Audit shows
6.7% of BM25 misses rescued at top-500.
Expected lift: +1-2% recall. Cost: ~N extra candidates per turn.

Suggested values to test: `--ql_pool 200 --bm25_sharp_pool 200`.
Both use `bm25_missing_floor` for fusion scoring, consistent with other pool sources.

### Not yet implemented

**2c. Tag-based retrieval**
Jaccard similarity between conversation tags and candidate `tag_list`. Targets "user
described a genre with different words than the metadata" gap.

**2d. Multi-query TT**
Two TT queries per turn: current full query + "what came before" (last 2 played tracks
only). Targets history-driven turns where the user says little.

### Steps

1. [ ] Run recall audit: compare current pool vs `+ql_pool 200 +bm25_sharp_pool 200`
       on golden-200. Measure: pool recall %, mean pool size, nDCG@20 with old LTR.
2. [ ] If either source lifts recall: integrate into production eval command
3. [ ] Consider 2c/2d only if pool recall still below 0.85 after 2a+2b

```bash
# Recall audit command (golden-200, ~7 min)
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
python scripts/inference/run_inference_fusion_recall_expansion.py \
  --tid phase_d_recall_audit \
  --session_ids_file plan/GOLDEN_HOLDOUT_SESSIONS.json \
  --tt_model models/twotower_v6/final --tt_index cache/twotower_v6 \
  --tt_pool 2000 --ql_pool 200 --bm25_sharp_pool 200 \
  --artist_expansion --last_nn_k 100 --last_nn_src 2 \
  --bm25_missing_floor 0.05 \
  --qwen_pool 500 --cf_pool 200 --session_mean_k 100 \
  --cooccur_table cache/cooccur/next_song_leakfree.npz --cooccur_ks 300,150,50 \
  --ltr_model models/ltr/ltr_phase_b_reg_nl31_lr0p08.txt
```

---

## Track 3: Response Generation (ACTIVE)

### Blind results so far (2026-05-29)

| Version | LexDiv | LLM Judge | Composite | Note |
|---|---:|---:|---:|---|
| v04 DeepSeek (generic, 0/80 named) | 0.8099 | 1.10 | 0.2771 | High diversity, zero personalization |
| v07 Gemma-3-12b (78/80 named)      | 0.6752 | 4.40 | 0.4837 | Best composite so far |
| v08 Phase A + Gemma-3-12b (running) | TBD | TBD | TBD | Expected: nDCG ~0.37, composite > 0.4837 |

Key finding: LLM judge dominates composite. But v04 shows LexDiv 0.81 is achievable.
Target: push Gemma LexDiv from 0.67 toward 0.80 while keeping judge ~4.4.

### Track 3a: Lexical Diversity Improvement (TODO)

Gemma-3-12b scores LexDiv 0.6752 with judge 4.4. DeepSeek scored 0.8099 with judge 1.1.
Goal: close the LexDiv gap without sacrificing judge score.

**Root cause:** Gemma reuses closing phrases across sessions ("makes this the top pick",
"was an ideal match", "was the ideal next step"). These shared bigrams reduce Distinct-2.

**Ideas:**

3a-1. **Forbidden closing phrases** -- extend the hard-rules banned opener list to also
ban recurring closing patterns. Add: "makes this the top pick", "was the ideal",
"was an ideal", "makes it an ideal", "makes this selection", "makes this an ideal",
"ideal next step", "ideal match".

3a-2. **Temperature sweep** -- test temperature 0.75 (current), 0.9, 1.0 on 20-session
sample. Measure LexDiv and judge score. Higher temp = more varied vocab. Risk: incoherence.

3a-3. **Genre/era vocabulary injection** -- add a `VOCABULARY HINT` field to the user
message listing 3-5 genre-specific descriptors from the track's tags (e.g. "grunge:
raw, abrasive, distorted, apathetic" or "bossa nova: lilting, airy, intimate, understated").
Instruct the model to weave at least one into the response.

3a-4. **Explicit diversity instruction** -- add to SYSTEM: "Each response must use
vocabulary specific to this track's genre and era. Do not repeat phrases you would
use in other recommendations."

### Track 3b: Response Quality (DONE for now)

Gemma-3-12b with current prompt achieves judge 4.4/5. The 13 opening style hints +
full conversation history + played-tracks block + example in user message are working.
No immediate changes needed unless judge score drops in v08.

### Steps

1. [done] Measure blind LexDiv and judge: v07 = 0.6752 / 4.40
2. [ ] Implement 3a-1 (forbidden closing phrases) -- cheapest, no latency cost
3. [ ] Implement 3a-3 (vocabulary injection) -- moderate, requires tag lookup
4. [ ] A/B test temperature: 0.75 vs 0.9 on 20-session sample, measure both metrics
5. [ ] Target: LexDiv > 0.75 with judge >= 4.0

---

## Track 4: Blind Generalization (ACTIVE)

### What we know (2026-05-29 ablation)

Per-turn nDCG@20 on dev (1000 sessions):

| Turn | Phase A pool=1000, 27-feat | Phase B pool=2000, 29-feat | Phase D pool=2000, 39-feat | TT v8 pool=2000, 29-feat |
|------|---------------------------|---------------------------|---------------------------|--------------------------|
| T1   | 0.1865                    | 0.1910                    | **0.1948**                | 0.1888                   |
| T2   | 0.1942                    | 0.1942                    | **0.2015**                | 0.1973                   |
| T3   | 0.1674                    | 0.1722                    | 0.1714                    | **0.1729**               |
| T4   | 0.1543                    | 0.1588                    | 0.1582                    | **0.1602**               |
| T5-T8| ~0.15                     | ~0.15                     | ~0.15                     | ~0.15                    |
| Full | 0.1646                    | 0.1653                    | **0.1684**                | 0.1635                   |

Key findings:
- Phase D pool=2000 wins at T1 (+0.0083 over Phase A), T2 (+0.0073), and overall.
  **The turn-1 weakness hypothesis is wrong** -- Phase D is better at T1 on dev.
- TT v8 wins at T3-T4 specifically (+0.0015-0.0020 over Phase D). 512-tok window
  helps when conversation context is richest.
- Phase D gains are concentrated at early turns (T1-T2). Later turns all converge.
- **The blind nDCG gap (v04 0.3709 vs v07 0.3164) is distributional shift**, not
  a turn-position issue. Blind sessions have niche goals (album art, sonic quality,
  specific East Coast sub-genre) that the extra TT pool can't serve well.

### Root cause of blind gap

pool=2000 adds TT candidates ranked 1001-2000 that are "dev-distribution" tracks --
they benefit from n_sources agreement on dev-like sessions but are noise on blind
sessions with unusual or niche queries. n_sources (dominant feature, gain 497k) is
calibrated to dev pool composition and degrades when pool composition shifts.

### Solutions for next iteration

**4a. Turn-position-aware pool sizing (LOW RISK)**
Use `--tt_pool 1000` for T1-T2, `--tt_pool 2000` for T3+. Early turns are where
blind overfitting is most likely (no history to anchor retrieval). Requires adding
`--tt_pool_by_turn 1000,1000,2000,2000,2000,2000,2000,2000` flag to inference script.
Expected: dev nDCG drop < 0.002, blind nDCG improvement toward 0.37.

**4b. Goal-category stratified LTR training (MEDIUM)**
2000 training sessions are currently random. Oversample rare goal categories
(album art, sonic quality, era-specific) so the LTR sees more diversity. Check if
goal_category (gain 228, lowest of new features) variance is driving overfitting.

**4c. n_sources normalization (MEDIUM)**
n_sources distribution shifts when pool config changes. Add `n_sources_norm =
n_sources / log2(pool_size)` as a feature alongside raw n_sources. This makes the
dominant signal scale-invariant across pool configs.

**4d. Turn-1 separate model (MEDIUM)**
Train a turn-1-only LTR on cold-start sessions. At T1, history features
(dist_to_last, history_len, cf_dist_*) are all zero -- they add noise not signal.
A T1-specific model with only retrieval-signal features may generalize better to
blind cold-start turns.

**4e. CV fold stability filter (LOW COST)**
Extract per-fold feature importance from the 5-fold CV. Drop features with CV
importance std/mean > 0.5 (high variance = overfitting). Candidate: goal_category
(gain 228), history_len (754), years_since_release (924).

### Steps

1. [ ] Evaluate Phase D LTR on Phase A pool (tt_pool=1000) for safe blind config
2. [ ] Run feature ablation: drop one feature at a time, measure dev nDCG
3. [ ] Check CV fold importance stability for all 39 features
4. [ ] Produce recommended blind submission config

---

## Track 5: TT v8 Integration (TRAINING ~COMPLETE, 2026-05-29)

Phase C (TT v8 LoRA, multilingual-e5-base, 512-token context) is at step ~2165/2326.
Eval loss: 0.677→0.645→0.632→0.626 (all checkpoints improving). ETA: ~30 min.
Model: `models/twotower_v8/final/` (after merge_and_unload, vanilla SentenceTransformer).
Index will be: `cache/twotower_v8/` (shape 47071×768, `passage: ` prefix on docs).

When TT v8 finishes, all TT-derived features need re-dumping:
- `tt_cos`, `tt_rank_sig` distributions change (768-dim e5-base vs 384-dim MiniLM)
- `dist_to_last`, `dist_to_recent_mean` are in TT space
- Pool composition changes (different tracks enter via TT expansion)
- `cf_dist_to_last`, `cf_dist_to_mean` are independent (CF space, unchanged)

### Steps (auto-executing via cron job 3ac6ae3f)

1. [auto] Build v8 index: `build_twotower_index.py --doc_prefix "passage: " --batch_size 32`
2. [auto] Quick eval with old 29-feat LTR + `--tt_query_prefix "query: " --tt_text_turns 3 --tt_hist_turns 4`
          (miscalibrated TT features, but gives recall signal)
3. [ ] Dump 39-feature set with v8 TT embeddings (same 2000 TRAIN sessions, seed 42)
4. [ ] Train new LTR on v8 39-feature dump (all 4 variants: baseline, soft, poly, both)
5. [ ] Full dev eval + recall audit; compare vs v6-based results

```bash
# Step 3: v8 39-feature dump
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
python scripts/inference/run_inference_fusion_recall_expansion.py \
  --tid phase_c_ltr_features_v8 \
  --split train --sessions 2000 --shuffle_seed 42 \
  --tt_model models/twotower_v8/final --tt_index cache/twotower_v8 \
  --tt_query_prefix "query: " --tt_text_turns 3 --tt_hist_turns 4 \
  --tt_pool 2000 --ql_pool 200 --bm25_sharp_pool 200 \
  --artist_expansion --last_nn_k 100 --last_nn_src 2 \
  --bm25_missing_floor 0.05 \
  --qwen_pool 500 --cf_pool 200 --session_mean_k 100 \
  --cooccur_table cache/cooccur/next_song_leakfree.npz --cooccur_ks 300,150,50 \
  --write_features exp/analysis/ltr_phase_c_v8_train_features.npz
```

---

## Track 6: Catalog Diversity (MMR tuning)

Current catalog diversity: 0.5144 (51% of 47K catalog recommended).
This is already above the organizer baseline (0.3795) but could be higher.

MMR (Maximal Marginal Relevance) re-ranking is implemented in `src/infer/mmr.py`
with lambda=0.5. It is NOT currently applied in the production pipeline.

### Steps

1. [ ] Apply MMR post-processing to LTR output with lambda sweep (0.3, 0.5, 0.7)
2. [ ] Measure nDCG@20 vs catalog diversity tradeoff
3. [ ] Find lambda that maximizes catalog diversity with <0.001 nDCG@20 loss

---

## Priority Order

Given the June 30 deadline:

| Priority | Track | Expected Impact | Effort |
|---|---|---|---|
| 1 | Track 1: LTR features | +0.001 to +0.005 nDCG@20 | Low (code done, just train/eval) |
| 2 | Track 3: Response generation | Major lift on Distinct-2 + LLM-judge | Medium |
| 3 | Track 4: Blind generalization | Protect blind score (0.37 baseline) | Low |
| 4 | Track 5: TT v8 integration | +0.001 to +0.010 if recall improves | Blocked |
| 5 | Track 6: MMR diversity | Free points on catalog diversity | Low |
| 6 | Track 2: Recall improvement | +0.001 to +0.003 (diminishing returns) | Medium |

## Validation

- Track 1: Dev nDCG@20 > 0.1653
- Track 2: Pool recall > 0.8303
- Track 3: Distinct-2 > 0.2558 (beat organizer baseline)
- Track 4: Phase A pool + new LTR does not regress vs current Phase A blind (0.37)
- Track 5: Dev nDCG@20 > 0.1653 with v8 embeddings
- Track 6: Catalog diversity > 0.55 with <0.001 nDCG@20 loss
