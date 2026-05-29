# Plan: Feature Engineering v2 + Next Directions (Phase D)

## Goal

Beat dev nDCG@20 0.1653 and blind nDCG@20 0.37 through feature engineering,
training improvements, and response generation. Deadline: June 30, 2026.

## Current State (updated 2026-05-29)

- Dev nDCG@20: **0.1684** (39-feat Phase D LTR, TT v6, tt_pool=2000)
- Pool recall: 87.21% (up from 83.03%)
- Blind nDCG@20: 0.37 (Phase A pool -- Phase D not yet blind-tested)
- Catalog diversity: 0.5159 | Lexical diversity: 0.2086
- TT v8 index built. TT v8 with old LTR: 0.1635 (below gate -- needs v8 LTR retrain)
- Next: re-dump 39 features with TT v8, retrain LTR, full eval

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

## Track 1b: TT v8 + 39-feat LTR (IN PROGRESS)

TT v8 (multilingual-e5-base 512-tok, LoRA r=16) index built at `cache/twotower_v8`.
TT v8 with old Phase B LTR: 0.1635 (below Phase D 0.1684 -- LTR not calibrated to v8).
Need to retrain LTR on v8 embeddings.

### Steps

1. [done] Build TT v8 index
2. [done] Quick eval with old LTR: 0.1635 (expected, distribution shift)
3. [running] Re-dump 39 features with TT v8 embeddings
4. [ ] Retrain LTR baseline on v8 features
5. [ ] Golden-200 eval + full dev eval
6. [ ] Gate: dev nDCG@20 > 0.1684

```bash
# Re-dump command (running)
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
python scripts/inference/run_inference_fusion_recall_expansion.py \
  --tid phase_d_v8_ltr_features --split train --sessions 2000 --shuffle_seed 42 \
  --tt_model models/twotower_v8/final --tt_index cache/twotower_v8 \
  --tt_query_prefix "query: " --tt_text_turns 3 --tt_hist_turns 4 \
  --tt_pool 2000 --artist_expansion --last_nn_k 100 --last_nn_src 2 \
  --bm25_missing_floor 0.05 \
  --qwen_pool 500 --cf_pool 200 --session_mean_k 100 \
  --cooccur_table cache/cooccur/next_song_leakfree.npz --cooccur_ks 300,150,50 \
  --write_features exp/analysis/ltr_phase_d_v8_train_features.npz \
  2>&1 | tee /tmp/phase_d_v8_dump.log
```

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

## Track 3: Response Generation (MAJOR UNTAPPED DIMENSION)

Current system uses template responses on dev:
`I recommend "{name}" by {artist} based on your request.`

This scores:
- Lexical diversity: 0.2073 (BELOW organizer baseline 0.2558)
- LLM-as-Judge: not measured, but certainly poor (no personalization, no explanation)

The competition explicitly scores response quality via Gemini LLM judge and Distinct-2.
This is the biggest untapped dimension -- potentially worth as much as nDCG improvements.

### Ideas

**3a. Response templates with variation**
Replace the single template with 5-10 diverse templates that rotate based on:
- Turn position (first turn vs follow-up)
- Query type (specific artist vs mood vs "more like that")
- Track metadata (genre, decade, artist popularity)
This alone would substantially lift Distinct-2.

**3b. Local LLM response rewrite (dev-side)**
Apply the same Qwen/Gemma response generation currently used only for blind submissions
to the dev evaluation pipeline. This would let us measure and optimize response quality
during development instead of only at blind submission time.

**3c. Retrieval-grounded response generation**
Instead of just naming the top-1 track, generate responses that reference:
- Why the top track matches the user's stated preferences
- What tags/genres connect the recommendation to the request
- How the track relates to previously played tracks
This directly targets the LLM-as-Judge personalisation dimension.

**3d. Chain-of-Thought response generation**
README notes: "Systems that generate internally reasoned responses (even if the CoT is
hidden) tend to score higher on explanation quality." Generate internal reasoning
(which tracks in top-20 match which user preferences) then produce a grounded response.

### Steps

1. [ ] Measure current Distinct-2 and estimate LLM-as-Judge on dev with template
2. [ ] Implement diverse template rotation (3a) -- zero-model-cost baseline
3. [ ] Run LLM response generation on dev set and measure Distinct-2 lift
4. [ ] Iterate on response quality (grounding, personalization)

---

## Track 4: Blind Generalization (DEFENSIVE)

Phase B hurt blind (0.37 -> 0.30). Understanding why is critical before adding more features.

### Hypotheses

1. **tt_pool=2000 overfits to dev pool distribution.** Dev sessions are synthetic with
   consistent patterns. Blind sessions may have different query distributions.
2. **popularity feature has different distribution in blind.** Dev tracks skew popular
   (synthetic conversations favor well-known tracks). Blind may include more long-tail.
3. **track_year is noise.** Marginal gain (1813) in training, likely fitting to dev-specific
   year patterns that don't transfer.

### Mitigation strategies

**4a. Phase A pool with Phase D features**
Use the safer Phase A pool (tt_pool=1000) with the new 39 features. If this beats
0.1646 on dev, it may also beat 0.37 on blind.

**4b. Feature ablation on blind**
Systematically drop suspect features (popularity, track_year, popularity_pctile,
goal_category) and evaluate each on dev. Features that cause dev nDCG to drop
significantly are load-bearing; features with marginal dev impact but high blind
risk should be dropped.

**4c. Cross-validation stability check**
Compare feature importance across 5 CV folds. Features with high variance in
importance across folds are overfitting candidates.

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
