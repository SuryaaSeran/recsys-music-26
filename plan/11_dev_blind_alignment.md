# Plan: Dev-Blind Alignment (Phase 11)

## Goal

Make dev evaluation predictive of blind A performance. Current gap: dev 0.1653
vs blind 0.37 (2.2x). Phase B showed the danger: +0.0007 dev, -0.07 blind.

## Root Cause

Dev evaluates all 8 turns per session (8000 predictions, macro-averaged by
turn position). Blind A evaluates only the last turn per session (80 predictions).
Early turns with minimal context score low and dominate the dev average. Features
that help early turns (popularity as a prior) can hurt late turns where
personalized signals should dominate.

## Current State

Branch `feature/dev-blind-alignment`, forked from `feature/engineering-v2` which has:
- 39 LTR features (10 new Phase D + 3 Phase D2 intent features)
- Recall expansion: ql_pool, bm25_sharp_pool
- Progress-aware LTR labels
- Response prompt improvements
- TT v8 training (in progress)

None of this has been evaluated on blind yet.

## Hypothesis

By adding a "last-turn-only" dev metric that matches blind evaluation semantics,
and training LTR with late-turn weighting, we can:
1. Build a dev metric that predicts blind within 10% relative error
2. Avoid repeating the Phase B regression
3. Identify blind-safe vs blind-unsafe features

## Changes Made

### A1. Evaluator improvements (DONE)

`scripts/inference/evaluate_local.py`:
- `--last_turn_only`: score only the last music turn per session
- `--per_turn_breakdown`: print nDCG@20 per turn position table with
  early/late/last buckets

### B1. Turn-weighted LTR training (DONE)

`scripts/train/train_ltr_lightgbm.py`:
- `--turn_weight_mode uniform|last_only|late_only|late_heavy|progressive`
- `last_only`: train only on last turn per session (matches blind)
- `late_heavy`: turns 5-8 get weight 2.0, turns 1-4 get weight 0.5
- `progressive`: weight = turn_number / 8

### C1. Feature ablation by turn position (DONE)

New `scripts/analysis/feature_ablation_by_turn.py`:
- Trains LightGBM separately on all/early/late/last turn buckets
- Reports feature gain per bucket
- Flags features as safe/unsafe/mixed for blind submissions

### D3. Diverse response templates (DONE)

`scripts/inference/run_inference_fusion_recall_expansion.py`:
- 12 diverse templates replacing single template
- Keyed by hash(session_id, turn_number) for determinism
- Uses track metadata (tags, year, album) for variation
- Expected Distinct-2 lift: 0.20 -> 0.30+

## Validation Plan

### Step 1: Verify metric correlation

Run existing pred files through new evaluator:
```bash
python scripts/inference/evaluate_local.py \
  --pred exp/inference/devset/<phase_a_pred>.json --per_turn_breakdown

python scripts/inference/evaluate_local.py \
  --pred exp/inference/devset/<phase_a_pred>.json --last_turn_only
```

Confirm: Phase A last-turn-only ~ 0.35, Phase B last-turn-only ~ 0.28.

### Step 2: Turn-weighted LTR ablation

Dump features, then train 4 weight modes:
```bash
python scripts/train/train_ltr_lightgbm.py \
  --features <dump>.npz --out models/ltr/ltr_<mode>.txt \
  --turn_weight_mode <mode> --n_folds 5
```

Compare CV ndcg@20 across modes. Evaluate on dev with --last_turn_only.

### Step 3: Feature ablation

```bash
python scripts/analysis/feature_ablation_by_turn.py \
  --features <dump>.npz --out exp/analysis/feature_ablation.json
```

Drop blind-unsafe features from submission config.

### Step 4: Blind submission

Use best (last-turn-only dev score) config for blind A submission.

## Next Steps (not in this phase)

- D1: Cross-encoder as LTR feature
- D2: Full-context Qwen query
- D5: Train on more LTR sessions (5000-10000)
- D6: Two-stage LTR (coarse + fine)
- D7: TT v8 integration
