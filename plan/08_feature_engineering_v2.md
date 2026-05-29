# Plan: Feature Engineering v2 (Phase D)

## Goal

Add 10 new LTR features targeting structural signals, metadata utilization, and CF-space
distances. Beat dev nDCG@20 0.1653 without blind regression risk.

## Current State

29-feature LambdaMART (Phase B reg) scores 0.1653 on dev, but Phase B features (popularity,
track_year, tt_pool=2000) hurt blind (0.37 -> 0.30). This phase prioritizes structural and
lexical features over metadata-distribution-dependent ones.

## Hypothesis

The current feature set has clear gaps: no turn-position signal, no tag-match signal, no
CF-space distance features, no query-specificity proxy. Adding these gives the booster new
decision axes that are structurally robust (not distribution-dependent). Expected improvement:
+0.001 to +0.005 nDCG@20.

## New Features (10)

| Feature | Type | Range | Why |
|---|---|---|---|
| `n_sources` | count | [1, 7] | Multi-source agreement = confidence |
| `turn_number` | position | [1, 8] | Early vs late turn ranking behavior |
| `history_len` | count | [0, 7] | Graduated cold-start (replaces binary cold_user) |
| `popularity_pctile` | normalized | [0, 1] | Replaces raw popularity (more stable across splits) |
| `years_since_release` | derived | [0, 126] | Replaces raw year (2026 - year, 0 if missing) |
| `tag_overlap_count` | lexical | [0, 12] | Explicit genre/mood match between query and track |
| `query_len_tokens` | proxy | [1, 100+] | Query specificity (short = vague, long = specific) |
| `cf_dist_to_last` | CF similarity | [0, 1] | Behavioral similarity to last track (CF space) |
| `cf_dist_to_recent_mean` | CF similarity | [0, 1] | Behavioral trajectory in CF space |
| `goal_category` | categorical | int | Session goal type (discovery, specific_track, etc.) |

## Steps

1. [done] Add FEATURE_COLS entries for 10 new features
2. [done] Precompute popularity_pctile lookup at startup
3. [done] Add goal_category integer encoding
4. [done] Compute CF-space distance arrays per turn
5. [done] Compute tag overlap and query length per turn
6. [done] Extend feat[i] tuple with all 10 new values
7. [done] Add 5 new polynomial interaction pairs to LTR trainer
8. [done] Syntax-check both scripts
9. [ ] Dump features from TRAIN sessions (2000, seed 42)
10. [ ] Train LTR with new features (39 total), regularized
11. [ ] Evaluate on golden-200, then full dev 1000
12. [ ] Compare with/without soft_labels and poly_feats

## Validation

- Golden-200 nDCG@20 > 0.1595 (Phase B reg golden)
- Full dev nDCG@20 > 0.1653 (current best)
- No single new feature in top-3 by gain (overfitting signal)

## Result

(to be filled on conclusion)
