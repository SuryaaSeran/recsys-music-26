# Plan: Source-aware ranking on the high-recall pool

## Goal

Translate the 80.8% pool recall into nDCG@20 gain. Current best nDCG = 0.1519 on a
BM25@500-only pool (59.0% gold availability). The recall-expansion pool (artist +
TT@1000) hits 0.808 gold availability but produces nDCG = 0.1518 — rescued candidates
cannot rank because they enter with bm25_signal = 0 (or a flat floor).

## Hypothesis

Non-BM25 candidates need their own ranking features (rank-based, source-aware). Once
they do, the additional 21.8% of gold tracks in the pool should start landing in the
top 20.

## Production Recall Pool (frozen for ranking work)

```
BM25@500 + artist + TT-v6@1000 + last-track-NN@100   (pool ~1468, recall 0.808)
```

Iteration pool (faster, ~half the cost, same shape):

```
BM25@300 + artist + TT-v6@500  + last-track-NN@100   (pool ~876,  recall 0.745)
```

## Source-aware scoring

Each candidate carries the origin sources (BM25, artist, TT, NN) and per-source rank.
New ranking features on top of existing fusion:

```
score =   w_bm25       * bm25_norm                       # 0 if not in BM25 pool, replaced by floor
        + w_tt         * tt_cosine
        + w_qwen_meta  * qm_cosine
        + w_qwen_lyr   * ql_cosine
        + w_clap       * clap_cosine
        + w_cf         * cf_cosine
        + w_tt_rank    * 1 / log2(tt_rank + 2)           # NEW: rank-based TT prior
        + w_artist     * artist_hit                      # NEW: artist-expansion flag
        + w_nn         * nn_hit                          # NEW: history-NN flag
        + bm25_missing_floor                             # NEW: replaces 0.0 for non-BM25 cands
```

Sweep (Step 3):

```
bm25_missing_floor : 0.00, 0.05, 0.10, 0.15
w_tt_rank          : 0.00, 0.03, 0.05, 0.08
w_artist           : 0.00, 0.03, 0.05, 0.08
w_nn               : 0.00, 0.03, 0.05
```

Holding the v13_tuned base weights constant initially; co-tune in pass 2.

## Provenance (for the failure table)

Per turn JSONL with at minimum:

```
session_id, turn_number, gold_track_id,
found_in_pool, found_by[bm25|artist|tt|nn], pool_size,
bm25_rank, tt_rank, artist_match_source, nn_source_track,
final_rank, final_score, score_components{...}
```

Used to produce the failure breakdown:

```
BM25 found + top20 / BM25 found + miss
artist rescued + top20 / artist rescued + miss
TT rescued + top20 / TT rescued + miss
NN rescued + top20 / NN rescued + miss
unreachable
```

This decides whether v7 training should target unreachable recall, rescued-but-low-
ranked, or both.

## Files

To modify:
- `scripts/inference/run_inference_fusion_recall_expansion.py`
  - add `--last_nn_k`, `--last_nn_src` (default 2) flags
  - track per-candidate `sources` (set of bm25/artist/tt/nn) and per-source rank
  - new weights: `--w_tt_rank`, `--w_artist`, `--w_nn`
  - optional `--write_provenance <path.jsonl>`

To create:
- `scripts/inference/sweep_source_aware.py` — driver that runs the inference with
  combos and pipes through `evaluate_local.py`, writes a summary table.

## Steps

1. Plan written (this file). ✅
2. Add NN@K expansion to `run_inference_fusion_recall_expansion.py`.
3. Add per-candidate source tracking + new features + flags.
4. Add optional provenance JSONL writer (off by default to keep eval fast).
5. Single-config dev run with new pool to establish a fresh baseline nDCG.
6. Sweep source-aware weights. Best config goes to leaderboard table.
7. If nDCG > 0.1519: rerun blind with the new weights, update CURRENT_BEST_ITERATION.
8. If not: dump provenance, build failure table, decide v7 training data.

## Validation

- Step 5: dev nDCG@20 on full 1000 sessions, no NaNs, pool size mean ≈ 1450.
- Step 6: dev nDCG@20 per config table, ≥ 1 config beats 0.1519.
- Step 8 (if needed): failure-table counts sum to 8000 turns.

## Risks

- Adding NN candidates × NN sources doubles candidates in the worst case; mitigated
  by dedup and the cap (`last_nn_k * last_nn_src`).
- Rank features need consistent indexing (tt_rank = ∞ for non-TT-pool cands). Use
  a large sentinel so 1/log2 ≈ 0.
- Source flags collinear with rank scores; sweep should not co-vary them blindly.

## Score Targets

- Step 5 baseline (no calibration): expected ~0.1518 (parity).
- Step 6 with calibration: target ≥ 0.155 dev nDCG. Anything < 0.152 = falsifies
  hypothesis, escalate to provenance analysis.
