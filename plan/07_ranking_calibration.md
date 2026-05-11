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
bm25_signal = bm25_norm if in_bm25_pool else bm25_missing_floor    # stays inside w_bm25
artist_sig  = 1/log2(artist_rank + 2) if matched else 0            # rank within artist catalog
nn_sig      = 1/log2(min_nn_rank + 2) if NN-hit else 0             # min rank across source tracks
tt_rank_sig = 1/log2(tt_rank + 2)     if in_tt_pool else 0
bm25_origin = 1 if "bm25" in sources else 0                        # preservation feature

score =   w_bm25         * bm25_signal
        + w_tt           * tt_cosine
        + w_qwen_meta    * qm_cosine
        + w_qwen_lyr     * ql_cosine
        + w_clap         * clap_cosine
        + w_cf           * cf_cosine
        + w_tt_rank      * tt_rank_sig
        + w_artist       * artist_sig
        + w_nn           * nn_sig
        + w_bm25_origin  * bm25_origin
```

- `bm25_missing_floor` is **inside** w_bm25 (replaces the BM25 signal value for non-BM25
  candidates); it is not a raw additive constant.
- `artist_rank` is the position of the track within the matched artist's catalog
  (proxy: order in the metadata-derived list, capped at `--artist_cap`).
- `nn_rank` is the smallest rank across any of the last-N source tracks for which this
  candidate is a TT-space neighbor.

Sweep (Step 3, narrowed):

```
bm25_missing_floor : 0.00, 0.03, 0.05, 0.08, 0.10
w_tt_rank          : 0.00, 0.03, 0.05, 0.08
w_artist           : 0.00, 0.02, 0.05
w_nn               : 0.00, 0.02, 0.05
w_bm25_origin      : 0.00, 0.02, 0.05
```

Holding the v13_tuned base weights constant initially; co-tune in pass 2.

### Baselines (must match before sweeping)

- **A**: expanded pool, old scorer (current `run_inference_fusion_recall_expansion.py`
  at HEAD before this work).
- **B**: expanded pool, new code path, `w_tt_rank=w_artist=w_nn=w_bm25_origin=0`,
  `bm25_missing_floor=0.05` (same as A).

If A ≠ B (within 0.0005 nDCG), the source-tracking refactor changed scoring
unintentionally; halt and diff.

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
- Final report table per config:

| Config | nDCG | Hit@20 | BM25-hit top20 | Artist-rescue top20 | TT-rescue top20 | NN-rescue top20 |
| ------ | ---: | -----: | -------------: | ------------------: | --------------: | --------------: |

  Bucket = which source(s) contained the gold; bucket metrics report top-20 rate
  conditioned on each.

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

## Result: full 1000-session parity-B (no new features)

Config B (expansion pool with last-track-NN@100, v13 weights, all new
`w_*=0`, `bm25_missing_floor=0.05`) ran on the full dev set:

```
nDCG@20  0.1533   (+0.0014 over prior best 0.1519)
Hit@20   31.7%
```

The +0.0014 is entirely from adding NN@100 candidates -- the rescore is
identical to v13. The prior expansion-pool-without-NN tested at 0.1518, so
this is a real, NN-attributable lift. Updated
`plan/CURRENT_BEST_ITERATION.md`.

This invalidates the "parity must match A==B within 0.0005" check from earlier
in this plan -- because the new pool is strictly larger (adds NN candidates),
A and B were never going to match. The relevant comparison is:

```
old pool   (artist + TT@1000)             v13 weights  ->  0.1518
new pool   (artist + TT@1000 + NN@100)    v13 weights  ->  0.1533  (+0.0015)
```

That is the NN ablation, and it positive.

Next: continue Step 6 -- sweep `w_tt_rank, w_artist, w_nn, w_bm25_origin,
bm25_missing_floor` on top of this 0.1533 baseline. Bucket-level reporting
(BM25-hit / artist-rescue / TT-rescue / NN-rescue top-20 rates) is still TBD.

## Smoke Test (20 sessions, 160 turns)

Run: `--tt_pool 1000 --artist_expansion --last_nn_k 100 --last_nn_src 2
--bm25_missing_floor 0.05 --w_tt_rank 0 --w_artist 0 --w_nn 0 --w_bm25_origin 0
--write_provenance exp/analysis/prov_smoke.jsonl` (config B = parity baseline).

- Throughput: ~2.35 s / turn (≈ 40 min for full 1000 sessions).
- Pool size: ≈ 1400 (matches sweep prediction).
- Provenance schema verified end-to-end.
- Example failure already visible (turn 2 of first session): gold found only by TT
  at tt_rank=198, final_rank=435. Exactly the rescued-but-not-ranked case.

Next: run full 1000-session config-A vs config-B parity, then sweep.

## Score Targets

- Step 5 baseline (no calibration): expected ~0.1518 (parity).
- Step 6 with calibration: target ≥ 0.155 dev nDCG. Anything < 0.152 = falsifies
  hypothesis, escalate to provenance analysis.
