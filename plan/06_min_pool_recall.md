# Plan: Maximize gold-in-pool recall with minimum candidates

## Goal

Reach the highest possible pool recall at the smallest possible pool size.
Today: 80.6% recall at total pool ~1500 (BM25@500 + artist + TT-v6@1000).
Target: ≥80% at total pool ≤ 800, and ≥85% at total pool ≤ 1500.

Why this matters: the rescore step is BM25@500-only (gold available rate = 59.0%).
Every percentage point we move into a pool we actually rescore is a real ranking
opportunity. Bigger pool also costs scoring time linearly, so we want recall per
candidate, not just absolute recall.

## Current State

See `plan/CURRENT_BEST_ITERATION.md`. Summary:

```
BM25@500                                  0.590  (size 500)
BM25@500 + artist                         0.651  (size ~700)
BM25@500 + artist + TT-v6@500             0.755  (size ~1200)
BM25@500 + artist + TT-v6@1000            0.806  (size ~1500)
BM25@500 + ALL signals @1000 (audit)      0.846  (size ~2500)
```

BM25-miss rescue breakdown (3277 BM25 misses):
- TT@500 rescues 33.3%
- CF@500 (warm) rescues 11.4%
- Qwen-meta@500 rescues 9.7%
- Qwen-lyrics@500 rescues 6.7%
- CLAP @500 rescues 1.0%

## Assumptions

- The rescore pipeline can consume any candidate pool we hand it.
- TT-v6 is the strongest single dense signal; v7 retrain is out of scope here.
- Total pool size budget includes deduplication across sources.
- 8000-turn eval is the unit of truth.

## Hypotheses (ranked by expected recall-per-candidate)

1. **Last-track-NN recall** — for each of the last N played tracks, fetch top-k
   nearest tracks in the TT (or Qwen-attr) embedding space and union into pool.
   Targets history_driven (1256 turns) and more_like_this (2284 turns) buckets,
   which together are 44% of all turns. Expected: cheap, ~20-50 candidates per
   turn, hits the cases where the user said little but listening trajectory is
   informative.

2. **MMR-diversified TT@K** — TT@500 currently overlaps heavily with BM25@500 in
   easy cases and wastes the budget. Replace with MMR over TT scores so that the
   added 500 dense candidates are maximally orthogonal to the BM25 pool.

3. **Multi-query TT union** — encode two queries per turn (entity-focused vs
   mood-focused) and union top-k from each. Targets mood and lyrics buckets where
   the single compact query collapses two intents.

4. **Per-bucket adaptive K** — `specific` and `history_driven` buckets reach 70%+
   at BM25@500; `mood` and `lyrics` are 53-54%. Spend the pool budget where it
   pays off (e.g., bigger TT@K for mood, smaller for specific).

5. **Smarter dedup / overflow** — current BM25 retrieves `bm25_pool + len(seen)*3`
   and trims to 500. If gold sits at BM25 rank 501-700, raising bm25_k to 700 with
   tight dedup is a cheap test.

6. **Artist similarity expansion** — when artist X is mentioned, also add the top
   K nearest artists by TT artist-centroid similarity. Risky for noise; gate on
   small K (≤2).

## Files To Read (already inspected)

- `scripts/inference/measure_recall_artist_expansion.py` — artist mention scan
- `scripts/inference/measure_recall_combined.py` — BM25 + artist + TT@K sweep
- `scripts/inference/audit_recall.py` — per-signal recall audit
- `scripts/inference/run_inference_fusion_recall_expansion.py` — production-side
  expansion entrypoint (where any new recall source plugs in)
- `cache/twotower_v6/*` — TT-v6 index and ids
- `cache/qwen3_attr/*` — attr embeddings (for history-NN in attr space)

## Files To Modify / Create

- New: `scripts/inference/measure_recall_min_pool.py`
  Sweeps BM25@N (N in {300, 500, 750}) x artist on/off x TT@K (K in {0, 250, 500,
  1000}) x last-track-NN (M in {0, 50, 100, 200}) and reports recall vs *deduped*
  total pool size. Produces a Pareto frontier.
- New: `scripts/inference/measure_recall_history_nn.py`
  Isolated test: BM25@500 + artist + last-track-NN at varying M and embedding
  space (TT vs Qwen-attr). Reports rescue rate vs pool size and bucket lift.
- Update: `plan/05_recall_improvement.md` — append pointer to this plan.
- Update: `plan/PLAN.md` — record Phase 7 entry pointing here.

## Steps

1. Append a "Phase 7: Min-pool recall" entry to `plan/PLAN.md` and link.
2. Build `measure_recall_min_pool.py`. Measure full grid on 8000 turns. Save table
   to `exp/analysis/recall_min_pool_grid.txt`.
3. Build `measure_recall_history_nn.py`. Measure last-track-NN rescue rate at
   M ∈ {25, 50, 100, 200}, in TT and Qwen-attr space. Save to
   `exp/analysis/recall_history_nn.txt`.
4. Pick the smallest pool config that hits ≥80% recall. Report it.
5. If the new minimum-pool config beats the current
   `BM25@500 + artist + TT@1000` size at equal recall, integrate it into
   `run_inference_fusion_recall_expansion.py` behind flags and rerun dev nDCG.

## Validation

- Per-config table: `total_pool_size, recall@pool, bucket_recall` for the six
  query buckets.
- Pareto plot (text table is fine): pool_size vs recall, highlighting current
  baseline and new candidates.
- Success: ≥80% recall at total pool ≤ 800; or ≥85% at total pool ≤ 1500.

## Risks

- TT batch encoding over 8000 turns is ~minutes; full grid is N_configs * encoding
  but encoding can be cached once and reused (compute sims once, slice K).
- Dedup math: a 500+500+200 pool can dedup to 700-1100 depending on overlap. The
  grid must report deduped size, not nominal.
- Artist expansion size is unbounded (popular artists have 100+ tracks). Cap per
  artist may be needed to avoid blowing the budget.

## Notes

- Provenance logging (per-turn JSONL with `found_by`, `bm25_rank`, `tt_rank`,
  `artist_match_source`) is a sibling improvement and lives in a separate plan
  (see future `plan/07_recall_provenance.md` if/when ranking work resumes).
  This plan stays scoped to pool-recall efficiency.
- The 19.4% structurally unreachable set is acknowledged but out of scope here;
  closing it likely requires v7 training or new signals, not budget shuffling.
