# ReccysMusic: ACM RecSys 2026 Music CRS

**Task:** predict the next track per turn in a music conversational rec system.
**Metric:** nDCG@20 (primary), Hit@20 (secondary).
**Dataset:** TalkPlayData-Challenge — 1000 dev sessions, 8000 turns, 47071 tracks.
**Evaluator:** `python scripts/inference/evaluate_local.py --pred <file>`

## How to use this folder

Read `WORKFLOW.md` in the repo root for the plan lifecycle and update rules.
Quick path for a new session:

1. This file — score ladder + active phase.
2. `CURRENT_BEST_ITERATION.md` — the system to beat.
3. The single active plan below.

## Active phase

- **Phase 8: Source-aware ranking** — [07_ranking_calibration.md](07_ranking_calibration.md)
  Pool recall ceiling (0.808) reached; turning it into nDCG via rank-based features
  for artist/TT/NN candidates and a BM25-origin preservation term.

## Score ladder (full 1000-session dev nDCG@20)

```
0.1533  v6 fusion + recall expansion (artist + TT@1000 + last-NN@100), v13 wts  <- current best (2026-05-11)
0.1519  v6 fusion, v13_tuned weights, BM25@500 pool only
0.1518  v6 fusion + recall expansion (artist + TT@1000, no NN), v13 wts
0.1473  v6 fusion v6 (precursor to v13)
0.1418  v3 bi-encoder, BM25@500 pool, w=0.7
0.1313  BM25 + tag_list + seen exclusion
0.0960  BM25 + tag_list (no seen exclusion)
0.0861  BM25 name+artist+album only
```

## Phase history

| Phase | Outcome | Detail |
|---|---|---|
| 1: BM25 baseline           | done       | [archive/01_bm25_baseline.md](archive/01_bm25_baseline.md) — 0.1313 |
| 2: Two-tower fine-tune     | done       | [archive/02_twotower.md](archive/02_twotower.md) — v3 best (0.1418); v4 hard-neg regressed |
| 3: Cross-encoder           | inconclusive | [archive/03_crossencoder.md](archive/03_crossencoder.md) — pre-trained CE underperformed; v1 overfit |
| 4: v5 triplet loss         | failed     | [archive/04_v5_twotower.md](archive/04_v5_twotower.md) — model collapsed (0.0525) |
| 5: Fusion + recall lift    | done       | [archive/05_recall_improvement.md](archive/05_recall_improvement.md) — 0.1519 best (v13 weights) |
| 6: Min-pool recall         | done       | [archive/06_min_pool_recall.md](archive/06_min_pool_recall.md) — 0.808 @ ~1468; ceiling reached |
| 7: Semantic ID (LLM)       | abandoned  | [archive/SEMANTIC_ID_PLAN.md](archive/SEMANTIC_ID_PLAN.md) |
| **8: Source-aware ranking** | **active** | [07_ranking_calibration.md](07_ranking_calibration.md) |

## Blind A submissions

Best generated: `exp/inference/blind_a/blind_a_fusion_v13_tuned_qwen.json` (dev nDCG
0.1519). Fallbacks listed in `CURRENT_BEST_ITERATION.md`. Generation command:

```bash
python scripts/inference/run_inference_blind_fusion.py --tid blind_a_<id> [weights]
python scripts/inference/generate_responses_blind.py \
    --pred exp/inference/blind_a/blind_a_<id>.json
```
