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

## Active phases

- **Phase D: TT v8b + progress-aware LTR (6K sessions)** — [08_feature_engineering_v2.md](08_feature_engineering_v2.md)
  TT v8b trained (drop_rejected + 3 hard negs). 42 features (39 Phase D + 3 user-intent proxies).
  H1+H3 inference flags implemented. 6K session feature dump running (2026-05-30).
  Gate: dev nDCG@20 > 0.1684.

- **Phase F: Dev/Blind Alignment + Goal-Type Routing** — [09_generalization_routing.md](09_generalization_routing.md)
  Ablation complete (2026-06-01): 2x dev/blind gap explained — Phase D CF/cooccurrence displaces
  correct BM25/TT results on niche blind sessions (overlap 9.6/20 vs 15.3/20 on dev). Weakest
  categories C/I/K (0.13-0.14). Plan: goal-type routing, implicit progress labels, query rewriting,
  adaptive pool sizing. Target: dev > 0.1684, blind >= 0.37.

## Score ladder (full 1000-session dev nDCG@20)

```
0.1684  Phase D pool (tt_pool=2000, TT v6) + 39-feat LTR nl31 lr0.08       <- current best (2026-05-29)
0.1682  Phase D pool (TT v8b) + 42-feat progress-aware LTR (2K sessions)    (2026-05-30, below gate)
0.1678  Phase D pool + 39+14-feat poly LTR nl31 lr0.08                      (2026-05-29, baseline wins)
0.1672  Phase D pool (TT v8b) + 42-feat LTR + H1+H3                         (2026-05-30, H1+H3 hurt all-turns)
0.1653  Phase B pool (tt_pool=2000) + 29-feat reg LTR nl31 lr0.08 (l2+hessian+path_smooth) (2026-05-28)
0.1646  Phase A pool (tt_pool=1000) + 27-feat LTR nl31 lr0.08, 2000 train sessions           (2026-05-27)
0.1609  v6 fusion + expansion + LTR LambdaMART nl31 lr0.08 (train-only)           (2026-05-15 LTR v3)
0.1601  v6 fusion + expansion + LTR LambdaMART nl63 lr0.05 (train-only)           (2026-05-11 LTR v2)
0.1533  v6 fusion + recall expansion (artist + TT@1000 + last-NN@100), v13 wts
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
| 8: Source-aware ranking     | done       | [07_ranking_calibration.md](07_ranking_calibration.md) — 0.1646 (Phase A LTR) |
| **9: Phase B LTR + LLM prune** | **active** | [07_ranking_calibration.md](07_ranking_calibration.md) — pop+year features + Opus top-25 rerank |

## Blind A submissions

Versioned folder: `exp/inference/blind_a/submissions/` (README.md inside).

| Version | Blind nDCG@20 | LLM Judge | Composite | Dev nDCG@20 | Status |
|---|---:|---:|---:|---:|---|
| **v07** | **0.3164** | **4.40** | **0.4837** | 0.1684 | **current best composite** |
| v06 | 0.3000 | — | — | 0.1653 | Phase B retrieval hurt blind |
| v04 | 0.3709 | 1.10 | 0.2771 | 0.1646 | best nDCG, poor judge |

Key finding: composite dominated by LLM judge. v07 wins composite despite weaker nDCG.
Next target: Phase A pool (nDCG ~0.37) + Gemma-3-12b responses (judge ~4.4) → composite > 0.5.
Active submission zip: `exp/inference/blind_a/submission/submission.zip` (v07).
