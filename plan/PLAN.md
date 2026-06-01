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

None. All previous phases concluded (see archive/). Next phase TBD.

Current state:
- Dev gate: **0.1684** (Phase D v6, 39-feat LTR) — held since 2026-05-29.
- Blind A nDCG@20: **0.3701** (v10 H1+H3, v8b retrieval).
- Blind A composite: **0.4837** (v07, judge 4.4/5).
- Phase E (46-feat H2 history features) and Phase F (44/45-feat n_sources_norm + bm25_top1,
  adaptive pool, infer_progress, entity BM25): all below gate. Retrieval ceiling reached
  on current pool/LTR architecture.

## Score ladder (full 1000-session dev nDCG@20)

```
0.1684  Phase D pool (tt_pool=2000, TT v6) + 39-feat LTR nl31 lr0.08       <- current best (2026-05-29)
0.1682  Phase D pool (TT v8b) + 42-feat progress-aware LTR (2K sessions)    (2026-05-30, below gate)
0.1678  Phase D pool + 39+14-feat poly LTR nl31 lr0.08                      (2026-05-29, baseline wins)
0.1672  Phase D pool (TT v8b) + 42-feat LTR + H1+H3                         (2026-05-30, H1+H3 hurt all-turns)
0.1653  Phase B pool (tt_pool=2000) + 29-feat reg LTR nl31 lr0.08 (l2+hessian+path_smooth) (2026-05-28)
0.1646  Phase A pool (tt_pool=1000) + 27-feat LTR nl31 lr0.08, 2000 train sessions           (2026-05-27)
0.1615  Phase D pool (TT v8b 6K) + 42-feat LTR + H1+H3                      (2026-05-31, blind 0.3701)
0.1609  v6 fusion + expansion + LTR LambdaMART nl31 lr0.08 (train-only)           (2026-05-15 LTR v3)
0.1608  Phase D pool (TT v8b 6K) + 42-feat LTR (no H1H3)                    (2026-05-31)
0.1603  Phase D pool (TT v8b 6K) + 44-feat LTR (n_sources_norm + log1p_n_sources) (2026-06-01)
0.1602  Phase D pool (TT v8b 6K) + 45-feat LTR (+bm25_top1, too sparse)     (2026-06-01)
0.1601  v6 fusion + expansion + LTR LambdaMART nl63 lr0.05 (train-only)           (2026-05-11 LTR v2)
0.1583  Phase E pool (TT v8b 6K) + 46-feat LTR (H2 history features)        (2026-05-31)
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
| 8: Source-aware ranking    | done       | [archive/07_ranking_calibration.md](archive/07_ranking_calibration.md) — 0.1646 (Phase A LTR) |
| 9: Phase D feature engineering v2 | done | [archive/08_feature_engineering_v2.md](archive/08_feature_engineering_v2.md) — 0.1684 dev best |
| 10: Response prompt optimization | done | [archive/10_response_prompt_optimization.md](archive/10_response_prompt_optimization.md) — judge 4.4 (v07) |
| 11: Path to 0.55 composite (scoping) | done | [archive/09_path_to_055.md](archive/09_path_to_055.md) — scoping doc |
| 12: Phase E H2 history features | below gate | [archive/08_feature_engineering_v2.md](archive/08_feature_engineering_v2.md) — 0.1583 (46 feat) |
| 13: Phase F generalization routing | below gate | [archive/09_generalization_routing.md](archive/09_generalization_routing.md) — 44/45-feat 0.1603/0.1602 below gate |

## Blind A submissions

Versioned folder: `exp/inference/blind_a/submissions/` (README.md inside).

| Version | Blind nDCG@20 | LLM Judge | Composite | Dev nDCG@20 | Status |
|---|---:|---:|---:|---:|---|
| **v10 H1+H3** | **0.3701** | **3.60** | **0.4504** | 0.1615 | **best blind nDCG** (v8b retrieval) |
| v07 | 0.3164 | 4.40 | **0.4837** | 0.1684 | **best composite** (judge dominates) |
| v06 | 0.3000 | — | — | 0.1653 | Phase B retrieval hurt blind |
| v04 | 0.3709 | 1.10 | 0.2771 | 0.1646 | high nDCG, poor judge |

Key finding: composite dominated by LLM judge. v10 has best raw blind nDCG (0.3701)
from v8b H1+H3 retrieval but lower judge score; v07 wins composite at 0.4837 with
weaker retrieval (0.3164) but judge 4.4/5. Active submission zip:
`exp/inference/blind_a/submission/submission.zip` (v07).
