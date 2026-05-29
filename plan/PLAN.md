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

- **Phase 10: TT v8 — larger context window + LoRA** — [07_ranking_calibration.md](07_ranking_calibration.md)
  Phase B concluded (0.1653, reg booster). TT v8 replaces all-MiniLM-L6-v2 (256-tok)
  with intfloat/multilingual-e5-base (512-tok, LoRA r=16) and richer tokenizer-aware
  anchors. Training in progress (eval loss 0.677→0.626 across 4 checkpoints, 2026-05-29).
  After training: build index, quick eval with old LTR, then retrain LTR on v8 features.
  Gate: pool recall > 0.830 AND dev nDCG@20 > 0.1653.

## Score ladder (full 1000-session dev nDCG@20)

```
0.1653  Phase B pool (tt_pool=2000) + 29-feat reg LTR nl31 lr0.08 (l2+hessian+path_smooth) <- current best (2026-05-28)
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

| Version | Dev nDCG@20 | Retrieval | Response | Status |
|---|---:|---|---|---|
| **v06** | **0.1653** | Phase B pool (tt_pool=2000) + 29-feat reg LTR | Gemma-3-12b native API (76/80 track named) | **recommended** |
| v04 | 0.1646 | Phase A pool + LTR nl31 lr0.08 | DeepSeek V4 Flash (73/80 track named) | superseded |
| v05 | 0.1646 | same | Gemma-4-e4b local (9/80 track named) | submitted, inferior responses |

Active submission zip: `exp/inference/blind_a/submission/submission.zip` (currently v05;
copy v04 zip to switch). Retrieval for both: `run_inference_fusion_recall_expansion.py`
with Phase A expansion flags + `--ltr_model models/ltr/ltr_phase_a_nl31_lr0p08.txt`.
