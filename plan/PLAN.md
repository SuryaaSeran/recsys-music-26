# ReccysMusic: ACM RecSys 2026 Competition

**Task:** Music Conversational Recommendation System (CRS)
**Metric:** nDCG@20 (primary), Hit@20 (secondary)
**Dataset:** TalkPlayData-Challenge — 1000 dev sessions, 8000 turns, 47071 tracks
**Evaluator:** `python scripts/evaluate_local.py --pred <file>`

---

## Current Best

| Model | nDCG@20 | Hit@20 | Sessions | Script |
|---|---|---|---|---|
| **Two-tower v6, fusion pipeline, pool=500** | **0.1518** | 31.5% | 1000 | `scripts/inference/run_inference_fusion.py --tt_model models/twotower_v6/final --tt_index cache/twotower_v6` |
| Two-tower v3, BM25 pool=500, w=0.7 | 0.1418 | 29.8% | 1000 | (previous best) |

Best blind submission (not yet submitted): `exp/inference/blind_a/blind_a_twotower_v3_qwen.json`

---

## Experiment Timeline

### Phase 1: BM25 Baseline
**Status: Done** — details in [plan/01_bm25_baseline.md](01_bm25_baseline.md)

| System | nDCG@20 | Hit@20 | Notes |
|---|---|---|---|
| BM25 name+artist+album | 0.0861 | 21.9% | Baseline floor |
| + tag_list in query | 0.0960 | 25.9% | Tags help recall |
| + exclude seen tracks | **0.1313** | 27.4% | Big jump from dedup |
| + CF reranking | 0.1196 | ~31% | Hit@20 up but nDCG down (50 sess) |
| + Qwen query rewrite | ~0.10 | — | No improvement |

**What worked:** tag expansion + seen exclusion.
**What failed:** CF reranking hurt nDCG despite improving Hit@20.

---

### Phase 2: Dense Retrieval (Pre-trained)
**Status: Done, abandoned** — details in [plan/01_bm25_baseline.md](01_bm25_baseline.md)

| System | nDCG@20 | Notes |
|---|---|---|
| Dense only (all-MiniLM-L6-v2 pre-trained) | 0.0654 | Worse than BM25 |
| BM25+Dense RRF (pre-trained) | 0.0775 | Worse than BM25 alone |
| Qwen3 dense probe (50 sess) | ~0.15 | Small sample, not reliable |

**Diagnosis:** Pre-trained models do not align with this music retrieval query format.
Fine-tuning required.

---

### Phase 3: Two-Tower Fine-Tuning
**Status: Done** — details in [plan/02_twotower.md](02_twotower.md)

| System | nDCG@20 | Hit@20 | Sessions | Notes |
|---|---|---|---|---|
| v1 long query, w=0.3, pool=200 | 0.1358 | 28.9% | 1000 | First fine-tune |
| v1 long query, w=0.7 | 0.1358 | 28.9% | 1000 | Weight doesn't matter for v1 |
| v3 compact query, w=0.7, pool=300 | 0.1406 | 29.4% | 1000 | Compact query key insight |
| **v3 compact query, w=0.7, pool=500** | **0.1418** | 29.8% | 1000 | Best |
| v3 compact query, w=0.7, pool=1000 | 0.1417 | 29.8% | 1000 | Plateaus at 500 |
| v3 + cluster hybrid recall | 0.1403 | 29.4% | 1000 | Cluster-only tracks score low |
| v3 + Dense+BM25 RRF | 0.1401 | 30.2% | 1000 | Hit up, nDCG down |
| v4 hard negatives, w=0.7 (200 sess) | 0.1364 | 29.2% | 200 | Regression |

**What worked:** compact query format, dense_weight=0.7, pool=500.
**What failed:** hard negatives (v4), cluster expansion, RRF fusion.

**Root cause of ceiling:** BM25 rank-1 signal overwhelms dense score.
Gold at BM25 rank 50 with cosine=0.6 → score 0.426. BM25 rank-1 non-gold with cosine=0.35 → score 0.545.
Theoretical ceiling with perfect reranking: nDCG@20=0.588.

---

### Phase 4: Cross-Encoder Reranker
**Status: Inconclusive** — details in [plan/03_crossencoder.md](03_crossencoder.md)

| System | nDCG@20 | Hit@20 | Sessions | Notes |
|---|---|---|---|---|
| v3 pure dense (w=1.0) | 0.1199 | 26.8% | 200 | confirms BM25 rank signal needed |
| v3 + pre-trained CE hybrid, pool=50 | 0.1014 | 24.1% | 200 | pool=50 limits recall |
| CE pre-trained (ms-marco), pool=50 | 0.0968 | 23.2% | 200 | zero-shot CE baseline |
| CE fine-tuned (final/epoch 2), pool=50 | 0.0455 | 14.0% | 200 | epoch 2 overfit, worse than pre-trained |

**Hypothesis:** Cross-encoder jointly attends (query, track) and produces a direct relevance score,
bypassing the BM25 rank vs. cosine tradeoff entirely.

---

### Phase 5: Two-Tower v5 (Triplet Loss)
**Status: Training complete — inference running** — details in [plan/04_v5_twotower.md](04_v5_twotower.md)

Training from v3/final with TripletLoss (margin=0.5) + hard negatives. Completed 2h 58m.
Eval loss improved through all of epoch 2 (0.3027 → 0.2480). Best = `models/twotower_v5/final`.

| System | nDCG@20 | Sessions | Notes |
|---|---|---|---|
| v5 triplet, hard neg, w=0.7, pool=500 | 0.0525 | 200 | FAILED — catastrophic regression, model collapsed |

---

### Phase 7: Min-pool recall (active)
**Status: Active** — details in [plan/06_min_pool_recall.md](06_min_pool_recall.md)

Goal: maximize gold-in-pool recall at the smallest pool size.
Today: 80.6% @ size ~1500. Target: ≥80% @ size ≤ 800, ≥85% @ size ≤ 1500.
Current best retrieval+rescore snapshot: [plan/CURRENT_BEST_ITERATION.md](CURRENT_BEST_ITERATION.md).

---

### Phase 6: Semantic ID (LLM-as-retriever)
**Status: Abandoned** — archived in [plan/archive/SEMANTIC_ID_PLAN.md](archive/SEMANTIC_ID_PLAN.md)



Fine-tuning Qwen2.5-0.5B with custom semantic ID tokens for end-to-end retrieval+generation.
Abandoned because: training infrastructure complexity, 47K-track catalog too large for
clean 256x256 codebook, and the BM25+dense pipeline already shows a viable path.

---

## Pending / Next Steps

**v6 bi-encoder is new best (0.1518). Sweep fusion weights, then blind submission.**

1. **Fusion weight sweep with v6** (IN PROGRESS) — v6 at default weights hits 0.1518. Sweep w_tt, w_bm25 to find better balance.

2. **Blind A submission with v6** — once weight sweep done:
   ```bash
   python scripts/inference/run_inference_blind_fusion.py \
       --tt_model models/twotower_v6/final --tt_index cache/twotower_v6 \
       --tid blind_a_v6
   python scripts/inference/generate_responses_blind.py --pred exp/inference/blind_a/blind_a_v6.json
   ```

3. **v6 dense recall expansion** — try `--tt_pool 250` to add TT-retrieved candidates beyond BM25 pool, since v6 has better cosine quality.

4. **CE checkpoint-10830 eval** (deferred) — still worth trying if v6 weight sweep plateaus.

---

## File Map

```
scripts/
  train_twotower.py           Fine-tune all-MiniLM-L6-v2 bi-encoder
  train_crossencoder.py       Fine-tune ms-marco-MiniLM cross-encoder
  build_twotower_data.py      Build (query, track) pairs for bi-encoder
  build_crossencoder_data.py  Build (query, track, label) pairs for CE
  build_twotower_index.py     Encode all tracks with fine-tuned model
  build_dense_index.py        Encode tracks with pre-trained model
  evaluate_local.py           nDCG@20 evaluator against dev ground truth
  run_inference_twotower_v3.py    BEST: BM25+two-tower v3 devset inference
  run_inference_crossencoder.py   CE reranker devset inference
  run_inference_blind_twotower.py Blind A inference (two-tower v3)
  run_inference_blind.py          Blind A inference (BM25 only fallback)
  generate_responses_blind.py     Qwen3 response generation for blind preds
  archive/                    All superseded scripts

models/
  twotower_v3/final           BEST bi-encoder (2 epochs, compact query)
  twotower_v4/final           Hard-neg bi-encoder (worse than v3)
  crossencoder_v1/final       CE reranker (training in progress)

data/
  twotower_v3/                Bi-encoder training data (115K/6K)
  crossencoder_v1/            CE training data (115K train_small, 693K full)

cache/
  twotower_v3/                Dense track index (47071 x 384)
  bm25/track_metadata/        BM25 index

exp/inference/
  devset/                     Dev set predictions (JSON)
  blind_a/                    Blind A predictions
```

---

## Score Ladder

```
0.1518  v6 bi-encoder, fusion pipeline, pool=500 ← CURRENT BEST (1000 sess)
0.1418  v3 bi-encoder, pool=500, w=0.7
0.1423  v3 bi-encoder, pool=500, w=0.7         ← same config (200 sess sample)
0.1417  v3 bi-encoder, pool=1000, w=0.7
0.1406  v3 bi-encoder, pool=300, w=0.7
0.1403  v3 + cluster hybrid recall
0.1401  v3 dense + BM25 RRF
0.1364  v4 hard negs (200 sess only)
0.1358  v1 bi-encoder, long query
0.1313  BM25 + tag + seen exclusion
0.1199  v3 pure dense w=1.0 (200 sess)
0.1014  v3 + pre-trained CE hybrid pool=50 (200 sess)
0.0968  CE pre-trained (ms-marco) pool=50 (200 sess)
0.0960  BM25 + tag
0.0861  BM25 name+artist+album only
0.0654  Dense only (pre-trained)
0.0525  v5 TripletLoss, pool=500, w=0.7 (200 sess) — collapsed, worse than BM25
0.0455  CE fine-tuned (epoch 2 / final) pool=50 — overfit
```
