# Phase 3: Two-Tower Fine-Tuning

**Status: Complete (v3 is current best)**
**Best result:** nDCG@20=0.1418, Hit@20=29.8% (1000 sessions)
**Best config:** `models/twotower_v3/final`, pool=500, dense_weight=0.7
**Run:** `python scripts/run_inference_twotower_v3.py --model models/twotower_v3/final --bm25_pool 500 --dense_weight 0.7`

---

## Pipeline

```
Query (compact format)
    |
    v
BM25 top-500 candidates    (full long query for recall)
    |
    v
Two-tower cosine similarity (compact query for reranking)
    |
    v
Score = 0.7 * cosine_sim + 0.3 * bm25_reciprocal_rank
    |
    v
Exclude seen tracks → top-20
```

**Compact query format:**
```
{latest_user_turn} {goal} {culture} {last_2_track_name_artist}
```
Median 101 tokens. Avoids 256-token truncation (81% of full queries exceed limit).

**BM25 query format (recall layer):**
```
{goal} {culture} {last_4_track_name_artist_album_tags} {last_4_user_turns}
```

---

## Experiment History

### v1: Long Query (Failed to beat BM25 meaningfully)

Training data: `data/twotower/` — full long query as anchor
Base model: `all-MiniLM-L6-v2` (pre-trained)
Training: 1 epoch, batch=32, lr=2e-5, in-batch negatives
Script: `scripts/train_twotower.py`

| Config | nDCG@20 | Hit@20 | Result file |
|---|---|---|---|
| v1 pool=200, w=0.3 | 0.1358 | 28.9% | `devset/twotower_v1_pool200_w03.json` |
| v1 pool=200, w=0.7 | 0.1358 | 28.9% | `devset/twotower_v1_w07.json` |
| v1 pool=500 | ~0.136 | — | `devset/twotower_v1_pool200_dense500_w03.json` |

**Problem diagnosed:** 81% of full queries exceed 256-token limit → heavy truncation → model
sees garbled input. Dense weight barely matters because cosine similarity is unreliable.

### Key Insight: Compact Query Format

Switch anchor to: `{latest_user_turn} {goal} {culture} {last_2_track_name_artist}`
- Median 101 tokens (well within 256 limit)
- Prioritizes the user's current request (first token gets most attention)
- Drops stale history that confuses the model

### v3: Compact Query (Current Best)

Training data: `data/twotower_v3/` — 115K train / 6K valid, compact query format
Base model: `all-MiniLM-L6-v2`
Training: 2 epochs, batch=32, lr=2e-5, in-batch negatives only
Script: `python scripts/train_twotower.py --data_dir data/twotower_v3 --out_dir models/twotower_v3 --epochs 2`

| Config | nDCG@20 | Hit@20 | Sessions | Result file |
|---|---|---|---|---|
| v3 w=0.3, pool=200 | 0.1358 | 28.9% | 1000 | `devset/twotower_v3_w03.json` |
| v3 w=0.7, pool=200 | 0.1406 | 29.4% | 1000 | `devset/twotower_v3_w07.json` |
| v3 w=0.7, pool=300 | 0.1406 | 29.4% | 1000 | `devset/twotower_v3_pool300_w07.json` |
| **v3 w=0.7, pool=500** | **0.1418** | **29.8%** | **1000** | `devset/twotower_v3_pool500_w07.json` |
| v3 w=0.7, pool=1000 | 0.1417 | 29.8% | 1000 | `devset/twotower_v3_pool1000_w07.json` |
| v3 w=0.8 | 0.1364 | — | 1000 | `devset/twotower_v3_w08.json` |
| v3 w=0.9 | 0.1285 | — | 1000 | `devset/twotower_v3_w09.json` |

**Findings:**
- Optimal dense_weight=0.7. Higher weights hurt: model over-trusts cosine over BM25 precision.
- Pool size plateaus at 500. Beyond 500, new candidates don't contain gold tracks.
- v3b (variant) matched v3 at 0.1418 with pool=500.

### v3 + Cluster/Hybrid Expansion

Script: `scripts/archive/run_inference_hybrid_recall.py`

Added a second recall path: encode query with v3, find nearest cluster centroids, expand
all cluster members as additional candidates alongside BM25 top-500.

| Config | nDCG@20 | Hit@20 | Result file |
|---|---|---|---|
| K=500 c=3 w=0.7 | 0.1403 | 29.4% | `devset/hybrid_k500_c3_w07_full.json` |
| K=500 c=3 RRF | 0.1396 | 30.1% | `devset/hybrid_k500_c3_rrf_full.json` |
| v3 Dense+BM25 RRF | 0.1401 | 30.2% | `devset/rrf_d500_b500_full.json` |

**Finding:** Cluster expansion improves Hit@20 (+0.3-0.4%) but consistently hurts nDCG@20.
Cluster-only gold tracks appear in the pool but cannot beat BM25 rank-1 competitors in scoring.
The problem is the scoring formula, not the recall pool.

### v4: Hard Negatives (Failed)

Motivation: root cause analysis showed BM25 rank-1 competitors were beating gold tracks
despite lower cosine similarity. Hard negatives should sharpen the cosine gap.

Training: 1 epoch from v3/final, batch=32, lr=1e-5, using `negative_1` column (BM25 top-5 non-gold)
Script: `python scripts/train_twotower.py --data_dir data/twotower_v3 --base_model models/twotower_v3/final --hard_neg --epochs 1 --lr 1e-5`

| Config | nDCG@20 | Hit@20 | Sessions | Result file |
|---|---|---|---|---|
| v4 w=0.70 | 0.1364 | 29.2% | 200 | `devset/v4_w070_200.json` |
| v4 w=0.75 | 0.1364 | 28.9% | 200 | `devset/v4_w075_200.json` |
| v4 w=0.80 | 0.1336 | 28.3% | 200 | `devset/v4_w080_200.json` |
| v4 w=0.85 | 0.1285 | 27.2% | 200 | `devset/v4_w085_200.json` |

**Finding:** v4 regressed at all weight settings. Hard negatives from BM25 top-5 may be
too easy (not actually competitive with the gold track in cosine space), causing the model
to overfit to superficial differences. Moving to cross-encoder instead.

---

## Root Cause Analysis

Diagnostic on 50 sessions, 261 turns where gold IS in BM25 pool:

| Metric | Value |
|---|---|
| Gold track in BM25 top-10 | 40.6% |
| Gold track in BM25 ranks 11-50 | 28.4% |
| Gold track in BM25 ranks 51-200 | 21.5% |
| Gold track in BM25 ranks 201-500 | 9.6% |
| Gold cosine sim > 0.5 | 82.8% |
| Gold in final top-20 (after scoring) | 58.2% |
| Gold in final ranks 21-100 | 26.1% |
| Gold in final ranks 100+ | 15.7% |

**The failure mode:** `score = 0.7 * cosine + 0.3 * (1/(1 + bm25_rank))`.
A BM25 rank-1 track with cosine=0.35 scores: 0.7*0.35 + 0.3*1.0 = **0.545**.
A gold track at BM25 rank 50 with cosine=0.60 scores: 0.7*0.60 + 0.3*(1/51) = **0.426**.
The BM25 rank-1 signal is too powerful to overcome even with much higher cosine.

**Theoretical ceiling:** nDCG@20 = 0.588 (with perfect reranking within BM25 pool).
Current system is at 0.1418 — about 24% of ceiling. Huge room for a better reranker.

---

## Data Files

- `data/twotower_v3/train.jsonl` — 115K training examples (compact query format)
- `data/twotower_v3/valid.jsonl` — 6K validation examples
- `data/twotower_v3/` has columns: `anchor` (query), `positive` (gold track text), `negative_1..5` (BM25 non-gold)
- `cache/twotower_v3/` — encoded track index: `track_embeddings.npy` (47071 × 384) + `track_ids.json`
