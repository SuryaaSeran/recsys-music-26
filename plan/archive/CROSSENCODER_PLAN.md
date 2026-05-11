# Plan: Cross-Encoder Reranker

## Goal

Replace the bi-encoder cosine score in the hybrid pipeline with a cross-encoder reranker.
Target: beat v3 best of nDCG@20=0.1418.

## Current State

Best system: v3 bi-encoder + BM25, dense_weight=0.7, pool=500 → nDCG@20=0.1418

Root cause of failure: BM25 rank-1 signal overwhelms cosine. Gold at BM25 rank 50 with
cosine=0.6 scores 0.426; BM25 rank-1 competitor with cosine=0.35 scores 0.545.

Hard negative approach (v4) made things worse: 0.1364 vs 0.1418 (on comparable sessions).
Cross-encoder directly attends to (query, track) jointly — avoids the dual-encoder
bottleneck and should produce much sharper relevance scores.

## Assumptions

- Cross-encoder (ms-marco-MiniLM-L-6-v2) fine-tuned on (query, BM25-candidate, label) pairs
  will produce more accurate reranking than bi-encoder cosine similarity.
- Using BM25 top-200 candidates as the pool (current training data uses top-5 negatives
  per positive from BM25 pool).
- Inference: BM25 pool=200, cross-encoder reranks all 200, take top-20.

## Training (IN PROGRESS)

Script: `scripts/train_crossencoder.py`
PID: 89689
Args: --data_dir data/crossencoder_v1 --out_dir models/crossencoder_v1 --epochs 2 --batch_size 16
Base model: cross-encoder/ms-marco-MiniLM-L-6-v2
Data: 115,520 train / 36,432 valid (train_small.jsonl)
Total steps: 14,440
Progress: ~1805/14440 (13%), checkpoint-1805 saved
Eval loss @ epoch 0.25: 0.3481 (started at ~0.38)
ETA: ~02:13 AM May 4

Log: /tmp/ce_train.log
Checkpoint: models/crossencoder_v1/checkpoint-1805
Final will be: models/crossencoder_v1/final

## Files To Read (already done)

- scripts/train_crossencoder.py — confirmed structure
- data/crossencoder_v1/ — 115K train, 36K valid, 693K full train

## Files To Modify / Create

- scripts/run_inference_crossencoder.py — new inference script (BM25 pool + CE rerank)

## Steps

### Step 1 (DONE): Train cross-encoder
Wait for training to complete (~02:13 AM). Check final model at models/crossencoder_v1/final.

### Step 2: Build inference script
After training completes, create `scripts/run_inference_crossencoder.py`:
- Load BM25 index from cache/bm25/track_metadata/
- For each turn: BM25 top-200, cross-encoder scores all 200, exclude seen, take top-20
- Use compact query format (same as v3): {latest_user_turn} {goal} {culture}
- Track text: same as BM25 document (name + artist + album + tags)

### Step 3: Eval on 200-session subset first
```bash
python scripts/run_inference_crossencoder.py \
    --model models/crossencoder_v1/final \
    --bm25_pool 200 --n 200 \
    --out exp/inference/devset/ce_v1_pool200_n200.json
python scripts/evaluate_local.py --pred exp/inference/devset/ce_v1_pool200_n200.json
```
Compare to v4 best (0.1364 on 200 sessions) and v3 (0.1418 on 1000).

### Step 4: Pool size sweep if Step 3 is promising
Try pool=100, 200, 500. CE is slow at inference (~0.1s/pair × 200 = 20s/turn).
May need to batch queries.

### Step 5: Full 1000-session eval
```bash
python scripts/run_inference_crossencoder.py \
    --model models/crossencoder_v1/final \
    --bm25_pool 200 \
    --out exp/inference/devset/ce_v1_pool200_full.json
python scripts/evaluate_local.py --pred exp/inference/devset/ce_v1_pool200_full.json
```

### Step 6: Hybrid CE+bi-encoder (if CE beats v3)
Score = alpha * CE_score + (1-alpha) * cosine_sim
Sweep alpha. This combines the cross-encoder's joint attention with bi-encoder's
precomputed dense index.

### Step 7: Blind A submission if CE pipeline improves on v3
```bash
python scripts/run_inference_crossencoder.py \
    --model models/crossencoder_v1/final --blind_a \
    --out exp/inference/blind_a/blind_a_ce_v1.json
python scripts/generate_responses_blind.py \
    --pred exp/inference/blind_a/blind_a_ce_v1.json
```

## Validation

Success: nDCG@20 > 0.1418 on full 1000 sessions
Stretch: nDCG@20 > 0.15

## Risks

- CE inference is slow. Pool=200 × 8000 turns = 1.6M pairs to score. Need batched inference.
- Training data uses only top-5 BM25 negatives per positive. Model may not generalize to
  ranking within all 200. If so, retrain with full 200-candidate negatives.
- train_small.jsonl (115K) vs full train.jsonl (693K). May need to retrain on full data
  if results are disappointing.

## Notes

Experiment history:
- v3 bi-encoder (compact query, 2 epochs, in-batch negs): 0.1418 → BEST
- v4 bi-encoder (hard negs, 1 epoch from v3): 0.1364 → WORSE
- Cross-encoder approach: in progress

Data files:
- data/crossencoder_v1/train_small.jsonl — 115,520 examples (5:1 neg ratio)
- data/crossencoder_v1/train.jsonl — 693,120 examples (full)
- data/crossencoder_v1/valid.jsonl — 36,432 examples
