# Phase 5: Two-Tower v5 (Triplet Loss)

**Status: Training complete — ready for eval**
**Motivation:** v4 hard negatives with MultipleNegativesRankingLoss regressed.
TripletLoss directly optimizes `sim(anchor, pos) - sim(anchor, neg) > margin`,
matching the root cause: gold track needs a strictly higher cosine than BM25 competitors.

---

## Training

**Command:**
```bash
python scripts/train_twotower.py \
    --data_dir data/twotower_v3 \
    --out_dir models/twotower_v5 \
    --base_model models/twotower_v3/final \
    --epochs 2 --batch_size 16 --lr 1e-5 --warmup_steps 100 \
    --hard_neg --triplet --triplet_margin 0.5
```

**Differences from v4:**
| | v4 | v5 |
|---|---|---|
| Loss | MultipleNegativesRankingLoss | TripletLoss (margin=0.5) |
| Batch size | 32 | 16 |
| Epochs | 1 | 2 |
| Hard neg | yes | yes |

**Status:**
- Started: ~04:27 AM May 4 2026
- Completed: 14440/14440 steps, 2h 58m, train_loss=0.2627
- Eval losses: 0.3027 → 0.2793 → 0.2654 → 0.2573 → 0.2575 → 0.2504 → 0.2491 → **0.2480 (ep 2.0)**
- **Best checkpoint: `models/twotower_v5/final`** (epoch 2.0, eval_loss=0.2480) — improved through all of epoch 2
- Log: `/tmp/twotower_v5_train.log`
- Log: `/tmp/twotower_v5_train.log`
- Output: `models/twotower_v5/final`

---

## Evaluation Plan

Once training completes, rebuild the dense index and run inference:

```bash
# Step 1: Build new dense index
python scripts/build_twotower_index.py \
    --model models/twotower_v5/final \
    --out_dir cache/twotower_v5

# Step 2: Eval on 200 sessions
python scripts/run_inference_twotower_v3.py \
    --model models/twotower_v5/final \
    --index_dir cache/twotower_v5 \
    --bm25_pool 500 --dense_weight 0.7 \
    --sessions 200 --tid twotower_v5_pool500_w07

python scripts/evaluate_local.py \
    --pred exp/inference/devset/twotower_v5_pool500_w07.json
```

Compare to v3 baseline on same 200 sessions (0.1423).

---

## Results

| Config | nDCG@20 | Hit@20 | Sessions | Notes |
|---|---|---|---|---|
| v5 final, pool=500, w=0.7 | **0.0525** | 9.9% | 200 | Catastrophic — far worse than v3 (0.1423) |

**FAILED.** v5 TripletLoss is worse than pure BM25 (0.1313) and worse than dense-only (0.1199).
Hit@20=9.9% vs v3's 30.9% — the model is actively destroying BM25 recall via bad cosine scores.

**Failure analysis:**
- TripletLoss with margin=0.5 may have caused representation collapse: all embeddings in a tight cluster, cosine similarities near-uniform across tracks
- Or model overfit to training negatives (BM25 top-5) but doesn't generalize to 47K catalog
- Result: v5 cosine scores are uninformative, but dense_weight=0.7 forces them to dominate scoring

**Conclusion:** TripletLoss approach abandoned. v3 (nDCG=0.1418) remains best.

---

## Context: What Was Tried Before v5

The scoring formula `0.7*cosine + 0.3*bm25_rr` means a BM25 rank-1 track with
cosine=0.35 beats a gold track at BM25 rank 50 with cosine=0.60. The model needs to
produce cosine=0.75+ for gold vs ~0.35 for BM25 competitors to win.

- **v4** (hard negs, MultipleNegativesRankingLoss): regressed to 0.1364 — loss doesn't
  directly penalize the margin between gold and hard negative cosine similarity.
- **CE approach**: pre-trained ms-marco beats fine-tuned CE at same pool size (0.0968 vs 0.0455)
  — fine-tuned CE (epoch 2) appears to have collapsed. Checkpoint-10830 untested.
- **v5 TripletLoss**: explicitly enforces `sim(q, gold) - sim(q, neg) > 0.5`.
  If it works, the gold track will have a cosine 0.5+ higher than hard negatives.
