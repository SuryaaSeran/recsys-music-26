# Phase 4: Cross-Encoder Reranker

**Status: Training complete — ready for inference**
**Hypothesis:** Joint (query, track) attention directly predicts relevance, bypassing the
BM25-rank-vs-cosine scoring tradeoff that limits the bi-encoder pipeline.
**Target:** nDCG@20 > 0.1418

---

## Motivation

From Phase 3 root cause analysis: the bi-encoder scoring formula
`score = 0.7 * cosine + 0.3 * bm25_rr` allows BM25 rank-1 noise to beat gold tracks
with higher cosine. A cross-encoder reads both query and track in a single pass,
producing a relevance score not dependent on any formula.

---

## Training

**Model:** `cross-encoder/ms-marco-MiniLM-L-6-v2` (fine-tuned on MS-MARCO passage ranking)
**Task:** Binary classification — (query, track_text) → relevant (1) or not (0)
**Data:** `data/crossencoder_v1/train_small.jsonl` — 115,520 examples
**Data format:**
```json
{"query": "...", "document": "track_name artist tag1 tag2 ...", "label": 0|1}
```
Positives: (compact_query, gold_track_text, 1)
Negatives: top-5 BM25 non-gold tracks per turn (5:1 ratio)

**Training command:**
```bash
python scripts/train_crossencoder.py \
    --data_dir data/crossencoder_v1 \
    --out_dir models/crossencoder_v1 \
    --epochs 2 --batch_size 16
```

**Status:**
- Started: ~00:13 AM May 4 2026
- **COMPLETE** — 14440/14440 steps, 2h 46m, train_loss=0.3059
- Eval loss curve: 0.3481 → 0.3395 → 0.3299 → 0.3265 → 0.3297 → **0.3242 (epoch 1.5)** → 0.3331 → 0.3297 (epoch 2)
- **Best checkpoint on disk: `models/crossencoder_v1/checkpoint-12635`** (epoch 1.75, eval_loss=0.3331)
- checkpoint-10830 (epoch 1.5, best eval 0.3242) was overwritten by trainer save_total_limit=2
- `final` = epoch 2.0, eval_loss=0.3297 (also on disk, slightly better eval than checkpoint-12635)

**Alternative:** `data/crossencoder_v1/train.jsonl` has 693K examples (full dataset).
If train_small results are weak, retrain on full data.

---

## Inference Pipeline

Script: `scripts/run_inference_crossencoder.py`

```
Compact query (latest_user_turn + goal + culture)
    |
    v
BM25 top-CE_POOL candidates   (lexical recall, full long query)
    |
    v
Cross-encoder scores all CE_POOL (query, track) pairs in batch
    |
    v
Sort by CE score, exclude seen tracks → top-20
```

**Run (200-session quick eval):**
```bash
python scripts/run_inference_crossencoder.py \
    --ce_model models/crossencoder_v1/checkpoint-10830 \
    --ce_pool 200 --tid ce_v1_pool200 --sessions 200
python scripts/evaluate_local.py --pred exp/inference/devset/ce_v1_pool200.json
```

**Pool size tradeoff:**
- Pool=50: fast (~0.5s/turn), but recall limited — gold track may not be in BM25 top-50
- Pool=200: ~2s/turn, reasonable recall
- Pool=500: ~5s/turn, same recall as bi-encoder pipeline
For full 1000-session eval (8000 turns), pool=200 is preferred for speed.

---

## Evaluation Plan

### Step 1: 200-session quick check (DONE — wrong model/pool, rerun needed)
First run used `final` + pool=50 → nDCG@20=0.0455 (pool too small, gold recall destroyed).

**Rerun with correct settings:**
```bash
python scripts/run_inference_crossencoder.py \
    --ce_model models/crossencoder_v1/checkpoint-10830 \
    --ce_pool 200 --tid ce_v1_ck10830_pool200 --sessions 200
python scripts/evaluate_local.py --pred exp/inference/devset/ce_v1_ck10830_pool200.json
```
Compare to v3 on same 200 sessions (0.1423).

### Step 1b: Pre-trained CE baseline (running — 133/200 sessions, ETA ~5 min)
```bash
# ce_pretrained_pool50_200.json — zero-shot ms-marco, pool=50
python scripts/evaluate_local.py --pred exp/inference/devset/ce_pretrained_pool50_200.json
```

### Step 1c: Fine-tuned CE pool=200 (CRASHED)
Run `models/crossencoder_v1/final` with pool=200 crashed after 1 session (OOM/semaphore leak).
Must run sequentially (not parallel) to avoid memory contention. Use checkpoint-10830.
```bash
python scripts/run_inference_crossencoder.py \
    --ce_model models/crossencoder_v1/checkpoint-10830 \
    --ce_pool 200 --tid ce_v1_ck10830_pool200 --sessions 200
```

### Step 2: Pool sweep (if Step 1 is promising)
```bash
for pool in 50 100 200 500; do
  python scripts/run_inference_crossencoder.py \
      --ce_model models/crossencoder_v1/checkpoint-10830 \
      --ce_pool $pool --tid ce_v1_pool${pool} --sessions 200
  python scripts/evaluate_local.py --pred exp/inference/devset/ce_v1_pool${pool}.json
done
```

### Step 3: Full 1000-session eval
```bash
python scripts/run_inference_crossencoder.py \
    --ce_model models/crossencoder_v1/checkpoint-10830 \
    --ce_pool 200 --tid ce_v1_pool200_full
python scripts/evaluate_local.py --pred exp/inference/devset/ce_v1_pool200_full.json
```

### Step 4: CE+bi-encoder hybrid (if CE beats v3)
Score = alpha * CE_score + (1-alpha) * cosine_sim
Sweep alpha in [0.3, 0.5, 0.7, 1.0].

---

## Results

| Config | nDCG@20 | Hit@20 | Sessions | Result file | Notes |
|---|---|---|---|---|---|
| v3 bi-encoder (baseline) | 0.1423 | 30.9% | 200 | `devset/v3_w070_200.json` | Best system |
| Pre-trained CE (ms-marco), pool=50 | 0.0968 | 23.2% | 200 | `devset/ce_pretrained_pool50_200.json` | Zero-shot, pool too small |
| Fine-tuned CE (final, epoch 2), pool=50 | 0.0455 | 14.0% | 200 | `devset/ce_v1_pool50_200.json` | Worse than pre-trained — red flag |

**Red flag:** Fine-tuned CE (epoch 2 / `final`) scores WORSE than zero-shot pre-trained CE at the same pool size.
Pre-trained 0.0968 vs fine-tuned 0.0455. Both use pool=50 so pool size is not the differentiator here.

**Possible causes:**
1. `final` is epoch 2 — eval_loss rose from best (0.3242 at epoch 1.5) to ~0.33. Overfit.
2. Score collapse: fine-tuned may output near-identical scores for all pairs → random ordering
3. Query format mismatch: inference query format may differ from training format

**Next diagnostic:** Run checkpoint-10830 (best eval loss) at pool=50 solo. If still worse than pre-trained, the training approach is broken. If better, epoch 2 overfit is confirmed.

---

## Decision Tree After Eval

**CE > 0.1418 on 200 sessions:**
→ Run full 1000-session eval
→ If still beats v3, submit blind A with CE pipeline

**CE between 0.1364 and 0.1418:**
→ Try CE+bi-encoder hybrid (alpha sweep)
→ Try retraining on full 693K examples

**CE < 0.1364 (worse than v4):**
→ CE model may be undertrained. Check eval_loss trend.
→ Try checkpoint-based eval (checkpoint-1805 vs final)
→ Consider: harder negatives (BM25 top-200 instead of top-5)

---

## Data Files

- `data/crossencoder_v1/train_small.jsonl` — 115,520 examples (5:1 neg/pos ratio)
- `data/crossencoder_v1/train.jsonl` — 693,120 examples (full)
- `data/crossencoder_v1/valid.jsonl` — 36,432 examples
- Built by: `scripts/build_crossencoder_data.py`
