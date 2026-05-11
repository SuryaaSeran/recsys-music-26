# Current Best Iteration

Snapshot of the best-tested recall pool and rescore method, including what has been
run on the blind dataset.

## Headline

- Dev nDCG@20: 0.1519 (1000 sessions, full 8000 turns)
- Dev Hit@20: 31.5%
- Pool recall ceiling (where gold is reachable at all): 80.6% at total pool ≈ 1500
- Pool recall actually used in scored pipeline: BM25@500 only (59.0%)
- Gap: pool recall reachable (80.6%) vs ranking ceiling realized (gold-in-top-20)
  is dominated by the rescore step, not retrieval.

## Best Retrieval Pool (recall-only, no rescore)

Measured on full 1000 dev sessions / 8000 turns. TT = two-tower v6.

| Pool | Recall | Total pool size |
|---|---|---|
| BM25@500 only | 0.590 | 500 |
| BM25@500 + artist expansion | 0.651 | ~700 (var.) |
| BM25@500 + TT-v6@500 | 0.729 | ~1000 |
| BM25@500 + artist + TT-v6@500 | 0.755 | ~1200 |
| BM25@500 + artist + TT-v6@1000 | 0.806 | ~1500 |

Source files:
- `exp/analysis/recall_v6_baseline.txt`
- `exp/analysis/recall_v6_artist_expansion.txt`
- `exp/analysis/recall_v6_combined.txt`
- `exp/analysis/recall_audit_summary.txt`

Notes:
- Artist expansion = scan all conversation text + played-track artist names, add every
  catalog track for each verbatim-mentioned artist. Implementation in
  `scripts/inference/measure_recall_artist_expansion.py`.
- Qwen-meta @500 union adds ~+1.5%; Qwen-lyrics adds ~+1%; CLAP negligible; CF (warm
  only) adds ~+2%. None beat the cost of TT depth expansion.
- BM25 query/index tuning has not improved pool recall (any reformulation tested so
  far either matches or hurts; field-weighted index hurt -1.9%).

## Best Rescore Method (currently submitted on blind)

Script (dev): `scripts/inference/run_inference_fusion.py`
Script (blind): `scripts/inference/run_inference_blind_fusion.py`

Pool used by rescore: BM25@500 only (no artist or TT recall expansion).
Per-candidate score (linear fusion):

```
score = w_tt          * tt_cosine          (two-tower v6)
      + w_qwen_meta   * qwen3_meta_cosine
      + w_qwen_lyrics * qwen3_lyrics_cosine
      + w_clap        * clap_text_cosine
      + w_cf          * cf_bpr_cosine      (warm users only; 0 for cold)
      + w_bm25        * bm25_norm          (s / s_max within retrieved pool)
```

Weights (grid-searched, v13_tuned):

```
w_tt          = 0.32
w_qwen_meta   = 0.40
w_qwen_lyrics = 0.08
w_clap        = 0.05
w_cf          = 0.10
w_bm25        = 0.24
bm25_norm     = True   (s / s_max, not reciprocal rank)
```

Dev nDCG@20 = 0.1519, Hit@20 = 31.5% (1000 sessions, verified via
`score_precomputed.py` exact reproduction).

## Tested on Blind A

| File | System | Status |
|---|---|---|
| `exp/inference/blind_a/blind_a_fusion_v13_tuned.json` | v13 fusion weights above | Predictions generated |
| `exp/inference/blind_a/blind_a_fusion_v13_tuned_qwen.json` | + Qwen responses | Generated |
| `exp/inference/blind_a/blind_a_fusion_v9_norm_qwen.json` | v9 fusion (dev 0.1489) | Fallback |
| `exp/inference/blind_a/blind_a_fusion_v6_qwen.json` | v6 fusion (dev 0.1473) | Fallback |
| `exp/inference/blind_a/blind_a_twotower_v3_qwen.json` | TT-v3 + BM25 only (dev 0.1418) | Fallback |
| `exp/inference/blind_a/blind_a_v2_qwen.json` | BM25 + tag + seen excl (dev 0.1313) | Submitted earlier |

No blind submission uses recall expansion (artist or TT pool). The rescore-on-expanded
pool variant (`run_inference_fusion_recall_expansion.py`) was tested on dev only: with
BM25_missing_floor=0.05 it produced nDCG = 0.1518 — no gain over the BM25@500-only pool.
Cause: rescued candidates start with bm25_signal=floor, can't out-score BM25 rank-1
incumbents under current weights.

## What this implies for next work

- Pool recall at the size we actually rescore (≈500) is the bottleneck: 59.0% BM25
  vs 80.6% reachable. Closing this gap at small pool size is the highest-leverage move.
- Rescore-side fixes (source-aware weights, BM25 floor sweep) need to be paired with
  pool-side fixes; rescore alone with BM25@500 cannot exceed 59.0% gold availability.
- For blind submission, current best stays `blind_a_fusion_v13_tuned_qwen.json` until
  a higher dev nDCG run is reproduced.
