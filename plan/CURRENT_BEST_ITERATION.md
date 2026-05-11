# Current Best Iteration

Live snapshot. Update only when full 1000-session dev nDCG@20 strictly beats this.

## Best (as of 2026-05-11)

- **Dev nDCG@20: 0.1533**
- **Hit@20: 31.7%** (2538 / 8000 turns)
- Script: `scripts/inference/run_inference_fusion_recall_expansion.py`
- Run id: `v07_parity_B` (`exp/inference/devset/v07_parity_B.json`)
- One-line reason it beat the prior best: adding last-track TT-NN@100 to the
  candidate pool lets a small fraction of NN-source gold tracks outscore BM25
  incumbents under the existing fusion weights.

### Retrieval pool

```
BM25@500
+ artist expansion (popularity-sorted catalog, --artist_cap 50)
+ TT-v6@1000
+ last-track-NN@100 in TT space (last_nn_src=2)
```

Mean deduped pool size: ~1450.

Reach metrics (8000 turns, audit):

| Source | Cumulative pool recall |
|---|---|
| BM25@500                                | 0.590 |
| + artist expansion                      | 0.651 |
| + TT-v6@1000                            | 0.806 |
| + last-track-NN@100                     | 0.808 |

### Rescore

```
score = w_tt          * tt_cosine
      + w_qwen_meta   * qm_cosine
      + w_qwen_lyrics * ql_cosine
      + w_clap        * clap_cosine
      + w_cf          * cf_cosine                 # warm users only
      + w_bm25        * bm25_signal               # = bm25_norm if in BM25 pool else floor
      + w_tt_rank     * tt_rank_sig               # 0 in this iter
      + w_artist      * artist_sig                # 0 in this iter
      + w_nn          * nn_sig                    # 0 in this iter
      + w_bm25_origin * bm25_origin               # 0 in this iter
```

Weights (v13_tuned + new features at 0):

```
w_tt          = 0.32
w_qwen_meta   = 0.40
w_qwen_lyrics = 0.08
w_clap        = 0.05
w_cf          = 0.10
w_bm25        = 0.24
bm25_norm     = True
bm25_missing_floor = 0.05
w_tt_rank = w_artist = w_nn = w_bm25_origin = 0
```

Reproduction:

```bash
python scripts/inference/run_inference_fusion_recall_expansion.py \
  --tid current_best \
  --tt_model models/twotower_v6/final --tt_index cache/twotower_v6 \
  --tt_pool 1000 --artist_expansion --last_nn_k 100 --last_nn_src 2 \
  --bm25_missing_floor 0.05 \
  --w_tt 0.32 --w_cf 0.10 --w_qwen_meta 0.40 --w_qwen_lyrics 0.08 \
  --w_clap 0.05 --w_bm25 0.24 \
  --w_tt_rank 0 --w_artist 0 --w_nn 0 --w_bm25_origin 0
```

### Tested on Blind A

Not yet re-run with this config. Next blind submission should regenerate using
the same flags above against `talkpl-ai/TalkPlayData-Challenge-Blind-A` via
`run_inference_blind_fusion.py` (needs --last_nn_k and --artist_expansion ported).

## Organizer baselines (full 1000-session dev nDCG@20)

Measured locally with `scripts/inference/evaluate_local.py` against the shipped
baseline scripts under `music-crs-baselines/`.

| Baseline | Source | nDCG@1 | nDCG@10 | nDCG@20 | Hit@20 |
|---|---|---:|---:|---:|---:|
| Random sample (20 tracks)        | `music-crs-baselines/lowerbound/random_sample.py` | 0.0000 | 0.0002 | 0.0002 | 0.1% |
| Popularity (top-20 train tracks) | `music-crs-baselines/lowerbound/popularity.py`    | 0.0005 | 0.0018 | 0.0024 | 0.6% |
| BM25 (Llama-3.2-1B response)     | `music-crs-baselines/run_inference_devset.py` cfg `llama1b_bm25_devset` | — | — | not measured (LLM-gated) | — |
| BERT dense (Llama-3.2-1B resp.)  | `music-crs-baselines/run_inference_devset.py` cfg `llama1b_bert_devset` | — | — | not measured (LLM-gated) | — |

BM25 / BERT require the Llama-3.2-1B inference pipeline (flash-attn, GPU); not
run locally. Our own BM25-only configuration (corpus = name+artist+album,
identical to the organizer BM25 retrieval side) scored 0.0861 nDCG@20 — that's
the comparable retrieval-only floor.

## Current system vs baselines

| System | nDCG@20 | Hit@20 | Lift vs popularity | Lift vs our BM25 floor |
|---|---:|---:|---:|---:|
| Random                                                               | 0.0002 | 0.1%  | -      | -      |
| Popularity                                                           | 0.0024 | 0.6%  | -      | -      |
| Our BM25 floor (name+artist+album)                                   | 0.0861 | 21.9% | x36    | -      |
| Our BM25 + tag_list + seen exclusion                                 | 0.1313 | 27.4% | x55    | +52%   |
| Our TT-v3 fusion (pool=500, w=0.7)                                   | 0.1418 | 29.8% | x59    | +65%   |
| Our v6 fusion v13_tuned (BM25@500 only)                              | 0.1519 | 31.5% | x63    | +76%   |
| **Our v6 fusion + expansion (artist + TT@1000 + NN@100), v13 wts**   | **0.1533** | **31.7%** | **x64** | **+78%** |

## Previous bests

| Date | nDCG@20 | Pool | Rescore | Note |
|---|---|---|---|---|
| 2026-05-05 | 0.1519 | BM25@500 only | v13_tuned weights | Prior best. Blind file: `blind_a_fusion_v13_tuned_qwen.json`. |
| 2026-05-04 | 0.1518 | BM25@500 + artist + TT-v6@1000 | v13_tuned + floor=0.05 | Expansion pool without NN; same weights. |
| 2026-04-30 | 0.1473 | BM25@500 | v6 fusion (precursor to v13) |  |
| 2026-04-25 | 0.1418 | BM25@500 | TT-v3 + w=0.7 | First two-tower production. |
| 2026-04-15 | 0.1313 | BM25@500 | BM25 only + tag + seen exclusion | BM25 ceiling. |
