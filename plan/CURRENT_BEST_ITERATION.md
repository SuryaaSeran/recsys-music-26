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

## Previous bests

| Date | nDCG@20 | Pool | Rescore | Note |
|---|---|---|---|---|
| 2026-05-05 | 0.1519 | BM25@500 only | v13_tuned weights | Prior best. Blind file: `blind_a_fusion_v13_tuned_qwen.json`. |
| 2026-05-04 | 0.1518 | BM25@500 + artist + TT-v6@1000 | v13_tuned + floor=0.05 | Expansion pool without NN; same weights. |
| 2026-04-30 | 0.1473 | BM25@500 | v6 fusion (precursor to v13) |  |
| 2026-04-25 | 0.1418 | BM25@500 | TT-v3 + w=0.7 | First two-tower production. |
| 2026-04-15 | 0.1313 | BM25@500 | BM25 only + tag + seen exclusion | BM25 ceiling. |
