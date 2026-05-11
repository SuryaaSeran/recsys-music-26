# Current Best Iteration

Live snapshot. Update only when full 1000-session dev nDCG@20 strictly beats this.

## Best (as of 2026-05-11, LTR v2)

- **Dev nDCG@20: 0.1601**
- **Hit@20: 34.0%** (2719 / 8000 turns)
- nDCG@1 0.0525  |  nDCG@10 0.1373  |  catalog_div 0.568  |  lex_div 0.203
- Script: `scripts/inference/run_inference_fusion_recall_expansion.py --ltr_model models/ltr/ltr_v2_train.txt`
- Run id: `ltr_v2_dev_eval` (`exp/inference/devset/ltr_v2_dev_eval.json`)
- Booster: `models/ltr/ltr_v2_train.txt` (LightGBM LambdaMART, 44 trees, 15
  features) trained on 2000 random TRAIN-split sessions
  (`--shuffle_seed 42`, ~17.7M candidate rows). 5-fold CV ndcg@20 on train =
  0.4063 (std 0.0094). No dev-set leakage.
- One-line reason it beat prior best: a tree ranker on the same 15 features
  picks up source-aware interactions (BM25-origin × bm25_signal,
  cold_user × cf_cos, multi-source × cosine combos) that the linear fusion
  cannot express.

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

### Rescore (LTR booster, replaces linear sum)

For each candidate the inference script builds the 15-feature vector
documented in `FEATURE_COLS` (in `run_inference_fusion_recall_expansion.py`).
The booster outputs a single relevance score per candidate; top-20 by that
score is the prediction.

Top feature gains (training data):
```
tt_cos          ████████████████████
tt_rank_sig     ████████████
artist_sig      ██████████
bm25_signal     ████████
nn_sig          ██
qm_cos, cf_cos, pool_size, artist_origin, ql_cos  (small)
```

Linear baseline kept as a fallback path in the same script if `--ltr_model`
is not passed:

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

Reproduction (LTR best):

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
python scripts/inference/run_inference_fusion_recall_expansion.py \
  --tid current_best \
  --tt_model models/twotower_v6/final --tt_index cache/twotower_v6 \
  --tt_pool 1000 --artist_expansion --last_nn_k 100 --last_nn_src 2 \
  --bm25_missing_floor 0.05 \
  --ltr_model models/ltr/ltr_v2_train.txt
```

Booster retraining:

```bash
# 1) feature dump from a sample of TRAIN sessions
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
python scripts/inference/run_inference_fusion_recall_expansion.py \
  --tid v07_train_features --split train --sessions 2000 --shuffle_seed 42 \
  --tt_model models/twotower_v6/final --tt_index cache/twotower_v6 \
  --tt_pool 1000 --artist_expansion --last_nn_k 100 --last_nn_src 2 \
  --bm25_missing_floor 0.05 \
  --write_features exp/analysis/ltr_train_features.npz

# 2) train LightGBM LambdaMART
python scripts/train/train_ltr_lightgbm.py \
  --features exp/analysis/ltr_train_features.npz \
  --out models/ltr/ltr_v2_train.txt \
  --n_folds 5 --num_leaves 63 --lr 0.05 --num_iter 1000 --early_stop 50
```

### Tested on Blind A

Not yet re-run with this config. Next blind submission should regenerate using
the same flags above against `talkpl-ai/TalkPlayData-Challenge-Blind-A` via
`run_inference_blind_fusion.py` (needs --last_nn_k and --artist_expansion ported).

## Evaluation standard

The official evaluator is at `music-crs-evaluator/` (mirrored in this repo).
Numbers below are produced by that evaluator. Our local
`scripts/inference/evaluate_local.py` mirrors it (per-turn-number macro-mean,
no-duplicate check, plus catalog/lexical diversity). Reproduce ours with:

```bash
python scripts/inference/evaluate_local.py --pred exp/inference/devset/<tid>.json
```

## Organizer baselines (official scores, devset 1000 sessions)

Source: `music-crs-evaluator/exp/scores/devset/{random,popularity,llama1b_bm25_devset}.json`.

| Baseline | nDCG@1 | nDCG@10 | nDCG@20 | Catalog div. | Lexical div. |
|---|---:|---:|---:|---:|---:|
| Random              | 0.0000 | 0.0001 | 0.0001 | 0.9652 | 0.0000 |
| Popularity          | 0.0005 | 0.0018 | 0.0024 | 0.0004 | 0.0000 |
| LLaMA-1B + BM25     | 0.0098 | 0.0627 | 0.0815 | 0.3795 | 0.2558 |

LLaMA-1B + BM25 is the organizer's reference retrieval baseline.

## Current system vs baselines (official metrics)

| System | nDCG@1 | nDCG@10 | nDCG@20 | Catalog div. | Lexical div. | Hit@20 |
|---|---:|---:|---:|---:|---:|---:|
| Random                                                               | 0.0000 | 0.0001 | 0.0001 | 0.9652 | 0.0000 | 0.1% |
| Popularity                                                           | 0.0005 | 0.0018 | 0.0024 | 0.0004 | 0.0000 | 0.6% |
| LLaMA-1B + BM25 (organizer)                                          | 0.0098 | 0.0627 | 0.0815 | 0.3795 | 0.2558 | — |
| Our BM25 floor (name+artist+album)                                   | —      | —      | 0.0861 | —      | —      | 21.9% |
| Our BM25 + tag_list + seen exclusion                                 | —      | —      | 0.1313 | —      | —      | 27.4% |
| Our TT-v3 fusion (pool=500, w=0.7)                                   | —      | —      | 0.1418 | —      | —      | 29.8% |
| Our v6 fusion v13_tuned (BM25@500 only)                              | —      | —      | 0.1519 | —      | —      | 31.5% |
| Our v6 fusion + expansion (artist + TT@1000 + NN@100), v13 wts       | 0.0551 | 0.1328 | 0.1533 | 0.5119 | 0.1844 | 31.7% |
| **Our v6 fusion + expansion + LTR LambdaMART (train-only)**          | **0.0525** | **0.1373** | **0.1601** | **0.5677** | **0.2026** | **34.0%** |

Notes:
- Our current best (0.1533 nDCG@20) is **+0.0718 over the strongest organizer
  baseline** (LLaMA-1B + BM25, 0.0815) — roughly 88% relative improvement.
- Catalog diversity 0.512 (we recommend ~51% of the 47,071-track catalog overall)
  vs LLaMA-1B + BM25 at 0.380. Higher coverage is better here.
- Lexical diversity 0.184 vs LLaMA-1B + BM25 at 0.256. Our template responses
  are less varied; addressing this is a response-generation problem, not
  retrieval, and is out of scope for the current phase.

## Response generation

Two paths are used depending on the run target.

### Dev (inside the inference loop)

`run_inference_fusion_recall_expansion.py` emits a fixed template per turn:

```python
response = f'I recommend "{name}" by {artist} based on your request.'
```

`name` and `artist` are taken from the top-1 predicted track's metadata. This
is intentionally cheap; the dev evaluator does not score response quality.
Lexical diversity (Distinct-2) on this template scores ~0.18 on the official
evaluator -- enough to confirm the path works end-to-end, but well below the
LLaMA-1B + BM25 baseline at 0.256.

### Blind (post-process step)

For blind submissions, `scripts/inference/generate_responses_blind.py` rewrites
each turn's response with an LLM. Pipeline:

1. Load the prediction JSON (template responses from the dev path).
2. Load Qwen via `mlx-lm` from `models/qwen_sid_patched` (local copy of a
   Qwen-family instruct checkpoint).
3. For each turn, build a chat prompt with:
   - System: "You are a friendly music recommendation assistant. Give a brief
     (2-3 sentence) recommendation that references the user's request and
     explains why the top track fits."
   - Last 4 turns of conversation history. Played-music turns are converted to
     `assistant: I recommend "<name>" by <artist>.` so the model sees what
     was already recommended.
   - Current user query.
   - A trailing user turn that lists the top-3 recommended tracks (name,
     artist, first 5 tags) and asks for a short response about the top track.
4. Generate with `max_tokens=120` on the patched Qwen.
5. On empty output or generation exception, fall back to the dev template.

This step replaces only `predicted_response`; `predicted_track_ids` is
preserved exactly from the retrieval step, so nDCG is unchanged.

Reproduction:

```bash
python scripts/inference/generate_responses_blind.py \
    --pred exp/inference/blind_a/<blind_id>.json \
    --max_tokens 120 \
    --dataset talkpl-ai/TalkPlayData-Challenge-Blind-A
# writes <blind_id>_qwen.json next to the input.
```

Limitations:

- Devset is never scored on response quality locally, so the LLM rewrite is
  not applied there in the standard flow. To measure lexical/personalisation
  judge scores on the devset, the same script can be pointed at a devset
  prediction file via `--dataset talkpl-ai/TalkPlayData-Challenge-Dataset
  --split test` (with the dev-side `session_map`).
- The patched checkpoint at `models/qwen_sid_patched` is not in this repo
  (gitignored); regenerate from the base Qwen-Instruct if missing.
- No LLM-as-Judge tuning has been done yet; raising the lexical/diversity
  numbers is a separate workstream that does not interact with retrieval.

## Previous bests

| Date | nDCG@20 | Pool | Rescore | Note |
|---|---|---|---|---|
| 2026-05-11 | 0.1533 | BM25@500 + artist + TT@1000 + NN@100 | v13_tuned linear | Pool expansion (NN@100) over BM25-only pool, same weights. |
| 2026-05-05 | 0.1519 | BM25@500 only | v13_tuned weights | Prior best. Blind file: `blind_a_fusion_v13_tuned_qwen.json`. |
| 2026-05-04 | 0.1518 | BM25@500 + artist + TT-v6@1000 | v13_tuned + floor=0.05 | Expansion pool without NN; same weights. |
| 2026-04-30 | 0.1473 | BM25@500 | v6 fusion (precursor to v13) |  |
| 2026-04-25 | 0.1418 | BM25@500 | TT-v3 + w=0.7 | First two-tower production. |
| 2026-04-15 | 0.1313 | BM25@500 | BM25 only + tag + seen exclusion | BM25 ceiling. |
