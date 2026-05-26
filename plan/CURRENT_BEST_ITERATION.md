# Current Best Iteration

Live snapshot. Update only when full 1000-session dev nDCG@20 strictly beats this.

## Best (as of 2026-05-24, Phase A LTR retrained)

- **Dev nDCG@20: 0.1646**
- **Hit@20: (not counted)**
- nDCG@1 0.0569  |  nDCG@10 0.1423  |  catalog_div 0.5631  |  lex_div 0.2168
- Script: `scripts/inference/run_inference_fusion_recall_expansion.py` with Phase A expansion flags
- Run id: `phase_a_ltr_retrained` (`exp/inference/devset/phase_a_ltr_retrained.json`)
- Booster: `models/ltr/ltr_phase_a_nl31_lr0p08.txt` (LightGBM LambdaMART, 73
  trees, 27 features, num_leaves=31, lr=0.08) trained on 2000 random
  TRAIN-split sessions (`--shuffle_seed 42`). 5-fold CV ndcg@20 on train =
  0.3987 (std 0.0052). No dev-set leakage. Co-occurrence table uses
  `next_song_leakfree.npz` (excludes the same 2000 LTR training sessions).
- One-line reason it beat prior best: Phase A expansion (qwen_pool=500, cf_pool=200,
  session_mean_k=100, cooccur_ks=300/150/50) pushed pool recall from 80.8% to 83.0%,
  and retraining the LTR on the expanded pool's 27-feature vectors gave the booster
  source-awareness for the new candidate types.

### Retrieval pool

```
BM25@500
+ artist expansion (popularity-sorted catalog, --artist_cap 50)
+ TT-v6@1000
+ last-track-NN@100 in TT space (last_nn_src=2)
+ Qwen-meta global top-500
+ CF global top-200 (warm users only)
+ session-mean-vector NN top-100
+ co-occurrence top-300/150/50 (last 3 played tracks, leakfree table)
```

Mean deduped pool size: ~2550 (vs ~1450 before Phase A).
Pool recall: 0.8303 (vs 0.808 before).

### Rescore (LTR booster, replaces linear sum)

27-feature LambdaMART booster. Top feature gains (training data):
```
tt_rank_sig          323159
tt_cos               184164
cf_cos                91930
nn_sig                53752
bm25_signal           36987
dist_to_recent_mean   35394
artist_sig            29710
dist_to_last          18397
collab_rank_sig       13942
mean_nn_rank_sig       8991
```

Reproduction (current best):

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
python scripts/inference/run_inference_fusion_recall_expansion.py \
  --tid current_best \
  --tt_model models/twotower_v6/final --tt_index cache/twotower_v6 \
  --tt_pool 1000 --artist_expansion --last_nn_k 100 --last_nn_src 2 \
  --bm25_missing_floor 0.05 \
  --qwen_pool 500 --cf_pool 200 --session_mean_k 100 \
  --cooccur_table cache/cooccur/next_song_leakfree.npz --cooccur_ks 300,150,50 \
  --ltr_model models/ltr/ltr_phase_a_nl31_lr0p08.txt
```

Booster retraining:

```bash
# 1) feature dump from TRAIN sessions
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
python scripts/inference/run_inference_fusion_recall_expansion.py \
  --tid phase_a_train_features --split train --sessions 2000 --shuffle_seed 42 \
  --tt_model models/twotower_v6/final --tt_index cache/twotower_v6 \
  --tt_pool 1000 --artist_expansion --last_nn_k 100 --last_nn_src 2 \
  --bm25_missing_floor 0.05 \
  --qwen_pool 500 --cf_pool 200 --session_mean_k 100 \
  --cooccur_table cache/cooccur/next_song_leakfree.npz --cooccur_ks 300,150,50 \
  --write_features exp/analysis/ltr_phase_a_train_features.npz

# 2) train LightGBM LambdaMART
python scripts/train/train_ltr_lightgbm.py \
  --features exp/analysis/ltr_phase_a_train_features.npz \
  --out models/ltr/ltr_phase_a_nl31_lr0p08.txt \
  --n_folds 5 --num_leaves 31 --lr 0.08 --num_iter 1000 --early_stop 50
```

### Tested on Blind A

Not yet re-run with this config. Blind script needs Phase A expansion flags ported
(currently only has `--last_nn_k` and `--artist_expansion`; needs `--qwen_pool`,
`--cf_pool`, `--session_mean_k`, `--cooccur_table`, `--cooccur_ks`).

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
| Our v6 fusion + expansion + LTR LambdaMART v2 (nl63 lr0.05)          | 0.0525 | 0.1373 | 0.1601 | 0.5677 | 0.2026 | 34.0% |
| **Our v6 fusion + expansion + LTR LambdaMART v3 (nl31 lr0.08)**      | **0.0534** | **0.1377** | **0.1609** | **0.5645** | **0.2030** | **—** |

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
| 2026-05-24 | 0.1609 | BM25@500 + artist + TT@1000 + NN@100 | LTR v3 (nl31 lr0.08, 73 trees, 15 feat) | `models/ltr/sweep/ltr_nl31_lr0p08.txt`. |
| 2026-05-15 | 0.1601 | BM25@500 + artist + TT@1000 + NN@100 | LTR v2 (nl63 lr0.05, 44 trees) | LambdaMART on TRAIN features; `models/ltr/ltr_v2_train.txt`. |
| 2026-05-11 | 0.1533 | BM25@500 + artist + TT@1000 + NN@100 | v13_tuned linear | Pool expansion (NN@100) over BM25-only pool, same weights. |
| 2026-05-05 | 0.1519 | BM25@500 only | v13_tuned weights | Prior best. Blind file: `blind_a_fusion_v13_tuned_qwen.json`. |
| 2026-05-04 | 0.1518 | BM25@500 + artist + TT-v6@1000 | v13_tuned + floor=0.05 | Expansion pool without NN; same weights. |
| 2026-04-30 | 0.1473 | BM25@500 | v6 fusion (precursor to v13) |  |
| 2026-04-25 | 0.1418 | BM25@500 | TT-v3 + w=0.7 | First two-tower production. |
| 2026-04-15 | 0.1313 | BM25@500 | BM25 only + tag + seen exclusion | BM25 ceiling. |
