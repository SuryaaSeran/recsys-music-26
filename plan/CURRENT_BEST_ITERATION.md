# Current Best Iteration

Live snapshot. Update only when full 1000-session dev nDCG@20 strictly beats this.

## Blind A best submission: v19 (blind A nDCG@20 = 0.3997, composite 0.3370)

v19 = v18 retrieval + Stage 3 SASRec semantic-bucket expansion (sasrec_runC2_L2C64, cap=300).
All metrics improved over v18: nDCG +0.0007, judge 1.9→2.0, composite +0.0079. LexDiv unchanged (0.5909).

v19 prediction file: `exp/inference/blind_a/blind_a_v8d_tier1_s3cap300_v19.json`
v19 submission zip: `exp/inference/blind_a/submissions/v19_v8d_s3cap300/submission.zip`

**Do NOT apply any reranker until CE is retrained on v8d format.**
Zero-shot rerankers collapse diversity (LexDiv 0.5909 → 0.0125 with Qwen3-8B).

---

## Best (as of 2026-06-13, v8d Tier 1 gpa-aware + entity features)

- **Dev nDCG@20: 0.1864** (+0.0116 over prior v8d 0.1748)
- **Blind A nDCG@20: 0.3990** (submitted 2026-06-14, composite 0.3291)
- nDCG@1 0.0716 | nDCG@10 0.1634 | catalog_div 0.4814 | lex_div 0.2045
- Run id: `v8d_tier1_dev1000` (`exp/inference/devset/v8d_tier1_dev1000.json`)

---

## Full Architecture

### 1. Retrieval Pool (7 sources, fused)

Every turn produces a candidate pool from 7 independent sources, then merged:

| Source | Candidates | Notes |
|---|---:|---|
| BM25 (Okapi BM25) | 500 | Query = latest user message + goal text + culture. `--bm25_missing_floor 0.05` smooths missing-vocab turns. |
| TT-v8d (Two-Tower) | 2000 | Fine-tuned anchor → positive cosine similarity. `--last_nn_k 100 --last_nn_src 2` adds 100 nearest neighbours of the last 2 history tracks in TT space. `--artist_expansion` adds full discographies of retrieved artists. |
| Qwen3-Embedding-0.6B | 500 | Pre-computed track embeddings (`metadata-qwen3`). Query encoded at inference by frozen Qwen3-Embedding-0.6B. |
| CF-BPR (collaborative filtering) | 200 | Pre-computed user × item BPR embeddings from competition dataset. Warm users only. |
| Session-mean NN | 100 | Mean of the TT embeddings of all history tracks; NN lookup in TT space. |
| Co-occurrence (leakfree) | 300/150/50 | `cache/cooccur/next_song_leakfree_6k_excluded.npz`. Tracks co-occurring with last 1/2/3 history tracks in the 6K training dump, built with 6K training sessions excluded to avoid pool leakage. Ks: 300 for last track, 150 for second-to-last, 50 for third-to-last. |

**Pool recall: 87.42%** (6994/8000 dev turns — gold found in pool).

### 2. gpa-Aware Inference Modulations (H1 + H3)

Applied before LTR scoring, using `goal_progress_assessment` labels from the conversation:

| Flag | Behaviour |
|---|---|
| `--use_goal_progress` (H1) | Strips tracks with P_{i+1} = DOES_NOT from BM25/TT seed history. Prevents the retrieval from anchoring on explicitly rejected tracks. Also activates H2 features. |
| `--goal_substitute_positive` (H3a) | Substitutes the `[GOAL]` slot in the TT anchor with the text of the most recent MOVES track. Tells the TT "more like the last accepted track." |
| `--rejection_drop_threshold 2` (H3b) | Drops the goal slot entirely after 2 consecutive DOES_NOT turns (the goal text has become misleading). |

### 3. LTR Reranker (LightGBM LambdaMART)

**Model**: `models/ltr/ltr_v8d_tier1_nl31_lr0p08.txt`

**60 features** in 5 groups:

#### Group A — Retrieval scores (core signal)
| Feature | Description |
|---|---|
| `tt_cos` | Two-Tower v8d cosine similarity (anchor ↔ track embedding) |
| `tt_rank_sig` | Sigmoid-normalised TT rank within pool |
| `qm_origin`, `ql_origin` | Qwen3 metadata / lyrics cosine |
| `clap_cos` | LAION CLAP audio cosine (pre-computed, dim 512) |
| `cf_cos` | CF-BPR cosine (user × item, dim 128) |
| `bm25_signal` | BM25 reciprocal rank signal |
| `collab_rank_sig` | Co-occurrence rank signal |

#### Group B — Multi-source agreement (highest gain at 57%)
| Feature | Description |
|---|---|
| `n_sources` | Number of distinct retrieval sources that returned this candidate (0–7) |
| `log1p_n_sources` | log(1 + n_sources) |
| `n_sources_norm` | n_sources / 7 |

#### Group C — Track metadata
| Feature | Description |
|---|---|
| `popularity_pctile` | Popularity percentile within catalog (pre-computed) |
| `yrs_since_release` | 2026 - release_year (float, NaN → imputed) |
| `tag_overlap` | Count of track tags appearing in BM25 query tokens |
| `tag_query_sim` | Fraction of track tags matching latest user message words |

#### Group D — History / gpa features (H2, activated by `--use_goal_progress`)
| Feature | Description |
|---|---|
| `sim_to_pos_hist_mean` | Cosine of candidate TT embedding vs mean of MOVES history track embeddings |
| `sim_to_neg_hist_mean` | Cosine vs mean of DOES_NOT history track embeddings |
| `artist_in_rejected_set` | Binary: candidate's primary artist appeared in a DOES_NOT turn |
| `n_rejected_in_history` | Count of DOES_NOT turns in session so far |
| `turns_toward_goal` | Count of MOVES labels in history |
| `consec_rej` | Count of consecutive DOES_NOT labels at the end of history |
| `cf_dist_last`, `cf_dist_mean` | CF cosine of candidate vs last/mean history track |

#### Group E — Tier 1 new features (2026-06-13)
| Feature | Description |
|---|---|
| `same_album_as_last_history` | Binary: candidate shares album_id with last history track |
| `n_same_album_in_history` | Count of history tracks sharing any album_id with candidate (clipped/normalised) |
| `album_in_recent_window` | Binary: candidate's album appears in last 3 history tracks |
| `q_has_era`, `q_era_year` | Era keyword present in Q_t + goal; extracted center year |
| `q_genre_count`, `q_mood_count`, `q_instrument_count` | Count of genre/mood/instrument keywords in Q_t + goal |
| `cand_genre_match` | Overlap between Q_t genre keywords and candidate's tag_list |
| `cand_era_match` | Binary: candidate release year falls within extracted era range |
| `within_artist_*` | Within-artist popularity/transition features (using artist_id, not string) |
| `cand_sem_l0_match_count` | Count of history tracks sharing candidate's L0 semantic bucket (C2 RQ-VAE) |
| `cand_sem_l0_match_moves` | Count of MOVES history tracks sharing candidate's L0 bucket |

**LTR training configuration:**
- Algorithm: LightGBM LambdaMART (ndcg@20 objective)
- Leaves: 31 | LR: 0.08 | L2: 0.1 | min_sum_hessian: 0.1 | path_smooth: 1.0
- Feature fraction: 0.8 | Bagging fraction: 0.8 | Truncation level: 30
- Folds: 5-fold CV on training groups | Early stopping: 75 rounds | max_iter: 1000
- Mean best iter: 111 | CV nDCG@20: 0.3152 ± 0.0061

---

## Two-Tower Model (TT-v8d)

### Base model
`intfloat/multilingual-e5-base` (12-layer XLM-RoBERTa, 768-dim, 279M params, 512-token max)

### LoRA fine-tuning
- Rank: r=32, alpha=64, dropout=0.05
- Target modules: `query, key, value` (all attention projection matrices)
- Trainable params: ~1.8M (0.64% of total)
- Memory: LoRA optimizer states ~40MB vs ~3GB for full fine-tune

### Anchor format (role-tagged, v8d)

```
query: [PROFILE] {age_group} · {country_code} · {gender} · {culture} · {language}
[GOAL] {listener_goal_text}  ({specificity})
[T1] USER: {Q_1} | REC: {track_name} – {artist} | ASST: {R_1} | REACTION: liked/rejected
[T2] USER: {Q_2} | REC: {track_name} – {artist} | ASST: {R_2} | REACTION: liked/rejected
...
[NOW] USER: {Q_t}
```

Token budget: 510 (E5 max 512 minus 2 special tokens). Greedy history insertion most-recent-first; falls back to shorter formats if budget exceeded. Thought fields excluded (null at eval). Target M_t excluded.

### Positive labeling

Each `[music]` turn t → anchor (turns 1..t-1 history) + positive (M_t text).

Label weight by P_{t+1} (gpa at turn t+1, the listener's response to M_t):
- `MOVES_TOWARD_GOAL` → weight 1.0 (kept)
- `DOES_NOT_MOVE_TOWARD_GOAL` → weight 0.3 (dropped with probability 0.7)
- Missing (last turn, no gpa_9) → weight 1.0

Implementation: probabilistic dropping at data-build time (approximates MNRL weighted loss).

### Positive text format (document)
```
passage: {track_name} by {artist_name} | Album: {album_name} | Tags: {tag1} {tag2}... | {year}
```

### Hard negative mining (per anchor, in priority order)

1. **Confirmed session rejections** (up to 2): tracks from earlier in the same session with P_{i+1} = DOES_NOT. Strongest hard neg — same intent context, explicitly rejected.
2. **BM25 hard negs** (up to 2, HH/LH specificity only): top BM25 results for the anchor, excluding gold and MOVES tracks.
3. **Artist-repeat distractors** (up to 1): tracks by artists already in history (targets artist-repetition failure mode for discovery goals).
4. **In-batch negatives**: all other positives in the batch (free, semi-hard via MNRL).

### MOVES protection (false negative prevention)
- Never use a track that was MOVES in the same session as any negative.
- Specificity gating: LL/HL sessions skip same-session positional negs (many acceptable tracks); HH/LH mine them aggressively (only one correct track).

### Training hyperparameters
- Loss: MultipleNegativesRankingLoss (InfoNCE, in-batch negatives)
- Epochs: 3 | Batch size: 8 per device | Grad accumulation: 4 (effective batch 32)
- LR: 1e-4 | Warmup steps: 200 | Gradient checkpointing: enabled
- Data: TalkPlayData-Challenge-Dataset `train` split, 15,199 sessions
- Train/val split: 95/5 at session level
- Output: `models/twotower_v8d/final` (LoRA merged into base weights)
- Index: `cache/twotower_v8d/` (47,071 tracks × 768-dim, L2-normalised)

---

## LTR Training Data

**Feature dump script**: `scripts/inference/run_inference_fusion_recall_expansion.py --write_features`

**Training sessions**: 6,000 sessions from `train` split, shuffle_seed=42
**Active flags**: `--skip_no_progress --use_goal_progress` (drops turns where gold = DOES_NOT, activates H2 features)
**Groups after filter**: 24,718 (dropped 2,448 all-zero groups = 9.0%)
**Rows**: 77,718,693 | **Positive rate**: 0.00035

Per group (turn):
- 1 positive row (gold track, label=2)
- ~3,100 negative rows (label=0)
- ~0 rows with label=1 (unused in this config)

Feature dump: `exp/analysis/ltr_v8d_tier1_6k_features.npz` (18GB)

---

## Reproduction Commands

### Build TT training data
```bash
python scripts/train/build_twotower_v8d_data.py \
  --out_dir data/twotower_v8d \
  --sessions 15000 --shuffle_seed 42
```

### Train TT-v8d
```bash
python scripts/train/train_twotower_lora.py \
  --data_dir data/twotower_v8d \
  --out_dir models/twotower_v8d \
  --base_model intfloat/multilingual-e5-base \
  --lora_r 32 --lora_alpha 64 --lora_dropout 0.05 \
  --epochs 3 --batch_size 8 --grad_accum 4 \
  --lr 1e-4 --warmup_steps 200 --gradient_checkpointing \
  --use_hard_neg --n_hard_negs 2
```

### Build TT index
```bash
python scripts/train/build_twotower_index.py \
  --model models/twotower_v8d/final \
  --out_dir cache/twotower_v8d \
  --doc_prefix "passage: " --batch_size 32
```

### Feature dump (LTR training data)
```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
python scripts/inference/run_inference_fusion_recall_expansion.py \
  --split train --sessions 6000 --shuffle_seed 42 \
  --tt_model models/twotower_v8d/final --tt_index cache/twotower_v8d \
  --anchor_v8d \
  --tt_pool 2000 --artist_expansion --last_nn_k 100 --last_nn_src 2 \
  --bm25_missing_floor 0.05 \
  --qwen_pool 500 --cf_pool 200 --session_mean_k 100 \
  --cooccur_table cache/cooccur/next_song_leakfree_6k_excluded.npz \
  --cooccur_ks 300,150,50 \
  --skip_no_progress --use_goal_progress \
  --semantic_ids_dir cache/semantic_ids/runC2_attributes_L2C64 \
  --write_features exp/analysis/ltr_v8d_tier1_6k_features.npz
```

### Train LTR
```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
python scripts/train/train_ltr_lightgbm.py \
  --features exp/analysis/ltr_v8d_tier1_6k_features.npz \
  --out models/ltr/ltr_v8d_tier1_nl31_lr0p08.txt \
  --n_folds 5 --num_leaves 31 --lr 0.08 --num_iter 1000 --early_stop 75 \
  --lambda_l2 0.1 --min_sum_hessian 0.1 --path_smooth 1.0 \
  --feature_fraction 0.8 --bagging_fraction 0.8 --truncation_level 30
```

### Dev eval (1000 sessions)
```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
python scripts/inference/run_inference_fusion_recall_expansion.py \
  --tid v8d_tier1_dev1000 \
  --tt_model models/twotower_v8d/final --tt_index cache/twotower_v8d \
  --anchor_v8d \
  --tt_pool 2000 --artist_expansion --last_nn_k 100 --last_nn_src 2 \
  --bm25_missing_floor 0.05 \
  --qwen_pool 500 --cf_pool 200 --session_mean_k 100 \
  --cooccur_table cache/cooccur/next_song_leakfree_6k_excluded.npz \
  --cooccur_ks 300,150,50 \
  --use_goal_progress --goal_substitute_positive --rejection_drop_threshold 2 \
  --semantic_ids_dir cache/semantic_ids/runC2_attributes_L2C64 \
  --ltr_model models/ltr/ltr_v8d_tier1_nl31_lr0p08.txt

python scripts/inference/evaluate_local.py \
  --pred exp/inference/devset/v8d_tier1_dev1000.json
```

### Blind A inference (v18)
```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
python scripts/inference/run_inference_fusion_recall_expansion.py \
  --tid blind_a_v8d_tier1_v18 \
  --dataset talkpl-ai/TalkPlayData-Challenge-Blind-A \
  --blind_mode --out_dir exp/inference/blind_a \
  --tt_model models/twotower_v8d/final --tt_index cache/twotower_v8d \
  --anchor_v8d \
  --tt_pool 2000 --artist_expansion --last_nn_k 100 --last_nn_src 2 \
  --bm25_missing_floor 0.05 \
  --qwen_pool 500 --cf_pool 200 --session_mean_k 100 \
  --cooccur_table cache/cooccur/next_song_leakfree_6k_excluded.npz \
  --cooccur_ks 300,150,50 \
  --use_goal_progress --goal_substitute_positive --rejection_drop_threshold 2 \
  --semantic_ids_dir cache/semantic_ids/runC2_attributes_L2C64 \
  --ltr_model models/ltr/ltr_v8d_tier1_nl31_lr0p08.txt
```

---

## Scores

### Dev (1000 sessions, 8000 turns)
| Metric | Value |
|---|---|
| nDCG@20 | **0.1864** |
| nDCG@10 | 0.1634 |
| nDCG@1 | 0.0716 |
| Hit@20 | 0.400 |
| catalog_diversity | 0.4814 |
| lexical_diversity | 0.2045 |
| Pool recall | 87.42% |

### Blind A — v19 (80 sessions, 80 turns)
| Metric | Value |
|---|---|
| **nDCG@20** | **0.3997** |
| catalog_diversity | 0.0304 |
| lexical_diversity | 0.5909 |
| llm_judge_score | 2.00 / 5 |
| **composite_score** | 0.3370 |

### Blind A — v18 (previous best, for comparison)
| Metric | Value |
|---|---|
| nDCG@20 | 0.3990 |
| lexical_diversity | 0.5909 |
| llm_judge_score | 1.90 / 5 |
| composite_score | 0.3291 |

### BLINDPROXY_MIXED (dev proxy, 992 MOVES turns)
| Metric | Value |
|---|---|
| nDCG@20 (flat mean) | 0.1988 |
| Hit@20 | 0.400 |
| Turn 5 | 0.2123 |
| Turn 6 | 0.2050 |
| Turn 7 | 0.2112 |

---

## Reranker Verdict (2026-06-15)

**Zero-shot Qwen3-8B reranker makes blind A WORSE.**

| Metric | v18 (no rerank) | v18 + Qwen3-8B |
|---|---:|---:|
| blind A nDCG@20 | **0.3990** | 0.3310 (-0.068) |
| lexical_diversity | 0.5909 | 0.0125 (collapsed) |
| llm_judge | 1.90 | 1.15 |
| composite | 0.3291 | 0.1811 |

Root cause: zero-shot Qwen3-8B has no TalkPlay fine-tuning; collapses diversity by picking near-identical tracks across sessions. CE v3 (existing fine-tuned model) is also unusable — trained on `[TURN-N]` format, not v8d anchor format.

**Do NOT use any reranker until one of:**
a) CE retrained on v8d anchor format data (update `build_crossencoder_v3_data.py`, ~6h retrain)
b) Qwen3-8B fine-tuned on TalkPlay (GPU needed)

---

## Previous Bests

| Date | Dev nDCG@20 | Blind A nDCG@20 | TT model | LTR | Notes |
|---|---|---|---|---|---|
| 2026-06-15 | 0.1854 | **0.3997** | TT v8d (r=32 LoRA) | 60-feat tier1 | ← current (v19, +Stage3 cap=300) |
| 2026-06-13 | **0.1864** | 0.3990 | TT v8d (r=32 LoRA) | 60-feat tier1 | v18, best dev |
| 2026-06-07 | 0.1748 | 0.3182 | TT v8d | 50-feat ltr_v8d | |
| 2026-06-06 | 0.1729 | — | TT v8c (r=32) | 50-feat ltr_v8c_fixed | |
| 2026-05-31 | 0.1615 | 0.3701 | TT v8b | 42-feat H1+H3 | |
| 2026-05-29 | 0.1684 | — | TT v6 | 39-feat LTR Phase D | |
| 2026-05-28 | 0.1653 | — | TT v6 | 29-feat Phase B reg | |
