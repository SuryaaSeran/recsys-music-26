# C2 Semantic ID Artifacts (canonical branch)

## Decision

`attributes-qwen3_embedding_0.6b`, 2 levels × 64 codes (Run C2) is the chosen
semantic tokenizer. Beats metadata-qwen3 (Run A) on tag purity (0.154 vs 0.085,
+80%) and decade purity (0.517 vs 0.435) while retaining strong session
coherence (45.6× random baseline for consecutive track pairs, vs 67× for Run A).

## Artifacts

| Artifact | Path |
|---|---|
| Parquet (attributes-qwen3 embeddings) | `third_party/semantic-ids-llm/data/output/TalkPlay_items_attributes_qwen3.parquet` |
| RQ-VAE checkpoint (final) | `models/rqvae/runC2_attributes_L2C64/final_model.pth` |
| Semantic ID assignments | `cache/semantic_ids/runC2_attributes_L2C64/semantic_ids.npy` |
| Track ID index | `cache/semantic_ids/runC2_attributes_L2C64/track_ids.npy` |
| Codebook meta | `cache/semantic_ids/runC2_attributes_L2C64/meta.json` |
| SASRec checkpoint (best, NDCG@10=0.6745) | `models/sasrec/sasrec_runC2_L2C64/best_model.pth` |
| SASRec training sequences (train) | `third_party/semantic-ids-llm/data/output/TalkPlay_sequences_with_semantic_ids_train.parquet` |
| SASRec training sequences (eval) | `third_party/semantic-ids-llm/data/output/TalkPlay_sequences_with_semantic_ids_eval.parquet` |
| LTR feature dump (no Stage 3) | `exp/analysis/ltr_v8d_tier1_semC2_6k_features.npz` |
| LTR booster (C2, no Stage 3) | `models/ltr/ltr_v8d_tier1_semC2_nl31_lr0p08.txt` (pending) |

## Config

```
RQ-VAE:
  embedding: attributes-qwen3_embedding_0.6b (1024-dim)
  codebook_quantization_levels: 2
  codebook_size: 64
  codebook_embedding_dim: 32
  encoder_hidden_dims: [512, 256, 128]
  use_kmeans_init: True
  reset_unused_codes: True
  epochs: 400, batch: 8192, max_lr: 3e-4, warmup: 200 steps

SASRec:
  num_levels: 2, codebook_size: 64, vocab_size: 128
  max_seq_length: 8 (all TalkPlay sessions = 8 music turns)
  num_blocks: 2, num_heads: 4, head_dim: 64 (hidden=256)
  epochs: 50, batch: 256, max_lr: 1e-3
  training data: 15,199 sessions, 91,194 training samples
  val NDCG@10: 0.6745, HR@10: 0.8869

Feature dump:
  --semantic_ids_dir cache/semantic_ids/runC2_attributes_L2C64
  4 features: cand_sem_l0_match_last, cand_sem_leaf_match_last,
              cand_sem_l0_match_count, cand_sem_l0_match_moves
```

## Codebook quality (Run C2 vs Run A)

| Metric | Run A (metadata) | **Run C2 (attributes)** |
|---|---:|---:|
| Tag purity L0 | 0.085 | **0.154** |
| Artist purity L0 | 0.025 | 0.022 |
| Decade purity L0 | 0.435 | **0.517** |
| Session coherence (leaf, consec pairs) | 1.64% (67×) | **1.11% (45.6×)** |
| Leaf bucket max size | 1144 | **221** |
| Leaf bucket median | 7 | **14** |
| L0/L1 code usage | 100%/100% | 100%/100% |

## Stage 3 recall (C2 SASRec, top-3 L0 expansion)

Over 200 dev sessions (1400 turns with known gold IDs):
- Gold in predicted top-3 L0 pool: **27.79%** (5.9× random baseline)
- Average pool size: 3,484 candidates per turn

## Stage 3 training requirement

Stage 3 pool expansion MUST be included in the LTR feature dump for the ranker
to rank expanded candidates correctly. Dumping without `--sasrec_ckpt` and then
adding Stage 3 at eval time drops nDCG (0.1864 → 0.1827) because the LTR has
never seen semantic-bucket candidates during training.

Re-dump command (Stage 3 enabled):
```bash
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
  --sasrec_ckpt models/sasrec/sasrec_runC2_L2C64/best_model.pth \
  --sasrec_top_k_l0 3 \
  --write_features exp/analysis/ltr_v8d_tier1_semC2_stage3_6k_features.npz
```
