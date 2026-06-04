# Plan: Per-modality RQ-VAE semantic-ID codebooks (Stage 1)

## Context
Stage 1 only: turn each of the 47071 catalog tracks into a SEMANTIC ID by quantizing
its precomputed embeddings, following EdoardoBotta/RQ-VAE-Recommender. A separate
codebook per embedding modality (late fusion), trained independently; per-modality
code tuples concatenated into one semantic ID per track. Stage 2 (generative-retrieval
decoder, or IDs-as-features in a ranker) is later; the ID format is locked now so it
stays decoder-ready.

Decisions: tiered depth (cf-bpr L=4, others L=3, K=256); 5 modalities (drop cover
image / image-siglip2); learned RQ-VAE; concatenate -> one ID + dedup counter +
collision/utilization report.

## ID shape (locked)
`ID_MODALITY_ORDER`: cf-bpr (L=4), audio-laion_clap (L=3), attributes-qwen3 (L=3),
lyrics-qwen3 (L=3), metadata-qwen3 (L=3). Total = 16 code positions + 1 dedup = 17.
`MISSING_CODE = -1` for absent modality positions. Presence/47071: cf-bpr 46455
(616 absent), clap/attributes/lyrics/metadata 46579 (492 absent each).

## Files
- src/rqvae/config.py   ModalityConfig, ID_MODALITY_ORDER, MISSING_CODE, default_configs()
- src/rqvae/cache.py    per-modality float16 cache (L2-norm, optional PCA off by default)
- src/rqvae/model.py    RqVae, Quantize, MLP, kmeans init, STE, commitment, utilization
- src/rqvae/train.py    train ONE modality branch -> exp/codebooks/<m>/{ckpt,codebook,log}
- src/rqvae/encode.py   assign codes, concatenate, dedup, collision/utilization report
- scripts/build_cache.py / build_codebooks.py / export_ids.py
Inputs (read-only): src/tracks.py (load_track_embeddings, EMB_DIMS, load_catalog).
Outputs (gitignored): data/cache/, exp/codebooks/, exp/ids/.

## Pipeline
1. build_cache: read parquet once via src.tracks, L2-normalize present rows, write
   data/cache/{track_ids.json, <m>.f16.npy, <m>.present.npy, norm_stats.json}.
2. build_codebooks: train one RqVae per modality (independent, parallel-friendly).
3. export_ids: encode all rows per modality, concatenate in fixed order, -1 for
   absent, dedup counter for collisions; write exp/ids/{per_modality_codes.npy,
   semantic_ids.json, codes_to_tracks.json, report.json}.

## Model (faithful to the reference repo)
- MLP encoder [input]+hidden->embed, ReLU between layers; decoder mirrors it.
- Residual quantization: res=encoder(x); per layer pick nearest code by L2, STE
  q=z+(e-z).detach(), res-=q, accumulate commitment ||sg(z)-e||^2+beta||z-sg(e)||^2.
- Loss = recon_mse + sum(commitment). AdamW lr 1e-4. kmeans codebook init on first
  batch (sklearn KMeans on encoder-output distribution, CPU, subsample 20k).
- Device auto: mps if available else cpu; train float32 on device (cache is float16
  on disk only).

## Verification (export_ids --verify -> report.json)
- Per-branch/layer utilization; expect L1 util > 0.8, flag any layer < 0.3.
- Collision histogram of full-tuple buckets; expect ~100% size-1; report max bucket
  and count needing dedup>0.
- Per-modality L1 coherence: sample track + bucket-mates via Track.text() (should be
  similar artist/genre/era).
- Missing-modality audit: confirm 616/492/492/492/492 carry MISSING_CODE.

## Success checks
Cache absent counts match; recon mse plateaus and L1 util > 0.8 per modality; all
47071 tracks exported with full length 16, absent = -1, ~100% size-1 buckets, every
collision resolved by dedup; re-export reproduces identical IDs.

## Risks
MPS float16 flakiness -> float32 on device, CPU fallback. Dead codes in deep residual
layers expected; flag only if L1 low. Stage-2 contract (order, MISSING_CODE, dedup
position) locked now; changing later invalidates exported IDs.
