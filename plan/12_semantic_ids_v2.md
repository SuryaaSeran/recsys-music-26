# Plan 12: Semantic IDs v2 — Improvements for Conversational Context

## Current state

**RQ-VAE codebook (runC2_attributes_L2C64):**
- Input: Qwen3-attributes 1024-dim item embeddings (track metadata, item-side only)
- 2 levels, 64 codes per level → 64 L0 buckets, 64×64=4096 leaf slots (2544 used)
- L0 bucket size: mean 728, max 1958, min 59
- SASRec (sasrec_runC2_L2C64) predicts next L0 bucket from track ID sequence history
- Stage 3 expands pool with all tracks in top-3 predicted L0 buckets, capped at 300

**What's working:**
- Stage 3 cap=300 + LTR retrain improved blind proxy from 0.1796 → 0.1867 (+0.0071)
- Diversity is preserved (LexDiv unchanged at 0.5909)

**Key weaknesses in the current approach:**
1. RQ-VAE trained on item-side embeddings only — buckets are semantic in track-metadata space, not in conversation-relevance space
2. SASRec trained on raw listening sequences including rejected tracks — doesn't know about DOES_NOT
3. SASRec input is track IDs → L0 bucket prediction ignores conversation goal, user profile, listener thought
4. Within-bucket selection is random (capped at 300) — no intra-bucket ranking
5. Two zero-importance features (`cand_sem_leaf_match_last`, `sem_bucket_l0_rank`) suggest current LTR features don't exploit semantic IDs well
6. L0 buckets are large (mean 728 tracks) → top-3 buckets ≈ 2184 candidates before cap

---

## Ideas (ordered by EV × effort)

### A — MOVES-only SASRec training sequences [HIGH EV, LOW EFFORT]

**What:** Rebuild SASRec training data using only MOVES_TOWARD_GOAL tracks per session. Skip DOES_NOT tracks from the sequence. Re-train SASRec.

**Why:** SASRec currently learns "what comes after X regardless of preference". With MOVES-only sequences, it learns "what the user actually liked next", which is the signal we want for next-bucket prediction.

**How:** In `build_sasrec_semantic_data.py`, filter each session's track sequence to only include tracks where `goal_progress_assessment = MOVES_TOWARD_GOAL`. Re-run `run_sasrec_talkplay.py`.

**Risk:** MOVES sequences are shorter, reducing training signal. But higher quality. Net positive expected.

---

### B — MOVES-only SASRec INPUT at inference [HIGH EV, ZERO EFFORT]

**What:** At inference time, feed SASRec only the MOVES history tracks (same H1 filter used for the TT). Currently we feed the full `music_history`.

**Where:** `scripts/inference/semantic_id_retrieval.py`, the `expand()` call — pass `pos_hist` instead of `music_history`.

**Why:** Even without retraining, filtering rejected tracks from the SASRec input avoids the model predicting buckets in the neighborhood of bad recommendations. Very simple change.

**Estimated gain:** +0.001–0.003 blind proxy. Can test in 30 minutes.

---

### C — Negative bucket blacklist [MEDIUM EV, LOW EFFORT]

**What:** After SASRec predicts top-k L0 buckets, remove any bucket that contains ≥1 DOES_NOT track from the session history. Only expand into "clean" buckets.

**Why:** If the user rejected a track in L0 bucket 7, expanding bucket 7 is likely to surface more unwanted tracks. The bucket defines a semantic neighborhood — rejection of one track suggests the whole neighborhood is off-target for this turn.

**Where:** `semantic_id_retrieval.py` or the inference loop after Stage 3 expansion. Get the L0 codes of all DOES_NOT tracks in `music_history`, blacklist those codes, skip bucket expansion for them.

---

### D — Rebuild RQ-VAE with v8e (TT) embeddings [MEDIUM EV, MEDIUM EFFORT ~2h]

**What:** After v8e training completes, encode all 47K tracks with the v8e item encoder (`passage: ...` format) and train a new RQ-VAE on those 768-dim embeddings instead of Qwen3-attributes 1024-dim.

**Why:** Qwen3-attributes embeddings are from a general-purpose model. v8e embeddings are fine-tuned on TalkPlay positives/negatives — tracks in the same v8e L0 bucket are genuinely similar in the way TalkPlay users experience them, not just in generic metadata space.

**Config:** L2C64 to match runC2. Expected to outperform runC2 because the embedding space is task-specific.

**After:** Re-run SASRec training on new semantic IDs. Re-dump LTR features with new IDs.

---

### E — Direct query→bucket retrieval using v8e query embedding [HIGH EV, MEDIUM EFFORT]

**What:** At inference time, compute the v8e QUERY embedding (the anchor for this turn) and find its nearest L0 bucket centroid. Expand all tracks from that bucket as Stage 3b candidates.

**Why:** SASRec predicts next bucket from track history — it doesn't see the conversation goal or the current user message. The v8e query embedding encodes goal + history + listener thought directly. Matching query to nearest codebook centroid gives goal-aware bucket prediction without a sequence model.

**How:**
1. After RQ-VAE training, save the L0 codebook centroids (64 vectors × embedding_dim)
2. At inference (after v8e index is built), compute the query embedding, cosine-match to centroids, expand the top-k nearest buckets
3. This runs in parallel with SASRec Stage 3 — merge the two candidate sets

This is essentially a non-autoregressive next-item prediction that uses the full conversational context.

---

### F — Finer L1 bucket expansion with score-based cap [MEDIUM EV, MEDIUM EFFORT]

**What:** Instead of expanding all tracks in top-3 L0 buckets, predict the most likely leaf (L0_code, L1_code) pair using SASRec and expand only tracks in those narrow leaves. Apply cap at leaf level.

**Why:** Current L0 buckets have mean 728 tracks — the cap=300 samples blindly from a very large set. Leaves have median 14 tracks, max 221. A leaf expansion of top-10 leaves would give ~140 tracks with much higher precision.

**Effort:** Requires SASRec to output (L0, L1) joint predictions, not just L0. The current SASRec predicts L0 only. Need to extend to 2-level prediction.

---

### G — Intra-bucket ranking before cap [MEDIUM EV, LOW EFFORT]

**What:** After Stage 3 identifies candidate tracks from a bucket, rank them by cosine similarity to the v8e query embedding before applying the cap. Currently the cap is applied to an arbitrary (likely unsorted) list.

**Why:** With mean bucket size 728 and cap 300, we're keeping ~41% of the bucket. If we rank by v8e query similarity first, the 300 kept are the most query-relevant tracks in the bucket, not random ones.

**Where:** `semantic_id_retrieval.py` or the inference Stage 3 integration. After getting `_s3_cands`, re-rank by TT cosine before `_s3_cap`.

---

### H — Larger codebook for finer granularity [MEDIUM EV, MEDIUM EFFORT]

**What:** Try L2C128 or L3C64 (larger/deeper codebook). L2C128 gives 128×128=16384 leaf slots, much finer than current 4096.

**Why:** Current L0 bucket size (mean 728) is too coarse for recall precision. With 128 codes, mean L0 bucket ≈ 364 tracks (half the size), giving more precise expansion.

**Existing attempt:** `runD_metadata_L2C128` already exists in `cache/semantic_ids/`. Check if it performed better than C2 before investing.

**Config:** `runE_metadata_L3C64` also exists — 3 levels gives even finer granularity.

---

### I — SASRec with goal-conditioned input [LOW-MEDIUM EV, HIGH EFFORT]

**What:** Modify SASRec to accept the goal embedding (encoded by v8e) as a prefix or conditioning vector in addition to track sequence. At each prediction step, the model attends to both track history and the current goal.

**Why:** Pure sequence SASRec is session-agnostic about the goal. If the goal changes mid-session (e.g., pivot from discovery to specific track), SASRec doesn't know. Goal conditioning would make bucket predictions goal-aware.

**Effort:** Requires architectural changes to SASRec. Non-trivial. Deferred unless simpler ideas plateau.

---

### J — Use SASRec prediction confidence as LTR feature [LOW EFFORT, UNCERTAIN EV]

**What:** Add `sasrec_bucket_confidence` (softmax probability of the predicted L0 bucket) as an LTR feature. High-confidence SASRec predictions should be weighted more.

**Why:** The two zero-gain features suggest current semantic ID LTR features are poorly designed. Confidence score might correlate with actual relevance — when SASRec is confident, those candidates should be ranked higher.

---

## Recommended execution order

| Priority | Idea | Effort | When |
|---|---|---|---|
| 1 | B: MOVES-only SASRec input at inference | ~30 min | Now (before v8e eval) |
| 2 | C: Negative bucket blacklist | ~1h | Now |
| 3 | G: Intra-bucket ranking by v8e similarity | ~2h | After v8e index built |
| 4 | D: Rebuild RQ-VAE with v8e embeddings | ~2h train | After v8e index built |
| 5 | A: MOVES-only SASRec training sequences | ~2h train | After new RQ-VAE |
| 6 | E: Direct query→bucket retrieval | ~3h | After v8e + new RQ-VAE |
| 7 | F: Finer L1 expansion | medium | After above |
| 8 | H: Check runD/runE codebooks | 30 min check | Now |

## Quick wins to test before v8e eval

Steps B and C can be implemented today on the current v8d+Stage3 stack and tested on blind proxy without waiting for v8e. If they help, add to v20 pipeline and submit as v20b.
