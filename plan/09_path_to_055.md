# Plan: Path to 0.55 Blind nDCG@20

## Goal

Reach 0.55 nDCG@20 on Blind A (from current 0.37). Deadline: June 30, 2026.

## Gap Analysis

| Metric | Current | Target | Multiplier |
|---|---|---|---|
| Blind nDCG@20 | 0.37 | 0.55 | 1.49x |
| Dev nDCG@20 | 0.1653 | ~0.25+ | 1.51x |
| Pool recall | 83.03% | 90%+ | needed for ceiling room |

The blind score (0.37) is higher than dev (0.1653) because blind evaluation methodology
differs -- likely fewer turns per session, different macro-averaging, or different gold
track difficulty distribution. Regardless, the 0.37 -> 0.55 gap requires a 49% relative
improvement that incremental feature tuning cannot deliver.

### Where points are lost today

1. **17% of gold tracks never enter the pool.** No reranker can fix this. 5% are truly
   unreachable (no signal's top-5000); 12% are "soft" (in some signal's top-5000 but
   not in our pool config).
2. **Of the 83% reachable, reranking converts only ~45% into top-20 hits.** The LTR
   booster sees 29 features but they are all derived from 6 independent retrieval signals
   with no cross-attention or dialogue awareness.
3. **BM25 score signal dominates reranking.** A track at BM25 rank 1 scores higher than
   a semantically superior track at BM25 rank 50 because the linear fusion and even LTR
   cannot fully overcome the BM25 position bias.
4. **Query encoding throws away information.** TT query is ~100 tokens (compressed goal +
   culture + last 2 tracks). The full conversation (300-500 tokens) is available but
   unused. User mood shifts, contradictions, and refinements across turns are invisible.
5. **Response generation scores zero.** Template responses (`I recommend "X" by Y...`)
   yield Distinct-2 = 0.2073, BELOW the organizer baseline (0.2558). LLM-as-Judge is
   unmeasured but certainly poor.

---

## Architecture: Current vs Target

### Current (0.37 blind)
```
6 independent retrievers (BM25, TT, Qwen, CF, CLAP, cooccur)
    -> union pool (~2550 candidates)
    -> 29-feature LambdaMART reranker
    -> top-20
    -> template response
```

### Target (0.55 blind)
```
8+ retrieval sources (add lyrics pool, dual BM25, expanded NN)
    -> union pool (~3000-3500 candidates, 90%+ recall)
    -> rich feature set (39+ features, dialogue-aware)
    -> cross-encoder reranker on top-100 (deep query-track interaction)
    -> top-20
    -> LLM response generation (grounded, diverse, personalized)
```

---

## Major Improvement Areas (6 areas, ordered by expected impact)

### Area 1: Cross-Encoder Reranker (expected +0.04 to +0.06 blind nDCG)

**Why:** The LTR booster sees 29 hand-crafted features derived from independent retrieval
signals. It cannot learn interactions between query terms and track metadata that a
cross-encoder can. Previous CE attempt failed because of wrong training methodology
(binary BCE loss on easy negatives, not LambdaRank on hard negatives).

**What failed before (Phase B addon):**
- bge-reranker-v2-m3 fine-tuned with binary loss on BM25 top-5 negatives
- Score collapse (all pairs scored similarly)
- Resulted in 0.1228 dev, large regression

**What to do differently:**
1. Use a pre-trained cross-encoder (cross-encoder/ms-marco-MiniLM-L-12-v2) as
   a SCORING signal, not as a replacement. Add CE score as a new LTR feature.
2. Rerank only the LTR top-100 (not full pool). CE is expensive but top-100 is
   tractable (~2 seconds per turn on CPU).
3. Feed the CE rich input: `[CLS] {full_user_query} [SEP] {track_name} by {artist} |
   {tags} | {album} | {year} [SEP]`. No truncation needed at 100 candidates.
4. If fine-tuning: use LambdaRank loss with in-pool hard negatives (LTR top-5 that
   are NOT gold), not random/easy negatives.

**Implementation:**
- New script: `scripts/inference/rescore_with_crossencoder_v3.py`
- New feature in FEATURE_COLS: `ce_score` (cross-encoder logit for top-100, 0 for rest)
- Retrain LTR with ce_score as 40th feature

**Risk:** CE inference is slow (~2s per turn on CPU for 100 candidates). With 8000 dev
turns, that is ~4.4 hours. Acceptable for final eval, not for iteration. Use a
200-session pilot for direction signals.

---

### Area 2: Pool Recall to 90% (expected +0.02 to +0.03 blind nDCG)

**Why:** 17% unreachable = 17% guaranteed zeros. Closing even half of the "soft
unreachable" 12% adds ~480 retrievable gold tracks across 8000 turns.

**Concrete sources to add:**

**2a. Qwen-lyrics pool expansion (rescue rate: 6.7% of BM25 misses)**
Currently lyrics embeddings only contribute a scoring signal (ql_cos), never pool
candidates. Add top-200 Qwen-lyrics candidates to pool.
- Expected recall lift: +1.5-2%
- Cost: ~200 extra candidates per turn

**2b. Dual BM25 query (mood-focused)**
Current BM25 query: goal + culture + last 4 tracks + last 4 text turns.
For mood queries, track history dilutes mood keywords. Add a second BM25 retrieval
using only `latest_user_message + goal` (no history). Union top-200 from each.
- Expected recall lift: +1-2% on mood/lyrics buckets
- Cost: one extra BM25 call

**2c. Larger TT pool with v8 embeddings**
TT v8 (512-token context, multilingual-e5-base) should retrieve better than v6
(256-token, all-MiniLM). Once v8 training completes:
- Increase tt_pool from 1000 to 2000 (with v8, not v6)
- v8 should give better recall-per-candidate than v6 at same pool size
- Expected recall lift: +2-3%

**2d. Expand last-track NN depth**
Current: NN of last 2 tracks, k=100 each. Increase to last 3 tracks, k=150 each.
The min-pool recall analysis showed NN is the best recall-per-candidate signal.
- Expected recall lift: +0.5-1%

**Combined target: 83% -> 89-91% pool recall.**

---

### Area 3: Query Enrichment (expected +0.02 to +0.03 blind nDCG)

**Why:** The TT query compresses 300-500 tokens of conversation into ~100 tokens.
Information about mood shifts, user contradictions, and refinements is lost.

**3a. Full-context Qwen query**
Qwen3-Embedding-0.6B supports 8192-token context. Currently we feed it ~100 tokens
(cleaned user msg + goal + culture + 2 track names). Feed it the FULL conversation:
all user turns, all assistant turns, all played tracks with metadata. Let the model's
attention mechanism decide what matters.
- Change: Increase `sem_hist` from 2 to 8, include all text turns
- Expected: better Qwen-meta scores, especially for follow-up queries

**3b. TT v8 rich query (already planned in Phase C)**
v8 encodes up to 512 tokens: latest user + goal + culture + 4 full track texts +
3 prior text turns. This is the single biggest query enrichment.

**3c. Conversation-derived features for LTR**
Extract from the conversation:
- `user_mentioned_artist`: binary, did user name an artist in this turn?
- `user_mentioned_genre`: binary, did user mention a genre word?
- `is_followup`: binary, did user say "more", "another", "similar", "like that"?
- `mood_shift`: binary, did user change mood from previous turn?
These help LTR learn that different turn types need different ranking strategies.

---

### Area 4: Response Generation (expected impact on Distinct-2 and LLM-Judge, not nDCG)

**Why:** The competition scores 4 dimensions. You are currently leaving 2 dimensions
(Distinct-2, LLM-as-Judge) at zero or below baseline. Response generation is completely
orthogonal to retrieval quality -- improving it costs nothing on nDCG while gaining
points on the other dimensions.

**4a. Diverse template rotation (zero model cost)**
Replace single template with 10+ templates that vary by:
- Turn position: "Let's start with..." (turn 1) vs "Following up on..." (turn 3+)
- Query type: "Since you mentioned {artist}..." vs "For that {mood} vibe..."
- Track metadata: "This {year} {genre} track..." vs "From the album {album}..."
Rotate deterministically by session_id + turn_number to ensure reproducibility.
- Expected Distinct-2 lift: 0.2073 -> 0.30+

**4b. LLM response generation on dev (measure what we are missing)**
Apply the same Gemma-3-12b response generation from blind submissions to dev set.
Measure Distinct-2 and estimate LLM-as-Judge quality. This gives us a feedback loop
to optimize response quality during development.

**4c. Retrieval-grounded CoT responses**
Generate responses that reference:
- Why the top track matches the user's stated preferences
- How the track relates to previously played tracks
- What genre/mood connection exists
The "Speak Spotify" paper and README both emphasize that CoT reasoning in responses
significantly lifts LLM-as-Judge scores.

**4d. Joint response + ranking (long-term)**
Train an LLM that produces both track IDs and natural language response in one forward
pass. This is the "Speak Spotify" architecture: semantic IDs + response generation.
Highest potential but also highest effort.

---

### Area 5: Blind-Safe Training (expected +0.02 to +0.04 by preventing regression)

**Why:** Phase B showed that features improving dev can HURT blind. The 0.37 -> 0.30
regression on blind cost more than all dev gains combined. Defensive training is as
important as offensive improvement.

**5a. Leave-one-out session groups**
Instead of training LTR on 2000 sessions and evaluating on a fixed golden-200 holdout,
use session-stratified 5-fold CV and evaluate on the held-out fold. This catches
overfitting to dev-specific patterns.

**5b. Feature stability analysis**
For each feature, compute importance variance across the 5 CV folds. Features with
high variance (importance changes >50% across folds) are overfitting candidates.
Drop them from the blind submission config.

**5c. Dual-config submission strategy**
Maintain two configs:
- "Aggressive": all features, all pools, latest improvements (for dev iteration)
- "Conservative": only structurally robust features, Phase A pool (for blind submission)
Submit the conservative config to blind. Upgrade it only when the aggressive config
has been validated on multiple CV folds AND the feature stability check passes.

**5d. Train on more sessions**
Current LTR trains on 2000 TRAIN sessions. There are 15000 available. Training on
5000-10000 sessions would reduce overfitting to small-sample patterns. The co-occurrence
table would need to be rebuilt excluding the larger training set.

---

### Area 6: Semantic ID Generative Retrieval (highest risk, highest potential)

**Why:** The current architecture has a fundamental ceiling: 6 independent retrievers
produce scores that are combined, but none of them "understands" the conversation.
A generative model that has seen the full dialogue and learned to output track IDs
directly can bypass this limitation.

**Status:** The semantic ID infrastructure exists in the codebase but was NEVER TRAINED.
- `src/quantize/build_semantic_ids.py`: RQ-KMeans codebook builder (2 levels x 256 codes)
- `src/model/music_crs_model.py`: Llama-3.2-1B with 512 new ID tokens + LoRA
- `src/infer/constrained_decoding.py`: Beam search constrained to valid IDs
- `src/train/dataset.py`: SFT dataset builder

**What to do:**
1. Build the codebook on the competition's multimodal track embeddings (CF + Qwen-meta
   + CLAP concatenated, then RQ-KMeans 256x256). This gives each of 47K tracks a 2-token ID.
2. Build SFT training data: for each music turn, the input is the full conversation
   history + user profile + listening history (as ID tokens), and the target is the
   gold track's semantic ID (2 tokens).
3. Fine-tune Llama-3.2-1B (or Qwen-2.5-1.5B) with LoRA on this data. LoRA r=16,
   freeze base, only train new ID token embeddings + LoRA adapters.
4. At inference: constrained beam search over valid 2-token IDs, generate top-20.
5. Use the generative retrieval scores as features in LTR (ensemble with existing system),
   or replace the entire pipeline if it dominates.

**Expected impact:** Text2Tracks paper reports +127% over bi-encoder on similar tasks.
Even a 50% improvement would be transformative here. But training takes 2-3 GPU days
and debugging is complex.

**Risk:** This is the highest-effort, highest-risk item. If it works, it could jump
straight to 0.55+. If it fails (as it did when abandoned earlier), it wastes 3-5 days.

**Mitigation:** Run a 200-session pilot after 1 epoch of training. If nDCG@20 on the
pilot is below baseline BM25 (0.0815), stop. If above 0.10, continue to full training.

---

## Execution Roadmap

### Week 1 (May 29 - June 4): Foundation

| Day | Task | Expected Gain |
|---|---|---|
| 1-2 | Track 1 from 08_plan: train LTR with 39 features + soft_labels + poly_feats | +0.001-0.005 dev nDCG |
| 2-3 | Area 2a+2b: add Qwen-lyrics pool + dual BM25 query. Measure recall. | +2-3% pool recall |
| 3-4 | Area 3a: full-context Qwen query (increase sem_hist to 8). Re-dump features. | +0.001-0.003 dev nDCG |
| 4-5 | Area 4a: diverse template rotation (10+ templates). Measure Distinct-2. | Distinct-2: 0.20 -> 0.30 |
| 5 | Integrate TT v8 if training complete (Phase C). Build index, quick eval. | +0.001-0.010 dev nDCG |

**Week 1 checkpoint:** Dev nDCG > 0.17, pool recall > 86%, Distinct-2 > 0.28.

### Week 2 (June 5 - June 11): Cross-Encoder + Semantic IDs

| Day | Task | Expected Gain |
|---|---|---|
| 1-2 | Area 1: pre-trained cross-encoder on LTR top-100 as new feature | +0.02-0.04 dev nDCG |
| 2-3 | Area 6: build semantic ID codebook + SFT data | (preparation) |
| 3-4 | Area 6: train semantic ID model (1-2 epochs, pilot on 200 sessions) | pilot signal |
| 4-5 | Area 5: blind-safe training (5-fold CV, feature stability, conservative config) | prevent regression |
| 5 | Area 4b: LLM response generation on dev, measure Distinct-2 + quality | Distinct-2: 0.30 -> 0.40 |

**Week 2 checkpoint:** Dev nDCG > 0.20, blind estimate > 0.42 (extrapolated from dev).

### Week 3 (June 12 - June 18): Integration + Semantic ID Full Training

| Day | Task | Expected Gain |
|---|---|---|
| 1-2 | If semantic ID pilot passed: full training (15K sessions, 2 epochs) | +0.03-0.05 dev nDCG |
| 2-3 | Ensemble: semantic ID scores as LTR feature + existing pipeline | combine strengths |
| 3-4 | Area 4c: CoT response generation (grounded in retrieval results) | LLM-Judge lift |
| 4-5 | Full system integration: best retrieval + best LTR + best responses | end-to-end eval |

**Week 3 checkpoint:** Dev nDCG > 0.22, Distinct-2 > 0.35.

### Week 4 (June 19 - June 25): Polish + Blind Submission

| Day | Task | Expected Gain |
|---|---|---|
| 1-2 | Hyperparameter sweep: LTR regularization, CE top-K, pool sizes | fine-tuning |
| 2-3 | Blind-safe config selection (conservative vs aggressive) | risk management |
| 3-4 | Final blind evaluation + submission | target: 0.55 |
| 4-5 | Buffer for debugging, reruns, response quality polish | safety margin |

---

## Risk Assessment

| Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|
| Cross-encoder training fails again | Medium | High | Use pre-trained CE as-is (no fine-tuning), just add score as LTR feature |
| Semantic ID model does not converge | Medium | Very High | Early pilot (200 sessions, 1 epoch). Stop if below BM25 baseline. |
| New features hurt blind (Phase B repeat) | Medium | High | Conservative config for blind. Feature stability check. |
| TT v8 does not improve over v6 | Low | Medium | v6 is a known-good fallback. |
| Response generation hurts nDCG somehow | Low | Low | Response is post-hoc, does not affect track_ids. |
| Time runs out before semantic IDs | High | Medium | Semantic IDs are additive (ensemble). Rest of system improves regardless. |

## nDCG Budget (how we get to 0.55)

Starting from 0.37 blind:

| Improvement | Estimated Lift | Running Total |
|---|---|---|
| LTR feature engineering (39 features, soft_labels) | +0.01 | 0.38 |
| Pool recall 83% -> 90% (lyrics pool, dual BM25, v8) | +0.03 | 0.41 |
| Cross-encoder top-100 reranking | +0.04 | 0.45 |
| Query enrichment (full-context Qwen, v8 TT) | +0.02 | 0.47 |
| Semantic ID ensemble (if it works) | +0.04 | 0.51 |
| Blind-safe training (prevent regression, more data) | +0.02 | 0.53 |
| Cumulative compounding effects | +0.02 | 0.55 |

**Note:** These estimates are optimistic. Gains do not always compound linearly.
A realistic floor is 0.45-0.48 if semantic IDs fail. 0.55 requires everything to work.

## What I Need From You

1. **GPU access for semantic ID training?** LoRA on Llama-3.2-1B needs ~8GB VRAM,
   2-3 days training. Can you run this on the M4 16GB (MPS) or do you have cloud GPU?
2. **Cross-encoder model download:** `cross-encoder/ms-marco-MiniLM-L-12-v2` (~130MB).
   Do I have internet access to download it during inference?
3. **LM Studio / Gemma-3-12b access for response generation?** The blind response
   generation script references `generate_responses_lmstudio.py`. Is this available?
4. **How many blind submissions can we make?** If limited, we need to be strategic
   about when to submit (after major improvements, not incrementally).
