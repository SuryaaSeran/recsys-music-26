# Plan: Dev/Blind Alignment + Goal-Type Routing (Phase F)

## Goal

Close the dev/blind alignment gap, improve weakest dev categories (C/I/K), and
implement goal-type-aware inference routing. Target: dev nDCG@20 > 0.1684,
blind nDCG@20 > 0.37 without sacrificing composite.

## Current State (2026-06-01)

- Dev nDCG@20: **0.1684** (Phase D, 39-feat, TT v6, tt_pool=2000)
- Blind A nDCG@20: **0.3701** (v10c, v8b H1+H3 42-feat)
- Blind A composite: **0.4837** (v07, judge 4.4/5)
- Phase D/E work complete: TT v8b, 6K dump, H1+H3 tested
- Active branch: `feature/engineering-v2`

## Ablation Findings (2026-06-01)

Source of the 2x dev/blind gap is NOT:
- Turn distribution (weighted dev = 0.1756, still far below 0.37)
- Goal category distribution (weighted dev = 0.1683)
- nDCG formula (verified identical)

Source IS:
1. **Blind A is a different distribution.** Contains "quiz-like" sessions
   ("Put on 'License to Drive' by Work Drugs", "Play Robert Johnson at the
   crossroads") that score near 1.0 for any decent retrieval system.
2. **Phase D sources hurt blind A via pool dilution.** Phase A vs Phase D top-20
   overlap = 9.6/20 on blind A (vs 15.3/20 on dev). Phase D's cooccurrence/CF/
   session-mean sources, which are calibrated to mainstream dev sessions, displace
   correct BM25/TT results on niche blind A sessions.
3. **No "magic" blind A signal.** Phase A (BM25+TT alone) already achieves 0.3709
   on blind. Blind A is near-ceiling for retrieval-only improvement.

Per-category dev nDCG (Phase D, weakest highlighted):

| Cat | nDCG | Type |
|-----|------|------|
| **I** | **0.1326** | specific globally popular song |
| **C** | **0.1332** | specific album by description |
| **K** | **0.1367** | discover multi instrumental broad era |
| J | 0.1585 | specific hit |
| G | 0.1605 | multi mood |
| A | 0.1639 | specific instrumental (game/movie) |
| D | 0.1653 | specific soundtrack |
| B | 0.1807 | specific by lyric phrase |
| H | 0.1884 | multi artists |
| E | 0.1945 | musical journey multi |
| F | 0.1953 | multi era/genre |

Phase D hurt category I (-0.0039 vs Phase A) and barely moved C (-0.0001).
Phase D helped everywhere else (especially J +0.0087, F +0.0063, B +0.0061).

---

## Track 1: Mimic Blind A in LTR Training

The LTR currently trains on 2K/6K random TRAIN sessions, macro-averaged over
all 8 turns. Blind A evaluates one specific turn per session, skewed toward
sessions with specific-goal types and MOVES_TOWARD_GOAL turns only. The
training distribution does not match this.

### 1a. Goal-category stratification (LOW RISK)

Blind A overrepresents B (17.5%) and C (13.7%) vs dev's K-heavy distribution
(15.6%). Category K (our weakest) is only 5% of blind A.

Approach: oversample categories B, C, I, A in the LTR training dump. When
sampling the N training sessions, select sessions such that each goal category
contributes proportionally to the blind A distribution rather than uniformly.

Expected effect: LTR learns to rank better for specific-track-finding queries
at the cost of slightly worse K-type discovery sessions.

Concrete: add `--goal_cat_weights B:3,C:3,I:3,A:2,D:2,J:2,E:1,F:1,G:1,H:1,K:1`
flag to the training session sampler in the inference script.

### 1b. Last-turn-prioritized training (MEDIUM)

Blind A evaluates the last turn of each session. Our LTR trains on all 8 turns
equally. Turns 1-2 have high dev nDCG (0.1948/0.2015) and dominate the training
gradient, but blind A weights them only proportionally.

Approach: duplicate last-turn examples N times (N=3) in the LTR training data
so the model gives more weight to learning final-turn ranking.

OR: train a second LTR model exclusively on last-turn data and blend predictions
(e.g. `0.7 * all_turns_ltr + 0.3 * last_turn_ltr` score).

### 1c. Implicit progress label generation for training augmentation (MEDIUM)

Goal progress assessments from the dataset are gold labels. At inference time
for test/blind sessions, we DON'T have them. The H1/H3 features currently only
fire when gold labels are available.

Train a lightweight classifier:
- Input: user follow-up message text (the message AFTER a music turn)
- Output: MOVES_TOWARD_GOAL / DOES_NOT_MOVE / NEUTRAL
- Training data: 6184 MOVES + 816 DOES_NOT_MOVE labeled turns from the dataset

Signals present in user text:
- Positive: "yes", "perfect", "that's it", "love it", "exactly", "keep going"
- Negative: "no", "not quite", "something different", "too slow/fast", "not really"
- Continuation: "more like", "similar to", "another one"

Rule-based baseline (2a): regex classification costs nothing at inference.
Embedding-based upgrade (2b): cosine similarity to positive/negative template
embeddings using TT model, no extra model required.
LLM-based (2c): use Qwen/Gemma to classify each turn at inference; most accurate
but adds latency.

Once we have inferred progress labels, apply H1 (filter rejected artists from
expansion seeds) and H3 (goal substitution) at inference time for ALL splits,
including blind test.

### 1d. Session augmentation: inject "quiz-like" sessions (MEDIUM-HIGH)

Blind A contains exact-match sessions that are underrepresented in TRAIN. We can
synthesize them from the catalog:

For each artist with >= 3 tracks in catalog:
- Construct a turn-1 session: query = "Play [track_name] by [artist_name]"
- Gold = that track_id
- session_goal category = I (specific globally popular) or B (specific by lyric)

These trivial sessions would teach the LTR that when BM25 returns the exact-named
track at rank 1, the LTR should not override it with popularity or CF-based
candidates. This directly fixes the Phase D regression on Category I.

Expected: eliminates category I regression, improves blind A on exact-match turns.
Risk: adding synthetic sessions may distort LTR calibration for real sessions.
Mitigation: cap synthetic sessions at 20% of training mix.

---

## Track 2: Goal Progress Assessment at Inference (No Gold Labels)

Currently H1/H3 only fire when `goal_progress_assessments` key exists in the
session (dataset sessions only). In blind test mode, the sessions have the full
conversation history but no gold progress labels.

### 2a. Rule-based turn classifier (ZERO COST)

Classify the user message immediately following each music turn:
- POSITIVE if message contains: ("yes", "that's", "exactly", "perfect", "love",
  "great", "it!", "found it", "keep", "more like this")
- NEGATIVE if message contains: ("no", "not", "different", "something else",
  "too slow", "too fast", "rather", "instead", "actually")
- NEUTRAL otherwise

Apply H1 using this signal: exclude NEGATIVE-turn tracks from expansion seeds.
Apply H3 using this signal: substitute goal with most-recent POSITIVE track.

This costs nothing. Add as `--infer_progress_labels` flag. Run ablation on dev
to measure vs gold labels (upper bound) and no labels (current).

### 2b. Embedding similarity classifier (LOW COST)

Use the TT model to compute similarity between the user follow-up message and
a bank of positive/negative template sentences:

Positive templates: "Yes, that's exactly what I was looking for!", "Perfect track!",
"This is great, keep going."
Negative templates: "That's not quite right.", "I want something different.",
"No, not this one."

Classify by majority vote among top-3 templates. Faster than LLM, more
accurate than rule-based. Does not require a new model.

### 2c. Goal tracking features as LTR signals (HIGH VALUE)

Even without inferred labels, add features that capture implicit progress signals:

- `user_followup_positive_score`: TT similarity of follow-up message to positive
  template bank (float 0-1)
- `user_followup_negative_score`: TT similarity to negative template bank
- `conversation_consistency`: TT cosine between current query and mean of all
  prior positive recommendations (estimated from conversation flow)
- `n_estimated_rejections`: count of user messages containing negation words

These features can be trained using the gold progress labels on the train set
and generalize to test sessions where exact labels are unavailable.

---

## Track 3: Goal-Type Routing (Different Strategies Per Conversation Type)

The biggest insight from the category analysis: discovery sessions (E, F, H)
score 0.19+ while specific-track sessions (C, I, K) score 0.13-0.14. These
require fundamentally different retrieval strategies:

| Goal type | Best retrieval signals | Worst signals |
|-----------|----------------------|---------------|
| Specific track (A,B,C,D,I,J) | BM25 exact match, artist expansion | CF, cooccurrence, session-mean |
| Multi-track discovery (E,F,H) | TT similarity, CF neighbors, cooccurrence | BM25 (query too vague) |
| Broad era discovery (K) | Popularity-filtered TT, era/genre cluster | Artist-specific CF |

### 3a. Goal category classifier at inference (LOW COST)

We already have `goal_category` as an LTR feature. Extend this by:
1. Classifying each test session's goal type at inference time using the initial
   user query (rule-based: detect "looking for", "find the song", "discover",
   "recommend multiple", etc.)
2. Using the classified type to route to the appropriate source-weighting config.

This is essentially a meta-feature: `inferred_goal_specificity` ∈ {specific=1,
exploration=2, broad=3}. Add as LTR feature alongside existing `goal_category`.

### 3b. Source-weighted pool per goal type (MEDIUM)

Instead of one fixed pool config, use goal-type-conditional pool configs:

**Specific-track mode** (when goal_specificity = specific):
```
BM25@1000 (more BM25, exact match priority)
+ artist expansion (cap=100, more artist neighbors)
+ TT@500 (less TT noise)
+ NO cooccurrence, NO session-mean, NO CF
```

**Discovery mode** (when goal_specificity = exploration):
```
BM25@300
+ TT@2000
+ CF@300 (more CF weight)
+ cooccurrence@500/250/100
+ session-mean@200
```

**Broad exploration mode** (K category):
```
BM25@200
+ TT@2000
+ Qwen-meta@1000 (era/instrument-aware)
+ CF@100 (diversity)
+ NO cooccurrence (too specific)
```

Implementation: three `--pool_config {specific,discovery,broad}` presets in the
inference script. Route using the inferred goal type.

### 3c. Query rewriting for exact-match sessions (MEDIUM)

For specific-track sessions (types A, B, C, D, I, J), the user query often
contains rich entity information that BM25 can exploit if given cleanly:

Current BM25 query: full user message + conversation history + goal text
Rewritten: extract artist/album/track mentions + key descriptors only

Example:
- Raw: "Can you tell me which song from Peter Doherty's 'Grace/Wastelands' was
  frequently mentioned as one of the standout tracks?"
- Rewritten BM25 query: "Peter Doherty Grace/Wastelands"

Run rewritten query as a parallel BM25 retrieval (`--bm25_entity_pool N`) and
merge with the standard pool. This directly addresses the C-category weakness
(specific album by description).

Implementation: add a named-entity extraction step using the TT tokenizer or
regex patterns for artist/album/song names in quotes.

### 3d. Turn-position-specific LTR models (MEDIUM-HIGH)

Cold-start turns (T1) have zero history features (dist_to_last, history_len,
cf_dist all = 0). These features add noise, not signal. A T1-specific model
trained only on turn-1 examples with history features removed would generalize
better to blind A's turn-1 sessions.

Models to train:
- `ltr_cold_start.txt`: trained on T1 only (1000 sessions), drops history features
- `ltr_warm.txt`: trained on T3+ (3000 sessions), full feature set
- At inference: route T1 predictions to cold_start model, T2+ to warm model

This matches blind A's observation: 20/80 blind sessions are T1, and our Phase D
T1 score (0.1948) still has a ~2x gap with blind T1 performance.

### 3e. Ensemble voting by goal type (HIGH COST)

For each test session, generate top-20 candidates from BOTH the specific-track
config and the discovery config. Blend them using goal-type-confidence weights:

```
final_score = p_specific * score_specific_config +
              p_discovery * score_discovery_config
```

Where p_specific = sigmoid(LTR(inferred_goal_specificity feature)). The LTR
learns which source blend is correct from training data.

This is the most expensive option (2x pool generation) but also the most
accurate because it doesn't commit to a hard classification.

---

## Track 4: Expensive Inference Options

These are higher latency but should significantly improve precision. Use for
blind submissions where we have time budget.

### 4a. Cross-encoder re-ranking (MEDIUM COST, HIGH IMPACT)

Cross-encoder v3 (trained in Phase 3) re-ranks the top-100 pool candidates using
full query-document pairs. Currently not used in production.

Usage: `--ce_rerank_k 100 --ce_model models/crossencoder_v3/...`
Expected: +0.01-0.02 nDCG@20 for specific-track sessions where the correct track
is in the pool but ranked sub-optimally by LTR.
Cost: ~100x more encoder calls per turn. Acceptable for blind submissions.

Best used after goal-type routing: apply CE only in specific-track mode where
precision (ranking the exact right track at #1) matters more than diversity.

### 4b. LLM-based pool pruning for hard cases (EXPENSIVE)

For turns where the LTR confidence is low (top-1 score < threshold), ask Gemma/
Qwen to evaluate the top-25 candidates and pick the most relevant 10.

Prompt format:
```
Given this conversation:
[conversation history]

The user wants: [goal text]

Rank these tracks by relevance (track_name by artist_name):
[candidate list]
```

This is used as a re-ranker, not a generator. Cost: ~1 LLM call per ambiguous
turn. For blind submissions (~80 turns), this is feasible.

### 4c. Adaptive pool sizing based on retrieval confidence (MEDIUM)

Current: fixed pool sizes per source regardless of session.
Better: when BM25 returns a very high-scoring exact match at rank 1 (query
term overlap > threshold), reduce TT pool and suppress CF/cooccurrence entirely.
When BM25 top score is low (vague query), expand TT pool aggressively.

Signal: `bm25_top_score_normalized` (already computed). Add threshold routing:
- If bm25_top_score > 0.8: specific mode (suppress noisy sources)
- If bm25_top_score < 0.3: exploration mode (expand TT aggressively)

This is the automatic version of 3b without needing goal classification. Can be
computed per-turn dynamically. Directly addresses the Phase D regression on blind A
without requiring goal category inference.

Concrete implementation: add `--adaptive_pool` flag that routes per-turn using
the BM25 confidence signal.

### 4d. Multi-hypothesis retrieval with oracle blending (RESEARCH)

Generate two candidate lists per turn:
- List A: optimized for recall (large pool, many sources)
- List B: optimized for precision (small pool, BM25-heavy, exact-match)

Train a meta-selector that predicts which list is better given the query features.
At inference, use the meta-selector to choose the list.

This is a two-stage approach: Stage 1 generates candidate sets, Stage 2 selects
among them. More robust than fixed routing because the selector can learn from
cases where both approaches fail or succeed.

---

## Track 5: Quick Alignment Fixes

### 5a. n_sources normalization

n_sources is the dominant feature (gain 497K) but its distribution shifts when
pool config changes. On blind A, the pool is effectively a different config than
dev (different session music style → different retrieval patterns).

Add: `n_sources_norm = n_sources / log2(1 + pool_source_count)` where
`pool_source_count` is the number of active retrieval sources for this turn.
This makes n_sources scale-invariant.

### 5b. Phase A pool as blind default

Phase A pool (BM25 + TT@1000 + artist expansion) achieves 0.3709 on blind A.
Phase D adds CF/cooccurrence which hurt blind A (0.3164).
TT v8b recovers blind (0.3701) because it improves TT precision without adding
noisy sources.

Current best blind config: v8b TT + Phase A pool + 42-feat LTR.
Do NOT add CF/cooccurrence to blind submissions until we confirm they don't hurt.
Blind submission should always use `--no_cf --no_cooccur --no_session_mean`
flags unless a controlled blind A experiment confirms otherwise.

### 5d. Popularity distribution signal

Blind A history tracks: mean_pop=36, median=34, only 4% < popularity_score 10.
Dev history tracks: mean_pop=43, median=46, 15% < popularity_score 10.

Blind A sits in the "medium popularity" band (10-50), where CF and cooccurrence
are weakest (fewer co-occurrence entries for mid-popularity artists, CF signal
sparse). This confirms why Phase D's CF/cooccurrence sources help dev (higher
median popularity) but hurt blind A (medium popularity, weak CF signal).

Actionable: add `history_mean_popularity` as an LTR feature. When low, suppress
CF source weight. When high, trust CF signal. This is a continuous version of
the adaptive routing in Track 4c.

### 5c. Category I investigation

Category I ("specific globally popular song") is our worst category (0.1326) and
Phase D HURT it (-0.0039). This seems wrong -- globally popular songs should be
easy to find.

Hypothesis: category I sessions ask for very specific versions of popular songs
(e.g. a specific live recording, or "that popular song from the [film/game]")
where the catalog has multiple tracks with similar artist/title and BM25 returns
the wrong version. The CF/cooccurrence sources then crowd out the correct version.

Action: sample 20 category I sessions from dev, inspect the gold tracks and the
retrieved pool. Check whether the gold track IS in our pool (pool recall for I
category) vs whether it's in pool but ranked wrong.

---

## Priority Order

Given remaining time before July 2026 deadline:

| Priority | Track | Expected Impact | Cost | Effort |
|---|---|---|---|---|
| 1 | 2a: Rule-based progress classifier | H1/H3 on all test sessions | Zero latency | Low |
| 2 | 5c: Category I investigation | Find root cause, direct fix | None | Low |
| 3 | 3c: Query rewriting for exact-match | Fix C category (0.1332) | +1 BM25 call | Low |
| 4 | 4c: Adaptive pool sizing | Fix Phase D blind regression | Near-zero | Medium |
| 5 | 3a: Goal category classifier | Enable routing | No extra calls | Medium |
| 6 | 1a: Goal-category stratification | Better LTR for B/C/I | Re-dump required | Medium |
| 7 | 4a: CE re-ranking (blind only) | +0.01-0.02 precision | 100x calls | Medium |
| 8 | 3d: Turn-position-specific LTR | Fix T1 generalization | Separate train | Medium |
| 9 | 1c: Implicit progress classifier | H1/H3 quality at test | Small model | Medium-High |
| 10 | 3b: Source-weighted pool | Full routing | 3x configs | High |
| 11 | 4b: LLM pool pruning | Hard-case precision | 1 LLM call/turn | High |
| 12 | 1d: Quiz-like session augmentation | Fix Cat I & blind A | Synthetic data | High |

## Validation

- Primary gate: dev nDCG@20 > 0.1684
- Secondary: blind nDCG@20 >= 0.37 (don't regress v10c)
- Blind composite target: > 0.4837 (don't regress v07)
- Category-specific targets: C > 0.14, I > 0.14, K > 0.15

## BA100 Baselines (2026-06-01)

Eval set: `plan/BLIND_A_STYLE_100.json` — 100 dev sessions stratified by blind A
goal-category distribution (A:5, B:17, C:14, D:11, E:8, F:10, G:11, H:9, I:2, J:8, K:5).
Evaluator flag: `--session_ids_file plan/BLIND_A_STYLE_100.json --last_turn_only`

| System | All | Last | T1 | T4-8 | Notes |
|--------|-----|------|-----|------|-------|
| Phase A (v6+A_ltr) | 0.1624 | **0.1213** | 0.1878 | **0.1444** | Baseline; beats blind A proxy |
| Phase D (v6+D_ltr) | 0.1602 | 0.1060 | **0.2054** | 0.1358 | Strong T1-3, weak late |
| v10 H1H3 | 0.1523 | 0.1042 | 0.1880 | 0.1274 | Weakest on BA100 |
| Blend D(T1-3)+A(T4+) | **0.1656** | **0.1213** | **0.2054** | **0.1444** | Best overall; targets > this |

Key findings:
- Phase A wins last-turn and T4-8 because its LTR doesn't rely on n_sources (absent in Phase A pool)
- Phase D wins T1-3 because larger TT pool helps cold-start/early turns
- Blend is the best zero-retrain configuration
- Stripping CF/cooccur from Phase D HURTS (these sources help on dev-distribution sessions)
- v8b + Phase A LTR fails: Phase A LTR is calibrated for v6 TT feature distributions

Gate for new systems on BA100: last-turn > 0.1213, all-turns > 0.1656.

### Extended BA100 results (2026-06-01)

| System | All | Last | T1 | T4-8 | Cat-A | Cat-C | Notes |
|--------|-----|------|-----|------|-------|-------|-------|
| Phase A (v6+A_ltr) | 0.1624 | **0.1213** | 0.1878 | **0.1444** | 0.400 | 0.169 | Baseline |
| Phase D v6 | 0.1602 | 0.1060 | **0.2054** | 0.1358 | 0.330 | 0.125 | |
| v10 H1+H3 (42feat v8b) | 0.1523 | 0.1042 | 0.1880 | 0.1274 | 0.253 | 0.195 | |
| 42feat v8b noH1H3 | 0.1492 | 0.1080 | 0.1880 | 0.1232 | 0.244 | 0.195 | |
| **44feat v8b noH1H3** | **0.1520** | **0.1108** | 0.1852 | 0.1301 | 0.244 | 0.191 | n_sources_norm helps |
| Blend D(T1-3)+A(T4+) | **0.1656** | **0.1213** | **0.2054** | **0.1444** | — | — | Best overall, zero retrain |
| PhaseD+entity200 | 0.1597 | 0.1091 | 0.2087 | 0.1344 | — | — | entity BM25 small +lift |
| 44feat+infer_progress | 0.1544 | 0.1052 | 0.1852 | 0.1282 | 0.238 | 0.220 | inferred labels hurt last |

Category A (specific instrumental) gap is the main remaining issue: 0.244 (v8b) vs 0.400 (Phase A).
Root cause: n_sources penalizes specific BM25-only matches.

bm25_top1 (45-feat model) DOES NOT help: sparsity=0.04%, gain=370. LambdaMART can't learn
from a feature this sparse. 45-feat BA100 last-turn = 0.1045 (worse than 44-feat 0.1108).

**Phase F feature engineering conclusion:**
- n_sources_norm (+log1p_n_sources): 44-feat = 0.1108 last-turn → real improvement (+0.003)
- bm25_top1: too sparse to be useful at LambdaMART training scale
- entity BM25: +0.003 pool recall, small category D lift, no meaningful ranking improvement
- Closing the Phase A gap (0.1213) requires retraining on Phase A pool features (v8b TT)

**Next action:** submit v13 (Phase A pool + Gemma v07-prompt). Expected: composite > 0.5.

### Full 1000-session dev results

| System | Dev nDCG@20 | BA100 last-turn | Notes |
|--------|------------|-----------------|-------|
| Phase D v6 (current gate) | **0.1684** | 0.1060 | Gate; worst on BA100 |
| Phase A | 0.1646 | **0.1213** | Best BA100; correlates with blind A |
| v10 H1+H3 (42-feat v8b) | 0.1615 | 0.1042 | Below gate |
| 44-feat v8b (no H1H3) | 0.1603 | 0.1108 | Below gate; better BA100 than v10 |
| Blend D(T1-3)+A(T4+) | 0.1671 | **0.1213** | Best of both; zero retrain |

Key trade-off: Phase D optimizes full-dev (uses n_sources/CF/cooccur) but degrades blind A proxy.
Phase A/blend optimizes blind A proxy but is below Phase D on full dev.
Blind A actual: Phase A=0.3709, Phase D=0.3164, confirming BA100 last-turn is the better signal.

## Assumptions

- Blind A distribution has ~25% trivially-easy exact-match sessions. We can't
  significantly improve beyond 0.37 on blind A through retrieval alone without
  better handling of exact-match queries.
- Goal category is available at test time via the dataset's `conversation_goal`
  field. For true test sessions (blind B, etc.), this must be inferred.
- Progress labels are NOT available at blind test time. Any feature using them
  must be derived from conversation text, not from gold labels.
- CF signals are unreliable for niche/non-mainstream sessions. Routing that
  suppresses CF for niche queries is directionally correct.

## Notes

- The 2x blind/dev gap should be considered a calibration artifact, not a signal
  that dev optimizations are wrong. Dev captures real improvements.
- Blind A is near-ceiling for retrieval-only improvements. Future blind gains
  will come from: (a) better response quality (LLM judge), (b) exact-match
  routing for specific-goal sessions, (c) better generalization of TT model.
- Phase D cooccurrence/CF hurt blind A but help dev. The safe blind default is
  Phase A pool + best TT model + best LTR.
