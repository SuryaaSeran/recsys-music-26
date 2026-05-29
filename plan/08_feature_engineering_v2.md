# Plan: Feature Engineering v2 + Next Directions (Phase D)

## Goal

Beat dev nDCG@20 0.1653 and blind nDCG@20 0.37 through feature engineering,
training improvements, and response generation. Deadline: June 30, 2026.

## Current State

- Dev nDCG@20: 0.1653 (29-feat Phase B reg LTR)
- Blind nDCG@20: 0.37 (Phase A pool, v04 submission with DeepSeek responses)
- Pool recall: 83.03% (~17% of gold tracks unreachable)
- Catalog diversity: 0.5144 (vs organizer baseline 0.3795)
- Lexical diversity: 0.2073 (vs organizer baseline 0.2558 -- we are WORSE)
- LLM-as-Judge: not measured (template responses only on dev)
- TT v8 LoRA training in progress (Phase C, orthogonal)

## Key Constraints

1. Phase B features (popularity, track_year, tt_pool=2000) hurt blind 0.37 -> 0.30.
   New features must be structurally robust, not distribution-dependent.
2. Competition scores on 4 dimensions: nDCG@20, catalog diversity, lexical diversity,
   LLM-as-Judge. We are currently optimizing only nDCG@20.
3. Blind B includes cold-start stress test. CF-dependent features must degrade gracefully.
4. 391 turns (4.9%) are "truly unreachable" -- gold track not in any signal's top-5000.
   These are a hard ceiling.

---

## Track 1: LTR Feature Engineering (IMPLEMENTED, needs eval)

### New Features (10, total 39)

| Feature | Type | Range | Why |
|---|---|---|---|
| `n_sources` | count | [1, 7] | Multi-source agreement = confidence |
| `turn_number` | position | [1, 8] | Early vs late turn ranking behavior |
| `history_len` | count | [0, 7] | Graduated cold-start (replaces binary cold_user) |
| `popularity_pctile` | normalized | [0, 1] | Stable rank-percentile (replaces raw popularity) |
| `years_since_release` | derived | [0, 126] | Inverted year with NaN for missing |
| `tag_overlap_count` | lexical | [0, 12] | Explicit genre/mood match vs query |
| `query_len_tokens` | proxy | [1, 100+] | Query specificity |
| `cf_dist_to_last` | CF cosine | [-1, 1] | Behavioral similarity to last track |
| `cf_dist_to_recent_mean` | CF cosine | [-1, 1] | Behavioral trajectory |
| `goal_category` | categorical | int | Session goal type |

### Fixes to existing features

- `popularity`: NaN for missing (was 0.0, confused LightGBM)
- `track_year`: NaN for missing (was 0.0)
- CF distance arrays: now computed for all users (track-track, not user-dependent)

### Steps

1. [done] Code changes to inference script + LTR trainer
2. [ ] Dump features from TRAIN sessions (2000, seed 42) with new 39-feature schema
3. [ ] Train LTR baseline: 39 features, same regularization as Phase B
4. [ ] Train LTR + soft_labels (already implemented, free experiment)
5. [ ] Train LTR + poly_feats (14 interaction features, already implemented)
6. [ ] Train LTR + soft_labels + poly_feats
7. [ ] Evaluate all 4 variants on golden-200, promote best to full 1000-session dev
8. [ ] Feature importance analysis: flag overfitting risks

---

## Track 2: Recall Improvement (ANALYSIS NEEDED)

Pool recall is 83.03%. The 17% miss rate is split into:
- ~12% "soft unreachable": gold exists in some signal's top-5000 but not in our pool config
- ~5% "truly unreachable": gold not in any signal's top-5000 at all

### Ideas to close the soft-unreachable gap

**2a. BM25 query sharpening for tag-heavy turns**
The current BM25 query concatenates goal + culture + last 4 tracks + last 4 text turns.
For mood/vibe queries ("something chill for studying"), the track history text dilutes
the mood keywords. A second BM25 query using only the latest user message + goal
(no history) would rescue tracks that match mood but not history.
- Expected lift: +1-2% recall on mood/lyrics buckets
- Cost: one extra BM25 call per turn (~negligible)

**2b. Qwen-lyrics pool expansion**
Qwen-lyrics currently only contributes a scoring signal (ql_cos), not pool candidates.
Audit shows it rescues 6.7% of BM25 misses at top-500. Adding top-200 Qwen-lyrics
candidates to the pool would lift recall for lyrics-described tracks.
- Expected lift: +1-2% recall
- Cost: ~200 extra candidates per turn

**2c. Tag-based retrieval (new signal)**
Build a tag-matching retrieval source: for each candidate, compute Jaccard similarity
between the conversation's extracted tags/genres and the candidate's tag_list. Add
top-100 tag-matched tracks to pool. This targets the "user described a genre but
used different words than the track metadata" gap.

**2d. Multi-query TT**
Encode two TT queries per turn: (1) the current full query, (2) a "what came before"
query using only the last 2 played tracks. Union top-K from each. Targets
history-driven turns where the user says little but the trajectory is informative.

### Steps

1. [ ] Run recall gap analysis on current Phase D features (may have improved with v8)
2. [ ] Implement and test 2a (dual BM25 query)
3. [ ] Implement and test 2b (Qwen-lyrics pool)
4. [ ] Evaluate recall lift vs pool size tradeoff
5. [ ] Integrate winning sources into production pipeline

---

## Track 3: Response Generation (MAJOR UNTAPPED DIMENSION)

Current system uses template responses on dev:
`I recommend "{name}" by {artist} based on your request.`

This scores:
- Lexical diversity: 0.2073 (BELOW organizer baseline 0.2558)
- LLM-as-Judge: not measured, but certainly poor (no personalization, no explanation)

The competition explicitly scores response quality via Gemini LLM judge and Distinct-2.
This is the biggest untapped dimension -- potentially worth as much as nDCG improvements.

### Ideas

**3a. Response templates with variation**
Replace the single template with 5-10 diverse templates that rotate based on:
- Turn position (first turn vs follow-up)
- Query type (specific artist vs mood vs "more like that")
- Track metadata (genre, decade, artist popularity)
This alone would substantially lift Distinct-2.

**3b. Local LLM response rewrite (dev-side)**
Apply the same Qwen/Gemma response generation currently used only for blind submissions
to the dev evaluation pipeline. This would let us measure and optimize response quality
during development instead of only at blind submission time.

**3c. Retrieval-grounded response generation**
Instead of just naming the top-1 track, generate responses that reference:
- Why the top track matches the user's stated preferences
- What tags/genres connect the recommendation to the request
- How the track relates to previously played tracks
This directly targets the LLM-as-Judge personalisation dimension.

**3d. Chain-of-Thought response generation**
README notes: "Systems that generate internally reasoned responses (even if the CoT is
hidden) tend to score higher on explanation quality." Generate internal reasoning
(which tracks in top-20 match which user preferences) then produce a grounded response.

### Steps

1. [ ] Measure current Distinct-2 and estimate LLM-as-Judge on dev with template
2. [ ] Implement diverse template rotation (3a) -- zero-model-cost baseline
3. [ ] Run LLM response generation on dev set and measure Distinct-2 lift
4. [ ] Iterate on response quality (grounding, personalization)

---

## Track 4: Blind Generalization (DEFENSIVE)

Phase B hurt blind (0.37 -> 0.30). Understanding why is critical before adding more features.

### Hypotheses

1. **tt_pool=2000 overfits to dev pool distribution.** Dev sessions are synthetic with
   consistent patterns. Blind sessions may have different query distributions.
2. **popularity feature has different distribution in blind.** Dev tracks skew popular
   (synthetic conversations favor well-known tracks). Blind may include more long-tail.
3. **track_year is noise.** Marginal gain (1813) in training, likely fitting to dev-specific
   year patterns that don't transfer.

### Mitigation strategies

**4a. Phase A pool with Phase D features**
Use the safer Phase A pool (tt_pool=1000) with the new 39 features. If this beats
0.1646 on dev, it may also beat 0.37 on blind.

**4b. Feature ablation on blind**
Systematically drop suspect features (popularity, track_year, popularity_pctile,
goal_category) and evaluate each on dev. Features that cause dev nDCG to drop
significantly are load-bearing; features with marginal dev impact but high blind
risk should be dropped.

**4c. Cross-validation stability check**
Compare feature importance across 5 CV folds. Features with high variance in
importance across folds are overfitting candidates.

### Steps

1. [ ] Evaluate Phase D LTR on Phase A pool (tt_pool=1000) for safe blind config
2. [ ] Run feature ablation: drop one feature at a time, measure dev nDCG
3. [ ] Check CV fold importance stability for all 39 features
4. [ ] Produce recommended blind submission config

---

## Track 5: TT v8 Integration (BLOCKED on training completion)

Phase C (TT v8 LoRA, multilingual-e5-base, 512-token context) is training.
When it completes, all features need to be re-dumped because:
- tt_cos and tt_rank_sig distributions will change
- dist_to_last and dist_to_recent_mean are in TT space
- The pool itself changes (different tracks enter via TT expansion)

### Steps (after v8 training completes)

1. [ ] Build v8 index
2. [ ] Quick eval with old LTR (direction signal only)
3. [ ] Dump 39-feature set with v8 TT embeddings
4. [ ] Train new LTR on v8 features
5. [ ] Full dev eval; compare vs v6-based LTR

---

## Track 6: Catalog Diversity (MMR tuning)

Current catalog diversity: 0.5144 (51% of 47K catalog recommended).
This is already above the organizer baseline (0.3795) but could be higher.

MMR (Maximal Marginal Relevance) re-ranking is implemented in `src/infer/mmr.py`
with lambda=0.5. It is NOT currently applied in the production pipeline.

### Steps

1. [ ] Apply MMR post-processing to LTR output with lambda sweep (0.3, 0.5, 0.7)
2. [ ] Measure nDCG@20 vs catalog diversity tradeoff
3. [ ] Find lambda that maximizes catalog diversity with <0.001 nDCG@20 loss

---

## Priority Order

Given the June 30 deadline:

| Priority | Track | Expected Impact | Effort |
|---|---|---|---|
| 1 | Track 1: LTR features | +0.001 to +0.005 nDCG@20 | Low (code done, just train/eval) |
| 2 | Track 3: Response generation | Major lift on Distinct-2 + LLM-judge | Medium |
| 3 | Track 4: Blind generalization | Protect blind score (0.37 baseline) | Low |
| 4 | Track 5: TT v8 integration | +0.001 to +0.010 if recall improves | Blocked |
| 5 | Track 6: MMR diversity | Free points on catalog diversity | Low |
| 6 | Track 2: Recall improvement | +0.001 to +0.003 (diminishing returns) | Medium |

## Validation

- Track 1: Dev nDCG@20 > 0.1653
- Track 2: Pool recall > 0.8303
- Track 3: Distinct-2 > 0.2558 (beat organizer baseline)
- Track 4: Phase A pool + new LTR does not regress vs current Phase A blind (0.37)
- Track 5: Dev nDCG@20 > 0.1653 with v8 embeddings
- Track 6: Catalog diversity > 0.55 with <0.001 nDCG@20 loss
