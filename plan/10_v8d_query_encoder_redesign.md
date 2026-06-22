# Phase 10: v8d Query Encoder Redesign (FUTURE)

Locked-in spec for the next TT retrain. Not active yet — pending current LTR-only intermediate.

## Goal

Rebuild the TT v8 query encoder to match what blind eval actually exposes,
add reaction-aware history, weighted positives, and specificity-gated hard
negatives. Target: improve blind A nDCG@20 beyond v8c (currently 0.1729 dev).

## Anchor format (role-tagged)

What's exposed at blind eval at target turn t:
- Profile (always): age_group, country_code, gender, preferred_musical_culture, preferred_language
- Goal (always): listener_goal text + specificity (control token)
- Conversation history 1..t-1: Q_i (user), R_i (assistant), M_i (track), P_{i+1} (reaction)
- Live query: Q_t
- Reaction state: P_1..P_t (last P_t = continue vs pivot signal)

Exclude: every `thought` (nulled at eval), the target M_t.

Serialized format:
```
query: [PROFILE] 20s · BR · female · Western Alternative Rock · English
[GOAL] explore new artists, intense/dramatic  (LL)
[T1] USER: ... | REC: The Fiend – Alesana | ASST: ... | REACTION: liked
[T2] USER: ... | REC: A Forbidden Dance – Alesana | ASST: ... | REACTION: rejected
[NOW] USER: <Q_t>
```

Note: specificity in parens after goal text = control token. HH/LH = peaked match.
LL/HL = broader match.

Empty history (t=1 cold) → just [PROFILE] [GOAL] [NOW] block.

## Positive labeling

For anchor at turn t, gold = M_t. Weight by P_{t+1}:
- P_{t+1} = MOVES → weight 1.0 (clean positive)
- P_{t+1} = DOES_NOT → weight 0.3 (weak positive — still must reproduce M_t)
- P_{t+1} missing (M_8, last turn) → weight 1.0 (neutral)

Implementation: probabilistic subsampling (drop with probability 1 - weight).
Matches the MNRL training loop without custom loss code.

## Hard negative mining

Per anchor at turn t, in order of value:

1. **Confirmed rejections in same session**: any M_i with P_{i+1} = DOES_NOT.
   Paired with this context (same intent), explicitly rejected. Strongest hard neg.

2. **Same-session non-accepted tracks**: other surfaced tracks from same hidden
   pool. Same broad intent but not the gold.

3. **Artist-repeat distractors**: tracks by artists already recommended this
   session (for discovery goals — directly addresses row-0 failure).

4. **In-batch positives from other sessions**: free, semi-hard.

5. **Random catalog tracks**: easy negs, cover full retrieval space.

## False-negative prevention rules

- **MOVES protection**: never use a track that was P_{i+1} = MOVES anywhere
  in the same session as a negative within that session — it's still on-goal.

- **specificity gating**:
  - HH / LH (one correct track): mine hard negatives aggressively,
    including same-session positional negatives.
  - LL / HL (many acceptable): skip same-session positional/co-relevant negatives.
    Use only confirmed DOES_NOT + in-batch + random.

## Training

InfoNCE = SentenceTransformer MultipleNegativesRankingLoss (mathematically
equivalent). Temperature-scaled cosine (default τ via scale=20 in MNRL).
P_{t+1} weights applied via subsampling. Specificity gating applied at
data-building time (different neg pools per specificity).

LoRA r=32, alpha=64. Reuse v8c training hyperparams: lr 1e-4, batch 8 ×
grad_accum 4, 5 hard negs, plateau-based stopping via watcher.

## Inference parity

`run_inference_fusion_recall_expansion.py` must reproduce the exact same
anchor format. Add `--anchor_v8d` flag that switches the tt_query construction
to the new tagged format. Without the flag, falls back to current behavior
(preserves v8c reproducibility).

## LTR layer (unchanged spec)

LTR reranker sits on encoder's top-K. Features: novelty / last-rejected /
goal-match. Already implemented; will benefit from better encoder.

## Outstanding questions

- specificity control token format: `(LL)` literal vs `[SPEC=LL]` token? Default to literal.
- Profile fields: include `age` (numeric) or just `age_group` (band)? Default `age_group`.
- Token budget: with structured format, anchor may exceed 512 tokens. Greedy budgeting
  strategy: keep [PROFILE] [GOAL] [NOW] core, drop turns from oldest first.

## Estimated cost

- New data builder: ~400 LOC
- Inference anchor path: ~50 LOC
- TT training: ~10hr on MPS
- Index rebuild: ~10 min
- LTR re-dump: ~3hr
- LTR retrain: ~15 min
- Dev eval: ~30 min

Total: ~14-15 hours wall clock.
